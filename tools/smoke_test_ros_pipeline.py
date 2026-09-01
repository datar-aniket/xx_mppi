#!/usr/bin/env python3
"""Read-only live validation of EKF-state to MPPI-trajectory ROS flow."""

import argparse
import math
import sys
import time

import rclpy
from ekf_mcu_driver.msg import EkfState
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from xxcar_msgs.msg import VehicleControlTrajectory


def stamp_tuple(stamp):
    return (int(stamp.sec), int(stamp.nanosec))


def stamp_seconds(stamp):
    return float(stamp.sec) + 1.0e-9 * float(stamp.nanosec)


class PipelineMonitor(Node):
    def __init__(self, args):
        super().__init__("xx_mppi_pipeline_smoke_test")
        self.args = args
        self.state_stamps = set()
        self.solution_stamps = []
        self.trajectory_arrivals = []
        self.errors = []
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(EkfState, args.state_topic, self.on_state, qos)
        self.create_subscription(
            VehicleControlTrajectory,
            args.trajectory_topic,
            self.on_trajectory,
            qos,
        )

    def fail(self, text):
        if len(self.errors) < 50:
            self.errors.append(text)

    def on_state(self, message):
        self.state_stamps.add(stamp_tuple(message.header.stamp))

    def on_trajectory(self, message):
        self.trajectory_arrivals.append(time.monotonic())
        self.solution_stamps.append(stamp_tuple(message.solution_pose_time))
        horizon = int(message.horizon)
        if horizon <= 0:
            self.fail("trajectory horizon is zero")
        if self.args.expected_horizon and horizon != self.args.expected_horizon:
            self.fail(f"horizon {horizon} != expected {self.args.expected_horizon}")
        if len(message.states) != horizon or len(message.controls) != horizon:
            self.fail("state/control arrays do not match horizon")
        if not math.isfinite(message.dt) or message.dt <= 0.0:
            self.fail("trajectory dt is invalid")
        age_ms = 1000.0 * (
            stamp_seconds(message.current_time)
            - stamp_seconds(message.solution_pose_time)
        )
        if age_ms < -1.0 or age_ms > self.args.maximum_solution_age_ms:
            self.fail(f"trajectory solution age is {age_ms:.2f} ms")
        for state in message.states:
            if not math.isfinite(state.x_m) or not math.isfinite(state.y_m):
                self.fail("trajectory contains a non-finite state")
                break
        for control in message.controls:
            if not math.isfinite(control.steering_angle_rad) or not math.isfinite(
                control.torque_nm
            ):
                self.fail("trajectory contains a non-finite control")
                break
            if abs(control.steering_angle_rad) > self.args.maximum_steering_rad + 1e-5:
                self.fail("trajectory steering exceeds configured bound")
                break
            if abs(control.torque_nm) > self.args.maximum_torque_nm + 1e-5:
                self.fail("trajectory torque exceeds configured bound")
                break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--state-topic", default="/ekf/state")
    parser.add_argument(
        "--trajectory-topic", default="/vehicle_control_trajectory"
    )
    parser.add_argument("--minimum-rate-hz", type=float, default=80.0)
    parser.add_argument("--maximum-solution-age-ms", type=float, default=100.0)
    parser.add_argument("--expected-horizon", type=int, default=50)
    parser.add_argument("--maximum-steering-rad", type=float, default=0.5)
    parser.add_argument("--maximum-torque-nm", type=float, default=5.0)
    args, ros_args = parser.parse_known_args()
    if args.duration <= 0.0 or args.minimum_rate_hz < 0.0:
        parser.error("duration must be positive and minimum rate nonnegative")

    rclpy.init(args=ros_args)
    monitor = PipelineMonitor(args)
    deadline = time.monotonic() + args.duration
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(monitor, timeout_sec=0.05)
    finally:
        monitor.destroy_node()
        rclpy.shutdown()

    missing_stamps = set(monitor.solution_stamps) - monitor.state_stamps
    if missing_stamps:
        monitor.fail(
            f"{len(missing_stamps)} solution timestamp(s) were not observed on the state topic"
        )
    count = len(monitor.trajectory_arrivals)
    if count < 2:
        monitor.fail(f"received only {count} trajectory message(s)")
        rate_hz = 0.0
    else:
        elapsed = monitor.trajectory_arrivals[-1] - monitor.trajectory_arrivals[0]
        rate_hz = float(count - 1) / max(elapsed, 1.0e-9)
        if rate_hz < args.minimum_rate_hz:
            monitor.fail(
                f"trajectory rate {rate_hz:.2f} Hz is below {args.minimum_rate_hz:.2f} Hz"
            )

    if monitor.errors:
        print("MPPI ROS pipeline: FAIL", file=sys.stderr)
        for error in monitor.errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(
        f"MPPI ROS pipeline: PASS ({count} trajectories, {rate_hz:.2f} Hz, "
        f"{len(monitor.state_stamps)} state stamps)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
