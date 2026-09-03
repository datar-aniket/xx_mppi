#!/usr/bin/env python3
"""Publish valid synthetic EKF states around config/raceline.csv."""

import math

import rclpy
from xxcar_msgs.msg import EkfState
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy


class SampleEkfStatePublisher(Node):
    def __init__(self):
        super().__init__("xx_mppi_sample_ekf_state")
        self.declare_parameter("state_topic", "xx_mppi/sample_ekf_state")
        self.declare_parameter("publish_rate_hz", 100.0)
        self.declare_parameter("speed_mps", 1.5)

        topic = self.get_parameter("state_topic").value
        rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.speed_mps = float(self.get_parameter("speed_mps").value)
        if not topic or not math.isfinite(rate_hz) or rate_hz <= 0.0:
            raise ValueError("state_topic must be nonempty and publish_rate_hz positive")
        if not math.isfinite(self.speed_mps) or self.speed_mps <= 0.0:
            raise ValueError("speed_mps must be finite and positive")

        self.radius_m = 5.0
        self.center_east_m = -5.0
        self.center_north_m = 0.0
        self.start_time_ns = self.get_clock().now().nanoseconds
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.publisher = self.create_publisher(EkfState, topic, qos)
        self.timer = self.create_timer(1.0 / rate_hz, self.publish_state)
        self.get_logger().warning(
            "Publishing sample-only EKF states on '%s'; do not use for vehicle control"
            % topic
        )

    def publish_state(self):
        now = self.get_clock().now()
        elapsed_s = 1.0e-9 * float(now.nanoseconds - self.start_time_ns)
        yaw_rate = self.speed_mps / self.radius_m
        theta = math.fmod(yaw_rate * elapsed_s, 2.0 * math.pi)
        yaw_enu = theta + 0.5 * math.pi

        message = EkfState()
        message.header.stamp = now.to_msg()
        message.header.frame_id = "map"
        message.pose.position.x = self.center_east_m + self.radius_m * math.cos(theta)
        message.pose.position.y = self.center_north_m + self.radius_m * math.sin(theta)
        message.pose.orientation.z = math.sin(0.5 * yaw_enu)
        message.pose.orientation.w = math.cos(0.5 * yaw_enu)
        message.twist.linear.x = self.speed_mps
        message.angular_velocity.z = yaw_rate
        message.side_slip_rad = 0.0
        message.wheel_torque_nm = 0.0
        message.steering_angle = math.atan(0.312 / self.radius_m)
        message.motor_speed_ms = self.speed_mps
        message.solution_status = (
            EkfState.SOLUTION_STATUS_ATTITUDE_VALID
            | EkfState.SOLUTION_STATUS_YAW_ABSOLUTE
            | EkfState.SOLUTION_STATUS_VEL_HORIZ
            | EkfState.SOLUTION_STATUS_POS_HORIZ
        )
        message.source_valid = (
            EkfState.SOURCE_VALID_ESTIMATOR
            | EkfState.SOURCE_VALID_GYRO
            | EkfState.SOURCE_VALID_VESC
        )
        message.mcu_timestamp_us = now.nanoseconds // 1000
        self.publisher.publish(message)


def main():
    rclpy.init()
    node = SampleEkfStatePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # ros2 launch may deliver another SIGINT while rclpy is tearing down.
        # Treat it as the same requested shutdown instead of emitting a traceback.
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
