# ROS 2 integration

`xx_mppi_node` subscribes to `ekf_mcu_driver/msg/EkfState` on `ekf/state` and
publishes `xxcar_msgs/msg/VehicleControlTrajectory` on
`vehicle_control_trajectory`. Both topic names are ROS parameters and launch
arguments. No MCU/UART publisher belongs to this package; the downstream
communication controller consumes the trajectory.

The adapter uses `header.stamp` as `solution_pose_time`, the ENU pose quaternion
for yaw, body-FLU horizontal twist magnitude for speed, `angular_velocity.z`
for yaw rate, `linear_acceleration.x`, sideslip, wheel torque, steering, and
motor peripheral speed. The default scale factors are one. Change
`steering_scale_to_rad`, `steering_offset_rad`, `torque_scale_to_nm`, or
`motor_speed_scale_to_mps` only when the driver values are not already SI.

Before accepting a sample, the node requires:

- attitude, horizontal velocity, and horizontal position solution bits;
- absolute yaw by default, because relative yaw is not guaranteed to share the
  map datum;
- estimator and gyro source-valid bits;
- the VESC source-valid bit by default;
- a finite nonzero quaternion and finite controller inputs;
- a strictly increasing pose timestamp no older than `maximum_state_age_s`
  and no farther into the future than `future_tolerance_s`.

Set `maximum_state_age_s:=0.0` only for deliberate bag replay. Setting
`require_absolute_yaw:=false` accepts either relative or absolute yaw, and
`require_vesc:=false` permits testing without fresh actuator feedback. These
relaxations are not recommended for autonomous vehicle operation.

An EKF `reset_counter` change resets the continuous `s` projection, MPPI warm
start, and accepted timestamp history before the new state is installed.
Sideslip may be NaN below 0.3 m/s; the controller substitutes zero only in that
low-speed case. A non-finite sideslip while moving is rejected.

`current_time` in the output is publication time and `solution_pose_time` is
the accepted input pose time. For every index `i`, `states[i]` and `controls[i]`
refer to `solution_pose_time + i * dt`; both arrays contain `horizon` entries.

## Vehicle startup

Build on the Orin with CUDA enabled, supply the real raceline in the selected
configuration directory, then start the EKF driver and MPPI node:

```bash
ros2 launch ekf_mcu_driver ekf_mcu_driver.launch.py enable_send_control:=false
ros2 launch xx_mppi mppi.launch.py config_directory:=/absolute/path/to/config
```

Verify the safety gate and output before enabling the downstream actuator
controller:

```bash
ros2 topic hz /ekf/state
ros2 topic echo /ekf/state --once
ros2 topic hz /vehicle_control_trajectory
ros2 topic echo /vehicle_control_trajectory --once
ros2 run xx_mppi smoke_test_ros_pipeline.py --duration 10
```

The UART-backed VESC channels must be calibrated before they can be treated as
radians, Nm, and m/s. In particular, confirm `VESC_STEER_K`, `VESC_TORQUE_K`,
and `VESC_SPEED_K` or apply the equivalent node scale parameters.
