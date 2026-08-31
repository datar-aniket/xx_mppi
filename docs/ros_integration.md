# ROS 2 integration boundary

`xxcar_msgs/msg/VehicleControlTrajectory` is complete and maps a planned
trajectory to `vehicle_control_trajectory`. `current_time` is publication time;
`solution_pose_time` is the timestamp of the input pose used as `x0`. For every
index `i`, `states[i]` and `controls[i]` refer to
`solution_pose_time + i * dt`. Both arrays have `horizon` entries.

The incoming vehicle-state message is intentionally not invented here. The
controller exposes a ROS-independent `VehicleObservation` containing the fields
needed from the UART-backed state: pose time, ENU pose/yaw, speed, yaw rate,
linear acceleration, sideslip, measured torque, measured steering, driven-wheel
speed, and status. Once the final custom `.msg` is supplied, the ROS adapter only
needs to:

1. copy its timestamp and scalar fields;
2. convert `geometry_msgs/Pose.orientation` to standard ENU yaw;
3. map body-FLU `Twist` to `hypot(linear.x, linear.y)` speed and `angular.z`
   yaw rate;
4. call `MppiController::UpdateObservation` for every accepted state update;
5. call `PlanLatest` at `solve_rate_hz` (or immediately when state rate is no
   higher), then call `ToRosMessage` and publish on
   `vehicle_control_trajectory`.

`MppiRosRuntime` already implements steps 4 and 5, including reliable depth-1
publishing, a `solve_rate_hz` wall timer, no duplicate solve without a new
state, and throttled error reporting. The concrete subscriber only owns steps
1 through 3 and calls `OnObservation`.

No MCU/UART publisher belongs in this package; the downstream communication
controller consumes the ROS trajectory.

The UART document states that sideslip is NaN below 0.3 m/s; the core treats
that specific low-speed case as zero. A non-finite sideslip while moving is
rejected. The MCU's default VESC scalars expose uncalibrated ADC/current/count
values, so `VESC_STEER_K`, `VESC_TORQUE_K`, and `VESC_SPEED_K` must be
characterized before the adapter may label them radians, Nm, and m/s.
