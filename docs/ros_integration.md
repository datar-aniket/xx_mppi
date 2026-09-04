# ROS 2 integration

`xx_mppi_node` subscribes to `xxcar_msgs/msg/EkfState` on `ekf/state`. It can
publish `xxcar_msgs/msg/VehicleControlTrajectory` on
`vehicle_control_trajectory`. The checked-in vehicle configuration instead publishes
`geometry_msgs/msg/Twist` on `cmd_vel`, using the same field contract as
`PID_lanekeeping`. Topic names and output selection are ROS parameters and
launch arguments. No MCU/UART publisher belongs to this package.

Obstacle avoidance subscribes to `/scan` (`sensor_msgs/msg/LaserScan`) in frame
`laser`. It obtains `base_link <- laser` once from TF, then reuses the cached
static transform. SDF construction runs on a latest-scan-wins worker thread and
does not execute in the state or MPPI timer callbacks. See
`docs/obstacle_avoidance.md` for deskew and cost details.

The adapter uses `header.stamp` as `solution_pose_time`, the ENU pose quaternion
for yaw, body-FLU horizontal twist magnitude for speed, `angular_velocity.z`
for yaw rate, `linear_acceleration.x`, sideslip, wheel torque, steering, and
motor peripheral speed. These three feedback channels are already in radians,
newton-metres, and metres per second and are copied without scale or offset.
`observation.use_measured_control_feedback: true` therefore starts steering
velocity and torque-rate costs from the real actuator state. Direct-control
output mapping is independent and does not alter EKF feedback.

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
`require_solution_validity:=false` bypasses all `solution_status` bitmask checks
for bench testing. It does not disable source-validity, quaternion,
finite-value, freshness, or timestamp-order checks. Setting
`require_absolute_yaw:=false` accepts either relative or absolute yaw when
solution validity is enabled, and `require_vesc:=false` permits testing without
fresh actuator feedback. These relaxations are not recommended for autonomous
vehicle operation.

For a launch-time bench test against an EKF that has not yet asserted its
solution bits:

```bash
ros2 launch xx_mppi mppi.launch.py require_solution_validity:=false
```

Restore the default (`true`) before connecting the normal trajectory output to
an actuator controller.

An EKF `reset_counter` change resets the continuous `s` projection, MPPI warm
start, and accepted timestamp history before the new state is installed.
Sideslip may be NaN below 0.3 m/s; the controller substitutes zero only in that
low-speed case. A non-finite sideslip while moving is rejected.

`current_time` in the output is publication time and `solution_pose_time` is
the accepted input pose time. For every index `i`, `states[i]` and `controls[i]`
refer to `solution_pose_time + i * dt`; both arrays contain `horizon` entries.

`solve_rate_hz` and `control_publish_rate_hz` in `mppi.yaml` are independent.
The solver consumes at most one solve per accepted EKF generation and atomically
replaces the latest solution. The publication timer emits each solution at most
once, so the configured control rate is an upper bound rather than a stale
command hold loop. `maximum_solution_age_s` drops a completed solution if its
source pose is already too old when the publication timer runs; set it to zero
only for deliberate bag replay.

The physical derivative costs are configured in `weights.yaml`:

- `longitudinal_acceleration.acceleration_weight` and
  `deceleration_weight` apply to `(V[t+1]-V[t])/dt`;
- `control_rate.steering_velocity_radps` and
  `wheel_torque_rate_nmps` apply to command differences divided by `dt`.

These are evaluated identically by the CPU reference evaluator, analytic CUDA
rollouts, and TensorRT-neural CUDA rollouts.

## Solver information

The latest successfully published control solution is printed to the node
terminal with ROS `INFO` logging at `info_log_rate_hz` (10 Hz by default).
Each line reports:

- `solve_time_ms`, `lambda`, `steering_sigma_rad`, and
  `wheel_torque_sigma_nm`;
- `steering_command_rad` and `wheel_torque_command_nm` in MPPI physical units;
- `minimum_cost`, `effective_sample_size`, `finite_rollouts`, and
  `solution_age_ms`.

The command fields come from the solution that was actually sent by the control
publisher, rather than a newer unpublished solve. No additional ROS topic is
created. The output looks like:

```bash
[xx_mppi_node] [INFO] MPPI info: solve=... lambda=... sigma=[...] command=[...]
```

## Direct control output

Set `publish_direct_control: true` in `config/mppi.yaml`, or override it with
`publish_direct_control:=true` at launch, to disable trajectory publication and
publish the first optimized MPPI control as `geometry_msgs/msg/Twist`:

- `angular.z` is steering angle in radians, multiplied by
  `direct_control_steering_scale` and clamped to
  `+/- direct_control_steering_limit_rad`. MPPI steering is positive-left,
  matching the positive-left yaw rate its vehicle model produces; set the scale
  to `-1.0` when the actuator chain downstream of `cmd_vel` is positive-right;
- with `control_mode: duty_cycle`, `linear.x` is
  `wheel_torque_nm * direct_control_torque_to_throttle_scale`, clamped between
  `direct_control_throttle_min` and `direct_control_throttle_max`;
- with `control_mode: torque`, `linear.x` is the optimized wheel torque in Nm,
  unchanged and without the duty-cycle clamp;
- all other `Twist` fields are zero.

The checked-in YAML defaults are solution-validity checking enabled, direct
output selected, topic `cmd_vel`, mode `duty_cycle`, and vehicle-specific
mapping/limits. Duty-cycle mode matches `control_throttle_type:=0` in
`ekf_mcu_driver`; calibrate its scale for the vehicle. Torque mode requires a
consumer that interprets `linear.x` as Nm—it must not be connected to a driver
that interprets the same field as duty cycle or current.

```bash
ros2 launch xx_mppi mppi.launch.py \
  publish_direct_control:=true \
  direct_control_topic:=cmd_vel \
  control_mode:=duty_cycle \
  direct_control_torque_to_throttle_scale:=1.0 \
  direct_control_throttle_min:=-1.0 \
  direct_control_throttle_max:=1.0 \
  direct_control_steering_scale:=1.0
```

Direct mode can actuate the vehicle. Ensure no other controller (including
`PID_lanekeeping`) publishes to the selected topic, keep the vehicle lifted or
otherwise secured during initial calibration, and verify the driver's command
watchdog before driving. `ekf_mcu_driver` must have direct control sending
enabled only when you are ready for it to consume this topic. All control
publishers use best-effort reliability.

These defaults are loaded from the top level of `config/mppi.yaml` before ROS
parameters are declared:

```yaml
require_solution_validity: true
publish_direct_control: true
direct_control_topic: cmd_vel
control_mode: duty_cycle
direct_control_torque_to_throttle_scale: 25.0
direct_control_throttle_min: -50.0
direct_control_throttle_max: 50.0
direct_control_steering_scale: -4.5
direct_control_steering_limit_rad: 0.5

observation:
  use_measured_control_feedback: false
  maximum_model_sideslip_rad: 0.8
```

An explicitly supplied ROS parameter or launch argument takes precedence over
the YAML value. The node validates unknown modes, empty topics, non-finite
mapping values, and reversed duty-cycle limits at startup.

## Vehicle startup

Build on the Orin with CUDA enabled, supply the real raceline in the selected
configuration directory, then start the EKF driver and MPPI node in the default
trajectory-output mode:

```bash
ros2 launch ekf_mcu_driver ekf_mcu_driver.launch.py enable_send_control:=false
ros2 launch xx_mppi mppi.launch.py
```

The default configuration is resolved as `share/xx_mppi/config` using the
ament package index. With a symlink install this points back to the package's
source configuration, so no absolute path is required and configuration-only
edits do not require rebuilding. Set `current_map` to the selected map directory;
the raceline filename is derived from its folder name. Other relative paths in
`model.yaml` resolve against the configuration directory. Pass an absolute
`config_directory` only to use an external configuration set.

Verify the safety gate and output before enabling the downstream actuator
controller:

```bash
ros2 topic hz /ekf/state
ros2 topic echo /ekf/state --once
ros2 topic hz /vehicle_control_trajectory
ros2 topic echo /vehicle_control_trajectory --once
ros2 topic hz /scan
ros2 run tf2_ros tf2_echo base_link laser
ros2 run xx_mppi smoke_test_ros_pipeline.py --duration 10
```

The checker loads the expected trajectory horizon from the selected
`config/mppi.yaml`. It uses the installed package config by default; pass
`--config-directory /absolute/config` when the controller uses an external
configuration set. `--expected-horizon` remains available as an explicit
override.

Confirm on the bench that the driver publishes the documented radians, Nm, and
m/s and that their signs agree with the model. The MPPI adapter intentionally
has no feedback scale parameters.

## Sample-map bench test

To test without a globally valid vehicle EKF state, start the sample publisher
and controller together. The publisher uses `/xx_mppi/sample_ekf_state`, so it
does not replace or modify the real `/ekf/state` stream:

```bash
ros2 launch xx_mppi sample_test.launch.py
```

In another terminal, validate the resulting trajectories. The checker derives
its default threshold from `control_publish_rate_hz` (80% to allow startup and
scheduling jitter):

```bash
ros2 run xx_mppi smoke_test_ros_pipeline.py \
  --state-topic /xx_mppi/sample_ekf_state \
  --trajectory-topic /xx_mppi/sample_vehicle_control_trajectory \
  --duration 10
```

The sample publisher follows `config/raceline.csv`. Both the state and
trajectory topics are isolated under `/xx_mppi`; they must never be remapped to
a vehicle actuation input.

## RViz visualization

Visualization is optional and disabled by default so it adds no publication
work to the control loop. Enable it at launch:

```bash
ros2 launch xx_mppi mppi.launch.py \
  publish_visualization:=true
```

The sample bench launch enables it automatically. Set RViz's fixed frame to
`map`, then add the following displays:

| Topic | Type | Contents |
|---|---|---|
| `/xx_mppi/expected_path` | `nav_msgs/Path` | Current expected Cartesian horizon |
| `/xx_mppi/rollouts` | `visualization_msgs/MarkerArray` | One line strip per best-weighted sampled rollout |
| `/xx_mppi/raceline` | `nav_msgs/Path` | Static raceline center |
| `/xx_mppi/track_left_boundary` | `nav_msgs/Path` | Static positive-`e` boundary |
| `/xx_mppi/track_right_boundary` | `nav_msgs/Path` | Static negative-`e` boundary |

All xx_mppi ROS publishers and subscriptions use **best-effort reliability**.
Set each RViz display's Reliability Policy to `Best Effort`; a Reliable RViz
subscription is not QoS-compatible with these publishers. The three static
paths are also transient-local and are republished at 1 Hz by the visualization
worker, so RViz can discover them even when it starts after the controller or
uses volatile durability.

The expected path and rollout array are built and published by a latest-only
visualization worker after the solve; they do not run in the control publication
callback. Their reduced rate and rollout count come from `mppi.yaml`:

```yaml
visualization_rate_hz: 10.0
num_rollouts: 15  # number of best-weighted sampled rollouts to draw
```

GPU rollout snapshots are requested only at this reduced cadence. Pose stamps
in the optimized path are the predicted solution times; its top-level header
uses publication time.

All names and the frame are launch arguments: `visualization_frame_id`,
`expected_path_topic`, `rollouts_topic`, `raceline_path_topic`,
`track_left_boundary_topic`, and `track_right_boundary_topic`. Restart the node
after changing them.

If RViz shows nothing, verify all three conditions:

1. Launch with `publish_visualization:=true` (it is disabled by default).
2. Set RViz Fixed Frame to `map`.
3. Set every Path and MarkerArray display's Reliability Policy to `Best Effort`.
