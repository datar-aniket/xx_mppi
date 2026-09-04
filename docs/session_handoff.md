# MPPI port session handoff

Last updated: 2026-09-02

This document captures the decisions, implementation state, validation, and
remaining work from the MPPI porting session so work can continue in another
chat without reconstructing the conversation.

## Goal and scope

Port the tried-and-tested EPIC MPPI controller architecture from
`/home/aniket/Documents/TRI/epic_workspace/src/` into the standalone ROS 2
package `xx_mppi` in the `xxCar_Nav`/navigation workspace. EPIC is a behavioral
reference only; the new implementation must not depend on EPIC packages.

The target is a Jetson Orin NX running JetPack 6.2.2, Ubuntu 22.04, and ROS 2
Humble. C++17, CUDA, and TensorRT were selected instead of JAX/XLA for the
vehicle deployment. The reasons were direct ROS 2 integration, predictable
native deployment, access to fused CUDA rollouts, TensorRT support for the
future neural model, and fewer Python/XLA runtime concerns on the vehicle.

Requirements agreed during the session:

- Remove perception and online mapping entirely.
- Retain a static raceline, static track bounds, continuous closed-loop `s`,
  coordinate frames, integrators, MPPI sampling/cost logic, and robustness
  additions.
- Keep the structure and terminology close to EPIC where practical:
  `MppiController`, builder, raceline, map boundary, integrator, models, and
  frames.
- Use real control units: steering in radians and total wheel-side torque in
  Nm, with bounds supplied by `mppi.yaml`.
- Keep both kinematic bicycle and dynamic Fiala models. Use Fiala initially.
- Later support a PyTorch neural model that outputs the same four body-state
  derivatives as the EPIC learned model. C++/CUDA retains integration and path
  dynamics.
- Current MPPI problem: `K=2001`, `T=35`, `dt=0.1 s`; solve and control
  publication rates are configured independently.
- Runtime YAML changes, including sample count, do not require recompilation.
  Restart the node so it reloads the configuration and reallocates buffers.
- The controller supports ROS trajectory output and direct first-command
  output; the checked-in configuration currently selects direct
  `geometry_msgs/msg/Twist` publication. `ekf_mcu_driver` still owns MCU/UART
  communication.
- Make small, tested changes and commit each feature so development remains
  traceable.

## Vehicle assumptions

The vehicle is a Traxxas TRX-4 Sport 1/10-scale car with approximately 2 kg
ready-to-run mass, stock tires, and a locked four-wheel-drive drivetrain.
`config/vehicle.yaml` contains the current starting profile:

- mass: 2.0 kg;
- wheelbase: 0.312 m, split evenly around the current assumed CG;
- stock tire radius: 0.05842 m;
- locked AWD enabled;
- yaw inertia, cornering stiffness, friction, and equivalent driveline inertia
  are initial estimates and require identification from vehicle logs.

For locked AWD, the Fiala model evaluates combined longitudinal/lateral slip at
both axles using a common effective wheel-speed state. The torque command is
total wheel-side driveline torque, not torque per wheel.

## State, frames, and map

The internal float32 Frenet state ordering is:

1. yaw rate `[rad/s]`;
2. speed `[m/s]`;
3. sideslip `[rad]`;
4. common driven-wheel peripheral speed `[m/s]`;
5. lateral deviation `e` `[m]`, positive left of the raceline;
6. relative course heading `dphi` `[rad]`;
7. continuous path evolution `s` `[m]`.

Controls are `[steering_angle_rad, wheel_torque_nm]`.

ROS poses use ENU coordinates and standard yaw. The EPIC CSV heading convention
is converted with `phi = yaw_enu - pi/2`. Each accepted vehicle pose is
projected onto the static raceline. The projector uses the previous unwrapped
`s` as a hint and unwraps across the closed-track seam; it globally reacquires
when a local projection fails.

The raceline CSV remains EPIC-compatible and includes centerline geometry,
speed profile, and static `e_min`/`e_max` bounds. See `docs/map_contract.md` for
the exact accepted columns. There is no perception, semantic map, occupancy
map, or online map-update path.

## ROS interfaces

The input is the existing
`xxcar_msgs/msg/EkfState` message from
`../xxCAR_msgs/msg/EkfState.msg`. The node subscribes to `ekf/state` by
default. The adapter maps:

- `header.stamp` to the solution pose time;
- ENU `geometry_msgs/Pose` quaternion to yaw;
- body-FLU horizontal twist magnitude to speed;
- `angular_velocity.z` to yaw rate;
- `linear_acceleration.x` to longitudinal acceleration;
- side slip, wheel torque, steering, and motor speed to their controller fields.

The steering, torque, and motor-speed fields are already expressed in the model's
physical units and are copied without scaling or offset. With measured control
feedback enabled, the first-step smoothness/rate cost uses those exact values.

The default safety gate requires attitude, horizontal velocity, horizontal
position, absolute yaw, estimator source, gyro source, VESC source, a valid
finite quaternion, finite required values, a fresh timestamp, and strictly
increasing pose time. The default maximum age is 0.10 s and future tolerance is
0.02 s. An EKF reset-counter change clears timestamp history, continuous `s`,
and the MPPI warm start. NaN sideslip is replaced with zero only below 0.3 m/s;
it is rejected while moving.

For bench testing, `require_solution_validity:=false` bypasses only the
`solution_status` bitmask gate. Source validity, quaternion, finite-value,
freshness, and timestamp-order checks remain enabled. The default stays `true`.
This mode was verified against the live EKF stream (`solution_status=83`) on an
isolated trajectory topic: the smoke checker passed with 334 trajectories at
33.37 Hz and 1,001 correlated state messages.

The default output is `xxcar_msgs/msg/VehicleControlTrajectory` on
`vehicle_control_trajectory`. It contains publication time, solution pose time,
`dt`, horizon, `horizon` ENU `(x,y)` states, and `horizon` steering/torque
controls. The terminal internal state `x[T]` is intentionally not published.
With `publish_direct_control:=true`, trajectory publication is disabled and the
first optimized command is published as `geometry_msgs/msg/Twist` on `cmd_vel`
by default. This follows `PID_lanekeeping`: `angular.z` is steering radians and
`linear.x` is the longitudinal command. In `control_mode: duty_cycle`, MPPI
wheel torque is converted using `direct_control_torque_to_throttle_scale` and
clamped to the configured direct throttle limits (default `[-1, 1]`). In
`control_mode: torque`, the wheel torque in Nm passes unchanged without that
mapping or clamp. Safety/output defaults, including
`require_solution_validity: true`, are loaded from `mppi.yaml`; explicit ROS
parameters override them. No UART message is published by `xx_mppi`.

The last successfully published solution is also printed to the node terminal
at 10 Hz with ROS `INFO` logging. It contains solve time, adaptive lambda and
steering/torque sigmas, the published physical steering/torque command, minimum
cost, ESS, finite rollout count, and solution age. `info_log_rate_hz` is set
in `mppi.yaml`; no additional diagnostics topic is created.

Optional visualization (`publish_visualization:=true`) publishes the expected
horizon as `nav_msgs/Path`, the best-weighted sampled rollouts as
`visualization_msgs/MarkerArray`, plus latched raceline, positive-`e` left
boundary, and negative-`e` right boundary paths. A dedicated latest-only worker
does the Frenet-to-map conversion and ROS visualization publication after each
requested snapshot. `visualization_rate_hz` and `num_rollouts` in `mppi.yaml`
control its reduced frequency and rollout count. All ROS endpoints use
best-effort reliability. Static paths retain transient-local durability and are
also republished at 1 Hz so late RViz displays work with volatile durability.
The sample launch enables visualization.

For vehicle testing, launch `ekf_mcu_driver` with its direct control sender
disabled:

```bash
ros2 launch ekf_mcu_driver ekf_mcu_driver.launch.py enable_send_control:=false
```

## Neural-model contract

The future PyTorch model replaces only the four-state body derivative. Its
TensorRT input is float32 `model_input` with shape `[batch,6]`:

```text
[yaw_rate, speed, sideslip, driven_wheel_speed, steering, wheel_torque]
```

Its float32 `state_derivative` output has shape `[batch,4]`:

```text
[yaw_acceleration, speed_acceleration, sideslip_rate,
 driven_wheel_speed_rate]
```

The provided exporter converts a TorchScript/module/checkpoint to ONNX and can
bake normalization into the graph. Build the serialized TensorRT engine on the
Orin because plans are tied to the TensorRT version and GPU. The neural path
currently supports Euler integration; RK4 is rejected because four batched
engine evaluations per substep materially alter the 100 Hz budget.

## Implemented commits

The development repository is `/home/aniket/Documents/nav_ws/src/xx_mppi`.
At handoff, `origin/main` is at `8eef8b9`; local `main` contains the following
five implementation commits plus the commit that adds this handoff document:

- `f88865e` — Configure TRX-4 Sport vehicle parameters.
- `e3ad262` — Model locked AWD Fiala driveline.
- `77468a9` — Add the `EkfState` ROS controller wrapper, launch file, safety
  validation, trajectory publication, smoke checker, tests, and documentation.
- `799b853` — Support symlink-installed default config resolution.
- `cae730d` — Expose CUDA headers to the TensorRT engine builder.

These commits have not been pushed by the assistant. They must be pushed or
otherwise synchronized to the vehicle workspace before rebuilding there.

There is also an existing user modification to `config/weights.yaml`. It was
deliberately left uncommitted and untouched by all assistant commits.

The current Orin workspace has an additional uncommitted TensorRT 10 build fix
in `tools/build_tensorrt_engine.cpp`, plus this handoff update. It must be
committed or otherwise preserved together with the earlier local commits.

## Build and config resolution

On the Orin workspace (reported as `~/fireball_ws`), build with:

```bash
cd ~/fireball_ws
source /opt/ros/humble/setup.bash

colcon build --symlink-install --packages-up-to xx_mppi \
  --cmake-args \
  -DCMAKE_BUILD_TYPE=Release \
  -DXX_MPPI_ENABLE_CUDA=ON \
  -DXX_MPPI_ENABLE_TENSORRT=ON \
  -DXX_MPPI_CUDA_ARCHITECTURES=87

source install/setup.bash
```

The launch file finds `share/xx_mppi/config` through the ament package index,
so the normal launch requires no absolute configuration path:

```bash
ros2 launch xx_mppi mppi.launch.py
```

With `--symlink-install`, the installed YAML files point to
`src/xx_mppi/config`; configuration-only changes are visible without rebuilding
after restarting the node. Paths inside `model.yaml`, including
`vehicle.yaml` and `model.plan`, resolve relative to the chosen config directory.
The raceline uses `raceline_path: "${current_map}"`; `current_map` names a map
folder and the loader derives `<folder>/<folder>_frenet_map.csv`. An external
configuration set can still be selected with
`config_directory:=/absolute/path/to/config`.

## Jetson build issues and fixes

The first Orin build used CUDA 12.6.68 and TensorRT 10.3.0.30. It failed while
compiling `xxcar_build_trt_engine` because `NvInferRuntimeBase.h` included
`cuda_runtime_api.h`, but that target did not inherit the CUDA Toolkit include
directory. Commit `cae730d` changes its link dependencies to include
`CUDA::cudart`, which supplies the required CUDA headers and runtime dependency.

After that fix, the build exposed a TensorRT 10 API ownership error:
`IOptimizationProfile` has a protected destructor, but the engine-builder tool
stored the builder-owned profile in `std::unique_ptr`. The current correction
keeps the result of `createOptimizationProfile()` as a non-owning pointer,
checks the return value of `addOptimizationProfile()`, and uses zero network
creation flags on TensorRT 10 because explicit batch is now unconditional.
This also removes the TensorRT 10 deprecation warning for `kEXPLICIT_BATCH`.

The GCC messages about `std::pair<float,float>` parameter passing changing in
GCC 10.1 are ABI notes, not build errors. Warnings that MPPI CMake variables are
unused by `xxcar_msgs` or `ekf_mcu_driver` are also harmless because colcon
passes the same CMake arguments to every package in `--packages-up-to`.

The following clean-cache controller build now succeeds on the Orin with CUDA
12.6.68 and TensorRT 10.3.0.30:

```bash
cd ~/fireball_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

colcon build --symlink-install --packages-select xx_mppi \
  --cmake-clean-cache \
  --event-handlers console_direct+ \
  --cmake-args \
  -DCMAKE_BUILD_TYPE=Release \
  -DXX_MPPI_ENABLE_CUDA=ON \
  -DXX_MPPI_ENABLE_TENSORRT=ON \
  -DXX_MPPI_CUDA_ARCHITECTURES=87
```

## Verification completed

- Clean CPU package build passed.
- Clean CUDA build targeting `sm_87` passed on the development host.
- Clean CUDA 12.6.68 and TensorRT 10.3.0.30 package build targeting `sm_87`
  passed on the Orin after correcting optimization-profile ownership.
- The default config now includes a sample 10 m diameter circular raceline and
  companion map metadata. The installed default config loads without an
  override, and a 2,001-sample CUDA solve completed with all rollouts finite.
- `sample_test.launch.py` uses isolated `/xx_mppi/sample_ekf_state` and
  `/xx_mppi/sample_vehicle_control_trajectory` topics and runs the controller
  without weakening the production EKF safety gate or exposing sample output
  to the normal actuator path. The ROS smoke checker passed with 336
  trajectories at 33.60 Hz and 1,001 correlated sample states.
- The CPU-only test build passed all 18 individual controller GTests; the
  colcon result reported 22 total test records with no errors or failures.
- In the CUDA/TensorRT build, 17 of 18 individual GTests passed. The remaining
  projection test could not create a CUDA stream because the current execution
  environment cannot initialize the Jetson memory manager; its CPU equivalent
  passed.
- The launch file loads and exposes all expected arguments.
- A clean `--symlink-install` build was verified: installed YAML files were
  symlinks to source YAML files and package-share lookup found them.
- An actual ROS DDS `EkfState` message traversed subscription, adapter validity,
  timestamp checks, Frenet projection, and reached the solver boundary.
- A synthetic DDS state/trajectory publisher passed the read-only pipeline
  checker with 301 trajectories at approximately 100.01 Hz.
- An actual GPU solve, TensorRT engine creation from a real ONNX model, and the
  100 Hz vehicle performance target still require runtime validation.

## Frame conventions review (2026-09-02)

The geometric frame chain was audited end to end and is correct. ENU yaw to CSV
`phi`, the left-positive lateral deviation, the `k = dphi/ds` sign, the Frenet
derivative, and `ToCartesian` all agree with each other and with every surveyed
map under `~/map`; the `test_frames` suite now pins each of them, including a
rollout parity check that the Frenet and Cartesian frames trace the same ENU
path on a surveyed loop.

`mppi.yaml` gained `frame: frenet | cartesian`. The Cartesian frame integrates
ENU position and heading and reprojects each rollout state onto the raceline on
the GPU, so costs, messages and visualization are unchanged. Measured on the
Orin with `K=1001, T=40`: 8.8 ms per solve in `frenet`, 13.7 ms in `cartesian`,
and the two frames' published paths agree within 0.1-0.2 m over the full-lap
horizon.

The current `xxcar_msgs/EkfState` contract provides steering radians, wheel
torque Nm, and motor speed m/s. The adapter passes all three through unchanged,
and `observation.use_measured_control_feedback` is enabled. Direct command
mapping remains separately configurable because it belongs to the downstream
actuator interface, not the EKF feedback path.
- `side_slip_rad` is exactly `atan2(v_y, v_x)` of the reported body twist and
  reaches 2.7 rad in recorded runs. That is the cause of the reported
  "path goes backward then forward": the projection's relative course heading is
  `yaw + beta - phi_path`, so a 2 rad sideslip pushes `|dphi|` past 90 degrees,
  `s_dot = V cos(dphi)` turns negative, and the rollout retreats along the track
  before the model turns it around. `ConditionedSideslip` now bounds the
  estimate once, at `observation.maximum_model_sideslip_rad`, and the projection
  and the body model both use that same value so the two frames share an initial
  condition. `MppiRosRuntime` logs a throttled warning whenever the bound bites.

Two solver defects were fixed alongside: the crash and sideslip latches were
being set from the initial state, which is shared by every sample and therefore
removed all boundary discrimination from the solve; and the expected-trajectory
average was linear over angle channels, which is wrong across the +/-pi seam and
matters most for the Cartesian ENU yaw.

Also worth knowing when reading RViz: `horizon: 40` at `dt: 0.1` with the 5 m/s
reference speed covers 20 m, which is 98% of the `last_run_sept1`/`iso1` lap and
123% of the `sept_2_00` lap. The expected path therefore wraps the whole loop and
draws over itself. Shorten `horizon`, shorten `dt`, or lower the map's speed
profile if a shorter preview is wanted.

Still open: with `lambda: 2.0` against a cost scale near 2e5, the first solves
after a reset are effectively an argmin over the sampled population (ESS 1.0).
Adaptive lambda climbs to roughly 580 and ESS to 4-20 within a few hundred
solves, but `ess_fraction_min/max` of `0.002 .. 0.02` targets only 2-20 of 1001
samples, which is close to argmin by design. Worth revisiting with the tuned
weights.

## Remaining blockers and next actions

1. Commit and push the local implementation and TensorRT 10 ownership fix when
   ready, while keeping the existing user change to `config/weights.yaml`
   separate unless it is intentionally included.
2. Replace the sample `config/raceline.csv` with the surveyed EPIC-format
   vehicle raceline, or select an external production config directory. The
   included circle is only for bench and ROS pipeline testing.
3. Confirm on a stationary bench test that the EKF feedback signs agree with
   positive-left steering and positive drive torque; the adapter intentionally
   performs no unit scale or offset.
4. Run the CUDA benchmark on the Orin after warm-up in the intended Jetson power
   mode. The 100 Hz target requires p99 solve time below 10 ms; it has not yet
   been demonstrated on the vehicle hardware.
5. With downstream actuation disabled or the wheels safely unloaded, inspect
   `/ekf/state` and `/vehicle_control_trajectory`, then run the smoke checker:

   ```bash
   ros2 topic hz /ekf/state
   ros2 topic echo /ekf/state --once
   ros2 topic hz /vehicle_control_trajectory
   ros2 topic echo /vehicle_control_trajectory --once
   ros2 run xx_mppi smoke_test_ros_pipeline.py --duration 10
   ```

6. Only enable the separate actuator/UART controller after the checker passes,
   timestamps/status bits are valid, coordinate projection matches the physical
   track, and steering/torque signs and units have been verified.

## Useful source documents

- `docs/architecture.md` — internal state, frames, MPPI flow, and robustness.
- `docs/map_contract.md` — raceline CSV schema.
- `docs/model_pipeline.md` — PyTorch/ONNX/TensorRT contract.
- `docs/ros_integration.md` — state validity and vehicle launch.
- `docs/orin_deployment.md` — Orin build and benchmark procedure.
