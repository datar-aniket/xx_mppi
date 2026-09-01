# MPPI port session handoff

Last updated: 2026-09-01

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
- Use real control units: steering in radians with bounds `[-0.5, 0.5]`, and
  total wheel-side torque in Nm with provisional bounds `[-5, 5]`.
- Keep both kinematic bicycle and dynamic Fiala models. Use Fiala initially.
- Later support a PyTorch neural model that outputs the same four body-state
  derivatives as the EPIC learned model. C++/CUDA retains integration and path
  dynamics.
- Default MPPI problem: `K=2001`, `T=50`, `dt=0.1 s`; target at least 100 Hz on
  the Orin NX.
- Runtime YAML changes, including sample count, do not require recompilation.
  Restart the node so it reloads the configuration and reallocates buffers.
- The controller publishes a ROS trajectory only. A separate downstream
  controller owns MCU/UART communication.
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
`ekf_mcu_driver/msg/EkfState` message from
`../ekf_mcu_driver/msg/EkfState.msg`. The node subscribes to `ekf/state` by
default. The adapter maps:

- `header.stamp` to the solution pose time;
- ENU `geometry_msgs/Pose` quaternion to yaw;
- body-FLU horizontal twist magnitude to speed;
- `angular_velocity.z` to yaw rate;
- `linear_acceleration.x` to longitudinal acceleration;
- side slip, wheel torque, steering, and motor speed to their controller fields.

The adapter supports these calibration parameters:

- `steering_scale_to_rad`;
- `steering_offset_rad`;
- `torque_scale_to_nm`;
- `motor_speed_scale_to_mps`.

The default safety gate requires attitude, horizontal velocity, horizontal
position, absolute yaw, estimator source, gyro source, VESC source, a valid
finite quaternion, finite required values, a fresh timestamp, and strictly
increasing pose time. The default maximum age is 0.10 s and future tolerance is
0.02 s. An EKF reset-counter change clears timestamp history, continuous `s`,
and the MPPI warm start. NaN sideslip is replaced with zero only below 0.3 m/s;
it is rejected while moving.

The output is `xxcar_msgs/msg/VehicleControlTrajectory` on
`vehicle_control_trajectory`. It contains publication time, solution pose time,
`dt`, horizon, `horizon` ENU `(x,y)` states, and `horizon` steering/torque
controls. The terminal internal state `x[T]` is intentionally not published.
No UART message is published by `xx_mppi`.

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
`raceline.csv`, `vehicle.yaml`, and `model.plan`, resolve relative to the chosen
config directory. An external set can still be selected explicitly with
`config_directory:=/absolute/path/to/config`.

## Latest Jetson build issue and fix

The first Orin build used CUDA 12.6.68 and TensorRT 10.3.0.30. It failed while
compiling `xxcar_build_trt_engine` because `NvInferRuntimeBase.h` included
`cuda_runtime_api.h`, but that target did not inherit the CUDA Toolkit include
directory. Commit `cae730d` changes its link dependencies to include
`CUDA::cudart`, which supplies the required CUDA headers and runtime dependency.

The GCC messages about `std::pair<float,float>` parameter passing changing in
GCC 10.1 are ABI notes, not build errors. Warnings that MPPI CMake variables are
unused by `xxcar_msgs` or `ekf_mcu_driver` are also harmless because colcon
passes the same CMake arguments to every package in `--packages-up-to`.

After synchronizing `cae730d`, retry just the controller package with:

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

This exact TensorRT 10.3 compilation still needs confirmation on the Orin; the
development host does not have TensorRT installed.

## Verification completed

- Clean CPU package build passed.
- Clean CUDA build targeting `sm_87` passed on the development host.
- All 18 individual controller GTests passed; the colcon result also reported
  22 total test records with no errors or failures.
- The launch file loads and exposes all expected arguments.
- A clean `--symlink-install` build was verified: installed YAML files were
  symlinks to source YAML files and package-share lookup found them.
- An actual ROS DDS `EkfState` message traversed subscription, adapter validity,
  timestamp checks, Frenet projection, and reached the solver boundary.
- A synthetic DDS state/trajectory publisher passed the read-only pipeline
  checker with 301 trajectories at approximately 100.01 Hz.
- The development host has no usable NVIDIA driver and no TensorRT libraries,
  so it cannot validate an actual GPU solve, TensorRT compilation, or the 100 Hz
  vehicle performance target.

## Remaining blockers and next actions

1. Synchronize the local implementation commits, especially `cae730d`, to the
   Orin and rebuild. Capture the first compiler/linker error if another
   TensorRT 10.3 API incompatibility appears.
2. Add the real EPIC-format `raceline.csv` to `xx_mppi/config` or select an
   external config directory. The production `model.yaml` references this file,
   but it was absent at the last check. The node intentionally fails fast
   without it.
3. Confirm the EKF/VESC field units on the actual MCU pipeline. In particular,
   calibrate steering to radians, torque to wheel-side Nm, and motor speed to
   m/s using driver constants (`VESC_STEER_K`, `VESC_TORQUE_K`,
   `VESC_SPEED_K`) or the equivalent ROS parameters.
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
