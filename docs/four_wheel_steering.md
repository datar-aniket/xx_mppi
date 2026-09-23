# Four-wheel steering

The car has an independently actuated rear steering servo. This describes how
the MPPI stack models and commands it, and what is still missing.

## Status

The solver, the dynamics model and the ROS command transport are in place. The
**MCU wire protocol is not**, so the rear servo does not yet move. The rear
control channel therefore ships pinned to zero, and must stay that way until the
firmware carries a rear angle — see "Enabling it" below.

## The control channel

`kControlDim` is 3: `[steering_angle_rad, wheel_torque_nm, rear_steering_angle_rad]`.

A channel is either **active** (positive sigma, and `control_min < control_max`)
or **pinned** (sigma, min and max all exactly zero, which makes `BuildCandidates`
clamp every candidate to zero). Config load rejects anything in between, because
a half-pinned channel — sampled but clamped flat, or zero-sigma with open bounds
— runs without doing anything and is almost always a mistake.

Config load pins the rear channel automatically whenever the selected model does
not steer the rear axle, which is every model except `dynamic_bicycle_fiala_4ws`.
The node logs which state it is in at startup.

## The model

`dynamic_bicycle_fiala_4ws` is a deliberate copy of `dynamic_bicycle_fiala`, not
a generalization of it, so the front-steer-only model stays a known-good
fallback that a mistake in the rear-steer terms cannot reach. The rear angle
enters in four places:

- the rear slip angle gains a `- delta_r` term;
- the rear contact-patch velocity is rotated into the steered rear wheel frame
  before forming the rear slip ratio, mirroring what the front already did;
- the rear yaw moment becomes `-b * (F_ry * cos(delta_r) + F_rx * sin(delta_r))`;
- the speed and sideslip rates use `cos(delta_r - beta)` / `sin(delta_r - beta)`
  where they previously used `cos(beta)` / `sin(beta)`.

The driveline balance is unchanged: it needs wheel-frame longitudinal forces,
which is what the tire model already returns. Rear steer reaches the wheel speed
through the rear slip ratio instead.

`Dynamics.ZeroRearSteerReproducesTheFrontSteerOnlyModel` is what keeps the two
copies in agreement — at `delta_r = 0` every rear term must collapse to the
front-steer-only form exactly. If you change either model, that test is the one
that matters.

## The command path

`geometry_msgs/Twist` has only two usable scalars (`angular.z`, `linear.x`) and
cannot express a rear angle, so four-wheel steering uses `xxcar_msgs/DirectControl`
instead. Set `direct_control_four_wheel:=true` to publish it on
`direct_control_four_wheel_topic` in place of the Twist; exactly one transport is
ever created, since publishing both would give the driver two command streams for
the same actuator. `cmd_vel` is left alone for `PID_lanekeeping`.

`ekf_mcu_driver` subscribes when `subscribe_four_wheel_control_topic` is set. It
converts and transmits the front angle and throttle exactly as the Twist path
does, and **drops the rear angle**: `RawDirectControlPacket` has no field for it.

## What is still missing

- **Firmware.** `RawDirectControlPacket` is 24 bytes with `uint8_t pad[7]` at
  offsets 17-23. A rear `float32` fits at 17-20 without changing the packet
  length, but needs a matching change in `~/xxCar_MCU/apps/companion/comp_proto.h`,
  which lives outside this workspace.
- **Feedback.** `EkfState` carries one `steering_angle`, and
  `RawVehicleStatePacket` has a single spare byte, so there is no rear steering
  measurement. The rear channel falls back to its commanded value.
- **Rear servo calibration.** The rear angle currently reuses the front's
  `steering_scale` and `steering_limit_rad`, which assumes the two servos share a
  calibration. Revisit once the rear is characterized.
- **The learned model.** `tensorrt_neural_derivative` is front-steer-only; see
  `model_pipeline.md`.

## Enabling it

Only after the firmware carries a rear angle and a bench test shows the rear
servo tracking a commanded one:

1. `config/model.yaml`: `name: dynamic_bicycle_fiala_4ws`
2. `config/mppi.yaml`: give `rear_steering_angle_rad` a nonzero `sigma` and open
   its `control_bounds`.
3. `config/weights.yaml` already carries rear entries mirroring the front
   steering weights. They must stay non-zero — a zero-weight channel is free to
   oscillate at no cost.

Enabling the model without the firmware makes MPPI plan rear-axle motion the car
cannot execute, which degrades tracking rather than improving it.

## Cost

The third channel costs about 18% of solve time even while pinned (median 1.16 ms
to 1.37 ms at K=2001, T=50 on the bench), because the sampling buffers and noise
generation scale with `kControlDim`. If that headroom is needed before rear steer
is actually in use, reduce `num_samples`.
