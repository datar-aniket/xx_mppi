# Four-wheel steering

The car has an independently actuated rear steering servo. This describes how
the MPPI stack models and commands it, and what is still missing.

## Status

The solver, the dynamics model, the ROS command transport, the MCU wire protocol
(`delta_rear` at bytes 24-27 of the 32-byte `DIRECT_CONTROL` payload) and the
rear servo calibration path are in place. The rear control channel still ships
pinned to zero until the rear servo is calibrated and bench-tested — see
"Enabling it" below.

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

`ekf_mcu_driver` subscribes when `subscribe_four_wheel_control_topic` is set
(`ekf_mcu.subscribe_four_wheel_control_topic` in `bringup_params.yaml`). With
`enable_steering_calibration` on, both angles arrive in radians and each axle is
mapped through its own fit: the front through `steering_calibration`, the rear
through `rear_steering_calibration`, both in the same
`xxCAR_bringup/calibration/steering_calibration.yaml`. The driver refuses to start
the four-wheel subscription with calibration on and no rear fit, because there is
no correct way to turn a rear angle into a servo command without one. Because
of that, the four-wheel transport requires `direct_control_steering_scale: 1.0`.

## Calibrating the rear servo

Same OptiTrack procedure as the front, with `--axle rear`:

1. Put the wheel rigid bodies on the rear wheels, or name rear bodies with
   `--left_wheel_body` / `--right_wheel_body`.
2. Run the driver with `ekf_mcu_enable_steering_calibration:=false` and
   `ekf_mcu_subscribe_four_wheel_control_topic:=direct_control`, so the sweep's
   normalised commands reach the servo unchanged. Nothing else may publish
   `cmd_vel`: the Twist path commands the rear centred.
3. `ros2 run robot_bringup calibrate_steering_ackermann.py --car_name carxx --axle rear --auto_sweep`

The script writes only the `rear_*` sections of `steering_calibration.yaml`, so a
front calibration is preserved (and a later front run preserves the rear). It
prints which way a positive command steers the rear wheels; the fit absorbs that
sign, so a mirrored rear linkage needs no special handling.

If the steering ADC feedback does not work on an axle, add `--no_feedback`: only
the command fit (from OptiTrack) is written, and the driver publishes that axle's
feedback raw. The feedback fit is also skipped automatically when the feedback
never changes during the sweep. Raw feedback is not an angle, so keep
`use_measured_control_feedback: false` in `mppi.yaml` for such an axle.

## What is still missing

- **Feedback.** `EkfState.steering_angle_rear` is the pulse last sent to the rear
  servo, not a measurement (converted to radians once the rear is calibrated).
  `MppiController::PreviousControl` does not read it yet and uses zero for the
  rear.
- **The learned model.** `tensorrt_neural_derivative` is front-steer-only; see
  `model_pipeline.md`.

## Enabling it

Only after the rear is calibrated and a bench test shows it tracking a commanded
angle in the right direction:

1. Driver: `enable_steering_calibration: true` and
   `subscribe_four_wheel_control_topic: direct_control`.
2. `config/mppi.yaml`: `direct_control_steering_scale: 1.0` (the front's
   calibration must then be enabled too, since it also receives radians).
3. `config/model.yaml`: `name: dynamic_bicycle_fiala_4ws`
4. `config/mppi.yaml`: give `rear_steering_angle_rad` a nonzero `sigma` and open
   its `control_bounds` inside the rear servo's calibrated range.
5. `config/weights.yaml` already carries rear entries mirroring the front
   steering weights. They must stay non-zero — a zero-weight channel is free to
   oscillate at no cost.
6. Launch `xx_mppi` with `direct_control_four_wheel:=true`.

## Cost

The third channel costs about 18% of solve time even while pinned (median 1.16 ms
to 1.37 ms at K=2001, T=50 on the bench), because the sampling buffers and noise
generation scale with `kControlDim`. If that headroom is needed before rear steer
is actually in use, reduce `num_samples`.
