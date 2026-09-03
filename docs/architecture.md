# Architecture

The code retains the EPIC concepts and names: `MppiController`,
`MppiControllerBuilder`, `Raceline`, `ContinuousProjector`, `map_boundary`,
integrators, frames, and model backends.

## Rollout frame

`mppi.yaml` selects the frame the rollout integrates in with
`frame: frenet | cartesian`. Both frames share the same seven-wide float32
state, the same body models, the same cost function, and the same published
messages; only the trailing three slots and their derivative differ.

| Slot | `frenet` | `cartesian` |
|---|---|---|
| 5 | lateral deviation `e` [m], positive left of the raceline | east [m], map ENU |
| 6 | relative course heading `dphi` [rad] | north [m], map ENU |
| 7 | continuous path evolution `s` [m] | heading [rad], ROS ENU yaw |

`frenet` integrates the path terms against the raceline curvature and needs no
projection during the rollout. `cartesian` integrates ENU position and heading
directly and reprojects every rollout state onto the raceline so the cost
function still sees `(e, dphi, s)`. That projection is a device twin of
`Raceline::Project`, restricted to a window of segments around the previous
arc length and unwrapped against it, which keeps `s` loop-continuous exactly as
the host `ContinuousProjector` does. The window is sized from
`projection_window_m` in `model.yaml`.

`cartesian` costs more per solve because of that per-state projection, and its
accuracy no longer depends on the curvature column being consistent with the
surveyed centreline. `frenet` remains the default.

## Controller state

The internal state is float32 and ordered as:

1. yaw rate `r` [rad/s]
2. speed `V` [m/s]
3. sideslip `beta` [rad]
4. driven-wheel peripheral speed [m/s]; for `locked_awd`, this is the common
   effective peripheral speed of the rigidly coupled four-wheel driveline
5. lateral deviation `e` [m], positive left of the raceline
6. relative course heading `dphi` [rad]
7. continuous path evolution `s` [m]

Controls are `[steering_angle_rad, wheel_torque_nm]`. Default bounds are
`[-0.5, 0.5] rad` and the provisional `[-5, 5] Nm`; both are runtime YAML
parameters.

The TRX-4 profile enables `locked_awd`. Its Fiala model evaluates combined
longitudinal/lateral slip at both axles using the common wheel-speed state and
feeds both axle-force reactions into the equivalent driveline inertia. The
torque control is therefore total wheel-side driveline torque, not per-wheel
torque.

Every vehicle-state update is projected onto the static raceline. The projector
uses the previous unwrapped `s` as a local hint and unwraps the new result near
that hint, so `s` continues increasing through a closed-track seam. It globally
reacquires if the local hint produces no valid segment.

ROS uses standard ENU yaw (CCW from east). EPIC CSV `phi` uses tangent
`(-sin(phi), cos(phi))`, so `phi = yaw_enu - pi/2`. Course heading includes
sideslip before computing `dphi`.

The conventions that follow from that single choice, all of which the frame
tests pin down:

- positive `e` is the left normal of the path tangent, and
  `Raceline::ToCartesian` maps it back with `(E - e cos(phi), N - e sin(phi))`;
- `k` is `dphi/ds`, positive for a left (counter-clockwise) turn, matching the
  signed curvature `raceline_generator` writes;
- positive steering produces a positive yaw rate, so positive steering,
  positive `k` and positive `e` all mean "left";
- `e_max` is the left boundary and `e_min` the right one.

The `cmd_vel` steering command is the one place where this vehicle does not
follow the model. Set `direct_control_steering_scale: -1.0` when the actuator
chain downstream of `cmd_vel` is positive-right; `PID_lanekeeping` compensates
for the same hardware with its `invert_steering` parameter.

## Observation conditioning

`EkfState` carries the VESC channels in raw actuator units until
`steering_scale_to_rad`, `steering_offset_rad`, `torque_scale_to_nm` and
`motor_speed_scale_to_mps` are calibrated. Two guards keep an uncalibrated
stream from steering the solve:

- `observation.use_measured_control_feedback` (default `false`) makes the
  control-smoothness warm start use the controller's own last applied command.
  With the raw feedback the warm start clamps to a control bound every cycle,
  and the smoothness weight then drags the first optimized command toward that
  bound.
- `observation.maximum_model_sideslip_rad` bounds the sideslip through
  `ConditionedSideslip`, which the projection and the body model both call, so
  the two never disagree about the vehicle's course. The bound matters
  geometrically as well as physically: `dphi = yaw + beta - phi_path`, so an
  implausible sideslip pushes `|dphi|` past 90 degrees and makes
  `s_dot = V cos(dphi)` negative, which draws a planned path that retreats along
  the track before turning around.

## MPPI iteration

Each iteration shifts the previous nominal sequence by the measured fractional
pose-time change, samples centered time-smoothed Gaussian perturbations, inserts
warm-start/reference/braking special samples, clips in physical units, rolls out
all samples, and computes robust softmax weights. Non-finite rollouts are assigned
the worst finite cost; an all-invalid population falls back to uniform weights.

The robustness costs retained from EPIC include own-`s` speed and boundary
lookup, boundary ramp, latched discounted crash cost, sideslip shaping and a
latched kill threshold, lateral damping, wheel-slip cost, progress, effort, and
control smoothness. The crash and sideslip latches start at the first
integrated state rather than at the initial state, which every sample shares:
latching there would set the same flag across the whole population and remove
all boundary discrimination from the solve. Online lambda and diagonal-sigma adaptation use effective
sample size and selection-pressure statistics.

Analytic Fiala/kinematic rollouts are fused per sample. Neural dynamics perform
one TensorRT batch over all `K` samples per integration stage. CUDA buffers and
Philox RNG states persist between solves; only the small reference horizon,
initial state, and final trajectory/diagnostics cross the host/device boundary.

Analytic CUDA rollouts support Euler and RK4 with configurable substeps. The
TensorRT path currently supports Euler with configurable substeps; choosing RK4
for a neural model fails at startup because it would require four batched engine
evaluations per substep and materially changes the 100 Hz budget.
