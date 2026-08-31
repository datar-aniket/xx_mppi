# Architecture

The code retains the EPIC concepts and names: `MppiController`,
`MppiControllerBuilder`, `Raceline`, `ContinuousProjector`, `map_boundary`,
integrators, frames, and model backends.

## Controller state

The internal Frenet state is float32 and ordered as:

1. yaw rate `r` [rad/s]
2. speed `V` [m/s]
3. sideslip `beta` [rad]
4. driven-wheel peripheral speed [m/s]
5. lateral deviation `e` [m], positive left of the raceline
6. relative course heading `dphi` [rad]
7. continuous path evolution `s` [m]

Controls are `[steering_angle_rad, wheel_torque_nm]`. Default bounds are
`[-0.5, 0.5] rad` and the provisional `[-5, 5] Nm`; both are runtime YAML
parameters.

Every vehicle-state update is projected onto the static raceline. The projector
uses the previous unwrapped `s` as a local hint and unwraps the new result near
that hint, so `s` continues increasing through a closed-track seam. It globally
reacquires if the local hint produces no valid segment.

ROS uses standard ENU yaw (CCW from east). EPIC CSV `phi` uses tangent
`(-sin(phi), cos(phi))`, so `phi = yaw_enu - pi/2`. Course heading includes
sideslip before computing `dphi`.

## MPPI iteration

Each iteration shifts the previous nominal sequence by the measured fractional
pose-time change, samples centered time-smoothed Gaussian perturbations, inserts
warm-start/reference/braking special samples, clips in physical units, rolls out
all samples, and computes robust softmax weights. Non-finite rollouts are assigned
the worst finite cost; an all-invalid population falls back to uniform weights.

The robustness costs retained from EPIC include own-`s` speed and boundary
lookup, boundary ramp, latched discounted crash cost, sideslip shaping and a
latched kill threshold, lateral damping, wheel-slip cost, progress, effort, and
control smoothness. Online lambda and diagonal-sigma adaptation use effective
sample size and selection-pressure statistics.

Analytic Fiala/kinematic rollouts are fused per sample. Neural dynamics perform
one TensorRT batch over all `K` samples per integration stage. CUDA buffers and
Philox RNG states persist between solves; only the small reference horizon,
initial state, and final trajectory/diagnostics cross the host/device boundary.

Analytic CUDA rollouts support Euler and RK4 with configurable substeps. The
TensorRT path currently supports Euler with configurable substeps; choosing RK4
for a neural model fails at startup because it would require four batched engine
evaluations per substep and materially changes the 100 Hz budget.
