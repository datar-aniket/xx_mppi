# Laser obstacle avoidance

`xx_mppi_node` consumes `sensor_msgs/msg/LaserScan` from `/scan`. The expected
scan frame is `laser`; the static `base_link <- laser` transform is looked up
once through TF and cached.

The node retains 100 ms of accepted EKF poses. Since a LaserScan stamp belongs
to its first ray, each valid return uses
`header.stamp + index * time_increment`. Poses are interpolated in SE(2), with
velocity/sideslip/yaw-rate extrapolation permitted only for the configured
short edge interval. A scan without complete timestamp coverage is dropped.

Scan callbacks only replace a one-element pending slot. A dedicated worker
deskews the newest scan, builds a rolling ENU signed Euclidean distance field,
and atomically publishes the immutable result. If the worker is busy, older
pending scans are discarded. Each completed scan replaces the previous field;
points are never accumulated. If scans stop, the last complete field remains
active. Cells outside the rolling field carry no observed-obstacle penalty.

The default 12 m by 12 m, 5 cm grid contains 57,600 float cells (225 KiB). The
CPU worker uses a linear-time two-pass distance transform. A new generation is
copied through persistent pinned host memory onto the controller's existing
CUDA stream before the next solve, so rollouts never observe a partial field.

Each predicted Frenet state is converted to ENU on the GPU. Clearance is the
minimum SDF value over a conservative three-circle approximation of the TRX-4
body. For clearance `d`, the shaping cost is:

```text
distance_weight * max(0, influence_distance_m - d)^2
```

When `d < latch_threshold_m`, the rollout's obstacle latch remains active for
the rest of the horizon and adds `latching_weight / (T + 1)` per state, matching
the existing sideslip-latch behavior. Parameters are in `config/obstacle.yaml`.

The field models only currently visible scan returns as static obstacles. It
does not infer occluded space or predict obstacle motion.

Benchmark the default field builder independently on the Orin with:

```bash
ros2 run xx_mppi xxcar_benchmark_sdf 500
```
