# Orin NX deployment

The CMake default is CUDA architecture `87`, the Orin GPU. It can be overridden
with `-DXX_MPPI_CUDA_ARCHITECTURES=...` for a separate development GPU.

On the JetPack 6.2.2 / Ubuntu 22.04 / ROS 2 Humble target:

```bash
cd /path/to/nav_ws
source /opt/ros/humble/setup.bash
colcon build --packages-up-to xx_mppi \
  --symlink-install \
  --cmake-args -DCMAKE_BUILD_TYPE=Release \
  -DXX_MPPI_ENABLE_CUDA=ON \
  -DXX_MPPI_ENABLE_TENSORRT=ON \
  -DXX_MPPI_CUDA_ARCHITECTURES=87
```

Source `install/setup.bash` after building. The normal launch command needs no
configuration path because it finds `share/xx_mppi/config` through the ament
index. With `--symlink-install`, edits to files in `src/xx_mppi/config` are
visible through that installed path without rebuilding:

```bash
source install/setup.bash
ros2 launch xx_mppi mppi.launch.py
```

Relative asset paths in `config/model.yaml` (for example `raceline.csv`,
`vehicle.yaml`, and `model.plan`) are resolved relative to the same config
directory. An external configuration remains available with
`config_directory:=/absolute/path/to/config`.

TensorRT is optional at configure time so CPU/CUDA analytic development remains
possible before a neural model exists. Selecting neural dynamics at runtime
without TensorRT or a valid plan fails explicitly.

The 100 Hz requirement means the complete solve must remain below 10 ms,
including state/reference copies and output synchronization. Measure after a
warm-up period in the intended Jetson power mode and record median, p95, p99,
maximum, effective sample size, and finite rollout count. The controller reports
GPU-event `solve_time_ms`; end-to-end ROS callback latency should be measured
separately. Run the supplied warm-started analytic benchmark with:

```bash
ros2 run xx_mppi xxcar_benchmark_mppi /path/to/raceline.csv 1000
```

It prints GPU median/p95/p99/maximum, wall time per solve, minimum effective
sample size, finite-rollout count, and a p99 100 Hz verdict. The repository does
not claim the 100 Hz target until this benchmark is run on the actual Orin NX.
