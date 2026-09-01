# xxCar CUDA/TensorRT MPPI

This package ports the tested EPIC MPPI controller architecture to ROS 2 Humble
and Jetson Orin NX. It is a new implementation for `nav_ws`; EPIC is only the
behavioral reference. There are no perception, semantic-map, or online-mapping
dependencies.

The default runtime problem is `K=2001`, `T=50`, and `dt=0.1 s`. Controls are
physical steering angle in radians and driven-wheel torque in Nm. Analytic
kinematic and dynamic Fiala models execute as fused CUDA rollouts. A learned
PyTorch body-derivative model can be exported through ONNX and run batched with
TensorRT; the same Frenet integrator supplies the path dynamics.

The ROS output is `xxcar_msgs/msg/VehicleControlTrajectory` on
`vehicle_control_trajectory`. It contains aligned arrays of exactly `T` planar
states and `T` controls; internal terminal state `x[T]` is deliberately omitted.
No UART packets are emitted.

The runnable ROS node subscribes to `ekf_mcu_driver/msg/EkfState` on
`ekf/state`, validates estimator/VESC status and sample freshness, projects each
accepted ENU pose into continuous Frenet coordinates, and resets its warm start
when the EKF reset counter changes:

```bash
cd /home/aniket/Documents/nav_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-up-to xx_mppi \
  --cmake-args -DCMAKE_BUILD_TYPE=Release \
  -DXX_MPPI_ENABLE_CUDA=ON \
  -DXX_MPPI_ENABLE_TENSORRT=ON \
  -DXX_MPPI_CUDA_ARCHITECTURES=87
source install/setup.bash
ros2 launch xx_mppi mppi.launch.py
```

The launch file finds the installed package share through the ament index and
uses its `config` directory by default. Consequently, `model.yaml` may keep
`raceline.csv`, `vehicle.yaml`, and `model.plan` as relative paths; they resolve
against that directory. Use `config_directory:=/absolute/alternate/config`
only when intentionally loading a configuration outside this package.

See:

- `docs/architecture.md` for state ordering, frames, and controller flow.
- `docs/map_contract.md` for the EPIC-compatible static CSV format.
- `docs/model_pipeline.md` for PyTorch-to-TensorRT conversion.
- `docs/ros_integration.md` for state validity, parameters, and vehicle startup.
- `docs/orin_deployment.md` for the JetPack build and benchmark procedure.
