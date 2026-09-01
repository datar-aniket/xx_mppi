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
ros2 launch xx_mppi mppi.launch.py config_directory:=/absolute/path/to/config
```

See:

- `docs/architecture.md` for state ordering, frames, and controller flow.
- `docs/map_contract.md` for the EPIC-compatible static CSV format.
- `docs/model_pipeline.md` for PyTorch-to-TensorRT conversion.
- `docs/ros_integration.md` for state validity, parameters, and vehicle startup.
- `docs/orin_deployment.md` for the JetPack build and benchmark procedure.
