# Learned dynamics model pipeline

The learned model replaces only the four-state body derivative. Frenet path
dynamics and numerical integration remain C++/CUDA.

The TensorRT contract has one float32 input named `model_input`, shape
`[batch,6]`:

`[yaw_rate, speed, sideslip, driven_wheel_speed, steering, wheel_torque]`

It has one float32 output named `state_derivative`, shape `[batch,4]`:

`[yaw_acceleration, speed_acceleration, sideslip_rate, driven_wheel_speed_rate]`

All values at the runtime boundary are SI physical units. If training used
normalization, provide its four vectors in JSON and the exporter bakes both
normalization and output de-normalization into ONNX.

**This model cannot steer the rear axle.** The six-wide input has no slot for a
rear steering angle, so `tensorrt_neural_derivative` is front-steer-only and
config load pins the rear control channel to zero for it, exactly as it does for
`kinematic_bicycle`. Supporting four-wheel steering here means retraining on
4WS driving data and widening the contract to seven inputs — which touches
`kInputWidth` in `dynamics/tensorrt_model.hpp`, the packing stride in both
`NeuralStageCostAndPack` and `PackNeuralInputs` in `cuda/cuda_mppi.cu`,
`tools/export_pytorch_dynamics.py`, and the dimension check in
`tools/build_tensorrt_engine.cpp`. Use `dynamic_bicycle_fiala_4ws` instead.

Export a TorchScript or serialized `nn.Module`:

```bash
ros2 run xx_mppi export_pytorch_dynamics.py model.pt model.onnx \
  --model-input concatenated --normalization-json normalization.json
```

For a state-dict checkpoint, add `--factory package.module:function`. For a
model whose Python forward method accepts body state and control separately, use
`--model-input split`.

Build the TensorRT plan on the Orin itself, because serialized plans are tied to
the TensorRT version and target GPU:

```bash
ros2 run xx_mppi xxcar_build_trt_engine model.onnx model.plan 2001 2001
```

The builder uses an explicit dynamic batch profile `1..2001`, an optimum batch
of 2001, FP16 kernels when supported, and float32 I/O. Set `name` in
`config/model.yaml` to `tensorrt_neural_derivative` and point
`neural_model_path` at the plan.

The exporter validates the Python model's `[batch,4]` output before writing. An
ONNX Python package is required by `torch.onnx.export`; it is a development-only
dependency and is not needed on the vehicle at runtime.
