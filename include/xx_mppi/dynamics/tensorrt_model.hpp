#pragma once

#include <cstddef>
#include <memory>
#include <string>

namespace xxcar::mppi {

// Loads a TensorRT plan whose input is [batch, 6] in physical units:
// [yaw_rate, speed, sideslip, driven_wheel_speed, steering, wheel_torque].
// Its output is [batch, 4], the body-state derivative. Both buffers must be
// device-resident float32 arrays. The exporter may bake normalization into the
// ONNX graph, keeping this runtime contract independent of training details.
class TensorRtDerivativeModel {
 public:
  static constexpr const char * kInputName = "model_input";
  static constexpr const char * kOutputName = "state_derivative";
  static constexpr std::size_t kInputWidth = 6U;
  static constexpr std::size_t kOutputWidth = 4U;

  explicit TensorRtDerivativeModel(const std::string & engine_path);
  ~TensorRtDerivativeModel();

  TensorRtDerivativeModel(const TensorRtDerivativeModel &) = delete;
  TensorRtDerivativeModel & operator=(const TensorRtDerivativeModel &) = delete;
  TensorRtDerivativeModel(TensorRtDerivativeModel &&) noexcept;
  TensorRtDerivativeModel & operator=(TensorRtDerivativeModel &&) noexcept;

  // cuda_stream is a cudaStream_t passed as void* to keep CUDA/TensorRT out of
  // the public header on CPU-only developer machines.
  void InferAsync(
    const float * device_input, float * device_output,
    std::size_t batch_size, void * cuda_stream);

  [[nodiscard]] bool available() const noexcept;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace xxcar::mppi
