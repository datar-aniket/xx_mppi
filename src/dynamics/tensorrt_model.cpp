#include "xx_mppi/dynamics/tensorrt_model.hpp"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace xxcar::mppi {
namespace {

class Logger final : public nvinfer1::ILogger {
 public:
  void log(const Severity severity, const char * message) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cerr << "[TensorRT] " << message << '\n';
    }
  }
};

std::vector<char> ReadBinary(const std::string & path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) {
    throw std::runtime_error("cannot open TensorRT engine: " + path);
  }
  const auto end = stream.tellg();
  if (end <= 0) {
    throw std::runtime_error("TensorRT engine is empty: " + path);
  }
  std::vector<char> bytes(static_cast<std::size_t>(end));
  stream.seekg(0, std::ios::beg);
  stream.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  if (!stream) {
    throw std::runtime_error("failed to read TensorRT engine: " + path);
  }
  return bytes;
}

void ValidateTensor(
  const nvinfer1::ICudaEngine & engine, const char * name,
  const nvinfer1::TensorIOMode expected_mode, const std::int32_t width)
{
  if (engine.getTensorIOMode(name) != expected_mode) {
    throw std::runtime_error(std::string("TensorRT engine is missing tensor '") + name + "'");
  }
  if (engine.getTensorDataType(name) != nvinfer1::DataType::kFLOAT) {
    throw std::runtime_error(std::string("TensorRT tensor '") + name + "' must be float32");
  }
  const auto dimensions = engine.getTensorShape(name);
  if (dimensions.nbDims != 2 || dimensions.d[1] != width) {
    throw std::runtime_error(
      std::string("TensorRT tensor '") + name + "' must have shape [batch," +
      std::to_string(width) + "]");
  }
}

}  // namespace

class TensorRtDerivativeModel::Impl {
 public:
  explicit Impl(const std::string & path) {
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) {
      throw std::runtime_error("TensorRT createInferRuntime failed");
    }
    const auto bytes = ReadBinary(path);
    engine_.reset(runtime_->deserializeCudaEngine(bytes.data(), bytes.size()));
    if (!engine_) {
      throw std::runtime_error(
        "TensorRT could not deserialize the engine; plans must be built on the target Orin");
    }
    ValidateTensor(
      *engine_, TensorRtDerivativeModel::kInputName,
      nvinfer1::TensorIOMode::kINPUT,
      static_cast<std::int32_t>(TensorRtDerivativeModel::kInputWidth));
    ValidateTensor(
      *engine_, TensorRtDerivativeModel::kOutputName,
      nvinfer1::TensorIOMode::kOUTPUT,
      static_cast<std::int32_t>(TensorRtDerivativeModel::kOutputWidth));
    context_.reset(engine_->createExecutionContext());
    if (!context_) {
      throw std::runtime_error("TensorRT createExecutionContext failed");
    }
  }

  void InferAsync(
    const float * input, float * output, const std::size_t batch_size,
    void * stream_pointer)
  {
    if (batch_size == 0U || batch_size > static_cast<std::size_t>(INT32_MAX)) {
      throw std::invalid_argument("TensorRT batch size is outside int32 range");
    }
    const auto batch = static_cast<std::int32_t>(batch_size);
    if (!context_->setInputShape(
        TensorRtDerivativeModel::kInputName,
        nvinfer1::Dims2{batch, static_cast<std::int32_t>(TensorRtDerivativeModel::kInputWidth)}))
    {
      throw std::runtime_error("TensorRT rejected the model input shape");
    }
    if (!context_->setTensorAddress(
        TensorRtDerivativeModel::kInputName, const_cast<float *>(input)) ||
      !context_->setTensorAddress(TensorRtDerivativeModel::kOutputName, output))
    {
      throw std::runtime_error("TensorRT rejected an inference buffer address");
    }
    const auto stream = reinterpret_cast<cudaStream_t>(stream_pointer);
    if (!context_->enqueueV3(stream)) {
      throw std::runtime_error("TensorRT enqueueV3 failed");
    }
  }

 private:
  Logger logger_{};
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
};

TensorRtDerivativeModel::TensorRtDerivativeModel(const std::string & engine_path)
: impl_(std::make_unique<Impl>(engine_path)) {}

TensorRtDerivativeModel::~TensorRtDerivativeModel() = default;
TensorRtDerivativeModel::TensorRtDerivativeModel(TensorRtDerivativeModel &&) noexcept = default;
TensorRtDerivativeModel & TensorRtDerivativeModel::operator=(TensorRtDerivativeModel &&) noexcept =
  default;

void TensorRtDerivativeModel::InferAsync(
  const float * device_input, float * device_output,
  const std::size_t batch_size, void * cuda_stream)
{
  impl_->InferAsync(device_input, device_output, batch_size, cuda_stream);
}

bool TensorRtDerivativeModel::available() const noexcept { return true; }

}  // namespace xxcar::mppi
