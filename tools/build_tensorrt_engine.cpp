#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

constexpr const char * kInputName = "model_input";

class Logger final : public nvinfer1::ILogger {
 public:
  void log(const Severity severity, const char * message) noexcept override {
    if (severity <= Severity::kINFO) {
      std::cerr << "[TensorRT] " << message << '\n';
    }
  }
};

std::uint32_t ParseBatch(const char * value, const char * label) {
  const auto parsed = std::stoul(value);
  if (parsed == 0UL || parsed > static_cast<unsigned long>(INT32_MAX)) {
    throw std::invalid_argument(std::string(label) + " must be in [1, INT32_MAX]");
  }
  return static_cast<std::uint32_t>(parsed);
}

}  // namespace

int main(int argc, char ** argv) {
  if (argc < 3 || argc > 5) {
    std::cerr << "usage: xxcar_build_trt_engine MODEL.onnx MODEL.plan "
      "[OPT_BATCH=2001] [MAX_BATCH=2001]\n";
    return 2;
  }
  try {
    const std::uint32_t opt_batch = argc >= 4 ? ParseBatch(argv[3], "OPT_BATCH") : 2001U;
    const std::uint32_t max_batch = argc >= 5 ? ParseBatch(argv[4], "MAX_BATCH") : opt_batch;
    if (max_batch < opt_batch) {
      throw std::invalid_argument("MAX_BATCH must be greater than or equal to OPT_BATCH");
    }

    Logger logger;
    std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
    if (!builder) {
      throw std::runtime_error("createInferBuilder failed");
    }
    const std::uint32_t explicit_batch =
      1U << static_cast<std::uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    std::unique_ptr<nvinfer1::INetworkDefinition> network(
      builder->createNetworkV2(explicit_batch));
    std::unique_ptr<nvonnxparser::IParser> parser(
      nvonnxparser::createParser(*network, logger));
    std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    if (!network || !parser || !config) {
      throw std::runtime_error("could not create TensorRT builder objects");
    }
    if (!parser->parseFromFile(argv[1], static_cast<int>(nvinfer1::ILogger::Severity::kINFO))) {
      throw std::runtime_error("ONNX parsing failed");
    }
    if (network->getNbInputs() != 1 || std::string(network->getInput(0)->getName()) != kInputName) {
      throw std::runtime_error("ONNX must contain one input named 'model_input'");
    }
    const auto dimensions = network->getInput(0)->getDimensions();
    if (dimensions.nbDims != 2 || dimensions.d[1] != 6) {
      throw std::runtime_error("model_input must have shape [batch,6]");
    }

    std::unique_ptr<nvinfer1::IOptimizationProfile> profile(
      builder->createOptimizationProfile());
    if (!profile ||
      !profile->setDimensions(
        kInputName, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims2{1, 6}) ||
      !profile->setDimensions(
        kInputName, nvinfer1::OptProfileSelector::kOPT,
        nvinfer1::Dims2{static_cast<std::int32_t>(opt_batch), 6}) ||
      !profile->setDimensions(
        kInputName, nvinfer1::OptProfileSelector::kMAX,
        nvinfer1::Dims2{static_cast<std::int32_t>(max_batch), 6}))
    {
      throw std::runtime_error("failed to configure TensorRT batch profile");
    }
    config->addOptimizationProfile(profile.release());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30U);
    if (builder->platformHasFastFp16()) {
      config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    std::unique_ptr<nvinfer1::IHostMemory> plan(
      builder->buildSerializedNetwork(*network, *config));
    if (!plan) {
      throw std::runtime_error("TensorRT engine build failed");
    }
    std::ofstream output(argv[2], std::ios::binary | std::ios::trunc);
    if (!output) {
      throw std::runtime_error(std::string("cannot open output plan: ") + argv[2]);
    }
    output.write(static_cast<const char *>(plan->data()),
      static_cast<std::streamsize>(plan->size()));
    if (!output) {
      throw std::runtime_error("failed while writing TensorRT plan");
    }
    std::cout << "wrote " << argv[2] << " for batch 1.." << max_batch
      << " (FP16=" << (builder->platformHasFastFp16() ? "yes" : "no") << ")\n";
    return 0;
  } catch (const std::exception & error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
