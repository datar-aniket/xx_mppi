#include "xx_mppi/dynamics/tensorrt_model.hpp"

#include <stdexcept>
#include <utility>

namespace xxcar::mppi {

class TensorRtDerivativeModel::Impl {};

TensorRtDerivativeModel::TensorRtDerivativeModel(const std::string &)
: impl_(std::make_unique<Impl>())
{
  throw std::runtime_error(
    "xx_mppi was built without TensorRT; install JetPack TensorRT development packages");
}

TensorRtDerivativeModel::~TensorRtDerivativeModel() = default;
TensorRtDerivativeModel::TensorRtDerivativeModel(TensorRtDerivativeModel &&) noexcept = default;
TensorRtDerivativeModel & TensorRtDerivativeModel::operator=(TensorRtDerivativeModel &&) noexcept =
  default;

void TensorRtDerivativeModel::InferAsync(const float *, float *, std::size_t, void *) {
  throw std::runtime_error("TensorRT is unavailable");
}

bool TensorRtDerivativeModel::available() const noexcept { return false; }

}  // namespace xxcar::mppi
