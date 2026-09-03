#include "xx_mppi/controller/cuda_mppi.hpp"

#include <stdexcept>
#include <utility>

namespace xxcar::mppi {

class CudaMppiController::Impl {
 public:
  explicit Impl(MppiConfig config) : config_(std::move(config)) {}

  [[noreturn]] MppiSolution Solve() const {
    throw std::runtime_error(
      "xx_mppi was built without CUDA; enable XX_MPPI_ENABLE_CUDA on the Orin target");
  }

  MppiConfig config_;
};

CudaMppiController::CudaMppiController(
  MppiConfig config, CostWeights, VehicleParameters, ModelKind, const Raceline &,
  IntegratorKind, std::string, float)
: impl_(std::make_unique<Impl>(std::move(config))) {}

CudaMppiController::~CudaMppiController() = default;
CudaMppiController::CudaMppiController(CudaMppiController &&) noexcept = default;
CudaMppiController & CudaMppiController::operator=(CudaMppiController &&) noexcept = default;

MppiSolution CudaMppiController::Solve(
  const State &, const ReferenceHorizon &, const Control &, float, float, bool,
  std::uint32_t)
{
  return impl_->Solve();
}

const MppiConfig & CudaMppiController::config() const noexcept { return impl_->config_; }
bool CudaMppiController::using_cuda() const noexcept { return false; }

}  // namespace xxcar::mppi
