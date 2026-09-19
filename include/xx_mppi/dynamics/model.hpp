#pragma once

#include <array>
#include <cstdint>

#include "xx_mppi/types.hpp"

namespace xxcar::mppi {

enum class ModelKind : std::uint8_t {
  kKinematicBicycle = 0,
  kDynamicBicycleFiala = 1,
  kTensorRtNeuralDerivative = 2,
  kDynamicBicycleFiala4ws = 3,
};

// Front-steer-only models ignore control[kRearSteering] entirely. Config load
// pins that channel to zero for them so the solver cannot spend samples on a
// control the active model does not read.
constexpr bool ModelSteersRearAxle(const ModelKind kind) noexcept {
  return kind == ModelKind::kDynamicBicycleFiala4ws;
}

using StateDerivative = std::array<float, kStateDim>;
using BodyDerivative = std::array<float, kBodyStateDim>;

}  // namespace xxcar::mppi
