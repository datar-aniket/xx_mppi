#pragma once

#include <array>
#include <cstdint>

#include "xx_mppi/types.hpp"

namespace xxcar::mppi {

enum class ModelKind : std::uint8_t {
  kKinematicBicycle = 0,
  kDynamicBicycleFiala = 1,
  kTensorRtNeuralDerivative = 2,
};

using StateDerivative = std::array<float, kStateDim>;
using BodyDerivative = std::array<float, kBodyStateDim>;

}  // namespace xxcar::mppi
