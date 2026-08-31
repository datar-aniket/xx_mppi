#pragma once

#include <cstdint>

#include "xx_mppi/dynamics/model.hpp"
#include "xx_mppi/types.hpp"

namespace xxcar::mppi {

enum class IntegratorKind : std::uint8_t { kEuler = 0, kRungeKutta4 = 1 };

template<typename DerivativeFunction>
State IntegrateStep(
  const State & state, const Control & control, const float dt_s,
  const std::uint16_t substeps, const IntegratorKind kind,
  DerivativeFunction && derivative)
{
  State current = state;
  const auto count = substeps == 0 ? 1U : substeps;
  const float h = dt_s / static_cast<float>(count);
  for (std::uint16_t step = 0; step < count; ++step) {
    if (kind == IntegratorKind::kEuler) {
      const auto k1 = derivative(current, control);
      for (std::size_t i = 0; i < kStateDim; ++i) {
        current[i] += h * k1[i];
      }
      continue;
    }
    const auto k1 = derivative(current, control);
    State temporary{};
    for (std::size_t i = 0; i < kStateDim; ++i) {
      temporary[i] = current[i] + 0.5F * h * k1[i];
    }
    const auto k2 = derivative(temporary, control);
    for (std::size_t i = 0; i < kStateDim; ++i) {
      temporary[i] = current[i] + 0.5F * h * k2[i];
    }
    const auto k3 = derivative(temporary, control);
    for (std::size_t i = 0; i < kStateDim; ++i) {
      temporary[i] = current[i] + h * k3[i];
    }
    const auto k4 = derivative(temporary, control);
    for (std::size_t i = 0; i < kStateDim; ++i) {
      current[i] += h / 6.0F * (k1[i] + 2.0F * k2[i] + 2.0F * k3[i] + k4[i]);
    }
  }
  return current;
}

}  // namespace xxcar::mppi
