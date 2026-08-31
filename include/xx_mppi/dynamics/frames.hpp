#pragma once

#include <algorithm>
#include <cmath>

#include "xx_mppi/dynamics/model.hpp"

#if defined(__CUDACC__)
#define XXCAR_FRAME_HD __host__ __device__
#else
#define XXCAR_FRAME_HD
#endif

namespace xxcar::mppi {

XXCAR_FRAME_HD inline StateDerivative FrenetDerivative(
  const State & state, const BodyDerivative & body_derivative,
  const float curvature_inv_m)
{
  StateDerivative result{};
  for (std::size_t i = 0; i < kBodyStateDim; ++i) {
    result[i] = body_derivative[i];
  }
  const float denominator = fmaxf(
    1.0F - curvature_inv_m * state[kLateralDeviation], 0.05F);
  const float path_rate = state[kSpeed] * cosf(state[kRelativeHeading]) / denominator;
  result[kLateralDeviation] = state[kSpeed] * sinf(state[kRelativeHeading]);
  result[kRelativeHeading] =
    body_derivative[kSideslip] + state[kYawRate] - curvature_inv_m * path_rate;
  result[kPathEvolution] = path_rate;
  return result;
}

}  // namespace xxcar::mppi

#undef XXCAR_FRAME_HD
