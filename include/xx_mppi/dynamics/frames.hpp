#pragma once

#include <algorithm>
#include <cmath>

#include "xx_mppi/dynamics/model.hpp"
#include "xx_mppi/math.hpp"

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

// Cartesian counterpart of FrenetDerivative. Position advances along the
// course heading (yaw plus sideslip) and heading advances at the yaw rate, so
// the same body model drives both frames. Heading is ENU yaw; the raceline is
// only consulted by the cost function, which projects each state.
XXCAR_FRAME_HD inline StateDerivative CartesianDerivative(
  const State & state, const BodyDerivative & body_derivative)
{
  StateDerivative result{};
  for (std::size_t i = 0; i < kBodyStateDim; ++i) {
    result[i] = body_derivative[i];
  }
  const float course_enu = state[kHeadingEnu] + state[kSideslip];
  result[kEastM] = state[kSpeed] * cosf(course_enu);
  result[kNorthM] = state[kSpeed] * sinf(course_enu);
  result[kHeadingEnu] = state[kYawRate];
  return result;
}

}  // namespace xxcar::mppi

#undef XXCAR_FRAME_HD
