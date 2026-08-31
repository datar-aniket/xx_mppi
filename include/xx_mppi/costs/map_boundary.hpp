#pragma once

#include <cmath>

#if defined(__CUDACC__)
#define XXCAR_BOUNDARY_HD __host__ __device__
#else
#define XXCAR_BOUNDARY_HD
#endif

namespace xxcar::mppi {

struct MapBoundaryEvaluation {
  float shaping_cost{};
  bool violated{};
};

// Static-track corridor cost shared by the CPU oracle and CUDA rollout. Bounds
// are evaluated at each rollout state's own s before calling this function.
XXCAR_BOUNDARY_HD inline MapBoundaryEvaluation EvaluateMapBoundary(
  const float lateral_deviation_m, const float e_min_m, const float e_max_m,
  const float shaping_weight, const float requested_margin_m,
  const float crash_buffer_m)
{
  const float upper_margin = fminf(
    fmaxf(0.9F * e_max_m, 1.0e-3F), requested_margin_m);
  const float lower_margin = fminf(
    fmaxf(0.9F * -e_min_m, 1.0e-3F), requested_margin_m);
  const float upper = fmaxf(
    lateral_deviation_m - (e_max_m - upper_margin), 0.0F) / upper_margin;
  const float lower = fmaxf(
    (e_min_m + lower_margin) - lateral_deviation_m, 0.0F) / lower_margin;
  return MapBoundaryEvaluation{
    shaping_weight * (upper * upper + lower * lower),
    lateral_deviation_m > e_max_m - crash_buffer_m ||
    lateral_deviation_m < e_min_m + crash_buffer_m};
}

}  // namespace xxcar::mppi

#undef XXCAR_BOUNDARY_HD
