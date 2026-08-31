#pragma once

#include <cmath>

#include "xx_mppi/math.hpp"

#if defined(__CUDACC__)
#define XXCAR_DYN_HD __host__ __device__
#else
#define XXCAR_DYN_HD
#endif

namespace xxcar::mppi {

struct TireForces {
  float lateral_n{};
  float longitudinal_n{};
};

XXCAR_DYN_HD inline TireForces FialaSimpleCoupling(
  float slip_angle_rad, const float cornering_stiffness_nprad,
  const float friction_coefficient, const float normal_load_n,
  const float requested_longitudinal_force_n = 0.0F)
{
  constexpr float alpha_max = 0.5F * kPi - 1.0e-2F;
  slip_angle_rad = clampf(slip_angle_rad, -alpha_max, alpha_max);
  const float friction_limit = fmaxf(friction_coefficient * normal_load_n, 1.0e-6F);
  float longitudinal = copysignf(
    fminf(fabsf(requested_longitudinal_force_n), 0.999999F * friction_limit),
    requested_longitudinal_force_n);
  longitudinal = fmaxf(longitudinal, -friction_limit * cosf(slip_angle_rad));
  const float eta = sqrtf(fmaxf(
    friction_limit * friction_limit - longitudinal * longitudinal, 0.0F)) /
    friction_limit;
  const float lateral_limit = fmaxf(eta * friction_limit, 1.0e-6F);
  const float saturation_angle = atanf(
    3.0F * lateral_limit / fmaxf(cornering_stiffness_nprad, 1.0e-6F));
  const float tangent = tanf(slip_angle_rad);
  const float stiffness = cornering_stiffness_nprad;
  const float unsaturated =
    -stiffness * tangent +
    stiffness * stiffness / (3.0F * lateral_limit) * tangent * fabsf(tangent) -
    stiffness * stiffness * stiffness / (27.0F * lateral_limit * lateral_limit) *
    tangent * tangent * tangent;
  const float lateral = fabsf(slip_angle_rad) > saturation_angle ?
    -copysignf(lateral_limit, slip_angle_rad) : unsaturated;
  return TireForces{lateral, longitudinal};
}

XXCAR_DYN_HD inline TireForces FialaCombinedSlip(
  float slip_angle_rad, const float slip_ratio,
  const float cornering_stiffness_nprad, const float friction_coefficient,
  const float normal_load_n)
{
  constexpr float alpha_max = 0.5F * kPi - 1.0e-2F;
  slip_angle_rad = clampf(slip_angle_rad, -alpha_max, alpha_max);
  const float friction_limit = fmaxf(friction_coefficient * normal_load_n, 1.0e-6F);
  const float stiffness = fmaxf(cornering_stiffness_nprad, 1.0e-6F);
  const float saturation_sigma = atanf(3.0F * friction_limit / stiffness);
  const float tangent = tanf(slip_angle_rad);
  const float sigma = fmaxf(sqrtf(tangent * tangent + slip_ratio * slip_ratio + 1.0e-12F),
    1.0e-10F);
  const float unsaturated =
    stiffness * sigma - stiffness * stiffness / (3.0F * friction_limit) * sigma * sigma +
    stiffness * stiffness * stiffness / (27.0F * friction_limit * friction_limit) *
    sigma * sigma * sigma;
  const float magnitude = fabsf(sigma) > saturation_sigma ? friction_limit : unsaturated;
  return TireForces{-magnitude * tangent / sigma, magnitude * slip_ratio / sigma};
}

}  // namespace xxcar::mppi

#undef XXCAR_DYN_HD
