#pragma once

#include <algorithm>
#include <cmath>

#if defined(__CUDACC__)
#define XXCAR_HD __host__ __device__
#else
#define XXCAR_HD
#endif

namespace xxcar::mppi {

constexpr float kPi = 3.14159265358979323846F;

XXCAR_HD inline float clampf(const float value, const float low, const float high) {
  return value < low ? low : (value > high ? high : value);
}

XXCAR_HD inline float wrap_to_pi(float angle) {
  while (angle > kPi) {
    angle -= 2.0F * kPi;
  }
  while (angle <= -kPi) {
    angle += 2.0F * kPi;
  }
  return angle;
}

// Standard ROS ENU yaw is CCW from east. EPIC-format CSV heading is CCW from
// north and uses tangent=(-sin(phi), cos(phi)).
XXCAR_HD inline float enu_yaw_to_heading_from_north(const float yaw_enu_rad) {
  return wrap_to_pi(yaw_enu_rad - 0.5F * kPi);
}

XXCAR_HD inline float heading_from_north_to_enu_yaw(const float heading_rad) {
  return wrap_to_pi(heading_rad + 0.5F * kPi);
}

}  // namespace xxcar::mppi

#undef XXCAR_HD
