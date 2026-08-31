#pragma once

#include <cmath>

#include "xx_mppi/dynamics/model.hpp"

#if defined(__CUDACC__)
#define XXCAR_KIN_HD __host__ __device__
#else
#define XXCAR_KIN_HD
#endif

namespace xxcar::mppi {

class KinematicBicycle {
 public:
  explicit XXCAR_KIN_HD KinematicBicycle(const VehicleParameters & parameters)
  : parameters_(parameters) {}

  [[nodiscard]] XXCAR_KIN_HD BodyDerivative Derivative(
    const BodyState & state, const Control & control) const
  {
    constexpr float relaxation_time_s = 0.05F;
    const float wheelbase = fmaxf(
      parameters_.cg_to_front_m + parameters_.cg_to_rear_m, 1.0e-6F);
    const float speed = state[kSpeed];
    const float steering = control[kSteering];
    const float beta_target = atanf(
      parameters_.cg_to_rear_m / wheelbase * tanf(steering));
    const float yaw_rate_target = speed * cosf(beta_target) * tanf(steering) / wheelbase;
    const float speed_rate = control[kWheelTorque] /
      fmaxf(parameters_.mass_kg * parameters_.wheel_radius_m, 1.0e-6F);
    return BodyDerivative{
      (yaw_rate_target - state[kYawRate]) / relaxation_time_s,
      speed_rate,
      (beta_target - state[kSideslip]) / relaxation_time_s,
      (speed - state[kDrivenWheelSpeed]) / relaxation_time_s};
  }

 private:
  VehicleParameters parameters_;
};

}  // namespace xxcar::mppi

#undef XXCAR_KIN_HD
