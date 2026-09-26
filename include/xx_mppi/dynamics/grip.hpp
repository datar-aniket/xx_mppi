#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

#include "xx_mppi/dynamics/model.hpp"
#include "xx_mppi/dynamics/models/dynamic_bicycle_fiala_4ws.hpp"
#include "xx_mppi/types.hpp"

namespace xxcar::mppi {

// Grip the Fiala body model uses at one state and control. Host only.
struct GripSample {
  float ax_mps2{std::numeric_limits<float>::quiet_NaN()};
  float ay_mps2{std::numeric_limits<float>::quiet_NaN()};
  float front_utilization{std::numeric_limits<float>::quiet_NaN()};
  float rear_utilization{std::numeric_limits<float>::quiet_NaN()};
  float front_slip_angle_rad{std::numeric_limits<float>::quiet_NaN()};
  float rear_slip_angle_rad{std::numeric_limits<float>::quiet_NaN()};
  float rear_slip_ratio{std::numeric_limits<float>::quiet_NaN()};
};

[[nodiscard]] inline bool ModelHasTireForces(const ModelKind kind) noexcept {
  return kind == ModelKind::kDynamicBicycleFiala || kind == ModelKind::kDynamicBicycleFiala4ws;
}

// Load-weighted friction limit of the whole car [m/s^2], using the same static
// axle loads as the Fiala models.
[[nodiscard]] inline float FrictionLimitMps2(const VehicleParameters & parameters) noexcept {
  constexpr float gravity = 9.81F;
  const float wheelbase =
    std::max(parameters.cg_to_front_m + parameters.cg_to_rear_m, 1.0e-6F);
  const float front_share = parameters.cg_to_rear_m / wheelbase;
  return gravity * (parameters.front_friction_coefficient * front_share +
         parameters.rear_friction_coefficient * (1.0F - front_share));
}

// The front-steer-only Fiala model is the 4WS model at zero rear steer (the
// equivalence test holds them together), so both kinds are evaluated with the
// 4WS model. Other kinds have no tire forces and return NaN.
[[nodiscard]] inline GripSample PredictGrip(
  const VehicleParameters & parameters, const ModelKind kind, const BodyState & state,
  Control control)
{
  GripSample sample;
  if (!ModelHasTireForces(kind)) {
    return sample;
  }
  if (kind == ModelKind::kDynamicBicycleFiala) {
    control[kRearSteering] = 0.0F;
  }
  TireReport report;
  (void)DynamicBicycleFiala4ws(parameters).Evaluate(state, control, &report);
  const auto utilization = [](const TireForces & force, const float mu, const float load) {
      return std::hypot(force.longitudinal_n, force.lateral_n) /
             std::max(mu * load, 1.0e-6F);
    };
  sample.ax_mps2 = report.body_ax_mps2;
  sample.ay_mps2 = report.body_ay_mps2;
  sample.front_utilization = utilization(
    report.front, parameters.front_friction_coefficient, report.front_load_n);
  sample.rear_utilization = utilization(
    report.rear, parameters.rear_friction_coefficient, report.rear_load_n);
  sample.front_slip_angle_rad = report.front_slip_angle_rad;
  sample.rear_slip_angle_rad = report.rear_slip_angle_rad;
  sample.rear_slip_ratio = report.rear_slip_ratio;
  return sample;
}

}  // namespace xxcar::mppi
