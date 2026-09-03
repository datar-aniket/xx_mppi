#pragma once

#include <cstdint>
#include <string>

#include "xx_mppi/dynamics/integrator.hpp"
#include "xx_mppi/dynamics/model.hpp"
#include "xx_mppi/types.hpp"

namespace xxcar::mppi {

struct ControllerConfig {
  MppiConfig mppi{};
  CostWeights costs{};
  VehicleParameters vehicle{};
  ModelKind model_kind{ModelKind::kDynamicBicycleFiala};
  IntegratorKind integrator{IntegratorKind::kEuler};
  std::string raceline_path;
  std::string neural_model_path;
  float projection_window_m{30.0F};
  // When false the warm start uses the controller's own last applied command
  // instead of the measured VESC feedback. The feedback channels arrive in raw
  // actuator units and must be calibrated with the ekf adapter scales before
  // they can be trusted as radians and newton-metres.
  bool use_measured_control_feedback{false};
  // Sideslip magnitude the body model is allowed to see. The projector still
  // uses the full measured course heading, so only the vehicle model and the
  // sideslip costs are protected from an implausible estimate.
  float maximum_model_sideslip_rad{0.8F};
  float solve_rate_hz{100.0F};
  float visualization_rate_hz{10.0F};
  std::uint32_t num_rollouts{15U};
};

ControllerConfig LoadControllerConfig(const std::string & config_directory);

}  // namespace xxcar::mppi
