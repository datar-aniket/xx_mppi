#pragma once

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
  float solve_rate_hz{100.0F};
};

ControllerConfig LoadControllerConfig(const std::string & config_directory);

}  // namespace xxcar::mppi
