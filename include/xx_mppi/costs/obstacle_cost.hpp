#pragma once

#include "xx_mppi/reference/raceline.hpp"
#include "xx_mppi/types.hpp"

namespace xxcar::mppi {

[[nodiscard]] float VehicleObstacleClearance(
  const State & state, const Raceline & raceline, const ObstacleField & field,
  const ObstacleConfig & config);

[[nodiscard]] float EvaluateObstacleCost(
  float clearance_m, const ObstacleConfig & config, bool & latched,
  std::size_t horizon_states);

}  // namespace xxcar::mppi
