#pragma once

#include <vector>

#include "xx_mppi/dynamics/analytic_dynamics.hpp"
#include "xx_mppi/dynamics/integrator.hpp"
#include "xx_mppi/reference/raceline.hpp"
#include "xx_mppi/types.hpp"

namespace xxcar::mppi {

std::vector<State> RolloutAnalytic(
  const AnalyticDynamics & dynamics, const Raceline & raceline,
  const State & initial_state, const std::vector<Control> & controls,
  float dt_s, std::uint16_t integration_substeps,
  IntegratorKind integrator = IntegratorKind::kEuler);

}  // namespace xxcar::mppi
