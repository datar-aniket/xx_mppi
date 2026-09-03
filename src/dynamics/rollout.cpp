#include "xx_mppi/dynamics/rollout.hpp"

#include <stdexcept>

namespace xxcar::mppi {

std::vector<State> RolloutAnalytic(
  const AnalyticDynamics & dynamics, const Raceline & raceline,
  const State & initial_state, const std::vector<Control> & controls,
  const float dt_s, const std::uint16_t integration_substeps,
  const IntegratorKind integrator)
{
  if (!(dt_s > 0.0F)) {
    throw std::invalid_argument("rollout dt must be positive");
  }
  std::vector<State> states(controls.size() + 1U);
  states.front() = initial_state;
  for (std::size_t t = 0; t < controls.size(); ++t) {
    const auto derivative = [&](const State & state, const Control & control) {
        const float curvature = dynamics.frame() == FrameKind::kCartesian ?
          0.0F : raceline.Interpolate(state[kPathEvolution]).curvature_inv_m;
        return dynamics.Derivative(state, control, curvature);
      };
    states[t + 1U] = IntegrateStep(
      states[t], controls[t], dt_s, integration_substeps, integrator, derivative);
  }
  return states;
}

}  // namespace xxcar::mppi
