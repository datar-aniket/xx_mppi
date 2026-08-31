#include "xx_mppi/costs/cost_evaluator.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "xx_mppi/costs/map_boundary.hpp"

namespace xxcar::mppi {

float CostEvaluator::InterpolateByS(
  const std::vector<float> & s_grid, const std::vector<float> & values, const float s)
{
  if (s_grid.size() != values.size() || s_grid.empty()) {
    throw std::invalid_argument("invalid reference interpolation arrays");
  }
  const auto upper = std::upper_bound(s_grid.begin(), s_grid.end(), s);
  if (upper == s_grid.begin()) {
    return values.front();
  }
  if (upper == s_grid.end()) {
    return values.back();
  }
  const std::size_t high = static_cast<std::size_t>(upper - s_grid.begin());
  const std::size_t low = high - 1U;
  const float fraction = (s - s_grid[low]) / (s_grid[high] - s_grid[low]);
  return values[low] + fraction * (values[high] - values[low]);
}

float CostEvaluator::Evaluate(
  const std::vector<State> & states, const std::vector<Control> & controls,
  const ReferenceHorizon & reference, const Control & previous_control) const
{
  if (states.size() != controls.size() + 1U ||
    reference.states.size() != states.size() || reference.controls.size() != controls.size())
  {
    throw std::invalid_argument("cost inputs do not share a horizon");
  }

  float cost = 0.0F;
  bool crashed = false;
  bool excessive_sideslip = false;
  float discount = 1.0F;
  for (std::size_t t = 0; t < states.size(); ++t) {
    const auto & state = states[t];
    const auto & desired = reference.states[t];
    for (const float value : state) {
      if (!std::isfinite(value)) {
        return std::numeric_limits<float>::infinity();
      }
    }
    for (std::size_t i = 0; i < kStateDim; ++i) {
      const float error = state[i] - desired[i];
      cost += weights_.reference_tracking[i] * error * error;
    }

    const float desired_speed = InterpolateByS(
      reference.s_grid, reference.speed_profile, state[kPathEvolution]);
    const float speed_error = state[kSpeed] - desired_speed;
    const float speed_weight = speed_error > 0.0F ?
      weights_.velocity_profile * weights_.velocity_overspeed_multiplier :
      weights_.velocity_profile;
    cost += speed_weight * speed_error * speed_error;

    const float e_min = InterpolateByS(
      reference.s_grid, reference.e_min, state[kPathEvolution]);
    const float e_max = InterpolateByS(
      reference.s_grid, reference.e_max, state[kPathEvolution]);
    const auto boundary = EvaluateMapBoundary(
      state[kLateralDeviation], e_min, e_max, weights_.boundary,
      weights_.boundary_margin_m, weights_.crash_buffer_m);
    cost += boundary.shaping_cost;
    crashed = crashed || boundary.violated;
    if (crashed) {
      cost += weights_.crash * discount;
    }
    discount *= weights_.crash_discount;

    const float sideslip = state[kSideslip];
    cost += weights_.sideslip * sideslip * sideslip;
    excessive_sideslip = excessive_sideslip ||
      std::abs(sideslip) > weights_.maximum_sideslip_rad;
    if (excessive_sideslip) {
      cost += weights_.sideslip_kill / static_cast<float>(states.size());
    }

    const float lateral_error_rate = state[kSpeed] * std::sin(state[kRelativeHeading]);
    const float damped_error = lateral_error_rate +
      weights_.lateral_decay_rate * state[kLateralDeviation];
    cost += weights_.lateral_damping * damped_error * damped_error;

    const float longitudinal_speed = std::max(
      state[kSpeed] * std::cos(state[kSideslip]), 1.0F);
    const float slip_ratio =
      (state[kDrivenWheelSpeed] - longitudinal_speed) / longitudinal_speed;
    const float excess_slip = std::max(
      std::abs(slip_ratio) - weights_.wheel_slip_band, 0.0F);
    cost += weights_.wheel_slip * excess_slip * excess_slip;
  }

  cost -= weights_.progress *
    (states.back()[kPathEvolution] - states.front()[kPathEvolution]);

  Control prior = previous_control;
  for (const auto & control : controls) {
    for (std::size_t i = 0; i < kControlDim; ++i) {
      if (!std::isfinite(control[i])) {
        return std::numeric_limits<float>::infinity();
      }
      cost += weights_.control_effort[i] * control[i] * control[i];
      const float delta = control[i] - prior[i];
      cost += weights_.control_smoothness[i] * delta * delta;
    }
    prior = control;
  }
  return std::isfinite(cost) ? cost : std::numeric_limits<float>::infinity();
}

}  // namespace xxcar::mppi
