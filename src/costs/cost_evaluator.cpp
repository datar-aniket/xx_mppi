#include "xx_mppi/costs/cost_evaluator.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include "xx_mppi/costs/map_boundary.hpp"
#include "xx_mppi/costs/obstacle_cost.hpp"

namespace xxcar::mppi {

CostEvaluator::CostEvaluator(CostWeights weights, const float dt_s)
: weights_(std::move(weights)), dt_s_(dt_s)
{
  if (!(dt_s_ > 0.0F) || !std::isfinite(dt_s_)) {
    throw std::invalid_argument("cost dt must be finite and positive");
  }
}

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
  const ReferenceHorizon & reference, const Control & previous_control,
  const ObstacleField * const obstacle_field, const Raceline * const raceline,
  const ObstacleConfig * const obstacle_config, CostTerms * const terms) const
{
  if (states.size() != controls.size() + 1U ||
    reference.states.size() != states.size() || reference.controls.size() != controls.size())
  {
    throw std::invalid_argument("cost inputs do not share a horizon");
  }
  if (terms != nullptr) {
    *terms = CostTerms{};
    terms->minimum_clearance_m = obstacle_config != nullptr ?
      obstacle_config->maximum_distance_m : 0.0F;
  }

  float cost = 0.0F;
  const auto add = [&cost, terms](const std::size_t index, const float value) {
      cost += value;
      if (terms != nullptr) {
        terms->values[index] += value;
      }
    };
  bool crashed = false;
  bool excessive_sideslip = false;
  bool obstacle_latched = false;
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
      const float penalty = weights_.reference_tracking[i] * error * error;
      add(kTermReferenceTracking, penalty);
      if (terms != nullptr) {
        terms->reference_tracking[i] += penalty;
      }
    }

    const float desired_speed = InterpolateByS(
      reference.s_grid, reference.speed_profile, state[kPathEvolution]);
    const float speed_error = state[kSpeed] - desired_speed;
    const float speed_weight = speed_error > 0.0F ?
      weights_.velocity_profile * weights_.velocity_overspeed_multiplier :
      weights_.velocity_profile;
    add(kTermVelocityProfile, speed_weight * speed_error * speed_error);

    const float e_min = InterpolateByS(
      reference.s_grid, reference.e_min, state[kPathEvolution]);
    const float e_max = InterpolateByS(
      reference.s_grid, reference.e_max, state[kPathEvolution]);
    const auto boundary = EvaluateMapBoundary(
      state[kLateralDeviation], e_min, e_max, weights_.boundary,
      weights_.boundary_margin_m, weights_.crash_buffer_m);
    add(kTermBoundary, boundary.shaping_cost);
    crashed = crashed || boundary.violated;
    if (crashed) {
      add(kTermCrash, weights_.crash * discount);
      if (terms != nullptr && terms->first_crash_step == kNoCostLatch) {
        terms->first_crash_step = static_cast<std::uint16_t>(t);
      }
    }
    discount *= weights_.crash_discount;

    const float sideslip = state[kSideslip];
    add(kTermSideslip, weights_.sideslip * sideslip * sideslip);
    excessive_sideslip = excessive_sideslip ||
      std::abs(sideslip) > weights_.maximum_sideslip_rad;
    if (excessive_sideslip) {
      add(kTermSideslipKill, weights_.sideslip_kill / static_cast<float>(states.size()));
      if (terms != nullptr && terms->first_sideslip_step == kNoCostLatch) {
        terms->first_sideslip_step = static_cast<std::uint16_t>(t);
      }
    }
    if (obstacle_field != nullptr && raceline != nullptr && obstacle_config != nullptr) {
      const float clearance = VehicleObstacleClearance(
        state, *raceline, *obstacle_field, *obstacle_config);
      const float forward_clearance = VehicleForwardObstacleClearance(
        state, *raceline, *obstacle_field, *obstacle_config);
      const bool latched_before = obstacle_latched;
      const float obstacle_cost = EvaluateObstacleCost(
        clearance, forward_clearance, *obstacle_config, obstacle_latched, states.size());
      // EvaluateObstacleCost returns the deficit and latching penalties summed.
      // The latching half is a constant per step, so splitting them back apart
      // needs no second copy of the deficit shaping.
      const float latching = obstacle_latched && obstacle_config->enabled ?
        obstacle_config->latching_weight / static_cast<float>(states.size()) : 0.0F;
      add(kTermObstacleDistance, obstacle_cost - latching);
      add(kTermObstacleLatching, latching);
      if (terms != nullptr) {
        terms->minimum_clearance_m = std::min(terms->minimum_clearance_m, clearance);
        if (obstacle_latched && !latched_before && terms->first_obstacle_step == kNoCostLatch) {
          terms->first_obstacle_step = static_cast<std::uint16_t>(t);
        }
      }
    }

    const float lateral_error_rate = state[kSpeed] * std::sin(state[kRelativeHeading]);
    const float damped_error = lateral_error_rate +
      weights_.lateral_decay_rate * state[kLateralDeviation];
    add(kTermLateralDamping, weights_.lateral_damping * damped_error * damped_error);

    const float longitudinal_speed = std::max(
      state[kSpeed] * std::cos(state[kSideslip]), 1.0F);
    const float slip_ratio =
      (state[kDrivenWheelSpeed] - longitudinal_speed) / longitudinal_speed;
    const float excess_slip = std::max(
      std::abs(slip_ratio) - weights_.wheel_slip_band, 0.0F);
    add(kTermWheelSlip, weights_.wheel_slip * excess_slip * excess_slip);
  }

  add(kTermProgress, -weights_.progress *
    (states.back()[kPathEvolution] - states.front()[kPathEvolution]));

  Control prior = previous_control;
  for (std::size_t t = 0; t < controls.size(); ++t) {
    const auto & control = controls[t];
    for (std::size_t i = 0; i < kControlDim; ++i) {
      if (!std::isfinite(control[i])) {
        return std::numeric_limits<float>::infinity();
      }
      const float effort = weights_.control_effort[i] * control[i] * control[i];
      add(kTermControlEffort, effort);
      const float delta = control[i] - prior[i];
      const float smoothness = weights_.control_smoothness[i] * delta * delta;
      add(kTermControlSmoothness, smoothness);
      const float rate = delta / dt_s_;
      const float rate_penalty = weights_.control_rate[i] * rate * rate;
      add(kTermControlRate, rate_penalty);
      if (terms != nullptr) {
        terms->control_effort[i] += effort;
        terms->control_smoothness[i] += smoothness;
        terms->control_rate[i] += rate_penalty;
      }
    }
    const float acceleration = (states[t + 1U][kSpeed] - states[t][kSpeed]) / dt_s_;
    const bool speeding_up = acceleration >= 0.0F;
    const float acceleration_weight = speeding_up ?
      weights_.longitudinal_acceleration : weights_.longitudinal_deceleration;
    add(speeding_up ? kTermLongitudinalAcceleration : kTermLongitudinalDeceleration,
      acceleration_weight * acceleration * acceleration);
    prior = control;
  }
  return std::isfinite(cost) ? cost : std::numeric_limits<float>::infinity();
}

}  // namespace xxcar::mppi
