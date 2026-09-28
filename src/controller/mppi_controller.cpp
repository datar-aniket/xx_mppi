#include "xx_mppi/controller/mppi_controller.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include "xx_mppi/math.hpp"

namespace xxcar::mppi {

float ConditionedSideslip(
  const float measured_sideslip_rad, const float speed_mps,
  const float maximum_rad) noexcept
{
  if (!std::isfinite(measured_sideslip_rad)) {
    return std::abs(speed_mps) < 0.3F ? 0.0F : measured_sideslip_rad;
  }
  return std::clamp(measured_sideslip_rad, -maximum_rad, maximum_rad);
}

MppiController::MppiController(ControllerConfig config, Raceline raceline)
: config_(std::move(config)),
  raceline_(std::move(raceline)),
  projector_(raceline_, config_.projection_window_m),
  optimizer_(
    config_.mppi, config_.costs, config_.vehicle, config_.obstacles, config_.model_kind,
    raceline_, config_.integrator, config_.neural_model_path,
    config_.projection_window_m)
{
}

void MppiController::UpdateObstacleField(const ObstacleField & field) {
  optimizer_.UpdateObstacleField(field);
  obstacle_points_ = field.points;
}

void MppiController::ClearObstacleField() {
  optimizer_.ClearObstacleField();
  obstacle_points_.clear();
}

std::optional<float> FindLeadGap(
  const Raceline & raceline, const float s0_m, const std::vector<Point2D> & points,
  const float lookahead_m, const float wall_margin_m)
{
  // Brute-force nearest centreline sample: ~100 samples against a few hundred
  // confirmed returns is well under a millisecond, where Raceline::Project
  // would scan the whole raceline for every point.
  constexpr float kStepM = 0.05F;
  const auto count = static_cast<std::size_t>(std::ceil(lookahead_m / kStepM)) + 1U;
  std::vector<ReferencePoint> samples;
  samples.reserve(count);
  float largest_bound = 0.0F;
  for (std::size_t i = 0; i < count; ++i) {
    samples.push_back(raceline.Interpolate(s0_m + static_cast<float>(i) * kStepM));
    largest_bound = std::max(
      largest_bound, std::max(std::abs(samples.back().e_min_m), samples.back().e_max_m));
  }
  // Nothing farther than this from the window's first sample can be inside it.
  const float reach = lookahead_m + largest_bound + kStepM;
  const auto & origin = samples.front();

  std::optional<float> gap;
  for (const auto & point : points) {
    const float origin_east = point.east_m - origin.east_m;
    const float origin_north = point.north_m - origin.north_m;
    if (origin_east * origin_east + origin_north * origin_north > reach * reach) {
      continue;
    }
    std::size_t nearest = 0U;
    float nearest_squared = std::numeric_limits<float>::infinity();
    for (std::size_t i = 0; i < count; ++i) {
      const float d_east = point.east_m - samples[i].east_m;
      const float d_north = point.north_m - samples[i].north_m;
      const float squared = d_east * d_east + d_north * d_north;
      if (squared < nearest_squared) {
        nearest_squared = squared;
        nearest = i;
      }
    }
    const auto & sample = samples[nearest];
    // EPIC path tangent is (-sin(phi), cos(phi)); positive e is its left normal,
    // as in Raceline::Project.
    const float tangent_east = -std::sin(sample.heading_from_north_rad);
    const float tangent_north = std::cos(sample.heading_from_north_rad);
    const float d_east = point.east_m - sample.east_m;
    const float d_north = point.north_m - sample.north_m;
    const float along = d_east * tangent_east + d_north * tangent_north;
    const float lateral = d_east * (-tangent_north) + d_north * tangent_east;
    // A nearest sample at either end with a large along-track offset means the
    // point lies outside the window rather than beside it.
    if (std::abs(along) > kStepM) {
      continue;
    }
    const float ahead = static_cast<float>(nearest) * kStepM + along;
    if (!(ahead > 0.0F && ahead <= lookahead_m) ||
      lateral <= sample.e_min_m + wall_margin_m || lateral >= sample.e_max_m - wall_margin_m)
    {
      continue;
    }
    gap = gap ? std::min(*gap, ahead) : ahead;
  }
  return gap;
}

Control SlewLimitControl(
  const Control & target, const Control & previous, const float elapsed_s,
  const std::array<float, kControlDim> & limit) noexcept
{
  Control result = target;
  const float elapsed = std::isfinite(elapsed_s) ? std::max(elapsed_s, 0.0F) : 0.0F;
  for (std::size_t channel = 0; channel < kControlDim; ++channel) {
    if (limit[channel] > 0.0F) {
      const float step = limit[channel] * elapsed;
      result[channel] = std::clamp(
        target[channel], previous[channel] - step, previous[channel] + step);
    }
  }
  return result;
}

// Warm-start reference for control smoothness/rate costs. EkfState provides
// radians and newton-metres already, so preserve the feedback exactly. An
// out-of-bound measured actuator state is meaningful: the first feasible
// candidate should pay the actual transition back into the admissible range.
Control MppiController::PreviousControl(const VehicleObservation & observation) const {
  if (!config_.use_measured_control_feedback) {
    return last_applied_control_;
  }
  // No rear steering measurement exists: EkfState carries a single steering
  // angle. The rear channel falls back to zero, which is exact while it is
  // pinned and an approximation once it is not.
  return Control{{observation.measured_steering_rad, observation.measured_torque_nm, 0.0F}};
}

Projection MppiController::UpdateObservation(const VehicleObservation & observation) {
  const float sideslip = ConditionedSideslip(
    observation.sideslip_rad, observation.speed_mps,
    config_.maximum_model_sideslip_rad);
  if (!std::isfinite(observation.east_m) || !std::isfinite(observation.north_m) ||
    !std::isfinite(observation.yaw_enu_rad) || !std::isfinite(observation.speed_mps) ||
    !std::isfinite(observation.yaw_rate_radps) || !std::isfinite(sideslip) ||
    !std::isfinite(observation.measured_torque_nm) ||
    !std::isfinite(observation.measured_steering_rad) ||
    !std::isfinite(observation.driven_wheel_speed_mps))
  {
    throw std::invalid_argument("vehicle observation contains a non-finite value");
  }

  // ENU yaw (CCW from east) becomes the EPIC CSV heading phi (CCW from north,
  // path tangent (-sin phi, cos phi)). The course heading adds sideslip so the
  // projection's relative heading is the direction the vehicle actually moves.
  const float body_heading = enu_yaw_to_heading_from_north(observation.yaw_enu_rad);
  const Projection projection = projector_.Update(
    observation.east_m, observation.north_m, body_heading,
    sideslip);
  if (!projection.valid) {
    reset_next_ = true;
    throw std::runtime_error("vehicle position could not be projected onto the raceline");
  }
  latest_ = PreparedObservation{observation, sideslip, projection};
  return projection;
}

PlannedTrajectory MppiController::PlanLatest(
  const std::uint32_t num_visualization_rollouts, const bool capture_cost_terms)
{
  if (!latest_) {
    throw std::runtime_error("no vehicle observation is available for planning");
  }
  const auto & observation = latest_->observation;
  const float sideslip = latest_->sideslip_rad;
  const Projection projection = latest_->projection;

  float shift_fraction = 0.0F;
  bool reset = reset_next_;
  if (previous_pose_time_ns_) {
    const double elapsed_s = static_cast<double>(
      observation.pose_time_ns - *previous_pose_time_ns_) * 1.0e-9;
    if (elapsed_s < 0.0 || elapsed_s > 1.0) {
      reset = true;
    } else {
      shift_fraction = static_cast<float>(elapsed_s / config_.mppi.dt_s);
    }
  }

  const bool cartesian = config_.mppi.frame == FrameKind::kCartesian;
  const State initial{{
    observation.yaw_rate_radps,
    observation.speed_mps,
    sideslip,
    observation.driven_wheel_speed_mps,
    cartesian ? observation.east_m : projection.e_m,
    cartesian ? observation.north_m : projection.relative_course_rad,
    cartesian ? observation.yaw_enu_rad : projection.s_m}};
  const Control previous_control = PreviousControl(observation);
  SpeedLimit speed_limit;
  std::optional<float> lead_gap;
  if (observation.no_overtake) {
    const auto & mode = config_.no_overtake;
    speed_limit.scale = mode.speed_scale;
    speed_limit.deceleration_mps2 = mode.deceleration_mps2;
    lead_gap = FindLeadGap(
      raceline_, projection.s_m, obstacle_points_, mode.lookahead_m, mode.wall_margin_m);
    if (lead_gap) {
      speed_limit.stop_s_m = projection.s_m + *lead_gap - mode.gap_minimum_m;
    }
  }
  optimizer_.SetVelocityOverspeedMultiplier(
    observation.no_overtake ? config_.no_overtake.overspeed_multiplier :
    config_.costs.velocity_overspeed_multiplier);
  auto reference = raceline_.Sample(
    projection.s_m, config_.mppi.horizon, config_.mppi.dt_s, speed_limit);
  if (lead_gap) {
    reference.pass_limit_s_m =
      projection.s_m + *lead_gap - config_.no_overtake.pass_limit_gap_m;
  }
  auto solution = optimizer_.Solve(
    initial, reference, previous_control, projection.s_m, shift_fraction, reset,
    num_visualization_rollouts, capture_cost_terms);

  PlannedTrajectory result;
  result.solution_pose_time_ns = observation.pose_time_ns;
  result.dt_s = config_.mppi.dt_s;
  result.controls = std::move(solution.controls);
  result.diagnostics = solution.diagnostics;
  result.projection = projection;
  result.frame = config_.mppi.frame;
  result.states.reserve(config_.mppi.horizon);
  result.body_states.reserve(config_.mppi.horizon);
  for (std::size_t i = 0; i < config_.mppi.horizon; ++i) {
    const auto point = StateToEnu(raceline_, solution.states[i], config_.mppi.frame);
    result.states.push_back(CartesianTrajectoryState{point.first, point.second});
    BodyState body;
    for (std::size_t j = 0; j < kBodyStateDim; ++j) {
      body[j] = solution.states[i][j];
    }
    result.body_states.push_back(body);
  }
  result.capture_id = solution.capture_id;
  result.no_overtake = observation.no_overtake;
  if (lead_gap) {
    result.lead_gap_m = *lead_gap;
  }

  previous_pose_time_ns_ = observation.pose_time_ns;
  reset_next_ = false;
  return result;
}

void MppiController::RecordPublishedControl(const Control & control) noexcept {
  for (std::size_t channel = 0; channel < kControlDim; ++channel) {
    if (!std::isfinite(control[channel])) {
      return;
    }
  }
  last_applied_control_ = control;
}

PlannedTrajectory MppiController::Plan(
  const VehicleObservation & observation,
  const std::uint32_t num_visualization_rollouts, const bool capture_cost_terms)
{
  (void)UpdateObservation(observation);
  return PlanLatest(num_visualization_rollouts, capture_cost_terms);
}

void MppiController::Reset() noexcept {
  projector_.Reset();
  latest_.reset();
  previous_pose_time_ns_.reset();
  last_applied_control_ = Control{};
  reset_next_ = true;
}

}  // namespace xxcar::mppi
