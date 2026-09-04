#include "xx_mppi/controller/mppi_controller.hpp"

#include <algorithm>
#include <cmath>
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
    config_.mppi, config_.costs, config_.vehicle, config_.model_kind,
    raceline_, config_.integrator, config_.neural_model_path,
    config_.projection_window_m)
{
}

// Warm-start reference for control smoothness/rate costs. EkfState provides
// radians and newton-metres already, so preserve the feedback exactly. An
// out-of-bound measured actuator state is meaningful: the first feasible
// candidate should pay the actual transition back into the admissible range.
Control MppiController::PreviousControl(const VehicleObservation & observation) const {
  if (!config_.use_measured_control_feedback) {
    return last_applied_control_;
  }
  return Control{{observation.measured_steering_rad, observation.measured_torque_nm}};
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
  const std::uint32_t num_visualization_rollouts)
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
  const auto reference = raceline_.Sample(
    projection.s_m, config_.mppi.horizon, config_.mppi.dt_s);
  auto solution = optimizer_.Solve(
    initial, reference, previous_control, projection.s_m, shift_fraction, reset,
    num_visualization_rollouts);

  PlannedTrajectory result;
  result.solution_pose_time_ns = observation.pose_time_ns;
  result.dt_s = config_.mppi.dt_s;
  result.controls = std::move(solution.controls);
  result.diagnostics = solution.diagnostics;
  result.projection = projection;
  result.frame = config_.mppi.frame;
  result.states.reserve(config_.mppi.horizon);
  for (std::size_t i = 0; i < config_.mppi.horizon; ++i) {
    const auto point = StateToEnu(raceline_, solution.states[i], config_.mppi.frame);
    result.states.push_back(CartesianTrajectoryState{point.first, point.second});
  }
  result.sampled_rollouts = std::move(solution.sampled_rollouts);

  previous_pose_time_ns_ = observation.pose_time_ns;
  reset_next_ = false;
  return result;
}

void MppiController::RecordPublishedControl(const Control & control) noexcept {
  if (std::isfinite(control[kSteering]) && std::isfinite(control[kWheelTorque])) {
    last_applied_control_ = control;
  }
}

PlannedTrajectory MppiController::Plan(
  const VehicleObservation & observation,
  const std::uint32_t num_visualization_rollouts)
{
  (void)UpdateObservation(observation);
  return PlanLatest(num_visualization_rollouts);
}

void MppiController::Reset() noexcept {
  projector_.Reset();
  latest_.reset();
  previous_pose_time_ns_.reset();
  last_applied_control_ = Control{};
  reset_next_ = true;
}

}  // namespace xxcar::mppi
