#include "xx_mppi/controller/mppi_controller.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "xx_mppi/math.hpp"

namespace xxcar::mppi {

MppiController::MppiController(ControllerConfig config, Raceline raceline)
: config_(std::move(config)),
  raceline_(std::move(raceline)),
  projector_(raceline_, config_.projection_window_m),
  optimizer_(
    config_.mppi, config_.costs, config_.vehicle, config_.model_kind,
    raceline_, config_.integrator, config_.neural_model_path)
{
}

Projection MppiController::UpdateObservation(const VehicleObservation & observation) {
  const float sideslip = std::isfinite(observation.sideslip_rad) ?
    observation.sideslip_rad :
    (std::abs(observation.speed_mps) < 0.3F ? 0.0F : observation.sideslip_rad);
  if (!std::isfinite(observation.east_m) || !std::isfinite(observation.north_m) ||
    !std::isfinite(observation.yaw_enu_rad) || !std::isfinite(observation.speed_mps) ||
    !std::isfinite(observation.yaw_rate_radps) || !std::isfinite(sideslip) ||
    !std::isfinite(observation.measured_torque_nm) ||
    !std::isfinite(observation.measured_steering_rad) ||
    !std::isfinite(observation.driven_wheel_speed_mps))
  {
    throw std::invalid_argument("vehicle observation contains a non-finite value");
  }

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

PlannedTrajectory MppiController::PlanLatest() {
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

  const State initial{{
    observation.yaw_rate_radps,
    observation.speed_mps,
    sideslip,
    observation.driven_wheel_speed_mps,
    projection.e_m,
    projection.relative_course_rad,
    projection.s_m}};
  const Control previous_control{{
    std::clamp(
      observation.measured_steering_rad,
      config_.mppi.control_min[kSteering], config_.mppi.control_max[kSteering]),
    std::clamp(
      observation.measured_torque_nm,
      config_.mppi.control_min[kWheelTorque], config_.mppi.control_max[kWheelTorque])}};
  const auto reference = raceline_.Sample(
    projection.s_m, config_.mppi.horizon, config_.mppi.dt_s);
  auto solution = optimizer_.Solve(
    initial, reference, previous_control, shift_fraction, reset);

  PlannedTrajectory result;
  result.solution_pose_time_ns = observation.pose_time_ns;
  result.dt_s = config_.mppi.dt_s;
  result.controls = std::move(solution.controls);
  result.diagnostics = solution.diagnostics;
  result.projection = projection;
  result.states.reserve(config_.mppi.horizon);
  for (std::size_t i = 0; i < config_.mppi.horizon; ++i) {
    const auto point = raceline_.ToCartesian(
      solution.states[i][kPathEvolution], solution.states[i][kLateralDeviation]);
    result.states.push_back(CartesianTrajectoryState{point.first, point.second});
  }

  previous_pose_time_ns_ = observation.pose_time_ns;
  reset_next_ = false;
  return result;
}

PlannedTrajectory MppiController::Plan(const VehicleObservation & observation) {
  (void)UpdateObservation(observation);
  return PlanLatest();
}

void MppiController::Reset() noexcept {
  projector_.Reset();
  latest_.reset();
  previous_pose_time_ns_.reset();
  reset_next_ = true;
}

}  // namespace xxcar::mppi
