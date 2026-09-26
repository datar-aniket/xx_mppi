#include "xx_mppi/ros/grip_status_message.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "xx_mppi/dynamics/grip.hpp"

namespace xxcar::mppi {
namespace {

float FrictionCircleRemaining(const float limit, const float along, const float across) {
  return std::sqrt(std::max(limit * limit - across * across, 0.0F)) - std::abs(along);
}

}  // namespace

xxcar_msgs::msg::GripStatus ToGripStatusMessage(
  const VehicleObservation & observation, const PlannedTrajectory & trajectory,
  const VehicleParameters & vehicle, const ModelKind model_kind, GripMonitor & monitor)
{
  constexpr float nan = std::numeric_limits<float>::quiet_NaN();
  xxcar_msgs::msg::GripStatus message;
  message.header.stamp.sec = static_cast<std::int32_t>(observation.pose_time_ns / 1000000000LL);
  message.header.stamp.nanosec = static_cast<std::uint32_t>(
    observation.pose_time_ns % 1000000000LL);
  message.speed_mps = observation.speed_mps;

  const float limit = FrictionLimitMps2(vehicle);
  message.friction_limit_mps2 = limit;
  const float ax = observation.longitudinal_acceleration_mps2;
  const float ay = observation.lateral_acceleration_mps2;
  message.measured_ax_mps2 = ax;
  message.measured_ay_mps2 = ay;
  message.measured_utilization = std::hypot(ax, ay) / std::max(limit, 1.0e-6F);
  message.measured_lateral_remaining_mps2 = FrictionCircleRemaining(limit, ay, ax);
  message.measured_longitudinal_remaining_mps2 = FrictionCircleRemaining(limit, ax, ay);

  const std::size_t steps = std::min(trajectory.body_states.size(), trajectory.controls.size());
  message.horizon_dt_s = trajectory.dt_s;
  message.horizon_front_utilization.reserve(steps);
  message.horizon_rear_utilization.reserve(steps);
  message.horizon_ax_mps2.reserve(steps);
  message.horizon_ay_mps2.reserve(steps);
  GripSample now;
  float peak = nan;
  float peak_time_s = nan;
  for (std::size_t i = 0; i < steps; ++i) {
    const auto sample = PredictGrip(
      vehicle, model_kind, trajectory.body_states[i], trajectory.controls[i]);
    if (i == 0U) {
      now = sample;
    }
    message.horizon_front_utilization.push_back(sample.front_utilization);
    message.horizon_rear_utilization.push_back(sample.rear_utilization);
    message.horizon_ax_mps2.push_back(sample.ax_mps2);
    message.horizon_ay_mps2.push_back(sample.ay_mps2);
    const float use = std::max(sample.front_utilization, sample.rear_utilization);
    if (std::isfinite(use) && !(use <= peak)) {
      peak = use;
      peak_time_s = static_cast<float>(i) * trajectory.dt_s;
    }
  }
  message.predicted_ax_mps2 = now.ax_mps2;
  message.predicted_ay_mps2 = now.ay_mps2;
  message.predicted_front_utilization = now.front_utilization;
  message.predicted_rear_utilization = now.rear_utilization;
  message.predicted_front_slip_angle_rad = now.front_slip_angle_rad;
  message.predicted_rear_slip_angle_rad = now.rear_slip_angle_rad;
  message.predicted_rear_slip_ratio = now.rear_slip_ratio;
  message.horizon_peak_utilization = peak;
  message.horizon_peak_time_s = peak_time_s;

  const auto check = monitor.Update(
    static_cast<double>(observation.pose_time_ns) * 1.0e-9, observation.speed_mps, ax, ay,
    now.ax_mps2, now.ay_mps2, std::max(now.front_utilization, now.rear_utilization));
  message.acceleration_shortfall_mps2 = check.acceleration_shortfall_mps2;
  message.observed_peak_mu = check.observed_peak_mu;
  message.model_overestimates_grip = check.model_overestimates_grip;
  return message;
}

}  // namespace xxcar::mppi
