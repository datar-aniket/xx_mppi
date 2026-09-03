#include "xx_mppi/ros/ekf_state_adapter.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace xxcar::mppi {
namespace {

using EkfState = xxcar_msgs::msg::EkfState;

void RequireBits(
  const std::uint8_t value, const std::uint8_t required, const char * label)
{
  if ((value & required) != required) {
    throw std::invalid_argument(std::string("EKF state is missing required ") + label);
  }
}

float FiniteFloat(const double value, const char * label) {
  if (!std::isfinite(value) ||
    std::abs(value) > static_cast<double>(std::numeric_limits<float>::max()))
  {
    throw std::invalid_argument(std::string("EKF state has invalid ") + label);
  }
  return static_cast<float>(value);
}

}  // namespace

void ValidateEkfStateAdapterConfig(const EkfStateAdapterConfig & config) {
  if (!std::isfinite(config.steering_scale_to_rad) ||
    !std::isfinite(config.steering_offset_rad) ||
    !std::isfinite(config.torque_scale_to_nm) ||
    !std::isfinite(config.motor_speed_scale_to_mps) ||
    config.steering_scale_to_rad == 0.0F || config.torque_scale_to_nm == 0.0F ||
    config.motor_speed_scale_to_mps == 0.0F)
  {
    throw std::invalid_argument("EKF state adapter scales must be finite and nonzero");
  }
}

VehicleObservation ToVehicleObservation(
  const EkfState & message, const EkfStateAdapterConfig & config)
{
  ValidateEkfStateAdapterConfig(config);

  if (config.require_solution_validity) {
    std::uint8_t required_solution = static_cast<std::uint8_t>(
      EkfState::SOLUTION_STATUS_ATTITUDE_VALID |
      EkfState::SOLUTION_STATUS_VEL_HORIZ |
      EkfState::SOLUTION_STATUS_POS_HORIZ);
    if (config.require_absolute_yaw) {
      required_solution = static_cast<std::uint8_t>(
        required_solution | EkfState::SOLUTION_STATUS_YAW_ABSOLUTE);
    } else if ((message.solution_status & static_cast<std::uint8_t>(
        EkfState::SOLUTION_STATUS_YAW_RELATIVE |
        EkfState::SOLUTION_STATUS_YAW_ABSOLUTE)) == 0U)
    {
      throw std::invalid_argument("EKF state has no valid yaw solution");
    }
    RequireBits(message.solution_status, required_solution, "solution status bits");
  }

  std::uint8_t required_sources = static_cast<std::uint8_t>(
    EkfState::SOURCE_VALID_ESTIMATOR | EkfState::SOURCE_VALID_GYRO);
  if (config.require_vesc) {
    required_sources = static_cast<std::uint8_t>(
      required_sources | EkfState::SOURCE_VALID_VESC);
  }
  RequireBits(message.source_valid, required_sources, "source validity bits");

  if (message.header.stamp.sec < 0 || message.header.stamp.nanosec >= 1'000'000'000U) {
    throw std::invalid_argument("EKF state has an invalid header timestamp");
  }
  const auto pose_time_ns =
    static_cast<std::int64_t>(message.header.stamp.sec) * 1'000'000'000LL +
    static_cast<std::int64_t>(message.header.stamp.nanosec);

  const double qx = message.pose.orientation.x;
  const double qy = message.pose.orientation.y;
  const double qz = message.pose.orientation.z;
  const double qw = message.pose.orientation.w;
  const double norm_squared = qx * qx + qy * qy + qz * qz + qw * qw;
  if (!std::isfinite(norm_squared) || norm_squared < 1.0e-12) {
    throw std::invalid_argument("EKF state has an invalid orientation quaternion");
  }
  const double inverse_norm = 1.0 / std::sqrt(norm_squared);
  const double nx = qx * inverse_norm;
  const double ny = qy * inverse_norm;
  const double nz = qz * inverse_norm;
  const double nw = qw * inverse_norm;
  const double yaw = std::atan2(
    2.0 * (nw * nz + nx * ny),
    1.0 - 2.0 * (ny * ny + nz * nz));

  const float velocity_x = FiniteFloat(message.twist.linear.x, "forward velocity");
  const float velocity_y = FiniteFloat(message.twist.linear.y, "lateral velocity");
  const float sideslip = message.side_slip_rad;
  if (!std::isfinite(sideslip) && !std::isnan(sideslip)) {
    throw std::invalid_argument("EKF state has an invalid sideslip");
  }

  VehicleObservation observation;
  observation.pose_time_ns = pose_time_ns;
  observation.east_m = FiniteFloat(message.pose.position.x, "east position");
  observation.north_m = FiniteFloat(message.pose.position.y, "north position");
  observation.yaw_enu_rad = FiniteFloat(yaw, "ENU yaw");
  observation.speed_mps = std::hypot(velocity_x, velocity_y);
  observation.yaw_rate_radps = FiniteFloat(message.angular_velocity.z, "yaw rate");
  observation.longitudinal_acceleration_mps2 = FiniteFloat(
    message.linear_acceleration.x, "longitudinal acceleration");
  observation.sideslip_rad = sideslip;
  observation.measured_torque_nm = FiniteFloat(
    static_cast<double>(message.wheel_torque_nm) * config.torque_scale_to_nm,
    "wheel torque");
  observation.measured_steering_rad = FiniteFloat(
    static_cast<double>(message.steering_angle) * config.steering_scale_to_rad +
    config.steering_offset_rad, "steering angle");
  observation.driven_wheel_speed_mps = FiniteFloat(
    static_cast<double>(message.motor_speed_ms) * config.motor_speed_scale_to_mps,
    "motor speed");
  observation.status = static_cast<std::uint32_t>(message.solution_status) |
    (static_cast<std::uint32_t>(message.source_valid) << 8U);
  return observation;
}

void ValidateObservationTime(
  const std::int64_t pose_time_ns, const std::int64_t now_ns,
  const std::optional<std::int64_t> previous_pose_time_ns,
  const double maximum_age_s, const double future_tolerance_s)
{
  if (pose_time_ns <= 0) {
    throw std::invalid_argument("EKF state pose timestamp must be positive");
  }
  if (!std::isfinite(maximum_age_s) || maximum_age_s < 0.0 ||
    !std::isfinite(future_tolerance_s) || future_tolerance_s < 0.0)
  {
    throw std::invalid_argument("state timing limits must be finite and nonnegative");
  }
  const double age_s = static_cast<double>(now_ns - pose_time_ns) * 1.0e-9;
  if (maximum_age_s > 0.0 && age_s > maximum_age_s) {
    throw std::invalid_argument("EKF state is stale");
  }
  if (age_s < -future_tolerance_s) {
    throw std::invalid_argument("EKF state timestamp is too far in the future");
  }
  if (previous_pose_time_ns && pose_time_ns <= *previous_pose_time_ns) {
    throw std::invalid_argument("EKF state timestamp is duplicate or out of order");
  }
}

}  // namespace xxcar::mppi
