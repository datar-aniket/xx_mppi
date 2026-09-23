#include "xx_mppi/ros/direct_control_message.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace xxcar::mppi {

DirectControlMode ParseDirectControlMode(const std::string & value) {
  if (value == "duty_cycle") {
    return DirectControlMode::kDutyCycle;
  }
  if (value == "torque") {
    return DirectControlMode::kTorque;
  }
  throw std::invalid_argument(
    "unknown control_mode '" + value + "'; expected duty_cycle or torque");
}

const char * DirectControlModeName(DirectControlMode mode) {
  switch (mode) {
    case DirectControlMode::kDutyCycle:
      return "duty_cycle";
    case DirectControlMode::kTorque:
      return "torque";
  }
  throw std::invalid_argument("invalid direct control mode");
}

void ValidateDirectControlConfig(const DirectControlConfig & config) {
  static_cast<void>(DirectControlModeName(config.mode));
  if (config.topic.empty()) {
    throw std::invalid_argument("direct control topic must not be empty");
  }
  if (!std::isfinite(config.torque_to_throttle_scale) ||
    !std::isfinite(config.throttle_min) || !std::isfinite(config.throttle_max) ||
    !std::isfinite(config.steering_scale) || config.steering_scale == 0.0F ||
    !std::isfinite(config.steering_limit_rad) || !(config.steering_limit_rad > 0.0F))
  {
    throw std::invalid_argument("direct control conversion parameters must be finite");
  }
  if (config.throttle_min > config.throttle_max) {
    throw std::invalid_argument(
      "direct control throttle_min must not exceed throttle_max");
  }
}

namespace {

struct ConvertedControl {
  float steering;
  float rear_steering;
  float throttle;
};

// Shared by both transports so the Twist path and the DirectControl path can
// never disagree about sign, limit or torque mapping.
ConvertedControl ConvertFirstControl(
  const PlannedTrajectory & trajectory, const DirectControlConfig & config)
{
  ValidateDirectControlConfig(config);
  if (trajectory.controls.empty()) {
    throw std::invalid_argument("planned trajectory has no direct control sample");
  }

  const float steering_rad = trajectory.controls.front()[kSteering];
  const float wheel_torque_nm = trajectory.controls.front()[kWheelTorque];
  const float rear_steering_rad = trajectory.controls.front()[kRearSteering];
  if (!std::isfinite(steering_rad) || !std::isfinite(wheel_torque_nm) ||
    !std::isfinite(rear_steering_rad))
  {
    throw std::invalid_argument("planned direct control sample must be finite");
  }

  float output = wheel_torque_nm;
  if (config.mode == DirectControlMode::kDutyCycle) {
    const float unbounded_throttle = wheel_torque_nm * config.torque_to_throttle_scale;
    if (!std::isfinite(unbounded_throttle)) {
      throw std::invalid_argument("converted direct control throttle must be finite");
    }
    output = std::clamp(
      unbounded_throttle, config.throttle_min, config.throttle_max);
  }

  const float steering_command = std::clamp(
    steering_rad * config.steering_scale,
    -config.steering_limit_rad, config.steering_limit_rad);
  // The rear angle reuses the front's scale and limit, which assumes the two
  // servos share a calibration. Revisit when the rear servo is characterized.
  const float rear_steering_command = std::clamp(
    rear_steering_rad * config.steering_scale,
    -config.steering_limit_rad, config.steering_limit_rad);
  if (!std::isfinite(steering_command) || !std::isfinite(rear_steering_command)) {
    throw std::invalid_argument("converted direct control steering must be finite");
  }

  return ConvertedControl{steering_command, rear_steering_command, output};
}

}  // namespace

geometry_msgs::msg::Twist ToDirectControlMessage(
  const PlannedTrajectory & trajectory, const DirectControlConfig & config)
{
  const auto converted = ConvertFirstControl(trajectory, config);
  geometry_msgs::msg::Twist message;
  message.angular.z = static_cast<double>(converted.steering);
  message.linear.x = static_cast<double>(converted.throttle);
  return message;
}

xxcar_msgs::msg::DirectControl ToDirectControlMessage(
  const PlannedTrajectory & trajectory, const DirectControlConfig & config,
  const rclcpp::Time & stamp)
{
  const auto converted = ConvertFirstControl(trajectory, config);
  xxcar_msgs::msg::DirectControl message;
  message.stamp = stamp;
  message.steering_angle_rad = converted.steering;
  message.rear_steering_angle_rad = converted.rear_steering;
  message.throttle = converted.throttle;
  message.throttle_type = config.mode == DirectControlMode::kDutyCycle ?
    xxcar_msgs::msg::DirectControl::THROTTLE_DUTY_CYCLE :
    xxcar_msgs::msg::DirectControl::THROTTLE_TORQUE;
  return message;
}

}  // namespace xxcar::mppi
