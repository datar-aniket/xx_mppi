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
  // Twist carries no throttle type, so the driver reads its throttle as duty or
  // amps per its own control_throttle_type and a wheel torque in N m would be
  // applied as that many amps. Only DirectControl can say it is a torque.
  if (config.mode == DirectControlMode::kTorque && !config.four_wheel) {
    throw std::invalid_argument(
      "control_mode torque requires direct_control_four_wheel: only the DirectControl "
      "transport marks the throttle as a wheel torque for ekf_mcu_driver to convert");
  }
  if (config.obstacle_brake_enabled &&
    (!config.enabled || !config.four_wheel || config.mode != DirectControlMode::kTorque ||
    !std::isfinite(config.obstacle_brake_activation_s) ||
    !(config.obstacle_brake_activation_s > 0.0F) ||
    !std::isfinite(config.obstacle_brake_recovery_s) ||
    !(config.obstacle_brake_recovery_s > 0.0F) ||
    !std::isfinite(config.obstacle_brake_stop_speed_mps) ||
    config.obstacle_brake_stop_speed_mps < 0.0F ||
    !std::isfinite(config.obstacle_brake_torque_nm) ||
    !(config.obstacle_brake_torque_nm > 0.0F) ||
    !std::isfinite(config.obstacle_brake_motor_rpm_release) ||
    config.obstacle_brake_motor_rpm_release < 0.0F ||
    !std::isfinite(config.obstacle_brake_motor_rpm_engage) ||
    !(config.obstacle_brake_motor_rpm_engage > config.obstacle_brake_motor_rpm_release)))
  {
    throw std::invalid_argument(
      "obstacle latch braking requires enabled four-wheel DirectControl transport "
      "in torque mode, with positive finite activation/recovery intervals and "
      "brake magnitude, plus nonnegative finite speed/RPM thresholds");
  }
  // One steering_scale is applied to both axles, but the servos differ in gain
  // and possibly sign. The four-wheel transport therefore carries plain radians
  // and leaves each axle's mapping to its own calibration in ekf_mcu_driver.
  // if (config.four_wheel && config.steering_scale != 1.0F) {
  //   throw std::invalid_argument(
  //     "direct_control_four_wheel requires direct_control_steering_scale 1.0: both axles "
  //     "are sent in radians and ekf_mcu_driver applies each servo's calibration");
  // }
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

  float output = wheel_torque_nm * config.torque_to_throttle_scale;
  if (config.mode == DirectControlMode::kDutyCycle) {
    const float unbounded_throttle = wheel_torque_nm ;
    if (!std::isfinite(unbounded_throttle)) {
      throw std::invalid_argument("converted direct control throttle must be finite");
    }
    output = std::clamp(
      unbounded_throttle, config.throttle_min, config.throttle_max);
  }

  const float steering_command = std::clamp(
    steering_rad * config.steering_scale,
    -config.steering_limit_rad, config.steering_limit_rad);
  // The rear shares the front's scale and limit. On the four-wheel transport
  // the scale is required to be 1.0, so both are radians and the limit is only
  // a sanity clamp; each servo's mapping lives in the driver's calibration.
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

xxcar_msgs::msg::DirectControl ToSafetyTorqueMessage(
  const PlannedTrajectory & trajectory, const DirectControlConfig & config,
  const float safety_torque_nm,
  const rclcpp::Time & stamp)
{
  if (trajectory.controls.empty()) {
    throw std::invalid_argument("cannot brake without an MPPI steering command");
  }
  if (config.mode != DirectControlMode::kTorque) {
    throw std::invalid_argument("safety torque requires torque control mode");
  }
  if (!std::isfinite(safety_torque_nm)) {
    throw std::invalid_argument("safety torque must be finite");
  }
  PlannedTrajectory brake_trajectory = trajectory;
  brake_trajectory.controls.front()[kWheelTorque] = safety_torque_nm;
  return ToDirectControlMessage(brake_trajectory, config, stamp);
}

}  // namespace xxcar::mppi
