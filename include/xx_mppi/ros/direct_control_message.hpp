#pragma once

#include <string>

#include <geometry_msgs/msg/twist.hpp>
#include <rclcpp/time.hpp>
#include <xxcar_msgs/msg/direct_control.hpp>

#include "xx_mppi/controller/mppi_controller.hpp"

namespace xxcar::mppi {

enum class DirectControlMode {
  kDutyCycle,
  kTorque,
};

[[nodiscard]] DirectControlMode ParseDirectControlMode(const std::string & value);
[[nodiscard]] const char * DirectControlModeName(DirectControlMode mode);

// PID_lanekeeping-compatible direct actuator output. angular.z is steering in
// radians; linear.x is either mapped duty cycle or unchanged MPPI wheel torque,
// according to mode.
struct DirectControlConfig {
  bool enabled{false};
  std::string topic{"cmd_vel"};
  DirectControlMode mode{DirectControlMode::kDutyCycle};
  float torque_to_throttle_scale{1.0F};
  float throttle_min{-1.0F};
  float throttle_max{1.0F};
  // MPPI steering is positive-left, matching the positive-left yaw rate its
  // vehicle model produces. Set this to -1 when the actuator chain downstream
  // of cmd_vel is positive-right, which is what PID_lanekeeping's
  // invert_steering parameter compensates for on this vehicle.
  float steering_scale{1.0F};
  float steering_limit_rad{0.5F};
  // Publish xxcar_msgs/DirectControl on four_wheel_topic instead of a Twist on
  // topic. Twist cannot carry a rear steering angle, so four-wheel steering
  // needs this transport. It requires steering_scale 1.0 and ekf_mcu_driver
  // running with steering calibration, which maps each axle from radians.
  bool four_wheel{false};
  std::string four_wheel_topic{"direct_control"};
  // If every MPPI sample latches an obstacle in the configured near-term
  // window, retain steering and command torque opposite measured motor rotation
  // until vehicle speed reaches the release threshold.
  bool obstacle_brake_enabled{false};
  float obstacle_brake_activation_s{0.5F};
  float obstacle_brake_recovery_s{0.5F};
  float obstacle_brake_stop_speed_mps{0.05F};
  float obstacle_brake_torque_nm{1.6F};
  float obstacle_brake_motor_rpm_release{50.0F};
  float obstacle_brake_motor_rpm_engage{80.0F};
};

void ValidateDirectControlConfig(const DirectControlConfig & config);

geometry_msgs::msg::Twist ToDirectControlMessage(
  const PlannedTrajectory & trajectory, const DirectControlConfig & config);

// Four-wheel-steering transport. Carries the same converted front steering and
// throttle as the Twist form, plus the rear steering angle Twist cannot express.
xxcar_msgs::msg::DirectControl ToDirectControlMessage(
  const PlannedTrajectory & trajectory, const DirectControlConfig & config,
  const rclcpp::Time & stamp);

// Retain the selected MPPI steering and replace only wheel torque with the
// selected safety torque. The message remains in torque mode.
xxcar_msgs::msg::DirectControl ToSafetyTorqueMessage(
  const PlannedTrajectory & trajectory, const DirectControlConfig & config,
  float safety_torque_nm,
  const rclcpp::Time & stamp);

}  // namespace xxcar::mppi
