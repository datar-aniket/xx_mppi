#pragma once

#include <string>

#include <geometry_msgs/msg/twist.hpp>

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
};

void ValidateDirectControlConfig(const DirectControlConfig & config);

geometry_msgs::msg::Twist ToDirectControlMessage(
  const PlannedTrajectory & trajectory, const DirectControlConfig & config);

}  // namespace xxcar::mppi
