#include "xx_mppi/ros/diagnostics_message.hpp"

#include <cstdint>
#include <string>
#include <utility>

#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <diagnostic_msgs/msg/key_value.hpp>

namespace xxcar::mppi {
namespace {

diagnostic_msgs::msg::KeyValue Value(const char * key, const float value) {
  diagnostic_msgs::msg::KeyValue result;
  result.key = key;
  result.value = std::to_string(value);
  return result;
}

diagnostic_msgs::msg::KeyValue Value(const char * key, const std::uint32_t value) {
  diagnostic_msgs::msg::KeyValue result;
  result.key = key;
  result.value = std::to_string(value);
  return result;
}

}  // namespace

diagnostic_msgs::msg::DiagnosticArray ToDiagnosticsMessage(
  const PlannedTrajectory & trajectory, const rclcpp::Time & publication_time)
{
  diagnostic_msgs::msg::DiagnosticArray message;
  message.header.stamp = publication_time;

  diagnostic_msgs::msg::DiagnosticStatus status;
  status.level = trajectory.diagnostics.finite_rollouts > 0U ?
    diagnostic_msgs::msg::DiagnosticStatus::OK :
    diagnostic_msgs::msg::DiagnosticStatus::ERROR;
  status.name = "xx_mppi/solver";
  status.message = trajectory.diagnostics.finite_rollouts > 0U ?
    "latest published solution" : "no finite rollouts";
  status.hardware_id = "jetson";
  status.values.reserve(10U);
  status.values.push_back(Value("solve_time_ms", trajectory.diagnostics.solve_time_ms));
  status.values.push_back(Value("lambda", trajectory.diagnostics.lambda_used));
  status.values.push_back(Value(
    "steering_sigma_rad", trajectory.diagnostics.sigma_used[kSteering]));
  status.values.push_back(Value(
    "wheel_torque_sigma_nm", trajectory.diagnostics.sigma_used[kWheelTorque]));
  if (!trajectory.controls.empty()) {
    status.values.push_back(Value(
      "steering_command_rad", trajectory.controls.front()[kSteering]));
    status.values.push_back(Value(
      "wheel_torque_command_nm", trajectory.controls.front()[kWheelTorque]));
  }
  status.values.push_back(Value("minimum_cost", trajectory.diagnostics.minimum_cost));
  status.values.push_back(Value(
    "effective_sample_size", trajectory.diagnostics.effective_sample_size));
  status.values.push_back(Value(
    "finite_rollouts", trajectory.diagnostics.finite_rollouts));
  const float solution_age_ms = static_cast<float>(
    static_cast<double>(publication_time.nanoseconds() - trajectory.solution_pose_time_ns) *
    1.0e-6);
  status.values.push_back(Value("solution_age_ms", solution_age_ms));
  message.status.push_back(std::move(status));
  return message;
}

}  // namespace xxcar::mppi
