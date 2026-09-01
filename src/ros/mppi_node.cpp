#include "xx_mppi/ros/mppi_node.hpp"

#include <exception>
#include <functional>
#include <stdexcept>
#include <string>

#include <ament_index_cpp/get_package_share_directory.hpp>

namespace xxcar::mppi {
namespace {

std::string DefaultConfigDirectory() {
  return ament_index_cpp::get_package_share_directory("xx_mppi") + "/config";
}

}  // namespace

MppiNode::MppiNode(const rclcpp::NodeOptions & options)
: Node("xx_mppi_node", options)
{
  const auto config_directory = declare_parameter<std::string>(
    "config_directory", DefaultConfigDirectory());
  const auto state_topic = declare_parameter<std::string>("state_topic", "ekf/state");
  const auto trajectory_topic = declare_parameter<std::string>(
    "trajectory_topic", "vehicle_control_trajectory");
  maximum_state_age_s_ = declare_parameter<double>("maximum_state_age_s", 0.10);
  future_tolerance_s_ = declare_parameter<double>("future_tolerance_s", 0.02);
  adapter_config_.require_absolute_yaw = declare_parameter<bool>(
    "require_absolute_yaw", true);
  adapter_config_.require_vesc = declare_parameter<bool>("require_vesc", true);
  adapter_config_.steering_scale_to_rad = static_cast<float>(declare_parameter<double>(
      "steering_scale_to_rad", 1.0));
  adapter_config_.steering_offset_rad = static_cast<float>(declare_parameter<double>(
      "steering_offset_rad", 0.0));
  adapter_config_.torque_scale_to_nm = static_cast<float>(declare_parameter<double>(
      "torque_scale_to_nm", 1.0));
  adapter_config_.motor_speed_scale_to_mps = static_cast<float>(declare_parameter<double>(
      "motor_speed_scale_to_mps", 1.0));
  const auto state_qos_depth = declare_parameter<int>("state_qos_depth", 1);

  if (config_directory.empty() || state_topic.empty() || trajectory_topic.empty()) {
    throw std::invalid_argument("config directory and ROS topics must not be empty");
  }
  if (state_qos_depth <= 0) {
    throw std::invalid_argument("state_qos_depth must be positive");
  }
  // Validate timing and scale parameters before opening the runtime.
  ValidateEkfStateAdapterConfig(adapter_config_);
  ValidateObservationTime(1, 1, std::nullopt, maximum_state_age_s_, future_tolerance_s_);

  runtime_ = std::make_unique<MppiRosRuntime>(*this, config_directory, trajectory_topic);
  const auto qos = rclcpp::QoS(rclcpp::KeepLast(
      static_cast<std::size_t>(state_qos_depth))).reliable().durability_volatile();
  state_subscription_ = create_subscription<ekf_mcu_driver::msg::EkfState>(
    state_topic, qos,
    std::bind(&MppiNode::StateCallback, this, std::placeholders::_1));

  RCLCPP_INFO(
    get_logger(), "MPPI listening on '%s' and publishing '%s' with config '%s'",
    state_topic.c_str(), trajectory_topic.c_str(), config_directory.c_str());
}

void MppiNode::StateCallback(
  const ekf_mcu_driver::msg::EkfState::ConstSharedPtr message)
{
  try {
    const auto observation = ToVehicleObservation(*message, adapter_config_);
    const bool reset_changed = previous_reset_counter_ &&
      *previous_reset_counter_ != message->reset_counter;
    ValidateObservationTime(
      observation.pose_time_ns, get_clock()->now().nanoseconds(),
      reset_changed ? std::nullopt : previous_pose_time_ns_,
      maximum_state_age_s_, future_tolerance_s_);

    if (reset_changed) {
      runtime_->Reset();
      previous_pose_time_ns_.reset();
      RCLCPP_WARN(
        get_logger(), "EKF reset counter changed from %u to %u; MPPI warm start reset",
        static_cast<unsigned>(*previous_reset_counter_),
        static_cast<unsigned>(message->reset_counter));
    }

    const auto projection = runtime_->OnObservation(observation);
    previous_pose_time_ns_ = observation.pose_time_ns;
    previous_reset_counter_ = message->reset_counter;
    RCLCPP_DEBUG(
      get_logger(), "state accepted: s=%.3f e=%.3f dphi=%.3f",
      projection.s_m, projection.e_m, projection.relative_course_rad);
  } catch (const std::exception & error) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 1000, "Rejecting EKF state: %s", error.what());
  }
}

}  // namespace xxcar::mppi
