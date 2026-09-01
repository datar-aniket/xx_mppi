#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include <ekf_mcu_driver/msg/ekf_state.hpp>
#include <rclcpp/rclcpp.hpp>

#include "xx_mppi/ros/ekf_state_adapter.hpp"
#include "xx_mppi/ros/runtime.hpp"

namespace xxcar::mppi {

class MppiNode : public rclcpp::Node {
 public:
  explicit MppiNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions{});

 private:
  void StateCallback(const ekf_mcu_driver::msg::EkfState::ConstSharedPtr message);

  EkfStateAdapterConfig adapter_config_;
  double maximum_state_age_s_{};
  double future_tolerance_s_{};
  std::unique_ptr<MppiRosRuntime> runtime_;
  rclcpp::Subscription<ekf_mcu_driver::msg::EkfState>::SharedPtr state_subscription_;
  std::optional<std::int64_t> previous_pose_time_ns_;
  std::optional<std::uint8_t> previous_reset_counter_;
};

}  // namespace xxcar::mppi
