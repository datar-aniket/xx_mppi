#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <xxcar_msgs/msg/vehicle_control_trajectory.hpp>

#include "xx_mppi/controller/mppi_controller.hpp"

namespace xxcar::mppi {

// Reusable ROS runtime beneath the final custom vehicle-state subscriber. The
// adapter calls OnObservation from every message callback; this class projects
// each update and publishes at the configured solve rate.
class MppiRosRuntime {
 public:
  MppiRosRuntime(
    rclcpp::Node & node, const std::string & config_directory,
    const std::string & trajectory_topic = "vehicle_control_trajectory");

  [[nodiscard]] Projection OnObservation(const VehicleObservation & observation);
  void Reset();

 private:
  void TimerCallback();

  rclcpp::Node & node_;
  std::unique_ptr<MppiController> controller_;
  rclcpp::Publisher<xxcar_msgs::msg::VehicleControlTrajectory>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::mutex mutex_;
  std::uint64_t observation_generation_{};
  std::uint64_t solved_generation_{};
};

}  // namespace xxcar::mppi
