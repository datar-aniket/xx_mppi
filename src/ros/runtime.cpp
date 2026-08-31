#include "xx_mppi/ros/runtime.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <utility>

#include "xx_mppi/controller/builder.hpp"
#include "xx_mppi/ros/trajectory_message.hpp"

namespace xxcar::mppi {

MppiRosRuntime::MppiRosRuntime(
  rclcpp::Node & node, const std::string & config_directory,
  const std::string & trajectory_topic)
: node_(node),
  controller_(MppiControllerBuilder::FromConfigDirectory(config_directory)),
  publisher_(node_.create_publisher<xxcar_msgs::msg::VehicleControlTrajectory>(
      trajectory_topic, rclcpp::QoS(1).reliable()))
{
  const auto period_ns = static_cast<std::int64_t>(std::llround(
      1.0e9 / static_cast<double>(controller_->config().solve_rate_hz)));
  timer_ = node_.create_wall_timer(
    std::chrono::nanoseconds(period_ns),
    [this]() {TimerCallback();});
}

Projection MppiRosRuntime::OnObservation(const VehicleObservation & observation) {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto projection = controller_->UpdateObservation(observation);
  ++observation_generation_;
  return projection;
}

void MppiRosRuntime::Reset() {
  std::lock_guard<std::mutex> lock(mutex_);
  controller_->Reset();
  observation_generation_ = 0U;
  solved_generation_ = 0U;
}

void MppiRosRuntime::TimerCallback() {
  PlannedTrajectory trajectory;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (observation_generation_ == solved_generation_) {
      return;
    }
    try {
      trajectory = controller_->PlanLatest();
      solved_generation_ = observation_generation_;
    } catch (const std::exception & error) {
      RCLCPP_ERROR_THROTTLE(
        node_.get_logger(), *node_.get_clock(), 1000,
        "MPPI solve failed: %s", error.what());
      return;
    }
  }
  publisher_->publish(ToRosMessage(trajectory, node_.get_clock()->now()));
}

}  // namespace xxcar::mppi
