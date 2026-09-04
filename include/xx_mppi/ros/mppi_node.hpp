#pragma once

#include <cstdint>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

#include <xxcar_msgs/msg/ekf_state.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include "xx_mppi/obstacles/laser_deskew.hpp"
#include "xx_mppi/obstacles/signed_distance_field.hpp"
#include "xx_mppi/ros/ekf_state_adapter.hpp"
#include "xx_mppi/ros/runtime.hpp"

namespace xxcar::mppi {

class MppiNode : public rclcpp::Node {
 public:
  explicit MppiNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions{});
  ~MppiNode() override;

 private:
  void StateCallback(const xxcar_msgs::msg::EkfState::ConstSharedPtr message);
  void ScanCallback(const sensor_msgs::msg::LaserScan::ConstSharedPtr message);
  void ObstacleWorker();
  [[nodiscard]] std::optional<RigidTransform2D> GetLaserToBaseTransform(
    const std::string & scan_frame);

  EkfStateAdapterConfig adapter_config_;
  double maximum_state_age_s_{};
  double future_tolerance_s_{};
  std::unique_ptr<MppiRosRuntime> runtime_;
  rclcpp::Subscription<xxcar_msgs::msg::EkfState>::SharedPtr state_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_subscription_;
  std::optional<std::int64_t> previous_pose_time_ns_;
  std::optional<std::uint8_t> previous_reset_counter_;
  std::string base_frame_;
  std::string laser_frame_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
  std::unique_ptr<PoseHistory> pose_history_;
  std::unique_ptr<SignedDistanceFieldBuilder> field_builder_;
  std::mutex pose_history_mutex_;
  std::uint64_t pose_epoch_{};
  std::mutex scan_mutex_;
  std::condition_variable scan_condition_;
  sensor_msgs::msg::LaserScan::ConstSharedPtr pending_scan_;
  bool stop_obstacle_worker_{false};
  std::thread obstacle_worker_;
  std::optional<RigidTransform2D> laser_to_base_;
  std::uint64_t obstacle_generation_{};
};

}  // namespace xxcar::mppi
