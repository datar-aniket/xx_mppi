#include "xx_mppi/ros/mppi_node.hpp"

#include <cmath>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <tf2/exceptions.h>
#include <tf2/time.h>

#include "xx_mppi/ros/direct_control_message.hpp"
#include "xx_mppi/ros/runtime_config.hpp"

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
  const auto yaml_defaults = LoadRosRuntimeConfig(config_directory);
  const auto state_topic = declare_parameter<std::string>("state_topic", "ekf/state");
  const auto trajectory_topic = declare_parameter<std::string>(
    "trajectory_topic", "vehicle_control_trajectory");
  const auto scan_topic = declare_parameter<std::string>("scan_topic", "/scan");
  base_frame_ = declare_parameter<std::string>("base_frame", "base_link");
  laser_frame_ = declare_parameter<std::string>("laser_frame", "laser");
  DirectControlConfig direct_control;
  direct_control.enabled = declare_parameter<bool>(
    "publish_direct_control", yaml_defaults.direct_control.enabled);
  direct_control.topic = declare_parameter<std::string>(
    "direct_control_topic", yaml_defaults.direct_control.topic);
  direct_control.mode = ParseDirectControlMode(declare_parameter<std::string>(
      "control_mode", DirectControlModeName(yaml_defaults.direct_control.mode)));
  direct_control.torque_to_throttle_scale = static_cast<float>(declare_parameter<double>(
      "direct_control_torque_to_throttle_scale",
      static_cast<double>(yaml_defaults.direct_control.torque_to_throttle_scale)));
  direct_control.throttle_min = static_cast<float>(declare_parameter<double>(
      "direct_control_throttle_min",
      static_cast<double>(yaml_defaults.direct_control.throttle_min)));
  direct_control.throttle_max = static_cast<float>(declare_parameter<double>(
      "direct_control_throttle_max",
      static_cast<double>(yaml_defaults.direct_control.throttle_max)));
  direct_control.steering_scale = static_cast<float>(declare_parameter<double>(
      "direct_control_steering_scale",
      static_cast<double>(yaml_defaults.direct_control.steering_scale)));
  direct_control.steering_limit_rad = static_cast<float>(declare_parameter<double>(
      "direct_control_steering_limit_rad",
      static_cast<double>(yaml_defaults.direct_control.steering_limit_rad)));
  maximum_state_age_s_ = declare_parameter<double>("maximum_state_age_s", 0.10);
  future_tolerance_s_ = declare_parameter<double>("future_tolerance_s", 0.02);
  adapter_config_.require_solution_validity = declare_parameter<bool>(
    "require_solution_validity", yaml_defaults.require_solution_validity);
  adapter_config_.require_absolute_yaw = declare_parameter<bool>(
    "require_absolute_yaw", true);
  adapter_config_.require_vesc = declare_parameter<bool>("require_vesc", true);
  const auto state_qos_depth = declare_parameter<int>("state_qos_depth", 1);
  const auto scan_qos_depth = declare_parameter<int>("scan_qos_depth", 1);

  VisualizationConfig visualization;
  visualization.enabled = declare_parameter<bool>("publish_visualization", false);
  visualization.frame_id = declare_parameter<std::string>("visualization_frame_id", "map");
  visualization.planned_path_topic = declare_parameter<std::string>(
    "expected_path_topic", "xx_mppi/expected_path");
  visualization.marker_topic = declare_parameter<std::string>(
    "rollouts_topic", "xx_mppi/rollouts");
  visualization.raceline_topic = declare_parameter<std::string>(
    "raceline_path_topic", "xx_mppi/raceline");
  visualization.left_boundary_topic = declare_parameter<std::string>(
    "track_left_boundary_topic", "xx_mppi/track_left_boundary");
  visualization.right_boundary_topic = declare_parameter<std::string>(
    "track_right_boundary_topic", "xx_mppi/track_right_boundary");
  visualization.obstacle_costmap_topic = declare_parameter<std::string>(
    "obstacle_costmap_topic", "xx_mppi/obstacle_costmap");

  if (config_directory.empty() || state_topic.empty() ||
    (!direct_control.enabled && trajectory_topic.empty()))
  {
    throw std::invalid_argument("config directory and selected ROS topics must not be empty");
  }
  if (state_qos_depth <= 0 || scan_qos_depth <= 0) {
    throw std::invalid_argument("state_qos_depth and scan_qos_depth must be positive");
  }
  // Validate timing and adapter parameters before opening the runtime.
  ValidateEkfStateAdapterConfig(adapter_config_);
  ValidateObservationTime(1, 1, std::nullopt, maximum_state_age_s_, future_tolerance_s_);
  if (direct_control.enabled) {
    ValidateDirectControlConfig(direct_control);
  }

  runtime_ = std::make_unique<MppiRosRuntime>(
    *this, config_directory, trajectory_topic, direct_control, std::move(visualization));
  const auto qos = rclcpp::QoS(rclcpp::KeepLast(
      static_cast<std::size_t>(state_qos_depth))).best_effort().durability_volatile();
  state_subscription_ = create_subscription<xxcar_msgs::msg::EkfState>(
    state_topic, qos,
    std::bind(&MppiNode::StateCallback, this, std::placeholders::_1));

  const auto obstacle_config = runtime_->config().obstacles;
  if (obstacle_config.enabled) {
    if (scan_topic.empty() || base_frame_.empty() || laser_frame_.empty()) {
      throw std::invalid_argument("obstacle scan topic and frame names must not be empty");
    }
    pose_history_ = std::make_unique<PoseHistory>(
      obstacle_config.pose_history_s, obstacle_config.maximum_extrapolation_s);
    field_builder_ = std::make_unique<SignedDistanceFieldBuilder>(obstacle_config);
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
    const auto scan_qos = rclcpp::SensorDataQoS().keep_last(
      static_cast<std::size_t>(scan_qos_depth));
    scan_subscription_ = create_subscription<sensor_msgs::msg::LaserScan>(
      scan_topic, scan_qos,
      std::bind(&MppiNode::ScanCallback, this, std::placeholders::_1));
    obstacle_worker_ = std::thread(&MppiNode::ObstacleWorker, this);
    RCLCPP_INFO(
      get_logger(), "Obstacle SDF listening on '%s'; static transform %s <- %s",
      scan_topic.c_str(), base_frame_.c_str(), laser_frame_.c_str());
  }

  if (direct_control.enabled) {
    RCLCPP_WARN(
      get_logger(),
      "MPPI listening on '%s' and publishing direct Twist control on '%s' "
      "in %s mode (scale %.6g, duty clamp [%.6g, %.6g], steering scale %.6g) "
      "with config '%s'",
      state_topic.c_str(), direct_control.topic.c_str(), DirectControlModeName(direct_control.mode),
      static_cast<double>(direct_control.torque_to_throttle_scale),
      static_cast<double>(direct_control.throttle_min),
      static_cast<double>(direct_control.throttle_max),
      static_cast<double>(direct_control.steering_scale), config_directory.c_str());
  } else {
    RCLCPP_INFO(
      get_logger(), "MPPI listening on '%s' and publishing trajectory '%s' with config '%s'",
      state_topic.c_str(), trajectory_topic.c_str(), config_directory.c_str());
  }
}

MppiNode::~MppiNode() {
  {
    std::lock_guard<std::mutex> lock(scan_mutex_);
    stop_obstacle_worker_ = true;
    pending_scan_.reset();
  }
  scan_condition_.notify_one();
  if (obstacle_worker_.joinable()) {
    obstacle_worker_.join();
  }
}

void MppiNode::StateCallback(
  const xxcar_msgs::msg::EkfState::ConstSharedPtr message)
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
      {
        std::lock_guard<std::mutex> lock(pose_history_mutex_);
        if (pose_history_) {
          pose_history_->Clear();
        }
        ++pose_epoch_;
      }
      runtime_->Reset();
      previous_pose_time_ns_.reset();
      RCLCPP_WARN(
        get_logger(), "EKF reset counter changed from %u to %u; MPPI warm start reset",
        static_cast<unsigned>(*previous_reset_counter_),
        static_cast<unsigned>(message->reset_counter));
    }

    const auto projection = runtime_->OnObservation(observation);
    if (pose_history_) {
      const float history_sideslip = std::isfinite(observation.sideslip_rad) ?
        observation.sideslip_rad : 0.0F;
      std::lock_guard<std::mutex> lock(pose_history_mutex_);
      pose_history_->Add(TimedVehiclePose{
        observation.pose_time_ns,
        Pose2D{observation.east_m, observation.north_m, observation.yaw_enu_rad},
        observation.speed_mps, history_sideslip, observation.yaw_rate_radps});
    }
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

void MppiNode::ScanCallback(
  const sensor_msgs::msg::LaserScan::ConstSharedPtr message)
{
  if (message->header.frame_id != laser_frame_) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "Rejecting scan frame '%s'; expected '%s'",
      message->header.frame_id.c_str(), laser_frame_.c_str());
    return;
  }
  {
    std::lock_guard<std::mutex> lock(scan_mutex_);
    pending_scan_ = message;
  }
  scan_condition_.notify_one();
}

std::optional<RigidTransform2D> MppiNode::GetLaserToBaseTransform(
  const std::string & scan_frame)
{
  if (laser_to_base_) {
    return laser_to_base_;
  }
  try {
    const auto transform = tf_buffer_->lookupTransform(
      base_frame_, scan_frame, tf2::TimePointZero);
    const auto & rotation = transform.transform.rotation;
    const double norm_squared = rotation.x * rotation.x + rotation.y * rotation.y +
      rotation.z * rotation.z + rotation.w * rotation.w;
    if (!(norm_squared > 1.0e-12) || !std::isfinite(norm_squared)) {
      throw std::runtime_error("laser transform has an invalid quaternion");
    }
    const double inverse_norm = 1.0 / std::sqrt(norm_squared);
    const double x = rotation.x * inverse_norm;
    const double y = rotation.y * inverse_norm;
    const double z = rotation.z * inverse_norm;
    const double w = rotation.w * inverse_norm;
    laser_to_base_ = RigidTransform2D{
      static_cast<float>(transform.transform.translation.x),
      static_cast<float>(transform.transform.translation.y),
      static_cast<float>(std::atan2(
          2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))};
    RCLCPP_INFO(
      get_logger(), "Cached static transform %s <- %s: x=%.3f y=%.3f yaw=%.3f",
      base_frame_.c_str(), scan_frame.c_str(), laser_to_base_->x_m,
      laser_to_base_->y_m, laser_to_base_->yaw_rad);
    return laser_to_base_;
  } catch (const tf2::TransformException & error) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "Waiting for static laser transform: %s", error.what());
  } catch (const std::exception & error) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "Invalid static laser transform: %s", error.what());
  }
  return std::nullopt;
}

void MppiNode::ObstacleWorker() {
  while (true) {
    sensor_msgs::msg::LaserScan::ConstSharedPtr message;
    {
      std::unique_lock<std::mutex> lock(scan_mutex_);
      scan_condition_.wait(lock, [this]() {
        return stop_obstacle_worker_ || static_cast<bool>(pending_scan_);
      });
      if (stop_obstacle_worker_) {
        return;
      }
      message = std::move(pending_scan_);
    }
    const auto transform = GetLaserToBaseTransform(message->header.frame_id);
    if (!transform) {
      continue;
    }
    const auto history_snapshot = [&]() {
      std::lock_guard<std::mutex> lock(pose_history_mutex_);
      return std::make_pair(*pose_history_, pose_epoch_);
    }();
    LaserScanData scan;
    scan.first_ray_stamp_ns = rclcpp::Time(message->header.stamp).nanoseconds();
    scan.angle_min_rad = message->angle_min;
    scan.angle_increment_rad = message->angle_increment;
    scan.time_increment_s = message->time_increment;
    scan.scan_time_s = message->scan_time;
    scan.range_min_m = message->range_min;
    scan.range_max_m = message->range_max;
    scan.ranges_m = message->ranges;
    const auto deskewed = DeskewLaserScan(scan, *transform, history_snapshot.first);
    if (!deskewed) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Dropping scan: pose history does not cover its ray timestamps");
      continue;
    }
    auto field = std::make_shared<ObstacleField>(field_builder_->Build(
        deskewed->obstacle_points, deskewed->reference_pose,
        deskewed->reference_stamp_ns, ++obstacle_generation_));
    {
      std::lock_guard<std::mutex> lock(pose_history_mutex_);
      if (history_snapshot.second != pose_epoch_) {
        continue;
      }
    }
    runtime_->SetObstacleField(std::move(field));
  }
}

}  // namespace xxcar::mppi
