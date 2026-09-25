#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>

#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <xxcar_msgs/msg/vehicle_control_trajectory.hpp>

#include "xx_mppi/controller/mppi_controller.hpp"
#include "xx_mppi/ros/cost_terms_message.hpp"
#include "xx_mppi/ros/direct_control_message.hpp"
#include "xx_mppi/ros/visualization.hpp"

namespace xxcar::mppi {

// Reusable ROS runtime beneath the final custom vehicle-state subscriber. The
// adapter calls OnObservation from every message callback. It only replaces a
// latest-state mailbox; projection and CUDA work belong exclusively to the
// solver worker. Solving and command publication have independent threads and
// rates; only the newest completed, non-stale solution is published.
class MppiRosRuntime {
 public:
  MppiRosRuntime(
    rclcpp::Node & node, const std::string & config_directory,
    const std::string & trajectory_topic = "vehicle_control_trajectory",
    DirectControlConfig direct_control = {},
    VisualizationConfig visualization = {},
    CostTermsConfig cost_terms = {});
  ~MppiRosRuntime();

  void OnObservation(const VehicleObservation & observation);
  void SetObstacleField(
    std::shared_ptr<const ObstacleField> field, std::uint64_t reset_epoch);
  void Reset();
  [[nodiscard]] const ControllerConfig & config() const noexcept {
    return controller_->config();
  }

 private:
  void PublishStaticVisualization();
  void PublishTrajectoryVisualization(
    const PlannedTrajectory & trajectory, const rclcpp::Time & publication_time);
  void PublishObstacleVisualization(
    const ObstacleField & field, const rclcpp::Time & publication_time);
  void PublishInfo(const PlannedTrajectory & trajectory, double publication_age_ms);
  void PublishCostTerms(const PlannedTrajectory & trajectory);
  void QueueVisualization(
    std::shared_ptr<const PlannedTrajectory> trajectory,
    const rclcpp::Time & publication_time);
  void VisualizationWorker();
  void SolverWorker();
  void ControlWorker();
  void InfoWorker();
  void SolveOnce(
    const VehicleObservation & observation, std::uint64_t observation_generation,
    std::uint64_t reset_epoch, const std::optional<Control> & published_control);
  void ControlPublicationCallback();
  void InfoLogCallback();

  rclcpp::Node & node_;
  std::unique_ptr<MppiController> controller_;
  DirectControlConfig direct_control_;
  rclcpp::Publisher<xxcar_msgs::msg::VehicleControlTrajectory>::SharedPtr trajectory_publisher_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr direct_control_publisher_;
  rclcpp::Publisher<xxcar_msgs::msg::DirectControl>::SharedPtr four_wheel_control_publisher_;
  VisualizationConfig visualization_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr planned_path_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_publisher_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr raceline_publisher_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr left_boundary_publisher_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr right_boundary_publisher_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr obstacle_costmap_publisher_;
  CostTermsConfig cost_terms_;
  rclcpp::Publisher<xxcar_msgs::msg::MppiCostTerms>::SharedPtr cost_terms_publisher_;
  std::chrono::nanoseconds solve_period_{};
  std::chrono::nanoseconds control_publication_period_{};
  std::chrono::nanoseconds info_log_period_{};
  bool limit_control_publication_rate_{false};
  std::mutex worker_mutex_;
  std::condition_variable worker_cv_;
  std::condition_variable solver_cv_;
  std::condition_variable control_cv_;
  std::optional<VehicleObservation> latest_observation_;
  std::optional<Control> pending_published_control_;
  // Last command actually sent and when, for the output slew limit. Touched
  // only by the control publication thread. Deliberately kept across Reset:
  // the servo is still wherever that command put it.
  std::optional<Control> last_sent_control_;
  std::int64_t last_sent_time_ns_{};
  std::uint64_t observation_generation_{};
  std::uint64_t solved_generation_{};
  std::uint64_t reset_epoch_{};
  bool reset_requested_{false};
  bool control_work_pending_{false};
  bool stop_workers_{false};
  std::shared_ptr<const ObstacleField> pending_obstacle_field_;
  std::uint64_t applied_obstacle_generation_{};
  std::mutex solution_mutex_;
  std::shared_ptr<const PlannedTrajectory> latest_solution_;
  std::shared_ptr<const PlannedTrajectory> published_solution_;
  std::uint64_t latest_solution_generation_{};
  std::uint64_t published_solution_generation_{};
  double published_solution_age_ms_{};
  std::chrono::nanoseconds visualization_period_{};
  std::chrono::steady_clock::time_point next_visualization_time_{};
  std::chrono::steady_clock::time_point next_costmap_time_{};
  std::chrono::nanoseconds cost_terms_period_{};
  std::chrono::steady_clock::time_point next_cost_terms_time_{};
  std::mutex visualization_mutex_;
  std::condition_variable visualization_cv_;
  std::optional<std::pair<std::shared_ptr<const PlannedTrajectory>, rclcpp::Time>>
  pending_visualization_;
  std::optional<std::pair<std::shared_ptr<const ObstacleField>, rclcpp::Time>>
  pending_obstacle_visualization_;
  // Cost term debug rides the visualization worker rather than a thread of its
  // own: it is gated to its own rate in SolveOnce and only needs to stay off
  // the solver and control threads.
  std::shared_ptr<const PlannedTrajectory> pending_cost_terms_;
  bool stop_visualization_{false};
  std::thread solver_thread_;
  std::thread control_thread_;
  std::thread info_thread_;
  std::thread visualization_thread_;
};

}  // namespace xxcar::mppi
