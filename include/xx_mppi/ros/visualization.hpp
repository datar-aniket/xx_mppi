#pragma once

#include <string>

#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <rclcpp/time.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "xx_mppi/controller/mppi_controller.hpp"
#include "xx_mppi/reference/raceline.hpp"

namespace xxcar::mppi {

struct VisualizationConfig {
  bool enabled{false};
  std::string frame_id{"map"};
  std::string planned_path_topic{"xx_mppi/expected_path"};
  std::string marker_topic{"xx_mppi/rollouts"};
  std::string raceline_topic{"xx_mppi/raceline"};
  std::string left_boundary_topic{"xx_mppi/track_left_boundary"};
  std::string right_boundary_topic{"xx_mppi/track_right_boundary"};
  std::string obstacle_costmap_topic{"xx_mppi/obstacle_costmap"};
};

struct StaticVisualizationPaths {
  nav_msgs::msg::Path raceline;
  nav_msgs::msg::Path left_boundary;
  nav_msgs::msg::Path right_boundary;
};

[[nodiscard]] nav_msgs::msg::Path ToPlannedPath(
  const PlannedTrajectory & trajectory, const rclcpp::Time & publication_time,
  const std::string & frame_id);

[[nodiscard]] visualization_msgs::msg::MarkerArray ToTrajectoryMarkers(
  const PlannedTrajectory & trajectory, const Raceline & raceline,
  const rclcpp::Time & publication_time, const std::string & frame_id);

[[nodiscard]] StaticVisualizationPaths ToStaticVisualizationPaths(
  const Raceline & raceline, const rclcpp::Time & publication_time,
  const std::string & frame_id);

[[nodiscard]] nav_msgs::msg::OccupancyGrid ToObstacleCostmap(
  const ObstacleField & field, const ObstacleConfig & config,
  const rclcpp::Time & publication_time, const std::string & frame_id);

}  // namespace xxcar::mppi
