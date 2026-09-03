#include "xx_mppi/ros/visualization.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>

#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_msgs/msg/header.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include "xx_mppi/math.hpp"

namespace xxcar::mppi {
namespace {

using Marker = visualization_msgs::msg::Marker;

geometry_msgs::msg::PoseStamped MakePose(
  const std_msgs::msg::Header & header, const float east_m, const float north_m,
  const float yaw_enu_rad)
{
  geometry_msgs::msg::PoseStamped pose;
  pose.header = header;
  pose.pose.position.x = east_m;
  pose.pose.position.y = north_m;
  pose.pose.orientation.z = std::sin(0.5F * yaw_enu_rad);
  pose.pose.orientation.w = std::cos(0.5F * yaw_enu_rad);
  return pose;
}

geometry_msgs::msg::Point MakePoint(
  const float east_m, const float north_m, const double height_m = 0.0)
{
  geometry_msgs::msg::Point point;
  point.x = east_m;
  point.y = north_m;
  point.z = height_m;
  return point;
}

void ConfigureMarker(
  Marker & marker, const std_msgs::msg::Header & header, const char * marker_namespace,
  const std::int32_t id, const std::int32_t type)
{
  marker.header = header;
  marker.ns = marker_namespace;
  marker.id = id;
  marker.type = type;
  marker.action = Marker::ADD;
  marker.pose.orientation.w = 1.0;
}

float PlannedYaw(const PlannedTrajectory & trajectory, const std::size_t index) {
  if (trajectory.states.size() < 2U) {
    return 0.0F;
  }
  const std::size_t a = index + 1U < trajectory.states.size() ? index : index - 1U;
  const std::size_t b = a + 1U;
  return std::atan2(
    trajectory.states[b].north_m - trajectory.states[a].north_m,
    trajectory.states[b].east_m - trajectory.states[a].east_m);
}

}  // namespace

nav_msgs::msg::Path ToPlannedPath(
  const PlannedTrajectory & trajectory, const rclcpp::Time & publication_time,
  const std::string & frame_id)
{
  if (frame_id.empty()) {
    throw std::invalid_argument("visualization frame_id must not be empty");
  }
  nav_msgs::msg::Path path;
  path.header.stamp = publication_time;
  path.header.frame_id = frame_id;
  path.poses.reserve(trajectory.states.size());
  for (std::size_t i = 0; i < trajectory.states.size(); ++i) {
    auto pose = MakePose(
      path.header, trajectory.states[i].east_m, trajectory.states[i].north_m,
      PlannedYaw(trajectory, i));
    const auto offset_ns = static_cast<std::int64_t>(std::llround(
        static_cast<double>(i) * static_cast<double>(trajectory.dt_s) * 1.0e9));
    pose.header.stamp = rclcpp::Time(trajectory.solution_pose_time_ns + offset_ns);
    path.poses.push_back(std::move(pose));
  }
  return path;
}

visualization_msgs::msg::MarkerArray ToTrajectoryMarkers(
  const PlannedTrajectory & trajectory, const Raceline & raceline,
  const rclcpp::Time & publication_time, const std::string & frame_id)
{
  if (frame_id.empty()) {
    throw std::invalid_argument("visualization frame_id must not be empty");
  }
  visualization_msgs::msg::MarkerArray array;
  if (trajectory.sampled_rollouts.empty()) {
    return array;
  }

  std_msgs::msg::Header header;
  header.stamp = publication_time;
  header.frame_id = frame_id;

  array.markers.reserve(trajectory.sampled_rollouts.size());
  const float denominator = static_cast<float>(
    std::max<std::size_t>(trajectory.sampled_rollouts.size() - 1U, 1U));
  for (std::size_t rank = 0; rank < trajectory.sampled_rollouts.size(); ++rank) {
    const auto & rollout = trajectory.sampled_rollouts[rank];
    Marker line;
    ConfigureMarker(
      line, header, "rollouts", static_cast<std::int32_t>(rank),
      Marker::LINE_STRIP);
    line.scale.x = 0.025;
    const float fraction = static_cast<float>(rank) / denominator;
    line.color.r = 0.1F + 0.8F * fraction;
    line.color.g = 0.9F - 0.55F * fraction;
    line.color.b = 1.0F - 0.65F * fraction;
    line.color.a = 0.75F - 0.35F * fraction;
    line.points.reserve(rollout.states.size());
    for (const auto & state : rollout.states) {
      const auto point = StateToEnu(raceline, state, trajectory.frame);
      line.points.push_back(MakePoint(point.first, point.second, 0.02));
    }
    array.markers.push_back(std::move(line));
  }
  return array;
}

StaticVisualizationPaths ToStaticVisualizationPaths(
  const Raceline & raceline, const rclcpp::Time & publication_time,
  const std::string & frame_id)
{
  if (frame_id.empty()) {
    throw std::invalid_argument("visualization frame_id must not be empty");
  }
  StaticVisualizationPaths result;
  result.raceline.header.stamp = publication_time;
  result.raceline.header.frame_id = frame_id;
  result.left_boundary.header = result.raceline.header;
  result.right_boundary.header = result.raceline.header;
  const auto count = raceline.points().size();
  result.raceline.poses.reserve(count);
  result.left_boundary.poses.reserve(count);
  result.right_boundary.poses.reserve(count);

  for (const auto & point : raceline.points()) {
    const float yaw_enu = heading_from_north_to_enu_yaw(point.heading_from_north_rad);
    const auto left = raceline.ToCartesian(point.s_m, point.e_max_m);
    const auto right = raceline.ToCartesian(point.s_m, point.e_min_m);
    result.raceline.poses.push_back(MakePose(
        result.raceline.header, point.east_m, point.north_m, yaw_enu));
    result.left_boundary.poses.push_back(MakePose(
        result.left_boundary.header, left.first, left.second, yaw_enu));
    result.right_boundary.poses.push_back(MakePose(
        result.right_boundary.header, right.first, right.second, yaw_enu));
  }
  return result;
}

}  // namespace xxcar::mppi
