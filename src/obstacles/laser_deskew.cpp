#include "xx_mppi/obstacles/laser_deskew.hpp"

#include <cmath>

namespace xxcar::mppi {

std::optional<DeskewedScan> DeskewLaserScan(
  const LaserScanData & scan, const RigidTransform2D & laser_to_base,
  const PoseHistory & pose_history)
{
  if (scan.first_ray_stamp_ns <= 0 || scan.ranges_m.empty() ||
    !std::isfinite(scan.angle_min_rad) || !std::isfinite(scan.angle_increment_rad) ||
    !(scan.range_max_m > scan.range_min_m))
  {
    return std::nullopt;
  }
  float time_increment = scan.time_increment_s;
  if (!(time_increment >= 0.0F) || !std::isfinite(time_increment)) {
    return std::nullopt;
  }
  if (time_increment == 0.0F && scan.ranges_m.size() > 1U && scan.scan_time_s > 0.0F) {
    time_increment = scan.scan_time_s /
      static_cast<float>(scan.ranges_m.size() - 1U);
  }
  const auto last_offset_ns = static_cast<std::int64_t>(std::llround(
      static_cast<double>(time_increment) *
      static_cast<double>(scan.ranges_m.size() - 1U) * 1.0e9));
  DeskewedScan result;
  result.reference_stamp_ns = scan.first_ray_stamp_ns + last_offset_ns;
  const auto reference_pose = pose_history.PoseAt(result.reference_stamp_ns);
  if (!reference_pose) {
    return std::nullopt;
  }
  result.reference_pose = *reference_pose;
  result.obstacle_points.reserve(scan.ranges_m.size());
  const float transform_cos = std::cos(laser_to_base.yaw_rad);
  const float transform_sin = std::sin(laser_to_base.yaw_rad);
  for (std::size_t index = 0; index < scan.ranges_m.size(); ++index) {
    const float range = scan.ranges_m[index];
    if (!std::isfinite(range) || range < scan.range_min_m || range > scan.range_max_m) {
      continue;
    }
    const auto ray_offset_ns = static_cast<std::int64_t>(std::llround(
        static_cast<double>(time_increment) * static_cast<double>(index) * 1.0e9));
    const auto pose = pose_history.PoseAt(scan.first_ray_stamp_ns + ray_offset_ns);
    if (!pose) {
      return std::nullopt;
    }
    const float angle = scan.angle_min_rad +
      static_cast<float>(index) * scan.angle_increment_rad;
    const float laser_x = range * std::cos(angle);
    const float laser_y = range * std::sin(angle);
    const float base_x = laser_to_base.x_m +
      transform_cos * laser_x - transform_sin * laser_y;
    const float base_y = laser_to_base.y_m +
      transform_sin * laser_x + transform_cos * laser_y;
    const float pose_cos = std::cos(pose->yaw_enu_rad);
    const float pose_sin = std::sin(pose->yaw_enu_rad);
    result.obstacle_points.push_back(Point2D{
      pose->east_m + pose_cos * base_x - pose_sin * base_y,
      pose->north_m + pose_sin * base_x + pose_cos * base_y});
  }
  return result;
}

}  // namespace xxcar::mppi
