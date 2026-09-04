#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "xx_mppi/obstacles/pose_history.hpp"
#include "xx_mppi/obstacles/signed_distance_field.hpp"

namespace xxcar::mppi {

struct RigidTransform2D {
  float x_m{};
  float y_m{};
  float yaw_rad{};
};

struct LaserScanData {
  std::int64_t first_ray_stamp_ns{};
  float angle_min_rad{};
  float angle_increment_rad{};
  float time_increment_s{};
  float scan_time_s{};
  float range_min_m{};
  float range_max_m{};
  std::vector<float> ranges_m;
};

struct DeskewedScan {
  std::int64_t reference_stamp_ns{};
  Pose2D reference_pose{};
  std::vector<Point2D> obstacle_points;
};

// LaserScan stamps identify the first ray. All valid returns are transformed at
// their own acquisition time. Failure to cover any valid ray rejects the scan.
[[nodiscard]] std::optional<DeskewedScan> DeskewLaserScan(
  const LaserScanData & scan, const RigidTransform2D & laser_to_base,
  const PoseHistory & pose_history);

}  // namespace xxcar::mppi
