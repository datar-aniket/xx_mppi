#pragma once

#include <cstdint>
#include <vector>

#include "xx_mppi/obstacles/pose_history.hpp"
#include "xx_mppi/types.hpp"

namespace xxcar::mppi {

struct Point2D {
  float east_m{};
  float north_m{};
};

class SignedDistanceFieldBuilder {
 public:
  explicit SignedDistanceFieldBuilder(ObstacleConfig config);

  [[nodiscard]] ObstacleField Build(
    const std::vector<Point2D> & obstacle_points, const Pose2D & center,
    std::int64_t stamp_ns, std::uint64_t generation) const;

 private:
  ObstacleConfig config_;
  std::uint32_t width_{};
  std::uint32_t height_{};
};

[[nodiscard]] float SampleSignedDistance(
  const ObstacleField & field, float east_m, float north_m,
  float outside_value_m);

}  // namespace xxcar::mppi
