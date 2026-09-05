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
    std::int64_t stamp_ns, std::uint64_t generation);

 private:
  ObstacleConfig config_;
  std::uint32_t width_{};
  std::uint32_t height_{};
  // Build runs on the dedicated obstacle worker. Reuse all transform scratch
  // storage so a 40 Hz scan does not repeatedly allocate and free full grids.
  std::vector<std::uint8_t> occupied_;
  std::vector<std::uint8_t> free_space_;
  std::vector<float> transform_first_;
  std::vector<float> distance_to_obstacle_;
  std::vector<float> distance_to_free_;
  std::vector<float> transform_input_;
  std::vector<float> transform_output_;
  std::vector<std::int32_t> transform_locations_;
  std::vector<float> transform_boundaries_;
};

[[nodiscard]] float SampleSignedDistance(
  const ObstacleField & field, float east_m, float north_m,
  float outside_value_m);

}  // namespace xxcar::mppi
