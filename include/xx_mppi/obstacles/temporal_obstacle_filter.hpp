#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "xx_mppi/obstacles/signed_distance_field.hpp"
#include "xx_mppi/types.hpp"

namespace xxcar::mppi {

// Confirms scan returns across consecutive updates and retains confirmed
// world-space returns for a bounded number of missed scan updates.
class TemporalObstacleFilter {
 public:
  explicit TemporalObstacleFilter(ObstacleConfig config);

  [[nodiscard]] std::vector<Point2D> Update(
    const std::vector<Point2D> & observations);
  void Clear() noexcept;
  [[nodiscard]] std::size_t track_count() const noexcept { return tracks_.size(); }

 private:
  struct Track {
    Point2D point;
    std::uint32_t consecutive_hits{1U};
    std::uint32_t missed_updates{};
    bool confirmed{false};
  };

  struct Cell {
    std::int64_t x{};
    std::int64_t y{};
  };

  struct SpatialEntry {
    Cell cell;
    std::size_t track_index{};
  };

  ObstacleConfig config_;
  std::vector<Track> tracks_;
  std::vector<Track> retained_tracks_;
  std::vector<SpatialEntry> spatial_index_;
  std::vector<std::uint8_t> matched_;
};

}  // namespace xxcar::mppi
