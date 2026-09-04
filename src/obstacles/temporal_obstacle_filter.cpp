#include "xx_mppi/obstacles/temporal_obstacle_filter.hpp"

#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace xxcar::mppi {
namespace {

struct Cell {
  std::int64_t x{};
  std::int64_t y{};

  bool operator==(const Cell & other) const noexcept {
    return x == other.x && y == other.y;
  }
};

struct CellHash {
  std::size_t operator()(const Cell & cell) const noexcept {
    const auto x = std::hash<std::int64_t>{}(cell.x);
    const auto y = std::hash<std::int64_t>{}(cell.y);
    return x ^ (y + 0x9e3779b9U + (x << 6U) + (x >> 2U));
  }
};

Cell ToCell(const Point2D & point, const float cell_size_m) {
  return Cell{
    static_cast<std::int64_t>(std::floor(point.east_m / cell_size_m)),
    static_cast<std::int64_t>(std::floor(point.north_m / cell_size_m))};
}

}  // namespace

TemporalObstacleFilter::TemporalObstacleFilter(ObstacleConfig config)
: config_(config)
{
  if (config_.confirmation_updates == 0U ||
    !(config_.association_distance_m > 0.0F) ||
    !std::isfinite(config_.association_distance_m))
  {
    throw std::invalid_argument("invalid temporal obstacle filter configuration");
  }
}

std::vector<Point2D> TemporalObstacleFilter::Update(
  const std::vector<Point2D> & observations)
{
  using SpatialHash = std::unordered_map<Cell, std::vector<std::size_t>, CellHash>;
  SpatialHash spatial_hash;
  spatial_hash.reserve(tracks_.size());
  for (std::size_t index = 0; index < tracks_.size(); ++index) {
    spatial_hash[ToCell(tracks_[index].point, config_.association_distance_m)].push_back(index);
  }

  std::vector<bool> matched(tracks_.size(), false);
  const float maximum_distance_squared =
    config_.association_distance_m * config_.association_distance_m;
  for (const auto & observation : observations) {
    if (!std::isfinite(observation.east_m) || !std::isfinite(observation.north_m)) {
      continue;
    }
    const Cell cell = ToCell(observation, config_.association_distance_m);
    std::size_t best = tracks_.size();
    float best_distance_squared = std::numeric_limits<float>::infinity();
    for (std::int64_t dy = -1; dy <= 1; ++dy) {
      for (std::int64_t dx = -1; dx <= 1; ++dx) {
        const auto candidates = spatial_hash.find(Cell{cell.x + dx, cell.y + dy});
        if (candidates == spatial_hash.end()) {
          continue;
        }
        for (const auto index : candidates->second) {
          if (matched[index]) {
            continue;
          }
          const float east_error = tracks_[index].point.east_m - observation.east_m;
          const float north_error = tracks_[index].point.north_m - observation.north_m;
          const float distance_squared =
            east_error * east_error + north_error * north_error;
          if (distance_squared <= maximum_distance_squared &&
            distance_squared < best_distance_squared)
          {
            best = index;
            best_distance_squared = distance_squared;
          }
        }
      }
    }
    if (best == tracks_.size()) {
      Track track{observation};
      track.confirmed = config_.confirmation_updates == 1U;
      tracks_.push_back(std::move(track));
      matched.push_back(true);
      continue;
    }
    auto & track = tracks_[best];
    matched[best] = true;
    track.point = observation;
    track.missed_updates = 0U;
    if (!track.confirmed) {
      if (track.consecutive_hits < config_.confirmation_updates) {
        ++track.consecutive_hits;
      }
      track.confirmed = track.consecutive_hits >= config_.confirmation_updates;
    }
  }

  std::vector<Track> retained;
  retained.reserve(tracks_.size());
  std::vector<Point2D> confirmed;
  confirmed.reserve(tracks_.size());
  for (std::size_t index = 0; index < tracks_.size(); ++index) {
    auto track = tracks_[index];
    if (!matched[index]) {
      if (!track.confirmed) {
        continue;
      }
      if (track.missed_updates < std::numeric_limits<std::uint32_t>::max()) {
        ++track.missed_updates;
      }
      if (track.missed_updates > config_.persistence_updates) {
        continue;
      }
    }
    if (track.confirmed) {
      confirmed.push_back(track.point);
    }
    retained.push_back(std::move(track));
  }
  tracks_ = std::move(retained);
  return confirmed;
}

void TemporalObstacleFilter::Clear() noexcept { tracks_.clear(); }

}  // namespace xxcar::mppi
