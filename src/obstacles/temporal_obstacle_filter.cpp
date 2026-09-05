#include "xx_mppi/obstacles/temporal_obstacle_filter.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace xxcar::mppi {

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
  const auto to_cell = [this](const Point2D & point) {
      return Cell{
        static_cast<std::int64_t>(std::floor(
          point.east_m / config_.association_distance_m)),
        static_cast<std::int64_t>(std::floor(
          point.north_m / config_.association_distance_m))};
    };
  const auto cell_less = [](const Cell & lhs, const Cell & rhs) {
      return lhs.y < rhs.y || (lhs.y == rhs.y && lhs.x < rhs.x);
    };
  spatial_index_.clear();
  spatial_index_.reserve(tracks_.size());
  for (std::size_t index = 0; index < tracks_.size(); ++index) {
    spatial_index_.push_back(SpatialEntry{to_cell(tracks_[index].point), index});
  }
  std::sort(
    spatial_index_.begin(), spatial_index_.end(),
    [&cell_less](const SpatialEntry & lhs, const SpatialEntry & rhs) {
      if (cell_less(lhs.cell, rhs.cell)) {
        return true;
      }
      if (cell_less(rhs.cell, lhs.cell)) {
        return false;
      }
      return lhs.track_index < rhs.track_index;
    });

  matched_.assign(tracks_.size(), 0U);
  const float maximum_distance_squared =
    config_.association_distance_m * config_.association_distance_m;
  for (const auto & observation : observations) {
    if (!std::isfinite(observation.east_m) || !std::isfinite(observation.north_m)) {
      continue;
    }
    const Cell cell = to_cell(observation);
    std::size_t best = tracks_.size();
    float best_distance_squared = std::numeric_limits<float>::infinity();
    for (std::int64_t dy = -1; dy <= 1; ++dy) {
      for (std::int64_t dx = -1; dx <= 1; ++dx) {
        const Cell candidate_cell{cell.x + dx, cell.y + dy};
        auto candidate = std::lower_bound(
          spatial_index_.begin(), spatial_index_.end(), candidate_cell,
          [&cell_less](const SpatialEntry & entry, const Cell & value) {
            return cell_less(entry.cell, value);
          });
        for (; candidate != spatial_index_.end() &&
          !cell_less(candidate_cell, candidate->cell) &&
          !cell_less(candidate->cell, candidate_cell); ++candidate)
        {
          const auto index = candidate->track_index;
          if (matched_[index]) {
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
      matched_.push_back(1U);
      continue;
    }
    auto & track = tracks_[best];
    matched_[best] = 1U;
    track.point = observation;
    track.missed_updates = 0U;
    if (!track.confirmed) {
      if (track.consecutive_hits < config_.confirmation_updates) {
        ++track.consecutive_hits;
      }
      track.confirmed = track.consecutive_hits >= config_.confirmation_updates;
    }
  }

  retained_tracks_.clear();
  retained_tracks_.reserve(tracks_.size());
  std::vector<Point2D> confirmed;
  confirmed.reserve(tracks_.size());
  for (std::size_t index = 0; index < tracks_.size(); ++index) {
    auto track = tracks_[index];
    if (!matched_[index]) {
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
    retained_tracks_.push_back(std::move(track));
  }
  tracks_.swap(retained_tracks_);
  return confirmed;
}

void TemporalObstacleFilter::Clear() noexcept { tracks_.clear(); }

}  // namespace xxcar::mppi
