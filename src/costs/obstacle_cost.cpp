#include "xx_mppi/costs/obstacle_cost.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "xx_mppi/obstacles/signed_distance_field.hpp"

namespace xxcar::mppi {

float VehicleObstacleClearance(
  const State & state, const Raceline & raceline, const ObstacleField & field,
  const ObstacleConfig & config)
{
  if (!config.enabled || !field.valid()) {
    return config.maximum_distance_m;
  }
  const auto center = raceline.ToCartesian(
    state[kPathEvolution], state[kLateralDeviation]);
  const auto track = raceline.Interpolate(state[kPathEvolution]);
  const float body_heading_from_north = track.heading_from_north_rad +
    state[kRelativeHeading] - state[kSideslip];
  const float yaw = body_heading_from_north + 0.5F * 3.14159265358979323846F;
  const float segment_length = config.footprint_length_m /
    static_cast<float>(config.footprint_circles);
  const float radius = std::sqrt(
    0.25F * segment_length * segment_length +
    0.25F * config.footprint_width_m * config.footprint_width_m);
  float clearance = std::numeric_limits<float>::infinity();
  for (std::uint16_t i = 0; i < config.footprint_circles; ++i) {
    const float offset = -0.5F * config.footprint_length_m +
      (static_cast<float>(i) + 0.5F) * segment_length;
    const float east = center.first + offset * std::cos(yaw);
    const float north = center.second + offset * std::sin(yaw);
    clearance = std::min(clearance, SampleSignedDistance(
        field, east, north, config.maximum_distance_m) - radius);
  }
  return clearance;
}

float EvaluateObstacleCost(
  const float clearance_m, const ObstacleConfig & config, bool & latched,
  const std::size_t horizon_states)
{
  if (!config.enabled || horizon_states == 0U) {
    return 0.0F;
  }
  const float deficit = std::max(config.influence_distance_m - clearance_m, 0.0F);
  float cost = config.distance_weight * deficit * deficit;
  latched = latched || clearance_m < config.latch_threshold_m;
  if (latched) {
    cost += config.latching_weight / static_cast<float>(horizon_states);
  }
  return cost;
}

}  // namespace xxcar::mppi
