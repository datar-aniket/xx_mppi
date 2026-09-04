#pragma once

#include <cstdint>
#include <deque>
#include <optional>

namespace xxcar::mppi {

struct Pose2D {
  float east_m{};
  float north_m{};
  float yaw_enu_rad{};
};

struct TimedVehiclePose {
  std::int64_t stamp_ns{};
  Pose2D pose{};
  float speed_mps{};
  float sideslip_rad{};
  float yaw_rate_radps{};
};

// Keeps one sample before the retention boundary so timestamps at the edge can
// still be interpolated. Extrapolation is deliberately short and velocity based.
class PoseHistory {
 public:
  PoseHistory(float retention_s, float maximum_extrapolation_s);

  void Add(const TimedVehiclePose & sample);
  void Clear() noexcept;
  [[nodiscard]] std::optional<Pose2D> PoseAt(std::int64_t stamp_ns) const;
  [[nodiscard]] std::size_t size() const noexcept { return samples_.size(); }

 private:
  [[nodiscard]] static Pose2D Extrapolate(
    const TimedVehiclePose & sample, float elapsed_s);

  std::int64_t retention_ns_{};
  std::int64_t maximum_extrapolation_ns_{};
  std::deque<TimedVehiclePose> samples_;
};

}  // namespace xxcar::mppi
