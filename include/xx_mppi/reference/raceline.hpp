#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "xx_mppi/types.hpp"

namespace xxcar::mppi {

class Raceline {
 public:
  static Raceline LoadCsv(const std::string & path, float closed_track_gap_m = 5.0F);

  [[nodiscard]] const std::vector<ReferencePoint> & points() const noexcept { return points_; }
  [[nodiscard]] bool closed() const noexcept { return closed_; }
  [[nodiscard]] float s_min() const noexcept { return points_.front().s_m; }
  [[nodiscard]] float s_max() const noexcept { return points_.back().s_m; }
  [[nodiscard]] float length() const noexcept { return s_max() - s_min(); }

  // course_heading_from_north includes sideslip. A supplied s hint is kept
  // continuous across a closed-track seam and restricts projection to a local
  // arc-length window so nearby branches cannot steal the projection.
  [[nodiscard]] Projection Project(
    float east_m, float north_m, float course_heading_from_north_rad,
    std::optional<float> unwrapped_s_hint = std::nullopt,
    float projection_window_m = 30.0F) const;

  [[nodiscard]] ReferencePoint Interpolate(float unwrapped_s_m) const;
  [[nodiscard]] std::pair<float, float> ToCartesian(
    float unwrapped_s_m, float lateral_deviation_m) const;
  [[nodiscard]] ReferenceHorizon Sample(
    float unwrapped_s0_m, std::uint16_t horizon, float dt_s) const;

 private:
  [[nodiscard]] float WrapOrClamp(float s_m) const;
  [[nodiscard]] float UnwrapNear(float wrapped_s_m, float hint_m) const;

  std::vector<ReferencePoint> points_;
  bool closed_{false};
};

class ContinuousProjector {
 public:
  explicit ContinuousProjector(const Raceline & raceline, float window_m = 30.0F)
  : raceline_(raceline), window_m_(window_m) {}

  [[nodiscard]] Projection Update(
    float east_m, float north_m, float body_heading_from_north_rad,
    float sideslip_rad);

  void Reset() noexcept { s_hint_.reset(); }
  void Seed(float unwrapped_s_m) noexcept { s_hint_ = unwrapped_s_m; }
  [[nodiscard]] std::optional<float> s_hint() const noexcept { return s_hint_; }

 private:
  const Raceline & raceline_;
  float window_m_;
  std::optional<float> s_hint_;
};

}  // namespace xxcar::mppi
