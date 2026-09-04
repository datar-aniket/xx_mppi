#include "xx_mppi/obstacles/pose_history.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "xx_mppi/math.hpp"

namespace xxcar::mppi {
namespace {

std::int64_t SecondsToNanoseconds(const float seconds) {
  if (!(seconds >= 0.0F) || !std::isfinite(seconds)) {
    throw std::invalid_argument("pose-history timing must be finite and nonnegative");
  }
  return static_cast<std::int64_t>(std::llround(static_cast<double>(seconds) * 1.0e9));
}

}  // namespace

PoseHistory::PoseHistory(
  const float retention_s, const float maximum_extrapolation_s)
: retention_ns_(SecondsToNanoseconds(retention_s)),
  maximum_extrapolation_ns_(SecondsToNanoseconds(maximum_extrapolation_s))
{
  if (retention_ns_ <= 0) {
    throw std::invalid_argument("pose-history retention must be positive");
  }
}

void PoseHistory::Add(const TimedVehiclePose & sample) {
  if (sample.stamp_ns <= 0 || !std::isfinite(sample.pose.east_m) ||
    !std::isfinite(sample.pose.north_m) || !std::isfinite(sample.pose.yaw_enu_rad) ||
    !std::isfinite(sample.speed_mps) || !std::isfinite(sample.sideslip_rad) ||
    !std::isfinite(sample.yaw_rate_radps))
  {
    throw std::invalid_argument("pose-history sample is invalid");
  }
  if (!samples_.empty() && sample.stamp_ns <= samples_.back().stamp_ns) {
    throw std::invalid_argument("pose-history timestamps must increase");
  }
  samples_.push_back(sample);
  const std::int64_t cutoff = sample.stamp_ns - retention_ns_;
  while (samples_.size() > 2U && samples_[1].stamp_ns < cutoff) {
    samples_.pop_front();
  }
}

void PoseHistory::Clear() noexcept { samples_.clear(); }

Pose2D PoseHistory::Extrapolate(
  const TimedVehiclePose & sample, const float elapsed_s)
{
  const float body_vx = sample.speed_mps * std::cos(sample.sideslip_rad);
  const float body_vy = sample.speed_mps * std::sin(sample.sideslip_rad);
  const float midpoint_yaw = sample.pose.yaw_enu_rad +
    0.5F * sample.yaw_rate_radps * elapsed_s;
  return Pose2D{
    sample.pose.east_m + elapsed_s *
      (std::cos(midpoint_yaw) * body_vx - std::sin(midpoint_yaw) * body_vy),
    sample.pose.north_m + elapsed_s *
      (std::sin(midpoint_yaw) * body_vx + std::cos(midpoint_yaw) * body_vy),
    wrap_to_pi(sample.pose.yaw_enu_rad + sample.yaw_rate_radps * elapsed_s)};
}

std::optional<Pose2D> PoseHistory::PoseAt(const std::int64_t stamp_ns) const {
  if (samples_.empty()) {
    return std::nullopt;
  }
  if (stamp_ns < samples_.front().stamp_ns) {
    const auto delta = samples_.front().stamp_ns - stamp_ns;
    if (delta > maximum_extrapolation_ns_) {
      return std::nullopt;
    }
    return Extrapolate(samples_.front(), -static_cast<float>(delta) * 1.0e-9F);
  }
  if (stamp_ns > samples_.back().stamp_ns) {
    const auto delta = stamp_ns - samples_.back().stamp_ns;
    if (delta > maximum_extrapolation_ns_) {
      return std::nullopt;
    }
    return Extrapolate(samples_.back(), static_cast<float>(delta) * 1.0e-9F);
  }

  const auto upper = std::lower_bound(
    samples_.begin(), samples_.end(), stamp_ns,
    [](const TimedVehiclePose & sample, const std::int64_t time) {
      return sample.stamp_ns < time;
    });
  if (upper == samples_.begin() || upper->stamp_ns == stamp_ns) {
    return upper->pose;
  }
  const auto lower = std::prev(upper);
  const float fraction = static_cast<float>(stamp_ns - lower->stamp_ns) /
    static_cast<float>(upper->stamp_ns - lower->stamp_ns);
  const float yaw_delta = wrap_to_pi(upper->pose.yaw_enu_rad - lower->pose.yaw_enu_rad);
  return Pose2D{
    lower->pose.east_m + fraction * (upper->pose.east_m - lower->pose.east_m),
    lower->pose.north_m + fraction * (upper->pose.north_m - lower->pose.north_m),
    wrap_to_pi(lower->pose.yaw_enu_rad + fraction * yaw_delta)};
}

}  // namespace xxcar::mppi
