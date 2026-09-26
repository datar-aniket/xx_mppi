#include "xx_mppi/controller/grip_monitor.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace xxcar::mppi {

GripMonitor::GripMonitor(GripMonitorConfig config)
: config_(config) {}

void GripMonitor::Reset() {
  initialized_ = false;
  latched_ = false;
  condition_time_s_ = 0.0F;
  peak_window_.clear();
}

GripMonitor::Output GripMonitor::Update(
  const double time_s, const float speed_mps, const float measured_ax_mps2,
  const float measured_ay_mps2, const float predicted_ax_mps2, const float predicted_ay_mps2,
  const float predicted_peak_utilization)
{
  constexpr float gravity = 9.81F;
  const float measured = std::hypot(measured_ax_mps2, measured_ay_mps2);
  const float predicted = std::hypot(predicted_ax_mps2, predicted_ay_mps2);
  float dt_s = 0.0F;
  if (initialized_) {
    dt_s = static_cast<float>(time_s - last_time_s_);
    // A time jump (bag restart, long pause) starts the filters over.
    if (!(dt_s >= 0.0F) || dt_s > 1.0F) {
      Reset();
      dt_s = 0.0F;
    }
  }
  last_time_s_ = time_s;

  const float alpha = config_.time_constant_s > 0.0F ?
    1.0F - std::exp(-dt_s / config_.time_constant_s) : 1.0F;
  if (std::isfinite(measured)) {
    measured_filtered_ = initialized_ ?
      measured_filtered_ + alpha * (measured - measured_filtered_) : measured;
  }
  if (std::isfinite(predicted)) {
    predicted_filtered_ = initialized_ && std::isfinite(predicted_filtered_) ?
      predicted_filtered_ + alpha * (predicted - predicted_filtered_) : predicted;
  } else {
    predicted_filtered_ = std::numeric_limits<float>::quiet_NaN();
  }
  initialized_ = true;

  Output output;
  output.acceleration_shortfall_mps2 = predicted_filtered_ - measured_filtered_;
  const bool condition = std::isfinite(output.acceleration_shortfall_mps2) &&
    speed_mps >= config_.minimum_speed_mps &&
    predicted_peak_utilization >= config_.minimum_model_utilization &&
    output.acceleration_shortfall_mps2 >= config_.shortfall_threshold_mps2;
  // condition_time_s_ counts how long the input has disagreed with the latch.
  condition_time_s_ = condition != latched_ ? condition_time_s_ + dt_s : 0.0F;
  if (condition_time_s_ >= config_.latch_time_s) {
    latched_ = condition;
    condition_time_s_ = 0.0F;
  }
  output.model_overestimates_grip = latched_;

  peak_window_.emplace_back(time_s, measured_filtered_ / gravity);
  while (!peak_window_.empty() && time_s - peak_window_.front().first > config_.peak_window_s) {
    peak_window_.pop_front();
  }
  for (const auto & entry : peak_window_) {
    output.observed_peak_mu = std::max(output.observed_peak_mu, entry.second);
  }
  return output;
}

}  // namespace xxcar::mppi
