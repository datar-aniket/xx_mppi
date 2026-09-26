#pragma once

#include <deque>
#include <utility>

namespace xxcar::mppi {

struct GripMonitorConfig {
  // Low-pass time constant applied to both acceleration magnitudes before they
  // are compared, so EKF noise and the model's step response do not trip it.
  float time_constant_s{0.15F};
  // The check only runs where grip is the question: above this speed and with
  // the model itself using at least this much of an axle's limit.
  float minimum_speed_mps{1.0F};
  float minimum_model_utilization{0.7F};
  float shortfall_threshold_mps2{1.0F};
  // The flag sets after the condition has held this long and clears after it
  // has been false this long.
  float latch_time_s{0.3F};
  float peak_window_s{5.0F};
};

// Compares the acceleration the MPPI model predicts for the command being
// applied with the acceleration the car actually shows. Host only, not
// thread-safe: call Update from one thread.
class GripMonitor {
 public:
  struct Output {
    float acceleration_shortfall_mps2{};
    float observed_peak_mu{};
    bool model_overestimates_grip{false};
  };

  explicit GripMonitor(GripMonitorConfig config = {});

  // Predicted values may be NaN (model without tires); the shortfall is then
  // NaN and the flag stays clear.
  Output Update(
    double time_s, float speed_mps, float measured_ax_mps2, float measured_ay_mps2,
    float predicted_ax_mps2, float predicted_ay_mps2, float predicted_peak_utilization);
  void Reset();

 private:
  GripMonitorConfig config_;
  bool initialized_{false};
  double last_time_s_{};
  float measured_filtered_{};
  float predicted_filtered_{};
  bool latched_{false};
  float condition_time_s_{};
  std::deque<std::pair<double, float>> peak_window_;
};

}  // namespace xxcar::mppi
