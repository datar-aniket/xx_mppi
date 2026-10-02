#pragma once

#include <vector>

#include "xx_mppi/types.hpp"

namespace xxcar::mppi {

struct MuEstimatorConfig {
  // Grid the fit searches, in load-weighted mu. Front and rear keep their
  // vehicle.yaml ratio at every grid point.
  float mu_min{0.2F};
  float mu_max{1.2F};
  float mu_step{0.02F};
  // Below this speed the Fiala slip angles run on the model's speed floor and
  // the fit is meaningless.
  float minimum_speed_mps{1.5F};
  // |d a_y,predicted / d mu| at the fit must reach this [m/s^2 per unit mu]
  // for the sample to count. In the linear tire region it is near zero; with
  // an axle sliding it approaches g.
  float minimum_sensitivity{2.0F};
  // A fit that still misses the measurement by more than this [m/s^2 RMS]
  // means the model state is wrong (bad sideslip, EKF transient), not mu.
  float maximum_residual_mps2{1.5F};
  // Weight of the longitudinal residual relative to the lateral one. Zero by
  // default: the model's a_x rests on the slip ratio, which needs an EKF vx
  // that is least trustworthy exactly when the tires slide.
  float longitudinal_weight{0.0F};
  // Forgetting time constant of the filtered estimate.
  float time_constant_s{2.0F};
};

// Fits the friction coefficient that makes the Fiala body model reproduce the
// measured body acceleration at the measured state and actuation. Host only,
// not thread-safe: call Update from one thread.
class MuEstimator {
 public:
  struct Output {
    float mu{};
    float nominal_mu{};
    float confidence{};
    bool has_estimate{false};
    bool sample_accepted{false};
    float sample_mu{};
    float sample_sensitivity{};
    float sample_residual_mps2{};
    float predicted_ay_mps2{};
  };

  explicit MuEstimator(VehicleParameters nominal, MuEstimatorConfig config = {});

  // state is the measured body state (yaw rate, speed, sideslip, driven wheel
  // speed); control the actuation actually applied, rear steering zero for a
  // front-steer-only car. Accelerations are the IMU body-frame specific force.
  Output Update(
    double time_s, const BodyState & state, const Control & control,
    float measured_ax_mps2, float measured_ay_mps2);

  [[nodiscard]] float nominal_mu() const noexcept { return nominal_mu_; }
  // The nominal parameters with both friction coefficients scaled to a
  // load-weighted mu, keeping their front/rear ratio.
  [[nodiscard]] VehicleParameters WithMu(float mu) const noexcept;

 private:
  [[nodiscard]] float Estimate() const noexcept;

  VehicleParameters nominal_;
  MuEstimatorConfig config_;
  float nominal_mu_{};
  std::vector<float> grid_;
  bool initialized_{false};
  double last_time_s_{};
  double weighted_sum_{};
  double weight_{};
};

}  // namespace xxcar::mppi
