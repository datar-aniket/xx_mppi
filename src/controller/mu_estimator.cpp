#include "xx_mppi/controller/mu_estimator.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "xx_mppi/dynamics/grip.hpp"

namespace xxcar::mppi {
namespace {

constexpr float kGravity = 9.81F;

}  // namespace

MuEstimator::MuEstimator(VehicleParameters nominal, MuEstimatorConfig config)
: nominal_(nominal), config_(config)
{
  if (!(config_.mu_min > 0.0F) || !(config_.mu_max > config_.mu_min) ||
    !(config_.mu_step > 0.0F) || !(config_.time_constant_s > 0.0F) ||
    !(config_.longitudinal_weight >= 0.0F))
  {
    throw std::invalid_argument(
      "mu estimator needs 0 < mu_min < mu_max, a positive step and time constant, "
      "and a nonnegative longitudinal weight");
  }
  nominal_mu_ = FrictionLimitMps2(nominal_) / kGravity;
  if (!(nominal_mu_ > 0.0F)) {
    throw std::invalid_argument("mu estimator needs positive vehicle.yaml friction coefficients");
  }
  const auto count = static_cast<std::size_t>(
    std::floor((config_.mu_max - config_.mu_min) / config_.mu_step + 1.0e-3F)) + 1U;
  grid_.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    grid_.push_back(config_.mu_min + static_cast<float>(i) * config_.mu_step);
  }
  if (grid_.size() < 3U) {
    throw std::invalid_argument("mu estimator grid needs at least three points");
  }
}

VehicleParameters MuEstimator::WithMu(const float mu) const noexcept {
  const float scale = mu / nominal_mu_;
  auto parameters = nominal_;
  parameters.front_friction_coefficient *= scale;
  parameters.rear_friction_coefficient *= scale;
  return parameters;
}

float MuEstimator::Estimate() const noexcept {
  return weight_ > 0.0 ? static_cast<float>(weighted_sum_ / weight_) : nominal_mu_;
}

MuEstimator::Output MuEstimator::Update(
  const double time_s, const BodyState & state, const Control & control,
  const float measured_ax_mps2, const float measured_ay_mps2)
{
  constexpr float nan = std::numeric_limits<float>::quiet_NaN();
  // Forget by elapsed time. A backwards or long jump (bag restart, pause)
  // forgets nothing extra: the estimate should survive the car standing still.
  double dt_s = 0.0;
  if (initialized_) {
    dt_s = std::clamp(time_s - last_time_s_, 0.0, 1.0);
  }
  initialized_ = true;
  last_time_s_ = time_s;
  const double retain = std::exp(-dt_s / static_cast<double>(config_.time_constant_s));
  weighted_sum_ *= retain;
  weight_ *= retain;

  Output output;
  output.nominal_mu = nominal_mu_;
  output.sample_mu = nan;
  output.sample_sensitivity = nan;
  output.sample_residual_mps2 = nan;
  output.predicted_ay_mps2 = nan;

  bool finite = std::isfinite(measured_ax_mps2) && std::isfinite(measured_ay_mps2);
  for (const float value : state) {
    finite = finite && std::isfinite(value);
  }
  for (const float value : control) {
    finite = finite && std::isfinite(value);
  }
  if (finite && std::abs(state[kSpeed]) >= config_.minimum_speed_mps) {
    const std::size_t count = grid_.size();
    std::vector<float> ax(count);
    std::vector<float> ay(count);
    std::vector<float> residual(count);
    std::size_t best = 0U;
    for (std::size_t i = 0; i < count; ++i) {
      // The front-steer-only model is the 4WS model at zero rear steer, so one
      // evaluation serves both cars.
      const auto sample = PredictGrip(
        WithMu(grid_[i]), ModelKind::kDynamicBicycleFiala4ws, state, control);
      ax[i] = sample.ax_mps2;
      ay[i] = sample.ay_mps2;
      const float ey = measured_ay_mps2 - ay[i];
      const float ex = measured_ax_mps2 - ax[i];
      residual[i] = ey * ey + config_.longitudinal_weight * ex * ex;
      if (!(residual[i] >= residual[best])) {
        best = i;
      }
    }

    // Parabolic refinement between grid points, and the local slope of the
    // prediction there. Both use a one-sided stencil at the grid edges.
    float fitted = grid_[best];
    if (best > 0U && best + 1U < count) {
      const float curvature = residual[best - 1U] - 2.0F * residual[best] + residual[best + 1U];
      if (curvature > 0.0F) {
        const float offset = std::clamp(
          0.5F * (residual[best - 1U] - residual[best + 1U]) / curvature, -0.5F, 0.5F);
        fitted += offset * config_.mu_step;
      }
    }
    const std::size_t lower = best > 0U ? best - 1U : best;
    const std::size_t upper = best + 1U < count ? best + 1U : best;
    const float span = grid_[upper] - grid_[lower];
    const float slope_y = (ay[upper] - ay[lower]) / span;
    const float slope_x = (ax[upper] - ax[lower]) / span;
    const float sensitivity = std::sqrt(
      slope_y * slope_y + config_.longitudinal_weight * slope_x * slope_x);
    const float rms = std::sqrt(residual[best] / (1.0F + config_.longitudinal_weight));

    output.sample_sensitivity = sensitivity;
    output.sample_residual_mps2 = rms;
    output.predicted_ay_mps2 = ay[best];
    // A best fit on the grid edge is censored: the true mu is there or beyond,
    // or no mu explains the sample at all. Either way it is not a measurement.
    const bool interior = best > 0U && best + 1U < count;
    if (interior && std::isfinite(fitted) && std::isfinite(sensitivity) &&
      sensitivity >= config_.minimum_sensitivity && rms <= config_.maximum_residual_mps2)
    {
      // Weight by information: a fully sliding axle (slope ~ g) counts fully,
      // a barely saturated one proportionally less.
      const double weight = std::pow(static_cast<double>(sensitivity / kGravity), 2.0);
      weighted_sum_ += weight * static_cast<double>(fitted);
      weight_ += weight;
      output.sample_accepted = true;
      output.sample_mu = fitted;
    }
  }

  output.mu = Estimate();
  output.confidence = static_cast<float>(weight_);
  output.has_estimate = weight_ > 0.0;
  if (!output.sample_accepted && !std::isfinite(output.predicted_ay_mps2) && finite) {
    output.predicted_ay_mps2 = PredictGrip(
      WithMu(output.mu), ModelKind::kDynamicBicycleFiala4ws, state, control).ay_mps2;
  }
  return output;
}

}  // namespace xxcar::mppi
