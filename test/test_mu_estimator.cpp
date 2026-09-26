#include <cmath>
#include <limits>

#include <gtest/gtest.h>

#include "xx_mppi/controller/mu_estimator.hpp"
#include "xx_mppi/dynamics/grip.hpp"

namespace xxcar::mppi {
namespace {

// vehicle.yaml for carxx, whose tires the logs put nearer mu 0.45 than 0.55.
VehicleParameters Carxx(const float mu = 0.55F) {
  VehicleParameters parameters;
  parameters.mass_kg = 2.465F;
  parameters.yaw_inertia_kgm2 = 0.0305F;
  parameters.cg_to_front_m = 0.13F;
  parameters.cg_to_rear_m = 0.13F;
  parameters.front_cornering_stiffness_nprad = 100.0F;
  parameters.rear_cornering_stiffness_nprad = 100.0F;
  parameters.front_friction_coefficient = mu;
  parameters.rear_friction_coefficient = mu;
  parameters.wheel_radius_m = 0.031F;
  parameters.driven_wheel_inertia_kgm2 = 1.5e-4F;
  parameters.locked_awd = true;
  parameters.front_brake_bias = 0.5F;
  return parameters;
}

BodyState Rolling(const float yaw_rate, const float speed) {
  // Wheels rolling at ground speed, so no slip ratio and the fit is lateral.
  return BodyState{{yaw_rate, speed, 0.0F, speed}};
}

Control Steer(const float steer) {
  Control control;
  control[kSteering] = steer;
  return control;
}

constexpr float kTrueMu = 0.45F;

TEST(MuEstimator, NominalIsTheLoadWeightedVehicleYamlValue) {
  auto parameters = Carxx();
  parameters.front_friction_coefficient = 0.5F;
  parameters.rear_friction_coefficient = 0.7F;
  parameters.cg_to_front_m = 0.10F;  // 60 % of the load on the front axle
  parameters.cg_to_rear_m = 0.15F;
  const MuEstimator estimator(parameters);
  EXPECT_NEAR(estimator.nominal_mu(), 0.6F * 0.5F + 0.4F * 0.7F, 1.0e-5F);
}

TEST(MuEstimator, SaturatedFrontAxleRecoversTheTrueMu) {
  // 0.3 rad of steer at 3 m/s puts the front slip angle far past saturation.
  const auto state = Rolling(1.2F, 3.0F);
  const auto control = Steer(0.3F);
  const auto truth = PredictGrip(
    Carxx(kTrueMu), ModelKind::kDynamicBicycleFiala4ws, state, control);

  MuEstimator estimator(Carxx());
  const auto output = estimator.Update(0.0, state, control, truth.ax_mps2, truth.ay_mps2);
  ASSERT_TRUE(output.sample_accepted);
  EXPECT_NEAR(output.sample_mu, kTrueMu, 0.01F);
  EXPECT_GT(output.sample_sensitivity, 2.0F);
  EXPECT_NEAR(output.mu, kTrueMu, 0.01F);
  EXPECT_TRUE(output.has_estimate);
}

TEST(MuEstimator, LinearTiresSayNothingAboutMu) {
  // Small steer: both slip angles stay in the linear region, where every mu
  // on the grid predicts almost the same acceleration.
  const auto state = Rolling(0.9F, 3.0F);
  const auto control = Steer(0.06F);
  const auto truth = PredictGrip(
    Carxx(kTrueMu), ModelKind::kDynamicBicycleFiala4ws, state, control);

  MuEstimator estimator(Carxx());
  const auto output = estimator.Update(0.0, state, control, truth.ax_mps2, truth.ay_mps2);
  EXPECT_FALSE(output.sample_accepted);
  EXPECT_TRUE(std::isnan(output.sample_mu));
  EXPECT_LT(output.sample_sensitivity, 2.0F);
  EXPECT_FALSE(output.has_estimate);
  EXPECT_FLOAT_EQ(output.mu, estimator.nominal_mu());
}

TEST(MuEstimator, SlowAndNonFiniteSamplesAreRejected) {
  MuEstimator estimator(Carxx());
  const auto slow = estimator.Update(0.0, Rolling(0.5F, 0.8F), Steer(0.3F), 0.0F, 3.0F);
  EXPECT_FALSE(slow.sample_accepted);

  auto state = Rolling(1.2F, 3.0F);
  state[kSideslip] = std::numeric_limits<float>::quiet_NaN();
  const auto nan = estimator.Update(0.02, state, Steer(0.3F), 0.0F, 3.0F);
  EXPECT_FALSE(nan.sample_accepted);
  EXPECT_FLOAT_EQ(nan.mu, estimator.nominal_mu());
}

TEST(MuEstimator, EstimateHoldsThroughUninformativeStretches) {
  const auto limit_state = Rolling(1.2F, 3.0F);
  const auto limit_control = Steer(0.3F);
  const auto limit = PredictGrip(
    Carxx(kTrueMu), ModelKind::kDynamicBicycleFiala4ws, limit_state, limit_control);
  const auto cruise_state = Rolling(0.0F, 3.0F);
  const auto cruise_control = Steer(0.0F);

  MuEstimator estimator(Carxx());
  double time_s = 0.0;
  for (int i = 0; i < 50; ++i, time_s += 0.02) {
    (void)estimator.Update(time_s, limit_state, limit_control, limit.ax_mps2, limit.ay_mps2);
  }
  MuEstimator::Output output;
  for (int i = 0; i < 500; ++i, time_s += 0.02) {
    output = estimator.Update(time_s, cruise_state, cruise_control, 0.0F, 0.0F);
  }
  // Ten seconds of straight driving decays the confidence, not the value.
  EXPECT_FALSE(output.sample_accepted);
  EXPECT_NEAR(output.mu, kTrueMu, 0.01F);
  EXPECT_LT(output.confidence, 0.1F);
  EXPECT_TRUE(output.has_estimate);
}

TEST(MuEstimator, FitThatCannotMatchTheMeasurementIsRejected) {
  // Measured lateral acceleration pointing the wrong way: no mu explains it,
  // so the model state is wrong, not the floor.
  MuEstimator estimator(Carxx());
  const auto output = estimator.Update(0.0, Rolling(1.2F, 3.0F), Steer(0.3F), 0.0F, -3.0F);
  EXPECT_FALSE(output.sample_accepted);
  EXPECT_GT(output.sample_residual_mps2, 1.5F);
}

TEST(MuEstimator, FitOnTheGridEdgeIsRejected) {
  // Truth outside the grid: the best fit clips to mu_max and says only that
  // mu is at least that, so it must not pull the estimate.
  const auto state = Rolling(1.2F, 3.0F);
  const auto control = Steer(0.3F);
  const auto truth = PredictGrip(
    Carxx(0.9F), ModelKind::kDynamicBicycleFiala4ws, state, control);

  MuEstimatorConfig config;
  config.mu_max = 0.6F;
  MuEstimator estimator(Carxx(), config);
  const auto output = estimator.Update(0.0, state, control, truth.ax_mps2, truth.ay_mps2);
  EXPECT_FALSE(output.sample_accepted);
  EXPECT_FALSE(output.has_estimate);
}

}  // namespace
}  // namespace xxcar::mppi
