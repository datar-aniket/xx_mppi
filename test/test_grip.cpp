#include <cmath>
#include <limits>

#include <gtest/gtest.h>

#include "xx_mppi/controller/grip_monitor.hpp"
#include "xx_mppi/dynamics/grip.hpp"
#include "xx_mppi/dynamics/models/dynamic_bicycle_fiala.hpp"

namespace xxcar::mppi {
namespace {

VehicleParameters Carxx() {
  VehicleParameters parameters;
  parameters.mass_kg = 2.465F;
  parameters.yaw_inertia_kgm2 = 0.03F;
  parameters.cg_to_front_m = 0.13F;
  parameters.cg_to_rear_m = 0.13F;
  parameters.front_cornering_stiffness_nprad = 200.0F;
  parameters.rear_cornering_stiffness_nprad = 200.0F;
  parameters.front_friction_coefficient = 0.5F;
  parameters.rear_friction_coefficient = 0.6F;
  parameters.wheel_radius_m = 0.031F;
  parameters.driven_wheel_inertia_kgm2 = 1.5e-4F;
  parameters.locked_awd = true;
  parameters.front_brake_bias = 0.5F;
  parameters.rolling_resistance_n = 1.8F;
  return parameters;
}

BodyState Cornering() {
  return BodyState{{1.2F, 3.0F, -0.08F, 3.1F}};
}

Control Command(const float steer, const float torque, const float rear_steer = 0.0F) {
  Control control;
  control[kSteering] = steer;
  control[kWheelTorque] = torque;
  control[kRearSteering] = rear_steer;
  return control;
}

TEST(Grip, ReportingDoesNotChangeTheDerivative) {
  const DynamicBicycleFiala4ws model(Carxx());
  const auto state = Cornering();
  const auto control = Command(0.2F, 0.15F, -0.1F);
  TireReport report;
  const auto plain = model.Derivative(state, control);
  const auto reported = model.Evaluate(state, control, &report);
  for (std::size_t i = 0; i < kBodyStateDim; ++i) {
    EXPECT_EQ(plain[i], reported[i]);
  }
}

TEST(Grip, BodyAccelerationMatchesTheModelsOwnForceBalance) {
  const auto parameters = Carxx();
  const DynamicBicycleFiala4ws model(parameters);
  const auto state = Cornering();
  const auto control = Command(0.2F, 0.15F, -0.1F);
  const auto derivative = model.Derivative(state, control);
  const auto sample = PredictGrip(
    parameters, ModelKind::kDynamicBicycleFiala4ws, state, control);
  const float beta = state[kSideslip];
  const float speed = state[kSpeed];
  // Speed rate is the body acceleration along the velocity; v (r + beta_dot)
  // is the part normal to it.
  EXPECT_NEAR(
    derivative[kSpeed],
    sample.ax_mps2 * std::cos(beta) + sample.ay_mps2 * std::sin(beta), 1.0e-4F);
  EXPECT_NEAR(
    speed * (state[kYawRate] + derivative[kSideslip]),
    -sample.ax_mps2 * std::sin(beta) + sample.ay_mps2 * std::cos(beta), 1.0e-4F);
}

TEST(Grip, UtilizationIsBoundedByTheTireModelAndReachesOneWhenSliding) {
  const auto parameters = Carxx();
  const auto gentle = PredictGrip(
    parameters, ModelKind::kDynamicBicycleFiala4ws,
    BodyState{{0.3F, 2.0F, 0.0F, 2.0F}}, Command(0.02F, 0.0F));
  EXPECT_GT(gentle.front_utilization, 0.0F);
  EXPECT_LT(gentle.front_utilization, 0.5F);
  // A large front slip angle saturates the front axle.
  const auto sliding = PredictGrip(
    parameters, ModelKind::kDynamicBicycleFiala4ws,
    BodyState{{0.0F, 3.0F, 0.0F, 3.0F}}, Command(0.4F, 0.0F));
  EXPECT_NEAR(sliding.front_utilization, 1.0F, 1.0e-3F);
  EXPECT_LE(sliding.rear_utilization, 1.0F + 1.0e-4F);
}

TEST(Grip, FrontSteerOnlyModelIgnoresRearSteerAndOthersHaveNoTires) {
  const auto parameters = Carxx();
  const auto state = Cornering();
  const auto two = PredictGrip(
    parameters, ModelKind::kDynamicBicycleFiala, state, Command(0.2F, 0.1F, 0.2F));
  const auto four = PredictGrip(
    parameters, ModelKind::kDynamicBicycleFiala4ws, state, Command(0.2F, 0.1F, 0.0F));
  EXPECT_FLOAT_EQ(two.ay_mps2, four.ay_mps2);
  EXPECT_TRUE(std::isnan(PredictGrip(
      parameters, ModelKind::kKinematicBicycle, state, Command(0.2F, 0.1F)).ay_mps2));
  // Equal axle loads: the load-weighted limit is the mean of the two mu.
  EXPECT_NEAR(FrictionLimitMps2(parameters), 9.81F * 0.55F, 1.0e-4F);
}

TEST(Grip, RollingResistanceSlowsACoastingCarAndVanishesAtRest) {
  auto parameters = Carxx();
  // Free rolling: wheel speed equals ground speed, no torque, no steer.
  const BodyState rolling{{0.0F, 3.0F, 0.0F, 3.0F}};
  const auto with = DynamicBicycleFiala4ws(parameters).Derivative(rolling, Command(0.0F, 0.0F));
  parameters.rolling_resistance_n = 0.0F;
  const auto without = DynamicBicycleFiala4ws(parameters).Derivative(
    rolling, Command(0.0F, 0.0F));
  EXPECT_NEAR(with[kSpeed] - without[kSpeed], -1.8F / 2.465F, 1.0e-4F);
  EXPECT_EQ(with[kYawRate], without[kYawRate]);
  EXPECT_EQ(with[kSideslip], without[kSideslip]);
  // Both Fiala models apply it identically at zero rear steer.
  parameters.rolling_resistance_n = 1.8F;
  const auto two = DynamicBicycleFiala(parameters).Derivative(rolling, Command(0.1F, 0.1F));
  const auto four = DynamicBicycleFiala4ws(parameters).Derivative(rolling, Command(0.1F, 0.1F));
  EXPECT_NEAR(two[kSpeed], four[kSpeed], 1.0e-5F);
  // A stationary car gets no resistance force, so it cannot be pushed backward.
  const BodyState rest{{0.0F, 0.0F, 0.0F, 0.0F}};
  const auto stopped = DynamicBicycleFiala4ws(parameters).Derivative(rest, Command(0.0F, 0.0F));
  parameters.rolling_resistance_n = 0.0F;
  EXPECT_EQ(
    stopped[kSpeed],
    DynamicBicycleFiala4ws(parameters).Derivative(rest, Command(0.0F, 0.0F))[kSpeed]);
}

TEST(GripMonitor, LatchesWhenTheCarFallsShortOfTheModelAndReleases) {
  GripMonitor monitor;
  const double dt = 0.02;
  double t = 0.0;
  GripMonitor::Output output;
  // Model predicts 6 m/s^2 near its limit, the car only shows 4.5.
  for (int i = 0; i < 10; ++i, t += dt) {
    output = monitor.Update(t, 3.0F, 0.0F, 4.5F, 0.0F, 6.0F, 0.95F);
  }
  EXPECT_NEAR(output.acceleration_shortfall_mps2, 1.5F, 0.05F);
  EXPECT_FALSE(output.model_overestimates_grip);  // held 0.18 s, latch needs 0.3 s
  for (int i = 0; i < 20; ++i, t += dt) {
    output = monitor.Update(t, 3.0F, 0.0F, 4.5F, 0.0F, 6.0F, 0.95F);
  }
  EXPECT_TRUE(output.model_overestimates_grip);
  EXPECT_NEAR(output.observed_peak_mu, 4.5F / 9.81F, 0.01F);
  // Agreement clears it again after the latch time.
  for (int i = 0; i < 40; ++i, t += dt) {
    output = monitor.Update(t, 3.0F, 0.0F, 4.5F, 0.0F, 4.6F, 0.95F);
  }
  EXPECT_FALSE(output.model_overestimates_grip);
}

TEST(GripMonitor, IgnoresShortfallWhenTheModelIsFarFromItsLimitOrHasNoTires) {
  GripMonitor monitor;
  GripMonitor::Output output;
  double t = 0.0;
  for (int i = 0; i < 100; ++i, t += 0.02) {
    output = monitor.Update(t, 3.0F, 0.0F, 1.0F, 0.0F, 3.0F, 0.3F);
  }
  EXPECT_FALSE(output.model_overestimates_grip);
  const float nan = std::numeric_limits<float>::quiet_NaN();
  for (int i = 0; i < 100; ++i, t += 0.02) {
    output = monitor.Update(t, 3.0F, 0.0F, 1.0F, nan, nan, nan);
  }
  EXPECT_TRUE(std::isnan(output.acceleration_shortfall_mps2));
  EXPECT_FALSE(output.model_overestimates_grip);
}

}  // namespace
}  // namespace xxcar::mppi
