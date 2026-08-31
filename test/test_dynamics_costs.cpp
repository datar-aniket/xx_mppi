#include <cmath>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "xx_mppi/costs/cost_evaluator.hpp"
#include "xx_mppi/dynamics/analytic_dynamics.hpp"
#include "xx_mppi/dynamics/rollout.hpp"
#include "xx_mppi/reference/raceline.hpp"

namespace xxcar::mppi {
namespace {

std::string TestCsv() {
  return std::string(XX_MPPI_TEST_DATA_DIR) + "/test_raceline.csv";
}

VehicleParameters TestVehicle() {
  VehicleParameters parameters;
  parameters.mass_kg = 20.0F;
  parameters.yaw_inertia_kgm2 = 1.2F;
  parameters.cg_to_front_m = 0.18F;
  parameters.cg_to_rear_m = 0.18F;
  parameters.front_cornering_stiffness_nprad = 1000.0F;
  parameters.rear_cornering_stiffness_nprad = 1200.0F;
  parameters.wheel_radius_m = 0.05F;
  parameters.driven_wheel_inertia_kgm2 = 0.02F;
  return parameters;
}

TEST(Dynamics, FialaRolloutRemainsFinite) {
  const auto raceline = Raceline::LoadCsv(TestCsv());
  const AnalyticDynamics dynamics(ModelKind::kDynamicBicycleFiala, TestVehicle());
  const State initial{0.0F, 3.0F, 0.0F, 3.0F, 0.0F, 0.0F, 0.0F};
  const std::vector<Control> controls(50U, Control{0.05F, 1.0F});
  const auto states = RolloutAnalytic(
    dynamics, raceline, initial, controls, 0.1F, 4U, IntegratorKind::kEuler);
  ASSERT_EQ(states.size(), 51U);
  for (const auto & state : states) {
    for (const float value : state) {
      EXPECT_TRUE(std::isfinite(value));
    }
  }
}

TEST(Dynamics, LockedAwdUsesBothAxlesForLongitudinalSlip) {
  auto rear_drive_parameters = TestVehicle();
  rear_drive_parameters.locked_awd = false;
  auto awd_parameters = rear_drive_parameters;
  awd_parameters.locked_awd = true;

  const BodyState slipping{{0.0F, 3.0F, 0.0F, 4.0F}};
  const Control coasting{{0.0F, 0.0F}};
  const auto rear_drive = DynamicBicycleFiala(rear_drive_parameters).Derivative(
    slipping, coasting);
  const auto locked_awd = DynamicBicycleFiala(awd_parameters).Derivative(
    slipping, coasting);

  EXPECT_GT(rear_drive[kSpeed], 0.0F);
  EXPECT_GT(locked_awd[kSpeed], 1.8F * rear_drive[kSpeed]);
  EXPECT_LT(locked_awd[kDrivenWheelSpeed], rear_drive[kDrivenWheelSpeed]);
}

TEST(Costs, CrashedRolloutCostsMoreThanInBoundsRollout) {
  const auto raceline = Raceline::LoadCsv(TestCsv());
  const auto reference = raceline.Sample(0.0F, 5U, 0.1F);
  std::vector<State> safe = reference.states;
  std::vector<State> crashed = safe;
  crashed[2U][kLateralDeviation] = 2.0F;
  const CostEvaluator evaluator(CostWeights{});
  const float safe_cost = evaluator.Evaluate(
    safe, reference.controls, reference, Control{});
  const float crashed_cost = evaluator.Evaluate(
    crashed, reference.controls, reference, Control{});
  EXPECT_GT(crashed_cost, safe_cost + 1000.0F);
}

}  // namespace
}  // namespace xxcar::mppi
