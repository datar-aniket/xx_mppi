#include <cmath>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "xx_mppi/costs/map_boundary.hpp"
#include "xx_mppi/costs/cost_evaluator.hpp"
#include "xx_mppi/costs/obstacle_cost.hpp"
#include "xx_mppi/dynamics/analytic_dynamics.hpp"
#include "xx_mppi/dynamics/rollout.hpp"
#include "xx_mppi/obstacles/signed_distance_field.hpp"
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

// A stationary car with stationary wheels has zero slip, not locked wheels.
// Flooring the speed inside the slip RATIO made the model report a full braking
// force at standstill, so every rollout reversed for the first few steps and the
// published path ran backwards along the track before turning around.
TEST(Dynamics, StandstillProducesNoPhantomBrakingForce) {
  const DynamicBicycleFiala model(TestVehicle());
  for (const float speed : {0.0F, 0.2F, 0.5F, 0.9F}) {
    const BodyState rolling{{0.0F, speed, 0.0F, speed}};
    const auto coasting = model.Derivative(rolling, Control{{0.0F, 0.0F}});
    EXPECT_NEAR(coasting[kSpeed], 0.0F, 1.0e-3F) << "speed " << speed;
    // Drive torque may only ever help below the model's speed floor.
    const auto driving = model.Derivative(rolling, Control{{0.0F, 1.0F}});
    EXPECT_GE(driving[kSpeed], -1.0e-3F) << "speed " << speed;
    EXPECT_GT(driving[kDrivenWheelSpeed], 0.0F) << "speed " << speed;
  }
}

TEST(Dynamics, RolloutFromRestAdvancesInsteadOfReversing) {
  const auto raceline = Raceline::LoadCsv(TestCsv());
  const AnalyticDynamics dynamics(ModelKind::kDynamicBicycleFiala, TestVehicle());
  const State rest{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
  const std::vector<Control> controls(40U, Control{0.0F, 1.0F});
  // This profile's driven-wheel loop needs a step below I / (r^2 C) = 6.7 ms;
  // LoadControllerConfig enforces the same bound on the runtime configuration.
  const auto states = RolloutAnalytic(
    dynamics, raceline, rest, controls, 0.05F, 8U, IntegratorKind::kEuler);
  for (std::size_t i = 1; i < states.size(); ++i) {
    EXPECT_GE(states[i][kSpeed], -1.0e-3F) << "step " << i;
    EXPECT_GE(states[i][kPathEvolution], states[i - 1U][kPathEvolution] - 1.0e-4F)
      << "step " << i;
  }
  EXPECT_GT(states.back()[kPathEvolution], 0.5F);
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

void ExpectDerivativeNear(
  const BodyDerivative & actual, const BodyDerivative & expected,
  const float tolerance = 1.0e-5F)
{
  for (std::size_t i = 0; i < kBodyStateDim; ++i) {
    EXPECT_NEAR(actual[i], expected[i], tolerance);
  }
}

// The 1 m/s floor exists for the slip-ANGLE and beta-rate equations, which are
// singular near zero speed. It must not reach the slip RATIO: there the floor
// would compare the real wheel speed against a fictitious 1 m/s of travel, so a
// stationary car with stationary wheels reads as locked wheels under braking.
TEST(Dynamics, FialaFloorsLowSpeedInTheSlipAngleEquationsOnly) {
  const DynamicBicycleFiala model(TestVehicle());
  const Control control{{0.08F, 0.3F}};
  // Matching each state's wheel speed to its own longitudinal speed zeroes the
  // slip ratio in both, leaving the floored slip-angle path as the only term.
  const float beta = 0.04F;
  const BodyState low_speed{{0.2F, 0.25F, beta, 0.25F * std::cos(beta)}};
  const BodyState floored_speed{{0.2F, 1.0F, beta, 1.0F * std::cos(beta)}};
  ExpectDerivativeNear(
    model.Derivative(low_speed, control), model.Derivative(floored_speed, control));

  // With the wheel speed held fixed instead, the two states must differ: their
  // real longitudinal slip is genuinely different.
  const BodyState low_fixed_wheel{{0.2F, 0.25F, beta, 0.4F}};
  const BodyState floored_fixed_wheel{{0.2F, 1.0F, beta, 0.4F}};
  EXPECT_GT(
    model.Derivative(low_fixed_wheel, control)[kSpeed],
    model.Derivative(floored_fixed_wheel, control)[kSpeed] + 1.0F);
}

TEST(Dynamics, FialaFloorsLowReverseSpeedAtNegativeOneMeterPerSecond) {
  const DynamicBicycleFiala model(TestVehicle());
  const Control control{{-0.08F, -0.3F}};
  const float beta = -0.04F;
  const BodyState low_speed{{-0.2F, -0.25F, beta, -0.25F * std::cos(beta)}};
  const BodyState floored_speed{{-0.2F, -1.0F, beta, -1.0F * std::cos(beta)}};

  ExpectDerivativeNear(
    model.Derivative(low_speed, control), model.Derivative(floored_speed, control));
}

TEST(Dynamics, FialaUsesPositiveFloorAtZeroSpeed) {
  const DynamicBicycleFiala model(TestVehicle());
  const Control control{{0.05F, 0.0F}};
  // Zero longitudinal slip in both, so only the floored slip-angle path acts.
  const BodyState stopped{{0.0F, 0.0F, 0.0F, 0.0F}};
  const BodyState positive_floor{{0.0F, 1.0F, 0.0F, 1.0F}};
  ExpectDerivativeNear(
    model.Derivative(stopped, control), model.Derivative(positive_floor, control));

  // Negative zero selects the -1 m/s floor instead, so it behaves like the
  // reverse-floored state rather than the forward one.
  const BodyState negative_zero{{0.0F, -0.0F, 0.0F, 0.0F}};
  const BodyState negative_floor{{0.0F, -1.0F, 0.0F, -1.0F}};
  ExpectDerivativeNear(
    model.Derivative(negative_zero, control), model.Derivative(negative_floor, control));
  EXPECT_NE(
    model.Derivative(negative_zero, control)[kYawRate],
    model.Derivative(stopped, control)[kYawRate]);
}

// The 4WS model is an independent copy of the Fiala physics, not a wrapper
// around it, so that a mistake in the rear-steer terms cannot reach a
// front-steer-only run. This test is what keeps the two copies honest: with the
// rear angle at zero every rear term must collapse to the front-steer-only
// form, exactly, with no small-angle approximation.
TEST(Dynamics, ZeroRearSteerReproducesTheFrontSteerOnlyModel) {
  const auto parameters = TestVehicle();
  const DynamicBicycleFiala front_only(parameters);
  const DynamicBicycleFiala4ws four_wheel(parameters);

  const BodyState states[] = {
    BodyState{{0.0F, 3.0F, 0.0F, 3.0F}},
    BodyState{{0.4F, 6.0F, 0.08F, 6.4F}},
    BodyState{{-0.7F, 2.0F, -0.15F, 1.6F}},
    BodyState{{0.0F, 0.0F, 0.0F, 0.0F}},
    BodyState{{0.2F, -1.5F, 0.05F, -1.2F}},
  };
  const Control controls[] = {
    Control{{0.0F, 0.0F, 0.0F}},
    Control{{0.25F, 1.0F, 0.0F}},
    Control{{-0.30F, -1.0F, 0.0F}},
  };
  for (const auto & state : states) {
    for (const auto & control : controls) {
      ExpectDerivativeNear(
        four_wheel.Derivative(state, control),
        front_only.Derivative(state, control));
    }
  }
}

// Opposite-phase rear steer (rear opposite the front) turns the car harder for
// the same front angle, which is the low-speed agility case for 4WS. Same-phase
// rear steer crabs the car instead, reducing the yaw it develops.
TEST(Dynamics, RearSteerPhaseSetsYawAuthority) {
  const DynamicBicycleFiala4ws model(TestVehicle());
  const BodyState cruising{{0.0F, 4.0F, 0.0F, 4.0F}};
  constexpr float front = 0.20F;

  const float neutral = model.Derivative(
    cruising, Control{{front, 0.0F, 0.0F}})[kYawRate];
  const float opposite = model.Derivative(
    cruising, Control{{front, 0.0F, -0.10F}})[kYawRate];
  const float same = model.Derivative(
    cruising, Control{{front, 0.0F, 0.10F}})[kYawRate];

  EXPECT_GT(neutral, 0.0F);
  EXPECT_GT(opposite, neutral);
  EXPECT_LT(same, neutral);
}

// With the front straight, rear steer alone must still develop a yaw moment,
// and it must point the opposite way a front input of the same sign would.
TEST(Dynamics, RearSteerAloneYawsOppositeTheFrontAxle) {
  const DynamicBicycleFiala4ws model(TestVehicle());
  const BodyState cruising{{0.0F, 4.0F, 0.0F, 4.0F}};

  const float from_front = model.Derivative(
    cruising, Control{{0.15F, 0.0F, 0.0F}})[kYawRate];
  const float from_rear = model.Derivative(
    cruising, Control{{0.0F, 0.0F, 0.15F}})[kYawRate];

  EXPECT_GT(from_front, 0.0F);
  EXPECT_LT(from_rear, 0.0F);
}

TEST(Costs, CrashedRolloutCostsMoreThanInBoundsRollout) {
  const auto raceline = Raceline::LoadCsv(TestCsv());
  const auto reference = raceline.Sample(0.0F, 5U, 0.1F);
  std::vector<State> safe = reference.states;
  std::vector<State> crashed = safe;
  crashed[2U][kLateralDeviation] = 2.0F;
  const CostEvaluator evaluator(CostWeights{}, 0.1F);
  const float safe_cost = evaluator.Evaluate(
    safe, reference.controls, reference, Control{});
  const float crashed_cost = evaluator.Evaluate(
    crashed, reference.controls, reference, Control{});
  EXPECT_GT(crashed_cost, safe_cost + 1000.0F);
}

TEST(Costs, CrashPaddingShrinksBothBoundsTowardRaceline) {
  constexpr float e_min = -1.0F;
  constexpr float e_max = 1.0F;
  constexpr float padding = 0.10F;

  EXPECT_FALSE(EvaluateMapBoundary(0.90F, e_min, e_max, 1.0F, 0.2F, padding).violated);
  EXPECT_TRUE(EvaluateMapBoundary(0.91F, e_min, e_max, 1.0F, 0.2F, padding).violated);
  EXPECT_FALSE(EvaluateMapBoundary(-0.90F, e_min, e_max, 1.0F, 0.2F, padding).violated);
  EXPECT_TRUE(EvaluateMapBoundary(-0.91F, e_min, e_max, 1.0F, 0.2F, padding).violated);

  // These states remain inside the physical CSV bounds when padding is zero.
  EXPECT_FALSE(EvaluateMapBoundary(0.91F, e_min, e_max, 1.0F, 0.2F, 0.0F).violated);
  EXPECT_FALSE(EvaluateMapBoundary(-0.91F, e_min, e_max, 1.0F, 0.2F, 0.0F).violated);
}

TEST(Costs, AddsAsymmetricAccelerationAndPhysicalControlRatePenalties) {
  const auto raceline = Raceline::LoadCsv(TestCsv());
  auto reference = raceline.Sample(0.0F, 2U, 0.1F);
  auto states = reference.states;
  states[0U][kSpeed] = 1.0F;
  states[1U][kSpeed] = 1.2F;  // +2 m/s^2
  states[2U][kSpeed] = 0.9F;  // -3 m/s^2
  std::vector<Control> controls{
    Control{{0.1F, 0.2F}}, Control{{-0.1F, 0.1F}}};
  const Control previous{{0.0F, 0.0F}};

  CostWeights baseline_weights;
  CostWeights rate_weights = baseline_weights;
  rate_weights.longitudinal_acceleration = 2.0F;
  rate_weights.longitudinal_deceleration = 3.0F;
  rate_weights.control_rate[kSteering] = 0.5F;
  rate_weights.control_rate[kWheelTorque] = 0.25F;
  const CostEvaluator baseline(baseline_weights, 0.1F);
  const CostEvaluator with_rates(rate_weights, 0.1F);

  const float added = with_rates.Evaluate(states, controls, reference, previous) -
    baseline.Evaluate(states, controls, reference, previous);
  // acceleration: 2*2^2 + 3*3^2 = 35
  // steering rate: .5*(1^2 + (-2)^2) = 2.5
  // torque rate: .25*(2^2 + (-1)^2) = 1.25
  EXPECT_NEAR(added, 38.75F, 1.0e-3F);
}

TEST(Costs, ObstacleDistancePenalizesBlockedTrajectory) {
  const auto raceline = Raceline::LoadCsv(TestCsv());
  const auto reference = raceline.Sample(0.0F, 5U, 0.1F);
  ObstacleConfig obstacle_config;
  obstacle_config.enabled = true;
  obstacle_config.grid_resolution_m = 0.05F;
  obstacle_config.grid_width_m = 4.0F;
  obstacle_config.grid_height_m = 4.0F;
  obstacle_config.obstacle_inflation_radius_m = 0.05F;
  obstacle_config.distance_weight = 100.0F;
  obstacle_config.influence_distance_m = 0.5F;
  obstacle_config.latch_threshold_m = 0.1F;
  obstacle_config.latching_weight = 1000.0F;
  obstacle_config.footprint_length_m = 0.5F;
  obstacle_config.footprint_width_m = 0.25F;
  SignedDistanceFieldBuilder builder(obstacle_config);
  const auto clear_field = builder.Build({}, {}, 1, 1);
  const auto blocked_field = builder.Build({Point2D{0.0F, 0.0F}}, {}, 2, 2);
  const CostEvaluator evaluator(CostWeights{}, 0.1F);
  const float clear_cost = evaluator.Evaluate(
    reference.states, reference.controls, reference, Control{},
    &clear_field, &raceline, &obstacle_config);
  const float blocked_cost = evaluator.Evaluate(
    reference.states, reference.controls, reference, Control{},
    &blocked_field, &raceline, &obstacle_config);
  EXPECT_GT(blocked_cost, clear_cost + 10.0F);
}

TEST(Costs, ObstacleLatchRegionRejectsSideAndRearObstacles) {
  const auto raceline = Raceline::LoadCsv(TestCsv());
  const auto reference = raceline.Sample(0.0F, 5U, 0.1F);
  const State state = reference.states.front();
  ObstacleConfig config;
  config.enabled = true;
  config.grid_resolution_m = 0.01F;
  config.grid_width_m = 4.0F;
  config.grid_height_m = 4.0F;
  config.maximum_distance_m = 2.0F;
  config.obstacle_inflation_radius_m = 0.01F;
  config.latch_threshold_m = 0.05F;
  config.footprint_length_m = 0.65F;
  config.footprint_width_m = 0.50F;
  config.footprint_circles = 3U;
  // The test raceline points north at s=0. Side and rear points intersect the
  // conservative full footprint, but only the north/front point may latch.
  const auto make_field = [&config](const Point2D point, const std::uint64_t generation) {
      ObstacleField field;
      field.generation = generation;
      field.resolution_m = config.grid_resolution_m;
      field.width = static_cast<std::uint32_t>(
        std::ceil(config.grid_width_m / field.resolution_m));
      field.height = static_cast<std::uint32_t>(
        std::ceil(config.grid_height_m / field.resolution_m));
      field.origin_east_m = -0.5F * config.grid_width_m;
      field.origin_north_m = -0.5F * config.grid_height_m;
      field.signed_distance_m.resize(
        static_cast<std::size_t>(field.width) * field.height);
      for (std::uint32_t y = 0; y < field.height; ++y) {
        for (std::uint32_t x = 0; x < field.width; ++x) {
          const float east = field.origin_east_m +
            (static_cast<float>(x) + 0.5F) * field.resolution_m;
          const float north = field.origin_north_m +
            (static_cast<float>(y) + 0.5F) * field.resolution_m;
          field.signed_distance_m[static_cast<std::size_t>(y) * field.width + x] =
            std::hypot(east - point.east_m, north - point.north_m) -
            config.obstacle_inflation_radius_m;
        }
      }
      return field;
    };
  const auto front = make_field(Point2D{0.0F, 0.45F}, 1U);
  const auto side = make_field(Point2D{0.30F, 0.0F}, 2U);
  const auto rear = make_field(Point2D{0.0F, -0.45F}, 3U);

  const auto vehicle_center = raceline.ToCartesian(
    state[kPathEvolution], state[kLateralDeviation]);
  EXPECT_NEAR(vehicle_center.first, 0.0F, 1.0e-5F);
  EXPECT_NEAR(vehicle_center.second, 0.0F, 1.0e-5F);
  const float front_full = VehicleObstacleClearance(state, raceline, front, config);
  const float front_forward = VehicleForwardObstacleClearance(state, raceline, front, config);
  const float side_full = VehicleObstacleClearance(state, raceline, side, config);
  const float side_forward = VehicleForwardObstacleClearance(state, raceline, side, config);
  const float rear_full = VehicleObstacleClearance(state, raceline, rear, config);
  const float rear_forward = VehicleForwardObstacleClearance(state, raceline, rear, config);
  EXPECT_LT(front_full, config.latch_threshold_m);
  EXPECT_LT(front_forward, config.latch_threshold_m);
  EXPECT_LT(side_full, config.latch_threshold_m);
  EXPECT_GT(side_forward, config.latch_threshold_m);
  EXPECT_LT(rear_full, config.latch_threshold_m);
  EXPECT_GT(rear_forward, config.latch_threshold_m);

  bool latched = false;
  (void)EvaluateObstacleCost(side_full, side_forward, config, latched, 6U);
  EXPECT_FALSE(latched);
  (void)EvaluateObstacleCost(rear_full, rear_forward, config, latched, 6U);
  EXPECT_FALSE(latched);
  (void)EvaluateObstacleCost(front_full, front_forward, config, latched, 6U);
  EXPECT_TRUE(latched);
}

// The decomposition is only useful if it accounts for the whole cost. A term
// added to the evaluator but forgotten in the breakdown shows up here as a
// mismatch rather than as a quietly wrong debug topic.
TEST(Costs, BreakdownTermsSumToTheTotalCost) {
  const auto raceline = Raceline::LoadCsv(TestCsv());
  auto reference = raceline.Sample(0.0F, 5U, 0.1F);
  auto states = reference.states;
  // Exercise every branch: an off-raceline excursion, asymmetric acceleration
  // and deceleration, and a control sequence that is not the feed-forward one.
  states[2U][kLateralDeviation] = 0.4F;
  states[3U][kLateralDeviation] = 2.0F;  // outside the bounds, latches the crash
  states[1U][kSpeed] = states[0U][kSpeed] + 0.3F;
  states[2U][kSpeed] = states[1U][kSpeed] - 0.5F;
  states[4U][kSideslip] = 1.2F;  // latches the sideslip kill
  std::vector<Control> controls(reference.controls.size());
  for (std::size_t t = 0; t < controls.size(); ++t) {
    controls[t][kSteering] = 0.1F * static_cast<float>(t % 3U) - 0.1F;
    controls[t][kWheelTorque] = 0.5F - 0.2F * static_cast<float>(t % 2U);
  }
  CostWeights weights;
  weights.longitudinal_acceleration = 2.0F;
  weights.longitudinal_deceleration = 3.0F;
  weights.control_rate[kSteering] = 0.5F;
  weights.control_rate[kWheelTorque] = 0.25F;
  const CostEvaluator evaluator(weights, 0.1F);

  CostTerms terms;
  const float total = evaluator.Evaluate(
    states, controls, reference, Control{}, nullptr, nullptr, nullptr, &terms);

  ASSERT_TRUE(std::isfinite(total));
  EXPECT_NEAR(terms.total(), total, 1.0e-3F * std::abs(total));
  // The detail splits must partition their grouped term, not shadow it.
  float tracking = 0.0F;
  for (std::size_t i = 0; i < kStateDim; ++i) {
    tracking += terms.reference_tracking[i];
  }
  EXPECT_NEAR(tracking, terms.values[kTermReferenceTracking],
    1.0e-3F * std::abs(tracking));
  for (std::size_t i = 0; i < kControlDim; ++i) {
    EXPECT_GE(terms.control_effort[i], 0.0F);
  }
  EXPECT_NEAR(
    terms.control_rate[kSteering] + terms.control_rate[kWheelTorque],
    terms.values[kTermControlRate], 1.0e-3F);
  // Latches fired at the step the trajectory first violated each limit.
  EXPECT_EQ(terms.first_crash_step, 3U);
  EXPECT_EQ(terms.first_sideslip_step, 4U);
  EXPECT_EQ(terms.first_obstacle_step, kNoCostLatch);
  // progress rewards arc length, so it is the one term that must be negative.
  EXPECT_LT(terms.values[kTermProgress], 0.0F);
}

}  // namespace
}  // namespace xxcar::mppi
