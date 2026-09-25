#include <cmath>
#include <string>

#include <gtest/gtest.h>

#include "xx_mppi/controller/cuda_mppi.hpp"
#include "xx_mppi/controller/mppi_controller.hpp"
#include "xx_mppi/reference/raceline.hpp"

using namespace xxcar::mppi;

namespace {

std::string TestCsv() {
  return std::string(XX_MPPI_TEST_DATA_DIR) + "/curved_raceline.csv";
}

// Wide sampling and no smoothing cost make the unconstrained solver jump
// freely, so only the hard limit can keep the plan inside it.
TEST(ControlRateLimit, EveryPlannedStepRespectsTheSlewLimit) {
  const auto raceline = Raceline::LoadCsv(TestCsv());
  MppiConfig config;
  config.num_samples = 1024U;
  config.horizon = 30U;
  config.dt_s = 0.1F;
  config.integration_substeps = 8U;
  config.noise_smoothing_window = 1U;
  config.sigma = {0.6F, 0.5F, 0.0F};
  config.control_min = {-0.4F, -1.0F, 0.0F};
  config.control_max = {0.4F, 0.8F, 0.0F};
  config.control_rate_limit = {1.0F, 0.0F, 0.0F};
  CostWeights costs;
  costs.control_smoothness = {0.0F, 0.0F, 0.0F};
  CudaMppiController controller(
    config, costs, VehicleParameters{}, ObstacleConfig{},
    ModelKind::kDynamicBicycleFiala, raceline);
  if (!controller.using_cuda()) {
    GTEST_SKIP() << "CUDA backend unavailable";
  }
  const auto reference = raceline.Sample(raceline.s_min(), config.horizon, config.dt_s);
  const float step = config.control_rate_limit[kSteering] * config.dt_s;
  // Start at full left lock so the first transition is itself constrained.
  Control previous{{0.4F, 0.0F, 0.0F}};
  for (int solve = 0; solve < 20; ++solve) {
    const auto solution = controller.Solve(
      reference.states.front(), reference, previous, reference.s_grid.front(),
      0.0F, solve == 0);
    ASSERT_EQ(solution.controls.size(), config.horizon);
    Control prior = previous;
    for (const auto & control : solution.controls) {
      EXPECT_LE(std::fabs(control[kSteering] - prior[kSteering]), step + 1.0e-5F);
      prior = control;
    }
    previous = solution.controls.front();
  }
}

TEST(ControlRateLimit, SlewLimitUsesElapsedTimeAndSkipsUnlimitedChannels) {
  const Control previous{{0.0F, 0.0F, 0.1F}};
  const Control target{{0.5F, -2.0F, -0.5F}};
  const std::array<float, kControlDim> limit{5.4F, 0.0F, 5.6F};
  const auto limited = SlewLimitControl(target, previous, 0.01F, limit);
  EXPECT_NEAR(limited[kSteering], 0.054F, 1.0e-6F);
  EXPECT_FLOAT_EQ(limited[kWheelTorque], -2.0F);
  EXPECT_NEAR(limited[kRearSteering], 0.1F - 0.056F, 1.0e-6F);
  // A step already inside the limit passes through unchanged.
  const auto small = SlewLimitControl(Control{{0.01F, 0.0F, 0.1F}}, previous, 0.01F, limit);
  EXPECT_FLOAT_EQ(small[kSteering], 0.01F);
  // Non-positive or non-finite elapsed time holds the previous command.
  EXPECT_FLOAT_EQ(SlewLimitControl(target, previous, -1.0F, limit)[kSteering], 0.0F);
  EXPECT_FLOAT_EQ(SlewLimitControl(target, previous, NAN, limit)[kSteering], 0.0F);
}

}  // namespace
