#include <cmath>
#include <string>

#include <gtest/gtest.h>

#include "xx_mppi/math.hpp"
#include "xx_mppi/reference/raceline.hpp"

namespace xxcar::mppi {
namespace {

std::string TestCsv() {
  return std::string(XX_MPPI_TEST_DATA_DIR) + "/test_raceline.csv";
}

TEST(Raceline, LoadsAndDetectsClosedTrack) {
  const auto raceline = Raceline::LoadCsv(TestCsv());
  EXPECT_TRUE(raceline.closed());
  EXPECT_EQ(raceline.points().size(), 5U);
  EXPECT_FLOAT_EQ(raceline.length(), 4.0F);
}

TEST(Raceline, ProjectionTracksUnwrappedSAcrossSeam) {
  const auto raceline = Raceline::LoadCsv(TestCsv());
  const auto projection = raceline.Project(
    0.0F, 0.05F, 0.0F, 4.05F, 0.4F);
  ASSERT_TRUE(projection.valid);
  EXPECT_GT(projection.s_m, 3.5F);
  const auto next = raceline.Project(0.0F, 0.20F, 0.0F, 4.1F, 0.5F);
  ASSERT_TRUE(next.valid);
  EXPECT_GT(next.s_m, 4.0F);
}

TEST(Raceline, SamplesEqualControlAndPublishedStateHorizons) {
  const auto raceline = Raceline::LoadCsv(TestCsv());
  const auto horizon = raceline.Sample(0.0F, 50U, 0.1F);
  EXPECT_EQ(horizon.states.size(), 51U);
  EXPECT_EQ(horizon.controls.size(), 50U);
  EXPECT_EQ(horizon.s_grid.size(), 51U);
  EXPECT_NEAR(horizon.s_grid.back(), 5.0F, 1.0e-5F);
}

TEST(Raceline, ConvertsPositiveLeftDeviationToCartesian) {
  const auto raceline = Raceline::LoadCsv(TestCsv());
  const auto point = raceline.ToCartesian(0.0F, 0.2F);
  EXPECT_NEAR(point.first, -0.2F, 1.0e-5F);
  EXPECT_NEAR(point.second, 0.0F, 1.0e-5F);
}

TEST(Frames, ConvertsRosYawToEpicCsvHeading) {
  EXPECT_NEAR(enu_yaw_to_heading_from_north(0.5F * kPi), 0.0F, 1.0e-6F);
  EXPECT_NEAR(enu_yaw_to_heading_from_north(0.0F), -0.5F * kPi, 1.0e-6F);
}

}  // namespace
}  // namespace xxcar::mppi
