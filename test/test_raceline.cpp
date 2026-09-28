#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "xx_mppi/controller/mppi_controller.hpp"
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

TEST(Raceline, AcceptsWindowsCrlfAndClosesSampledLoop) {
  const auto path = std::filesystem::temp_directory_path() /
    "xx_mppi_test_raceline_crlf.csv";
  {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(stream.good());
    stream << "s, k, E, N, phi, V, e_min, e_max\r\n"
           << "0, 0, 0, 0, 0, 1, -1, 1\r\n"
           << "1, 0, 0, 1, 0, 1, -1, 1\r\n"
           << "2, 0, 0.1, 0, 0, 1, -1, 1\r\n";
  }

  const auto raceline = Raceline::LoadCsv(path.string());
  std::filesystem::remove(path);
  EXPECT_TRUE(raceline.closed());
  ASSERT_EQ(raceline.points().size(), 4U);
  EXPECT_NEAR(raceline.length(), 2.1F, 1.0e-6F);
  EXPECT_FLOAT_EQ(raceline.points().back().east_m, 0.0F);
  EXPECT_FLOAT_EQ(raceline.points().back().north_m, 0.0F);
  EXPECT_FLOAT_EQ(raceline.points().back().e_max_m, 1.0F);
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

// Open 20 m straight heading north at 4 m/s with bounds of +/-1 m.
Raceline StraightRaceline() {
  const auto path = std::filesystem::temp_directory_path() /
    "xx_mppi_test_raceline_straight.csv";
  {
    std::ofstream stream(path, std::ios::trunc);
    stream << "s,k,E,N,phi,V,e_min,e_max\n";
    for (int i = 0; i <= 20; ++i) {
      stream << i << ",0,0," << i << ",0,4,-1,1\n";
    }
  }
  auto raceline = Raceline::LoadCsv(path.string());
  std::filesystem::remove(path);
  return raceline;
}

TEST(Raceline, SpeedLimitScalesAndStopsReference) {
  const auto raceline = StraightRaceline();
  const auto horizon = raceline.Sample(1.0F, 60U, 0.05F, SpeedLimit{0.5F, 4.0F, 2.0F});
  // sqrt(2 * 2 * 3) = 3.46 m/s allows the full scaled 2 m/s at the start.
  EXPECT_FLOAT_EQ(horizon.speed_profile.front(), 2.0F);
  EXPECT_FLOAT_EQ(horizon.states.front()[kSpeed], 2.0F);
  for (std::size_t i = 1; i < horizon.s_grid.size(); ++i) {
    EXPECT_GE(horizon.s_grid[i], horizon.s_grid[i - 1U]);
    EXPECT_LE(horizon.s_grid[i], 4.0F + 1.0e-4F);
  }
  EXPECT_LT(horizon.speed_profile.back(), 0.5F);

  const auto stopped = raceline.Sample(5.0F, 10U, 0.05F, SpeedLimit{1.0F, 4.0F, 2.0F});
  EXPECT_FLOAT_EQ(stopped.speed_profile.front(), 0.0F);
  EXPECT_FLOAT_EQ(stopped.s_grid.back(), 5.0F);
}

TEST(Raceline, LeadGapIgnoresWallsBehindAndBeyondLookahead) {
  const auto raceline = StraightRaceline();
  // Positive e is west (negative east) on a north-heading path.
  const std::vector<Point2D> points{
    {0.0F, 4.0F},    // car on the line, 3 m ahead
    {0.5F, 4.6F},    // its far corner
    {-0.95F, 2.0F},  // wall return inside wall_margin of e_max
    {0.95F, 2.5F},   // wall return inside wall_margin of e_min
    {0.0F, 0.5F},    // behind
    {0.0F, 7.5F}};   // beyond the lookahead
  const auto gap = FindLeadGap(raceline, 1.0F, points, 5.0F, 0.15F);
  ASSERT_TRUE(gap.has_value());
  EXPECT_NEAR(*gap, 3.0F, 1.0e-3F);

  const std::vector<Point2D> walls_only{{-0.95F, 2.0F}, {0.95F, 2.5F}, {0.0F, 0.5F}};
  EXPECT_FALSE(FindLeadGap(raceline, 1.0F, walls_only, 5.0F, 0.15F).has_value());
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
