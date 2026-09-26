#include <gtest/gtest.h>

#include <cmath>
#include <chrono>

#include "xx_mppi/obstacles/pose_history.hpp"
#include "xx_mppi/obstacles/laser_deskew.hpp"
#include "xx_mppi/obstacles/signed_distance_field.hpp"
#include "xx_mppi/obstacles/temporal_obstacle_filter.hpp"
#include "xx_mppi/safety/obstacle_brake_latch.hpp"
#include "xx_mppi/safety/motor_brake_hysteresis.hpp"

namespace xxcar::mppi {
namespace {

TEST(ObstacleBrakeLatch, RequiresContinuousHazardAndHealthyRecoveryWindows) {
  using namespace std::chrono_literals;
  ObstacleBrakeLatch latch(500ms, 500ms);
  const auto start = std::chrono::steady_clock::time_point{};

  EXPECT_FALSE(latch.Update(true, false, start));
  EXPECT_FALSE(latch.Update(true, false, start + 499ms));
  EXPECT_TRUE(latch.Update(true, false, start + 500ms));
  EXPECT_TRUE(latch.Update(false, false, start + 600ms));
  EXPECT_TRUE(latch.Update(false, false, start + 1099ms));
  EXPECT_FALSE(latch.Update(false, false, start + 1100ms));

  // One healthy population resets a pending activation debounce.
  EXPECT_FALSE(latch.Update(true, false, start + 1200ms));
  EXPECT_FALSE(latch.Update(false, false, start + 1600ms));
  EXPECT_FALSE(latch.Update(true, false, start + 1700ms));
  EXPECT_FALSE(latch.Update(true, false, start + 2199ms));
  EXPECT_TRUE(latch.Update(true, false, start + 2200ms));
}

TEST(ObstacleBrakeLatch, StopReleasesAndSuppressesReengagement) {
  using namespace std::chrono_literals;
  ObstacleBrakeLatch latch(500ms, 500ms);
  const auto start = std::chrono::steady_clock::time_point{};

  EXPECT_FALSE(latch.Update(true, false, start));
  EXPECT_TRUE(latch.Update(true, false, start + 500ms));
  EXPECT_FALSE(latch.Update(true, true, start + 600ms));
  EXPECT_FALSE(latch.Update(true, true, start + 2s));

  // Moving again starts a fresh debounce; stopped time cannot accumulate.
  EXPECT_FALSE(latch.Update(true, false, start + 2100ms));
  EXPECT_FALSE(latch.Update(true, false, start + 2599ms));
  EXPECT_TRUE(latch.Update(true, false, start + 2600ms));
  EXPECT_TRUE(latch.Pause());
}

TEST(MotorBrakeHysteresis, UsesSeparateEngageAndReleaseRpmThresholds) {
  MotorBrakeHysteresis brake(1.6F, 50.0F, 80.0F);

  EXPECT_FLOAT_EQ(brake.Update(80.0F), 0.0F);
  EXPECT_FLOAT_EQ(brake.Update(80.01F), -1.6F);
  EXPECT_FLOAT_EQ(brake.Update(60.0F), -1.6F);
  EXPECT_FLOAT_EQ(brake.Update(50.0F), -1.6F);
  EXPECT_FLOAT_EQ(brake.Update(49.99F), 0.0F);
  EXPECT_FLOAT_EQ(brake.Update(70.0F), 0.0F);
  EXPECT_FLOAT_EQ(brake.Update(-81.0F), 1.6F);
  EXPECT_FLOAT_EQ(brake.Update(-60.0F), 1.6F);
  EXPECT_FLOAT_EQ(brake.Update(-49.0F), 0.0F);
}

TEST(MotorBrakeHysteresis, SignJumpDisarmsBeforeOppositeTorqueCanEngage) {
  MotorBrakeHysteresis brake(1.6F, 50.0F, 80.0F);

  EXPECT_FLOAT_EQ(brake.Update(100.0F), -1.6F);
  EXPECT_FLOAT_EQ(brake.Update(60.0F), -1.6F);
  // A direct sign jump cannot produce a +3.2 Nm command transition.
  EXPECT_FLOAT_EQ(brake.Update(-100.0F), 0.0F);
  EXPECT_FALSE(brake.engaged());
  // A later consistent sample may establish the new direction.
  EXPECT_FLOAT_EQ(brake.Update(-100.0F), 1.6F);
  EXPECT_FLOAT_EQ(brake.Update(-60.0F), 1.6F);
  EXPECT_FLOAT_EQ(brake.Update(100.0F), 0.0F);
  EXPECT_FALSE(brake.engaged());
}

TEST(PoseHistory, InterpolatesYawAcrossWrapAndPosition) {
  PoseHistory history(0.1F, 0.02F);
  history.Add(TimedVehiclePose{
    1'000'000'000, {0.0F, 0.0F, 3.10F}, 1.0F, 0.0F, 0.0F});
  history.Add(TimedVehiclePose{
    1'100'000'000, {0.1F, 0.0F, -3.10F}, 1.0F, 0.0F, 0.0F});
  const auto pose = history.PoseAt(1'050'000'000);
  ASSERT_TRUE(pose);
  EXPECT_NEAR(pose->east_m, 0.05F, 1.0e-5F);
  EXPECT_NEAR(std::abs(pose->yaw_enu_rad), 3.14159265F, 1.0e-3F);
}

TEST(PoseHistory, UsesVelocityForBoundedExtrapolation) {
  PoseHistory history(0.1F, 0.02F);
  history.Add(TimedVehiclePose{
    1'000'000'000, {2.0F, 3.0F, 0.0F}, 2.0F, 0.0F, 1.0F});
  const auto pose = history.PoseAt(1'010'000'000);
  ASSERT_TRUE(pose);
  EXPECT_NEAR(pose->east_m, 2.02F, 1.0e-4F);
  EXPECT_NEAR(pose->yaw_enu_rad, 0.01F, 1.0e-5F);
  EXPECT_FALSE(history.PoseAt(1'030'000'000));
}

TEST(PoseHistory, RetainsBoundarySampleAndRejectsOldTime) {
  PoseHistory history(0.1F, 0.0F);
  history.Add(TimedVehiclePose{1'000'000'000, {}, 0.0F, 0.0F, 0.0F});
  history.Add(TimedVehiclePose{1'050'000'000, {}, 0.0F, 0.0F, 0.0F});
  history.Add(TimedVehiclePose{1'160'000'000, {}, 0.0F, 0.0F, 0.0F});
  EXPECT_EQ(history.size(), 2U);
  EXPECT_FALSE(history.PoseAt(1'049'000'000));
  EXPECT_TRUE(history.PoseAt(1'100'000'000));
}

TEST(SignedDistanceField, IsSignedAndReplacesOnlyCurrentPoints) {
  ObstacleConfig config;
  config.grid_resolution_m = 0.1F;
  config.grid_width_m = 4.0F;
  config.grid_height_m = 4.0F;
  config.maximum_distance_m = 2.0F;
  config.obstacle_inflation_radius_m = 0.15F;
  SignedDistanceFieldBuilder builder(config);
  const auto occupied = builder.Build({Point2D{0.0F, 0.0F}}, {}, 10, 1);
  EXPECT_LT(SampleSignedDistance(occupied, 0.0F, 0.0F, -99.0F), 0.0F);
  EXPECT_GT(SampleSignedDistance(occupied, 1.0F, 0.0F, -99.0F), 0.5F);

  const auto empty = builder.Build({}, {}, 20, 2);
  EXPECT_FLOAT_EQ(SampleSignedDistance(empty, 0.0F, 0.0F, -99.0F), 2.0F);
  EXPECT_EQ(empty.generation, 2U);
}

TEST(SignedDistanceField, BilinearSamplingRejectsOutsideGrid) {
  ObstacleConfig config;
  config.grid_resolution_m = 0.1F;
  config.grid_width_m = 2.0F;
  config.grid_height_m = 2.0F;
  SignedDistanceFieldBuilder builder(config);
  const auto field = builder.Build({Point2D{0.0F, 0.0F}}, {}, 10, 1);
  EXPECT_FLOAT_EQ(SampleSignedDistance(field, 10.0F, 0.0F, -0.25F), -0.25F);
}

TEST(TemporalObstacleFilter, ConfirmsAcrossConsecutiveNearbyUpdates) {
  ObstacleConfig config;
  config.confirmation_updates = 3U;
  config.persistence_updates = 2U;
  config.association_distance_m = 0.10F;
  TemporalObstacleFilter filter(config);

  EXPECT_TRUE(filter.Update({Point2D{1.00F, 2.00F}}).empty());
  EXPECT_TRUE(filter.Update({Point2D{1.04F, 2.01F}}).empty());
  const auto confirmed = filter.Update({Point2D{1.08F, 2.02F}});
  ASSERT_EQ(confirmed.size(), 1U);
  EXPECT_NEAR(confirmed.front().east_m, 1.08F, 1.0e-6F);
}

TEST(TemporalObstacleFilter, RetainsForExactlyConfiguredMissedUpdates) {
  ObstacleConfig config;
  config.confirmation_updates = 1U;
  config.persistence_updates = 2U;
  config.association_distance_m = 0.10F;
  TemporalObstacleFilter filter(config);

  ASSERT_EQ(filter.Update({Point2D{1.0F, 2.0F}}).size(), 1U);
  EXPECT_EQ(filter.Update({}).size(), 1U);
  EXPECT_EQ(filter.Update({}).size(), 1U);
  EXPECT_TRUE(filter.Update({}).empty());
  EXPECT_EQ(filter.track_count(), 0U);
}

TEST(TemporalObstacleFilter, UnconfirmedTrackRequiresConsecutiveUpdates) {
  ObstacleConfig config;
  config.confirmation_updates = 2U;
  config.persistence_updates = 4U;
  TemporalObstacleFilter filter(config);

  EXPECT_TRUE(filter.Update({Point2D{1.0F, 2.0F}}).empty());
  EXPECT_TRUE(filter.Update({}).empty());
  EXPECT_TRUE(filter.Update({Point2D{1.0F, 2.0F}}).empty());
  EXPECT_EQ(filter.Update({Point2D{1.0F, 2.0F}}).size(), 1U);
  filter.Clear();
  EXPECT_EQ(filter.track_count(), 0U);
}

TEST(LaserDeskew, UsesPerRayPoseAndStaticExtrinsic) {
  PoseHistory history(0.1F, 0.0F);
  history.Add(TimedVehiclePose{1'000'000'000, {0.0F, 0.0F, 0.0F}, 1.0F, 0.0F, 0.0F});
  history.Add(TimedVehiclePose{1'025'000'000, {0.025F, 0.0F, 0.0F}, 1.0F, 0.0F, 0.0F});
  LaserScanData scan;
  scan.first_ray_stamp_ns = 1'000'000'000;
  scan.angle_min_rad = 0.0F;
  scan.angle_increment_rad = 0.0F;
  scan.time_increment_s = 0.025F;
  scan.range_min_m = 0.1F;
  scan.range_max_m = 10.0F;
  scan.ranges_m = {1.0F, 1.0F};
  const auto result = DeskewLaserScan(scan, RigidTransform2D{0.1F, 0.0F, 0.0F}, history);
  ASSERT_TRUE(result);
  ASSERT_EQ(result->obstacle_points.size(), 2U);
  EXPECT_NEAR(result->obstacle_points[0].east_m, 1.1F, 1.0e-5F);
  EXPECT_NEAR(result->obstacle_points[1].east_m, 1.125F, 1.0e-5F);
  EXPECT_EQ(result->reference_stamp_ns, 1'025'000'000);
}

TEST(LaserDeskew, RejectsScanOutsidePoseCoverage) {
  PoseHistory history(0.1F, 0.005F);
  history.Add(TimedVehiclePose{1'000'000'000, {}, 0.0F, 0.0F, 0.0F});
  LaserScanData scan;
  scan.first_ray_stamp_ns = 1'000'000'000;
  scan.angle_increment_rad = 0.1F;
  scan.time_increment_s = 0.01F;
  scan.range_min_m = 0.1F;
  scan.range_max_m = 10.0F;
  scan.ranges_m = {1.0F, 1.0F};
  EXPECT_FALSE(DeskewLaserScan(scan, {}, history));
}

}  // namespace
}  // namespace xxcar::mppi
