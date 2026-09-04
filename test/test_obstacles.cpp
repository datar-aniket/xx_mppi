#include <gtest/gtest.h>

#include <cmath>

#include "xx_mppi/obstacles/pose_history.hpp"
#include "xx_mppi/obstacles/laser_deskew.hpp"
#include "xx_mppi/obstacles/signed_distance_field.hpp"

namespace xxcar::mppi {
namespace {

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
