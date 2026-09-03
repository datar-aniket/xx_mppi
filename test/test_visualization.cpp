#include <cmath>
#include <string>

#include <gtest/gtest.h>
#include <rclcpp/time.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include "xx_mppi/ros/visualization.hpp"

namespace xxcar::mppi {
namespace {

PlannedTrajectory TestTrajectory() {
  PlannedTrajectory trajectory;
  trajectory.solution_pose_time_ns = 2'000'000'000LL;
  trajectory.dt_s = 0.1F;
  trajectory.states = {{0.0F, 0.0F}, {1.0F, 0.0F}, {2.0F, 1.0F}};
  trajectory.controls.resize(trajectory.states.size());
  trajectory.sampled_rollouts = {
    WeightedRollout{{State{{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F}},
      State{{0.0F, 0.0F, 0.0F, 0.0F, 0.1F, 0.0F, 1.0F}},
      State{{0.0F, 0.0F, 0.0F, 0.0F, 0.2F, 0.0F, 2.0F}}}, 0.4F},
    WeightedRollout{{State{{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F}},
      State{{0.0F, 0.0F, 0.0F, 0.0F, -0.1F, 0.0F, 1.0F}},
      State{{0.0F, 0.0F, 0.0F, 0.0F, -0.2F, 0.0F, 2.0F}}}, 0.2F}};
  return trajectory;
}

TEST(Visualization, ConvertsExpectedTrajectoryToPath) {
  const auto path = ToPlannedPath(TestTrajectory(), rclcpp::Time(3'000'000'000LL), "map");

  ASSERT_EQ(path.poses.size(), 3U);
  EXPECT_EQ(path.header.frame_id, "map");
  EXPECT_EQ(rclcpp::Time(path.poses[0].header.stamp).nanoseconds(), 2'000'000'000LL);
  EXPECT_NEAR(
    static_cast<double>(rclcpp::Time(path.poses[2].header.stamp).nanoseconds()),
    2'200'000'000.0, 5.0);
  EXPECT_NEAR(path.poses[0].pose.orientation.z, 0.0, 1.0e-6);
  EXPECT_NEAR(path.poses[0].pose.orientation.w, 1.0, 1.0e-6);
}

TEST(Visualization, CreatesOneLineMarkerPerBestWeightedRollout) {
  const auto raceline = Raceline::LoadCsv(
    std::string(XX_MPPI_TEST_DATA_DIR) + "/test_raceline.csv");
  const auto markers = ToTrajectoryMarkers(
    TestTrajectory(), raceline, rclcpp::Time(3'000'000'000LL), "map");

  ASSERT_EQ(markers.markers.size(), 2U);
  EXPECT_EQ(markers.markers[0].type, visualization_msgs::msg::Marker::LINE_STRIP);
  EXPECT_EQ(markers.markers[1].type, visualization_msgs::msg::Marker::LINE_STRIP);
  EXPECT_EQ(markers.markers[0].ns, "rollouts");
  EXPECT_EQ(markers.markers[0].points.size(), 3U);
  EXPECT_TRUE(std::isfinite(markers.markers[0].points.back().x));
  EXPECT_TRUE(std::isfinite(markers.markers[1].points.back().y));
}

TEST(Visualization, CreatesRacelineAndBothTrackBoundaryPaths) {
  const auto raceline = Raceline::LoadCsv(
    std::string(XX_MPPI_TEST_DATA_DIR) + "/test_raceline.csv");
  const auto paths = ToStaticVisualizationPaths(
    raceline, rclcpp::Time(3'000'000'000LL), "map");

  const auto count = raceline.points().size();
  ASSERT_EQ(paths.raceline.poses.size(), count);
  ASSERT_EQ(paths.left_boundary.poses.size(), count);
  ASSERT_EQ(paths.right_boundary.poses.size(), count);
  EXPECT_NEAR(paths.raceline.poses.front().pose.position.x, 0.0, 1.0e-6);
  EXPECT_NEAR(paths.left_boundary.poses.front().pose.position.x, -1.0, 1.0e-6);
  EXPECT_NEAR(paths.right_boundary.poses.front().pose.position.x, 1.0, 1.0e-6);
  EXPECT_NEAR(
    paths.raceline.poses.front().pose.orientation.z, std::sqrt(0.5), 1.0e-6);
}

}  // namespace
}  // namespace xxcar::mppi
