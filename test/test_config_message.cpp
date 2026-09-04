#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>
#include <rclcpp/time.hpp>

#include "xx_mppi/controller/config.hpp"
#include "xx_mppi/reference/raceline.hpp"
#include "xx_mppi/ros/direct_control_message.hpp"
#include "xx_mppi/ros/runtime_config.hpp"
#include "xx_mppi/ros/trajectory_message.hpp"

namespace xxcar::mppi {
namespace {

TEST(Config, LoadsRuntimeProblemAndPhysicalControlBounds) {
  const auto config = LoadControllerConfig(XX_MPPI_CONFIG_DIR);
  EXPECT_GT(config.mppi.num_samples, 3U);
  EXPECT_GT(config.mppi.horizon, 0U);
  EXPECT_FLOAT_EQ(config.mppi.dt_s, 0.1F);
  EXPECT_LT(config.mppi.control_min[kSteering], 0.0F);
  EXPECT_GT(config.mppi.control_max[kSteering], 0.0F);
  EXPECT_LE(config.mppi.control_min[kWheelTorque], 0.0F);
  EXPECT_GT(config.mppi.control_max[kWheelTorque], 0.0F);
  EXPECT_GT(config.visualization_rate_hz, 0.0F);
  EXPECT_GE(config.solve_rate_hz, config.control_publish_rate_hz);
  EXPECT_GE(config.maximum_solution_age_s, 0.0F);
  EXPECT_GT(config.num_rollouts, 0U);
  EXPECT_GE(config.costs.longitudinal_acceleration, 0.0F);
  EXPECT_GE(config.costs.longitudinal_deceleration, 0.0F);
  EXPECT_GE(config.costs.control_rate[kSteering], 0.0F);
  EXPECT_GE(config.costs.control_rate[kWheelTorque], 0.0F);
  EXPECT_FLOAT_EQ(config.info_log_rate_hz, 10.0F);
  EXPECT_EQ(config.obstacles.confirmation_updates, 2U);
  EXPECT_EQ(config.obstacles.persistence_updates, 4U);
  EXPECT_FLOAT_EQ(config.obstacles.association_distance_m, 0.10F);
}

TEST(Config, LoadsTrx4SportVehicleProfile) {
  const auto config = LoadControllerConfig(XX_MPPI_CONFIG_DIR);
  EXPECT_FLOAT_EQ(config.vehicle.mass_kg, 3.2F);
  EXPECT_FLOAT_EQ(config.vehicle.yaw_inertia_kgm2, 0.030F);
  EXPECT_FLOAT_EQ(config.vehicle.cg_to_front_m + config.vehicle.cg_to_rear_m, 0.26F);
  EXPECT_FLOAT_EQ(config.vehicle.front_cornering_stiffness_nprad, 100.0F);
  EXPECT_FLOAT_EQ(config.vehicle.rear_cornering_stiffness_nprad, 100.0F);
  EXPECT_FLOAT_EQ(config.vehicle.wheel_radius_m, 0.05842F);
  EXPECT_FLOAT_EQ(config.vehicle.front_brake_bias, 0.5F);
  EXPECT_TRUE(config.vehicle.locked_awd);
}

namespace {

// Copies the shipped config into a scratch directory so a test can override one
// file without disturbing the config the vehicle actually runs.
std::filesystem::path MakeOverlayConfig(const std::string & name) {
  const auto directory = std::filesystem::temp_directory_path() / name;
  std::filesystem::remove_all(directory);
  std::filesystem::copy(
    XX_MPPI_CONFIG_DIR, directory,
    std::filesystem::copy_options::recursive |
    std::filesystem::copy_options::overwrite_existing);
  return directory;
}

void WriteFile(const std::filesystem::path & path, const std::string & contents) {
  std::ofstream stream(path, std::ios::trunc);
  ASSERT_TRUE(stream.good());
  stream << contents;
}

}  // namespace

TEST(Config, DerivesRacelineCsvFromCurrentMapDirectory) {
  const auto directory = MakeOverlayConfig("xx_mppi_test_current_map_config");
  WriteFile(
    directory / "model.yaml",
    "name: kinematic_bicycle\n"
    "raceline_path: \"${current_map}\"\n"
    "vehicle_params_path: vehicle.yaml\n"
    "projection_window_m: 10.0\n");
  const auto config = LoadControllerConfig(directory.string());
  std::filesystem::remove_all(directory);
  EXPECT_EQ(
    std::filesystem::path(config.raceline_path).filename(),
    "current_map_frenet_map.csv");
}

TEST(Config, SelectsTheRolloutFrame) {
  const auto directory = MakeOverlayConfig("xx_mppi_test_frame_config");
  const auto mppi_path = directory / "mppi.yaml";
  std::ifstream original(mppi_path);
  const std::string body(
    (std::istreambuf_iterator<char>(original)), std::istreambuf_iterator<char>());
  original.close();

  // An explicit key wins over whatever the shipped config currently selects.
  WriteFile(mppi_path, "frame: cartesian\n" + body);
  EXPECT_EQ(LoadControllerConfig(directory.string()).mppi.frame, FrameKind::kCartesian);
  WriteFile(mppi_path, "frame: frenet\n" + body);
  EXPECT_EQ(LoadControllerConfig(directory.string()).mppi.frame, FrameKind::kFrenet);

  WriteFile(mppi_path, "frame: polar\n" + body);
  EXPECT_THROW((void)LoadControllerConfig(directory.string()), std::runtime_error);
  std::filesystem::remove_all(directory);
}

TEST(Config, RejectsAnIntegrationStepTheDrivenWheelLoopCannotHold) {
  const auto directory = MakeOverlayConfig("xx_mppi_test_substep_config");
  const auto mppi_path = directory / "mppi.yaml";
  std::ifstream original(mppi_path);
  const std::string body(
    (std::istreambuf_iterator<char>(original)), std::istreambuf_iterator<char>());
  original.close();

  // One substep at dt = 0.1 s is above the stability limit for this profile and
  // makes the wheel speed ring, which reverses the rollout at low speed.
  WriteFile(mppi_path, "integration_substeps: 1\n" + body);
  EXPECT_THROW((void)LoadControllerConfig(directory.string()), std::runtime_error);
  WriteFile(mppi_path, "integration_substeps: 4\n" + body);
  EXPECT_NO_THROW((void)LoadControllerConfig(directory.string()));
  std::filesystem::remove_all(directory);
}

TEST(Config, UsesUnitPreservingMeasuredControlFeedback) {
  const auto config = LoadControllerConfig(XX_MPPI_CONFIG_DIR);
  EXPECT_TRUE(config.use_measured_control_feedback);
  EXPECT_GT(config.maximum_model_sideslip_rad, 0.0F);
}

TEST(Config, LoadsConfiguredClosedRaceline) {
  const auto config = LoadControllerConfig(XX_MPPI_CONFIG_DIR);
  // The shipped model.yaml may name a CSV directly or select a map directory
  // through ${current_map}; either way the loaded map must be a usable loop.
  EXPECT_EQ(std::filesystem::path(config.raceline_path).extension(), ".csv");
  const auto raceline = Raceline::LoadCsv(config.raceline_path);

  EXPECT_TRUE(raceline.closed());
  EXPECT_GT(raceline.points().size(), 2U);
  EXPECT_GT(raceline.length(), 0.0F);
  EXPECT_NEAR(
    raceline.points().front().east_m, raceline.points().back().east_m, 1.0e-5F);
  EXPECT_NEAR(
    raceline.points().front().north_m, raceline.points().back().north_m, 1.0e-5F);
  for (const auto & point : raceline.points()) {
    EXPECT_LT(point.e_min_m, 0.0F);
    EXPECT_GT(point.e_max_m, 0.0F);
    EXPECT_LT(point.e_min_m, point.e_max_m);
  }
}

TEST(Config, LoadsRosSafetyAndDirectControlDefaults) {
  const auto config = LoadRosRuntimeConfig(XX_MPPI_CONFIG_DIR);
  EXPECT_FALSE(config.direct_control.topic.empty());
  EXPECT_NO_THROW(static_cast<void>(DirectControlModeName(config.direct_control.mode)));
  EXPECT_TRUE(std::isfinite(config.direct_control.torque_to_throttle_scale));
  EXPECT_TRUE(std::isfinite(config.direct_control.throttle_min));
  EXPECT_TRUE(std::isfinite(config.direct_control.throttle_max));
  EXPECT_LE(config.direct_control.throttle_min, config.direct_control.throttle_max);
}

TEST(RosMessage, PreservesTimestampAndTTAlignment) {
  PlannedTrajectory trajectory;
  trajectory.solution_pose_time_ns = 2'000'000'123LL;
  trajectory.dt_s = 0.1F;
  trajectory.states = {{1.0F, 2.0F}, {3.0F, 4.0F}};
  trajectory.controls = {Control{{0.1F, 1.5F}}, Control{{-0.2F, -2.0F}}};

  const auto message = ToRosMessage(trajectory, rclcpp::Time(1'234'567'890LL));
  EXPECT_EQ(message.current_time.sec, 1);
  EXPECT_EQ(message.current_time.nanosec, 234'567'890U);
  EXPECT_EQ(message.solution_pose_time.sec, 2);
  EXPECT_EQ(message.solution_pose_time.nanosec, 123U);
  EXPECT_EQ(message.horizon, 2U);
  ASSERT_EQ(message.states.size(), 2U);
  ASSERT_EQ(message.controls.size(), 2U);
  EXPECT_FLOAT_EQ(message.states[1].x_m, 3.0F);
  EXPECT_FLOAT_EQ(message.controls[1].steering_angle_rad, -0.2F);
  EXPECT_FLOAT_EQ(message.controls[1].torque_nm, -2.0F);
}

TEST(DirectControlMessage, UsesFirstControlAndPidLanekeepingTwistFields) {
  PlannedTrajectory trajectory;
  trajectory.controls = {Control{{0.25F, 0.4F}}, Control{{-0.1F, -0.2F}}};
  DirectControlConfig config;
  config.torque_to_throttle_scale = 2.0F;

  const auto message = ToDirectControlMessage(trajectory, config);
  EXPECT_DOUBLE_EQ(message.angular.z, 0.25);
  EXPECT_NEAR(message.linear.x, 0.8, 1.0e-6);
  EXPECT_DOUBLE_EQ(message.angular.x, 0.0);
  EXPECT_DOUBLE_EQ(message.linear.y, 0.0);
}

TEST(DirectControlMessage, ClampsConvertedThrottle) {
  PlannedTrajectory trajectory;
  trajectory.controls = {Control{{-0.3F, 0.75F}}};
  DirectControlConfig config;
  config.torque_to_throttle_scale = 4.0F;
  config.throttle_min = -0.5F;
  config.throttle_max = 0.6F;

  const auto message = ToDirectControlMessage(trajectory, config);
  EXPECT_NEAR(message.angular.z, -0.3, 1.0e-6);
  EXPECT_NEAR(message.linear.x, 0.6, 1.0e-6);
}

TEST(DirectControlMessage, TorqueModePassesWheelTorqueWithoutMappingOrClamp) {
  PlannedTrajectory trajectory;
  trajectory.controls = {Control{{0.1F, 2.75F}}};
  DirectControlConfig config;
  config.mode = DirectControlMode::kTorque;
  config.torque_to_throttle_scale = 100.0F;
  config.throttle_min = -0.1F;
  config.throttle_max = 0.1F;

  const auto message = ToDirectControlMessage(trajectory, config);
  EXPECT_NEAR(message.linear.x, 2.75, 1.0e-6);
}

TEST(DirectControlMessage, AppliesSteeringSignAndLimit) {
  PlannedTrajectory trajectory;
  trajectory.controls = {Control{{0.25F, 0.0F}}};
  DirectControlConfig inverted;
  // MPPI steering is positive-left; an actuator chain that is positive-right is
  // configured here rather than by flipping the vehicle model.
  inverted.steering_scale = -1.0F;
  EXPECT_NEAR(ToDirectControlMessage(trajectory, inverted).angular.z, -0.25, 1.0e-6);

  DirectControlConfig limited;
  limited.steering_scale = 4.0F;
  limited.steering_limit_rad = 0.4F;
  EXPECT_NEAR(ToDirectControlMessage(trajectory, limited).angular.z, 0.4, 1.0e-6);

  DirectControlConfig invalid;
  invalid.steering_scale = 0.0F;
  EXPECT_THROW(ValidateDirectControlConfig(invalid), std::invalid_argument);
}

TEST(DirectControlMessage, RejectsUnknownControlMode) {
  EXPECT_THROW(
    {
      const auto mode = ParseDirectControlMode("current");
      static_cast<void>(mode);
    },
    std::invalid_argument);
}

TEST(DirectControlMessage, RejectsEmptyTrajectoryAndInvalidConversion) {
  PlannedTrajectory trajectory;
  DirectControlConfig config;
  EXPECT_THROW(ToDirectControlMessage(trajectory, config), std::invalid_argument);

  config.throttle_min = 1.0F;
  config.throttle_max = -1.0F;
  EXPECT_THROW(ValidateDirectControlConfig(config), std::invalid_argument);
}

}  // namespace
}  // namespace xxcar::mppi
