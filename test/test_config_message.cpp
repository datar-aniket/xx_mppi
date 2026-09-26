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
  EXPECT_GT(config.mppi.obstacle_latch_brake_steps, 0U);
  EXPECT_LE(config.mppi.obstacle_latch_brake_steps, config.mppi.horizon);
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
  EXPECT_TRUE(config.mppi.refinement.enabled);
  EXPECT_GT(config.mppi.refinement.sqp_iterations, 0U);
  EXPECT_GT(config.mppi.refinement.pcg_iterations, 0U);
  EXPECT_GT(config.mppi.refinement.constraint_tolerance, 0.0F);
  EXPECT_GT(config.mppi.refinement.merit_constraint_penalty, 0.0F);
  EXPECT_GT(config.mppi.refinement.maximum_control_rate[kSteering], 0.0F);
  EXPECT_GT(config.mppi.refinement.maximum_control_rate[kWheelTorque], 0.0F);
  EXPECT_GT(config.info_log_rate_hz, 0.0F);
  EXPECT_GT(config.obstacles.confirmation_updates, 0U);
  EXPECT_GE(
    config.obstacles.persistence_updates,
    config.obstacles.confirmation_updates);
  EXPECT_GT(config.obstacles.association_distance_m, 0.0F);
}

TEST(Config, LoadsConfiguredVehicleProfile) {
  const auto config = LoadControllerConfig(XX_MPPI_CONFIG_DIR);
  EXPECT_GT(config.vehicle.mass_kg, 0.0F);
  EXPECT_GT(config.vehicle.yaw_inertia_kgm2, 0.0F);
  EXPECT_GT(config.vehicle.cg_to_front_m, 0.0F);
  EXPECT_GT(config.vehicle.cg_to_rear_m, 0.0F);
  EXPECT_GT(config.vehicle.front_cornering_stiffness_nprad, 0.0F);
  EXPECT_GT(config.vehicle.rear_cornering_stiffness_nprad, 0.0F);
  EXPECT_GT(config.vehicle.wheel_radius_m, 0.0F);
  EXPECT_EQ(config.vehicle.motor_pole_pairs, 2);
  EXPECT_GE(config.vehicle.front_brake_bias, 0.0F);
  EXPECT_LE(config.vehicle.front_brake_bias, 1.0F);
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

// A model.yaml naming one model, with every other path resolved inside the
// overlay. The shipped file points its raceline at an environment variable,
// which a test must not depend on.
std::string ModelYaml(const std::string & name) {
  return "name: " + name + "\n"
         "raceline_path: raceline.csv\n"
         "vehicle_params_path: vehicle.yaml\n"
         "neural_model_path: psi_dynamics.plan\n"
         "projection_window_m: 10.0\n";
}

}  // namespace

TEST(Config, LoadsExternalNestedVehicleProfileAndPhysicalSteeringBounds) {
  const auto directory = MakeOverlayConfig("xx_mppi_test_external_vehicle_config");
  const auto vehicle_path = directory / "external_vehicle.yaml";
  WriteFile(
    vehicle_path,
    "vehicle:\n"
    "  mass_kg: 3.0\n"
    "  yaw_inertia_kgm2: 0.03\n"
    "  geometry: {cg_to_front_m: 0.13, cg_to_rear_m: 0.14}\n"
    "  steering: {min_angle_rad: -0.45, max_angle_rad: 0.40, "
    "rear_min_angle_rad: -0.1, rear_max_angle_rad: 0.1}\n"
    "  motor: {pole_pairs: 2, min_current_a: -50.0, max_current_a: 50.0}\n"
    "  drivetrain: {motor_to_wheel_ratio: 2.5, "
    "driven_wheel_inertia_kgm2: 0.02, locked_awd: true, front_brake_bias: 0.5}\n"
    "  tires: {wheel_radius_m: 0.062, front_cornering_stiffness_nprad: 150.0, "
    "rear_cornering_stiffness_nprad: 150.0, front_friction_coefficient: 0.7, "
    "rear_friction_coefficient: 0.7}\n");
  const auto config = LoadControllerConfig(directory.string(), vehicle_path.string());
  std::filesystem::remove_all(directory);
  EXPECT_FLOAT_EQ(config.vehicle.mass_kg, 3.0F);
  EXPECT_EQ(config.vehicle.motor_pole_pairs, 2);
  EXPECT_FLOAT_EQ(config.vehicle.min_current_a, -50.0F);
  EXPECT_FLOAT_EQ(config.vehicle.max_current_a, 50.0F);
  EXPECT_FLOAT_EQ(config.vehicle.motor_to_wheel_ratio, 2.5F);
  EXPECT_FLOAT_EQ(config.mppi.control_min[kSteering], -0.45F);
  EXPECT_FLOAT_EQ(config.mppi.control_max[kSteering], 0.40F);
}

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

// The shipped config must stay front-steer-only: the MCU wire protocol carries
// no rear angle, so a solver planning rear-axle motion would be planning
// something the car cannot execute.
TEST(Config, ShippedConfigPinsRearSteeringToZero) {
  const auto config = LoadControllerConfig(XX_MPPI_CONFIG_DIR);
  EXPECT_FALSE(ModelSteersRearAxle(config.model_kind));
  EXPECT_FLOAT_EQ(config.mppi.sigma[kRearSteering], 0.0F);
  EXPECT_FLOAT_EQ(config.mppi.control_min[kRearSteering], 0.0F);
  EXPECT_FLOAT_EQ(config.mppi.control_max[kRearSteering], 0.0F);
}

// A front-steer-only model ignores the rear channel entirely, so sampling it
// would spend part of every solve on a control that cannot move the
// trajectory. The pin must win over whatever the YAML asks for.
TEST(Config, PinsRearSteeringWhenTheModelDoesNotSteerTheRearAxle) {
  const auto directory = MakeOverlayConfig("xx_mppi_test_rear_pin_config");
  const auto mppi_path = directory / "mppi.yaml";
  std::ifstream original(mppi_path);
  const std::string body(
    (std::istreambuf_iterator<char>(original)), std::istreambuf_iterator<char>());
  original.close();

  const std::string rear_active =
    "sigma: {steering_angle_rad: 0.15, wheel_torque_nm: 0.3, "
    "rear_steering_angle_rad: 0.1}\n"
    "control_bounds:\n"
    "  min: {steering_angle_rad: -0.4, wheel_torque_nm: -1.0, "
    "rear_steering_angle_rad: -0.2}\n"
    "  max: {steering_angle_rad: 0.4, wheel_torque_nm: 0.8, "
    "rear_steering_angle_rad: 0.2}\n";
  WriteFile(mppi_path, rear_active + body);

  WriteFile(
    directory / "model.yaml",
    ModelYaml("dynamic_bicycle_fiala"));
  const auto pinned = LoadControllerConfig(directory.string());
  EXPECT_FLOAT_EQ(pinned.mppi.sigma[kRearSteering], 0.0F);
  EXPECT_FLOAT_EQ(pinned.mppi.control_min[kRearSteering], 0.0F);
  EXPECT_FLOAT_EQ(pinned.mppi.control_max[kRearSteering], 0.0F);

  // The 4WS model keeps what the YAML asked for.
  WriteFile(
    directory / "model.yaml",
    ModelYaml("dynamic_bicycle_fiala_4ws"));
  const auto active = LoadControllerConfig(directory.string());
  EXPECT_TRUE(ModelSteersRearAxle(active.model_kind));
  EXPECT_FLOAT_EQ(active.mppi.sigma[kRearSteering], 0.1F);
  EXPECT_FLOAT_EQ(active.mppi.control_min[kRearSteering], -0.2F);
  EXPECT_FLOAT_EQ(active.mppi.control_max[kRearSteering], 0.2F);

  // The exported TensorRT engine takes a fixed six-wide input with no slot for
  // a rear angle, so it is front-steer-only and gets pinned like any other.
  WriteFile(
    directory / "model.yaml",
    ModelYaml("tensorrt_neural_derivative"));
  const auto neural = LoadControllerConfig(directory.string());
  EXPECT_FALSE(ModelSteersRearAxle(neural.model_kind));
  EXPECT_FLOAT_EQ(neural.mppi.sigma[kRearSteering], 0.0F);
  std::filesystem::remove_all(directory);
}

// Half-pinned states are configuration mistakes: a collapsed interval that is
// still being sampled, or a zero sigma with bounds left open.
TEST(Config, RejectsAHalfPinnedControlChannel) {
  const auto directory = MakeOverlayConfig("xx_mppi_test_half_pin_config");
  const auto mppi_path = directory / "mppi.yaml";
  std::ifstream original(mppi_path);
  const std::string body(
    (std::istreambuf_iterator<char>(original)), std::istreambuf_iterator<char>());
  original.close();
  WriteFile(
    directory / "model.yaml",
    ModelYaml("dynamic_bicycle_fiala_4ws"));

  // Sampled, but with a collapsed interval that clamps every draw to zero.
  WriteFile(
    mppi_path,
    "sigma: {steering_angle_rad: 0.15, wheel_torque_nm: 0.3, "
    "rear_steering_angle_rad: 0.1}\n"
    "control_bounds:\n"
    "  min: {steering_angle_rad: -0.4, wheel_torque_nm: -1.0, "
    "rear_steering_angle_rad: 0.0}\n"
    "  max: {steering_angle_rad: 0.4, wheel_torque_nm: 0.8, "
    "rear_steering_angle_rad: 0.0}\n" + body);
  EXPECT_THROW((void)LoadControllerConfig(directory.string()), std::runtime_error);

  // Never sampled, but with bounds left open as though it were active.
  WriteFile(
    mppi_path,
    "sigma: {steering_angle_rad: 0.15, wheel_torque_nm: 0.3, "
    "rear_steering_angle_rad: 0.0}\n"
    "control_bounds:\n"
    "  min: {steering_angle_rad: -0.4, wheel_torque_nm: -1.0, "
    "rear_steering_angle_rad: -0.2}\n"
    "  max: {steering_angle_rad: 0.4, wheel_torque_nm: 0.8, "
    "rear_steering_angle_rad: 0.2}\n" + body);
  EXPECT_THROW((void)LoadControllerConfig(directory.string()), std::runtime_error);
  std::filesystem::remove_all(directory);
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

TEST(Config, ValidatesNearTermObstacleBrakeWindow) {
  const auto directory = MakeOverlayConfig("xx_mppi_test_obstacle_brake_window");
  const auto mppi_path = directory / "mppi.yaml";
  std::ifstream original(mppi_path);
  const std::string body(
    (std::istreambuf_iterator<char>(original)), std::istreambuf_iterator<char>());
  original.close();

  WriteFile(mppi_path, "obstacle_latch_brake_steps: 0\n" + body);
  EXPECT_THROW((void)LoadControllerConfig(directory.string()), std::runtime_error);
  WriteFile(
    mppi_path, "horizon: 4\nobstacle_latch_brake_steps: 5\n" + body);
  EXPECT_THROW((void)LoadControllerConfig(directory.string()), std::runtime_error);
  WriteFile(
    mppi_path, "horizon: 5\nobstacle_latch_brake_steps: 5\n" + body);
  EXPECT_NO_THROW((void)LoadControllerConfig(directory.string()));
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

TEST(Config, LoadsCrashPaddingWithLegacyBufferFallback) {
  const auto directory = MakeOverlayConfig("xx_mppi_test_crash_padding_config");
  const auto weights_path = directory / "weights.yaml";

  WriteFile(weights_path, "crash:\n  buffer: 0.12\n");
  EXPECT_FLOAT_EQ(LoadControllerConfig(directory.string()).costs.crash_buffer_m, 0.12F);

  WriteFile(weights_path, "crash:\n  buffer: 0.12\n  padding: 0.13\n");
  EXPECT_FLOAT_EQ(LoadControllerConfig(directory.string()).costs.crash_buffer_m, 0.13F);

  WriteFile(weights_path, "crash:\n  padding: -0.01\n");
  EXPECT_THROW((void)LoadControllerConfig(directory.string()), std::runtime_error);
  std::filesystem::remove_all(directory);
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
  EXPECT_GT(config.direct_control.obstacle_brake_activation_s, 0.0F);
  EXPECT_GT(config.direct_control.obstacle_brake_recovery_s, 0.0F);
  EXPECT_GE(config.direct_control.obstacle_brake_stop_speed_mps, 0.0F);
  EXPECT_GT(config.direct_control.obstacle_brake_torque_nm, 0.0F);
  EXPECT_GE(config.direct_control.obstacle_brake_motor_rpm_release, 0.0F);
  EXPECT_GT(
    config.direct_control.obstacle_brake_motor_rpm_engage,
    config.direct_control.obstacle_brake_motor_rpm_release);
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
  config.four_wheel = true;
  config.torque_to_throttle_scale = 100.0F;
  config.throttle_min = -0.1F;
  config.throttle_max = 0.1F;

  const auto message = ToDirectControlMessage(trajectory, config, rclcpp::Time(0, 0));
  EXPECT_NEAR(message.throttle, 2.75F, 1.0e-6F);
  EXPECT_EQ(message.throttle_type, xxcar_msgs::msg::DirectControl::THROTTLE_TORQUE);
}

TEST(DirectControlMessage, ObstacleBrakeUsesSafetyTorqueWithoutModeChange) {
  PlannedTrajectory trajectory;
  trajectory.controls = {Control{{0.25F, 2.75F, -0.10F}}};
  DirectControlConfig config;
  config.mode = DirectControlMode::kTorque;
  config.four_wheel = true;
  config.steering_scale = 1.0F;
  config.steering_limit_rad = 1.0F;

  const auto message = ToSafetyTorqueMessage(
    trajectory, config, -1.6F, rclcpp::Time(2, 3));
  EXPECT_FLOAT_EQ(message.steering_angle_rad, 0.25F);
  EXPECT_FLOAT_EQ(message.rear_steering_angle_rad, -0.10F);
  EXPECT_FLOAT_EQ(message.throttle, -1.6F);
  EXPECT_EQ(message.throttle_type, xxcar_msgs::msg::DirectControl::THROTTLE_TORQUE);

  // The fallback and recovered MPPI command use the same torque mode.
  EXPECT_EQ(config.mode, DirectControlMode::kTorque);
  const auto recovered = ToDirectControlMessage(trajectory, config, rclcpp::Time(3, 4));
  EXPECT_FLOAT_EQ(recovered.throttle, 2.75F);
  EXPECT_EQ(recovered.throttle_type, xxcar_msgs::msg::DirectControl::THROTTLE_TORQUE);
}

TEST(DirectControlMessage, ObstacleBrakeRequiresTypedDirectControlTransport) {
  DirectControlConfig config;
  config.obstacle_brake_enabled = true;
  EXPECT_THROW(ValidateDirectControlConfig(config), std::invalid_argument);
  config.enabled = true;
  config.four_wheel = true;
  config.mode = DirectControlMode::kTorque;
  EXPECT_NO_THROW(ValidateDirectControlConfig(config));
  config.obstacle_brake_activation_s = 0.0F;
  EXPECT_THROW(ValidateDirectControlConfig(config), std::invalid_argument);
  config.obstacle_brake_activation_s = 0.5F;
  config.obstacle_brake_recovery_s = 0.0F;
  EXPECT_THROW(ValidateDirectControlConfig(config), std::invalid_argument);
  config.obstacle_brake_recovery_s = 0.5F;
  config.obstacle_brake_stop_speed_mps = -0.01F;
  EXPECT_THROW(ValidateDirectControlConfig(config), std::invalid_argument);
  config.obstacle_brake_stop_speed_mps = 0.05F;
  config.obstacle_brake_motor_rpm_release = 80.0F;
  config.obstacle_brake_motor_rpm_engage = 80.0F;
  EXPECT_THROW(ValidateDirectControlConfig(config), std::invalid_argument);
}

// A Twist cannot mark its throttle as a torque, so the driver would apply N m
// as amps. Torque mode is only valid on the DirectControl transport.
TEST(DirectControlMessage, TorqueModeRequiresDirectControlTransport) {
  DirectControlConfig config;
  config.mode = DirectControlMode::kTorque;
  config.four_wheel = false;
  EXPECT_THROW(ValidateDirectControlConfig(config), std::invalid_argument);
  config.four_wheel = true;
  EXPECT_NO_THROW(ValidateDirectControlConfig(config));
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

// The four-wheel transport sends both axles in radians so the driver can map
// each through its own servo calibration. A non-unit scale would push the front
// servo's sign and gain onto the rear, which may be mirrored.
TEST(DirectControlMessage, FourWheelTransportCarriesRadiansForBothAxles) {
  PlannedTrajectory trajectory;
  trajectory.controls = {Control{{0.25F, 0.0F, -0.1F}}};
  DirectControlConfig config;
  config.four_wheel = true;
  const auto message = ToDirectControlMessage(trajectory, config, rclcpp::Time(0, 0));
  EXPECT_NEAR(message.steering_angle_rad, 0.25F, 1.0e-6F);
  EXPECT_NEAR(message.rear_steering_angle_rad, -0.1F, 1.0e-6F);

  config.steering_scale = -4.5F;
  EXPECT_THROW(ValidateDirectControlConfig(config), std::invalid_argument);
  config.four_wheel = false;
  EXPECT_NO_THROW(ValidateDirectControlConfig(config));
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
