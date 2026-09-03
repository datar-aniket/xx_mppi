#include <cmath>
#include <limits>
#include <optional>
#include <string>

#include <gtest/gtest.h>

#include "xx_mppi/controller/mppi_controller.hpp"
#include "xx_mppi/math.hpp"
#include "xx_mppi/ros/ekf_state_adapter.hpp"

namespace xxcar::mppi {
namespace {

using EkfState = ekf_mcu_driver::msg::EkfState;

EkfState ValidMessage() {
  EkfState message;
  message.header.stamp.sec = 10;
  message.header.stamp.nanosec = 20U;
  message.pose.position.x = 1.25;
  message.pose.position.y = -2.5;
  const double yaw = 0.75;
  message.pose.orientation.z = std::sin(0.5 * yaw);
  message.pose.orientation.w = std::cos(0.5 * yaw);
  message.twist.linear.x = 3.0;
  message.twist.linear.y = 4.0;
  message.angular_velocity.z = 0.2;
  message.linear_acceleration.x = 0.3;
  message.side_slip_rad = 0.1F;
  message.wheel_torque_nm = 2.0F;
  message.steering_angle = 0.4F;
  message.motor_speed_ms = 5.5F;
  message.solution_status = static_cast<std::uint8_t>(
    EkfState::SOLUTION_STATUS_ATTITUDE_VALID |
    EkfState::SOLUTION_STATUS_YAW_ABSOLUTE |
    EkfState::SOLUTION_STATUS_VEL_HORIZ |
    EkfState::SOLUTION_STATUS_POS_HORIZ);
  message.source_valid = static_cast<std::uint8_t>(
    EkfState::SOURCE_VALID_ESTIMATOR |
    EkfState::SOURCE_VALID_GYRO |
    EkfState::SOURCE_VALID_VESC);
  return message;
}

TEST(EkfStateAdapter, ConvertsFramesUnitsTimestampAndCalibration) {
  auto message = ValidMessage();
  EkfStateAdapterConfig config;
  config.steering_scale_to_rad = 0.5F;
  config.steering_offset_rad = -0.1F;
  config.torque_scale_to_nm = 2.0F;
  config.motor_speed_scale_to_mps = 0.25F;
  const auto observation = ToVehicleObservation(message, config);

  EXPECT_EQ(observation.pose_time_ns, 10'000'000'020LL);
  EXPECT_FLOAT_EQ(observation.east_m, 1.25F);
  EXPECT_FLOAT_EQ(observation.north_m, -2.5F);
  EXPECT_NEAR(observation.yaw_enu_rad, 0.75F, 1.0e-6F);
  EXPECT_FLOAT_EQ(observation.speed_mps, 5.0F);
  EXPECT_FLOAT_EQ(observation.yaw_rate_radps, 0.2F);
  EXPECT_FLOAT_EQ(observation.longitudinal_acceleration_mps2, 0.3F);
  EXPECT_FLOAT_EQ(observation.sideslip_rad, 0.1F);
  EXPECT_FLOAT_EQ(observation.measured_torque_nm, 4.0F);
  EXPECT_FLOAT_EQ(observation.measured_steering_rad, 0.1F);
  EXPECT_FLOAT_EQ(observation.driven_wheel_speed_mps, 1.375F);
}

TEST(EkfStateAdapter, RejectsMissingPositionValidity) {
  auto message = ValidMessage();
  message.solution_status = static_cast<std::uint8_t>(
    message.solution_status & ~EkfState::SOLUTION_STATUS_POS_HORIZ);
  EXPECT_THROW((void)ToVehicleObservation(message), std::invalid_argument);
}

TEST(EkfStateAdapter, AllowsMissingSolutionBitsWhenValidityCheckIsDisabled) {
  auto message = ValidMessage();
  message.solution_status = 0U;
  EkfStateAdapterConfig config;
  config.require_solution_validity = false;

  EXPECT_NO_THROW((void)ToVehicleObservation(message, config));
}

TEST(EkfStateAdapter, StillRequiresSourcesWhenSolutionValidityIsDisabled) {
  auto message = ValidMessage();
  message.solution_status = 0U;
  message.source_valid = 0U;
  EkfStateAdapterConfig config;
  config.require_solution_validity = false;

  EXPECT_THROW((void)ToVehicleObservation(message, config), std::invalid_argument);
}

TEST(EkfStateAdapter, RejectsMissingVescByDefault) {
  auto message = ValidMessage();
  message.source_valid = static_cast<std::uint8_t>(
    message.source_valid & ~EkfState::SOURCE_VALID_VESC);
  EXPECT_THROW((void)ToVehicleObservation(message), std::invalid_argument);

  EkfStateAdapterConfig config;
  config.require_vesc = false;
  EXPECT_NO_THROW((void)ToVehicleObservation(message, config));
}

TEST(EkfStateAdapter, RejectsZeroQuaternion) {
  auto message = ValidMessage();
  message.pose.orientation.x = 0.0;
  message.pose.orientation.y = 0.0;
  message.pose.orientation.z = 0.0;
  message.pose.orientation.w = 0.0;
  EXPECT_THROW((void)ToVehicleObservation(message), std::invalid_argument);
}

TEST(EkfStateAdapter, AllowsNanSideslipForCoreLowSpeedHandling) {
  auto message = ValidMessage();
  message.twist.linear.x = 0.1;
  message.twist.linear.y = 0.0;
  message.side_slip_rad = std::numeric_limits<float>::quiet_NaN();
  const auto observation = ToVehicleObservation(message);
  EXPECT_TRUE(std::isnan(observation.sideslip_rad));
}

TEST(EkfStateTiming, EnforcesFreshMonotonicPoseTime) {
  EXPECT_NO_THROW(ValidateObservationTime(
      9'950'000'000LL, 10'000'000'000LL, std::nullopt, 0.10, 0.02));
  EXPECT_THROW(ValidateObservationTime(
      9'800'000'000LL, 10'000'000'000LL, std::nullopt, 0.10, 0.02),
    std::invalid_argument);
  EXPECT_THROW(ValidateObservationTime(
      10'030'000'000LL, 10'000'000'000LL, std::nullopt, 0.10, 0.02),
    std::invalid_argument);
  EXPECT_THROW(ValidateObservationTime(
      9'950'000'000LL, 10'000'000'000LL, 9'950'000'000LL, 0.10, 0.02),
    std::invalid_argument);
}

TEST(EkfStatePipeline, ProjectsConvertedEnuPoseIntoFrenetFrame) {
  auto message = ValidMessage();
  message.pose.position.x = 0.0;
  message.pose.position.y = 0.0;
  message.pose.orientation.x = 0.0;
  message.pose.orientation.y = 0.0;
  message.pose.orientation.z = std::sin(0.25 * static_cast<double>(kPi));
  message.pose.orientation.w = std::cos(0.25 * static_cast<double>(kPi));
  message.twist.linear.x = 1.0;
  message.twist.linear.y = 0.0;
  message.side_slip_rad = 0.0F;
  message.motor_speed_ms = 1.0F;

  ControllerConfig controller_config;
  const auto raceline = Raceline::LoadCsv(
    std::string(XX_MPPI_TEST_DATA_DIR) + "/test_raceline.csv");
  MppiController controller(controller_config, raceline);
  const auto projection = controller.UpdateObservation(ToVehicleObservation(message));

  ASSERT_TRUE(projection.valid);
  EXPECT_NEAR(projection.s_m, 0.0F, 1.0e-5F);
  EXPECT_NEAR(projection.e_m, 0.0F, 1.0e-5F);
  EXPECT_NEAR(projection.relative_course_rad, 0.0F, 1.0e-5F);
}

}  // namespace
}  // namespace xxcar::mppi
