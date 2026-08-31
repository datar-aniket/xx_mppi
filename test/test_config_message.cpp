#include <string>

#include <gtest/gtest.h>
#include <rclcpp/time.hpp>

#include "xx_mppi/controller/config.hpp"
#include "xx_mppi/ros/trajectory_message.hpp"

namespace xxcar::mppi {
namespace {

TEST(Config, LoadsOrinProblemAndPhysicalControlBounds) {
  const auto config = LoadControllerConfig(XX_MPPI_CONFIG_DIR);
  EXPECT_EQ(config.mppi.num_samples, 2001U);
  EXPECT_EQ(config.mppi.horizon, 50U);
  EXPECT_FLOAT_EQ(config.mppi.dt_s, 0.1F);
  EXPECT_FLOAT_EQ(config.mppi.control_min[kSteering], -0.5F);
  EXPECT_FLOAT_EQ(config.mppi.control_max[kSteering], 0.5F);
  EXPECT_FLOAT_EQ(config.mppi.control_min[kWheelTorque], -5.0F);
  EXPECT_FLOAT_EQ(config.mppi.control_max[kWheelTorque], 5.0F);
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

}  // namespace
}  // namespace xxcar::mppi
