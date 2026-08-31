#pragma once

#include <rclcpp/time.hpp>
#include <xxcar_msgs/msg/vehicle_control_trajectory.hpp>

#include "xx_mppi/controller/mppi_controller.hpp"

namespace xxcar::mppi {

xxcar_msgs::msg::VehicleControlTrajectory ToRosMessage(
  const PlannedTrajectory & trajectory, const rclcpp::Time & current_time);

}  // namespace xxcar::mppi
