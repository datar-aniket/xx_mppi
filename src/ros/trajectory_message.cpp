#include "xx_mppi/ros/trajectory_message.hpp"

#include <limits>
#include <stdexcept>

namespace xxcar::mppi {

xxcar_msgs::msg::VehicleControlTrajectory ToRosMessage(
  const PlannedTrajectory & trajectory, const rclcpp::Time & current_time)
{
  if (trajectory.states.size() != trajectory.controls.size() ||
    trajectory.states.size() > std::numeric_limits<std::uint16_t>::max())
  {
    throw std::invalid_argument("planned trajectory arrays must be aligned and fit uint16 horizon");
  }
  xxcar_msgs::msg::VehicleControlTrajectory message;
  message.current_time = current_time;
  message.solution_pose_time = rclcpp::Time(trajectory.solution_pose_time_ns);
  message.dt = trajectory.dt_s;
  message.horizon = static_cast<std::uint16_t>(trajectory.states.size());
  message.states.resize(trajectory.states.size());
  message.controls.resize(trajectory.controls.size());
  for (std::size_t i = 0; i < trajectory.states.size(); ++i) {
    message.states[i].x_m = trajectory.states[i].east_m;
    message.states[i].y_m = trajectory.states[i].north_m;
    message.controls[i].steering_angle_rad = trajectory.controls[i][kSteering];
    message.controls[i].torque_nm = trajectory.controls[i][kWheelTorque];
  }
  return message;
}

}  // namespace xxcar::mppi
