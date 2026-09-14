#include "xx_mppi/ros/cost_terms_message.hpp"

namespace xxcar::mppi {

xxcar_msgs::msg::MppiCostTerms ToRosMessage(
  const CostTerms & terms, const std::int64_t solution_pose_time_ns,
  const rclcpp::Time & current_time)
{
  xxcar_msgs::msg::MppiCostTerms message;
  message.current_time = current_time;
  message.solution_pose_time = rclcpp::Time(solution_pose_time_ns);
  message.total = terms.total();
  message.reference_tracking = terms.values[kTermReferenceTracking];
  message.velocity_profile = terms.values[kTermVelocityProfile];
  message.boundary = terms.values[kTermBoundary];
  message.crash = terms.values[kTermCrash];
  message.sideslip = terms.values[kTermSideslip];
  message.sideslip_kill = terms.values[kTermSideslipKill];
  message.obstacle_distance = terms.values[kTermObstacleDistance];
  message.obstacle_latching = terms.values[kTermObstacleLatching];
  message.lateral_damping = terms.values[kTermLateralDamping];
  message.wheel_slip = terms.values[kTermWheelSlip];
  message.control_effort = terms.values[kTermControlEffort];
  message.control_smoothness = terms.values[kTermControlSmoothness];
  message.control_rate = terms.values[kTermControlRate];
  message.importance_sampling = terms.values[kTermImportanceSampling];
  message.longitudinal_acceleration = terms.values[kTermLongitudinalAcceleration];
  message.longitudinal_deceleration = terms.values[kTermLongitudinalDeceleration];
  message.progress = terms.values[kTermProgress];
  for (std::size_t i = 0; i < kStateDim; ++i) {
    message.reference_tracking_per_state[i] = terms.reference_tracking[i];
  }
  for (std::size_t i = 0; i < kControlDim; ++i) {
    message.control_effort_per_channel[i] = terms.control_effort[i];
    message.control_smoothness_per_channel[i] = terms.control_smoothness[i];
    message.control_rate_per_channel[i] = terms.control_rate[i];
  }
  message.first_crash_step = terms.first_crash_step;
  message.first_sideslip_step = terms.first_sideslip_step;
  message.first_obstacle_step = terms.first_obstacle_step;
  message.minimum_clearance = terms.minimum_clearance_m;
  return message;
}

}  // namespace xxcar::mppi
