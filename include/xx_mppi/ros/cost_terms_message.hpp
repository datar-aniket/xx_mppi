#pragma once

#include <string>

#include <rclcpp/time.hpp>
#include <xxcar_msgs/msg/mppi_cost_terms.hpp>

#include "xx_mppi/controller/mppi_controller.hpp"

namespace xxcar::mppi {

struct CostTermsConfig {
  bool enabled{false};
  std::string topic{"xx_mppi/cost_terms"};
};

xxcar_msgs::msg::MppiCostTerms ToRosMessage(
  const CostTerms & terms, std::int64_t solution_pose_time_ns,
  const rclcpp::Time & current_time);

}  // namespace xxcar::mppi
