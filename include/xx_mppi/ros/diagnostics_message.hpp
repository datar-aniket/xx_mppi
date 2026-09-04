#pragma once

#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <rclcpp/time.hpp>

#include "xx_mppi/controller/mppi_controller.hpp"

namespace xxcar::mppi {

// Builds a named, human-readable snapshot of the solution that was most
// recently published to the controller output.
[[nodiscard]] diagnostic_msgs::msg::DiagnosticArray ToDiagnosticsMessage(
  const PlannedTrajectory & trajectory, const rclcpp::Time & publication_time);

}  // namespace xxcar::mppi
