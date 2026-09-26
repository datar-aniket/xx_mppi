#pragma once

#include <string>

#include <xxcar_msgs/msg/grip_status.hpp>

#include "xx_mppi/controller/grip_monitor.hpp"
#include "xx_mppi/controller/mppi_controller.hpp"
#include "xx_mppi/dynamics/model.hpp"

namespace xxcar::mppi {

struct GripStatusConfig {
  bool enabled{true};
  std::string topic{"xx_mppi/grip"};
  double rate_hz{25.0};
};

// Measured grip from the observation, predicted grip from the published plan
// (its first control is the command actually sent) and the monitor's observed
// check. Updates monitor, so call it once per published message.
xxcar_msgs::msg::GripStatus ToGripStatusMessage(
  const VehicleObservation & observation, const PlannedTrajectory & trajectory,
  const VehicleParameters & vehicle, ModelKind model_kind, GripMonitor & monitor);

}  // namespace xxcar::mppi
