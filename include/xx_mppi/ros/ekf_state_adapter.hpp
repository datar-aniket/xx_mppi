#pragma once

#include <cstdint>
#include <optional>

#include <xxcar_msgs/msg/ekf_state.hpp>

#include "xx_mppi/controller/mppi_controller.hpp"

namespace xxcar::mppi {

struct EkfStateAdapterConfig {
  bool require_solution_validity{true};
  bool require_absolute_yaw{true};
  bool require_vesc{true};
};

void ValidateEkfStateAdapterConfig(const EkfStateAdapterConfig & config);

// Converts the MCU state contract into the ROS-independent controller input.
// Throws std::invalid_argument when a required validity bit or value is bad.
[[nodiscard]] VehicleObservation ToVehicleObservation(
  const xxcar_msgs::msg::EkfState & message,
  const EkfStateAdapterConfig & config = {});

// Validates freshness and strict timestamp ordering. A zero maximum age turns
// off the past-age check for bag replay, while future and ordering checks stay
// active.
void ValidateObservationTime(
  std::int64_t pose_time_ns, std::int64_t now_ns,
  std::optional<std::int64_t> previous_pose_time_ns,
  double maximum_age_s, double future_tolerance_s);

}  // namespace xxcar::mppi
