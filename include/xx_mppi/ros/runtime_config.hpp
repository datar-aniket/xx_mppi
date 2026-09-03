#pragma once

#include <string>

#include "xx_mppi/ros/direct_control_message.hpp"

namespace xxcar::mppi {

struct RosRuntimeConfig {
  bool require_solution_validity{true};
  DirectControlConfig direct_control{};
};

[[nodiscard]] RosRuntimeConfig LoadRosRuntimeConfig(
  const std::string & config_directory);

}  // namespace xxcar::mppi
