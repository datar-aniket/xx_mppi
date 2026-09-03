#include "xx_mppi/ros/runtime_config.hpp"

#include <filesystem>
#include <stdexcept>
#include <string>

#include <yaml-cpp/yaml.h>

namespace xxcar::mppi {
namespace {

template<typename T>
T GetOr(const YAML::Node & node, const char * key, const T & fallback) {
  return node && node[key] ? node[key].as<T>() : fallback;
}

}  // namespace

RosRuntimeConfig LoadRosRuntimeConfig(const std::string & config_directory) {
  const std::filesystem::path directory(config_directory);
  if (!std::filesystem::is_directory(directory)) {
    throw std::runtime_error("config directory does not exist: " + config_directory);
  }

  const YAML::Node yaml = YAML::LoadFile((directory / "mppi.yaml").string());
  RosRuntimeConfig config;
  config.require_solution_validity = GetOr(
    yaml, "require_solution_validity", config.require_solution_validity);
  config.direct_control.enabled = GetOr(
    yaml, "publish_direct_control", config.direct_control.enabled);
  config.direct_control.topic = GetOr(
    yaml, "direct_control_topic", config.direct_control.topic);
  const auto default_mode = std::string(DirectControlModeName(config.direct_control.mode));
  config.direct_control.mode = ParseDirectControlMode(
    GetOr(yaml, "control_mode", default_mode));
  config.direct_control.torque_to_throttle_scale = GetOr(
    yaml, "direct_control_torque_to_throttle_scale",
    config.direct_control.torque_to_throttle_scale);
  config.direct_control.throttle_min = GetOr(
    yaml, "direct_control_throttle_min", config.direct_control.throttle_min);
  config.direct_control.throttle_max = GetOr(
    yaml, "direct_control_throttle_max", config.direct_control.throttle_max);
  config.direct_control.steering_scale = GetOr(
    yaml, "direct_control_steering_scale", config.direct_control.steering_scale);
  config.direct_control.steering_limit_rad = GetOr(
    yaml, "direct_control_steering_limit_rad", config.direct_control.steering_limit_rad);
  ValidateDirectControlConfig(config.direct_control);
  return config;
}

}  // namespace xxcar::mppi
