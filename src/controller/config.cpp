#include "xx_mppi/controller/config.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>

#include <yaml-cpp/yaml.h>

namespace xxcar::mppi {
namespace {

template<typename T>
T GetOr(const YAML::Node & node, const char * key, const T & fallback) {
  return node && node[key] ? node[key].as<T>() : fallback;
}

std::filesystem::path Resolve(
  const std::filesystem::path & directory, const std::string & value)
{
  if (value.empty()) {
    return {};
  }
  const std::filesystem::path path(value);
  return path.is_absolute() ? path : directory / path;
}

void LoadVehicle(const YAML::Node & root, VehicleParameters & output) {
  const YAML::Node vehicle = root["vehicle"] ? root["vehicle"] : root;
  output.mass_kg = GetOr(vehicle, "mass_kg", output.mass_kg);
  output.yaw_inertia_kgm2 = GetOr(vehicle, "yaw_inertia_kgm2", output.yaw_inertia_kgm2);
  output.cg_to_front_m = GetOr(vehicle, "cg_to_front_m", output.cg_to_front_m);
  output.cg_to_rear_m = GetOr(vehicle, "cg_to_rear_m", output.cg_to_rear_m);
  output.front_cornering_stiffness_nprad = GetOr(
    vehicle, "front_cornering_stiffness_nprad", output.front_cornering_stiffness_nprad);
  output.rear_cornering_stiffness_nprad = GetOr(
    vehicle, "rear_cornering_stiffness_nprad", output.rear_cornering_stiffness_nprad);
  output.front_friction_coefficient = GetOr(
    vehicle, "front_friction_coefficient", output.front_friction_coefficient);
  output.rear_friction_coefficient = GetOr(
    vehicle, "rear_friction_coefficient", output.rear_friction_coefficient);
  output.wheel_radius_m = GetOr(vehicle, "wheel_radius_m", output.wheel_radius_m);
  output.driven_wheel_inertia_kgm2 = GetOr(
    vehicle, "driven_wheel_inertia_kgm2", output.driven_wheel_inertia_kgm2);
  output.front_brake_bias = GetOr(vehicle, "front_brake_bias", output.front_brake_bias);
}

void LoadNamedStateWeights(const YAML::Node & node, CostWeights & weights) {
  if (!node) {
    return;
  }
  weights.reference_tracking[kYawRate] = GetOr(node, "yaw_rate", 0.0F);
  weights.reference_tracking[kSpeed] = GetOr(node, "speed", 0.0F);
  weights.reference_tracking[kSideslip] = GetOr(node, "sideslip", 0.0F);
  weights.reference_tracking[kDrivenWheelSpeed] = GetOr(node, "driven_wheel_speed", 0.0F);
  weights.reference_tracking[kLateralDeviation] = GetOr(node, "lateral_deviation", 0.0F);
  weights.reference_tracking[kRelativeHeading] = GetOr(node, "relative_heading", 0.0F);
  weights.reference_tracking[kPathEvolution] = GetOr(node, "path_evolution", 0.0F);
}

}  // namespace

ControllerConfig LoadControllerConfig(const std::string & config_directory) {
  const std::filesystem::path directory(config_directory);
  if (!std::filesystem::is_directory(directory)) {
    throw std::runtime_error("config directory does not exist: " + config_directory);
  }
  const YAML::Node mppi_yaml = YAML::LoadFile((directory / "mppi.yaml").string());
  const YAML::Node model_yaml = YAML::LoadFile((directory / "model.yaml").string());
  const YAML::Node weights_yaml = YAML::LoadFile((directory / "weights.yaml").string());

  ControllerConfig config;
  config.mppi.num_samples = GetOr(mppi_yaml, "num_samples", config.mppi.num_samples);
  config.mppi.horizon = GetOr(mppi_yaml, "horizon", config.mppi.horizon);
  config.mppi.dt_s = GetOr(mppi_yaml, "dt", config.mppi.dt_s);
  config.mppi.integration_substeps = GetOr(
    mppi_yaml, "integration_substeps", config.mppi.integration_substeps);
  config.mppi.lambda = GetOr(mppi_yaml, "lambda", config.mppi.lambda);
  config.mppi.noise_smoothing_window = GetOr(
    mppi_yaml, "noise_smoothing_window", config.mppi.noise_smoothing_window);
  config.mppi.control_delay_steps = GetOr(
    mppi_yaml, "control_delay_steps", config.mppi.control_delay_steps);
  config.mppi.control_cost_gamma = GetOr(
    mppi_yaml, "control_cost_gamma", config.mppi.control_cost_gamma);
  config.mppi.special_samples = GetOr(
    mppi_yaml, "special_samples", config.mppi.special_samples);
  config.mppi.use_reference_controls = GetOr(
    mppi_yaml, "use_reference_controls", config.mppi.use_reference_controls);
  config.mppi.expected_trajectory = GetOr(
    mppi_yaml, "expected_trajectory", config.mppi.expected_trajectory);
  config.mppi.seed = GetOr(mppi_yaml, "seed", config.mppi.seed);
  config.solve_rate_hz = GetOr(mppi_yaml, "solve_rate_hz", config.solve_rate_hz);

  const auto sigma = mppi_yaml["sigma"];
  config.mppi.sigma[kSteering] = GetOr(sigma, "steering_angle_rad", config.mppi.sigma[kSteering]);
  config.mppi.sigma[kWheelTorque] = GetOr(sigma, "wheel_torque_nm", config.mppi.sigma[kWheelTorque]);
  const auto bounds = mppi_yaml["control_bounds"];
  if (bounds) {
    config.mppi.control_min[kSteering] = GetOr(
      bounds["min"], "steering_angle_rad", config.mppi.control_min[kSteering]);
    config.mppi.control_min[kWheelTorque] = GetOr(
      bounds["min"], "wheel_torque_nm", config.mppi.control_min[kWheelTorque]);
    config.mppi.control_max[kSteering] = GetOr(
      bounds["max"], "steering_angle_rad", config.mppi.control_max[kSteering]);
    config.mppi.control_max[kWheelTorque] = GetOr(
      bounds["max"], "wheel_torque_nm", config.mppi.control_max[kWheelTorque]);
  }
  const auto adaptation = mppi_yaml["adaptation"];
  config.mppi.adaptation.adaptive_lambda = GetOr(
    adaptation, "adaptive_lambda", config.mppi.adaptation.adaptive_lambda);
  config.mppi.adaptation.ess_fraction_min = GetOr(
    adaptation, "ess_fraction_min", config.mppi.adaptation.ess_fraction_min);
  config.mppi.adaptation.ess_fraction_max = GetOr(
    adaptation, "ess_fraction_max", config.mppi.adaptation.ess_fraction_max);
  config.mppi.adaptation.lambda_min = GetOr(
    adaptation, "lambda_min", config.mppi.adaptation.lambda_min);
  config.mppi.adaptation.lambda_max = GetOr(
    adaptation, "lambda_max", config.mppi.adaptation.lambda_max);
  config.mppi.adaptation.adaptive_sigma = GetOr(
    adaptation, "adaptive_sigma", config.mppi.adaptation.adaptive_sigma);
  config.mppi.adaptation.sigma_alpha = GetOr(
    adaptation, "sigma_alpha", config.mppi.adaptation.sigma_alpha);
  config.mppi.adaptation.sigma_scale_min = GetOr(
    adaptation, "sigma_scale_min", config.mppi.adaptation.sigma_scale_min);
  config.mppi.adaptation.sigma_scale_max = GetOr(
    adaptation, "sigma_scale_max", config.mppi.adaptation.sigma_scale_max);
  config.mppi.adaptation.speed_scaled_steering = GetOr(
    adaptation, "speed_scaled_steering", config.mppi.adaptation.speed_scaled_steering);
  config.mppi.adaptation.steering_reference_speed_mps = GetOr(
    adaptation, "steering_reference_speed_mps",
    config.mppi.adaptation.steering_reference_speed_mps);
  config.mppi.adaptation.steering_minimum_scale = GetOr(
    adaptation, "steering_minimum_scale", config.mppi.adaptation.steering_minimum_scale);

  const std::string integrator = GetOr(mppi_yaml, "integrator", std::string("euler"));
  if (integrator == "euler") {
    config.integrator = IntegratorKind::kEuler;
  } else if (integrator == "rk4") {
    config.integrator = IntegratorKind::kRungeKutta4;
  } else {
    throw std::runtime_error("unknown integrator: " + integrator);
  }

  const std::string model = GetOr(model_yaml, "name", std::string("dynamic_bicycle_fiala"));
  if (model == "kinematic_bicycle") {
    config.model_kind = ModelKind::kKinematicBicycle;
  } else if (model == "dynamic_bicycle_fiala") {
    config.model_kind = ModelKind::kDynamicBicycleFiala;
  } else if (model == "tensorrt_neural_derivative") {
    config.model_kind = ModelKind::kTensorRtNeuralDerivative;
  } else {
    throw std::runtime_error("unknown model name: " + model);
  }
  config.raceline_path = Resolve(
    directory, GetOr(model_yaml, "raceline_path", std::string{})).string();
  config.neural_model_path = Resolve(
    directory, GetOr(model_yaml, "neural_model_path", std::string{})).string();
  config.projection_window_m = GetOr(
    model_yaml, "projection_window_m", config.projection_window_m);
  if (model_yaml["vehicle_params_path"]) {
    LoadVehicle(YAML::LoadFile(Resolve(
      directory, model_yaml["vehicle_params_path"].as<std::string>()).string()), config.vehicle);
  } else {
    LoadVehicle(model_yaml, config.vehicle);
  }

  LoadNamedStateWeights(weights_yaml["reference_tracking"], config.costs);
  const auto velocity = weights_yaml["velocity_profile"];
  config.costs.velocity_profile = GetOr(velocity, "weight", config.costs.velocity_profile);
  config.costs.velocity_overspeed_multiplier = GetOr(
    velocity, "over_weight", config.costs.velocity_overspeed_multiplier);
  config.costs.progress = GetOr(weights_yaml["progress"], "weight", config.costs.progress);
  config.costs.boundary = GetOr(weights_yaml["boundary"], "weight", config.costs.boundary);
  config.costs.boundary_margin_m = GetOr(
    weights_yaml["boundary"], "margin", config.costs.boundary_margin_m);
  config.costs.crash = GetOr(weights_yaml["crash"], "weight", config.costs.crash);
  config.costs.crash_discount = GetOr(
    weights_yaml["crash"], "discount", config.costs.crash_discount);
  config.costs.crash_buffer_m = GetOr(
    weights_yaml["crash"], "buffer", config.costs.crash_buffer_m);
  config.costs.sideslip = GetOr(
    weights_yaml["sideslip_limit"], "weight", config.costs.sideslip);
  config.costs.maximum_sideslip_rad = GetOr(
    weights_yaml["sideslip_limit"], "max_slip", config.costs.maximum_sideslip_rad);
  config.costs.sideslip_kill = GetOr(
    weights_yaml["sideslip_limit"], "kill_weight", config.costs.sideslip_kill);
  config.costs.lateral_damping = GetOr(
    weights_yaml["lateral_damping"], "weight", config.costs.lateral_damping);
  config.costs.lateral_decay_rate = GetOr(
    weights_yaml["lateral_damping"], "decay_rate", config.costs.lateral_decay_rate);
  config.costs.wheel_slip = GetOr(
    weights_yaml["wheel_slip"], "weight", config.costs.wheel_slip);
  config.costs.wheel_slip_band = GetOr(
    weights_yaml["wheel_slip"], "band", config.costs.wheel_slip_band);
  const auto effort = weights_yaml["control_effort"];
  config.costs.control_effort[kSteering] = GetOr(
    effort, "steering_angle_rad", config.costs.control_effort[kSteering]);
  config.costs.control_effort[kWheelTorque] = GetOr(
    effort, "wheel_torque_nm", config.costs.control_effort[kWheelTorque]);
  const auto smoothness = weights_yaml["control_smoothness"];
  config.costs.control_smoothness[kSteering] = GetOr(
    smoothness, "steering_angle_rad", config.costs.control_smoothness[kSteering]);
  config.costs.control_smoothness[kWheelTorque] = GetOr(
    smoothness, "wheel_torque_nm", config.costs.control_smoothness[kWheelTorque]);

  if (config.mppi.num_samples <= 3U || config.mppi.horizon == 0U ||
    config.mppi.horizon == std::numeric_limits<std::uint16_t>::max() ||
    !(config.mppi.dt_s > 0.0F) || !std::isfinite(config.mppi.dt_s) ||
    !(config.mppi.lambda > 0.0F) || !std::isfinite(config.mppi.lambda) ||
    !(config.solve_rate_hz > 0.0F) || !std::isfinite(config.solve_rate_hz) ||
    !(config.projection_window_m > 0.0F) || !std::isfinite(config.projection_window_m))
  {
    throw std::runtime_error("invalid MPPI dimensions, timing, lambda, or projection window");
  }
  if (config.mppi.control_delay_steps >= config.mppi.horizon) {
    throw std::runtime_error("control_delay_steps must be less than horizon");
  }
  for (std::size_t i = 0; i < kControlDim; ++i) {
    if (!std::isfinite(config.mppi.control_min[i]) ||
      !std::isfinite(config.mppi.control_max[i]) ||
      !std::isfinite(config.mppi.sigma[i]) ||
      !(config.mppi.control_min[i] < config.mppi.control_max[i]) ||
      !(config.mppi.sigma[i] > 0.0F))
    {
      throw std::runtime_error("invalid control bounds or sampling sigma");
    }
  }
  if (!(config.costs.boundary_margin_m > 0.0F) ||
    !(config.costs.crash_discount > 0.0F && config.costs.crash_discount <= 1.0F) ||
    !(config.costs.maximum_sideslip_rad > 0.0F) ||
    !(config.costs.wheel_slip_band >= 0.0F) ||
    !(config.vehicle.mass_kg > 0.0F) || !(config.vehicle.yaw_inertia_kgm2 > 0.0F) ||
    !(config.vehicle.wheel_radius_m > 0.0F) ||
    !(config.vehicle.driven_wheel_inertia_kgm2 > 0.0F) ||
    !(config.vehicle.cg_to_front_m + config.vehicle.cg_to_rear_m > 0.0F))
  {
    throw std::runtime_error("invalid cost or vehicle physical parameters");
  }
  if (config.raceline_path.empty()) {
    throw std::runtime_error("model.yaml must define raceline_path");
  }
  return config;
}

}  // namespace xxcar::mppi
