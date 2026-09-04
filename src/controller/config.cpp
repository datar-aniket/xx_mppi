#include "xx_mppi/controller/config.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
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

std::string ExpandEnvironmentVariables(const std::string & value) {
  std::string result;
  result.reserve(value.size());
  for (std::size_t i = 0U; i < value.size();) {
    if (value[i] != '$') {
      result.push_back(value[i++]);
      continue;
    }

    std::size_t name_begin = i + 1U;
    std::size_t name_end = name_begin;
    if (name_begin < value.size() && value[name_begin] == '{') {
      name_begin++;
      name_end = value.find('}', name_begin);
      if (name_end == std::string::npos) {
        throw std::runtime_error("unterminated environment variable in path: " + value);
      }
      i = name_end + 1U;
    } else {
      while (name_end < value.size() &&
        (std::isalnum(static_cast<unsigned char>(value[name_end])) != 0 ||
        value[name_end] == '_'))
      {
        ++name_end;
      }
      i = name_end;
    }

    if (name_begin == name_end) {
      throw std::runtime_error("empty environment variable in path: " + value);
    }
    const std::string name = value.substr(name_begin, name_end - name_begin);
    const char * expanded = std::getenv(name.c_str());
    if (expanded == nullptr || expanded[0] == '\0') {
      throw std::runtime_error(
        "environment variable '" + name + "' referenced by path is not set");
    }
    result.append(expanded);
  }
  return result;
}

std::string ExpandHomeDirectory(const std::string & value) {
  if (value != "~" && value.rfind("~/", 0U) != 0U) {
    return value;
  }
  const char * home = std::getenv("HOME");
  if (home == nullptr || home[0] == '\0') {
    throw std::runtime_error("HOME is not set; cannot expand path: " + value);
  }
  return value == "~" ? std::string(home) : std::string(home) + value.substr(1U);
}

std::filesystem::path Resolve(
  const std::filesystem::path & directory, const std::string & value)
{
  if (value.empty()) {
    return {};
  }
  const std::filesystem::path path(
    ExpandHomeDirectory(ExpandEnvironmentVariables(value)));
  return path.is_absolute() ? path : directory / path;
}

std::filesystem::path ResolveRaceline(
  const std::filesystem::path & config_directory, const std::string & value)
{
  auto path = Resolve(config_directory, value).lexically_normal();
  if (path.empty()) {
    return {};
  }
  if (std::filesystem::is_directory(path)) {
    const auto map_name = path.filename().string();
    if (map_name.empty()) {
      throw std::runtime_error("raceline map directory has no folder name: " + path.string());
    }
    path /= map_name + "_frenet_map.csv";
  } else if (!std::filesystem::exists(path) && path.extension() != ".csv") {
    throw std::runtime_error("raceline map directory does not exist: " + path.string());
  }
  if (!std::filesystem::is_regular_file(path)) {
    throw std::runtime_error("raceline CSV does not exist: " + path.string());
  }
  return path;
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
  output.locked_awd = GetOr(vehicle, "locked_awd", output.locked_awd);
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
  config.control_publish_rate_hz = GetOr(
    mppi_yaml, "control_publish_rate_hz", config.control_publish_rate_hz);
  config.maximum_solution_age_s = GetOr(
    mppi_yaml, "maximum_solution_age_s", config.maximum_solution_age_s);
  config.info_publish_rate_hz = GetOr(
    mppi_yaml, "info_publish_rate_hz", config.info_publish_rate_hz);
  config.info_topic = GetOr(mppi_yaml, "info_topic", config.info_topic);
  config.visualization_rate_hz = GetOr(
    mppi_yaml, "visualization_rate_hz", config.visualization_rate_hz);
  config.num_rollouts = GetOr(mppi_yaml, "num_rollouts", config.num_rollouts);

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

  const std::string frame = GetOr(mppi_yaml, "frame", std::string("frenet"));
  if (frame == "frenet") {
    config.mppi.frame = FrameKind::kFrenet;
  } else if (frame == "cartesian") {
    config.mppi.frame = FrameKind::kCartesian;
  } else {
    throw std::runtime_error("unknown frame: " + frame + "; expected frenet or cartesian");
  }

  const auto observation = mppi_yaml["observation"];
  config.use_measured_control_feedback = GetOr(
    observation, "use_measured_control_feedback", config.use_measured_control_feedback);
  config.maximum_model_sideslip_rad = GetOr(
    observation, "maximum_model_sideslip_rad", config.maximum_model_sideslip_rad);

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
  config.raceline_path = ResolveRaceline(
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
  const auto acceleration = weights_yaml["longitudinal_acceleration"];
  config.costs.longitudinal_acceleration = GetOr(
    acceleration, "acceleration_weight", config.costs.longitudinal_acceleration);
  config.costs.longitudinal_deceleration = GetOr(
    acceleration, "deceleration_weight", config.costs.longitudinal_deceleration);
  const auto control_rate = weights_yaml["control_rate"];
  config.costs.control_rate[kSteering] = GetOr(
    control_rate, "steering_velocity_radps", config.costs.control_rate[kSteering]);
  config.costs.control_rate[kWheelTorque] = GetOr(
    control_rate, "wheel_torque_rate_nmps", config.costs.control_rate[kWheelTorque]);

  if (config.mppi.num_samples <= 3U || config.mppi.horizon == 0U ||
    config.mppi.horizon == std::numeric_limits<std::uint16_t>::max() ||
    !(config.mppi.dt_s > 0.0F) || !std::isfinite(config.mppi.dt_s) ||
    !(config.mppi.lambda > 0.0F) || !std::isfinite(config.mppi.lambda) ||
    !(config.solve_rate_hz > 0.0F) || !std::isfinite(config.solve_rate_hz) ||
    !(config.control_publish_rate_hz > 0.0F) ||
    !std::isfinite(config.control_publish_rate_hz) ||
    !(config.maximum_solution_age_s >= 0.0F) ||
    !std::isfinite(config.maximum_solution_age_s) ||
    !(config.info_publish_rate_hz > 0.0F) ||
    !std::isfinite(config.info_publish_rate_hz) || config.info_topic.empty() ||
    !(config.visualization_rate_hz > 0.0F) ||
    !std::isfinite(config.visualization_rate_hz) ||
    config.num_rollouts == 0U || config.num_rollouts > config.mppi.num_samples ||
    !(config.projection_window_m > 0.0F) || !std::isfinite(config.projection_window_m) ||
    !(config.maximum_model_sideslip_rad > 0.0F) ||
    !std::isfinite(config.maximum_model_sideslip_rad))
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
      !(config.mppi.sigma[i] > 0.0F) ||
      !std::isfinite(config.costs.control_rate[i]) || config.costs.control_rate[i] < 0.0F)
    {
      throw std::runtime_error("invalid control bounds or sampling sigma");
    }
  }
  if (!(config.costs.boundary_margin_m > 0.0F) ||
    !(config.costs.crash_discount > 0.0F && config.costs.crash_discount <= 1.0F) ||
    !(config.costs.maximum_sideslip_rad > 0.0F) ||
    !(config.costs.wheel_slip_band >= 0.0F) ||
    !std::isfinite(config.costs.longitudinal_acceleration) ||
    !std::isfinite(config.costs.longitudinal_deceleration) ||
    config.costs.longitudinal_acceleration < 0.0F ||
    config.costs.longitudinal_deceleration < 0.0F ||
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

  // The driven-wheel state is the stiffest mode in the model: the tire force
  // reacts to slip almost instantly, so the wheel-speed loop has eigenvalue
  // -r^2 C / I. Integrating it above the explicit stability limit makes the
  // wheel speed ring, which drags the body speed negative and produces rollouts
  // that reverse before moving off. Half the marginal limit is the usable step.
  const float stiffest_stiffness = std::max(
    config.vehicle.front_cornering_stiffness_nprad,
    config.vehicle.rear_cornering_stiffness_nprad);
  const float marginal_step_s = 2.0F * config.vehicle.driven_wheel_inertia_kgm2 /
    std::max(
      config.vehicle.wheel_radius_m * config.vehicle.wheel_radius_m *
      stiffest_stiffness, 1.0e-9F);
  const auto substeps = static_cast<float>(
    config.mppi.integration_substeps == 0U ? 1U : config.mppi.integration_substeps);
  const float step_s = config.mppi.dt_s / substeps;
  if (!(step_s < 0.5F * marginal_step_s)) {
    const auto required = static_cast<unsigned>(std::ceil(
        config.mppi.dt_s / (0.5F * marginal_step_s)));
    throw std::runtime_error(
            "integration step " + std::to_string(step_s) +
            " s exceeds the driven-wheel stability limit " +
            std::to_string(0.5F * marginal_step_s) +
            " s for this vehicle profile; raise integration_substeps to at least " +
            std::to_string(required) + ", lower dt, or re-identify "
            "driven_wheel_inertia_kgm2 / cornering stiffness");
  }
  return config;
}

}  // namespace xxcar::mppi
