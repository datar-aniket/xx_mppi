#pragma once

#include <cstdint>
#include <string>

#include "xx_mppi/dynamics/integrator.hpp"
#include "xx_mppi/dynamics/model.hpp"
#include "xx_mppi/types.hpp"

namespace xxcar::mppi {

// Operator no-overtake mode (RC trigger high while in AUTO). Two parts:
//  - a hard wall pass_limit_gap_m behind the nearest return on the racing
//    surface ahead. Any rollout state past it latches the crash cost, the same
//    as leaving the track, so no admissible plan passes the lead car. The lead
//    is treated as stationary over the horizon: conservative, never optimistic.
//  - the raceline speed is scaled and capped so the reference stops
//    gap_minimum_m short of that return, which sets the following distance:
//    gap = gap_minimum_m + v^2 / (2 deceleration_mps2).
struct NoOvertakeConfig {
  float speed_scale{0.7F};
  // Along-track distance from the vehicle centre (base_link) to the lead
  // car's nearest return at which the reference speed reaches zero. Includes
  // this car's front overhang.
  float gap_minimum_m{0.8F};
  // Hard wall distance from base_link to the lead car's nearest return. At
  // least this car's front overhang; keep it below gap_minimum_m.
  float pass_limit_gap_m{0.3F};
  float deceleration_mps2{3.0F};
  // The wall exists only for a car detected this far ahead, so it must exceed
  // the stopping distance from the fastest no-overtake speed.
  float lookahead_m{8.0F};
  // Returns closer than this to either track bound are walls, not cars.
  float wall_margin_m{0.15F};
  // Replaces velocity_profile.over_weight while the mode is on, so exceeding
  // the lowered reference is not nearly free.
  float overspeed_multiplier{1.0F};
};

struct ControllerConfig {
  MppiConfig mppi{};
  CostWeights costs{};
  ObstacleConfig obstacles{};
  VehicleParameters vehicle{};
  ModelKind model_kind{ModelKind::kDynamicBicycleFiala};
  IntegratorKind integrator{IntegratorKind::kEuler};
  std::string raceline_path;
  std::string neural_model_path;
  float projection_window_m{30.0F};
  // When false the warm start uses the controller's last published command
  // instead of the measured steering [rad] and wheel torque [N m] feedback.
  bool use_measured_control_feedback{false};
  // Sideslip magnitude the body model is allowed to see. The projector still
  // uses the full measured course heading, so only the vehicle model and the
  // sideslip costs are protected from an implausible estimate.
  float maximum_model_sideslip_rad{0.8F};
  float solve_rate_hz{100.0F};
  float control_publish_rate_hz{50.0F};
  // Do not emit a command solved from an observation older than this. Zero
  // disables this publication-time guard (useful for deterministic bag replay).
  float maximum_solution_age_s{0.1F};
  float info_log_rate_hz{10.0F};
  float visualization_rate_hz{10.0F};
  // Rate at which the solver is asked to decompose the published trajectory's
  // cost into its individual terms. Debug output; keep it well below
  // solve_rate_hz.
  float cost_terms_rate_hz{10.0F};
  std::uint32_t num_rollouts{15U};
  NoOvertakeConfig no_overtake{};
};

ControllerConfig LoadControllerConfig(const std::string & config_directory);

}  // namespace xxcar::mppi
