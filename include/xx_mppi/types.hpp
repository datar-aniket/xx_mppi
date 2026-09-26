#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace xxcar::mppi {

constexpr std::size_t kBodyStateDim = 4;
constexpr std::size_t kFrameStateDim = 3;
constexpr std::size_t kStateDim = kBodyStateDim + kFrameStateDim;
constexpr std::size_t kControlDim = 3;
constexpr std::size_t kMpcLineSearchCandidates = 7;

enum BodyIndex : std::size_t {
  kYawRate = 0,
  kSpeed = 1,
  kSideslip = 2,
  kDrivenWheelSpeed = 3,
};

// Frenet frame slots. e is positive to the LEFT of the raceline tangent,
// dphi is the course heading relative to the path tangent, and s is the
// loop-continuous (unwrapped) path evolution.
enum FrameIndex : std::size_t {
  kLateralDeviation = 4,
  kRelativeHeading = 5,
  kPathEvolution = 6,
};

// Cartesian frame slots occupy the same three trailing entries. Position is
// map-frame ENU metres and heading is standard ROS ENU yaw (CCW from east),
// not the EPIC CSV heading. Frenet quantities used by the cost function are
// recovered by projecting each rollout state onto the raceline.
enum CartesianFrameIndex : std::size_t {
  kEastM = 4,
  kNorthM = 5,
  kHeadingEnu = 6,
};

// Frame the rollout integrates in. Both frames share the same state width,
// controls, costs, and published messages.
enum class FrameKind : std::uint8_t {
  kFrenet = 0,
  kCartesian = 1,
};

// kRearSteering is only actuated by models that read it; front-steer-only
// models ignore it and config load pins its sigma and bounds to zero so the
// solver does not spend samples on a control that cannot move the trajectory.
enum ControlIndex : std::size_t {
  kSteering = 0,
  kWheelTorque = 1,
  kRearSteering = 2,
};

template<std::size_t Size>
struct FixedVector {
  float values[Size]{};

#if defined(__CUDACC__)
  __host__ __device__
#endif
  constexpr float & operator[](const std::size_t index) noexcept { return values[index]; }

#if defined(__CUDACC__)
  __host__ __device__
#endif
  constexpr const float & operator[](const std::size_t index) const noexcept {
    return values[index];
  }

  constexpr float * begin() noexcept { return values; }
  constexpr float * end() noexcept { return values + Size; }
  constexpr const float * begin() const noexcept { return values; }
  constexpr const float * end() const noexcept { return values + Size; }
};

using BodyState = FixedVector<kBodyStateDim>;
using State = FixedVector<kStateDim>;
using Control = FixedVector<kControlDim>;

struct VehicleParameters {
  float mass_kg{20.0F};
  float yaw_inertia_kgm2{1.0F};
  float cg_to_front_m{0.18F};
  float cg_to_rear_m{0.18F};
  float front_cornering_stiffness_nprad{1200.0F};
  float rear_cornering_stiffness_nprad{1400.0F};
  float front_friction_coefficient{1.0F};
  float rear_friction_coefficient{1.0F};
  float wheel_radius_m{0.05F};
  float driven_wheel_inertia_kgm2{0.01F};
  float front_brake_bias{0.0F};
  bool locked_awd{false};
  int motor_pole_pairs{1};
  float min_current_a{-50.0F};
  float max_current_a{50.0F};
  float min_steering_angle_rad{-0.5F};
  float max_steering_angle_rad{0.5F};
  float rear_min_steering_angle_rad{0.0F};
  float rear_max_steering_angle_rad{0.0F};
  float motor_to_wheel_ratio{1.0F};
};

struct AdaptationConfig {
  bool adaptive_lambda{true};
  float ess_fraction_min{0.002F};
  float ess_fraction_max{0.02F};
  float lambda_min{0.05F};
  float lambda_max{1.0e4F};
  bool adaptive_sigma{true};
  float sigma_alpha{0.1F};
  float sigma_scale_min{0.5F};
  float sigma_scale_max{2.0F};
  bool speed_scaled_steering{true};
  float steering_reference_speed_mps{4.0F};
  float steering_minimum_scale{0.2F};
};

// Local multiple-shooting SQP pass applied once to the control/state mean
// produced by MPPI.  MPPI remains the global optimizer; this pass improves the
// dynamically coupled trajectory and command smoothness before the sequence is
// shifted into the next warm start.
struct MpcRefinementConfig {
  bool enabled{false};
  std::uint16_t sqp_iterations{3};
  std::uint16_t pcg_iterations{48};
  float pcg_tolerance{1.0e-4F};
  float constraint_tolerance{2.0e-2F};
  float finite_difference_relative_step{1.0e-3F};
  float hessian_regularization{1.0e-3F};
  float merit_constraint_penalty{1.0e4F};
  float state_proximity_weight{1.0e-2F};
  float control_proximity_weight{1.0F};
  float maximum_state_step{5.0F};
  std::array<float, kControlDim> maximum_control_step{0.10F, 0.25F, 0.10F};
  std::array<float, kControlDim> maximum_control_rate{4.0F, 10.0F, 4.0F};
};

struct MppiConfig {
  std::uint32_t num_samples{2001};
  std::uint16_t horizon{50};
  float dt_s{0.1F};
  std::uint16_t integration_substeps{2};
  float lambda{2.0F};
  // The rear steering channel defaults to pinned: zero sigma with zero-width
  // bounds forces every candidate to exactly zero, so a config that never
  // mentions rear steering behaves precisely like the front-steer-only stack.
  std::array<float, kControlDim> sigma{0.10F, 1.0F, 0.0F};
  std::array<float, kControlDim> control_min{-0.5F, -5.0F, 0.0F};
  std::array<float, kControlDim> control_max{0.5F, 5.0F, 0.0F};
  std::uint16_t noise_smoothing_window{5};
  std::uint16_t control_delay_steps{0};
  // Number of predicted states after the current state that may trigger the
  // all-sample obstacle brake. The obstacle cost itself remains latched over
  // the complete horizon.
  std::uint16_t obstacle_latch_brake_steps{5};
  float control_cost_gamma{0.8F};
  bool special_samples{true};
  bool use_reference_controls{true};
  bool expected_trajectory{true};
  std::uint64_t seed{0};
  FrameKind frame{FrameKind::kFrenet};
  AdaptationConfig adaptation{};
  MpcRefinementConfig refinement{};
};

struct CostWeights {
  std::array<float, kStateDim> reference_tracking{};
  // Rear steering mirrors the front steering weights rather than defaulting to
  // zero: an unpinned rear channel with no cost would be free to oscillate.
  std::array<float, kControlDim> control_effort{1.0e-3F, 1.0e-4F, 1.0e-3F};
  std::array<float, kControlDim> control_smoothness{1.0e4F, 1.0e-1F, 1.0e4F};
  // Physical state/control-rate penalties. Acceleration and deceleration are
  // separated so braking comfort can be tuned independently from propulsion.
  float longitudinal_acceleration{0.0F};
  float longitudinal_deceleration{0.0F};
  std::array<float, kControlDim> control_rate{};
  float velocity_profile{30.0F};
  float velocity_overspeed_multiplier{4.0F};
  float progress{1.0F};
  float boundary{200.0F};
  float boundary_margin_m{1.0F};
  float crash{20000.0F};
  float crash_discount{0.9F};
  float crash_buffer_m{0.0F};
  float sideslip{200.0F};
  float maximum_sideslip_rad{0.8F};
  float sideslip_kill{60000.0F};
  float lateral_damping{1.0F};
  float lateral_decay_rate{0.2F};
  float wheel_slip{500.0F};
  float wheel_slip_band{0.2F};
};

struct ObstacleConfig {
  bool enabled{false};
  float grid_resolution_m{0.05F};
  float grid_width_m{12.0F};
  float grid_height_m{12.0F};
  float maximum_distance_m{5.0F};
  float obstacle_inflation_radius_m{0.03F};
  std::uint32_t confirmation_updates{2U};
  std::uint32_t persistence_updates{4U};
  float association_distance_m{0.10F};
  float pose_history_s{0.10F};
  float maximum_extrapolation_s{0.02F};
  float distance_weight{500.0F};
  float influence_distance_m{0.75F};
  float latch_threshold_m{0.10F};
  float latching_weight{60000.0F};
  float footprint_length_m{0.557F};
  float footprint_width_m{0.249F};
  std::uint16_t footprint_circles{3U};
};

struct ObstacleField {
  std::int64_t stamp_ns{};
  std::uint64_t generation{};
  float origin_east_m{};
  float origin_north_m{};
  float resolution_m{};
  std::uint32_t width{};
  std::uint32_t height{};
  std::vector<float> signed_distance_m;

  [[nodiscard]] bool valid() const noexcept {
    return resolution_m > 0.0F && width > 1U && height > 1U &&
      signed_distance_m.size() == static_cast<std::size_t>(width) * height;
  }
};

struct ReferencePoint {
  float s_m{};
  float curvature_inv_m{};
  float east_m{};
  float north_m{};
  float heading_from_north_rad{};
  float yaw_rate_radps{};
  float speed_mps{};
  float sideslip_rad{};
  float steering_rad{};
  float torque_nm{};
  float driven_wheel_speed_mps{};
  float e_min_m{-1.0F};
  float e_max_m{1.0F};
};

struct Projection {
  float s_m{};  // continuous/unwrapped on closed tracks
  float e_m{};
  float relative_course_rad{};
  std::size_t segment_index{};
  bool valid{false};
};

struct ReferenceHorizon {
  std::vector<State> states;          // T + 1
  std::vector<Control> controls;      // T
  std::vector<float> curvature;       // T
  std::vector<float> s_grid;          // T + 1, unwrapped
  std::vector<float> speed_profile;   // T + 1
  std::vector<float> e_min;           // T + 1
  std::vector<float> e_max;           // T + 1
};

// Per-term decomposition of the scalar cost the solver minimizes. The entries
// sum exactly to the cost of the trajectory they were measured on, so a debug
// consumer can see which term is driving a solve instead of only its sum.
// progress is negative because it rewards arc length travelled.
enum CostTermIndex : std::size_t {
  kTermReferenceTracking = 0,
  kTermVelocityProfile,
  kTermBoundary,
  kTermCrash,
  kTermSideslip,
  kTermSideslipKill,
  kTermObstacleDistance,
  kTermObstacleLatching,
  kTermLateralDamping,
  kTermWheelSlip,
  kTermControlEffort,
  kTermControlSmoothness,
  kTermControlRate,
  kTermImportanceSampling,
  kTermLongitudinalAcceleration,
  kTermLongitudinalDeceleration,
  kTermProgress,
  kCostTermCount,
};

// Value of the first_*_step fields when that latch never fired over the horizon.
constexpr std::uint16_t kNoCostLatch = 0xFFFFU;

struct CostTerms {
  float values[kCostTermCount]{};
  // Detail splits. Each array sums to its grouped entry in values, so adding
  // them to a total would double count.
  float reference_tracking[kStateDim]{};
  float control_effort[kControlDim]{};
  float control_smoothness[kControlDim]{};
  float control_rate[kControlDim]{};
  // Horizon step at which each latched penalty first fired, kNoCostLatch if it
  // never did. crash and sideslip_kill carry weights three orders of magnitude
  // above the shaping terms, so once one latches it swamps the decomposition
  // and only the step it fired at is informative.
  std::uint16_t first_crash_step{kNoCostLatch};
  std::uint16_t first_sideslip_step{kNoCostLatch};
  std::uint16_t first_obstacle_step{kNoCostLatch};
  // Smallest signed footprint clearance seen over the horizon. Equals the
  // obstacle field's maximum distance when obstacles are disabled or absent.
  float minimum_clearance_m{};

#if defined(__CUDACC__)
  __host__ __device__
#endif
  float total() const noexcept {
    float sum = 0.0F;
    for (std::size_t i = 0; i < kCostTermCount; ++i) {
      sum += values[i];
    }
    return sum;
  }
};

// Parallel to CostTermIndex; used for the terminal summary and message field
// documentation. Host only.
inline const char * CostTermName(const std::size_t index) noexcept {
  constexpr const char * kNames[kCostTermCount] = {
    "reference_tracking", "velocity_profile", "boundary", "crash", "sideslip",
    "sideslip_kill", "obstacle_distance", "obstacle_latching", "lateral_damping",
    "wheel_slip", "control_effort", "control_smoothness", "control_rate",
    "importance_sampling", "longitudinal_acceleration", "longitudinal_deceleration",
    "progress"};
  return index < kCostTermCount ? kNames[index] : "unknown";
}

struct MppiDiagnostics {
  float minimum_cost{std::numeric_limits<float>::infinity()};
  float effective_sample_size{};
  float lambda_used{};
  std::array<float, kControlDim> sigma_used{};
  float solve_time_ms{};
  std::uint32_t finite_rollouts{};
  // Near-term obstacle latches used by the safety brake, not full-horizon cost
  // latches. The window is configured by obstacle_latch_brake_steps.
  std::uint32_t obstacle_latched_rollouts{};
  std::uint32_t finite_unlatched_rollouts{};
  bool all_rollouts_obstacle_latched{};
  bool obstacle_field_active{};
  bool obstacle_brake_active{};
  bool refinement_attempted{};
  bool refinement_accepted{};
  std::uint16_t refinement_iterations{};
  std::uint16_t refinement_pcg_iterations{};
  float refinement_time_ms{};
  float refinement_cost_before{std::numeric_limits<float>::infinity()};
  float refinement_cost_after{std::numeric_limits<float>::infinity()};
  float refinement_merit_before{std::numeric_limits<float>::infinity()};
  float refinement_merit_after{std::numeric_limits<float>::infinity()};
  float refinement_constraint_residual_before{std::numeric_limits<float>::infinity()};
  float refinement_constraint_residual{std::numeric_limits<float>::infinity()};
  // Difference between the exact pre-SQP MPPI mean and the trajectory returned
  // by Solve. Rejected refinements report zero because MPPI remains published.
  std::array<float, kStateDim> refinement_state_rms_delta{};
  std::array<float, kStateDim> refinement_state_max_delta{};
  std::array<float, kControlDim> refinement_control_rms_delta{};
  std::array<float, kControlDim> refinement_control_max_delta{};
  std::array<float, kControlDim> refinement_first_control_delta{};
  std::array<float, kMpcLineSearchCandidates> refinement_trial_merits{};
  std::array<float, kMpcLineSearchCandidates> refinement_trial_constraint_residuals{};
  // Present only on solves the ROS runtime asked to decompose, which it does at
  // its own reduced rate. Measured on the published (expected) trajectory.
  std::optional<CostTerms> cost_terms{};
};

struct WeightedRollout {
  std::vector<State> states;  // T + 1
  float weight{};
};

struct MppiSolution {
  std::vector<State> states;       // T + 1 internally
  std::vector<Control> controls;   // T
  std::vector<WeightedRollout> sampled_rollouts;  // optional, highest weight first
  // Nonzero when Solve queued a device-side rollout snapshot for asynchronous
  // collection.  The visualization worker consumes it through
  // CudaMppiController::CollectVisualization rather than blocking Solve.
  std::uint64_t visualization_snapshot_id{};
  MppiDiagnostics diagnostics{};
};

}  // namespace xxcar::mppi
