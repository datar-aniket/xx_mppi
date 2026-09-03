#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace xxcar::mppi {

constexpr std::size_t kBodyStateDim = 4;
constexpr std::size_t kFrameStateDim = 3;
constexpr std::size_t kStateDim = kBodyStateDim + kFrameStateDim;
constexpr std::size_t kControlDim = 2;

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

enum ControlIndex : std::size_t {
  kSteering = 0,
  kWheelTorque = 1,
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

struct MppiConfig {
  std::uint32_t num_samples{2001};
  std::uint16_t horizon{50};
  float dt_s{0.1F};
  std::uint16_t integration_substeps{2};
  float lambda{2.0F};
  std::array<float, kControlDim> sigma{0.10F, 1.0F};
  std::array<float, kControlDim> control_min{-0.5F, -5.0F};
  std::array<float, kControlDim> control_max{0.5F, 5.0F};
  std::uint16_t noise_smoothing_window{5};
  std::uint16_t control_delay_steps{0};
  float control_cost_gamma{0.8F};
  bool special_samples{true};
  bool use_reference_controls{true};
  bool expected_trajectory{true};
  std::uint64_t seed{0};
  FrameKind frame{FrameKind::kFrenet};
  AdaptationConfig adaptation{};
};

struct CostWeights {
  std::array<float, kStateDim> reference_tracking{};
  std::array<float, kControlDim> control_effort{1.0e-3F, 1.0e-4F};
  std::array<float, kControlDim> control_smoothness{1.0e4F, 1.0e-1F};
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

struct MppiDiagnostics {
  float minimum_cost{std::numeric_limits<float>::infinity()};
  float effective_sample_size{};
  float lambda_used{};
  float solve_time_ms{};
  std::uint32_t finite_rollouts{};
};

struct WeightedRollout {
  std::vector<State> states;  // T + 1
  float weight{};
};

struct MppiSolution {
  std::vector<State> states;       // T + 1 internally
  std::vector<Control> controls;   // T
  std::vector<WeightedRollout> sampled_rollouts;  // optional, highest weight first
  MppiDiagnostics diagnostics{};
};

}  // namespace xxcar::mppi
