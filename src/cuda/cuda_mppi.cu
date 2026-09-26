#include "xx_mppi/controller/cuda_mppi.hpp"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cub/cub.cuh>
#include <math_constants.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "xx_mppi/dynamics/frames.hpp"
#include "xx_mppi/costs/map_boundary.hpp"
#include "xx_mppi/dynamics/models/dynamic_bicycle_fiala.hpp"
#include "xx_mppi/dynamics/models/dynamic_bicycle_fiala_4ws.hpp"
#include "xx_mppi/dynamics/models/kinematic_bicycle.hpp"
#include "xx_mppi/dynamics/tensorrt_model.hpp"

namespace xxcar::mppi {
namespace {

void CheckCuda(const cudaError_t status, const char * operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

struct DeviceTrack {
  const float * s;
  const float * curvature;
  const float * east;
  const float * north;
  const float * heading;  // EPIC phi, CCW from north; tangent is (-sin, cos).
  std::uint32_t count;
  float s_min;
  float s_max;
  float length;
  bool closed;
  // Segments scanned either side of the arc-length hint when a Cartesian
  // rollout state is projected back onto the raceline.
  std::uint32_t search_span;
};

struct DeviceObstacleField {
  const float * signed_distance;
  float origin_east;
  float origin_north;
  float resolution;
  std::uint32_t width;
  std::uint32_t height;
  bool valid;
};

// Frenet quantities the cost function needs. In the Frenet frame they are read
// straight out of the state; in the Cartesian frame they come from projecting
// the ENU state onto the raceline.
struct FrenetView {
  float lateral_deviation;
  float relative_course;
  float path_evolution;
};

struct DeviceReference {
  const State * states;
  const Control * controls;
  const float * s_grid;
  const float * speed;
  const float * e_min;
  const float * e_max;
  std::uint16_t horizon;
};

struct DeviceCosts {
  float reference_tracking[kStateDim];
  float control_effort[kControlDim];
  float control_smoothness[kControlDim];
  float longitudinal_acceleration;
  float longitudinal_deceleration;
  float control_rate[kControlDim];
  float velocity_profile;
  float velocity_overspeed_multiplier;
  float progress;
  float boundary;
  float boundary_margin;
  float crash;
  float crash_discount;
  float crash_buffer;
  float sideslip;
  float maximum_sideslip;
  float sideslip_kill;
  float lateral_damping;
  float lateral_decay_rate;
  float wheel_slip;
  float wheel_slip_band;
  float obstacle_distance;
  float obstacle_influence_distance;
  float obstacle_latch_threshold;
  float obstacle_latching;
  float obstacle_maximum_distance;
  float footprint_length;
  float footprint_width;
  std::uint16_t footprint_circles;
  bool obstacle_enabled;
};

struct DeviceMppi {
  std::uint32_t samples;
  std::uint16_t horizon;
  std::uint16_t substeps;
  std::uint16_t smoothing_window;
  std::uint16_t delay_steps;
  std::uint16_t obstacle_brake_steps;
  float dt;
  float lambda;
  float sigma[kControlDim];
  float control_min[kControlDim];
  float control_max[kControlDim];
  float gamma;
  bool special_samples;
  bool use_reference_controls;
  bool speed_scaled_steering;
  float steering_reference_speed;
  float steering_minimum_scale;
  ModelKind model_kind;
  IntegratorKind integrator_kind;
  FrameKind frame;
};

struct DeviceRefinement {
  bool enabled;
  std::uint16_t sqp_iterations;
  std::uint16_t pcg_iterations;
  float pcg_tolerance;
  float constraint_tolerance;
  float finite_difference_relative_step;
  float hessian_regularization;
  float merit_constraint_penalty;
  float state_proximity_weight;
  float control_proximity_weight;
  float maximum_state_step;
  float maximum_control_step[kControlDim];
  float maximum_control_rate[kControlDim];
};

struct RefinementMetrics {
  float cost;
  float merit;
  float maximum_dynamics_residual;
  std::uint8_t safe;
};

struct RefinementDelta {
  float state_rms[kStateDim];
  float state_max[kStateDim];
  float control_rms[kControlDim];
  float control_max[kControlDim];
  float first_control[kControlDim];
};

__device__ float WrapTrackS(float s, const DeviceTrack & track) {
  if (!track.closed) {
    return fminf(fmaxf(s, track.s_min), track.s_max);
  }
  float wrapped = fmodf(s - track.s_min, track.length);
  if (wrapped < 0.0F) {
    wrapped += track.length;
  }
  return track.s_min + wrapped;
}

__device__ float InterpolateTrackCurvature(const float unwrapped_s, const DeviceTrack & track) {
  const float query = WrapTrackS(unwrapped_s, track);
  std::uint32_t low = 0;
  std::uint32_t high = track.count - 1U;
  while (high - low > 1U) {
    const std::uint32_t middle = low + (high - low) / 2U;
    if (track.s[middle] <= query) {
      low = middle;
    } else {
      high = middle;
    }
  }
  const float denominator = fmaxf(track.s[high] - track.s[low], 1.0e-9F);
  const float fraction = (query - track.s[low]) / denominator;
  return track.curvature[low] + fraction * (track.curvature[high] - track.curvature[low]);
}

__device__ std::uint32_t LocateTrackSegment(const float wrapped_s, const DeviceTrack & track) {
  std::uint32_t low = 0;
  std::uint32_t high = track.count - 1U;
  while (high - low > 1U) {
    const std::uint32_t middle = low + (high - low) / 2U;
    if (track.s[middle] <= wrapped_s) {
      low = middle;
    } else {
      high = middle;
    }
  }
  return low;
}

__device__ float InterpolateAngleDevice(const float a, const float b, const float t) {
  return atan2f(
    a == b ? sinf(a) : sinf(a) + t * (sinf(b) - sinf(a)),
    a == b ? cosf(a) : cosf(a) + t * (cosf(b) - cosf(a)));
}

__device__ float SampleObstacleField(
  const DeviceObstacleField & field, const float east, const float north,
  const float outside_value)
{
  if (!field.valid) {
    return outside_value;
  }
  const float grid_x = (east - field.origin_east) / field.resolution - 0.5F;
  const float grid_y = (north - field.origin_north) / field.resolution - 0.5F;
  if (grid_x < 0.0F || grid_y < 0.0F ||
    grid_x >= static_cast<float>(field.width - 1U) ||
    grid_y >= static_cast<float>(field.height - 1U))
  {
    return outside_value;
  }
  const auto x0 = static_cast<std::uint32_t>(floorf(grid_x));
  const auto y0 = static_cast<std::uint32_t>(floorf(grid_y));
  const float tx = grid_x - static_cast<float>(x0);
  const float ty = grid_y - static_cast<float>(y0);
  const std::size_t row0 = static_cast<std::size_t>(y0) * field.width;
  const std::size_t row1 = static_cast<std::size_t>(y0 + 1U) * field.width;
  const float lower = field.signed_distance[row0 + x0] + tx *
    (field.signed_distance[row0 + x0 + 1U] - field.signed_distance[row0 + x0]);
  const float upper = field.signed_distance[row1 + x0] + tx *
    (field.signed_distance[row1 + x0 + 1U] - field.signed_distance[row1 + x0]);
  return lower + ty * (upper - lower);
}

__device__ float VehicleObstacleClearance(
  const State & state, const FrenetView & frenet, const DeviceTrack & track,
  const DeviceObstacleField & field, const DeviceMppi & config,
  const DeviceCosts & weights)
{
  if (!weights.obstacle_enabled || !field.valid) {
    return weights.obstacle_maximum_distance;
  }
  float center_east = state[kEastM];
  float center_north = state[kNorthM];
  float yaw = state[kHeadingEnu];
  if (config.frame == FrameKind::kFrenet) {
    const float wrapped_s = WrapTrackS(frenet.path_evolution, track);
    const std::uint32_t low = LocateTrackSegment(wrapped_s, track);
    const std::uint32_t high = low + 1U;
    const float fraction = (wrapped_s - track.s[low]) /
      fmaxf(track.s[high] - track.s[low], 1.0e-9F);
    const float track_heading = InterpolateAngleDevice(
      track.heading[low], track.heading[high], fraction);
    center_east = track.east[low] + fraction *
      (track.east[high] - track.east[low]) -
      frenet.lateral_deviation * cosf(track_heading);
    center_north = track.north[low] + fraction *
      (track.north[high] - track.north[low]) -
      frenet.lateral_deviation * sinf(track_heading);
    yaw = track_heading + frenet.relative_course - state[kSideslip] +
      0.5F * CUDART_PI_F;
  }
  const float segment_length = weights.footprint_length /
    static_cast<float>(weights.footprint_circles);
  const float radius = sqrtf(
    0.25F * segment_length * segment_length +
    0.25F * weights.footprint_width * weights.footprint_width);
  float clearance = CUDART_INF_F;
  for (std::uint16_t i = 0; i < weights.footprint_circles; ++i) {
    const float offset = -0.5F * weights.footprint_length +
      (static_cast<float>(i) + 0.5F) * segment_length;
    const float distance = SampleObstacleField(
      field, center_east + offset * cosf(yaw), center_north + offset * sinf(yaw),
      weights.obstacle_maximum_distance) - radius;
    clearance = fminf(clearance, distance);
  }
  return clearance;
}

// Device twin of Raceline::Project restricted to a window of segments around
// the previous arc length. Positive lateral deviation is left of the tangent
// and the returned s is unwrapped to stay continuous with the hint, matching
// the host projector exactly.
__device__ FrenetView ProjectOnTrack(
  const float east_m, const float north_m, const float course_from_north_rad,
  const float unwrapped_s_hint, const DeviceTrack & track)
{
  const int segments = static_cast<int>(track.count) - 1;
  const int center = static_cast<int>(
    LocateTrackSegment(WrapTrackS(unwrapped_s_hint, track), track));
  const int span = static_cast<int>(track.search_span);
  float best_distance_squared = CUDART_INF_F;
  FrenetView best{0.0F, 0.0F, unwrapped_s_hint};
  for (int offset = -span; offset <= span; ++offset) {
    int index = center + offset;
    if (track.closed) {
      index = ((index % segments) + segments) % segments;
    } else if (index < 0 || index >= segments) {
      continue;
    }
    const auto i = static_cast<std::uint32_t>(index);
    const float dx = track.east[i + 1U] - track.east[i];
    const float dy = track.north[i + 1U] - track.north[i];
    const float norm_squared = dx * dx + dy * dy;
    if (norm_squared <= 1.0e-12F) {
      continue;
    }
    const float fraction = fminf(fmaxf(
      ((east_m - track.east[i]) * dx + (north_m - track.north[i]) * dy) / norm_squared,
      0.0F), 1.0F);
    const float error_east = east_m - (track.east[i] + fraction * dx);
    const float error_north = north_m - (track.north[i] + fraction * dy);
    const float distance_squared = error_east * error_east + error_north * error_north;
    if (distance_squared >= best_distance_squared) {
      continue;
    }
    best_distance_squared = distance_squared;
    const float inverse_norm = rsqrtf(norm_squared);
    const float tangent_east = dx * inverse_norm;
    const float tangent_north = dy * inverse_norm;
    const float path_heading = InterpolateAngleDevice(
      track.heading[i], track.heading[i + 1U], fraction);
    const float projected_s =
      track.s[i] + fraction * (track.s[i + 1U] - track.s[i]);
    best.lateral_deviation =
      error_east * (-tangent_north) + error_north * tangent_east;
    best.relative_course = wrap_to_pi(course_from_north_rad - path_heading);
    best.path_evolution = track.closed ?
      projected_s + roundf((unwrapped_s_hint - projected_s) / track.length) * track.length :
      projected_s;
  }
  return best;
}

// Frenet view of a rollout state for the active frame.
__device__ FrenetView FrameView(
  const State & state, const float unwrapped_s_hint, const DeviceTrack & track,
  const DeviceMppi & config)
{
  if (config.frame == FrameKind::kFrenet) {
    return FrenetView{
      state[kLateralDeviation], state[kRelativeHeading], state[kPathEvolution]};
  }
  const float course_from_north = enu_yaw_to_heading_from_north(
    state[kHeadingEnu] + state[kSideslip]);
  return ProjectOnTrack(
    state[kEastM], state[kNorthM], course_from_north, unwrapped_s_hint, track);
}

__device__ float InterpolateReference(
  const float * values, const float query, const DeviceReference & reference)
{
  const std::uint16_t count = reference.horizon + 1U;
  if (query <= reference.s_grid[0]) {
    return values[0];
  }
  if (query >= reference.s_grid[count - 1U]) {
    return values[count - 1U];
  }
  std::uint16_t low = 0;
  std::uint16_t high = count - 1U;
  while (high - low > 1U) {
    const std::uint16_t middle = static_cast<std::uint16_t>(low + (high - low) / 2U);
    if (reference.s_grid[middle] <= query) {
      low = middle;
    } else {
      high = middle;
    }
  }
  const float fraction = (query - reference.s_grid[low]) /
    fmaxf(reference.s_grid[high] - reference.s_grid[low], 1.0e-9F);
  return values[low] + fraction * (values[high] - values[low]);
}

__device__ StateDerivative EvaluateDerivative(
  const State & state, const Control & control, const DeviceTrack & track,
  const DeviceMppi & config, const VehicleParameters & parameters)
{
  const BodyState body{
    state[kYawRate], state[kSpeed], state[kSideslip], state[kDrivenWheelSpeed]};
  BodyDerivative body_derivative{};
  if (config.model_kind == ModelKind::kKinematicBicycle) {
    body_derivative = KinematicBicycle(parameters).Derivative(body, control);
  } else if (config.model_kind == ModelKind::kDynamicBicycleFiala4ws) {
    body_derivative = DynamicBicycleFiala4ws(parameters).Derivative(body, control);
  } else {
    body_derivative = DynamicBicycleFiala(parameters).Derivative(body, control);
  }
  if (config.frame == FrameKind::kCartesian) {
    return CartesianDerivative(state, body_derivative);
  }
  return FrenetDerivative(
    state, body_derivative, InterpolateTrackCurvature(state[kPathEvolution], track));
}

__device__ State IntegrateAnalyticStep(
  State state, const Control & control, const DeviceTrack & track,
  const DeviceMppi & config, const VehicleParameters & parameters)
{
  const std::uint16_t substeps = config.substeps == 0U ? 1U : config.substeps;
  const float h = config.dt / static_cast<float>(substeps);
  for (std::uint16_t substep = 0; substep < substeps; ++substep) {
    const auto k1 = EvaluateDerivative(state, control, track, config, parameters);
    if (config.integrator_kind == IntegratorKind::kEuler) {
      for (std::size_t i = 0; i < kStateDim; ++i) {
        state[i] += h * k1[i];
      }
      continue;
    }
    State temporary{};
    for (std::size_t i = 0; i < kStateDim; ++i) {
      temporary[i] = state[i] + 0.5F * h * k1[i];
    }
    const auto k2 = EvaluateDerivative(temporary, control, track, config, parameters);
    for (std::size_t i = 0; i < kStateDim; ++i) {
      temporary[i] = state[i] + 0.5F * h * k2[i];
    }
    const auto k3 = EvaluateDerivative(temporary, control, track, config, parameters);
    for (std::size_t i = 0; i < kStateDim; ++i) {
      temporary[i] = state[i] + h * k3[i];
    }
    const auto k4 = EvaluateDerivative(temporary, control, track, config, parameters);
    for (std::size_t i = 0; i < kStateDim; ++i) {
      state[i] += h / 6.0F * (k1[i] + 2.0F * k2[i] + 2.0F * k3[i] + k4[i]);
    }
  }
  return state;
}

// Accumulates one cost contribution. kBreakdown is a template parameter so the
// sampling kernels instantiate StateCost with the attribution branches removed
// entirely: the hot path over every sample and horizon step is unchanged, and
// the debug decomposition cannot drift from the cost the solver minimizes
// because there is only one implementation of it.
template<bool kBreakdown>
__device__ inline void AddCost(
  float & cost, CostTerms * const terms, const std::size_t index, const float value)
{
  cost += value;
  if constexpr (kBreakdown) {
    terms->values[index] += value;
  }
}

// latch_violations is false for the initial state, which every sample shares:
// latching a crash or excessive sideslip there would set the same flag for the
// whole population and remove all boundary discrimination from the solve.
//
// terms and step are read only when kBreakdown is true, in which case the
// per-term contributions of this step are added into terms.
template<bool kBreakdown>
__device__ float StateCost(
  const State & state, const FrenetView & frenet, const State & desired,
  const DeviceTrack & track, const DeviceReference & reference,
  const DeviceObstacleField & obstacle_field, const DeviceMppi & config,
  const DeviceCosts & weights, bool & crashed, bool & excessive_sideslip,
  bool & obstacle_latched, const float crash_discount, const bool latch_violations,
  CostTerms * const terms = nullptr, const std::uint16_t step = 0U)
{
  float cost = 0.0F;
  for (std::size_t i = 0; i < kBodyStateDim; ++i) {
    if (!isfinite(state[i])) {
      return CUDART_INF_F;
    }
    const float error = state[i] - desired[i];
    const float penalty = weights.reference_tracking[i] * error * error;
    AddCost<kBreakdown>(cost, terms, kTermReferenceTracking, penalty);
    if constexpr (kBreakdown) {
      terms->reference_tracking[i] += penalty;
    }
  }
  if (!isfinite(frenet.lateral_deviation) || !isfinite(frenet.relative_course) ||
    !isfinite(frenet.path_evolution))
  {
    return CUDART_INF_F;
  }
  const float frame_error[kFrameStateDim] = {
    frenet.lateral_deviation - desired[kLateralDeviation],
    wrap_to_pi(frenet.relative_course - desired[kRelativeHeading]),
    frenet.path_evolution - desired[kPathEvolution]};
  for (std::size_t i = 0; i < kFrameStateDim; ++i) {
    const float penalty = weights.reference_tracking[kBodyStateDim + i] *
      frame_error[i] * frame_error[i];
    AddCost<kBreakdown>(cost, terms, kTermReferenceTracking, penalty);
    if constexpr (kBreakdown) {
      terms->reference_tracking[kBodyStateDim + i] += penalty;
    }
  }
  const float speed_target = InterpolateReference(
    reference.speed, frenet.path_evolution, reference);
  const float speed_error = state[kSpeed] - speed_target;
  AddCost<kBreakdown>(cost, terms, kTermVelocityProfile, weights.velocity_profile *
    (speed_error > 0.0F ? weights.velocity_overspeed_multiplier : 1.0F) *
    speed_error * speed_error);

  const float e_min = InterpolateReference(reference.e_min, frenet.path_evolution, reference);
  const float e_max = InterpolateReference(reference.e_max, frenet.path_evolution, reference);
  const auto boundary = EvaluateMapBoundary(
    frenet.lateral_deviation, e_min, e_max, weights.boundary,
    weights.boundary_margin, weights.crash_buffer);
  AddCost<kBreakdown>(cost, terms, kTermBoundary, boundary.shaping_cost);
  crashed = crashed || (latch_violations && boundary.violated);
  if (crashed) {
    AddCost<kBreakdown>(cost, terms, kTermCrash, weights.crash * crash_discount);
    if constexpr (kBreakdown) {
      if (terms->first_crash_step == kNoCostLatch) {
        terms->first_crash_step = step;
      }
    }
  }
  const float beta = state[kSideslip];
  AddCost<kBreakdown>(cost, terms, kTermSideslip, weights.sideslip * beta * beta);
  excessive_sideslip = excessive_sideslip ||
    (latch_violations && fabsf(beta) > weights.maximum_sideslip);
  if (excessive_sideslip) {
    AddCost<kBreakdown>(cost, terms, kTermSideslipKill,
      weights.sideslip_kill / static_cast<float>(reference.horizon + 1U));
    if constexpr (kBreakdown) {
      if (terms->first_sideslip_step == kNoCostLatch) {
        terms->first_sideslip_step = step;
      }
    }
  }
  if (weights.obstacle_enabled && obstacle_field.valid) {
    const float clearance = VehicleObstacleClearance(
      state, frenet, track, obstacle_field, config, weights);
    const float deficit = fmaxf(weights.obstacle_influence_distance - clearance, 0.0F);
    AddCost<kBreakdown>(cost, terms, kTermObstacleDistance,
      weights.obstacle_distance * deficit * deficit);
    if constexpr (kBreakdown) {
      terms->minimum_clearance_m = fminf(terms->minimum_clearance_m, clearance);
    }
    obstacle_latched = obstacle_latched ||
      (latch_violations && clearance < weights.obstacle_latch_threshold);
    if (obstacle_latched) {
      AddCost<kBreakdown>(cost, terms, kTermObstacleLatching,
        weights.obstacle_latching / static_cast<float>(reference.horizon + 1U));
      if constexpr (kBreakdown) {
        if (terms->first_obstacle_step == kNoCostLatch) {
          terms->first_obstacle_step = step;
        }
      }
    }
  }
  const float lateral_rate = state[kSpeed] * sinf(frenet.relative_course);
  const float damping =
    lateral_rate + weights.lateral_decay_rate * frenet.lateral_deviation;
  AddCost<kBreakdown>(cost, terms, kTermLateralDamping,
    weights.lateral_damping * damping * damping);
  const float longitudinal_speed = fmaxf(
    state[kSpeed] * cosf(state[kSideslip]), 1.0F);
  const float slip = (state[kDrivenWheelSpeed] - longitudinal_speed) / longitudinal_speed;
  const float excess_slip = fmaxf(fabsf(slip) - weights.wheel_slip_band, 0.0F);
  AddCost<kBreakdown>(cost, terms, kTermWheelSlip,
    weights.wheel_slip * excess_slip * excess_slip);
  return cost;
}

__global__ void InitRandomStates(
  curandStatePhilox4_32_10_t * states, const std::uint32_t count, const std::uint64_t seed)
{
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    curand_init(seed, index, 0, &states[index]);
  }
}

__global__ void GenerateNoise(
  curandStatePhilox4_32_10_t * random_states, float * raw_noise,
  const DeviceMppi config)
{
  const std::uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample >= config.samples) {
    return;
  }
  // Philox yields four normals per call and the draws are written by index, not
  // by channel name. Naming them individually is how a newly added channel ends
  // up reading uninitialized noise; the static_assert turns outgrowing one call
  // into a compile error instead.
  static_assert(kControlDim <= 4U, "GenerateNoise supplies four draws per step");
  auto random = random_states[sample];
  for (std::uint16_t t = 0; t < config.horizon; ++t) {
    const float4 draw = curand_normal4(&random);
    const float draws[4] = {draw.x, draw.y, draw.z, draw.w};
    const std::size_t offset =
      (static_cast<std::size_t>(sample) * config.horizon + t) * kControlDim;
    for (std::size_t channel = 0; channel < kControlDim; ++channel) {
      raw_noise[offset + channel] = draws[channel];
    }
  }
  random_states[sample] = random;
}

// Sampling sigma for one control channel at the current speed. Both steering
// channels taper as the car speeds up, so the sampled steering range narrows
// where a large angle would be violent; torque is unscaled. Four kernels need
// this identically — the candidate builder, both cost paths, and the debug
// breakdown — and they must agree, because the importance-sampling term divides
// by exactly the sigma the candidate was drawn with.
__device__ inline float ChannelSigma(
  const DeviceMppi & config, const std::size_t channel, const float speed)
{
  const float sigma = config.sigma[channel];
  const bool steering_channel = channel == kSteering || channel == kRearSteering;
  if (!steering_channel || !config.speed_scaled_steering) {
    return sigma;
  }
  return sigma * fminf(fmaxf(
      config.steering_reference_speed / fmaxf(speed, 1.0F),
      config.steering_minimum_scale), 1.0F);
}

__global__ void BuildCandidates(
  const float * raw_noise, const Control * nominal, const Control * reference_controls,
  Control * candidates, Control * perturbations, const State initial_state,
  const DeviceMppi config)
{
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t count =
    static_cast<std::size_t>(config.samples) * config.horizon * kControlDim;
  if (index >= count) {
    return;
  }
  const std::size_t control_channel = index % kControlDim;
  const std::size_t time = (index / kControlDim) % config.horizon;
  const std::size_t sample = index / (static_cast<std::size_t>(config.horizon) * kControlDim);
  const std::uint16_t window = config.smoothing_window == 0U ? 1U : config.smoothing_window;
  const int left = static_cast<int>(window / 2U);
  const int right = static_cast<int>((window - 1U) / 2U);
  float sum = 0.0F;
  for (int offset = -left; offset <= right; ++offset) {
    const int source_time = static_cast<int>(time) + offset;
    if (source_time >= 0 && source_time < config.horizon) {
      const std::size_t source =
        (sample * config.horizon + static_cast<std::size_t>(source_time)) * kControlDim +
        control_channel;
      sum += raw_noise[source];
    }
  }
  const float sigma = ChannelSigma(config, control_channel, initial_state[kSpeed]);
  float epsilon = sigma * sum / sqrtf(static_cast<float>(window));
  if (time < config.delay_steps || (config.special_samples && sample == 0U)) {
    epsilon = 0.0F;
  } else if (config.special_samples && sample == 1U && config.use_reference_controls) {
    epsilon = reference_controls[time][control_channel] - nominal[time][control_channel];
  } else if (config.special_samples && sample == 2U) {
    epsilon = control_channel == kWheelTorque ?
      config.control_min[control_channel] - nominal[time][control_channel] : 0.0F;
  }
  const float candidate = fminf(fmaxf(
    nominal[time][control_channel] + epsilon,
    config.control_min[control_channel]), config.control_max[control_channel]);
  const std::size_t candidate_index = sample * config.horizon + time;
  candidates[candidate_index][control_channel] = candidate;
  perturbations[candidate_index][control_channel] = candidate - nominal[time][control_channel];
}

__global__ void RolloutAndCost(
  const State initial_state, const float initial_path_s, const Control previous_control,
  const Control * nominal, const Control * candidates, State * trajectories, float * costs,
  std::uint8_t * obstacle_latches, std::uint8_t * obstacle_brake_latches,
  const DeviceTrack track, const DeviceReference reference,
  const DeviceObstacleField obstacle_field,
  const DeviceMppi config, const DeviceCosts weights,
  const VehicleParameters parameters)
{
  const std::uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample >= config.samples) {
    return;
  }
  State state = initial_state;
  float s_hint = initial_path_s;
  FrenetView frenet = FrameView(state, s_hint, track, config);
  const std::size_t state_base = static_cast<std::size_t>(sample) * (config.horizon + 1U);
  const std::size_t control_base = static_cast<std::size_t>(sample) * config.horizon;
  trajectories[state_base] = state;
  float cost = 0.0F;
  bool crashed = false;
  bool excessive_sideslip = false;
  bool obstacle_latched = false;
  bool obstacle_brake_latched = false;
  float discount = 1.0F;
  Control prior = previous_control;
  for (std::uint16_t t = 0; t < config.horizon; ++t) {
    cost += StateCost<false>(
      state, frenet, reference.states[t], track, reference, obstacle_field,
      config, weights, crashed, excessive_sideslip, obstacle_latched,
      discount, t != 0U);
    if (t != 0U && t <= config.obstacle_brake_steps) {
      obstacle_brake_latched = obstacle_brake_latched || obstacle_latched;
    }
    discount *= weights.crash_discount;
    const Control control = candidates[control_base + t];
    for (std::size_t channel = 0; channel < kControlDim; ++channel) {
      cost += weights.control_effort[channel] * control[channel] * control[channel];
      const float delta = control[channel] - prior[channel];
      cost += weights.control_smoothness[channel] * delta * delta;
      const float rate = delta / config.dt;
      cost += weights.control_rate[channel] * rate * rate;
      const float sigma = ChannelSigma(config, channel, initial_state[kSpeed]);
      cost += config.gamma * nominal[t][channel] /
        fmaxf(sigma * sigma, 1.0e-12F) * (control[channel] - nominal[t][channel]);
    }
    prior = control;
    const float previous_speed = state[kSpeed];
    state = IntegrateAnalyticStep(state, control, track, config, parameters);
    const float acceleration = (state[kSpeed] - previous_speed) / config.dt;
    const float acceleration_weight = acceleration >= 0.0F ?
      weights.longitudinal_acceleration : weights.longitudinal_deceleration;
    cost += acceleration_weight * acceleration * acceleration;
    frenet = FrameView(state, s_hint, track, config);
    s_hint = frenet.path_evolution;
    trajectories[state_base + t + 1U] = state;
  }
  cost += StateCost<false>(
    state, frenet, reference.states[config.horizon], track, reference,
    obstacle_field, config, weights, crashed, excessive_sideslip,
    obstacle_latched, discount, true);
  if (config.horizon <= config.obstacle_brake_steps) {
    obstacle_brake_latched = obstacle_brake_latched || obstacle_latched;
  }
  cost -= weights.progress * (frenet.path_evolution - initial_path_s);
  costs[sample] = isfinite(cost) ? cost : CUDART_INF_F;
  obstacle_latches[sample] = obstacle_latched ? 1U : 0U;
  obstacle_brake_latches[sample] = obstacle_brake_latched ? 1U : 0U;
}

__global__ void InitializeNeuralRollouts(
  const State initial_state, const float initial_path_s, State * current_states,
  State * trajectories, float * costs, std::uint8_t * crashed,
  std::uint8_t * excessive_sideslip, std::uint8_t * obstacle_latched,
  std::uint8_t * obstacle_brake_latched, float * path_s_hint,
  const DeviceMppi config)
{
  const std::uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample >= config.samples) {
    return;
  }
  current_states[sample] = initial_state;
  path_s_hint[sample] = initial_path_s;
  trajectories[static_cast<std::size_t>(sample) * (config.horizon + 1U)] = initial_state;
  costs[sample] = 0.0F;
  crashed[sample] = 0U;
  excessive_sideslip[sample] = 0U;
  obstacle_latched[sample] = 0U;
  obstacle_brake_latched[sample] = 0U;
}

__global__ void NeuralStageCostAndPack(
  const State initial_state, const Control previous_control,
  const State * current_states, const Control * nominal, const Control * candidates,
  float * model_input, float * costs, std::uint8_t * crashed,
  std::uint8_t * excessive_sideslip, std::uint8_t * obstacle_latched,
  std::uint8_t * obstacle_brake_latched, float * path_s_hint,
  const std::uint16_t time_index, const DeviceTrack track,
  const DeviceReference reference, const DeviceObstacleField obstacle_field,
  const DeviceMppi config, const DeviceCosts weights)
{
  const std::uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample >= config.samples) {
    return;
  }
  const State state = current_states[sample];
  bool crash_latch = crashed[sample] != 0U;
  bool sideslip_latch = excessive_sideslip[sample] != 0U;
  bool obstacle_latch = obstacle_latched[sample] != 0U;
  const float discount = powf(weights.crash_discount, static_cast<float>(time_index));
  const FrenetView frenet = FrameView(state, path_s_hint[sample], track, config);
  path_s_hint[sample] = frenet.path_evolution;
  float cost = StateCost<false>(
    state, frenet, reference.states[time_index], track, reference,
    obstacle_field, config, weights, crash_latch, sideslip_latch,
    obstacle_latch, discount, time_index != 0U);
  if (time_index != 0U && time_index <= config.obstacle_brake_steps && obstacle_latch) {
    obstacle_brake_latched[sample] = 1U;
  }
  const std::size_t control_offset =
    static_cast<std::size_t>(sample) * config.horizon + time_index;
  const Control control = candidates[control_offset];
  const Control prior = time_index == 0U ? previous_control : candidates[control_offset - 1U];
  for (std::size_t channel = 0; channel < kControlDim; ++channel) {
    cost += weights.control_effort[channel] * control[channel] * control[channel];
    const float delta = control[channel] - prior[channel];
    cost += weights.control_smoothness[channel] * delta * delta;
    const float rate = delta / config.dt;
    cost += weights.control_rate[channel] * rate * rate;
    const float sigma = ChannelSigma(config, channel, initial_state[kSpeed]);
    cost += config.gamma * nominal[time_index][channel] /
      fmaxf(sigma * sigma, 1.0e-12F) *
      (control[channel] - nominal[time_index][channel]);
  }
  costs[sample] += cost;
  crashed[sample] = crash_latch ? 1U : 0U;
  excessive_sideslip[sample] = sideslip_latch ? 1U : 0U;
  obstacle_latched[sample] = obstacle_latch ? 1U : 0U;

  const std::size_t input_offset = static_cast<std::size_t>(sample) * 6U;
  for (std::size_t channel = 0; channel < kBodyStateDim; ++channel) {
    model_input[input_offset + channel] = state[channel];
  }
  model_input[input_offset + 4U] = control[kSteering];
  model_input[input_offset + 5U] = control[kWheelTorque];
}

__global__ void PackNeuralInputs(
  const State * current_states, const Control * candidates, float * model_input,
  const std::uint16_t time_index, const DeviceMppi config)
{
  const std::uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample >= config.samples) {
    return;
  }
  const State state = current_states[sample];
  const Control control = candidates[
    static_cast<std::size_t>(sample) * config.horizon + time_index];
  const std::size_t input_offset = static_cast<std::size_t>(sample) * 6U;
  for (std::size_t channel = 0; channel < kBodyStateDim; ++channel) {
    model_input[input_offset + channel] = state[channel];
  }
  model_input[input_offset + 4U] = control[kSteering];
  model_input[input_offset + 5U] = control[kWheelTorque];
}

__global__ void NeuralIntegrateAndStore(
  State * current_states, const float * body_derivatives,
  State * trajectories, float * costs, const std::uint16_t time_index,
  const float step_dt, const bool store_trajectory, const DeviceTrack track,
  const DeviceMppi config, const DeviceCosts weights)
{
  const std::uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample >= config.samples) {
    return;
  }
  State state = current_states[sample];
  BodyDerivative body_derivative{};
  const std::size_t derivative_offset = static_cast<std::size_t>(sample) * kBodyStateDim;
  for (std::size_t channel = 0; channel < kBodyStateDim; ++channel) {
    body_derivative[channel] = body_derivatives[derivative_offset + channel];
  }
  const auto derivative = config.frame == FrameKind::kCartesian ?
    CartesianDerivative(state, body_derivative) :
    FrenetDerivative(
      state, body_derivative, InterpolateTrackCurvature(state[kPathEvolution], track));
  for (std::size_t channel = 0; channel < kStateDim; ++channel) {
    state[channel] += step_dt * derivative[channel];
  }
  current_states[sample] = state;
  if (store_trajectory) {
    const std::size_t trajectory_offset =
      static_cast<std::size_t>(sample) * (config.horizon + 1U) + time_index + 1U;
    const float acceleration =
      (state[kSpeed] - trajectories[trajectory_offset - 1U][kSpeed]) / config.dt;
    const float acceleration_weight = acceleration >= 0.0F ?
      weights.longitudinal_acceleration : weights.longitudinal_deceleration;
    costs[sample] += acceleration_weight * acceleration * acceleration;
    trajectories[trajectory_offset] = state;
  }
}

__global__ void FinalizeNeuralCosts(
  const float initial_path_s, const State * current_states, float * costs,
  const std::uint8_t * crashed, const std::uint8_t * excessive_sideslip,
  std::uint8_t * obstacle_latched, std::uint8_t * obstacle_brake_latched,
  const float * path_s_hint,
  const DeviceTrack track, const DeviceReference reference,
  const DeviceObstacleField obstacle_field, const DeviceMppi config,
  const DeviceCosts weights)
{
  const std::uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample >= config.samples) {
    return;
  }
  const State state = current_states[sample];
  bool crash_latch = crashed[sample] != 0U;
  bool sideslip_latch = excessive_sideslip[sample] != 0U;
  bool obstacle_latch = obstacle_latched[sample] != 0U;
  const float discount = powf(weights.crash_discount, static_cast<float>(config.horizon));
  const FrenetView frenet = FrameView(state, path_s_hint[sample], track, config);
  float total = costs[sample] + StateCost<false>(
    state, frenet, reference.states[config.horizon], track, reference,
    obstacle_field, config, weights, crash_latch, sideslip_latch,
    obstacle_latch, discount, true);
  total -= weights.progress * (frenet.path_evolution - initial_path_s);
  costs[sample] = isfinite(total) ? total : CUDART_INF_F;
  obstacle_latched[sample] = obstacle_latch ? 1U : 0U;
  if (config.horizon <= config.obstacle_brake_steps && obstacle_latch) {
    obstacle_brake_latched[sample] = 1U;
  }
}

__global__ void PrepareReductionInputs(
  const float * costs, float * minimum_input, float * maximum_input,
  const std::uint8_t * obstacle_latches, std::uint32_t * finite_flags,
  std::uint32_t * obstacle_latch_flags, std::uint32_t * finite_unlatched_flags,
  const std::uint32_t count)
{
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    const bool finite = isfinite(costs[index]);
    minimum_input[index] = finite ? costs[index] : CUDART_INF_F;
    maximum_input[index] = finite ? costs[index] : -CUDART_INF_F;
    finite_flags[index] = finite ? 1U : 0U;
    const bool obstacle_latched = obstacle_latches[index] != 0U;
    obstacle_latch_flags[index] = obstacle_latched ? 1U : 0U;
    finite_unlatched_flags[index] = finite && !obstacle_latched ? 1U : 0U;
  }
}

__global__ void ComputeUnnormalizedWeights(
  float * costs, float * weights, const float * minimum, const float * worst,
  const DeviceMppi config)
{
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= config.samples) {
    return;
  }
  if (!isfinite(*minimum)) {
    costs[index] = 0.0F;
    weights[index] = 1.0F;
    return;
  }
  const float sanitized = isfinite(costs[index]) ? costs[index] : *worst;
  costs[index] = sanitized;
  const float value = expf(-(sanitized - *minimum) / fmaxf(config.lambda, 1.0e-6F));
  weights[index] = isfinite(value) ? value : 0.0F;
}

__global__ void NormalizeWeights(
  float * weights, float * squares, const float * sum, const std::uint32_t count)
{
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    const float normalized = *sum > 1.0F / static_cast<float>(count) ?
      weights[index] / *sum : 1.0F / static_cast<float>(count);
    weights[index] = normalized;
    squares[index] = normalized * normalized;
  }
}

__global__ void ComputeEss(const float * sum_squares, float * ess) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *ess = 1.0F / fmaxf(*sum_squares, 1.0e-3F);
  }
}

// The weighted averages below reduce over all K samples for each output
// element. One thread per output element leaves the horizon-sized grid far too
// small to fill the GPU, and each thread then walks the sample axis alone.
// Giving one block to each output element lets its threads stride over the
// samples, which raises the memory-level parallelism on the strided sample axis
// by the block size. The block size is fixed, so the reduction order, and
// therefore the floating-point result, is deterministic.
constexpr std::uint32_t kWeightedThreads = 128U;

__device__ float BlockSum(float value, float * scratch) {
  scratch[threadIdx.x] = value;
  __syncthreads();
  for (std::uint32_t span = kWeightedThreads / 2U; span > 0U; span >>= 1U) {
    if (threadIdx.x < span) {
      scratch[threadIdx.x] += scratch[threadIdx.x + span];
    }
    __syncthreads();
  }
  return scratch[0];
}

__global__ void WeightedControls(
  const float * __restrict__ weights, const Control * __restrict__ candidates,
  Control * __restrict__ updated, const DeviceMppi config)
{
  const std::size_t index = blockIdx.x;
  const std::size_t count = static_cast<std::size_t>(config.horizon) * kControlDim;
  if (index >= count) {
    return;
  }
  const std::size_t t = index / kControlDim;
  const std::size_t channel = index % kControlDim;
  float value = 0.0F;
  for (std::uint32_t sample = threadIdx.x; sample < config.samples;
    sample += kWeightedThreads)
  {
    value += weights[sample] *
      candidates[static_cast<std::size_t>(sample) * config.horizon + t][channel];
  }
  __shared__ float scratch[kWeightedThreads];
  value = BlockSum(value, scratch);
  if (threadIdx.x == 0U) {
    updated[t][channel] = fminf(fmaxf(
      value, config.control_min[channel]), config.control_max[channel]);
  }
}

// Angle channels are averaged on the unit circle. A linear mean of headings
// that straddle +/-pi collapses toward zero and drags the expected trajectory
// off in the wrong direction, which matters most for the Cartesian ENU yaw.
__device__ bool IsCircularChannel(const std::size_t channel, const DeviceMppi & config) {
  if (channel == kSideslip) {
    return true;
  }
  return config.frame == FrameKind::kCartesian ?
    channel == kHeadingEnu : channel == kRelativeHeading;
}

__global__ void WeightedStates(
  const float * __restrict__ weights, const State * __restrict__ trajectories,
  State * __restrict__ expected, const DeviceMppi config)
{
  const std::size_t index = blockIdx.x;
  const std::size_t count = static_cast<std::size_t>(config.horizon + 1U) * kStateDim;
  if (index >= count) {
    return;
  }
  const std::size_t t = index / kStateDim;
  const std::size_t channel = index % kStateDim;
  const bool circular = IsCircularChannel(channel, config);
  float value = 0.0F;
  float quadrature = 0.0F;
  for (std::uint32_t sample = threadIdx.x; sample < config.samples;
    sample += kWeightedThreads)
  {
    const float entry =
      trajectories[static_cast<std::size_t>(sample) * (config.horizon + 1U) + t][channel];
    const float weight = weights[sample];
    if (circular) {
      value += weight * cosf(entry);
      quadrature += weight * sinf(entry);
    } else {
      value += weight * entry;
    }
  }
  __shared__ float scratch[kWeightedThreads];
  value = BlockSum(value, scratch);
  if (circular) {
    __syncthreads();
    quadrature = BlockSum(quadrature, scratch);
  }
  if (threadIdx.x == 0U) {
    expected[t][channel] = circular ? atan2f(quadrature, value) : value;
  }
}

__global__ void RolloutUpdatedControls(
  const State initial_state, const Control * updated, State * expected,
  const DeviceTrack track, const DeviceMppi config,
  const VehicleParameters parameters)
{
  if (blockIdx.x != 0U || threadIdx.x != 0U) {
    return;
  }
  State state = initial_state;
  expected[0] = state;
  for (std::uint16_t t = 0; t < config.horizon; ++t) {
    state = IntegrateAnalyticStep(state, updated[t], track, config, parameters);
    expected[t + 1U] = state;
  }
}

// Decomposes the cost of an already-computed trajectory into its terms. It
// replays the same StateCost and control penalties as RolloutAndCost but does
// not integrate dynamics: states and controls are read back as given. That
// makes one kernel cover both the analytic and the TensorRT rollout paths, and
// it is why the launch is a single thread over the horizon rather than a second
// population rollout.
//
// nominal must be the nominal sequence the candidates were drawn around, not
// the shifted sequence ShiftNominal leaves behind, or the importance-sampling
// term is attributed against the wrong mean.
__global__ void EvaluateCostBreakdown(
  const State initial_state, const float initial_path_s, const Control previous_control,
  const Control * nominal, const State * states, const Control * controls,
  CostTerms * out, const DeviceTrack track, const DeviceReference reference,
  const DeviceObstacleField obstacle_field, const DeviceMppi config,
  const DeviceCosts weights)
{
  if (blockIdx.x != 0U || threadIdx.x != 0U) {
    return;
  }
  CostTerms terms{};
  terms.minimum_clearance_m = weights.obstacle_maximum_distance;
  bool crashed = false;
  bool excessive_sideslip = false;
  bool obstacle_latched = false;
  float discount = 1.0F;
  float s_hint = initial_path_s;
  Control prior = previous_control;
  FrenetView frenet = FrameView(states[0], s_hint, track, config);
  for (std::uint16_t t = 0; t < config.horizon; ++t) {
    (void)StateCost<true>(
      states[t], frenet, reference.states[t], track, reference, obstacle_field,
      config, weights, crashed, excessive_sideslip, obstacle_latched, discount,
      t != 0U, &terms, t);
    discount *= weights.crash_discount;
    const Control control = controls[t];
    for (std::size_t channel = 0; channel < kControlDim; ++channel) {
      const float effort =
        weights.control_effort[channel] * control[channel] * control[channel];
      terms.values[kTermControlEffort] += effort;
      terms.control_effort[channel] += effort;
      const float delta = control[channel] - prior[channel];
      const float smoothness = weights.control_smoothness[channel] * delta * delta;
      terms.values[kTermControlSmoothness] += smoothness;
      terms.control_smoothness[channel] += smoothness;
      const float rate = delta / config.dt;
      const float rate_penalty = weights.control_rate[channel] * rate * rate;
      terms.values[kTermControlRate] += rate_penalty;
      terms.control_rate[channel] += rate_penalty;
      const float sigma = ChannelSigma(config, channel, initial_state[kSpeed]);
      terms.values[kTermImportanceSampling] += config.gamma * nominal[t][channel] /
        fmaxf(sigma * sigma, 1.0e-12F) * (control[channel] - nominal[t][channel]);
    }
    prior = control;
    const float acceleration = (states[t + 1U][kSpeed] - states[t][kSpeed]) / config.dt;
    const bool speeding_up = acceleration >= 0.0F;
    const float penalty = (speeding_up ?
      weights.longitudinal_acceleration : weights.longitudinal_deceleration) *
      acceleration * acceleration;
    terms.values[speeding_up ?
      kTermLongitudinalAcceleration : kTermLongitudinalDeceleration] += penalty;
    frenet = FrameView(states[t + 1U], s_hint, track, config);
    s_hint = frenet.path_evolution;
  }
  (void)StateCost<true>(
    states[config.horizon], frenet, reference.states[config.horizon], track,
    reference, obstacle_field, config, weights, crashed, excessive_sideslip,
    obstacle_latched, discount, true, &terms, config.horizon);
  terms.values[kTermProgress] =
    -weights.progress * (frenet.path_evolution - initial_path_s);
  *out = terms;
}

// Adaptive-sigma statistics. Both accumulators are flat sums over every
// (sample, step) pair,
//   weighted[c]   = (1 / T)     * sum_{k,t} w_k * p[k][t][c]^2
//   unweighted[c] = (1 / (K T)) * sum_{k,t}       p[k][t][c]^2
// because the per-sample mean over t is linear and factors out. Summing over
// the flat index rather than per channel keeps the perturbation reads coalesced
// and spreads the K*T reduction over the whole GPU. Reducing per channel meant
// kControlDim threads carried the entire reduction, which dominated the solve.
//
// The grid is a fixed size, so for a given K and T the summation order, and
// therefore the floating-point result, is identical on every solve.
constexpr std::uint32_t kSigmaBlocks = 64U;
constexpr std::uint32_t kSigmaThreads = 256U;

__global__ void SigmaPartials(
  const float * __restrict__ weights, const Control * __restrict__ perturbations,
  float * __restrict__ partials, const DeviceMppi config)
{
  const std::size_t total =
    static_cast<std::size_t>(config.samples) * config.horizon;
  const std::size_t stride =
    static_cast<std::size_t>(gridDim.x) * blockDim.x;
  float weighted[kControlDim] = {};
  float unweighted[kControlDim] = {};
  for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    index < total; index += stride)
  {
    const float weight = weights[index / config.horizon];
    for (std::size_t channel = 0; channel < kControlDim; ++channel) {
      const float value = perturbations[index][channel];
      const float square = value * value;
      weighted[channel] += weight * square;
      unweighted[channel] += square;
    }
  }

  // Reduce all channel/statistic pairs together.  The former shared-memory
  // tree performed six full block reductions for 3 controls x 2 statistics,
  // including 54 block-wide barriers.  Warp shuffles need one barrier total
  // and retain the same fixed per-block partitioning used by SigmaFinalize.
  constexpr std::uint32_t kWarpSize = 32U;
  constexpr std::uint32_t kWarpCount = kSigmaThreads / kWarpSize;
  __shared__ float warp_sums[kWarpCount][kControlDim][2U];
  const std::uint32_t lane = threadIdx.x % kWarpSize;
  const std::uint32_t warp = threadIdx.x / kWarpSize;
  for (std::size_t channel = 0; channel < kControlDim; ++channel) {
    for (std::uint32_t offset = kWarpSize / 2U; offset > 0U; offset >>= 1U) {
      weighted[channel] += __shfl_down_sync(0xffffffffU, weighted[channel], offset);
      unweighted[channel] += __shfl_down_sync(0xffffffffU, unweighted[channel], offset);
    }
    if (lane == 0U) {
      warp_sums[warp][channel][0U] = weighted[channel];
      warp_sums[warp][channel][1U] = unweighted[channel];
    }
  }
  __syncthreads();

  if (warp == 0U) {
    for (std::size_t channel = 0; channel < kControlDim; ++channel) {
      float block_weighted = lane < kWarpCount ? warp_sums[lane][channel][0U] : 0.0F;
      float block_unweighted = lane < kWarpCount ? warp_sums[lane][channel][1U] : 0.0F;
      for (std::uint32_t offset = kWarpSize / 2U; offset > 0U; offset >>= 1U) {
        block_weighted += __shfl_down_sync(0xffffffffU, block_weighted, offset);
        block_unweighted += __shfl_down_sync(0xffffffffU, block_unweighted, offset);
      }
      if (lane == 0U) {
        const std::size_t base = (blockIdx.x * kControlDim + channel) * 2U;
        partials[base] = block_weighted;
        partials[base + 1U] = block_unweighted;
      }
    }
  }
}

__global__ void SigmaFinalize(
  const float * __restrict__ partials, float * __restrict__ sigma_hat,
  const DeviceMppi config)
{
  const std::size_t channel = threadIdx.x;
  if (channel >= kControlDim) {
    return;
  }
  float weighted = 0.0F;
  float unweighted = 0.0F;
  for (std::uint32_t block = 0; block < kSigmaBlocks; ++block) {
    const std::size_t base = (block * kControlDim + channel) * 2U;
    weighted += partials[base];
    unweighted += partials[base + 1U];
  }
  weighted /= static_cast<float>(config.horizon);
  unweighted /=
    static_cast<float>(config.samples) * static_cast<float>(config.horizon);
  sigma_hat[channel] = config.sigma[channel] *
    sqrtf(weighted + 1.0e-12F) / sqrtf(unweighted + 1.0e-12F);
}

// The refinement solves the multiple-shooting QP in increments.  Its Hessian
// approximation is block diagonal (state and input blocks at each stage), so
// the Schur complement is block tridiagonal in the dynamics multipliers.  This
// is the structure used by the native one-block PCG implementation below.
constexpr std::uint16_t kMaximumRefinementHorizon = 128U;
constexpr std::uint32_t kRefinementThreads = 256U;
constexpr std::size_t kLineSearchCandidates = kMpcLineSearchCandidates;

__device__ float RefinementStateCost(
  const State & state, const State & anchor, const std::uint16_t time,
  const float initial_path_s, const DeviceTrack track,
  const DeviceReference reference, const DeviceObstacleField obstacle_field,
  const DeviceMppi config, const DeviceCosts weights,
  const DeviceRefinement refinement)
{
  const float hint = config.frame == FrameKind::kFrenet ?
    state[kPathEvolution] : reference.s_grid[time];
  const FrenetView frenet = FrameView(state, hint, track, config);
  bool crashed = false;
  bool excessive_sideslip = false;
  bool obstacle_latched = false;
  float cost = StateCost<false>(
    state, frenet, reference.states[time], track, reference, obstacle_field,
    config, weights, crashed, excessive_sideslip, obstacle_latched,
    powf(weights.crash_discount, static_cast<float>(time)), false);
  for (std::size_t i = 0; i < kStateDim; ++i) {
    const float difference = state[i] - anchor[i];
    cost += refinement.state_proximity_weight * difference * difference;
  }
  if (time == config.horizon) {
    cost -= weights.progress * (frenet.path_evolution - initial_path_s);
  }
  return cost;
}

__global__ void LinearizeRefinementDynamics(
  const State * states, const Control * controls, float * dynamics_a,
  float * dynamics_b, State * defects, const std::uint8_t * active,
  const DeviceTrack track,
  const DeviceMppi config, const DeviceRefinement refinement,
  const VehicleParameters parameters)
{
  if (*active == 0U) {
    return;
  }
  const std::uint16_t time = static_cast<std::uint16_t>(blockIdx.x);
  if (time >= config.horizon) {
    return;
  }
  const State state = states[time];
  const Control control = controls[time];
  if (threadIdx.x == 0U) {
    const State predicted = IntegrateAnalyticStep(state, control, track, config, parameters);
    State residual{};
    for (std::size_t i = 0; i < kStateDim; ++i) {
      residual[i] = states[time + 1U][i] - predicted[i];
    }
    defects[time] = residual;
  }
  if (threadIdx.x < kStateDim) {
    const std::size_t column = threadIdx.x;
    const float scale = fmaxf(fabsf(state[column]), 1.0F);
    const float epsilon = refinement.finite_difference_relative_step * scale;
    State positive = state;
    State negative = state;
    positive[column] += epsilon;
    negative[column] -= epsilon;
    const State next_positive = IntegrateAnalyticStep(
      positive, control, track, config, parameters);
    const State next_negative = IntegrateAnalyticStep(
      negative, control, track, config, parameters);
    for (std::size_t row = 0; row < kStateDim; ++row) {
      dynamics_a[(static_cast<std::size_t>(time) * kStateDim + row) * kStateDim + column] =
        (next_positive[row] - next_negative[row]) / (2.0F * epsilon);
    }
  }
  if (threadIdx.x < kControlDim) {
    const std::size_t column = threadIdx.x;
    const float scale = fmaxf(fabsf(control[column]), 1.0F);
    const float epsilon = refinement.finite_difference_relative_step * scale;
    Control positive = control;
    Control negative = control;
    positive[column] += epsilon;
    negative[column] -= epsilon;
    const State next_positive = IntegrateAnalyticStep(
      state, positive, track, config, parameters);
    const State next_negative = IntegrateAnalyticStep(
      state, negative, track, config, parameters);
    for (std::size_t row = 0; row < kStateDim; ++row) {
      dynamics_b[(static_cast<std::size_t>(time) * kStateDim + row) * kControlDim + column] =
        (next_positive[row] - next_negative[row]) / (2.0F * epsilon);
    }
  }
}

__global__ void QuadraticizeRefinementCost(
  const State * states, const Control * controls, const State * anchor_states,
  const Control * anchor_controls, const Control previous_control,
  float * state_gradient, float * state_inverse_hessian,
  float * control_gradient, float * control_inverse_hessian,
  const std::uint8_t * active,
  const float initial_path_s, const DeviceTrack track,
  const DeviceReference reference, const DeviceObstacleField obstacle_field,
  const DeviceMppi config, const DeviceCosts weights,
  const DeviceRefinement refinement)
{
  if (*active == 0U) {
    return;
  }
  const std::size_t flat = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t state_count = static_cast<std::size_t>(config.horizon + 1U) * kStateDim;
  if (flat < state_count) {
    const std::uint16_t time = static_cast<std::uint16_t>(flat / kStateDim);
    const std::size_t channel = flat % kStateDim;
    const State center_state = states[time];
    const float scale = fmaxf(fabsf(center_state[channel]), 1.0F);
    const float epsilon = refinement.finite_difference_relative_step * scale;
    State positive = center_state;
    State negative = center_state;
    positive[channel] += epsilon;
    negative[channel] -= epsilon;
    const float center = RefinementStateCost(
      center_state, anchor_states[time], time, initial_path_s, track, reference,
      obstacle_field, config, weights, refinement);
    const float upper = RefinementStateCost(
      positive, anchor_states[time], time, initial_path_s, track, reference,
      obstacle_field, config, weights, refinement);
    const float lower = RefinementStateCost(
      negative, anchor_states[time], time, initial_path_s, track, reference,
      obstacle_field, config, weights, refinement);
    const float gradient = (upper - lower) / (2.0F * epsilon);
    const float hessian = fmaxf(
      (upper - 2.0F * center + lower) / (epsilon * epsilon),
      refinement.hessian_regularization);
    state_gradient[flat] = isfinite(gradient) ? gradient : 0.0F;
    state_inverse_hessian[flat] = 1.0F /
      (isfinite(hessian) ? hessian : refinement.hessian_regularization);
  }

  const std::size_t control_count = static_cast<std::size_t>(config.horizon) * kControlDim;
  if (flat < control_count) {
    const std::uint16_t time = static_cast<std::uint16_t>(flat / kControlDim);
    const std::size_t channel = flat % kControlDim;
    const float value = controls[time][channel];
    const float anchor = anchor_controls[time][channel];
    const float prior = time == 0U ? previous_control[channel] : controls[time - 1U][channel];
    const float pair_weight = weights.control_smoothness[channel] +
      weights.control_rate[channel] / (config.dt * config.dt);
    float gradient = 2.0F * weights.control_effort[channel] * value +
      2.0F * refinement.control_proximity_weight * (value - anchor) +
      2.0F * pair_weight * (value - prior);
    float hessian = 2.0F * (weights.control_effort[channel] +
      refinement.control_proximity_weight + pair_weight);
    if (time + 1U < config.horizon) {
      gradient += 2.0F * pair_weight * (value - controls[time + 1U][channel]);
      hessian += 2.0F * pair_weight;
    }
    hessian = fmaxf(hessian, refinement.hessian_regularization);
    control_gradient[flat] = isfinite(gradient) ? gradient : 0.0F;
    control_inverse_hessian[flat] = 1.0F /
      (isfinite(hessian) ? hessian : refinement.hessian_regularization);
  }
}

__device__ bool InvertRefinementBlock(const float * source, float * inverse) {
  float augmented[kStateDim][2U * kStateDim];
  for (std::size_t row = 0; row < kStateDim; ++row) {
    for (std::size_t column = 0; column < kStateDim; ++column) {
      augmented[row][column] = source[row * kStateDim + column];
      augmented[row][kStateDim + column] = row == column ? 1.0F : 0.0F;
    }
  }
  for (std::size_t pivot = 0; pivot < kStateDim; ++pivot) {
    std::size_t best = pivot;
    float magnitude = fabsf(augmented[pivot][pivot]);
    for (std::size_t row = pivot + 1U; row < kStateDim; ++row) {
      const float candidate = fabsf(augmented[row][pivot]);
      if (candidate > magnitude) {
        magnitude = candidate;
        best = row;
      }
    }
    if (!(magnitude > 1.0e-12F) || !isfinite(magnitude)) {
      return false;
    }
    if (best != pivot) {
      for (std::size_t column = 0; column < 2U * kStateDim; ++column) {
        const float temporary = augmented[pivot][column];
        augmented[pivot][column] = augmented[best][column];
        augmented[best][column] = temporary;
      }
    }
    const float divisor = augmented[pivot][pivot];
    for (std::size_t column = 0; column < 2U * kStateDim; ++column) {
      augmented[pivot][column] /= divisor;
    }
    for (std::size_t row = 0; row < kStateDim; ++row) {
      if (row == pivot) {
        continue;
      }
      const float factor = augmented[row][pivot];
      for (std::size_t column = 0; column < 2U * kStateDim; ++column) {
        augmented[row][column] -= factor * augmented[pivot][column];
      }
    }
  }
  for (std::size_t row = 0; row < kStateDim; ++row) {
    for (std::size_t column = 0; column < kStateDim; ++column) {
      inverse[row * kStateDim + column] = augmented[row][kStateDim + column];
    }
  }
  return true;
}

__global__ void BuildRefinementSchur(
  const float * dynamics_a, const float * dynamics_b, const State * defects,
  const float * state_gradient, const float * state_inverse_hessian,
  const float * control_gradient, const float * control_inverse_hessian,
  float * diagonal, float * upper, float * inverse_diagonal, float * rhs,
  const std::uint8_t * active,
  const DeviceMppi config, const DeviceRefinement refinement)
{
  if (*active == 0U) {
    return;
  }
  const std::uint16_t row_index = static_cast<std::uint16_t>(blockIdx.x);
  if (row_index > config.horizon || threadIdx.x != 0U) {
    return;
  }
  float block[kStateDim * kStateDim]{};
  float row_rhs[kStateDim]{};
  if (row_index == 0U) {
    for (std::size_t i = 0; i < kStateDim; ++i) {
      block[i * kStateDim + i] = state_inverse_hessian[i];
      row_rhs[i] = -state_inverse_hessian[i] * state_gradient[i];
    }
  } else {
    const std::uint16_t time = static_cast<std::uint16_t>(row_index - 1U);
    const float * a = dynamics_a + static_cast<std::size_t>(time) * kStateDim * kStateDim;
    const float * b = dynamics_b + static_cast<std::size_t>(time) * kStateDim * kControlDim;
    const std::size_t state_offset = static_cast<std::size_t>(time) * kStateDim;
    const std::size_t next_state_offset = static_cast<std::size_t>(time + 1U) * kStateDim;
    const std::size_t control_offset = static_cast<std::size_t>(time) * kControlDim;
    for (std::size_t i = 0; i < kStateDim; ++i) {
      for (std::size_t j = 0; j < kStateDim; ++j) {
        float value = i == j ? state_inverse_hessian[next_state_offset + i] : 0.0F;
        for (std::size_t k = 0; k < kStateDim; ++k) {
          value += a[i * kStateDim + k] * state_inverse_hessian[state_offset + k] *
            a[j * kStateDim + k];
        }
        for (std::size_t k = 0; k < kControlDim; ++k) {
          value += b[i * kControlDim + k] * control_inverse_hessian[control_offset + k] *
            b[j * kControlDim + k];
        }
        block[i * kStateDim + j] = value;
      }
      float h_g_inverse = state_inverse_hessian[next_state_offset + i] *
        state_gradient[next_state_offset + i];
      for (std::size_t k = 0; k < kStateDim; ++k) {
        h_g_inverse -= a[i * kStateDim + k] * state_inverse_hessian[state_offset + k] *
          state_gradient[state_offset + k];
      }
      for (std::size_t k = 0; k < kControlDim; ++k) {
        h_g_inverse -= b[i * kControlDim + k] * control_inverse_hessian[control_offset + k] *
          control_gradient[control_offset + k];
      }
      row_rhs[i] = defects[time][i] - h_g_inverse;
    }
  }
  for (std::size_t i = 0; i < kStateDim; ++i) {
    block[i * kStateDim + i] += refinement.hessian_regularization;
  }
  float inverse[kStateDim * kStateDim]{};
  const bool invertible = InvertRefinementBlock(block, inverse);
  const std::size_t block_offset = static_cast<std::size_t>(row_index) * kStateDim * kStateDim;
  for (std::size_t i = 0; i < kStateDim * kStateDim; ++i) {
    diagonal[block_offset + i] = block[i];
    inverse_diagonal[block_offset + i] = invertible ? inverse[i] :
      ((i / kStateDim == i % kStateDim) ? 1.0F / refinement.hessian_regularization : 0.0F);
  }
  for (std::size_t i = 0; i < kStateDim; ++i) {
    rhs[static_cast<std::size_t>(row_index) * kStateDim + i] = row_rhs[i];
  }
  if (row_index < config.horizon) {
    const float * a = dynamics_a + static_cast<std::size_t>(row_index) * kStateDim * kStateDim;
    const std::size_t state_offset = static_cast<std::size_t>(row_index) * kStateDim;
    float * destination = upper + static_cast<std::size_t>(row_index) * kStateDim * kStateDim;
    for (std::size_t i = 0; i < kStateDim; ++i) {
      for (std::size_t j = 0; j < kStateDim; ++j) {
        destination[i * kStateDim + j] =
          -state_inverse_hessian[state_offset + i] * a[j * kStateDim + i];
      }
    }
  }
}

__device__ float RefinementBlockDot(float value, float * scratch) {
  // Reduce within each warp first, then reduce the (at most eight) warp sums.
  // The former tree reduction synchronized the full block eight times for each
  // dot product, twice per PCG iteration. This needs only two block barriers.
  constexpr std::uint32_t kWarpSize = 32U;
  const std::uint32_t lane = threadIdx.x & (kWarpSize - 1U);
  const std::uint32_t warp = threadIdx.x / kWarpSize;
  for (std::uint32_t offset = kWarpSize / 2U; offset > 0U; offset >>= 1U) {
    value += __shfl_down_sync(0xffffffffU, value, offset);
  }
  if (lane == 0U) {
    scratch[warp] = value;
  }
  __syncthreads();
  if (warp == 0U) {
    const std::uint32_t warp_count = (blockDim.x + kWarpSize - 1U) / kWarpSize;
    value = lane < warp_count ? scratch[lane] : 0.0F;
    for (std::uint32_t offset = kWarpSize / 2U; offset > 0U; offset >>= 1U) {
      value += __shfl_down_sync(0xffffffffU, value, offset);
    }
    if (lane == 0U) {
      scratch[0] = value;
    }
  }
  __syncthreads();
  return scratch[0];
}

__device__ float RefinementSchurEntry(
  const std::size_t flat, const float * vector, const float * diagonal,
  const float * upper, const std::size_t block_count)
{
  const std::size_t row = flat / kStateDim;
  const std::size_t component = flat % kStateDim;
  const float * center = diagonal + row * kStateDim * kStateDim;
  float value = 0.0F;
  for (std::size_t column = 0; column < kStateDim; ++column) {
    value += center[component * kStateDim + column] *
      vector[row * kStateDim + column];
  }
  if (row + 1U < block_count) {
    const float * next = upper + row * kStateDim * kStateDim;
    for (std::size_t column = 0; column < kStateDim; ++column) {
      value += next[component * kStateDim + column] *
        vector[(row + 1U) * kStateDim + column];
    }
  }
  if (row > 0U) {
    const float * previous = upper + (row - 1U) * kStateDim * kStateDim;
    for (std::size_t column = 0; column < kStateDim; ++column) {
      value += previous[column * kStateDim + component] *
        vector[(row - 1U) * kStateDim + column];
    }
  }
  return value;
}

__global__ void SolveRefinementPcg(
  const float * diagonal, const float * upper, const float * inverse_diagonal,
  const float * rhs, float * dual, std::uint16_t * iterations,
  const std::uint8_t * active,
  const DeviceMppi config, const DeviceRefinement refinement)
{
  if (*active == 0U) {
    return;
  }
  extern __shared__ float shared[];
  const std::size_t count = static_cast<std::size_t>(config.horizon + 1U) * kStateDim;
  float * solution = shared;
  float * residual = solution + count;
  float * preconditioned = residual + count;
  float * direction = preconditioned + count;
  float * product = direction + count;
  float * scratch = product + count;
  for (std::size_t index = threadIdx.x; index < count; index += blockDim.x) {
    solution[index] = dual[index];
  }
  __syncthreads();
  for (std::size_t index = threadIdx.x; index < count; index += blockDim.x) {
    residual[index] = rhs[index] - RefinementSchurEntry(
      index, solution, diagonal, upper, config.horizon + 1U);
  }
  __syncthreads();
  for (std::size_t index = threadIdx.x; index < count; index += blockDim.x) {
    const std::size_t row = index / kStateDim;
    const std::size_t component = index % kStateDim;
    const float * inverse = inverse_diagonal + row * kStateDim * kStateDim;
    float value = 0.0F;
    for (std::size_t column = 0; column < kStateDim; ++column) {
      value += inverse[component * kStateDim + column] *
        residual[row * kStateDim + column];
    }
    preconditioned[index] = value;
    direction[index] = value;
  }
  __syncthreads();
  float partial = 0.0F;
  for (std::size_t index = threadIdx.x; index < count; index += blockDim.x) {
    partial += residual[index] * preconditioned[index];
  }
  float rz = RefinementBlockDot(partial, scratch);
  const float convergence_threshold = 1.0e-12F +
    refinement.pcg_tolerance * refinement.pcg_tolerance * fabsf(rz);
  std::uint16_t completed = 0U;
  for (std::uint16_t iteration = 0U; iteration < refinement.pcg_iterations; ++iteration) {
    if (!isfinite(rz) || fabsf(rz) <= convergence_threshold) {
      break;
    }
    for (std::size_t index = threadIdx.x; index < count; index += blockDim.x) {
      product[index] = RefinementSchurEntry(
        index, direction, diagonal, upper, config.horizon + 1U);
    }
    __syncthreads();
    partial = 0.0F;
    for (std::size_t index = threadIdx.x; index < count; index += blockDim.x) {
      partial += direction[index] * product[index];
    }
    const float denominator = RefinementBlockDot(partial, scratch);
    if (!isfinite(denominator) || fabsf(denominator) < 1.0e-20F) {
      break;
    }
    const float alpha = rz / denominator;
    for (std::size_t index = threadIdx.x; index < count; index += blockDim.x) {
      solution[index] += alpha * direction[index];
      residual[index] -= alpha * product[index];
    }
    __syncthreads();
    for (std::size_t index = threadIdx.x; index < count; index += blockDim.x) {
      const std::size_t row = index / kStateDim;
      const std::size_t component = index % kStateDim;
      const float * inverse = inverse_diagonal + row * kStateDim * kStateDim;
      float value = 0.0F;
      for (std::size_t column = 0; column < kStateDim; ++column) {
        value += inverse[component * kStateDim + column] *
          residual[row * kStateDim + column];
      }
      preconditioned[index] = value;
    }
    __syncthreads();
    partial = 0.0F;
    for (std::size_t index = threadIdx.x; index < count; index += blockDim.x) {
      partial += residual[index] * preconditioned[index];
    }
    const float next_rz = RefinementBlockDot(partial, scratch);
    const float beta = next_rz / fmaxf(rz, 1.0e-20F);
    for (std::size_t index = threadIdx.x; index < count; index += blockDim.x) {
      direction[index] = preconditioned[index] + beta * direction[index];
    }
    __syncthreads();
    rz = next_rz;
    completed = static_cast<std::uint16_t>(iteration + 1U);
  }
  for (std::size_t index = threadIdx.x; index < count; index += blockDim.x) {
    dual[index] = isfinite(solution[index]) ? solution[index] : 0.0F;
  }
  if (threadIdx.x == 0U) {
    *iterations = completed;
  }
}

__global__ void RecoverRefinementStep(
  const float * dynamics_a, const float * dynamics_b,
  const float * state_gradient, const float * state_inverse_hessian,
  const float * control_gradient, const float * control_inverse_hessian,
  const float * dual, State * state_step, Control * control_step,
  const std::uint8_t * active,
  const DeviceMppi config, const DeviceRefinement refinement)
{
  if (*active == 0U) {
    return;
  }
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t state_count = static_cast<std::size_t>(config.horizon + 1U) * kStateDim;
  if (index < state_count) {
    const std::uint16_t time = static_cast<std::uint16_t>(index / kStateDim);
    const std::size_t channel = index % kStateDim;
    float stationarity = state_gradient[index] + dual[index];
    if (time < config.horizon) {
      const float * a = dynamics_a + static_cast<std::size_t>(time) * kStateDim * kStateDim;
      for (std::size_t row = 0; row < kStateDim; ++row) {
        stationarity -= a[row * kStateDim + channel] *
          dual[static_cast<std::size_t>(time + 1U) * kStateDim + row];
      }
    }
    float step = -state_inverse_hessian[index] * stationarity;
    step = fminf(fmaxf(step, -refinement.maximum_state_step), refinement.maximum_state_step);
    if (time == 0U) {
      step = 0.0F;
    }
    state_step[time][channel] = isfinite(step) ? step : 0.0F;
  }
  const std::size_t control_count = static_cast<std::size_t>(config.horizon) * kControlDim;
  if (index < control_count) {
    const std::uint16_t time = static_cast<std::uint16_t>(index / kControlDim);
    const std::size_t channel = index % kControlDim;
    const float * b = dynamics_b + static_cast<std::size_t>(time) * kStateDim * kControlDim;
    float stationarity = control_gradient[index];
    for (std::size_t row = 0; row < kStateDim; ++row) {
      stationarity -= b[row * kControlDim + channel] *
        dual[static_cast<std::size_t>(time + 1U) * kStateDim + row];
    }
    float step = -control_inverse_hessian[index] * stationarity;
    const float limit = refinement.maximum_control_step[channel];
    step = fminf(fmaxf(step, -limit), limit);
    if (time < config.delay_steps) {
      step = 0.0F;
    }
    control_step[time][channel] = isfinite(step) ? step : 0.0F;
  }
}

__device__ float RefinementAlpha(const std::size_t candidate) {
  constexpr float values[kLineSearchCandidates] = {0.0F, 1.0F, 0.7F, 0.3F, 0.1F, 0.03F, 0.01F};
  return values[candidate];
}

__global__ void DisableRefinementWhenAllObstacleLatched(
  const std::uint32_t * obstacle_latch_count, std::uint8_t * active,
  const DeviceMppi config)
{
  if (blockIdx.x == 0U && threadIdx.x == 0U &&
    *obstacle_latch_count == config.samples)
  {
    *active = 0U;
  }
}

__global__ void EvaluateRefinementLineSearch(
  const State initial_state, const float initial_path_s,
  const Control previous_control, const State * states, const Control * controls,
  const Control * control_step,
  const State * anchor_states, const Control * anchor_controls,
  State * candidate_states, Control * candidate_controls,
  RefinementMetrics * metrics, const std::uint8_t * active,
  const std::uint16_t sqp_iteration, const DeviceTrack track,
  const DeviceReference reference, const DeviceObstacleField obstacle_field,
  const DeviceMppi config, const DeviceCosts weights,
  const DeviceRefinement refinement, const VehicleParameters parameters)
{
  const std::size_t candidate = blockIdx.x;
  if (*active == 0U || candidate >= kLineSearchCandidates || threadIdx.x != 0U) {
    return;
  }
  const float alpha = RefinementAlpha(candidate);
  State * candidate_state = candidate_states + candidate * (config.horizon + 1U);
  Control * candidate_control = candidate_controls + candidate * config.horizon;
  candidate_state[0] = initial_state;
  Control prior = previous_control;
  for (std::uint16_t time = 0U; time < config.horizon; ++time) {
    Control value = controls[time];
    // Candidate zero is the exact current SQP primal and defines the line-search
    // baseline. Projecting it here would silently change MPPI's controls while
    // leaving its warm-start states untouched, manufacturing a dynamics defect.
    if (candidate != 0U) {
      for (std::size_t channel = 0; channel < kControlDim; ++channel) {
        value[channel] += alpha * control_step[time][channel];
        value[channel] = fminf(fmaxf(
          value[channel], config.control_min[channel]), config.control_max[channel]);
        const float rate_delta = refinement.maximum_control_rate[channel] * config.dt;
        value[channel] = fminf(fmaxf(
          value[channel], prior[channel] - rate_delta), prior[channel] + rate_delta);
        if (time < config.delay_steps) {
          value[channel] = anchor_controls[time][channel];
        }
      }
    }
    candidate_control[time] = value;
    prior = value;
  }

  if (candidate == 0U) {
    for (std::uint16_t time = 1U; time <= config.horizon; ++time) {
      candidate_state[time] = states[time];
    }
  } else {
    // The MPPI expected trajectory is accepted directly as the SQP warm start.
    // Only trial controls are rolled out here: this is the nonlinear
    // feasibility restoration used by line search, and the seven trials run in
    // parallel CUDA blocks.
    for (std::uint16_t time = 0U; time < config.horizon; ++time) {
      candidate_state[time + 1U] = IntegrateAnalyticStep(
        candidate_state[time], candidate_control[time], track, config, parameters);
    }
  }

  bool crashed = false;
  bool excessive_sideslip = false;
  bool obstacle_latched = false;
  float cost = 0.0F;
  float residual_l1 = 0.0F;
  float maximum_residual = 0.0F;
  float discount = 1.0F;
  float s_hint = initial_path_s;
  prior = previous_control;
  for (std::uint16_t time = 0U; time < config.horizon; ++time) {
    const FrenetView frenet = FrameView(candidate_state[time], s_hint, track, config);
    s_hint = frenet.path_evolution;
    cost += StateCost<false>(
      candidate_state[time], frenet, reference.states[time], track, reference,
      obstacle_field, config, weights, crashed, excessive_sideslip,
      obstacle_latched, discount, time != 0U);
    discount *= weights.crash_discount;
    for (std::size_t channel = 0; channel < kStateDim; ++channel) {
      const float difference = candidate_state[time][channel] - anchor_states[time][channel];
      cost += refinement.state_proximity_weight * difference * difference;
    }
    const Control control = candidate_control[time];
    for (std::size_t channel = 0; channel < kControlDim; ++channel) {
      cost += weights.control_effort[channel] * control[channel] * control[channel];
      const float delta = control[channel] - prior[channel];
      cost += weights.control_smoothness[channel] * delta * delta;
      const float rate = delta / config.dt;
      cost += weights.control_rate[channel] * rate * rate;
      const float anchor_difference = control[channel] - anchor_controls[time][channel];
      cost += refinement.control_proximity_weight * anchor_difference * anchor_difference;
    }
    prior = control;
    const float acceleration =
      (candidate_state[time + 1U][kSpeed] - candidate_state[time][kSpeed]) / config.dt;
    cost += (acceleration >= 0.0F ? weights.longitudinal_acceleration :
      weights.longitudinal_deceleration) * acceleration * acceleration;
    // Nonzero line-search candidates were generated immediately above by the
    // exact same nonlinear integration, so their dynamics residual is exactly
    // zero by construction.  Re-integrating all six feasible candidates here
    // doubled the dominant work of each SQP line search.  Only candidate zero,
    // whose multiple-shooting states may be infeasible, needs the residual.
    if (candidate == 0U && sqp_iteration == 0U) {
      const State predicted = IntegrateAnalyticStep(
        candidate_state[time], control, track, config, parameters);
      for (std::size_t channel = 0; channel < kStateDim; ++channel) {
        const float residual = fabsf(candidate_state[time + 1U][channel] - predicted[channel]);
        residual_l1 += residual;
        maximum_residual = fmaxf(maximum_residual, residual);
      }
    }
  }
  const FrenetView terminal_frenet = FrameView(
    candidate_state[config.horizon], s_hint, track, config);
  cost += StateCost<false>(
    candidate_state[config.horizon], terminal_frenet,
    reference.states[config.horizon], track, reference, obstacle_field,
    config, weights, crashed, excessive_sideslip, obstacle_latched, discount, true);
  for (std::size_t channel = 0; channel < kStateDim; ++channel) {
    const float difference = candidate_state[config.horizon][channel] -
      anchor_states[config.horizon][channel];
    cost += refinement.state_proximity_weight * difference * difference;
  }
  cost -= weights.progress * (terminal_frenet.path_evolution - initial_path_s);
  const bool finite = isfinite(cost) && isfinite(residual_l1) && isfinite(maximum_residual);
  metrics[candidate] = RefinementMetrics{
    finite ? cost : CUDART_INF_F,
    finite ? cost + refinement.merit_constraint_penalty * residual_l1 : CUDART_INF_F,
    finite ? maximum_residual : CUDART_INF_F,
    static_cast<std::uint8_t>(finite && !crashed && !excessive_sideslip && !obstacle_latched)};
}

__global__ void SelectRefinementLineSearch(
  const RefinementMetrics * metrics, std::uint32_t * selected,
  RefinementMetrics * initial_metrics, RefinementMetrics * final_metrics,
  std::uint8_t * active, std::uint16_t * completed_iterations,
  const std::uint16_t sqp_iteration)
{
  if (*active == 0U || blockIdx.x != 0U || threadIdx.x != 0U) {
    return;
  }
  *initial_metrics = metrics[0];
  std::uint32_t choice = 0U;
  for (std::uint32_t candidate = 1U; candidate < kLineSearchCandidates; ++candidate) {
    // All trials have already been evaluated, so choose the lowest-merit safe
    // trajectory rather than the first trial that happens to improve. The old
    // backtracking-style choice accepted alpha=1 for any tiny decrease even
    // when alpha=0.7 or 0.3 was substantially better, and it could advance the
    // next SQP linearization through an unsafe intermediate trajectory.
    if (metrics[candidate].safe != 0U &&
      metrics[candidate].merit < metrics[choice].merit)
    {
      choice = candidate;
    }
  }
  *selected = choice;
  *final_metrics = metrics[choice];
  *completed_iterations = static_cast<std::uint16_t>(sqp_iteration + 1U);
  // A rejected step leaves the primal unchanged. Re-linearizing the identical
  // point cannot improve it deterministically, so the remaining queued SQP
  // kernels can return immediately instead of burning the configured maximum.
  if (choice == 0U) {
    *active = 0U;
  }
}

__global__ void ApplyRefinementLineSearch(
  State * states, Control * controls, const State * candidate_states,
  const Control * candidate_controls, const std::uint32_t * selected,
  const std::uint8_t * active,
  const DeviceMppi config)
{
  if (*active == 0U) {
    return;
  }
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t state_count = static_cast<std::size_t>(config.horizon + 1U);
  if (index < state_count) {
    states[index] = candidate_states[
      static_cast<std::size_t>(*selected) * state_count + index];
  }
  if (index < config.horizon) {
    controls[index] = candidate_controls[
      static_cast<std::size_t>(*selected) * config.horizon + index];
  }
}

__global__ void CommitRefinement(
  const State * refined_states, const Control * refined_controls,
  State * expected, Control * updated, const RefinementMetrics * initial_metrics,
  const RefinementMetrics * final_metrics, std::uint8_t * accepted,
  const std::uint32_t * obstacle_latch_count,
  const DeviceMppi config, const DeviceRefinement refinement)
{
  if (*obstacle_latch_count == config.samples) {
    if (blockIdx.x == 0U && threadIdx.x == 0U) {
      *accepted = 0U;
    }
    return;
  }
  const bool valid = final_metrics->safe != 0U &&
    isfinite(final_metrics->cost) && isfinite(final_metrics->merit) &&
    final_metrics->merit <= initial_metrics->merit &&
    final_metrics->maximum_dynamics_residual <= refinement.constraint_tolerance;
  if (blockIdx.x == 0U && threadIdx.x == 0U) {
    *accepted = valid ? 1U : 0U;
  }
  if (!valid) {
    return;
  }
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index <= config.horizon) {
    expected[index] = refined_states[index];
  }
  if (index < config.horizon) {
    updated[index] = refined_controls[index];
  }
}

__global__ void MeasureRefinementDelta(
  const State * anchor_states, const Control * anchor_controls,
  const State * published_states, const Control * published_controls,
  const std::uint8_t * accepted, RefinementDelta * delta,
  const DeviceMppi config)
{
  if (blockIdx.x != 0U || threadIdx.x != 0U) {
    return;
  }
  RefinementDelta result{};
  if (*accepted != 0U) {
    for (std::uint16_t time = 0U; time <= config.horizon; ++time) {
      for (std::size_t channel = 0U; channel < kStateDim; ++channel) {
        const float difference = fabsf(
          published_states[time][channel] - anchor_states[time][channel]);
        result.state_rms[channel] += difference * difference;
        result.state_max[channel] = fmaxf(result.state_max[channel], difference);
      }
    }
    for (std::uint16_t time = 0U; time < config.horizon; ++time) {
      for (std::size_t channel = 0U; channel < kControlDim; ++channel) {
        const float difference = fabsf(
          published_controls[time][channel] - anchor_controls[time][channel]);
        result.control_rms[channel] += difference * difference;
        result.control_max[channel] = fmaxf(result.control_max[channel], difference);
      }
    }
    for (std::size_t channel = 0U; channel < kControlDim; ++channel) {
      result.first_control[channel] = fabsf(
        published_controls[0][channel] - anchor_controls[0][channel]);
    }
    for (std::size_t channel = 0U; channel < kStateDim; ++channel) {
      result.state_rms[channel] = sqrtf(
        result.state_rms[channel] / static_cast<float>(config.horizon + 1U));
    }
    for (std::size_t channel = 0U; channel < kControlDim; ++channel) {
      result.control_rms[channel] = sqrtf(
        result.control_rms[channel] / static_cast<float>(config.horizon));
    }
  }
  *delta = result;
}

__global__ void ShiftNominal(
  const Control * updated, Control * nominal, const float shift_fraction,
  const DeviceMppi config)
{
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t count = static_cast<std::size_t>(config.horizon) * kControlDim;
  if (index >= count) {
    return;
  }
  const std::size_t t = index / kControlDim;
  const std::size_t channel = index % kControlDim;
  const float query = static_cast<float>(t) + fmaxf(shift_fraction, 0.0F);
  const std::size_t last = static_cast<std::size_t>(config.horizon - 1U);
  const std::size_t requested_low = static_cast<std::size_t>(floorf(query));
  const std::size_t low = requested_low < last ? requested_low : last;
  const std::size_t high = low + 1U < last ? low + 1U : last;
  const float fraction = fminf(fmaxf(query - static_cast<float>(low), 0.0F), 1.0F);
  nominal[t][channel] =
    (1.0F - fraction) * updated[low][channel] + fraction * updated[high][channel];
}

template<typename T>
void Allocate(T *& pointer, const std::size_t count) {
  CheckCuda(cudaMalloc(reinterpret_cast<void **>(&pointer), count * sizeof(T)), "cudaMalloc");
}

template<typename T>
void Free(T *& pointer) noexcept {
  if (pointer != nullptr) {
    cudaFree(pointer);
    pointer = nullptr;
  }
}

DeviceCosts ToDeviceCosts(
  const CostWeights & source, const ObstacleConfig & obstacles)
{
  DeviceCosts result{};
  for (std::size_t i = 0; i < kStateDim; ++i) {
    result.reference_tracking[i] = source.reference_tracking[i];
  }
  for (std::size_t i = 0; i < kControlDim; ++i) {
    result.control_effort[i] = source.control_effort[i];
    result.control_smoothness[i] = source.control_smoothness[i];
    result.control_rate[i] = source.control_rate[i];
  }
  result.longitudinal_acceleration = source.longitudinal_acceleration;
  result.longitudinal_deceleration = source.longitudinal_deceleration;
  result.velocity_profile = source.velocity_profile;
  result.velocity_overspeed_multiplier = source.velocity_overspeed_multiplier;
  result.progress = source.progress;
  result.boundary = source.boundary;
  result.boundary_margin = source.boundary_margin_m;
  result.crash = source.crash;
  result.crash_discount = source.crash_discount;
  result.crash_buffer = source.crash_buffer_m;
  result.sideslip = source.sideslip;
  result.maximum_sideslip = source.maximum_sideslip_rad;
  result.sideslip_kill = source.sideslip_kill;
  result.lateral_damping = source.lateral_damping;
  result.lateral_decay_rate = source.lateral_decay_rate;
  result.wheel_slip = source.wheel_slip;
  result.wheel_slip_band = source.wheel_slip_band;
  result.obstacle_distance = obstacles.distance_weight;
  result.obstacle_influence_distance = obstacles.influence_distance_m;
  result.obstacle_latch_threshold = obstacles.latch_threshold_m;
  result.obstacle_latching = obstacles.latching_weight;
  result.obstacle_maximum_distance = obstacles.maximum_distance_m;
  result.footprint_length = obstacles.footprint_length_m;
  result.footprint_width = obstacles.footprint_width_m;
  result.footprint_circles = obstacles.footprint_circles;
  result.obstacle_enabled = obstacles.enabled;
  return result;
}

}  // namespace

class CudaMppiController::Impl {
 private:
  enum class VisualizationSlotState : std::uint8_t {kFree, kReady, kReading};

  struct VisualizationSlot {
    float * weights{nullptr};
    State * trajectories{nullptr};
    cudaEvent_t ready_event{nullptr};
    VisualizationSlotState state{VisualizationSlotState::kFree};
    std::uint64_t id{};
    std::uint32_t rollout_count{};
  };

  static constexpr std::size_t kVisualizationSlotCount = 2U;

 public:
  Impl(
    MppiConfig config, const CostWeights & costs, VehicleParameters vehicle,
    ObstacleConfig obstacle_config,
    const ModelKind model_kind, const Raceline & raceline,
    const IntegratorKind integrator_kind, std::string neural_engine_path,
    const float cartesian_projection_window_m)
  : config_(std::move(config)), vehicle_(vehicle), obstacle_config_(obstacle_config),
    costs_(ToDeviceCosts(costs, obstacle_config)),
    base_sigma_(config_.sigma)
  {
    if (!(cartesian_projection_window_m > 0.0F)) {
      throw std::invalid_argument("projection window must be positive");
    }
    if (config_.num_samples <= 3U || config_.horizon == 0U ||
      config_.horizon == std::numeric_limits<std::uint16_t>::max() ||
      config_.obstacle_latch_brake_steps == 0U ||
      config_.obstacle_latch_brake_steps > config_.horizon ||
      !(config_.dt_s > 0.0F) || !(config_.lambda > 0.0F) ||
      raceline.points().size() < 2U)
    {
      throw std::invalid_argument(
              "invalid MPPI dimensions/timing, obstacle brake window, or empty raceline");
    }
    if (model_kind == ModelKind::kTensorRtNeuralDerivative && neural_engine_path.empty()) {
      throw std::invalid_argument("neural dynamics requires a TensorRT engine path");
    }
    if (model_kind == ModelKind::kTensorRtNeuralDerivative &&
      integrator_kind != IntegratorKind::kEuler)
    {
      throw std::invalid_argument(
        "TensorRT neural dynamics currently supports Euler integration; "
        "RK4 would require four TensorRT evaluations per substep");
    }
    if (model_kind == ModelKind::kTensorRtNeuralDerivative &&
      !config_.expected_trajectory)
    {
      throw std::invalid_argument(
        "TensorRT neural dynamics currently requires expected_trajectory=true");
    }
    if (model_kind == ModelKind::kTensorRtNeuralDerivative &&
      config_.refinement.enabled)
    {
      throw std::invalid_argument(
        "MPC refinement currently requires analytic dynamics Jacobians; "
        "disable mpc_refinement for the TensorRT model");
    }
    if (config_.refinement.enabled && config_.horizon > kMaximumRefinementHorizon) {
      throw std::invalid_argument("MPC refinement horizon exceeds the CUDA PCG limit of 128");
    }
    if (model_kind == ModelKind::kTensorRtNeuralDerivative) {
      neural_model_ = std::make_unique<TensorRtDerivativeModel>(neural_engine_path);
    }
    device_config_.samples = config_.num_samples;
    device_config_.horizon = config_.horizon;
    device_config_.substeps = config_.integration_substeps;
    device_config_.smoothing_window = config_.noise_smoothing_window;
    device_config_.delay_steps = config_.control_delay_steps;
    device_config_.obstacle_brake_steps = config_.obstacle_latch_brake_steps;
    device_config_.dt = config_.dt_s;
    device_config_.lambda = config_.lambda;
    device_config_.gamma = config_.control_cost_gamma;
    device_config_.special_samples = config_.special_samples;
    device_config_.use_reference_controls = config_.use_reference_controls;
    device_config_.speed_scaled_steering = config_.adaptation.speed_scaled_steering;
    device_config_.steering_reference_speed = config_.adaptation.steering_reference_speed_mps;
    device_config_.steering_minimum_scale = config_.adaptation.steering_minimum_scale;
    device_config_.model_kind = model_kind;
    device_config_.integrator_kind = integrator_kind;
    device_config_.frame = config_.frame;
    device_refinement_.enabled = config_.refinement.enabled;
    device_refinement_.sqp_iterations = config_.refinement.sqp_iterations;
    device_refinement_.pcg_iterations = config_.refinement.pcg_iterations;
    device_refinement_.pcg_tolerance = config_.refinement.pcg_tolerance;
    device_refinement_.constraint_tolerance = config_.refinement.constraint_tolerance;
    device_refinement_.finite_difference_relative_step =
      config_.refinement.finite_difference_relative_step;
    device_refinement_.hessian_regularization = config_.refinement.hessian_regularization;
    device_refinement_.merit_constraint_penalty =
      config_.refinement.merit_constraint_penalty;
    device_refinement_.state_proximity_weight = config_.refinement.state_proximity_weight;
    device_refinement_.control_proximity_weight = config_.refinement.control_proximity_weight;
    device_refinement_.maximum_state_step = config_.refinement.maximum_state_step;
    for (std::size_t i = 0; i < kControlDim; ++i) {
      device_config_.sigma[i] = config_.sigma[i];
      device_config_.control_min[i] = config_.control_min[i];
      device_config_.control_max[i] = config_.control_max[i];
      device_refinement_.maximum_control_step[i] =
        config_.refinement.maximum_control_step[i];
      device_refinement_.maximum_control_rate[i] =
        config_.refinement.maximum_control_rate[i];
    }

    CheckCuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreate");
    CheckCuda(cudaStreamCreateWithFlags(&sigma_stream_, cudaStreamNonBlocking),
      "create adaptive sigma CUDA stream");
    CheckCuda(cudaStreamCreateWithFlags(
        &visualization_stream_, cudaStreamNonBlocking),
      "create visualization CUDA stream");
    CheckCuda(cudaEventCreate(&start_event_), "cudaEventCreate start");
    CheckCuda(cudaEventCreate(&stop_event_), "cudaEventCreate stop");
    CheckCuda(cudaEventCreate(&refinement_start_event_), "cudaEventCreate refinement start");
    CheckCuda(cudaEventCreate(&refinement_stop_event_), "cudaEventCreate refinement stop");
    CheckCuda(cudaEventCreateWithFlags(
        &sigma_inputs_ready_event_, cudaEventDisableTiming),
      "create adaptive sigma input event");
    CheckCuda(cudaEventCreateWithFlags(
        &sigma_ready_event_, cudaEventDisableTiming),
      "create adaptive sigma completion event");
    const std::size_t samples = config_.num_samples;
    const std::size_t horizon = config_.horizon;
    Allocate(random_states_, samples);
    Allocate(raw_noise_, samples * horizon * kControlDim);
    Allocate(candidates_, samples * horizon);
    Allocate(perturbations_, samples * horizon);
    Allocate(trajectories_, samples * (horizon + 1U));
    Allocate(costs_device_, samples);
    Allocate(weights_device_, samples);
    Allocate(reduction_a_, samples);
    Allocate(reduction_b_, samples);
    Allocate(nominal_, horizon);
    Allocate(nominal_snapshot_, horizon);
    Allocate(updated_, horizon);
    Allocate(cost_terms_, 1U);
    Allocate(expected_, horizon + 1U);
    Allocate(reference_states_, horizon + 1U);
    Allocate(reference_controls_, horizon);
    Allocate(reference_s_, horizon + 1U);
    Allocate(reference_speed_, horizon + 1U);
    Allocate(reference_e_min_, horizon + 1U);
    Allocate(reference_e_max_, horizon + 1U);
    Allocate(minimum_, 1U);
    Allocate(worst_, 1U);
    Allocate(sum_, 1U);
    Allocate(sum_squares_, 1U);
    Allocate(ess_, 1U);
    Allocate(sigma_hat_, kControlDim);
    Allocate(sigma_partials_, kSigmaBlocks * kControlDim * 2U);
    Allocate(finite_flags_, samples);
    Allocate(finite_count_, 1U);
    Allocate(obstacle_latches_, samples);
    Allocate(obstacle_brake_latches_, samples);
    Allocate(obstacle_latch_flags_, samples);
    Allocate(obstacle_latch_count_, 1U);
    Allocate(finite_unlatched_flags_, samples);
    Allocate(finite_unlatched_count_, 1U);
    for (auto & slot : visualization_slots_) {
      Allocate(slot.weights, samples);
      Allocate(slot.trajectories, samples * (horizon + 1U));
      CheckCuda(cudaEventCreateWithFlags(
          &slot.ready_event, cudaEventDisableTiming),
        "create visualization snapshot event");
    }
    if (config_.refinement.enabled) {
      const std::size_t state_count = horizon + 1U;
      const std::size_t multiplier_count = state_count * kStateDim;
      Allocate(refinement_states_, state_count);
      Allocate(refinement_controls_, horizon);
      Allocate(refinement_anchor_states_, state_count);
      Allocate(refinement_anchor_controls_, horizon);
      Allocate(refinement_dynamics_a_, horizon * kStateDim * kStateDim);
      Allocate(refinement_dynamics_b_, horizon * kStateDim * kControlDim);
      Allocate(refinement_defects_, horizon);
      Allocate(refinement_state_gradient_, state_count * kStateDim);
      Allocate(refinement_state_inverse_hessian_, state_count * kStateDim);
      Allocate(refinement_control_gradient_, horizon * kControlDim);
      Allocate(refinement_control_inverse_hessian_, horizon * kControlDim);
      Allocate(refinement_schur_diagonal_, state_count * kStateDim * kStateDim);
      Allocate(refinement_schur_upper_, horizon * kStateDim * kStateDim);
      Allocate(refinement_schur_inverse_diagonal_, state_count * kStateDim * kStateDim);
      Allocate(refinement_rhs_, multiplier_count);
      Allocate(refinement_dual_, multiplier_count);
      Allocate(refinement_state_step_, state_count);
      Allocate(refinement_control_step_, horizon);
      Allocate(refinement_candidate_states_, kLineSearchCandidates * state_count);
      Allocate(refinement_candidate_controls_, kLineSearchCandidates * horizon);
      Allocate(refinement_metrics_, kLineSearchCandidates);
      Allocate(refinement_original_metrics_, 1U);
      Allocate(refinement_initial_metrics_, 1U);
      Allocate(refinement_final_metrics_, 1U);
      Allocate(refinement_delta_, 1U);
      Allocate(refinement_selected_, 1U);
      Allocate(refinement_accepted_, 1U);
      Allocate(refinement_active_, 1U);
      Allocate(refinement_sqp_iterations_, 1U);
      Allocate(refinement_pcg_iterations_, 1U);
      CheckCuda(cudaMemsetAsync(
        refinement_dual_, 0, multiplier_count * sizeof(float), stream_),
        "zero MPC refinement dual warm start");
    }
    if (model_kind == ModelKind::kTensorRtNeuralDerivative) {
      Allocate(neural_states_, samples);
      Allocate(neural_input_, samples * TensorRtDerivativeModel::kInputWidth);
      Allocate(neural_derivative_, samples * TensorRtDerivativeModel::kOutputWidth);
      Allocate(crash_latches_, samples);
      Allocate(sideslip_latches_, samples);
      Allocate(neural_s_hint_, samples);
    }

    const std::size_t point_count = raceline.points().size();
    std::vector<float> track_s;
    std::vector<float> track_curvature;
    std::vector<float> track_east;
    std::vector<float> track_north;
    std::vector<float> track_heading;
    track_s.reserve(point_count);
    track_curvature.reserve(point_count);
    track_east.reserve(point_count);
    track_north.reserve(point_count);
    track_heading.reserve(point_count);
    for (const auto & point : raceline.points()) {
      track_s.push_back(point.s_m);
      track_curvature.push_back(point.curvature_inv_m);
      track_east.push_back(point.east_m);
      track_north.push_back(point.north_m);
      track_heading.push_back(point.heading_from_north_rad);
    }
    Allocate(track_s_, track_s.size());
    Allocate(track_curvature_, track_curvature.size());
    Allocate(track_east_, track_east.size());
    Allocate(track_north_, track_north.size());
    Allocate(track_heading_, track_heading.size());
    const auto copy_track = [&](float * device, const std::vector<float> & host,
        const char * label) {
        CheckCuda(cudaMemcpyAsync(
          device, host.data(), host.size() * sizeof(float),
          cudaMemcpyHostToDevice, stream_), label);
      };
    copy_track(track_s_, track_s, "copy track s");
    copy_track(track_curvature_, track_curvature, "copy track curvature");
    copy_track(track_east_, track_east, "copy track east");
    copy_track(track_north_, track_north, "copy track north");
    copy_track(track_heading_, track_heading, "copy track heading");

    // Cartesian rollouts reproject each state onto the raceline. Scanning a
    // fixed arc-length window either side of the previous projection keeps the
    // search bounded and stops a far branch of the track from stealing it, the
    // same guarantee the host ContinuousProjector gives.
    const float mean_spacing_m = raceline.length() /
      static_cast<float>(std::max<std::size_t>(point_count - 1U, 1U));
    const auto span = static_cast<std::uint32_t>(std::max(1.0F, std::ceil(
        cartesian_projection_window_m / std::max(mean_spacing_m, 1.0e-3F))));
    track_ = DeviceTrack{
      track_s_, track_curvature_, track_east_, track_north_, track_heading_,
      static_cast<std::uint32_t>(point_count), raceline.s_min(), raceline.s_max(),
      raceline.length(), raceline.closed(),
      std::min(span, static_cast<std::uint32_t>(point_count))};

    if (obstacle_config_.enabled) {
      obstacle_width_ = static_cast<std::uint32_t>(std::ceil(
          obstacle_config_.grid_width_m / obstacle_config_.grid_resolution_m));
      obstacle_height_ = static_cast<std::uint32_t>(std::ceil(
          obstacle_config_.grid_height_m / obstacle_config_.grid_resolution_m));
      const std::size_t obstacle_cells =
        static_cast<std::size_t>(obstacle_width_) * obstacle_height_;
      Allocate(obstacle_distance_, obstacle_cells);
      CheckCuda(cudaMallocHost(
          reinterpret_cast<void **>(&obstacle_staging_), obstacle_cells * sizeof(float)),
        "allocate pinned obstacle staging");
    }

    std::size_t min_bytes = 0;
    std::size_t max_bytes = 0;
    std::size_t sum_bytes = 0;
    std::size_t count_bytes = 0;
    cub::DeviceReduce::Min(nullptr, min_bytes, reduction_a_, minimum_, samples, stream_);
    cub::DeviceReduce::Max(nullptr, max_bytes, reduction_b_, worst_, samples, stream_);
    cub::DeviceReduce::Sum(nullptr, sum_bytes, weights_device_, sum_, samples, stream_);
    cub::DeviceReduce::Sum(
      nullptr, count_bytes, finite_flags_, finite_count_, samples, stream_);
    reduction_temp_bytes_ = std::max({min_bytes, max_bytes, sum_bytes, count_bytes});
    CheckCuda(cudaMalloc(&reduction_temp_, reduction_temp_bytes_), "allocate CUB reduction temp");

    const int blocks = static_cast<int>((samples + 255U) / 256U);
    InitRandomStates<<<blocks, 256, 0, stream_>>>(
      random_states_, config_.num_samples, config_.seed);
    CheckCuda(cudaGetLastError(), "launch InitRandomStates");
    CheckCuda(cudaMemsetAsync(nominal_, 0, horizon * sizeof(Control), stream_), "zero nominal");
    CheckCuda(cudaStreamSynchronize(stream_), "initialize CUDA MPPI");
  }

  ~Impl() {
    Free(random_states_);
    Free(raw_noise_);
    Free(candidates_);
    Free(perturbations_);
    Free(trajectories_);
    Free(costs_device_);
    Free(weights_device_);
    Free(reduction_a_);
    Free(reduction_b_);
    Free(nominal_);
    Free(nominal_snapshot_);
    Free(updated_);
    Free(cost_terms_);
    Free(expected_);
    Free(reference_states_);
    Free(reference_controls_);
    Free(reference_s_);
    Free(reference_speed_);
    Free(reference_e_min_);
    Free(reference_e_max_);
    Free(track_s_);
    Free(track_curvature_);
    Free(track_east_);
    Free(track_north_);
    Free(track_heading_);
    Free(obstacle_distance_);
    if (obstacle_staging_ != nullptr) {
      cudaFreeHost(obstacle_staging_);
    }
    Free(minimum_);
    Free(worst_);
    Free(sum_);
    Free(sum_squares_);
    Free(ess_);
    Free(sigma_hat_);
    Free(sigma_partials_);
    Free(finite_flags_);
    Free(finite_count_);
    Free(obstacle_latches_);
    Free(obstacle_brake_latches_);
    Free(obstacle_latch_flags_);
    Free(obstacle_latch_count_);
    Free(finite_unlatched_flags_);
    Free(finite_unlatched_count_);
    for (auto & slot : visualization_slots_) {
      Free(slot.weights);
      Free(slot.trajectories);
      if (slot.ready_event != nullptr) {
        cudaEventDestroy(slot.ready_event);
      }
    }
    Free(refinement_states_);
    Free(refinement_controls_);
    Free(refinement_anchor_states_);
    Free(refinement_anchor_controls_);
    Free(refinement_dynamics_a_);
    Free(refinement_dynamics_b_);
    Free(refinement_defects_);
    Free(refinement_state_gradient_);
    Free(refinement_state_inverse_hessian_);
    Free(refinement_control_gradient_);
    Free(refinement_control_inverse_hessian_);
    Free(refinement_schur_diagonal_);
    Free(refinement_schur_upper_);
    Free(refinement_schur_inverse_diagonal_);
    Free(refinement_rhs_);
    Free(refinement_dual_);
    Free(refinement_state_step_);
    Free(refinement_control_step_);
    Free(refinement_candidate_states_);
    Free(refinement_candidate_controls_);
    Free(refinement_metrics_);
    Free(refinement_original_metrics_);
    Free(refinement_initial_metrics_);
    Free(refinement_final_metrics_);
    Free(refinement_delta_);
    Free(refinement_selected_);
    Free(refinement_accepted_);
    Free(refinement_active_);
    Free(refinement_sqp_iterations_);
    Free(refinement_pcg_iterations_);
    Free(neural_states_);
    Free(neural_input_);
    Free(neural_derivative_);
    Free(crash_latches_);
    Free(sideslip_latches_);
    Free(neural_s_hint_);
    if (reduction_temp_ != nullptr) {
      cudaFree(reduction_temp_);
    }
    if (start_event_ != nullptr) {
      cudaEventDestroy(start_event_);
    }
    if (stop_event_ != nullptr) {
      cudaEventDestroy(stop_event_);
    }
    if (refinement_start_event_ != nullptr) {
      cudaEventDestroy(refinement_start_event_);
    }
    if (refinement_stop_event_ != nullptr) {
      cudaEventDestroy(refinement_stop_event_);
    }
    if (sigma_inputs_ready_event_ != nullptr) {
      cudaEventDestroy(sigma_inputs_ready_event_);
    }
    if (sigma_ready_event_ != nullptr) {
      cudaEventDestroy(sigma_ready_event_);
    }
    if (stream_ != nullptr) {
      cudaStreamDestroy(stream_);
    }
    if (sigma_stream_ != nullptr) {
      cudaStreamDestroy(sigma_stream_);
    }
    if (visualization_stream_ != nullptr) {
      cudaStreamDestroy(visualization_stream_);
    }
  }

  std::vector<WeightedRollout> CollectVisualization(const std::uint64_t snapshot_id) {
    if (snapshot_id == 0U) {
      return {};
    }
    VisualizationSlot * slot = nullptr;
    {
      std::lock_guard<std::mutex> lock(visualization_mutex_);
      for (auto & candidate : visualization_slots_) {
        if (candidate.state == VisualizationSlotState::kReady &&
          candidate.id == snapshot_id)
        {
          candidate.state = VisualizationSlotState::kReading;
          slot = &candidate;
          break;
        }
      }
    }
    // Visualization is latest-only. A ready snapshot may be replaced before
    // the worker reaches it; dropping that stale frame is preferable to ever
    // applying back-pressure to control.
    if (slot == nullptr) {
      return {};
    }

    const auto release_slot = [this, slot, snapshot_id]() {
        std::lock_guard<std::mutex> lock(visualization_mutex_);
        if (slot->id == snapshot_id && slot->state == VisualizationSlotState::kReading) {
          slot->state = VisualizationSlotState::kFree;
        }
      };
    try {
      std::vector<float> weights(config_.num_samples);
      CheckCuda(cudaStreamWaitEvent(
          visualization_stream_, slot->ready_event, 0U),
        "wait for visualization snapshot");
      CheckCuda(cudaMemcpyAsync(
          weights.data(), slot->weights, weights.size() * sizeof(float),
          cudaMemcpyDeviceToHost, visualization_stream_),
        "copy visualization weights");
      CheckCuda(cudaStreamSynchronize(visualization_stream_),
        "wait for visualization weights");

      std::vector<std::uint32_t> indices(config_.num_samples);
      std::iota(indices.begin(), indices.end(), 0U);
      const std::size_t rollout_count = std::min<std::size_t>(
        slot->rollout_count, config_.num_samples);
      std::partial_sort(
        indices.begin(), indices.begin() + static_cast<std::ptrdiff_t>(rollout_count),
        indices.end(),
        [&weights](const std::uint32_t lhs, const std::uint32_t rhs) {
          if (weights[lhs] == weights[rhs]) {
            return lhs < rhs;
          }
          return weights[lhs] > weights[rhs];
        });

      std::vector<WeightedRollout> rollouts(rollout_count);
      for (std::size_t rank = 0U; rank < rollout_count; ++rank) {
        const std::uint32_t sample = indices[rank];
        auto & rollout = rollouts[rank];
        rollout.weight = weights[sample];
        rollout.states.resize(static_cast<std::size_t>(config_.horizon) + 1U);
        const State * source = slot->trajectories +
          static_cast<std::size_t>(sample) * (config_.horizon + 1U);
        CheckCuda(cudaMemcpyAsync(
            rollout.states.data(), source, rollout.states.size() * sizeof(State),
            cudaMemcpyDeviceToHost, visualization_stream_),
          "copy visualization rollout");
      }
      CheckCuda(cudaStreamSynchronize(visualization_stream_),
        "wait for visualization rollouts");
      release_slot();
      return rollouts;
    } catch (...) {
      release_slot();
      throw;
    }
  }

  MppiSolution Solve(
    const State & initial_state, const ReferenceHorizon & reference,
    const Control & previous_control, const float initial_path_s_m,
    const float shift_fraction, const bool reset,
    const std::uint32_t num_visualization_rollouts, const bool capture_cost_terms)
  {
    const std::size_t horizon = config_.horizon;
    if (!std::isfinite(initial_path_s_m)) {
      throw std::invalid_argument("initial path arc length must be finite");
    }
    if (reference.states.size() != horizon + 1U || reference.controls.size() != horizon ||
      reference.s_grid.size() != horizon + 1U || reference.speed_profile.size() != horizon + 1U ||
      reference.e_min.size() != horizon + 1U || reference.e_max.size() != horizon + 1U)
    {
      throw std::invalid_argument("reference does not match configured MPPI horizon");
    }
    CheckCuda(cudaEventRecord(start_event_, stream_), "record MPPI start");
    CheckCuda(cudaMemcpyAsync(
      reference_states_, reference.states.data(), (horizon + 1U) * sizeof(State),
      cudaMemcpyHostToDevice, stream_), "copy reference states");
    CheckCuda(cudaMemcpyAsync(
      reference_controls_, reference.controls.data(), horizon * sizeof(Control),
      cudaMemcpyHostToDevice, stream_), "copy reference controls");
    CheckCuda(cudaMemcpyAsync(
      reference_s_, reference.s_grid.data(), (horizon + 1U) * sizeof(float),
      cudaMemcpyHostToDevice, stream_), "copy reference s");
    CheckCuda(cudaMemcpyAsync(
      reference_speed_, reference.speed_profile.data(), (horizon + 1U) * sizeof(float),
      cudaMemcpyHostToDevice, stream_), "copy reference speed");
    CheckCuda(cudaMemcpyAsync(
      reference_e_min_, reference.e_min.data(), (horizon + 1U) * sizeof(float),
      cudaMemcpyHostToDevice, stream_), "copy reference e_min");
    CheckCuda(cudaMemcpyAsync(
      reference_e_max_, reference.e_max.data(), (horizon + 1U) * sizeof(float),
      cudaMemcpyHostToDevice, stream_), "copy reference e_max");
    if (reset) {
      if (config_.use_reference_controls) {
        CheckCuda(cudaMemcpyAsync(
          nominal_, reference.controls.data(), horizon * sizeof(Control),
          cudaMemcpyHostToDevice, stream_), "reset nominal from reference");
      } else {
        CheckCuda(cudaMemsetAsync(nominal_, 0, horizon * sizeof(Control), stream_), "reset nominal");
      }
    }
    const DeviceReference device_reference{
      reference_states_, reference_controls_, reference_s_, reference_speed_,
      reference_e_min_, reference_e_max_, config_.horizon};
    const int sample_blocks = static_cast<int>((config_.num_samples + 255U) / 256U);
    GenerateNoise<<<sample_blocks, 256, 0, stream_>>>(
      random_states_, raw_noise_, device_config_);
    const std::size_t candidate_values =
      static_cast<std::size_t>(config_.num_samples) * horizon * kControlDim;
    BuildCandidates<<<static_cast<int>((candidate_values + 255U) / 256U), 256, 0, stream_>>>(
      raw_noise_, nominal_, reference_controls_, candidates_, perturbations_,
      initial_state, device_config_);
    if (device_config_.model_kind == ModelKind::kTensorRtNeuralDerivative) {
      InitializeNeuralRollouts<<<sample_blocks, 256, 0, stream_>>>(
        initial_state, initial_path_s_m, neural_states_, trajectories_, costs_device_,
        crash_latches_, sideslip_latches_, obstacle_latches_, obstacle_brake_latches_,
        neural_s_hint_, device_config_);
      const std::uint16_t substeps = device_config_.substeps == 0U ? 1U :
        device_config_.substeps;
      const float step_dt = device_config_.dt / static_cast<float>(substeps);
      for (std::uint16_t t = 0; t < device_config_.horizon; ++t) {
        NeuralStageCostAndPack<<<sample_blocks, 256, 0, stream_>>>(
          initial_state, previous_control, neural_states_, nominal_, candidates_,
          neural_input_, costs_device_, crash_latches_, sideslip_latches_,
          obstacle_latches_, obstacle_brake_latches_, neural_s_hint_, t,
          track_, device_reference,
          obstacle_field_, device_config_, costs_);
        for (std::uint16_t substep = 0; substep < substeps; ++substep) {
          if (substep != 0U) {
            PackNeuralInputs<<<sample_blocks, 256, 0, stream_>>>(
              neural_states_, candidates_, neural_input_, t, device_config_);
          }
          neural_model_->InferAsync(
            neural_input_, neural_derivative_, config_.num_samples,
            reinterpret_cast<void *>(stream_));
          NeuralIntegrateAndStore<<<sample_blocks, 256, 0, stream_>>>(
            neural_states_, neural_derivative_, trajectories_, costs_device_, t,
            step_dt, substep + 1U == substeps, track_, device_config_, costs_);
        }
      }
      FinalizeNeuralCosts<<<sample_blocks, 256, 0, stream_>>>(
        initial_path_s_m, neural_states_, costs_device_, crash_latches_,
        sideslip_latches_, obstacle_latches_, obstacle_brake_latches_,
        neural_s_hint_, track_,
        device_reference, obstacle_field_, device_config_, costs_);
    } else {
      RolloutAndCost<<<sample_blocks, 256, 0, stream_>>>(
        initial_state, initial_path_s_m, previous_control, nominal_, candidates_,
        trajectories_, costs_device_, obstacle_latches_, obstacle_brake_latches_,
        track_, device_reference, obstacle_field_, device_config_, costs_, vehicle_);
    }
    PrepareReductionInputs<<<sample_blocks, 256, 0, stream_>>>(
      costs_device_, reduction_a_, reduction_b_, obstacle_brake_latches_, finite_flags_,
      obstacle_latch_flags_, finite_unlatched_flags_, config_.num_samples);
    cub::DeviceReduce::Sum(
      reduction_temp_, reduction_temp_bytes_, finite_flags_, finite_count_,
      config_.num_samples, stream_);
    cub::DeviceReduce::Sum(
      reduction_temp_, reduction_temp_bytes_, obstacle_latch_flags_,
      obstacle_latch_count_, config_.num_samples, stream_);
    cub::DeviceReduce::Sum(
      reduction_temp_, reduction_temp_bytes_, finite_unlatched_flags_,
      finite_unlatched_count_, config_.num_samples, stream_);
    cub::DeviceReduce::Min(
      reduction_temp_, reduction_temp_bytes_, reduction_a_, minimum_,
      config_.num_samples, stream_);
    cub::DeviceReduce::Max(
      reduction_temp_, reduction_temp_bytes_, reduction_b_, worst_,
      config_.num_samples, stream_);
    ComputeUnnormalizedWeights<<<sample_blocks, 256, 0, stream_>>>(
      costs_device_, weights_device_, minimum_, worst_, device_config_);
    cub::DeviceReduce::Sum(
      reduction_temp_, reduction_temp_bytes_, weights_device_, sum_,
      config_.num_samples, stream_);
    NormalizeWeights<<<sample_blocks, 256, 0, stream_>>>(
      weights_device_, reduction_a_, sum_, config_.num_samples);
    cub::DeviceReduce::Sum(
      reduction_temp_, reduction_temp_bytes_, reduction_a_, sum_squares_,
      config_.num_samples, stream_);
    ComputeEss<<<1, 1, 0, stream_>>>(sum_squares_, ess_);
    const std::size_t control_values = horizon * kControlDim;
    WeightedControls<<<static_cast<int>(control_values), kWeightedThreads, 0, stream_>>>(
      weights_device_, candidates_, updated_, device_config_);
    const std::size_t state_values = (horizon + 1U) * kStateDim;
    if (config_.expected_trajectory) {
      WeightedStates<<<static_cast<int>(state_values), kWeightedThreads, 0, stream_>>>(
        weights_device_, trajectories_, expected_, device_config_);
    } else {
      RolloutUpdatedControls<<<1, 1, 0, stream_>>>(
        initial_state, updated_, expected_, track_, device_config_, vehicle_);
    }
    if (config_.adaptation.adaptive_sigma) {
      // Sigma only reads the completed MPPI weights and perturbations. Run its
      // reduction concurrently with SQP and join only before the tiny result
      // copy; normally it has finished several milliseconds earlier.
      CheckCuda(cudaEventRecord(sigma_inputs_ready_event_, stream_),
        "record adaptive sigma inputs");
      CheckCuda(cudaStreamWaitEvent(sigma_stream_, sigma_inputs_ready_event_, 0U),
        "wait for adaptive sigma inputs");
      SigmaPartials<<<kSigmaBlocks, kSigmaThreads, 0, sigma_stream_>>>(
        weights_device_, perturbations_, sigma_partials_, device_config_);
      SigmaFinalize<<<1, 32, 0, sigma_stream_>>>(
        sigma_partials_, sigma_hat_, device_config_);
      CheckCuda(cudaEventRecord(sigma_ready_event_, sigma_stream_),
        "record adaptive sigma completion");
    }
    if (config_.refinement.enabled) {
      CheckCuda(cudaEventRecord(refinement_start_event_, stream_),
        "record MPC refinement start");
      CheckCuda(cudaMemcpyAsync(
        refinement_states_, expected_, (horizon + 1U) * sizeof(State),
        cudaMemcpyDeviceToDevice, stream_), "warm start MPC states from MPPI expectation");
      CheckCuda(cudaMemcpyAsync(
        refinement_controls_, updated_, horizon * sizeof(Control),
        cudaMemcpyDeviceToDevice, stream_), "warm start MPC controls from MPPI mean");
      CheckCuda(cudaMemcpyAsync(
        refinement_anchor_states_, expected_, (horizon + 1U) * sizeof(State),
        cudaMemcpyDeviceToDevice, stream_), "snapshot MPC state anchor");
      CheckCuda(cudaMemcpyAsync(
        refinement_anchor_controls_, updated_, horizon * sizeof(Control),
        cudaMemcpyDeviceToDevice, stream_), "snapshot MPC control anchor");
      if (reset) {
        CheckCuda(cudaMemsetAsync(
          refinement_dual_, 0, (horizon + 1U) * kStateDim * sizeof(float), stream_),
          "reset MPC dual warm start");
      }
      CheckCuda(cudaMemsetAsync(
        refinement_active_, 1, sizeof(std::uint8_t), stream_),
        "activate MPC refinement");
      CheckCuda(cudaMemsetAsync(
        refinement_sqp_iterations_, 0, sizeof(std::uint16_t), stream_),
        "reset MPC SQP iteration count");
      CheckCuda(cudaMemsetAsync(
        refinement_pcg_iterations_, 0, sizeof(std::uint16_t), stream_),
        "reset MPC PCG iteration count");
      CheckCuda(cudaMemsetAsync(
        refinement_accepted_, 0, sizeof(std::uint8_t), stream_),
        "reset MPC acceptance");
      CheckCuda(cudaMemsetAsync(
        refinement_metrics_, 0,
        kLineSearchCandidates * sizeof(RefinementMetrics), stream_),
        "reset MPC trial metrics");
      CheckCuda(cudaMemsetAsync(
        refinement_original_metrics_, 0, sizeof(RefinementMetrics), stream_),
        "reset initial MPC metrics");
      CheckCuda(cudaMemsetAsync(
        refinement_initial_metrics_, 0, sizeof(RefinementMetrics), stream_),
        "reset current MPC metrics");
      CheckCuda(cudaMemsetAsync(
        refinement_final_metrics_, 0, sizeof(RefinementMetrics), stream_),
        "reset final MPC metrics");
      DisableRefinementWhenAllObstacleLatched<<<1, 1, 0, stream_>>>(
        obstacle_latch_count_, refinement_active_, device_config_);
      const std::size_t derivative_values = std::max(
        (horizon + 1U) * kStateDim, horizon * kControlDim);
      const int derivative_blocks = static_cast<int>((derivative_values + 255U) / 256U);
      const std::size_t trajectory_values = horizon + 1U;
      const int trajectory_blocks = static_cast<int>((trajectory_values + 255U) / 256U);
      const std::size_t multiplier_count = (horizon + 1U) * kStateDim;
      const std::size_t pcg_shared_bytes =
        (5U * multiplier_count + kRefinementThreads) * sizeof(float);
      for (std::uint16_t iteration = 0U;
        iteration < config_.refinement.sqp_iterations; ++iteration)
      {
        LinearizeRefinementDynamics<<<static_cast<int>(horizon), 32, 0, stream_>>>(
          refinement_states_, refinement_controls_, refinement_dynamics_a_,
          refinement_dynamics_b_, refinement_defects_, refinement_active_, track_, device_config_,
          device_refinement_, vehicle_);
        QuadraticizeRefinementCost<<<derivative_blocks, 256, 0, stream_>>>(
          refinement_states_, refinement_controls_, refinement_anchor_states_,
          refinement_anchor_controls_, previous_control,
          refinement_state_gradient_, refinement_state_inverse_hessian_,
          refinement_control_gradient_, refinement_control_inverse_hessian_,
          refinement_active_, initial_path_s_m, track_, device_reference,
          obstacle_field_, device_config_,
          costs_, device_refinement_);
        BuildRefinementSchur<<<static_cast<int>(horizon + 1U), 1, 0, stream_>>>(
          refinement_dynamics_a_, refinement_dynamics_b_, refinement_defects_,
          refinement_state_gradient_, refinement_state_inverse_hessian_,
          refinement_control_gradient_, refinement_control_inverse_hessian_,
          refinement_schur_diagonal_, refinement_schur_upper_,
          refinement_schur_inverse_diagonal_, refinement_rhs_, refinement_active_,
          device_config_, device_refinement_);
        SolveRefinementPcg<<<1, kRefinementThreads, pcg_shared_bytes, stream_>>>(
          refinement_schur_diagonal_, refinement_schur_upper_,
          refinement_schur_inverse_diagonal_, refinement_rhs_, refinement_dual_,
          refinement_pcg_iterations_, refinement_active_, device_config_, device_refinement_);
        RecoverRefinementStep<<<derivative_blocks, 256, 0, stream_>>>(
          refinement_dynamics_a_, refinement_dynamics_b_,
          refinement_state_gradient_, refinement_state_inverse_hessian_,
          refinement_control_gradient_, refinement_control_inverse_hessian_,
          refinement_dual_, refinement_state_step_, refinement_control_step_,
          refinement_active_, device_config_, device_refinement_);
        EvaluateRefinementLineSearch<<<kLineSearchCandidates, 1, 0, stream_>>>(
          initial_state, initial_path_s_m, previous_control, refinement_states_,
          refinement_controls_, refinement_control_step_,
          refinement_anchor_states_, refinement_anchor_controls_,
          refinement_candidate_states_, refinement_candidate_controls_,
          refinement_metrics_, refinement_active_, iteration, track_, device_reference, obstacle_field_,
          device_config_, costs_, device_refinement_, vehicle_);
        SelectRefinementLineSearch<<<1, 1, 0, stream_>>>(
          refinement_metrics_, refinement_selected_, refinement_initial_metrics_,
          refinement_final_metrics_, refinement_active_, refinement_sqp_iterations_, iteration);
        if (iteration == 0U) {
          CheckCuda(cudaMemcpyAsync(
            refinement_original_metrics_, refinement_initial_metrics_,
            sizeof(RefinementMetrics), cudaMemcpyDeviceToDevice, stream_),
            "snapshot initial MPC refinement metrics");
        }
        ApplyRefinementLineSearch<<<trajectory_blocks, 256, 0, stream_>>>(
          refinement_states_, refinement_controls_, refinement_candidate_states_,
          refinement_candidate_controls_, refinement_selected_, refinement_active_, device_config_);
      }
      CommitRefinement<<<trajectory_blocks, 256, 0, stream_>>>(
        refinement_states_, refinement_controls_, expected_, updated_,
        refinement_original_metrics_, refinement_final_metrics_,
        refinement_accepted_, obstacle_latch_count_, device_config_, device_refinement_);
      MeasureRefinementDelta<<<1, 1, 0, stream_>>>(
        refinement_anchor_states_, refinement_anchor_controls_, expected_, updated_,
        refinement_accepted_, refinement_delta_, device_config_);
      CheckCuda(cudaEventRecord(refinement_stop_event_, stream_),
        "record MPC refinement stop");
    }
    if (capture_cost_terms) {
      // ShiftNominal overwrites nominal_ in place, and the breakdown runs after
      // the solve has been measured, so the mean the candidates were drawn
      // around has to be kept aside first. Four hundred bytes on the solve path.
      CheckCuda(cudaMemcpyAsync(
        nominal_snapshot_, nominal_, horizon * sizeof(Control),
        cudaMemcpyDeviceToDevice, stream_), "snapshot nominal for cost breakdown");
    }
    ShiftNominal<<<static_cast<int>((control_values + 127U) / 128U), 128, 0, stream_>>>(
      updated_, nominal_, shift_fraction, device_config_);
    CheckCuda(cudaGetLastError(), "launch MPPI CUDA pipeline");

    MppiSolution solution;
    solution.controls.resize(horizon);
    solution.states.resize(horizon + 1U);
    std::array<float, kControlDim> sigma_hat{};
    float minimum = 0.0F;
    float ess = 0.0F;
    std::uint32_t finite_count = 0U;
    std::uint32_t obstacle_latch_count = 0U;
    std::uint32_t finite_unlatched_count = 0U;
    std::uint8_t refinement_accepted = 0U;
    std::uint16_t refinement_sqp_iterations = 0U;
    std::uint16_t refinement_pcg_iterations = 0U;
    RefinementMetrics refinement_initial_metrics{};
    RefinementMetrics refinement_final_metrics{};
    RefinementDelta refinement_delta{};
    std::array<RefinementMetrics, kLineSearchCandidates> refinement_trial_metrics{};
    CheckCuda(cudaMemcpyAsync(
      solution.controls.data(), updated_, horizon * sizeof(Control),
      cudaMemcpyDeviceToHost, stream_), "copy updated controls");
    CheckCuda(cudaMemcpyAsync(
      solution.states.data(), expected_, (horizon + 1U) * sizeof(State),
      cudaMemcpyDeviceToHost, stream_), "copy expected states");
    CheckCuda(cudaMemcpyAsync(&minimum, minimum_, sizeof(float), cudaMemcpyDeviceToHost, stream_),
      "copy minimum cost");
    CheckCuda(cudaMemcpyAsync(&ess, ess_, sizeof(float), cudaMemcpyDeviceToHost, stream_), "copy ESS");
    if (config_.adaptation.adaptive_sigma) {
      CheckCuda(cudaStreamWaitEvent(stream_, sigma_ready_event_, 0U),
        "join adaptive sigma result");
      CheckCuda(cudaMemcpyAsync(
        sigma_hat.data(), sigma_hat_, kControlDim * sizeof(float),
        cudaMemcpyDeviceToHost, stream_), "copy sigma statistics");
    }
    CheckCuda(cudaMemcpyAsync(
      &finite_count, finite_count_, sizeof(std::uint32_t),
      cudaMemcpyDeviceToHost, stream_), "copy finite rollout count");
    CheckCuda(cudaMemcpyAsync(
      &obstacle_latch_count, obstacle_latch_count_, sizeof(std::uint32_t),
      cudaMemcpyDeviceToHost, stream_), "copy obstacle-latched rollout count");
    CheckCuda(cudaMemcpyAsync(
      &finite_unlatched_count, finite_unlatched_count_, sizeof(std::uint32_t),
      cudaMemcpyDeviceToHost, stream_), "copy finite unlatched rollout count");
    if (config_.refinement.enabled) {
      CheckCuda(cudaMemcpyAsync(
        &refinement_accepted, refinement_accepted_, sizeof(std::uint8_t),
        cudaMemcpyDeviceToHost, stream_), "copy MPC refinement acceptance");
      CheckCuda(cudaMemcpyAsync(
        &refinement_sqp_iterations, refinement_sqp_iterations_, sizeof(std::uint16_t),
        cudaMemcpyDeviceToHost, stream_), "copy MPC refinement SQP iterations");
      CheckCuda(cudaMemcpyAsync(
        &refinement_pcg_iterations, refinement_pcg_iterations_, sizeof(std::uint16_t),
        cudaMemcpyDeviceToHost, stream_), "copy MPC refinement PCG iterations");
      CheckCuda(cudaMemcpyAsync(
        &refinement_initial_metrics, refinement_original_metrics_,
        sizeof(RefinementMetrics), cudaMemcpyDeviceToHost, stream_),
        "copy initial MPC refinement metrics");
      CheckCuda(cudaMemcpyAsync(
        &refinement_final_metrics, refinement_final_metrics_,
        sizeof(RefinementMetrics), cudaMemcpyDeviceToHost, stream_),
        "copy final MPC refinement metrics");
      CheckCuda(cudaMemcpyAsync(
        &refinement_delta, refinement_delta_, sizeof(RefinementDelta),
        cudaMemcpyDeviceToHost, stream_), "copy MPC refinement delta");
      CheckCuda(cudaMemcpyAsync(
        refinement_trial_metrics.data(), refinement_metrics_,
        refinement_trial_metrics.size() * sizeof(RefinementMetrics),
        cudaMemcpyDeviceToHost, stream_), "copy MPC refinement line-search metrics");
    }
    CheckCuda(cudaEventRecord(stop_event_, stream_), "record MPPI stop");
    const std::uint64_t visualization_snapshot_id = QueueVisualizationSnapshot(
      num_visualization_rollouts);
    CheckCuda(cudaEventSynchronize(stop_event_), "wait for MPPI solve");
    float elapsed_ms = 0.0F;
    CheckCuda(cudaEventElapsedTime(&elapsed_ms, start_event_, stop_event_), "measure MPPI solve");
    float refinement_elapsed_ms = 0.0F;
    if (config_.refinement.enabled) {
      CheckCuda(cudaEventElapsedTime(
        &refinement_elapsed_ms, refinement_start_event_, refinement_stop_event_),
        "measure MPC refinement");
    }

    // Decomposed on request only, at the ROS runtime's own reduced rate, and
    // after the stop event so it can neither delay the control solve nor inflate
    // the solve time the publication budget is checked against.
    if (capture_cost_terms) {
      EvaluateCostBreakdown<<<1, 1, 0, stream_>>>(
        initial_state, initial_path_s_m, previous_control, nominal_snapshot_,
        expected_, updated_, cost_terms_, track_, device_reference,
        obstacle_field_, device_config_, costs_);
      CheckCuda(cudaGetLastError(), "launch cost term breakdown");
      CostTerms terms{};
      CheckCuda(cudaMemcpyAsync(
        &terms, cost_terms_, sizeof(CostTerms), cudaMemcpyDeviceToHost, stream_),
        "copy cost term breakdown");
      CheckCuda(cudaStreamSynchronize(stream_), "wait for cost term breakdown");
      if (std::isfinite(terms.total())) {
        solution.diagnostics.cost_terms = terms;
      }
    }

    solution.diagnostics.minimum_cost = minimum;
    solution.visualization_snapshot_id = visualization_snapshot_id;
    solution.diagnostics.effective_sample_size = ess;
    solution.diagnostics.lambda_used = config_.lambda;
    solution.diagnostics.sigma_used = config_.sigma;
    solution.diagnostics.solve_time_ms = elapsed_ms;
    solution.diagnostics.finite_rollouts = finite_count;
    solution.diagnostics.obstacle_latched_rollouts = obstacle_latch_count;
    solution.diagnostics.finite_unlatched_rollouts = finite_unlatched_count;
    solution.diagnostics.all_rollouts_obstacle_latched =
      obstacle_latch_count == config_.num_samples;
    solution.diagnostics.obstacle_field_active =
      obstacle_config_.enabled && obstacle_field_.valid;
    solution.diagnostics.refinement_attempted = config_.refinement.enabled;
    solution.diagnostics.refinement_accepted = refinement_accepted != 0U;
    solution.diagnostics.refinement_iterations = refinement_sqp_iterations;
    solution.diagnostics.refinement_pcg_iterations = refinement_pcg_iterations;
    solution.diagnostics.refinement_time_ms = refinement_elapsed_ms;
    solution.diagnostics.refinement_cost_before = refinement_initial_metrics.cost;
    solution.diagnostics.refinement_cost_after = refinement_final_metrics.cost;
    solution.diagnostics.refinement_merit_before = refinement_initial_metrics.merit;
    solution.diagnostics.refinement_merit_after = refinement_final_metrics.merit;
    solution.diagnostics.refinement_constraint_residual_before =
      refinement_initial_metrics.maximum_dynamics_residual;
    solution.diagnostics.refinement_constraint_residual =
      refinement_final_metrics.maximum_dynamics_residual;
    for (std::size_t channel = 0U; channel < kStateDim; ++channel) {
      solution.diagnostics.refinement_state_rms_delta[channel] =
        refinement_delta.state_rms[channel];
      solution.diagnostics.refinement_state_max_delta[channel] =
        refinement_delta.state_max[channel];
    }
    for (std::size_t channel = 0U; channel < kControlDim; ++channel) {
      solution.diagnostics.refinement_control_rms_delta[channel] =
        refinement_delta.control_rms[channel];
      solution.diagnostics.refinement_control_max_delta[channel] =
        refinement_delta.control_max[channel];
      solution.diagnostics.refinement_first_control_delta[channel] =
        refinement_delta.first_control[channel];
    }
    for (std::size_t candidate = 0U; candidate < kLineSearchCandidates; ++candidate) {
      solution.diagnostics.refinement_trial_merits[candidate] =
        refinement_trial_metrics[candidate].merit;
      solution.diagnostics.refinement_trial_constraint_residuals[candidate] =
        refinement_trial_metrics[candidate].maximum_dynamics_residual;
    }

    if (config_.adaptation.adaptive_lambda) {
      if (ess < config_.adaptation.ess_fraction_min * config_.num_samples) {
        config_.lambda = std::min(config_.lambda * 1.2F, config_.adaptation.lambda_max);
      } else if (ess > config_.adaptation.ess_fraction_max * config_.num_samples) {
        config_.lambda = std::max(config_.lambda * 0.9F, config_.adaptation.lambda_min);
      }
      device_config_.lambda = config_.lambda;
    }
    if (config_.adaptation.adaptive_sigma) {
      for (std::size_t channel = 0; channel < kControlDim; ++channel) {
        const float low = config_.adaptation.sigma_scale_min * base_sigma_[channel];
        const float high = config_.adaptation.sigma_scale_max * base_sigma_[channel];
        config_.sigma[channel] = std::clamp(
          (1.0F - config_.adaptation.sigma_alpha) * config_.sigma[channel] +
          config_.adaptation.sigma_alpha * sigma_hat[channel], low, high);
        device_config_.sigma[channel] = config_.sigma[channel];
      }
    }
    return solution;
  }

  void UpdateObstacleField(const ObstacleField & field) {
    if (!obstacle_config_.enabled) {
      return;
    }
    if (!field.valid() || field.width != obstacle_width_ ||
      field.height != obstacle_height_ ||
      std::abs(field.resolution_m - obstacle_config_.grid_resolution_m) > 1.0e-6F)
    {
      throw std::invalid_argument("obstacle field does not match configured CUDA grid");
    }
    // This method and Solve are owned by the dedicated solver thread. Solve
    // synchronizes its completion event before returning, and this upload is
    // queued on the same stream before the next solve, so an extra full-stream
    // synchronization here only adds map-update jitter.
    std::copy(
      field.signed_distance_m.begin(), field.signed_distance_m.end(), obstacle_staging_);
    CheckCuda(cudaMemcpyAsync(
      obstacle_distance_, obstacle_staging_,
      field.signed_distance_m.size() * sizeof(float), cudaMemcpyHostToDevice, stream_),
      "copy obstacle signed distance field");
    obstacle_field_ = DeviceObstacleField{
      obstacle_distance_, field.origin_east_m, field.origin_north_m,
      field.resolution_m, field.width, field.height, true};
  }

  void ClearObstacleField() {
    // Solver-thread ownership guarantees no concurrent kernel reads here.
    obstacle_field_.valid = false;
  }

  const MppiConfig & config() const noexcept { return config_; }

 private:
  std::uint64_t QueueVisualizationSnapshot(const std::uint32_t rollout_count) {
    if (rollout_count == 0U) {
      return 0U;
    }
    std::lock_guard<std::mutex> lock(visualization_mutex_);
    VisualizationSlot * slot = nullptr;
    for (auto & candidate : visualization_slots_) {
      if (candidate.state == VisualizationSlotState::kFree) {
        slot = &candidate;
        break;
      }
    }
    if (slot == nullptr) {
      // Replace the oldest snapshot that the worker has not started reading.
      for (auto & candidate : visualization_slots_) {
        if (candidate.state == VisualizationSlotState::kReady &&
          (slot == nullptr || candidate.id < slot->id))
        {
          slot = &candidate;
        }
      }
    }
    if (slot == nullptr) {
      return 0U;
    }

    slot->id = next_visualization_snapshot_id_++;
    if (next_visualization_snapshot_id_ == 0U) {
      next_visualization_snapshot_id_ = 1U;
    }
    slot->rollout_count = std::min(rollout_count, config_.num_samples);
    slot->state = VisualizationSlotState::kReady;
    try {
      CheckCuda(cudaMemcpyAsync(
          slot->weights, weights_device_, config_.num_samples * sizeof(float),
          cudaMemcpyDeviceToDevice, stream_),
        "snapshot visualization weights");
      const std::size_t trajectory_count =
        static_cast<std::size_t>(config_.num_samples) * (config_.horizon + 1U);
      CheckCuda(cudaMemcpyAsync(
          slot->trajectories, trajectories_, trajectory_count * sizeof(State),
          cudaMemcpyDeviceToDevice, stream_),
        "snapshot visualization rollouts");
      CheckCuda(cudaEventRecord(slot->ready_event, stream_),
        "record visualization snapshot");
    } catch (...) {
      slot->state = VisualizationSlotState::kFree;
      throw;
    }
    return slot->id;
  }

  MppiConfig config_;
  VehicleParameters vehicle_;
  ObstacleConfig obstacle_config_;
  DeviceCosts costs_{};
  DeviceMppi device_config_{};
  DeviceRefinement device_refinement_{};
  std::array<float, kControlDim> base_sigma_{};
  cudaStream_t stream_{nullptr};
  cudaStream_t sigma_stream_{nullptr};
  cudaStream_t visualization_stream_{nullptr};
  cudaEvent_t start_event_{nullptr};
  cudaEvent_t stop_event_{nullptr};
  cudaEvent_t refinement_start_event_{nullptr};
  cudaEvent_t refinement_stop_event_{nullptr};
  cudaEvent_t sigma_inputs_ready_event_{nullptr};
  cudaEvent_t sigma_ready_event_{nullptr};
  curandStatePhilox4_32_10_t * random_states_{nullptr};
  float * raw_noise_{nullptr};
  Control * candidates_{nullptr};
  Control * perturbations_{nullptr};
  State * trajectories_{nullptr};
  float * costs_device_{nullptr};
  float * weights_device_{nullptr};
  float * reduction_a_{nullptr};
  float * reduction_b_{nullptr};
  Control * nominal_{nullptr};
  Control * nominal_snapshot_{nullptr};
  Control * updated_{nullptr};
  CostTerms * cost_terms_{nullptr};
  State * expected_{nullptr};
  State * reference_states_{nullptr};
  Control * reference_controls_{nullptr};
  float * reference_s_{nullptr};
  float * reference_speed_{nullptr};
  float * reference_e_min_{nullptr};
  float * reference_e_max_{nullptr};
  float * track_s_{nullptr};
  float * track_curvature_{nullptr};
  float * track_east_{nullptr};
  float * track_north_{nullptr};
  float * track_heading_{nullptr};
  DeviceTrack track_{};
  float * obstacle_distance_{nullptr};
  float * obstacle_staging_{nullptr};
  std::uint32_t obstacle_width_{};
  std::uint32_t obstacle_height_{};
  DeviceObstacleField obstacle_field_{};
  float * minimum_{nullptr};
  float * worst_{nullptr};
  float * sum_{nullptr};
  float * sum_squares_{nullptr};
  float * ess_{nullptr};
  float * sigma_hat_{nullptr};
  float * sigma_partials_{nullptr};
  std::uint32_t * finite_flags_{nullptr};
  std::uint32_t * finite_count_{nullptr};
  std::uint32_t * obstacle_latch_flags_{nullptr};
  std::uint32_t * obstacle_latch_count_{nullptr};
  std::uint32_t * finite_unlatched_flags_{nullptr};
  std::uint32_t * finite_unlatched_count_{nullptr};
  std::array<VisualizationSlot, kVisualizationSlotCount> visualization_slots_{};
  std::mutex visualization_mutex_;
  std::uint64_t next_visualization_snapshot_id_{1U};
  State * refinement_states_{nullptr};
  Control * refinement_controls_{nullptr};
  State * refinement_anchor_states_{nullptr};
  Control * refinement_anchor_controls_{nullptr};
  float * refinement_dynamics_a_{nullptr};
  float * refinement_dynamics_b_{nullptr};
  State * refinement_defects_{nullptr};
  float * refinement_state_gradient_{nullptr};
  float * refinement_state_inverse_hessian_{nullptr};
  float * refinement_control_gradient_{nullptr};
  float * refinement_control_inverse_hessian_{nullptr};
  float * refinement_schur_diagonal_{nullptr};
  float * refinement_schur_upper_{nullptr};
  float * refinement_schur_inverse_diagonal_{nullptr};
  float * refinement_rhs_{nullptr};
  float * refinement_dual_{nullptr};
  State * refinement_state_step_{nullptr};
  Control * refinement_control_step_{nullptr};
  State * refinement_candidate_states_{nullptr};
  Control * refinement_candidate_controls_{nullptr};
  RefinementMetrics * refinement_metrics_{nullptr};
  RefinementMetrics * refinement_original_metrics_{nullptr};
  RefinementMetrics * refinement_initial_metrics_{nullptr};
  RefinementMetrics * refinement_final_metrics_{nullptr};
  RefinementDelta * refinement_delta_{nullptr};
  std::uint32_t * refinement_selected_{nullptr};
  std::uint8_t * refinement_accepted_{nullptr};
  std::uint8_t * refinement_active_{nullptr};
  std::uint16_t * refinement_sqp_iterations_{nullptr};
  std::uint16_t * refinement_pcg_iterations_{nullptr};
  State * neural_states_{nullptr};
  float * neural_input_{nullptr};
  float * neural_derivative_{nullptr};
  std::uint8_t * crash_latches_{nullptr};
  std::uint8_t * sideslip_latches_{nullptr};
  std::uint8_t * obstacle_latches_{nullptr};
  std::uint8_t * obstacle_brake_latches_{nullptr};
  float * neural_s_hint_{nullptr};
  std::unique_ptr<TensorRtDerivativeModel> neural_model_;
  void * reduction_temp_{nullptr};
  std::size_t reduction_temp_bytes_{};
};

CudaMppiController::CudaMppiController(
  MppiConfig config, CostWeights costs, VehicleParameters vehicle,
  ObstacleConfig obstacle_config,
  const ModelKind model_kind, const Raceline & raceline,
  const IntegratorKind integrator_kind, std::string neural_engine_path,
  const float projection_window_m)
: impl_(std::make_unique<Impl>(
    std::move(config), costs, vehicle, obstacle_config, model_kind, raceline,
    integrator_kind, std::move(neural_engine_path), projection_window_m)) {}

CudaMppiController::~CudaMppiController() = default;
CudaMppiController::CudaMppiController(CudaMppiController &&) noexcept = default;
CudaMppiController & CudaMppiController::operator=(CudaMppiController &&) noexcept = default;

MppiSolution CudaMppiController::Solve(
  const State & initial_state, const ReferenceHorizon & reference,
  const Control & previous_control, const float initial_path_s_m,
  const float shift_fraction, const bool reset,
  const std::uint32_t num_visualization_rollouts, const bool capture_cost_terms)
{
  return impl_->Solve(
    initial_state, reference, previous_control, initial_path_s_m, shift_fraction,
    reset, num_visualization_rollouts, capture_cost_terms);
}

std::vector<WeightedRollout> CudaMppiController::CollectVisualization(
  const std::uint64_t snapshot_id)
{
  return impl_->CollectVisualization(snapshot_id);
}

const MppiConfig & CudaMppiController::config() const noexcept { return impl_->config(); }
bool CudaMppiController::using_cuda() const noexcept { return true; }
void CudaMppiController::UpdateObstacleField(const ObstacleField & field) {
  impl_->UpdateObstacleField(field);
}

void CudaMppiController::ClearObstacleField() { impl_->ClearObstacleField(); }

}  // namespace xxcar::mppi
