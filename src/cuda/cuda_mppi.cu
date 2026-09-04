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
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "xx_mppi/dynamics/frames.hpp"
#include "xx_mppi/costs/map_boundary.hpp"
#include "xx_mppi/dynamics/models/dynamic_bicycle_fiala.hpp"
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
};

struct DeviceMppi {
  std::uint32_t samples;
  std::uint16_t horizon;
  std::uint16_t substeps;
  std::uint16_t smoothing_window;
  std::uint16_t delay_steps;
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

// latch_violations is false for the initial state, which every sample shares:
// latching a crash or excessive sideslip there would set the same flag for the
// whole population and remove all boundary discrimination from the solve.
__device__ float StateCost(
  const State & state, const FrenetView & frenet, const State & desired,
  const DeviceReference & reference, const DeviceCosts & weights, bool & crashed,
  bool & excessive_sideslip, const float crash_discount, const bool latch_violations)
{
  float cost = 0.0F;
  for (std::size_t i = 0; i < kBodyStateDim; ++i) {
    if (!isfinite(state[i])) {
      return CUDART_INF_F;
    }
    const float error = state[i] - desired[i];
    cost += weights.reference_tracking[i] * error * error;
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
    cost += weights.reference_tracking[kBodyStateDim + i] *
      frame_error[i] * frame_error[i];
  }
  const float speed_target = InterpolateReference(
    reference.speed, frenet.path_evolution, reference);
  const float speed_error = state[kSpeed] - speed_target;
  cost += weights.velocity_profile *
    (speed_error > 0.0F ? weights.velocity_overspeed_multiplier : 1.0F) *
    speed_error * speed_error;

  const float e_min = InterpolateReference(reference.e_min, frenet.path_evolution, reference);
  const float e_max = InterpolateReference(reference.e_max, frenet.path_evolution, reference);
  const auto boundary = EvaluateMapBoundary(
    frenet.lateral_deviation, e_min, e_max, weights.boundary,
    weights.boundary_margin, weights.crash_buffer);
  cost += boundary.shaping_cost;
  crashed = crashed || (latch_violations && boundary.violated);
  if (crashed) {
    cost += weights.crash * crash_discount;
  }
  const float beta = state[kSideslip];
  cost += weights.sideslip * beta * beta;
  excessive_sideslip = excessive_sideslip ||
    (latch_violations && fabsf(beta) > weights.maximum_sideslip);
  if (excessive_sideslip) {
    cost += weights.sideslip_kill / static_cast<float>(reference.horizon + 1U);
  }
  const float lateral_rate = state[kSpeed] * sinf(frenet.relative_course);
  const float damping =
    lateral_rate + weights.lateral_decay_rate * frenet.lateral_deviation;
  cost += weights.lateral_damping * damping * damping;
  const float longitudinal_speed = fmaxf(
    state[kSpeed] * cosf(state[kSideslip]), 1.0F);
  const float slip = (state[kDrivenWheelSpeed] - longitudinal_speed) / longitudinal_speed;
  const float excess_slip = fmaxf(fabsf(slip) - weights.wheel_slip_band, 0.0F);
  cost += weights.wheel_slip * excess_slip * excess_slip;
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
  auto random = random_states[sample];
  for (std::uint16_t t = 0; t < config.horizon; ++t) {
    const float2 draw = curand_normal2(&random);
    const std::size_t offset =
      (static_cast<std::size_t>(sample) * config.horizon + t) * kControlDim;
    raw_noise[offset + kSteering] = draw.x;
    raw_noise[offset + kWheelTorque] = draw.y;
  }
  random_states[sample] = random;
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
  float sigma = config.sigma[control_channel];
  if (control_channel == kSteering && config.speed_scaled_steering) {
    sigma *= fminf(fmaxf(
      config.steering_reference_speed / fmaxf(initial_state[kSpeed], 1.0F),
      config.steering_minimum_scale), 1.0F);
  }
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
  const DeviceTrack track, const DeviceReference reference,
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
  float discount = 1.0F;
  Control prior = previous_control;
  for (std::uint16_t t = 0; t < config.horizon; ++t) {
    cost += StateCost(
      state, frenet, reference.states[t], reference, weights, crashed,
      excessive_sideslip, discount, t != 0U);
    discount *= weights.crash_discount;
    const Control control = candidates[control_base + t];
    for (std::size_t channel = 0; channel < kControlDim; ++channel) {
      cost += weights.control_effort[channel] * control[channel] * control[channel];
      const float delta = control[channel] - prior[channel];
      cost += weights.control_smoothness[channel] * delta * delta;
      const float rate = delta / config.dt;
      cost += weights.control_rate[channel] * rate * rate;
      float sigma = config.sigma[channel];
      if (channel == kSteering && config.speed_scaled_steering) {
        sigma *= fminf(fmaxf(
          config.steering_reference_speed / fmaxf(initial_state[kSpeed], 1.0F),
          config.steering_minimum_scale), 1.0F);
      }
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
  cost += StateCost(
    state, frenet, reference.states[config.horizon], reference, weights, crashed,
    excessive_sideslip, discount, true);
  cost -= weights.progress * (frenet.path_evolution - initial_path_s);
  costs[sample] = isfinite(cost) ? cost : CUDART_INF_F;
}

__global__ void InitializeNeuralRollouts(
  const State initial_state, const float initial_path_s, State * current_states,
  State * trajectories, float * costs, std::uint8_t * crashed,
  std::uint8_t * excessive_sideslip, float * path_s_hint, const DeviceMppi config)
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
}

__global__ void NeuralStageCostAndPack(
  const State initial_state, const Control previous_control,
  const State * current_states, const Control * nominal, const Control * candidates,
  float * model_input, float * costs, std::uint8_t * crashed,
  std::uint8_t * excessive_sideslip, float * path_s_hint,
  const std::uint16_t time_index, const DeviceTrack track,
  const DeviceReference reference, const DeviceMppi config, const DeviceCosts weights)
{
  const std::uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample >= config.samples) {
    return;
  }
  const State state = current_states[sample];
  bool crash_latch = crashed[sample] != 0U;
  bool sideslip_latch = excessive_sideslip[sample] != 0U;
  const float discount = powf(weights.crash_discount, static_cast<float>(time_index));
  const FrenetView frenet = FrameView(state, path_s_hint[sample], track, config);
  path_s_hint[sample] = frenet.path_evolution;
  float cost = StateCost(
    state, frenet, reference.states[time_index], reference, weights,
    crash_latch, sideslip_latch, discount, time_index != 0U);
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
    float sigma = config.sigma[channel];
    if (channel == kSteering && config.speed_scaled_steering) {
      sigma *= fminf(fmaxf(
        config.steering_reference_speed / fmaxf(initial_state[kSpeed], 1.0F),
        config.steering_minimum_scale), 1.0F);
    }
    cost += config.gamma * nominal[time_index][channel] /
      fmaxf(sigma * sigma, 1.0e-12F) *
      (control[channel] - nominal[time_index][channel]);
  }
  costs[sample] += cost;
  crashed[sample] = crash_latch ? 1U : 0U;
  excessive_sideslip[sample] = sideslip_latch ? 1U : 0U;

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
  const float * path_s_hint, const DeviceTrack track,
  const DeviceReference reference, const DeviceMppi config, const DeviceCosts weights)
{
  const std::uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample >= config.samples) {
    return;
  }
  const State state = current_states[sample];
  bool crash_latch = crashed[sample] != 0U;
  bool sideslip_latch = excessive_sideslip[sample] != 0U;
  const float discount = powf(weights.crash_discount, static_cast<float>(config.horizon));
  const FrenetView frenet = FrameView(state, path_s_hint[sample], track, config);
  float total = costs[sample] + StateCost(
    state, frenet, reference.states[config.horizon], reference, weights,
    crash_latch, sideslip_latch, discount, true);
  total -= weights.progress * (frenet.path_evolution - initial_path_s);
  costs[sample] = isfinite(total) ? total : CUDART_INF_F;
}

__global__ void PrepareReductionInputs(
  const float * costs, float * minimum_input, float * maximum_input,
  std::uint32_t * finite_flags, const std::uint32_t count)
{
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    const bool finite = isfinite(costs[index]);
    minimum_input[index] = finite ? costs[index] : CUDART_INF_F;
    maximum_input[index] = finite ? costs[index] : -CUDART_INF_F;
    finite_flags[index] = finite ? 1U : 0U;
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

__global__ void WeightedControls(
  const float * weights, const Control * candidates, Control * updated,
  const DeviceMppi config)
{
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t count = static_cast<std::size_t>(config.horizon) * kControlDim;
  if (index >= count) {
    return;
  }
  const std::size_t t = index / kControlDim;
  const std::size_t channel = index % kControlDim;
  float value = 0.0F;
  for (std::uint32_t sample = 0; sample < config.samples; ++sample) {
    value += weights[sample] * candidates[static_cast<std::size_t>(sample) * config.horizon + t][channel];
  }
  updated[t][channel] = fminf(fmaxf(
    value, config.control_min[channel]), config.control_max[channel]);
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
  const float * weights, const State * trajectories, State * expected,
  const DeviceMppi config)
{
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t count = static_cast<std::size_t>(config.horizon + 1U) * kStateDim;
  if (index >= count) {
    return;
  }
  const std::size_t t = index / kStateDim;
  const std::size_t channel = index % kStateDim;
  const bool circular = IsCircularChannel(channel, config);
  float value = 0.0F;
  float quadrature = 0.0F;
  for (std::uint32_t sample = 0; sample < config.samples; ++sample) {
    const float entry =
      trajectories[static_cast<std::size_t>(sample) * (config.horizon + 1U) + t][channel];
    if (circular) {
      value += weights[sample] * cosf(entry);
      quadrature += weights[sample] * sinf(entry);
    } else {
      value += weights[sample] * entry;
    }
  }
  expected[t][channel] = circular ? atan2f(quadrature, value) : value;
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

__global__ void SigmaStatistics(
  const float * weights, const Control * perturbations, float * sigma_hat,
  const DeviceMppi config)
{
  const std::size_t channel = blockIdx.x * blockDim.x + threadIdx.x;
  if (channel >= kControlDim) {
    return;
  }
  float weighted = 0.0F;
  float unweighted = 0.0F;
  for (std::uint32_t sample = 0; sample < config.samples; ++sample) {
    float mean_square = 0.0F;
    for (std::uint16_t t = 0; t < config.horizon; ++t) {
      const float value = perturbations[static_cast<std::size_t>(sample) * config.horizon + t][channel];
      mean_square += value * value;
    }
    mean_square /= static_cast<float>(config.horizon);
    weighted += weights[sample] * mean_square;
    unweighted += mean_square;
  }
  unweighted /= static_cast<float>(config.samples);
  sigma_hat[channel] = config.sigma[channel] *
    sqrtf(weighted + 1.0e-12F) / sqrtf(unweighted + 1.0e-12F);
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

DeviceCosts ToDeviceCosts(const CostWeights & source) {
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
  return result;
}

}  // namespace

class CudaMppiController::Impl {
 public:
  Impl(
    MppiConfig config, const CostWeights & costs, VehicleParameters vehicle,
    const ModelKind model_kind, const Raceline & raceline,
    const IntegratorKind integrator_kind, std::string neural_engine_path,
    const float cartesian_projection_window_m)
  : config_(std::move(config)), vehicle_(vehicle), costs_(ToDeviceCosts(costs)),
    base_sigma_(config_.sigma)
  {
    if (!(cartesian_projection_window_m > 0.0F)) {
      throw std::invalid_argument("projection window must be positive");
    }
    if (config_.num_samples <= 3U || config_.horizon == 0U ||
      config_.horizon == std::numeric_limits<std::uint16_t>::max() ||
      !(config_.dt_s > 0.0F) || !(config_.lambda > 0.0F) ||
      raceline.points().size() < 2U)
    {
      throw std::invalid_argument("invalid MPPI dimensions/timing or empty raceline");
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
    if (model_kind == ModelKind::kTensorRtNeuralDerivative) {
      neural_model_ = std::make_unique<TensorRtDerivativeModel>(neural_engine_path);
    }
    device_config_.samples = config_.num_samples;
    device_config_.horizon = config_.horizon;
    device_config_.substeps = config_.integration_substeps;
    device_config_.smoothing_window = config_.noise_smoothing_window;
    device_config_.delay_steps = config_.control_delay_steps;
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
    for (std::size_t i = 0; i < kControlDim; ++i) {
      device_config_.sigma[i] = config_.sigma[i];
      device_config_.control_min[i] = config_.control_min[i];
      device_config_.control_max[i] = config_.control_max[i];
    }

    CheckCuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreate");
    CheckCuda(cudaEventCreate(&start_event_), "cudaEventCreate start");
    CheckCuda(cudaEventCreate(&stop_event_), "cudaEventCreate stop");
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
    Allocate(updated_, horizon);
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
    Allocate(finite_flags_, samples);
    Allocate(finite_count_, 1U);
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
    Free(updated_);
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
    Free(minimum_);
    Free(worst_);
    Free(sum_);
    Free(sum_squares_);
    Free(ess_);
    Free(sigma_hat_);
    Free(finite_flags_);
    Free(finite_count_);
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
    if (stream_ != nullptr) {
      cudaStreamDestroy(stream_);
    }
  }

  MppiSolution Solve(
    const State & initial_state, const ReferenceHorizon & reference,
    const Control & previous_control, const float initial_path_s_m,
    const float shift_fraction, const bool reset,
    const std::uint32_t num_visualization_rollouts)
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
        crash_latches_, sideslip_latches_, neural_s_hint_, device_config_);
      const std::uint16_t substeps = device_config_.substeps == 0U ? 1U :
        device_config_.substeps;
      const float step_dt = device_config_.dt / static_cast<float>(substeps);
      for (std::uint16_t t = 0; t < device_config_.horizon; ++t) {
        NeuralStageCostAndPack<<<sample_blocks, 256, 0, stream_>>>(
          initial_state, previous_control, neural_states_, nominal_, candidates_,
          neural_input_, costs_device_, crash_latches_, sideslip_latches_,
          neural_s_hint_, t, track_, device_reference, device_config_, costs_);
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
        sideslip_latches_, neural_s_hint_, track_, device_reference, device_config_,
        costs_);
    } else {
      RolloutAndCost<<<sample_blocks, 256, 0, stream_>>>(
        initial_state, initial_path_s_m, previous_control, nominal_, candidates_,
        trajectories_, costs_device_, track_, device_reference, device_config_,
        costs_, vehicle_);
    }
    PrepareReductionInputs<<<sample_blocks, 256, 0, stream_>>>(
      costs_device_, reduction_a_, reduction_b_, finite_flags_, config_.num_samples);
    cub::DeviceReduce::Sum(
      reduction_temp_, reduction_temp_bytes_, finite_flags_, finite_count_,
      config_.num_samples, stream_);
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
    WeightedControls<<<static_cast<int>((control_values + 127U) / 128U), 128, 0, stream_>>>(
      weights_device_, candidates_, updated_, device_config_);
    const std::size_t state_values = (horizon + 1U) * kStateDim;
    if (config_.expected_trajectory) {
      WeightedStates<<<static_cast<int>((state_values + 127U) / 128U), 128, 0, stream_>>>(
        weights_device_, trajectories_, expected_, device_config_);
    } else {
      RolloutUpdatedControls<<<1, 1, 0, stream_>>>(
        initial_state, updated_, expected_, track_, device_config_, vehicle_);
    }
    SigmaStatistics<<<1, 32, 0, stream_>>>(
      weights_device_, perturbations_, sigma_hat_, device_config_);
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
    CheckCuda(cudaMemcpyAsync(
      solution.controls.data(), updated_, horizon * sizeof(Control),
      cudaMemcpyDeviceToHost, stream_), "copy updated controls");
    CheckCuda(cudaMemcpyAsync(
      solution.states.data(), expected_, (horizon + 1U) * sizeof(State),
      cudaMemcpyDeviceToHost, stream_), "copy expected states");
    CheckCuda(cudaMemcpyAsync(&minimum, minimum_, sizeof(float), cudaMemcpyDeviceToHost, stream_),
      "copy minimum cost");
    CheckCuda(cudaMemcpyAsync(&ess, ess_, sizeof(float), cudaMemcpyDeviceToHost, stream_), "copy ESS");
    CheckCuda(cudaMemcpyAsync(
      sigma_hat.data(), sigma_hat_, kControlDim * sizeof(float),
      cudaMemcpyDeviceToHost, stream_), "copy sigma statistics");
    CheckCuda(cudaMemcpyAsync(
      &finite_count, finite_count_, sizeof(std::uint32_t),
      cudaMemcpyDeviceToHost, stream_), "copy finite rollout count");
    CheckCuda(cudaEventRecord(stop_event_, stream_), "record MPPI stop");
    CheckCuda(cudaEventSynchronize(stop_event_), "wait for MPPI solve");
    float elapsed_ms = 0.0F;
    CheckCuda(cudaEventElapsedTime(&elapsed_ms, start_event_, stop_event_), "measure MPPI solve");

    // Visualization is intentionally sampled only when requested by the ROS
    // runtime. The control solve has completed at this point, so these optional
    // copies are outside the measured solve time and cannot delay GPU kernels.
    const std::size_t rollout_count = std::min<std::size_t>(
      num_visualization_rollouts, config_.num_samples);
    if (rollout_count > 0U) {
      std::vector<float> weights(config_.num_samples);
      CheckCuda(cudaMemcpy(
        weights.data(), weights_device_, weights.size() * sizeof(float),
        cudaMemcpyDeviceToHost), "copy visualization weights");
      std::vector<std::uint32_t> indices(config_.num_samples);
      std::iota(indices.begin(), indices.end(), 0U);
      std::partial_sort(
        indices.begin(), indices.begin() + static_cast<std::ptrdiff_t>(rollout_count),
        indices.end(),
        [&weights](const std::uint32_t lhs, const std::uint32_t rhs) {
          if (weights[lhs] == weights[rhs]) {
            return lhs < rhs;
          }
          return weights[lhs] > weights[rhs];
        });
      solution.sampled_rollouts.reserve(rollout_count);
      for (std::size_t rank = 0; rank < rollout_count; ++rank) {
        const auto sample = indices[rank];
        WeightedRollout rollout;
        rollout.weight = weights[sample];
        rollout.states.resize(horizon + 1U);
        const auto * source = trajectories_ +
          static_cast<std::size_t>(sample) * (horizon + 1U);
        CheckCuda(cudaMemcpy(
          rollout.states.data(), source, rollout.states.size() * sizeof(State),
          cudaMemcpyDeviceToHost), "copy visualization rollout");
        solution.sampled_rollouts.push_back(std::move(rollout));
      }
    }

    solution.diagnostics.minimum_cost = minimum;
    solution.diagnostics.effective_sample_size = ess;
    solution.diagnostics.lambda_used = config_.lambda;
    solution.diagnostics.sigma_used = config_.sigma;
    solution.diagnostics.solve_time_ms = elapsed_ms;
    solution.diagnostics.finite_rollouts = finite_count;

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

  const MppiConfig & config() const noexcept { return config_; }

 private:
  MppiConfig config_;
  VehicleParameters vehicle_;
  DeviceCosts costs_{};
  DeviceMppi device_config_{};
  std::array<float, kControlDim> base_sigma_{};
  cudaStream_t stream_{nullptr};
  cudaEvent_t start_event_{nullptr};
  cudaEvent_t stop_event_{nullptr};
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
  Control * updated_{nullptr};
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
  float * minimum_{nullptr};
  float * worst_{nullptr};
  float * sum_{nullptr};
  float * sum_squares_{nullptr};
  float * ess_{nullptr};
  float * sigma_hat_{nullptr};
  std::uint32_t * finite_flags_{nullptr};
  std::uint32_t * finite_count_{nullptr};
  State * neural_states_{nullptr};
  float * neural_input_{nullptr};
  float * neural_derivative_{nullptr};
  std::uint8_t * crash_latches_{nullptr};
  std::uint8_t * sideslip_latches_{nullptr};
  float * neural_s_hint_{nullptr};
  std::unique_ptr<TensorRtDerivativeModel> neural_model_;
  void * reduction_temp_{nullptr};
  std::size_t reduction_temp_bytes_{};
};

CudaMppiController::CudaMppiController(
  MppiConfig config, CostWeights costs, VehicleParameters vehicle,
  const ModelKind model_kind, const Raceline & raceline,
  const IntegratorKind integrator_kind, std::string neural_engine_path,
  const float projection_window_m)
: impl_(std::make_unique<Impl>(
    std::move(config), costs, vehicle, model_kind, raceline,
    integrator_kind, std::move(neural_engine_path), projection_window_m)) {}

CudaMppiController::~CudaMppiController() = default;
CudaMppiController::CudaMppiController(CudaMppiController &&) noexcept = default;
CudaMppiController & CudaMppiController::operator=(CudaMppiController &&) noexcept = default;

MppiSolution CudaMppiController::Solve(
  const State & initial_state, const ReferenceHorizon & reference,
  const Control & previous_control, const float initial_path_s_m,
  const float shift_fraction, const bool reset,
  const std::uint32_t num_visualization_rollouts)
{
  return impl_->Solve(
    initial_state, reference, previous_control, initial_path_s_m, shift_fraction,
    reset, num_visualization_rollouts);
}

const MppiConfig & CudaMppiController::config() const noexcept { return impl_->config(); }
bool CudaMppiController::using_cuda() const noexcept { return true; }

}  // namespace xxcar::mppi
