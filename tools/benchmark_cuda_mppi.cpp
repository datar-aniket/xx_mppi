#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "xx_mppi/controller/config.hpp"
#include "xx_mppi/controller/cuda_mppi.hpp"
#include "xx_mppi/dynamics/analytic_dynamics.hpp"
#include "xx_mppi/dynamics/rollout.hpp"
#include "xx_mppi/reference/raceline.hpp"

namespace {

float Percentile(const std::vector<float> & sorted, const float fraction) {
  const auto index = static_cast<std::size_t>(
    fraction * static_cast<float>(sorted.size() - 1U));
  return sorted[index];
}

float MaximumDynamicsResidual(
  const xxcar::mppi::MppiSolution & solution,
  const xxcar::mppi::AnalyticDynamics & dynamics,
  const xxcar::mppi::Raceline & raceline,
  const xxcar::mppi::MppiConfig & config,
  const xxcar::mppi::IntegratorKind integrator)
{
  if (solution.states.empty() || solution.controls.empty()) {
    return 0.0F;
  }
  const auto rerolled = xxcar::mppi::RolloutAnalytic(
    dynamics, raceline, solution.states.front(), solution.controls,
    config.dt_s, config.integration_substeps, integrator);
  float maximum = 0.0F;
  for (std::size_t time = 0U; time < rerolled.size(); ++time) {
    for (std::size_t channel = 0U; channel < xxcar::mppi::kStateDim; ++channel) {
      maximum = std::max(maximum, std::abs(
        solution.states[time][channel] - rerolled[time][channel]));
    }
  }
  return maximum;
}

}  // namespace

int main(int argc, char ** argv) {
  if (argc < 2 || argc > 3) {
    std::cerr << "usage: xxcar_benchmark_mppi RACELINE.csv [ITERATIONS=500]\n";
    return 2;
  }
  try {
    const std::size_t iterations = argc == 3 ?
      static_cast<std::size_t>(std::stoul(argv[2])) : 500U;
    if (iterations < 10U) {
      throw std::invalid_argument("ITERATIONS must be at least 10");
    }
    const auto raceline = xxcar::mppi::Raceline::LoadCsv(argv[1]);
    xxcar::mppi::MppiConfig config;
    xxcar::mppi::CostWeights costs;
    xxcar::mppi::VehicleParameters vehicle;
    xxcar::mppi::ObstacleConfig obstacles;
    auto model_kind = xxcar::mppi::ModelKind::kDynamicBicycleFiala;
    auto integrator = xxcar::mppi::IntegratorKind::kEuler;
    if (const char * config_directory = std::getenv("XX_MPPI_BENCH_CONFIG_DIR")) {
      const auto loaded = xxcar::mppi::LoadControllerConfig(config_directory);
      config = loaded.mppi;
      costs = loaded.costs;
      vehicle = loaded.vehicle;
      obstacles = loaded.obstacles;
      model_kind = loaded.model_kind;
      integrator = loaded.integrator;
    }
    if (std::getenv("XX_MPPI_BENCH_REFINEMENT") != nullptr) {
      config.refinement.enabled = true;
    }
    if (std::getenv("XX_MPPI_BENCH_DISABLE_REFINEMENT") != nullptr) {
      config.refinement.enabled = false;
    }
    if (const char * value = std::getenv("XX_MPPI_BENCH_SQP_ITERATIONS")) {
      config.refinement.sqp_iterations = static_cast<std::uint16_t>(std::stoul(value));
    }
    if (const char * value = std::getenv("XX_MPPI_BENCH_PCG_ITERATIONS")) {
      config.refinement.pcg_iterations = static_cast<std::uint16_t>(std::stoul(value));
    }
    if (const char * value = std::getenv("XX_MPPI_BENCH_FD_RELATIVE_STEP")) {
      config.refinement.finite_difference_relative_step = std::stof(value);
    }
    if (const char * value = std::getenv("XX_MPPI_BENCH_LAMBDA")) {
      config.lambda = std::stof(value);
    }
    if (std::getenv("XX_MPPI_BENCH_DISABLE_ADAPTIVE_SIGMA") != nullptr) {
      config.adaptation.adaptive_sigma = false;
    }
    if (const char * value = std::getenv("XX_MPPI_BENCH_SEED")) {
      config.seed = static_cast<std::uint64_t>(std::stoull(value));
    }
    std::uint32_t visualization_rollouts = 0U;
    if (const char * requested = std::getenv("XX_MPPI_BENCH_VISUALIZATION_ROLLOUTS")) {
      visualization_rollouts = static_cast<std::uint32_t>(std::stoul(requested));
    }
    const bool collect_visualization =
      std::getenv("XX_MPPI_BENCH_COLLECT_VISUALIZATION") != nullptr;
    const bool closed_loop = std::getenv("XX_MPPI_BENCH_CLOSED_LOOP") != nullptr;
    if (closed_loop && config.frame != xxcar::mppi::FrameKind::kFrenet) {
      throw std::invalid_argument("closed-loop benchmark currently requires frame=frenet");
    }
    xxcar::mppi::CudaMppiController controller(
      config, costs, vehicle, obstacles, model_kind, raceline, integrator);
    auto reference = raceline.Sample(
      raceline.s_min(), config.horizon, config.dt_s);
    xxcar::mppi::State initial = reference.states.front();
    if (const char * value = std::getenv("XX_MPPI_BENCH_INITIAL_LATERAL_M")) {
      initial[xxcar::mppi::kLateralDeviation] += std::stof(value);
    }
    if (const char * value = std::getenv("XX_MPPI_BENCH_INITIAL_HEADING_RAD")) {
      initial[xxcar::mppi::kRelativeHeading] += std::stof(value);
    }
    if (const char * value = std::getenv("XX_MPPI_BENCH_INITIAL_SPEED_MPS")) {
      initial[xxcar::mppi::kSpeed] = std::stof(value);
    }
    xxcar::mppi::Control previous = reference.controls.front();
    const xxcar::mppi::AnalyticDynamics dynamics(model_kind, vehicle, config.frame);

    for (std::size_t i = 0; i < 20U; ++i) {
      (void)controller.Solve(
        initial, reference, previous, reference.s_grid.front(), 0.1F, i == 0U);
    }
    std::vector<float> gpu_times;
    gpu_times.reserve(iterations);
    std::vector<float> effective_sample_sizes;
    effective_sample_sizes.reserve(iterations);
    float minimum_ess = static_cast<float>(config.num_samples);
    std::uint32_t minimum_finite = config.num_samples;
    std::size_t refinement_accepts = 0U;
    double refinement_cost_before_sum = 0.0;
    double refinement_cost_after_sum = 0.0;
    double refinement_merit_before_sum = 0.0;
    double refinement_merit_after_sum = 0.0;
    std::array<double, xxcar::mppi::kStateDim> state_delta_rms_square_sum{};
    std::array<float, xxcar::mppi::kStateDim> state_delta_max{};
    std::array<double, xxcar::mppi::kControlDim> control_delta_rms_square_sum{};
    std::array<float, xxcar::mppi::kControlDim> control_delta_max{};
    std::array<double, xxcar::mppi::kControlDim> first_control_delta_sum{};
    std::array<float, xxcar::mppi::kControlDim> first_control_delta_max{};
    std::uint64_t refinement_sqp_iterations = 0U;
    std::uint16_t maximum_refinement_sqp_iterations = 0U;
    std::uint16_t maximum_refinement_pcg_iterations = 0U;
    float maximum_returned_dynamics_residual = 0.0F;
    float maximum_steering_rate = 0.0F;
    float maximum_torque_rate = 0.0F;
    xxcar::mppi::Control prior_command{};
    bool have_prior_command = false;
    float maximum_refinement_ms = 0.0F;
    float maximum_refinement_residual = 0.0F;
    float final_lambda = config.lambda;
    float last_refinement_cost_before = 0.0F;
    float last_refinement_cost_after = 0.0F;
    float last_refinement_residual_before = 0.0F;
    float last_refinement_residual = 0.0F;
    std::array<float, xxcar::mppi::kMpcLineSearchCandidates> last_trial_merits{};
    std::array<float, xxcar::mppi::kMpcLineSearchCandidates> last_trial_residuals{};
    const auto wall_start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < iterations; ++i) {
      const auto solution = controller.Solve(
        initial, reference, previous, reference.s_grid.front(), 0.1F, false,
        visualization_rollouts);
      if (collect_visualization && visualization_rollouts > 0U) {
        const auto rollouts = controller.CollectVisualization(
          solution.visualization_snapshot_id);
        if (rollouts.size() != std::min<std::size_t>(
            visualization_rollouts, config.num_samples))
        {
          throw std::runtime_error("asynchronous visualization snapshot was dropped");
        }
      }
      gpu_times.push_back(solution.diagnostics.solve_time_ms);
      effective_sample_sizes.push_back(solution.diagnostics.effective_sample_size);
      minimum_ess = std::min(minimum_ess, solution.diagnostics.effective_sample_size);
      minimum_finite = std::min(minimum_finite, solution.diagnostics.finite_rollouts);
      refinement_accepts += solution.diagnostics.refinement_accepted ? 1U : 0U;
      if (solution.diagnostics.refinement_accepted) {
        refinement_cost_before_sum += solution.diagnostics.refinement_cost_before;
        refinement_cost_after_sum += solution.diagnostics.refinement_cost_after;
        refinement_merit_before_sum += solution.diagnostics.refinement_merit_before;
        refinement_merit_after_sum += solution.diagnostics.refinement_merit_after;
        for (std::size_t channel = 0U; channel < xxcar::mppi::kStateDim; ++channel) {
          const double rms = solution.diagnostics.refinement_state_rms_delta[channel];
          state_delta_rms_square_sum[channel] += rms * rms;
          state_delta_max[channel] = std::max(
            state_delta_max[channel],
            solution.diagnostics.refinement_state_max_delta[channel]);
        }
        for (std::size_t channel = 0U; channel < xxcar::mppi::kControlDim; ++channel) {
          const double rms = solution.diagnostics.refinement_control_rms_delta[channel];
          control_delta_rms_square_sum[channel] += rms * rms;
          control_delta_max[channel] = std::max(
            control_delta_max[channel],
            solution.diagnostics.refinement_control_max_delta[channel]);
          first_control_delta_sum[channel] +=
            solution.diagnostics.refinement_first_control_delta[channel];
          first_control_delta_max[channel] = std::max(
            first_control_delta_max[channel],
            solution.diagnostics.refinement_first_control_delta[channel]);
        }
      }
      refinement_sqp_iterations += solution.diagnostics.refinement_iterations;
      maximum_refinement_sqp_iterations = std::max(
        maximum_refinement_sqp_iterations, solution.diagnostics.refinement_iterations);
      maximum_refinement_pcg_iterations = std::max(
        maximum_refinement_pcg_iterations, solution.diagnostics.refinement_pcg_iterations);
      if (solution.diagnostics.refinement_accepted) {
        maximum_returned_dynamics_residual = std::max(
          maximum_returned_dynamics_residual,
          MaximumDynamicsResidual(solution, dynamics, raceline, config, integrator));
      }
      if (!solution.controls.empty()) {
        if (have_prior_command) {
          maximum_steering_rate = std::max(maximum_steering_rate, std::abs(
            solution.controls.front()[xxcar::mppi::kSteering] -
            prior_command[xxcar::mppi::kSteering]) / config.dt_s);
          maximum_torque_rate = std::max(maximum_torque_rate, std::abs(
            solution.controls.front()[xxcar::mppi::kWheelTorque] -
            prior_command[xxcar::mppi::kWheelTorque]) / config.dt_s);
        }
        prior_command = solution.controls.front();
        have_prior_command = true;
      }
      maximum_refinement_ms = std::max(
        maximum_refinement_ms, solution.diagnostics.refinement_time_ms);
      maximum_refinement_residual = std::max(
        maximum_refinement_residual,
        solution.diagnostics.refinement_constraint_residual);
      final_lambda = solution.diagnostics.lambda_used;
      last_refinement_cost_before = solution.diagnostics.refinement_cost_before;
      last_refinement_cost_after = solution.diagnostics.refinement_cost_after;
      last_refinement_residual_before =
        solution.diagnostics.refinement_constraint_residual_before;
      last_refinement_residual = solution.diagnostics.refinement_constraint_residual;
      last_trial_merits = solution.diagnostics.refinement_trial_merits;
      last_trial_residuals = solution.diagnostics.refinement_trial_constraint_residuals;
      if (closed_loop && solution.states.size() > 1U && !solution.controls.empty()) {
        initial = solution.states[1U];
        previous = solution.controls.front();
        reference = raceline.Sample(
          initial[xxcar::mppi::kPathEvolution], config.horizon, config.dt_s);
      }
    }
    const auto wall_end = std::chrono::steady_clock::now();
    std::sort(gpu_times.begin(), gpu_times.end());
    std::sort(effective_sample_sizes.begin(), effective_sample_sizes.end());
    const double wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
    std::cout << "K=" << config.num_samples << " T=" << config.horizon
      << " refinement=" << (config.refinement.enabled ? "on" : "off")
      << " SQP=" << config.refinement.sqp_iterations
      << " PCG=" << config.refinement.pcg_iterations
      << " adaptive_sigma=" << (config.adaptation.adaptive_sigma ? "on" : "off")
      << " closed_loop=" << (closed_loop ? "on" : "off")
      << " visualization_rollouts=" << visualization_rollouts
      << " dt=" << config.dt_s << " iterations=" << iterations << '\n'
      << "GPU ms median=" << Percentile(gpu_times, 0.50F)
      << " p95=" << Percentile(gpu_times, 0.95F)
      << " p99=" << Percentile(gpu_times, 0.99F)
      << " max=" << gpu_times.back() << '\n'
      << "wall ms/solve=" << wall_ms / static_cast<double>(iterations)
      << " minimum_ESS=" << minimum_ess
      << " median_ESS=" << Percentile(effective_sample_sizes, 0.50F)
      << " final_lambda=" << final_lambda
      << " minimum_finite=" << minimum_finite
      << " refinement_accepts=" << refinement_accepts << '/' << iterations
      << " SQP_actual_mean=" << static_cast<double>(refinement_sqp_iterations) /
        static_cast<double>(iterations)
      << " SQP_actual_max=" << maximum_refinement_sqp_iterations
      << " PCG_actual_max=" << maximum_refinement_pcg_iterations
      << " refinement_max_ms=" << maximum_refinement_ms
      << " refinement_max_residual=" << maximum_refinement_residual
      << " returned_dynamics_max_residual=" << maximum_returned_dynamics_residual
      << " command_rate_max=[steer " << maximum_steering_rate
      << " rad/s, torque " << maximum_torque_rate << " Nm/s]"
      << " refinement_last=[cost " << last_refinement_cost_before << " -> "
      << last_refinement_cost_after << ", residual "
      << last_refinement_residual_before << " -> " << last_refinement_residual << "]\n";
    if (config.refinement.enabled) {
      std::cout << "line_search merit/residual:";
      for (std::size_t candidate = 0U; candidate < last_trial_merits.size(); ++candidate) {
        std::cout << ' ' << last_trial_merits[candidate] << '/'
          << last_trial_residuals[candidate];
      }
      std::cout << '\n';
      if (refinement_accepts > 0U) {
        const double accepted = static_cast<double>(refinement_accepts);
        std::cout << "accepted SQP vs pre-SQP MPPI: mean_cost="
          << refinement_cost_before_sum / accepted << " -> "
          << refinement_cost_after_sum / accepted
          << " mean_merit=" << refinement_merit_before_sum / accepted << " -> "
          << refinement_merit_after_sum / accepted << '\n';
        std::cout << "state_delta order=" <<
          (config.frame == xxcar::mppi::FrameKind::kFrenet ?
            "[yaw_rate,speed,sideslip,wheel_speed,lateral,heading,path_s]" :
            "[yaw_rate,speed,sideslip,wheel_speed,east,north,heading]")
          << " rms=[";
        for (std::size_t channel = 0U; channel < state_delta_rms_square_sum.size(); ++channel) {
          std::cout << (channel == 0U ? "" : ",")
            << std::sqrt(state_delta_rms_square_sum[channel] / accepted);
        }
        std::cout << "] max=[";
        for (std::size_t channel = 0U; channel < state_delta_max.size(); ++channel) {
          std::cout << (channel == 0U ? "" : ",") << state_delta_max[channel];
        }
        std::cout << "]\ncontrol_delta order=[steering,torque,rear_steering] rms=[";
        for (std::size_t channel = 0U; channel < control_delta_rms_square_sum.size(); ++channel) {
          std::cout << (channel == 0U ? "" : ",")
            << std::sqrt(control_delta_rms_square_sum[channel] / accepted);
        }
        std::cout << "] max=[";
        for (std::size_t channel = 0U; channel < control_delta_max.size(); ++channel) {
          std::cout << (channel == 0U ? "" : ",") << control_delta_max[channel];
        }
        std::cout << "] first_mean_abs=[";
        for (std::size_t channel = 0U; channel < first_control_delta_sum.size(); ++channel) {
          std::cout << (channel == 0U ? "" : ",")
            << first_control_delta_sum[channel] / accepted;
        }
        std::cout << "] first_max_abs=[";
        for (std::size_t channel = 0U; channel < first_control_delta_max.size(); ++channel) {
          std::cout << (channel == 0U ? "" : ",") << first_control_delta_max[channel];
        }
        std::cout << "]\n";
      }
    }
    std::cout << "100Hz p99 target: "
      << (Percentile(gpu_times, 0.99F) <= 10.0F ? "PASS" : "FAIL") << '\n';
    return 0;
  } catch (const std::exception & error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
