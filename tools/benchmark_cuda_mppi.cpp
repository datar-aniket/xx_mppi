#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "xx_mppi/controller/cuda_mppi.hpp"
#include "xx_mppi/reference/raceline.hpp"

namespace {

float Percentile(const std::vector<float> & sorted, const float fraction) {
  const auto index = static_cast<std::size_t>(
    fraction * static_cast<float>(sorted.size() - 1U));
  return sorted[index];
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
    xxcar::mppi::CudaMppiController controller(
      config, costs, vehicle, xxcar::mppi::ObstacleConfig{},
      xxcar::mppi::ModelKind::kDynamicBicycleFiala, raceline);
    const auto reference = raceline.Sample(
      raceline.s_min(), config.horizon, config.dt_s);
    xxcar::mppi::State initial = reference.states.front();
    const xxcar::mppi::Control previous = reference.controls.front();

    for (std::size_t i = 0; i < 20U; ++i) {
      (void)controller.Solve(
        initial, reference, previous, reference.s_grid.front(), 0.1F, i == 0U);
    }
    std::vector<float> gpu_times;
    gpu_times.reserve(iterations);
    float minimum_ess = static_cast<float>(config.num_samples);
    std::uint32_t minimum_finite = config.num_samples;
    const auto wall_start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < iterations; ++i) {
      const auto solution = controller.Solve(
        initial, reference, previous, reference.s_grid.front(), 0.1F, false);
      gpu_times.push_back(solution.diagnostics.solve_time_ms);
      minimum_ess = std::min(minimum_ess, solution.diagnostics.effective_sample_size);
      minimum_finite = std::min(minimum_finite, solution.diagnostics.finite_rollouts);
    }
    const auto wall_end = std::chrono::steady_clock::now();
    std::sort(gpu_times.begin(), gpu_times.end());
    const double wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
    std::cout << "K=" << config.num_samples << " T=" << config.horizon
      << " dt=" << config.dt_s << " iterations=" << iterations << '\n'
      << "GPU ms median=" << Percentile(gpu_times, 0.50F)
      << " p95=" << Percentile(gpu_times, 0.95F)
      << " p99=" << Percentile(gpu_times, 0.99F)
      << " max=" << gpu_times.back() << '\n'
      << "wall ms/solve=" << wall_ms / static_cast<double>(iterations)
      << " minimum_ESS=" << minimum_ess
      << " minimum_finite=" << minimum_finite << '\n'
      << "100Hz p99 target: "
      << (Percentile(gpu_times, 0.99F) <= 10.0F ? "PASS" : "FAIL") << '\n';
    return 0;
  } catch (const std::exception & error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
