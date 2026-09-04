#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "xx_mppi/obstacles/signed_distance_field.hpp"

int main(int argc, char ** argv) {
  const int iterations = argc > 1 ? std::max(std::atoi(argv[1]), 1) : 200;
  xxcar::mppi::ObstacleConfig config;
  config.enabled = true;
  xxcar::mppi::SignedDistanceFieldBuilder builder(config);
  std::vector<xxcar::mppi::Point2D> points;
  points.reserve(1080U);
  constexpr float pi = 3.14159265358979323846F;
  for (std::size_t i = 0; i < 1080U; ++i) {
    const float angle = 2.0F * pi * static_cast<float>(i) / 1080.0F;
    points.push_back({4.0F * std::cos(angle), 4.0F * std::sin(angle)});
  }
  std::vector<double> elapsed_ms;
  elapsed_ms.reserve(static_cast<std::size_t>(iterations));
  for (int iteration = 0; iteration < iterations; ++iteration) {
    const auto start = std::chrono::steady_clock::now();
    const auto field = builder.Build(points, {}, iteration + 1, iteration + 1U);
    const auto stop = std::chrono::steady_clock::now();
    if (!field.valid()) {
      std::cerr << "invalid field\n";
      return 1;
    }
    elapsed_ms.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
  }
  std::sort(elapsed_ms.begin(), elapsed_ms.end());
  const auto percentile = [&elapsed_ms](const double fraction) {
      const auto index = static_cast<std::size_t>(fraction *
        static_cast<double>(elapsed_ms.size() - 1U));
      return elapsed_ms[index];
    };
  std::cout << "SDF " << config.grid_width_m << "x" << config.grid_height_m
    << " m @ " << config.grid_resolution_m << " m, 1080 returns: median="
    << percentile(0.50) << " ms p95=" << percentile(0.95)
    << " ms p99=" << percentile(0.99) << " ms\n";
  return 0;
}
