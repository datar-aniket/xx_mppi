#pragma once

#include <memory>
#include <string>

#include "xx_mppi/costs/cost_evaluator.hpp"
#include "xx_mppi/dynamics/model.hpp"
#include "xx_mppi/dynamics/integrator.hpp"
#include "xx_mppi/reference/raceline.hpp"
#include "xx_mppi/types.hpp"

namespace xxcar::mppi {

class CudaMppiController {
 public:
  CudaMppiController(
    MppiConfig config, CostWeights costs, VehicleParameters vehicle,
    ObstacleConfig obstacle_config, ModelKind model_kind, const Raceline & raceline,
    IntegratorKind integrator_kind = IntegratorKind::kEuler,
    std::string neural_engine_path = {},
    float projection_window_m = 30.0F);
  ~CudaMppiController();

  CudaMppiController(const CudaMppiController &) = delete;
  CudaMppiController & operator=(const CudaMppiController &) = delete;
  CudaMppiController(CudaMppiController &&) noexcept;
  CudaMppiController & operator=(CudaMppiController &&) noexcept;

  // shift_fraction is elapsed controller wall time / planning dt. Setting reset
  // seeds the nominal sequence from reference feed-forward controls.
  // initial_path_s_m is the loop-continuous arc length of the initial state. In
  // the Frenet frame it equals initial_state[kPathEvolution]; in the Cartesian
  // frame it seeds the on-device reprojection so s stays continuous.
  // capture_cost_terms decomposes the published trajectory's cost into its
  // individual terms and returns it in the diagnostics. It runs outside the
  // measured solve, so callers should still request it well below the solve
  // rate rather than on every cycle.
  [[nodiscard]] MppiSolution Solve(
    const State & initial_state, const ReferenceHorizon & reference,
    const Control & previous_control, float initial_path_s_m, float shift_fraction,
    bool reset, std::uint32_t num_visualization_rollouts = 0U,
    bool capture_cost_terms = false);
  void UpdateObstacleField(const ObstacleField & field);
  void ClearObstacleField();

  [[nodiscard]] const MppiConfig & config() const noexcept;
  [[nodiscard]] bool using_cuda() const noexcept;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace xxcar::mppi
