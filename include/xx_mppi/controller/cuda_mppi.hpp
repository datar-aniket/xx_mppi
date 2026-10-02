#pragma once

#include <memory>
#include <optional>
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
  // num_visualization_rollouts and capture_cost_terms stage a capture of the
  // highest-weight rollouts and of the per-term decomposition of the returned
  // trajectory's cost. Staging only enqueues GPU work behind the solve on a
  // second stream; Solve never waits for it. Only one capture is outstanding at
  // a time: while one is uncollected, further requests are ignored and the
  // solution's capture_id stays zero.
  [[nodiscard]] MppiSolution Solve(
    const State & initial_state, const ReferenceHorizon & reference,
    const Control & previous_control, float initial_path_s_m, float shift_fraction,
    bool reset, std::uint32_t num_visualization_rollouts = 0U,
    bool capture_cost_terms = false);
  // Blocks until the outstanding capture is complete, returns it and frees the
  // slot for the next one. Safe to call from a thread other than the one
  // calling Solve. Returns nullopt when no capture is outstanding.
  [[nodiscard]] std::optional<MppiCapture> CollectCapture();
  void UpdateObstacleField(const ObstacleField & field);
  void ClearObstacleField();
  // Takes effect from the next Solve. Call from the solver thread.
  void SetVelocityOverspeedMultiplier(float multiplier) noexcept;
  // Physical parameters the rollouts use from the next Solve (the online
  // friction estimate). Call from the solver thread.
  void SetVehicleParameters(const VehicleParameters & vehicle) noexcept;

  [[nodiscard]] const MppiConfig & config() const noexcept;
  [[nodiscard]] bool using_cuda() const noexcept;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace xxcar::mppi
