#pragma once

#include <vector>

#include "xx_mppi/reference/raceline.hpp"
#include "xx_mppi/types.hpp"

namespace xxcar::mppi {

class CostEvaluator {
 public:
  CostEvaluator(CostWeights weights, float dt_s);

  // When terms is non-null it is overwritten with the per-term decomposition of
  // the returned cost. The entries sum to the return value, which is what the
  // CUDA breakdown is checked against. The CUDA solver adds one further term,
  // the importance-sampling correction, which has no counterpart here because
  // this evaluator does not know the sampling distribution.
  [[nodiscard]] float Evaluate(
    const std::vector<State> & states, const std::vector<Control> & controls,
    const ReferenceHorizon & reference, const Control & previous_control,
    const ObstacleField * obstacle_field = nullptr,
    const Raceline * raceline = nullptr,
    const ObstacleConfig * obstacle_config = nullptr,
    CostTerms * terms = nullptr) const;

  [[nodiscard]] const CostWeights & weights() const noexcept { return weights_; }

 private:
  [[nodiscard]] static float InterpolateByS(
    const std::vector<float> & s_grid, const std::vector<float> & values, float s);

  CostWeights weights_;
  float dt_s_;
};

}  // namespace xxcar::mppi
