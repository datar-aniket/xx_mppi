#pragma once

#include <vector>

#include "xx_mppi/types.hpp"

namespace xxcar::mppi {

class CostEvaluator {
 public:
  explicit CostEvaluator(CostWeights weights) : weights_(weights) {}

  [[nodiscard]] float Evaluate(
    const std::vector<State> & states, const std::vector<Control> & controls,
    const ReferenceHorizon & reference, const Control & previous_control) const;

  [[nodiscard]] const CostWeights & weights() const noexcept { return weights_; }

 private:
  [[nodiscard]] static float InterpolateByS(
    const std::vector<float> & s_grid, const std::vector<float> & values, float s);

  CostWeights weights_;
};

}  // namespace xxcar::mppi
