#pragma once

#include <stdexcept>

#include "xx_mppi/dynamics/frames.hpp"
#include "xx_mppi/dynamics/model.hpp"
#include "xx_mppi/dynamics/models/dynamic_bicycle_fiala.hpp"
#include "xx_mppi/dynamics/models/dynamic_bicycle_fiala_4ws.hpp"
#include "xx_mppi/dynamics/models/kinematic_bicycle.hpp"
#include "xx_mppi/types.hpp"

namespace xxcar::mppi {

class AnalyticDynamics {
 public:
  AnalyticDynamics(
    ModelKind kind, const VehicleParameters & parameters,
    FrameKind frame = FrameKind::kFrenet)
  : kind_(kind), frame_(frame), kinematic_(parameters), fiala_(parameters),
    fiala_4ws_(parameters) {
    if (kind_ == ModelKind::kTensorRtNeuralDerivative) {
      throw std::invalid_argument("TensorRT neural dynamics is not an analytic model");
    }
  }

  // curvature_inv_m is only consulted in the Frenet frame; the Cartesian frame
  // integrates ENU position and heading and ignores it.
  [[nodiscard]] StateDerivative Derivative(
    const State & state, const Control & control, float curvature_inv_m) const
  {
    const BodyState body{
      state[kYawRate], state[kSpeed], state[kSideslip], state[kDrivenWheelSpeed]};
    BodyDerivative body_derivative{};
    if (kind_ == ModelKind::kKinematicBicycle) {
      body_derivative = kinematic_.Derivative(body, control);
    } else if (kind_ == ModelKind::kDynamicBicycleFiala4ws) {
      body_derivative = fiala_4ws_.Derivative(body, control);
    } else {
      body_derivative = fiala_.Derivative(body, control);
    }
    return frame_ == FrameKind::kCartesian ?
      CartesianDerivative(state, body_derivative) :
      FrenetDerivative(state, body_derivative, curvature_inv_m);
  }

  [[nodiscard]] ModelKind kind() const noexcept { return kind_; }
  [[nodiscard]] FrameKind frame() const noexcept { return frame_; }

 private:
  ModelKind kind_;
  FrameKind frame_;
  KinematicBicycle kinematic_;
  DynamicBicycleFiala fiala_;
  DynamicBicycleFiala4ws fiala_4ws_;
};

}  // namespace xxcar::mppi
