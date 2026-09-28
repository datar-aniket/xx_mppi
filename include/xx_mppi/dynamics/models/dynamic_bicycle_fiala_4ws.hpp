#pragma once

#include <cmath>

#include "xx_mppi/dynamics/model.hpp"
#include "xx_mppi/dynamics/tires.hpp"

#if defined(__CUDACC__)
#define XXCAR_MODEL_4WS_HD __host__ __device__
#else
#define XXCAR_MODEL_4WS_HD
#endif

namespace xxcar::mppi {

// Dynamic bicycle with a Fiala brush tire model and an independently steered
// rear axle. This is a deliberate copy of DynamicBicycleFiala rather than a
// generalization of it: the front-steer-only model stays the car's known-good
// fallback, so a mistake here cannot reach a two-wheel-steering run. The
// zero-rear-steer equivalence test is what holds the two in agreement.
//
// Setting control[kRearSteering] to zero reduces every equation below to the
// front-steer-only form exactly, with no small-angle approximation.
// Tire state behind one Derivative evaluation, for host-side diagnostics such as
// the grip monitor. Forces are in each axle's own wheel frame; body_a*_mps2 are
// the body-frame specific forces an accelerometer at the CG would read.
struct TireReport {
  TireForces front{};
  TireForces rear{};
  float front_load_n{};
  float rear_load_n{};
  float front_slip_angle_rad{};
  float rear_slip_angle_rad{};
  float front_slip_ratio{};
  float rear_slip_ratio{};
  float body_ax_mps2{};
  float body_ay_mps2{};
};

class DynamicBicycleFiala4ws {
 public:
  explicit XXCAR_MODEL_4WS_HD DynamicBicycleFiala4ws(const VehicleParameters & parameters)
  : parameters_(parameters) {}

  [[nodiscard]] XXCAR_MODEL_4WS_HD BodyDerivative Derivative(
    const BodyState & state, const Control & control) const
  {
    return Evaluate(state, control, nullptr);
  }

  // Derivative plus, when report is non-null, the tire state that produced it.
  // The rollout kernel only calls Derivative, where report is a constant null
  // and the reporting branch compiles away.
  [[nodiscard]] XXCAR_MODEL_4WS_HD BodyDerivative Evaluate(
    const BodyState & state, const Control & control, TireReport * report) const
  {
    constexpr float gravity = 9.81F;
    constexpr float minimum_longitudinal_speed = 1.0F;
    const float mass = fmaxf(parameters_.mass_kg, 1.0e-6F);
    const float yaw_inertia = fmaxf(parameters_.yaw_inertia_kgm2, 1.0e-6F);
    const float a = parameters_.cg_to_front_m;
    const float b = parameters_.cg_to_rear_m;
    const float wheelbase = fmaxf(a + b, 1.0e-6F);
    const float static_front_load = mass * gravity * b / wheelbase;
    const float static_rear_load = mass * gravity * a / wheelbase;

    const float yaw_rate = state[kYawRate];
    const float speed_input = state[kSpeed];
    // The Fiala slip-angle and beta-rate equations are singular around zero
    // speed. Preserve direction while applying the requested 1 m/s model
    // floor: (0, 1) -> +1 and (-1, 0) -> -1. Positive zero selects +1;
    // negative zero selects -1. The actual state is not overwritten.
    const float speed = copysignf(
      fmaxf(fabsf(speed_input), minimum_longitudinal_speed), speed_input);
    const float sideslip = fabsf(speed_input) > 1.0e-3F ? state[kSideslip] : 0.0F;
    const float wheel_speed = state[kDrivenWheelSpeed];
    const float steering = control[kSteering];
    const float rear_steering = control[kRearSteering];
    const float torque = control[kWheelTorque];

    const float cos_beta = cosf(sideslip);
    const float sin_beta = sinf(sideslip);
    const float cos_delta = cosf(steering);
    const float sin_delta = sinf(steering);
    const float cos_delta_minus_beta = cosf(steering - sideslip);
    const float sin_delta_minus_beta = sinf(steering - sideslip);
    const float cos_delta_rear = cosf(rear_steering);
    const float sin_delta_rear = sinf(rear_steering);
    const float cos_delta_rear_minus_beta = cosf(rear_steering - sideslip);
    const float sin_delta_rear_minus_beta = sinf(rear_steering - sideslip);
    const float longitudinal_speed = speed * cos_beta;
    const float lateral_speed = speed * sin_beta;
    // Slip ANGLES need the floored speed, because atan2 against a near-zero
    // longitudinal speed is ill-conditioned. Slip RATIOS must not use it: the
    // floor would make a stationary car with stationary wheels read as locked
    // wheels rolling at the floor speed, which is a full braking force at
    // standstill. Their denominators carry the floor on their own.
    const float true_longitudinal_speed = speed_input * cos_beta;
    const float front_slip_angle =
      atan2f(lateral_speed + a * yaw_rate, longitudinal_speed) - steering;
    const float rear_slip_angle =
      atan2f(lateral_speed - b * yaw_rate, longitudinal_speed) - rear_steering;

    // A locked TRX-4 driveline constrains all wheels to one effective angular
    // speed. Resolve each contact-patch velocity into its own steered wheel
    // frame before forming that axle's longitudinal slip ratio. With the rear
    // axle steered, the rear needs this rotation just as the front does.
    const float front_lateral_speed = lateral_speed + a * yaw_rate;
    const float front_longitudinal_speed =
      cos_delta * true_longitudinal_speed + sin_delta * front_lateral_speed;
    const float front_slip_denominator = fmaxf(
      fabsf(front_longitudinal_speed), minimum_longitudinal_speed);
    const float front_slip_ratio =
      (wheel_speed - front_longitudinal_speed) / front_slip_denominator;

    const float rear_lateral_speed = lateral_speed - b * yaw_rate;
    const float rear_longitudinal_speed =
      cos_delta_rear * true_longitudinal_speed + sin_delta_rear * rear_lateral_speed;
    const float rear_slip_denominator = fmaxf(
      fabsf(rear_longitudinal_speed), minimum_longitudinal_speed);
    const float rear_slip_ratio =
      (wheel_speed - rear_longitudinal_speed) / rear_slip_denominator;

    // The xxCar command is physical driven-wheel torque. Negative torque can
    // optionally be split toward the front using front_brake_bias; the default
    // zero bias represents a single rear VESC/regenerative brake.
    const float negative_torque = fminf(torque, 0.0F);
    const float front_brake_torque = parameters_.front_brake_bias * negative_torque;
    const float rear_torque = fmaxf(torque, 0.0F) +
      (1.0F - parameters_.front_brake_bias) * negative_torque;
    const float low_speed_fade = clampf(true_longitudinal_speed / 2.0F, 0.0F, 1.0F);
    const float requested_front_force =
      front_brake_torque / fmaxf(parameters_.wheel_radius_m, 1.0e-6F) * low_speed_fade;

    // Rolling resistance acts along the velocity, so it only enters the speed
    // equation and the body acceleration. tanh fades it through zero speed
    // instead of switching sign.
    const float rolling_resistance = parameters_.rolling_resistance_n *
      tanhf(speed_input / kRollingResistanceBlendMps);

    const auto front_tire = [&](const float load) {
        if (parameters_.locked_awd) {
          return FialaCombinedSlip(
            front_slip_angle, front_slip_ratio, parameters_.front_cornering_stiffness_nprad,
            parameters_.front_friction_coefficient, load);
        }
        return FialaSimpleCoupling(
          front_slip_angle, parameters_.front_cornering_stiffness_nprad,
          parameters_.front_friction_coefficient, load, requested_front_force);
      };
    const auto rear_tire = [&](const float load) {
        return FialaCombinedSlip(
          rear_slip_angle, rear_slip_ratio, parameters_.rear_cornering_stiffness_nprad,
          parameters_.rear_friction_coefficient, load);
      };
    // Body-frame specific force along x, what an accelerometer at the CG reads.
    const auto body_ax = [&](const TireForces & front_forces, const TireForces & rear_forces) {
        return (cos_delta * front_forces.longitudinal_n - sin_delta * front_forces.lateral_n +
               cos_delta_rear * rear_forces.longitudinal_n -
               sin_delta_rear * rear_forces.lateral_n - cos_beta * rolling_resistance) / mass;
      };

    float front_load = static_front_load;
    float rear_load = static_rear_load;
    TireForces front = front_tire(front_load);
    TireForces rear = rear_tire(rear_load);
    if (parameters_.load_transfer_height_m > 0.0F) {
      // The loads set the tire forces and the forces' acceleration sets the
      // loads. One fixed-point pass from the static loads closes that loop:
      // replayed against logged driving it matches carrying the acceleration
      // over from the previous substep, without adding a state. The clamp
      // keeps both axle loads non-negative.
      const float transfer = clampf(
        mass * parameters_.load_transfer_height_m * body_ax(front, rear) / wheelbase,
        -static_rear_load, static_front_load);
      front_load -= transfer;
      rear_load += transfer;
      front = front_tire(front_load);
      rear = rear_tire(rear_load);
    }

    // Each axle's tire forces are in its own wheel frame, so both rotate into
    // the body frame by their own steer angle before the moment and force
    // balances. At rear_steering = 0 the rear terms collapse to -b * F_ry,
    // cos_beta / sin_beta, matching the front-steer-only model exactly.
    const float yaw_acceleration =
      (a * front.longitudinal_n * sin_delta + a * front.lateral_n * cos_delta -
      b * rear.longitudinal_n * sin_delta_rear - b * rear.lateral_n * cos_delta_rear) /
      yaw_inertia;
    const float speed_acceleration =
      (cos_delta_minus_beta * front.longitudinal_n -
      sin_delta_minus_beta * front.lateral_n +
      cos_delta_rear_minus_beta * rear.longitudinal_n -
      sin_delta_rear_minus_beta * rear.lateral_n - rolling_resistance) / mass;
    const float sideslip_rate = -yaw_rate +
      (sin_delta_minus_beta * front.longitudinal_n +
      cos_delta_minus_beta * front.lateral_n +
      sin_delta_rear_minus_beta * rear.longitudinal_n +
      cos_delta_rear_minus_beta * rear.lateral_n) / (mass * speed);
    // The driveline balance needs wheel-frame longitudinal forces, which is
    // what the tire model already returns, so steering does not rotate these.
    // Rear steer reaches the wheel speed through rear_slip_ratio instead.
    const float tire_reaction_force = rear.longitudinal_n +
      (parameters_.locked_awd ? front.longitudinal_n : 0.0F);
    const float applied_driveline_torque = parameters_.locked_awd ? torque : rear_torque;
    const float wheel_speed_rate = parameters_.wheel_radius_m *
      (applied_driveline_torque - parameters_.wheel_radius_m * tire_reaction_force) /
      fmaxf(parameters_.driven_wheel_inertia_kgm2, 1.0e-6F);
    if (report != nullptr) {
      report->front = front;
      report->rear = rear;
      report->front_load_n = front_load;
      report->rear_load_n = rear_load;
      report->front_slip_angle_rad = front_slip_angle;
      report->rear_slip_angle_rad = rear_slip_angle;
      report->front_slip_ratio = front_slip_ratio;
      report->rear_slip_ratio = rear_slip_ratio;
      report->body_ax_mps2 = body_ax(front, rear);
      report->body_ay_mps2 =
        (sin_delta * front.longitudinal_n + cos_delta * front.lateral_n +
        sin_delta_rear * rear.longitudinal_n + cos_delta_rear * rear.lateral_n -
        sin_beta * rolling_resistance) / mass;
    }
    return BodyDerivative{
      yaw_acceleration, speed_acceleration, sideslip_rate, wheel_speed_rate};
  }

 private:
  VehicleParameters parameters_;
};

}  // namespace xxcar::mppi

#undef XXCAR_MODEL_4WS_HD
