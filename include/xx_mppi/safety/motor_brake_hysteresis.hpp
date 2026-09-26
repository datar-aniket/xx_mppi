#pragma once

#include <cmath>

namespace xxcar::mppi {

// Stateful motor brake with a deadband that cannot chatter around one RPM
// threshold. Torque always opposes the current direction of motor rotation.
class MotorBrakeHysteresis {
 public:
  MotorBrakeHysteresis(
    const float torque_magnitude_nm, const float release_rpm, const float engage_rpm)
  : torque_magnitude_nm_(torque_magnitude_nm),
    release_rpm_(release_rpm), engage_rpm_(engage_rpm) {}

  [[nodiscard]] float Update(const float motor_rpm) noexcept {
    const float magnitude = std::abs(motor_rpm);
    if (magnitude < release_rpm_) {
      Reset();
      return 0.0F;
    }

    const float measured_direction = motor_rpm > 0.0F ? 1.0F : -1.0F;
    if (engaged_ && measured_direction != latched_direction_) {
      // A sampled sign jump may have skipped the low-RPM region because of
      // telemetry rate, quantization, or drivetrain backlash. Never reverse a
      // full torque command on that sample: disarm first and require a later
      // sample above the engage threshold to establish the new direction.
      Reset();
      return 0.0F;
    }

    if (!engaged_ && magnitude > engage_rpm_) {
      engaged_ = true;
      latched_direction_ = measured_direction;
    }
    if (!engaged_) {
      return 0.0F;
    }
    return -latched_direction_ * torque_magnitude_nm_;
  }

  void Reset() noexcept {
    engaged_ = false;
    latched_direction_ = 0.0F;
  }
  [[nodiscard]] bool engaged() const noexcept {return engaged_;}

 private:
  float torque_magnitude_nm_{};
  float release_rpm_{};
  float engage_rpm_{};
  float latched_direction_{};
  bool engaged_{false};
};

}  // namespace xxcar::mppi
