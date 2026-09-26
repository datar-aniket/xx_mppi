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
    if (engaged_) {
      if (magnitude < release_rpm_) {
        engaged_ = false;
      }
    } else if (magnitude > engage_rpm_) {
      engaged_ = true;
    }
    if (!engaged_) {
      return 0.0F;
    }
    return motor_rpm > 0.0F ? -torque_magnitude_nm_ : torque_magnitude_nm_;
  }

  void Reset() noexcept {engaged_ = false;}
  [[nodiscard]] bool engaged() const noexcept {return engaged_;}

 private:
  float torque_magnitude_nm_{};
  float release_rpm_{};
  float engage_rpm_{};
  bool engaged_{false};
};

}  // namespace xxcar::mppi
