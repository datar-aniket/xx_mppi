#pragma once

#include <chrono>
#include <optional>

namespace xxcar::mppi {

// Debounces both sides of the zero-duty obstacle brake. An all-latched sample
// population must persist for the activation interval. Once active, reaching
// the configured velocity release condition clears it immediately; otherwise
// release requires one uninterrupted healthy interval.
class ObstacleBrakeLatch {
 public:
  ObstacleBrakeLatch(
    const std::chrono::steady_clock::duration activation,
    const std::chrono::steady_clock::duration recovery)
  : activation_(activation), recovery_(recovery) {}

  bool Update(
    const bool all_rollouts_obstacle_latched,
    const bool release_for_velocity,
    const std::chrono::steady_clock::time_point now) noexcept
  {
    if (release_for_velocity) {
      Reset();
      return false;
    }
    if (all_rollouts_obstacle_latched) {
      healthy_since_.reset();
      if (active_) {
        return true;
      }
      if (!latched_since_) {
        latched_since_ = now;
        return false;
      }
      if (now - *latched_since_ >= activation_) {
        active_ = true;
        latched_since_.reset();
      }
      return active_;
    }
    latched_since_.reset();
    if (!active_) {
      return false;
    }
    if (!healthy_since_) {
      healthy_since_ = now;
      return true;
    }
    if (now - *healthy_since_ >= recovery_) {
      active_ = false;
      healthy_since_.reset();
    }
    return active_;
  }

  void Reset() noexcept {
    active_ = false;
    latched_since_.reset();
    healthy_since_.reset();
  }

  // Invalid/stale solves may neither complete activation nor count as healthy
  // recovery. Preserve an already active brake while restarting both timers.
  bool Pause() noexcept {
    latched_since_.reset();
    healthy_since_.reset();
    return active_;
  }

  [[nodiscard]] bool active() const noexcept {return active_;}

 private:
  std::chrono::steady_clock::duration activation_{};
  std::chrono::steady_clock::duration recovery_{};
  bool active_{false};
  std::optional<std::chrono::steady_clock::time_point> latched_since_;
  std::optional<std::chrono::steady_clock::time_point> healthy_since_;
};

}  // namespace xxcar::mppi
