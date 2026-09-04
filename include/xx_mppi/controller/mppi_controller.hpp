#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "xx_mppi/controller/config.hpp"
#include "xx_mppi/controller/cuda_mppi.hpp"
#include "xx_mppi/reference/raceline.hpp"
#include "xx_mppi/types.hpp"

namespace xxcar::mppi {

// ROS-independent state contract. The eventual ROS adapter only copies fields
// from the xxCar vehicle-state message and converts its quaternion to ENU yaw.
struct VehicleObservation {
  std::int64_t pose_time_ns{};
  float east_m{};
  float north_m{};
  float yaw_enu_rad{};
  float speed_mps{};
  float yaw_rate_radps{};
  float longitudinal_acceleration_mps2{};
  float sideslip_rad{};
  float measured_torque_nm{};
  float measured_steering_rad{};
  float driven_wheel_speed_mps{};
  std::uint32_t status{};
};

struct CartesianTrajectoryState {
  float east_m{};
  float north_m{};
};

struct PlannedTrajectory {
  std::int64_t solution_pose_time_ns{};
  float dt_s{};
  std::vector<CartesianTrajectoryState> states;  // T, terminal x[T] omitted
  std::vector<Control> controls;                 // T
  std::vector<WeightedRollout> sampled_rollouts;  // optional, highest weight first
  MppiDiagnostics diagnostics{};
  Projection projection{};
  // Frame the rollout states in sampled_rollouts are expressed in.
  FrameKind frame{FrameKind::kFrenet};
};

// The sideslip both the Frenet projection and the body model use.
//
// EkfState reports side_slip_rad as atan2(v_y, v_x) of the body twist, so a
// disagreement between the reported heading and the reported velocity shows up
// here at full size. Left unbounded it rotates the projection's relative course
// heading past +/-90 degrees, which makes s_dot = V cos(dphi) negative: the
// planned path then retreats along the track before turning around. Bounding it
// once, here, keeps the geometry and the small-angle bicycle model consistent.
//
// A non-finite estimate becomes zero only below 0.3 m/s; while moving it is
// passed through so the caller's finite check rejects the sample.
[[nodiscard]] float ConditionedSideslip(
  float measured_sideslip_rad, float speed_mps, float maximum_rad) noexcept;

class MppiController {
 public:
  MppiController(ControllerConfig config, Raceline raceline);

  // Call for every incoming state message, even if optimization is throttled
  // to solve_rate_hz. This keeps the loop-continuous s hint current.
  [[nodiscard]] Projection UpdateObservation(const VehicleObservation & observation);
  [[nodiscard]] PlannedTrajectory PlanLatest(
    std::uint32_t num_visualization_rollouts = 0U);
  [[nodiscard]] PlannedTrajectory Plan(
    const VehicleObservation & observation,
    std::uint32_t num_visualization_rollouts = 0U);
  // Called only after the ROS command publisher accepts a solution. This keeps
  // the fallback feedback aligned with what left the controller, not merely
  // with the newest (possibly downsampled) solve.
  void RecordPublishedControl(const Control & control) noexcept;
  void UpdateObstacleField(const ObstacleField & field);
  void ClearObstacleField();
  void Reset() noexcept;

  [[nodiscard]] const ControllerConfig & config() const noexcept { return config_; }
  [[nodiscard]] const Raceline & raceline() const noexcept { return raceline_; }

 private:
  struct PreparedObservation {
    VehicleObservation observation;
    float sideslip_rad{};
    Projection projection;
  };

  [[nodiscard]] Control PreviousControl(const VehicleObservation & observation) const;

  ControllerConfig config_;
  Raceline raceline_;
  ContinuousProjector projector_;
  CudaMppiController optimizer_;
  std::optional<PreparedObservation> latest_;
  std::optional<std::int64_t> previous_pose_time_ns_;
  Control last_applied_control_{};
  bool reset_next_{true};
};

}  // namespace xxcar::mppi
