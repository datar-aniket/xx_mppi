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
  MppiDiagnostics diagnostics{};
  Projection projection{};
};

class MppiController {
 public:
  MppiController(ControllerConfig config, Raceline raceline);

  // Call for every incoming state message, even if optimization is throttled
  // to solve_rate_hz. This keeps the loop-continuous s hint current.
  [[nodiscard]] Projection UpdateObservation(const VehicleObservation & observation);
  [[nodiscard]] PlannedTrajectory PlanLatest();
  [[nodiscard]] PlannedTrajectory Plan(const VehicleObservation & observation);
  void Reset() noexcept;

  [[nodiscard]] const ControllerConfig & config() const noexcept { return config_; }
  [[nodiscard]] const Raceline & raceline() const noexcept { return raceline_; }

 private:
  struct PreparedObservation {
    VehicleObservation observation;
    float sideslip_rad{};
    Projection projection;
  };

  ControllerConfig config_;
  Raceline raceline_;
  ContinuousProjector projector_;
  CudaMppiController optimizer_;
  std::optional<PreparedObservation> latest_;
  std::optional<std::int64_t> previous_pose_time_ns_;
  bool reset_next_{true};
};

}  // namespace xxcar::mppi
