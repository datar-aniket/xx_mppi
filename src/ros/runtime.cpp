#include "xx_mppi/ros/runtime.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <stdexcept>
#include <thread>
#include <utility>

#include "xx_mppi/controller/builder.hpp"
#include "xx_mppi/ros/direct_control_message.hpp"
#include "xx_mppi/ros/trajectory_message.hpp"

namespace xxcar::mppi {

MppiRosRuntime::MppiRosRuntime(
  rclcpp::Node & node, const std::string & config_directory,
  const std::string & trajectory_topic, DirectControlConfig direct_control,
  VisualizationConfig visualization)
: node_(node),
  controller_(MppiControllerBuilder::FromConfigDirectory(config_directory)),
  direct_control_(std::move(direct_control)),
  visualization_(std::move(visualization))
{
  RCLCPP_INFO(
    node_.get_logger(), "Loaded raceline '%s' (%zu points, %.3f m lap)",
    controller_->config().raceline_path.c_str(), controller_->raceline().points().size(),
    static_cast<double>(controller_->raceline().length()));
  if (direct_control_.enabled) {
    ValidateDirectControlConfig(direct_control_);
    direct_control_publisher_ = node_.create_publisher<geometry_msgs::msg::Twist>(
      direct_control_.topic, rclcpp::QoS(1).best_effort());
  } else {
    if (trajectory_topic.empty()) {
      throw std::invalid_argument("trajectory topic must not be empty");
    }
    trajectory_publisher_ =
      node_.create_publisher<xxcar_msgs::msg::VehicleControlTrajectory>(
      trajectory_topic, rclcpp::QoS(1).best_effort());
  }
  if (visualization_.enabled) {
    if (visualization_.frame_id.empty() || visualization_.planned_path_topic.empty() ||
      visualization_.marker_topic.empty() || visualization_.raceline_topic.empty() ||
      visualization_.left_boundary_topic.empty() ||
      visualization_.right_boundary_topic.empty() ||
      visualization_.obstacle_costmap_topic.empty())
    {
      throw std::invalid_argument("visualization frame and topics must not be empty");
    }
    planned_path_publisher_ = node_.create_publisher<nav_msgs::msg::Path>(
      visualization_.planned_path_topic, rclcpp::QoS(1).best_effort());
    marker_publisher_ = node_.create_publisher<visualization_msgs::msg::MarkerArray>(
      visualization_.marker_topic, rclcpp::QoS(1).best_effort());
    const auto static_qos = rclcpp::QoS(1).best_effort().transient_local();
    raceline_publisher_ = node_.create_publisher<nav_msgs::msg::Path>(
      visualization_.raceline_topic, static_qos);
    left_boundary_publisher_ = node_.create_publisher<nav_msgs::msg::Path>(
      visualization_.left_boundary_topic, static_qos);
    right_boundary_publisher_ = node_.create_publisher<nav_msgs::msg::Path>(
      visualization_.right_boundary_topic, static_qos);
    if (controller_->config().obstacles.enabled) {
      obstacle_costmap_publisher_ = node_.create_publisher<nav_msgs::msg::OccupancyGrid>(
        visualization_.obstacle_costmap_topic, rclcpp::QoS(1).best_effort());
    }
    visualization_period_ = std::chrono::nanoseconds(
      static_cast<std::int64_t>(std::llround(
        1.0e9 / static_cast<double>(controller_->config().visualization_rate_hz))));
    next_visualization_time_ = std::chrono::steady_clock::now();
    next_costmap_time_ = next_visualization_time_;
  }
  const auto solve_period_ns = static_cast<std::int64_t>(std::llround(
      1.0e9 / static_cast<double>(controller_->config().solve_rate_hz)));
  solve_period_ = std::chrono::nanoseconds(solve_period_ns);
  const auto publication_period_ns = static_cast<std::int64_t>(std::llround(
      1.0e9 / static_cast<double>(controller_->config().control_publish_rate_hz)));
  control_publication_period_ = std::chrono::nanoseconds(publication_period_ns);
  limit_control_publication_rate_ =
    controller_->config().control_publish_rate_hz < controller_->config().solve_rate_hz;
  const auto info_period_ns = static_cast<std::int64_t>(std::llround(
      1.0e9 / static_cast<double>(controller_->config().info_log_rate_hz)));
  info_log_period_ = std::chrono::nanoseconds(info_period_ns);
  RCLCPP_INFO(
    node_.get_logger(), "MPPI solve rate %.3f Hz, control publication rate %.3f Hz",
    static_cast<double>(controller_->config().solve_rate_hz),
    static_cast<double>(controller_->config().control_publish_rate_hz));
  RCLCPP_INFO(
    node_.get_logger(), "MPPI terminal info logging at %.3f Hz",
    static_cast<double>(controller_->config().info_log_rate_hz));
  if (controller_->config().control_publish_rate_hz > controller_->config().solve_rate_hz) {
    RCLCPP_WARN(
      node_.get_logger(),
      "control_publish_rate_hz exceeds solve_rate_hz; new-only publication will be "
      "limited by the solve/state rate");
  }
  solver_thread_ = std::thread([this]() {SolverWorker();});
  control_thread_ = std::thread([this]() {ControlWorker();});
  info_thread_ = std::thread([this]() {InfoWorker();});
  visualization_thread_ = std::thread([this]() {VisualizationWorker();});
  if (visualization_.enabled) {
    RCLCPP_INFO(
      node_.get_logger(),
      "Best-effort visualization enabled: Path '%s', MarkerArray '%s', costmap '%s' at %.3f Hz",
      visualization_.planned_path_topic.c_str(), visualization_.marker_topic.c_str(),
      visualization_.obstacle_costmap_topic.c_str(),
      static_cast<double>(controller_->config().visualization_rate_hz));
  } else {
    RCLCPP_INFO(
      node_.get_logger(),
      "Visualization disabled; launch with publish_visualization:=true to enable RViz topics");
  }
}

MppiRosRuntime::~MppiRosRuntime() {
  {
    std::lock_guard<std::mutex> lock(worker_mutex_);
    stop_workers_ = true;
  }
  worker_cv_.notify_all();
  solver_cv_.notify_one();
  control_cv_.notify_one();
  if (solver_thread_.joinable()) {
    solver_thread_.join();
  }
  if (control_thread_.joinable()) {
    control_thread_.join();
  }
  if (info_thread_.joinable()) {
    info_thread_.join();
  }
  if (visualization_thread_.joinable()) {
    {
      std::lock_guard<std::mutex> lock(visualization_mutex_);
      stop_visualization_ = true;
      pending_visualization_.reset();
    }
    visualization_cv_.notify_one();
    visualization_thread_.join();
  }
}

void MppiRosRuntime::PublishStaticVisualization() {
  const auto paths = ToStaticVisualizationPaths(
    controller_->raceline(), node_.get_clock()->now(), visualization_.frame_id);
  raceline_publisher_->publish(paths.raceline);
  left_boundary_publisher_->publish(paths.left_boundary);
  right_boundary_publisher_->publish(paths.right_boundary);
}

void MppiRosRuntime::PublishTrajectoryVisualization(
  const PlannedTrajectory & trajectory, const rclcpp::Time & publication_time)
{
  planned_path_publisher_->publish(ToPlannedPath(
      trajectory, publication_time, visualization_.frame_id));
  marker_publisher_->publish(ToTrajectoryMarkers(
      trajectory, controller_->raceline(), publication_time, visualization_.frame_id));
}

void MppiRosRuntime::PublishObstacleVisualization(
  const ObstacleField & field, const rclcpp::Time & publication_time)
{
  if (obstacle_costmap_publisher_) {
    obstacle_costmap_publisher_->publish(ToObstacleCostmap(
        field, controller_->config().obstacles, publication_time,
        visualization_.frame_id));
  }
}

void MppiRosRuntime::PublishInfo(
  const PlannedTrajectory & trajectory, const double publication_age_ms)
{
  if (trajectory.controls.empty()) {
    return;
  }
  const auto & diagnostics = trajectory.diagnostics;
  const auto & command = trajectory.controls.front();
  const double solution_age_ms = static_cast<double>(
    node_.get_clock()->now().nanoseconds() - trajectory.solution_pose_time_ns) * 1.0e-6;
  RCLCPP_INFO(
    node_.get_logger(),
    "MPPI info: solve=%.3f ms lambda=%.6g sigma=[steer %.6g rad, torque %.6g Nm] "
    "command=[steer %.6g rad, torque %.6g Nm] cost=%.6g ESS=%.3f finite=%u "
    "publish_age=%.3f ms snapshot_age=%.3f ms",
    static_cast<double>(diagnostics.solve_time_ms),
    static_cast<double>(diagnostics.lambda_used),
    static_cast<double>(diagnostics.sigma_used[kSteering]),
    static_cast<double>(diagnostics.sigma_used[kWheelTorque]),
    static_cast<double>(command[kSteering]),
    static_cast<double>(command[kWheelTorque]),
    static_cast<double>(diagnostics.minimum_cost),
    static_cast<double>(diagnostics.effective_sample_size),
    static_cast<unsigned>(diagnostics.finite_rollouts), publication_age_ms,
    solution_age_ms);
}

void MppiRosRuntime::QueueVisualization(
  std::shared_ptr<const PlannedTrajectory> trajectory,
  const rclcpp::Time & publication_time)
{
  {
    std::lock_guard<std::mutex> lock(visualization_mutex_);
    pending_visualization_.emplace(std::move(trajectory), publication_time);
  }
  visualization_cv_.notify_one();
}

void MppiRosRuntime::VisualizationWorker() {
  auto next_static_publication = std::chrono::steady_clock::now();
  while (true) {
    std::optional<std::pair<std::shared_ptr<const PlannedTrajectory>, rclcpp::Time>> work;
    std::optional<std::pair<std::shared_ptr<const ObstacleField>, rclcpp::Time>> obstacle_work;
    bool publish_static = false;
    {
      std::unique_lock<std::mutex> lock(visualization_mutex_);
      const auto wake_time = visualization_.enabled ? next_static_publication :
        std::chrono::steady_clock::time_point::max();
      visualization_cv_.wait_until(lock, wake_time, [this]() {
        return stop_visualization_ || pending_visualization_.has_value() ||
               pending_obstacle_visualization_.has_value();
      });
      if (stop_visualization_) {
        return;
      }
      if (pending_visualization_) {
        work = std::move(pending_visualization_);
        pending_visualization_.reset();
      }
      if (pending_obstacle_visualization_) {
        obstacle_work = std::move(pending_obstacle_visualization_);
        pending_obstacle_visualization_.reset();
      }
      const auto now = std::chrono::steady_clock::now();
      if (visualization_.enabled && now >= next_static_publication) {
        publish_static = true;
        next_static_publication = now + std::chrono::seconds(1);
      }
    }
    if (publish_static) {
      try {
        PublishStaticVisualization();
      } catch (const std::exception & error) {
        RCLCPP_ERROR_THROTTLE(
          node_.get_logger(), *node_.get_clock(), 1000,
          "MPPI static visualization publication failed: %s", error.what());
      }
    }
    if (work) {
      try {
        PublishTrajectoryVisualization(*work->first, work->second);
      } catch (const std::exception & error) {
        RCLCPP_ERROR_THROTTLE(
          node_.get_logger(), *node_.get_clock(), 1000,
          "MPPI visualization publication failed: %s", error.what());
      }
    }
    if (obstacle_work) {
      try {
        PublishObstacleVisualization(*obstacle_work->first, obstacle_work->second);
      } catch (const std::exception & error) {
        RCLCPP_ERROR_THROTTLE(
          node_.get_logger(), *node_.get_clock(), 1000,
          "MPPI obstacle costmap publication failed: %s", error.what());
      }
    }
  }
}

void MppiRosRuntime::OnObservation(const VehicleObservation & observation) {
  const float maximum_sideslip = controller_->config().maximum_model_sideslip_rad;
  if (std::isfinite(observation.sideslip_rad) &&
    std::abs(observation.sideslip_rad) > maximum_sideslip)
  {
    // A sideslip this large is normally a disagreement between the reported
    // heading and the reported body velocity, not real cornering. It is bounded
    // before it can spin the Frenet relative heading past +/-90 degrees.
    RCLCPP_WARN_THROTTLE(
      node_.get_logger(), *node_.get_clock(), 2000,
      "EKF sideslip %.3f rad exceeds maximum_model_sideslip_rad %.3f rad and was "
      "bounded; check the EkfState twist/heading agreement",
      static_cast<double>(observation.sideslip_rad),
      static_cast<double>(maximum_sideslip));
  }
  const float sideslip = ConditionedSideslip(
    observation.sideslip_rad, observation.speed_mps, maximum_sideslip);
  if (!std::isfinite(observation.east_m) || !std::isfinite(observation.north_m) ||
    !std::isfinite(observation.yaw_enu_rad) || !std::isfinite(observation.speed_mps) ||
    !std::isfinite(observation.yaw_rate_radps) || !std::isfinite(sideslip) ||
    !std::isfinite(observation.measured_torque_nm) ||
    !std::isfinite(observation.measured_steering_rad) ||
    !std::isfinite(observation.driven_wheel_speed_mps))
  {
    throw std::invalid_argument("vehicle observation contains a non-finite value");
  }
  {
    std::lock_guard<std::mutex> lock(worker_mutex_);
    latest_observation_ = observation;
    ++observation_generation_;
  }
  solver_cv_.notify_one();
}

void MppiRosRuntime::SetObstacleField(
  std::shared_ptr<const ObstacleField> field, const std::uint64_t reset_epoch)
{
  {
    std::lock_guard<std::mutex> lock(worker_mutex_);
    if (reset_epoch != reset_epoch_) {
      return;
    }
    std::atomic_store_explicit(
      &pending_obstacle_field_, field, std::memory_order_release);
  }
  if (visualization_.enabled && obstacle_costmap_publisher_) {
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(visualization_mutex_);
    if (now >= next_costmap_time_) {
      pending_obstacle_visualization_.emplace(std::move(field), node_.get_clock()->now());
      next_costmap_time_ = now + visualization_period_;
      visualization_cv_.notify_one();
    }
  }
}

void MppiRosRuntime::Reset() {
  {
    std::lock_guard<std::mutex> lock(worker_mutex_);
    latest_observation_.reset();
    pending_published_control_.reset();
    ++observation_generation_;
    ++reset_epoch_;
    reset_requested_ = true;
    control_work_pending_ = false;
    std::atomic_store_explicit(
      &pending_obstacle_field_, std::shared_ptr<const ObstacleField>{},
      std::memory_order_release);
  }
  {
    std::lock_guard<std::mutex> solution_lock(solution_mutex_);
    latest_solution_.reset();
    published_solution_.reset();
    latest_solution_generation_ = 0U;
    published_solution_generation_ = 0U;
    published_solution_age_ms_ = 0.0;
  }
  {
    std::lock_guard<std::mutex> visualization_lock(visualization_mutex_);
    pending_visualization_.reset();
    pending_obstacle_visualization_.reset();
    next_costmap_time_ = std::chrono::steady_clock::now();
  }
  worker_cv_.notify_all();
  solver_cv_.notify_one();
  control_cv_.notify_one();
}

void MppiRosRuntime::SolveOnce(
  const VehicleObservation & observation, const std::uint64_t observation_generation,
  const std::uint64_t reset_epoch, const std::optional<Control> & published_control)
{
  PlannedTrajectory trajectory;
  bool capture_visualization = false;
  try {
    if (published_control) {
      controller_->RecordPublishedControl(*published_control);
    }
    (void)controller_->UpdateObservation(observation);
    const auto field = std::atomic_load_explicit(
      &pending_obstacle_field_, std::memory_order_acquire);
    if (field && field->generation != applied_obstacle_generation_) {
      controller_->UpdateObstacleField(*field);
      applied_obstacle_generation_ = field->generation;
    }
    capture_visualization = visualization_.enabled &&
      std::chrono::steady_clock::now() >= next_visualization_time_;
    trajectory = controller_->PlanLatest(
      capture_visualization ? controller_->config().num_rollouts : 0U);
    if (capture_visualization) {
      next_visualization_time_ += visualization_period_;
      const auto now = std::chrono::steady_clock::now();
      if (next_visualization_time_ <= now) {
        next_visualization_time_ = now + visualization_period_;
      }
    }
  } catch (const std::exception & error) {
    RCLCPP_ERROR_THROTTLE(
      node_.get_logger(), *node_.get_clock(), 1000,
      "MPPI solve failed: %s", error.what());
    return;
  }
  const float solve_budget_ms = 1000.0F / controller_->config().solve_rate_hz;
  if (trajectory.diagnostics.solve_time_ms > solve_budget_ms) {
    RCLCPP_WARN_THROTTLE(
      node_.get_logger(), *node_.get_clock(), 1000,
      "MPPI solve took %.3f ms and exceeded the %.3f ms configured period",
      static_cast<double>(trajectory.diagnostics.solve_time_ms),
      static_cast<double>(solve_budget_ms));
  }
  auto solution = std::make_shared<PlannedTrajectory>(std::move(trajectory));
  {
    std::lock_guard<std::mutex> worker_lock(worker_mutex_);
    if (reset_epoch != reset_epoch_) {
      return;
    }
    std::lock_guard<std::mutex> solution_lock(solution_mutex_);
    latest_solution_ = solution;
    latest_solution_generation_ = observation_generation;
    control_work_pending_ = true;
  }
  control_cv_.notify_one();
  if (capture_visualization) {
    QueueVisualization(std::move(solution), node_.get_clock()->now());
  }
}

void MppiRosRuntime::SolverWorker() {
  auto next_solve = std::chrono::steady_clock::now();
  while (true) {
    std::optional<VehicleObservation> observation;
    std::optional<Control> published_control;
    std::uint64_t generation = 0U;
    std::uint64_t reset_epoch = 0U;
    bool reset = false;
    {
      std::unique_lock<std::mutex> lock(worker_mutex_);
      while (!stop_workers_) {
        if (reset_requested_) {
          reset = true;
          reset_requested_ = false;
          break;
        }
        if (latest_observation_ && observation_generation_ != solved_generation_) {
          const auto now = std::chrono::steady_clock::now();
          if (now >= next_solve) {
            break;
          }
          solver_cv_.wait_until(lock, next_solve, [this]() {
            return stop_workers_ || reset_requested_;
          });
          continue;
        }
        solver_cv_.wait(lock, [this]() {
          return stop_workers_ || reset_requested_ ||
                 (latest_observation_ && observation_generation_ != solved_generation_);
        });
      }
      if (stop_workers_) {
        return;
      }
      if (latest_observation_ && observation_generation_ != solved_generation_ &&
        std::chrono::steady_clock::now() >= next_solve)
      {
        observation = latest_observation_;
        generation = observation_generation_;
        reset_epoch = reset_epoch_;
        published_control = std::move(pending_published_control_);
      }
    }
    if (reset) {
      controller_->Reset();
      controller_->ClearObstacleField();
      applied_obstacle_generation_ = 0U;
      next_visualization_time_ = std::chrono::steady_clock::now();
      next_solve = next_visualization_time_;
    }
    if (!observation) {
      continue;
    }
    SolveOnce(*observation, generation, reset_epoch, published_control);
    {
      std::lock_guard<std::mutex> lock(worker_mutex_);
      solved_generation_ = generation;
    }
    const auto now = std::chrono::steady_clock::now();
    next_solve += solve_period_;
    if (next_solve <= now) {
      next_solve = now + solve_period_;
    }
  }
}

void MppiRosRuntime::ControlWorker() {
  auto next_publication = std::chrono::steady_clock::now();
  while (true) {
    {
      std::unique_lock<std::mutex> lock(worker_mutex_);
      control_cv_.wait(lock, [this]() {
        return stop_workers_ || control_work_pending_;
      });
      if (stop_workers_) {
        return;
      }
      const auto now = std::chrono::steady_clock::now();
      if (limit_control_publication_rate_ && now < next_publication) {
        control_cv_.wait_until(lock, next_publication, [this]() {return stop_workers_;});
        if (stop_workers_) {
          return;
        }
      }
      control_work_pending_ = false;
    }
    ControlPublicationCallback();
    if (limit_control_publication_rate_) {
      next_publication = std::chrono::steady_clock::now() + control_publication_period_;
    }
  }
}

void MppiRosRuntime::InfoWorker() {
  auto next_log = std::chrono::steady_clock::now() + info_log_period_;
  while (true) {
    {
      std::unique_lock<std::mutex> lock(worker_mutex_);
      if (worker_cv_.wait_until(lock, next_log, [this]() {return stop_workers_;})) {
        return;
      }
    }
    InfoLogCallback();
    const auto now = std::chrono::steady_clock::now();
    next_log += info_log_period_;
    if (next_log <= now) {
      next_log = now + info_log_period_;
    }
  }
}

void MppiRosRuntime::ControlPublicationCallback() {
  std::shared_ptr<const PlannedTrajectory> solution;
  std::uint64_t generation = 0U;
  {
    std::lock_guard<std::mutex> lock(solution_mutex_);
    if (!latest_solution_ || latest_solution_generation_ == published_solution_generation_) {
      return;
    }
    solution = latest_solution_;
    generation = latest_solution_generation_;
  }

  const auto publication_time = node_.get_clock()->now();
  const double solution_age_s = static_cast<double>(
    publication_time.nanoseconds() - solution->solution_pose_time_ns) * 1.0e-9;
  if (controller_->config().maximum_solution_age_s > 0.0F &&
    solution_age_s > static_cast<double>(controller_->config().maximum_solution_age_s))
  {
    RCLCPP_WARN_THROTTLE(
      node_.get_logger(), *node_.get_clock(), 1000,
      "Skipping stale MPPI solution (age %.3f s, limit %.3f s)", solution_age_s,
      static_cast<double>(controller_->config().maximum_solution_age_s));
    std::lock_guard<std::mutex> lock(solution_mutex_);
    if (generation == latest_solution_generation_) {
      published_solution_generation_ = generation;
    }
    return;
  }

  try {
    if (direct_control_.enabled) {
      direct_control_publisher_->publish(ToDirectControlMessage(*solution, direct_control_));
    } else {
      trajectory_publisher_->publish(ToRosMessage(*solution, publication_time));
    }
  } catch (const std::exception & error) {
    RCLCPP_ERROR_THROTTLE(
      node_.get_logger(), *node_.get_clock(), 1000,
      "MPPI control publication failed: %s", error.what());
    return;
  }
  if (!solution->controls.empty()) {
    std::lock_guard<std::mutex> lock(worker_mutex_);
    pending_published_control_ = solution->controls.front();
  }
  {
    std::lock_guard<std::mutex> lock(solution_mutex_);
    // A newer solve may have completed during publication. Mark only the
    // solution actually sent so that the newer one remains eligible.
    published_solution_generation_ = std::max(
      published_solution_generation_, generation);
    published_solution_age_ms_ = solution_age_s * 1000.0;
    published_solution_ = std::move(solution);
  }
}

void MppiRosRuntime::InfoLogCallback() {
  std::shared_ptr<const PlannedTrajectory> solution;
  double publication_age_ms = 0.0;
  {
    std::lock_guard<std::mutex> lock(solution_mutex_);
    solution = published_solution_;
    publication_age_ms = published_solution_age_ms_;
  }
  if (!solution) {
    return;
  }
  PublishInfo(*solution, publication_age_ms);
}

}  // namespace xxcar::mppi
