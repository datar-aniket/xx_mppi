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
#include "xx_mppi/dynamics/grip.hpp"
#include "xx_mppi/ros/direct_control_message.hpp"
#include "xx_mppi/ros/trajectory_message.hpp"

namespace xxcar::mppi {

MppiRosRuntime::MppiRosRuntime(
  rclcpp::Node & node, const std::string & config_directory,
  const std::string & trajectory_topic, DirectControlConfig direct_control,
  VisualizationConfig visualization, CostTermsConfig cost_terms,
  GripStatusConfig grip_status, MuEstimateConfig mu_estimate)
: node_(node),
  controller_(MppiControllerBuilder::FromConfigDirectory(config_directory)),
  direct_control_(std::move(direct_control)),
  visualization_(std::move(visualization)),
  cost_terms_(std::move(cost_terms)),
  grip_status_(std::move(grip_status)),
  mu_estimate_(std::move(mu_estimate))
{
  RCLCPP_INFO(
    node_.get_logger(), "Loaded raceline '%s' (%zu points, %.3f m lap)",
    controller_->config().raceline_path.c_str(), controller_->raceline().points().size(),
    static_cast<double>(controller_->raceline().length()));
  // Config load pins the rear steering channel whenever the active model does
  // not steer the rear axle. Say so at startup, so a run that shows no rear
  // motion has an explanation that does not require reading the YAML.
  if (controller_->config().mppi.sigma[kRearSteering] > 0.0F) {
    RCLCPP_INFO(
      node_.get_logger(), "MPPI rear steering ACTIVE: sigma %.3f rad, bounds [%.3f, %.3f]",
      static_cast<double>(controller_->config().mppi.sigma[kRearSteering]),
      static_cast<double>(controller_->config().mppi.control_min[kRearSteering]),
      static_cast<double>(controller_->config().mppi.control_max[kRearSteering]));
  } else {
    RCLCPP_INFO(
      node_.get_logger(),
      "MPPI rear steering pinned to zero; the solver plans front steering only");
  }
  if (direct_control_.enabled) {
    ValidateDirectControlConfig(direct_control_);
    // Exactly one transport is ever created: a Twist, or the four-wheel
    // DirectControl message. Publishing both would give the driver two command
    // streams for the same actuator.
    if (direct_control_.four_wheel) {
      if (direct_control_.four_wheel_topic.empty()) {
        throw std::invalid_argument("four wheel direct control topic must not be empty");
      }
      four_wheel_control_publisher_ =
        node_.create_publisher<xxcar_msgs::msg::DirectControl>(
        direct_control_.four_wheel_topic, rclcpp::QoS(1).best_effort());
      RCLCPP_INFO(
        node_.get_logger(), "MPPI four-wheel direct control on '%s'",
        direct_control_.four_wheel_topic.c_str());
    } else {
      direct_control_publisher_ = node_.create_publisher<geometry_msgs::msg::Twist>(
        direct_control_.topic, rclcpp::QoS(1).best_effort());
    }
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
      visualization_.right_boundary_topic.empty())
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
  }
  if (visualization_.obstacle_costmap_enabled) {
    if (visualization_.frame_id.empty() || visualization_.obstacle_costmap_topic.empty()) {
      throw std::invalid_argument("obstacle costmap frame and topic must not be empty");
    }
    if (controller_->config().obstacles.enabled) {
      obstacle_costmap_publisher_ = node_.create_publisher<nav_msgs::msg::OccupancyGrid>(
        visualization_.obstacle_costmap_topic, rclcpp::QoS(1).best_effort());
    } else {
      RCLCPP_WARN(
        node_.get_logger(),
        "publish_obstacle_costmap is set but obstacles are disabled; no costmap is published");
    }
  }
  if (visualization_.enabled || obstacle_costmap_publisher_) {
    visualization_period_ = std::chrono::nanoseconds(
      static_cast<std::int64_t>(std::llround(
        1.0e9 / static_cast<double>(controller_->config().visualization_rate_hz))));
    next_visualization_time_ = std::chrono::steady_clock::now();
    next_costmap_time_ = next_visualization_time_;
  }
  if (cost_terms_.enabled) {
    if (cost_terms_.topic.empty()) {
      throw std::invalid_argument("cost terms topic must not be empty");
    }
    cost_terms_publisher_ = node_.create_publisher<xxcar_msgs::msg::MppiCostTerms>(
      cost_terms_.topic, rclcpp::QoS(1).best_effort());
    cost_terms_period_ = std::chrono::nanoseconds(
      static_cast<std::int64_t>(std::llround(
        1.0e9 / static_cast<double>(controller_->config().cost_terms_rate_hz))));
    next_cost_terms_time_ = std::chrono::steady_clock::now();
    RCLCPP_INFO(
      node_.get_logger(), "MPPI cost term debug on '%s' at %.3f Hz",
      cost_terms_.topic.c_str(),
      static_cast<double>(controller_->config().cost_terms_rate_hz));
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
  if (grip_status_.enabled) {
    if (grip_status_.topic.empty() || !(grip_status_.rate_hz > 0.0) ||
      !std::isfinite(grip_status_.rate_hz))
    {
      throw std::invalid_argument("grip status topic must be set and its rate positive");
    }
    grip_status_publisher_ = node_.create_publisher<xxcar_msgs::msg::GripStatus>(
      grip_status_.topic, rclcpp::QoS(1).best_effort());
    grip_thread_ = std::thread([this]() {GripWorker();});
    RCLCPP_INFO(
      node_.get_logger(), "Grip status on '%s' at %.3f Hz%s", grip_status_.topic.c_str(),
      grip_status_.rate_hz,
      ModelHasTireForces(controller_->config().model_kind) ? "" :
      " (this model has no tire forces; predicted fields are NaN)");
  }
  if (mu_estimate_.enabled) {
    if (mu_estimate_.topic.empty() || !(mu_estimate_.rate_hz > 0.0) ||
      !std::isfinite(mu_estimate_.rate_hz))
    {
      throw std::invalid_argument("mu estimate topic must be set and its rate positive");
    }
    // The fit always runs the Fiala tire model on vehicle.yaml, whatever model
    // the solver plans with.
    mu_estimator_ = std::make_unique<MuEstimator>(
      controller_->config().vehicle, mu_estimate_.estimator);
    mu_estimate_publisher_ = node_.create_publisher<std_msgs::msg::Float32>(
      mu_estimate_.topic, rclcpp::QoS(1).best_effort());
    mu_thread_ = std::thread([this]() {MuWorker();});
    RCLCPP_INFO(
      node_.get_logger(), "Online mu estimate on '%s' at %.3f Hz (vehicle.yaml mu %.3f)",
      mu_estimate_.topic.c_str(), mu_estimate_.rate_hz,
      static_cast<double>(mu_estimator_->nominal_mu()));
  } else {
    RCLCPP_INFO(node_.get_logger(), "Online mu estimate disabled");
  }
  visualization_thread_ = std::thread([this]() {VisualizationWorker();});
  if (visualization_.enabled) {
    RCLCPP_INFO(
      node_.get_logger(),
      "Best-effort visualization enabled: Path '%s', MarkerArray '%s' at %.3f Hz",
      visualization_.planned_path_topic.c_str(), visualization_.marker_topic.c_str(),
      static_cast<double>(controller_->config().visualization_rate_hz));
  } else {
    RCLCPP_INFO(
      node_.get_logger(),
      "Visualization disabled; launch with publish_visualization:=true to enable RViz topics");
  }
  if (obstacle_costmap_publisher_) {
    RCLCPP_WARN(
      node_.get_logger(),
      "Obstacle costmap enabled on '%s' at %.3f Hz; each grid is large, keep it off "
      "over WiFi while driving",
      visualization_.obstacle_costmap_topic.c_str(),
      static_cast<double>(controller_->config().visualization_rate_hz));
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
  if (grip_thread_.joinable()) {
    grip_thread_.join();
  }
  if (mu_thread_.joinable()) {
    mu_thread_.join();
  }
  if (visualization_thread_.joinable()) {
    {
      std::lock_guard<std::mutex> lock(visualization_mutex_);
      stop_visualization_ = true;
      pending_capture_.reset();
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
  const PlannedTrajectory & trajectory, const double publication_age_ms,
  const std::optional<CostTerms> & cost_terms)
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
  if (cost_terms) {
    // The single largest positive term, which is the one question the summed
    // cost above can never answer.
    const auto & terms = *cost_terms;
    std::size_t dominant = 0U;
    for (std::size_t i = 1U; i < kCostTermCount; ++i) {
      if (terms.values[i] > terms.values[dominant]) {
        dominant = i;
      }
    }
    RCLCPP_INFO(
      node_.get_logger(),
      "MPPI cost terms: total=%.6g dominant=%s (%.6g) clearance=%.3f m",
      static_cast<double>(terms.total()), CostTermName(dominant),
      static_cast<double>(terms.values[dominant]),
      static_cast<double>(terms.minimum_clearance_m));
  }
}

void MppiRosRuntime::PublishCostTerms(const PlannedTrajectory & trajectory) {
  if (!cost_terms_publisher_ || !trajectory.diagnostics.cost_terms) {
    return;
  }
  cost_terms_publisher_->publish(ToRosMessage(
      *trajectory.diagnostics.cost_terms, trajectory.solution_pose_time_ns,
      node_.get_clock()->now()));
}

void MppiRosRuntime::QueueCapture(CaptureWork work) {
  {
    std::lock_guard<std::mutex> lock(visualization_mutex_);
    pending_capture_ = std::move(work);
  }
  visualization_cv_.notify_one();
}

void MppiRosRuntime::PublishCapture(const CaptureWork & work) {
  // Waits here, on the visualization thread, for the GPU capture the solver
  // staged, and frees the slot so the solver can stage the next one.
  auto capture = controller_->CollectCapture();
  if (!capture || capture->id != work.trajectory->capture_id) {
    return;
  }
  {
    std::lock_guard<std::mutex> lock(worker_mutex_);
    if (work.reset_epoch != reset_epoch_) {
      return;
    }
  }
  auto trajectory = std::make_shared<PlannedTrajectory>(*work.trajectory);
  trajectory->sampled_rollouts = std::move(capture->sampled_rollouts);
  trajectory->diagnostics.cost_terms = capture->cost_terms;
  if (work.visualization) {
    try {
      PublishTrajectoryVisualization(*trajectory, work.publication_time);
    } catch (const std::exception & error) {
      RCLCPP_ERROR_THROTTLE(
        node_.get_logger(), *node_.get_clock(), 1000,
        "MPPI visualization publication failed: %s", error.what());
    }
  }
  if (trajectory->diagnostics.cost_terms) {
    {
      std::lock_guard<std::mutex> lock(solution_mutex_);
      latest_cost_terms_ = trajectory->diagnostics.cost_terms;
    }
    try {
      PublishCostTerms(*trajectory);
    } catch (const std::exception & error) {
      RCLCPP_ERROR_THROTTLE(
        node_.get_logger(), *node_.get_clock(), 1000,
        "MPPI cost term publication failed: %s", error.what());
    }
  }
}

void MppiRosRuntime::VisualizationWorker() {
  auto next_static_publication = std::chrono::steady_clock::now();
  while (true) {
    std::optional<CaptureWork> capture_work;
    std::optional<std::pair<std::shared_ptr<const ObstacleField>, rclcpp::Time>> obstacle_work;
    bool publish_static = false;
    {
      std::unique_lock<std::mutex> lock(visualization_mutex_);
      const auto wake_time = visualization_.enabled ? next_static_publication :
        std::chrono::steady_clock::time_point::max();
      visualization_cv_.wait_until(lock, wake_time, [this]() {
        return stop_visualization_ || pending_capture_.has_value() ||
               pending_obstacle_visualization_.has_value();
      });
      if (stop_visualization_) {
        return;
      }
      if (pending_capture_) {
        capture_work = std::move(pending_capture_);
        pending_capture_.reset();
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
    if (capture_work) {
      try {
        PublishCapture(*capture_work);
      } catch (const std::exception & error) {
        RCLCPP_ERROR_THROTTLE(
          node_.get_logger(), *node_.get_clock(), 1000,
          "MPPI capture readback failed: %s", error.what());
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
  if (obstacle_costmap_publisher_) {
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
    latest_cost_terms_.reset();
  }
  {
    // pending_capture_ is kept: the worker must still collect it to free the
    // capture slot, and drops it unpublished because its epoch is stale.
    std::lock_guard<std::mutex> visualization_lock(visualization_mutex_);
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
  bool capture_cost_terms = false;
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
    capture_cost_terms = cost_terms_.enabled &&
      std::chrono::steady_clock::now() >= next_cost_terms_time_;
    trajectory = controller_->PlanLatest(
      capture_visualization ? controller_->config().num_rollouts : 0U,
      capture_cost_terms);
    // A request the controller could not stage, because the worker has not yet
    // collected the previous capture, is retried on the next solve.
    if (trajectory.capture_id == 0U) {
      capture_visualization = false;
      capture_cost_terms = false;
    }
    if (capture_cost_terms) {
      next_cost_terms_time_ += cost_terms_period_;
      const auto now = std::chrono::steady_clock::now();
      if (next_cost_terms_time_ <= now) {
        next_cost_terms_time_ = now + cost_terms_period_;
      }
    }
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
  if (solution->capture_id != 0U) {
    // Queued even if a reset has made this solve stale, so the slot is freed.
    QueueCapture(CaptureWork{
        solution, node_.get_clock()->now(), reset_epoch, capture_visualization});
  }
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

void MppiRosRuntime::GripWorker() {
  const auto period = std::chrono::nanoseconds(
    static_cast<std::int64_t>(std::llround(1.0e9 / grip_status_.rate_hz)));
  auto next = std::chrono::steady_clock::now() + period;
  while (true) {
    {
      std::unique_lock<std::mutex> lock(worker_mutex_);
      if (worker_cv_.wait_until(lock, next, [this]() {return stop_workers_;})) {
        return;
      }
    }
    PublishGripStatus();
    const auto now = std::chrono::steady_clock::now();
    next += period;
    if (next <= now) {
      next = now + period;
    }
  }
}

void MppiRosRuntime::PublishGripStatus() {
  std::optional<VehicleObservation> observation;
  std::uint64_t reset_epoch = 0U;
  {
    std::lock_guard<std::mutex> lock(worker_mutex_);
    observation = latest_observation_;
    reset_epoch = reset_epoch_;
  }
  std::shared_ptr<const PlannedTrajectory> solution;
  {
    std::lock_guard<std::mutex> lock(solution_mutex_);
    solution = published_solution_;
  }
  if (reset_epoch != grip_reset_epoch_) {
    grip_monitor_.Reset();
    grip_reset_epoch_ = reset_epoch;
  }
  // Only new observations: the measured half would otherwise repeat.
  if (!observation || !solution || observation->pose_time_ns == grip_last_pose_time_ns_) {
    return;
  }
  grip_last_pose_time_ns_ = observation->pose_time_ns;
  try {
    grip_status_publisher_->publish(ToGripStatusMessage(
        *observation, *solution, controller_->config().vehicle,
        controller_->config().model_kind, grip_monitor_));
  } catch (const std::exception & error) {
    RCLCPP_ERROR_THROTTLE(
      node_.get_logger(), *node_.get_clock(), 1000,
      "MPPI grip status publication failed: %s", error.what());
  }
}

void MppiRosRuntime::MuWorker() {
  const auto period = std::chrono::nanoseconds(
    static_cast<std::int64_t>(std::llround(1.0e9 / mu_estimate_.rate_hz)));
  auto next = std::chrono::steady_clock::now() + period;
  while (true) {
    {
      std::unique_lock<std::mutex> lock(worker_mutex_);
      if (worker_cv_.wait_until(lock, next, [this]() {return stop_workers_;})) {
        return;
      }
    }
    PublishMuEstimate();
    const auto now = std::chrono::steady_clock::now();
    next += period;
    if (next <= now) {
      next = now + period;
    }
  }
}

void MppiRosRuntime::PublishMuEstimate() {
  std::optional<VehicleObservation> observation;
  {
    std::lock_guard<std::mutex> lock(worker_mutex_);
    observation = latest_observation_;
  }
  if (!observation || observation->pose_time_ns == mu_last_pose_time_ns_) {
    return;
  }
  mu_last_pose_time_ns_ = observation->pose_time_ns;
  BodyState state;
  state[kYawRate] = observation->yaw_rate_radps;
  state[kSpeed] = observation->speed_mps;
  state[kSideslip] = observation->sideslip_rad;
  state[kDrivenWheelSpeed] = observation->driven_wheel_speed_mps;
  Control control;
  control[kSteering] = observation->measured_steering_rad;
  control[kWheelTorque] = observation->measured_torque_nm;
  // The EKF carries no rear steering angle, so a 4WS car uses the rear command
  // being applied now; front-steer-only cars have none.
  if (controller_->config().model_kind == ModelKind::kDynamicBicycleFiala4ws) {
    std::lock_guard<std::mutex> lock(solution_mutex_);
    if (published_solution_ && !published_solution_->controls.empty()) {
      control[kRearSteering] = published_solution_->controls.front()[kRearSteering];
    }
  }
  const auto estimate = mu_estimator_->Update(
    static_cast<double>(observation->pose_time_ns) * 1.0e-9, state, control,
    observation->longitudinal_acceleration_mps2, observation->lateral_acceleration_mps2);
  try {
    std_msgs::msg::Float32 message;
    message.data = estimate.mu;
    mu_estimate_publisher_->publish(message);
  } catch (const std::exception & error) {
    RCLCPP_ERROR_THROTTLE(
      node_.get_logger(), *node_.get_clock(), 1000,
      "MPPI mu estimate publication failed: %s", error.what());
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

  const auto & rate_limit = controller_->config().mppi.control_rate_limit;
  if (last_sent_control_ && !solution->controls.empty()) {
    const float elapsed_s = static_cast<float>(
      publication_time.nanoseconds() - last_sent_time_ns_) * 1.0e-9F;
    const Control & planned = solution->controls.front();
    const Control limited = SlewLimitControl(
      planned, *last_sent_control_, elapsed_s, rate_limit);
    bool changed = false;
    for (std::size_t channel = 0; channel < kControlDim; ++channel) {
      changed = changed || limited[channel] != planned[channel];
    }
    if (changed) {
      auto copy = std::make_shared<PlannedTrajectory>(*solution);
      copy->controls.front() = limited;
      solution = std::move(copy);
    }
  }

  try {
    if (direct_control_.four_wheel && direct_control_.enabled) {
      four_wheel_control_publisher_->publish(
        ToDirectControlMessage(*solution, direct_control_, publication_time));
    } else if (direct_control_.enabled) {
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
  if (!solution->controls.empty()) {
    last_sent_control_ = solution->controls.front();
    last_sent_time_ns_ = publication_time.nanoseconds();
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
  std::optional<CostTerms> cost_terms;
  {
    std::lock_guard<std::mutex> lock(solution_mutex_);
    solution = published_solution_;
    publication_age_ms = published_solution_age_ms_;
    cost_terms = latest_cost_terms_;
  }
  if (!solution) {
    return;
  }
  PublishInfo(*solution, publication_age_ms, cost_terms);
}

}  // namespace xxcar::mppi
