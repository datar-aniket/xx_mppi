#include "xx_mppi/reference/raceline.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>

#include "xx_mppi/math.hpp"

namespace xxcar::mppi {
namespace {

std::string TrimCsvField(const std::string & value) {
  std::size_t first = 0U;
  while (first < value.size() &&
    std::isspace(static_cast<unsigned char>(value[first])) != 0)
  {
    ++first;
  }
  std::size_t last = value.size();
  while (last > first &&
    std::isspace(static_cast<unsigned char>(value[last - 1U])) != 0)
  {
    --last;
  }
  return value.substr(first, last - first);
}

std::vector<std::string> SplitCsvLine(const std::string & line) {
  std::vector<std::string> fields;
  std::string field;
  bool quoted = false;
  for (const char ch : line) {
    if (ch == '"') {
      quoted = !quoted;
    } else if (ch == ',' && !quoted) {
      fields.push_back(TrimCsvField(field));
      field.clear();
    } else {
      field.push_back(ch);
    }
  }
  fields.push_back(TrimCsvField(field));
  return fields;
}

std::size_t FindColumn(
  const std::unordered_map<std::string, std::size_t> & columns,
  const std::initializer_list<std::string_view> aliases,
  const bool required = true)
{
  for (const auto alias : aliases) {
    const auto it = columns.find(std::string(alias));
    if (it != columns.end()) {
      return it->second;
    }
  }
  if (!required) {
    return std::numeric_limits<std::size_t>::max();
  }
  std::ostringstream message;
  message << "raceline CSV is missing required column; accepted aliases:";
  for (const auto alias : aliases) {
    message << ' ' << alias;
  }
  throw std::runtime_error(message.str());
}

float ParseField(
  const std::vector<std::string> & fields, const std::size_t index,
  const float fallback = 0.0F)
{
  if (index == std::numeric_limits<std::size_t>::max()) {
    return fallback;
  }
  if (index >= fields.size() || fields[index].empty()) {
    throw std::runtime_error("raceline CSV row has a missing numeric field");
  }
  std::size_t parsed = 0;
  const float value = std::stof(fields[index], &parsed);
  if (parsed != fields[index].size() || !std::isfinite(value)) {
    throw std::runtime_error("raceline CSV contains a non-finite or invalid numeric field");
  }
  return value;
}

float Lerp(const float a, const float b, const float t) { return a + t * (b - a); }

float InterpolateAngle(const float a, const float b, const float t) {
  const float sin_value = Lerp(std::sin(a), std::sin(b), t);
  const float cos_value = Lerp(std::cos(a), std::cos(b), t);
  return std::atan2(sin_value, cos_value);
}

}  // namespace

Raceline Raceline::LoadCsv(const std::string & path, const float closed_track_gap_m) {
  std::ifstream stream(path);
  if (!stream) {
    throw std::runtime_error("failed to open raceline CSV: " + path);
  }

  std::string line;
  if (!std::getline(stream, line)) {
    throw std::runtime_error("raceline CSV is empty: " + path);
  }
  const auto header = SplitCsvLine(line);
  std::unordered_map<std::string, std::size_t> columns;
  for (std::size_t i = 0; i < header.size(); ++i) {
    columns.emplace(header[i], i);
  }

  const auto s = FindColumn(columns, {"s"});
  const auto curvature = FindColumn(columns, {"k", "kappa", "curvature"});
  const auto east = FindColumn(columns, {"E", "East", "east", "x"});
  const auto north = FindColumn(columns, {"N", "North", "north", "y"});
  const auto heading = FindColumn(columns, {"phi", "psi", "heading"});
  const auto speed = FindColumn(columns, {"V", "speed", "v"});
  const auto e_min = FindColumn(columns, {"e_min", "left_bound"});
  const auto e_max = FindColumn(columns, {"e_max", "right_bound"});
  const auto yaw_rate = FindColumn(columns, {"r", "yaw_rate"}, false);
  const auto sideslip = FindColumn(columns, {"beta", "sideslip"}, false);
  const auto steering = FindColumn(columns, {"delta_cmd", "delta", "steering_angle_rad"}, false);
  const auto torque = FindColumn(
    columns, {"wheel_torque_cmd", "torque_cmd", "engine_torque_cmd", "engine_torque"}, false);
  const auto wheel_speed = FindColumn(
    columns, {"rear_wr", "driven_wheel_speed", "motor_speed_mps"}, false);

  Raceline raceline;
  std::size_t row_number = 1;
  while (std::getline(stream, line)) {
    ++row_number;
    if (line.empty()) {
      continue;
    }
    try {
      const auto fields = SplitCsvLine(line);
      ReferencePoint point;
      point.s_m = ParseField(fields, s);
      point.curvature_inv_m = ParseField(fields, curvature);
      point.east_m = ParseField(fields, east);
      point.north_m = ParseField(fields, north);
      point.heading_from_north_rad = ParseField(fields, heading);
      point.yaw_rate_radps = ParseField(fields, yaw_rate);
      point.speed_mps = ParseField(fields, speed);
      point.sideslip_rad = ParseField(fields, sideslip);
      point.steering_rad = ParseField(fields, steering);
      point.torque_nm = ParseField(fields, torque);
      point.driven_wheel_speed_mps = ParseField(fields, wheel_speed, point.speed_mps);
      point.e_min_m = ParseField(fields, e_min);
      point.e_max_m = ParseField(fields, e_max);
      raceline.points_.push_back(point);
    } catch (const std::exception & error) {
      throw std::runtime_error(
              "failed to parse " + path + ":" + std::to_string(row_number) + ": " + error.what());
    }
  }

  if (raceline.points_.size() < 2) {
    throw std::runtime_error("raceline CSV must contain at least two points");
  }
  for (std::size_t i = 1; i < raceline.points_.size(); ++i) {
    if (!(raceline.points_[i].s_m > raceline.points_[i - 1].s_m)) {
      throw std::runtime_error("raceline s must be finite and strictly increasing");
    }
  }
  const float closing_distance_m = std::hypot(
    raceline.points_.back().east_m - raceline.points_.front().east_m,
    raceline.points_.back().north_m - raceline.points_.front().north_m);
  raceline.closed_ = closing_distance_m <= closed_track_gap_m;
  if (raceline.closed_ && closing_distance_m > 1.0e-5F) {
    auto closing_point = raceline.points_.front();
    closing_point.s_m = raceline.points_.back().s_m + closing_distance_m;
    raceline.points_.push_back(closing_point);
  }
  return raceline;
}

float Raceline::WrapOrClamp(const float s_m) const {
  if (!closed_) {
    return clampf(s_m, s_min(), s_max());
  }
  float wrapped = std::fmod(s_m - s_min(), length());
  if (wrapped < 0.0F) {
    wrapped += length();
  }
  return s_min() + wrapped;
}

float Raceline::UnwrapNear(const float wrapped_s_m, const float hint_m) const {
  if (!closed_) {
    return wrapped_s_m;
  }
  return wrapped_s_m + std::round((hint_m - wrapped_s_m) / length()) * length();
}

ReferencePoint Raceline::Interpolate(const float unwrapped_s_m) const {
  const float query = WrapOrClamp(unwrapped_s_m);
  const auto upper = std::upper_bound(
    points_.begin(), points_.end(), query,
    [](const float value, const ReferencePoint & point) { return value < point.s_m; });
  if (upper == points_.begin()) {
    return points_.front();
  }
  if (upper == points_.end()) {
    return points_.back();
  }
  const auto & b = *upper;
  const auto & a = *(upper - 1);
  const float t = (query - a.s_m) / (b.s_m - a.s_m);
  ReferencePoint point;
  point.s_m = unwrapped_s_m;
  point.curvature_inv_m = Lerp(a.curvature_inv_m, b.curvature_inv_m, t);
  point.east_m = Lerp(a.east_m, b.east_m, t);
  point.north_m = Lerp(a.north_m, b.north_m, t);
  point.heading_from_north_rad = InterpolateAngle(
    a.heading_from_north_rad, b.heading_from_north_rad, t);
  point.yaw_rate_radps = Lerp(a.yaw_rate_radps, b.yaw_rate_radps, t);
  point.speed_mps = Lerp(a.speed_mps, b.speed_mps, t);
  point.sideslip_rad = Lerp(a.sideslip_rad, b.sideslip_rad, t);
  point.steering_rad = Lerp(a.steering_rad, b.steering_rad, t);
  point.torque_nm = Lerp(a.torque_nm, b.torque_nm, t);
  point.driven_wheel_speed_mps = Lerp(
    a.driven_wheel_speed_mps, b.driven_wheel_speed_mps, t);
  point.e_min_m = Lerp(a.e_min_m, b.e_min_m, t);
  point.e_max_m = Lerp(a.e_max_m, b.e_max_m, t);
  return point;
}

std::pair<float, float> Raceline::ToCartesian(
  const float unwrapped_s_m, const float lateral_deviation_m) const
{
  const auto center = Interpolate(unwrapped_s_m);
  // EPIC's path tangent is (-sin(phi), cos(phi)); positive e is its left normal.
  return {
    center.east_m - lateral_deviation_m * std::cos(center.heading_from_north_rad),
    center.north_m - lateral_deviation_m * std::sin(center.heading_from_north_rad)};
}

Projection Raceline::Project(
  const float east_m, const float north_m, const float course_heading_from_north_rad,
  const std::optional<float> unwrapped_s_hint, const float projection_window_m) const
{
  float best_distance_squared = std::numeric_limits<float>::infinity();
  Projection best;
  for (std::size_t i = 0; i + 1 < points_.size(); ++i) {
    const auto & a = points_[i];
    const auto & b = points_[i + 1];
    if (unwrapped_s_hint) {
      const float candidate_a = UnwrapNear(a.s_m, *unwrapped_s_hint);
      const float candidate_b = candidate_a + (b.s_m - a.s_m);
      const float distance_to_segment = *unwrapped_s_hint < candidate_a ?
        candidate_a - *unwrapped_s_hint :
        (*unwrapped_s_hint > candidate_b ? *unwrapped_s_hint - candidate_b : 0.0F);
      if (distance_to_segment > projection_window_m) {
        continue;
      }
    }
    const float dx = b.east_m - a.east_m;
    const float dy = b.north_m - a.north_m;
    const float norm_squared = dx * dx + dy * dy;
    if (norm_squared <= 1.0e-12F) {
      continue;
    }
    const float fraction = clampf(
      ((east_m - a.east_m) * dx + (north_m - a.north_m) * dy) / norm_squared,
      0.0F, 1.0F);
    const float projected_east = a.east_m + fraction * dx;
    const float projected_north = a.north_m + fraction * dy;
    const float error_east = east_m - projected_east;
    const float error_north = north_m - projected_north;
    const float distance_squared = error_east * error_east + error_north * error_north;
    if (distance_squared >= best_distance_squared) {
      continue;
    }
    const float inverse_norm = 1.0F / std::sqrt(norm_squared);
    const float tangent_east = dx * inverse_norm;
    const float tangent_north = dy * inverse_norm;
    float projected_s = Lerp(a.s_m, b.s_m, fraction);
    if (unwrapped_s_hint) {
      projected_s = UnwrapNear(projected_s, *unwrapped_s_hint);
    }
    const float path_heading = InterpolateAngle(
      a.heading_from_north_rad, b.heading_from_north_rad, fraction);
    best = Projection{
      projected_s,
      error_east * (-tangent_north) + error_north * tangent_east,
      wrap_to_pi(course_heading_from_north_rad - path_heading),
      i,
      true};
    best_distance_squared = distance_squared;
  }

  // A bad hint must never silently map to segment zero. Reacquire globally.
  if (!best.valid && unwrapped_s_hint) {
    return Project(
      east_m, north_m, course_heading_from_north_rad, std::nullopt,
      projection_window_m);
  }
  return best;
}

ReferenceHorizon Raceline::Sample(
  const float unwrapped_s0_m, const std::uint16_t horizon, const float dt_s) const
{
  if (horizon == 0 || !(dt_s > 0.0F)) {
    throw std::invalid_argument("reference horizon and dt must be positive");
  }
  ReferenceHorizon result;
  result.states.resize(static_cast<std::size_t>(horizon) + 1U);
  result.controls.resize(horizon);
  result.curvature.resize(horizon);
  result.s_grid.resize(static_cast<std::size_t>(horizon) + 1U);
  result.speed_profile.resize(static_cast<std::size_t>(horizon) + 1U);
  result.e_min.resize(static_cast<std::size_t>(horizon) + 1U);
  result.e_max.resize(static_cast<std::size_t>(horizon) + 1U);

  float s_query = unwrapped_s0_m;
  for (std::size_t i = 0; i <= horizon; ++i) {
    const auto point = Interpolate(s_query);
    result.s_grid[i] = s_query;
    result.speed_profile[i] = point.speed_mps;
    result.e_min[i] = point.e_min_m;
    result.e_max[i] = point.e_max_m;
    result.states[i] = State{
      point.yaw_rate_radps, point.speed_mps, point.sideslip_rad,
      point.driven_wheel_speed_mps, 0.0F, 0.0F, s_query};
    if (i < horizon) {
      result.controls[i] = Control{point.steering_rad, point.torque_nm};
      result.curvature[i] = point.curvature_inv_m;
      s_query += point.speed_mps * dt_s;
    }
  }
  return result;
}

std::pair<float, float> StateToEnu(
  const Raceline & raceline, const State & state, const FrameKind frame)
{
  if (frame == FrameKind::kCartesian) {
    return {state[kEastM], state[kNorthM]};
  }
  return raceline.ToCartesian(state[kPathEvolution], state[kLateralDeviation]);
}

Projection ContinuousProjector::Update(
  const float east_m, const float north_m, const float body_heading_from_north_rad,
  const float sideslip_rad)
{
  const auto result = raceline_.Project(
    east_m, north_m, wrap_to_pi(body_heading_from_north_rad + sideslip_rad),
    s_hint_, window_m_);
  if (result.valid) {
    s_hint_ = result.s_m;
  }
  return result;
}

}  // namespace xxcar::mppi
