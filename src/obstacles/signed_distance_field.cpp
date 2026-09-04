#include "xx_mppi/obstacles/signed_distance_field.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace xxcar::mppi {
namespace {

constexpr float kLargeDistance = 1.0e20F;

void DistanceTransform1D(
  const float * input, float * output, const std::uint32_t count,
  std::vector<std::int32_t> & locations, std::vector<float> & boundaries)
{
  std::int32_t k = 0;
  locations[0] = 0;
  boundaries[0] = -std::numeric_limits<float>::infinity();
  boundaries[1] = std::numeric_limits<float>::infinity();
  for (std::uint32_t q = 1; q < count; ++q) {
    float intersection = 0.0F;
    do {
      const auto location = locations[static_cast<std::size_t>(k)];
      intersection = ((input[q] + static_cast<float>(q * q)) -
        (input[static_cast<std::size_t>(location)] +
        static_cast<float>(location * location))) /
        (2.0F * static_cast<float>(static_cast<std::int32_t>(q) - location));
      if (intersection <= boundaries[static_cast<std::size_t>(k)]) {
        --k;
      }
    } while (k >= 0 && intersection <= boundaries[static_cast<std::size_t>(k)]);
    ++k;
    locations[static_cast<std::size_t>(k)] = static_cast<std::int32_t>(q);
    boundaries[static_cast<std::size_t>(k)] = intersection;
    boundaries[static_cast<std::size_t>(k + 1)] =
      std::numeric_limits<float>::infinity();
  }
  k = 0;
  for (std::uint32_t q = 0; q < count; ++q) {
    while (boundaries[static_cast<std::size_t>(k + 1)] < static_cast<float>(q)) {
      ++k;
    }
    const float delta = static_cast<float>(
      static_cast<std::int32_t>(q) - locations[static_cast<std::size_t>(k)]);
    output[q] = delta * delta + input[
      static_cast<std::size_t>(locations[static_cast<std::size_t>(k)])];
  }
}

std::vector<float> SquaredDistanceTransform(
  const std::vector<std::uint8_t> & feature, const std::uint32_t width,
  const std::uint32_t height)
{
  const std::size_t cells = static_cast<std::size_t>(width) * height;
  std::vector<float> first(cells);
  std::vector<float> result(cells);
  const std::uint32_t longest = std::max(width, height);
  std::vector<float> input(longest);
  std::vector<float> output(longest);
  std::vector<std::int32_t> locations(longest);
  std::vector<float> boundaries(static_cast<std::size_t>(longest) + 1U);

  for (std::uint32_t y = 0; y < height; ++y) {
    for (std::uint32_t x = 0; x < width; ++x) {
      input[x] = feature[static_cast<std::size_t>(y) * width + x] ? 0.0F : kLargeDistance;
    }
    DistanceTransform1D(input.data(), output.data(), width, locations, boundaries);
    for (std::uint32_t x = 0; x < width; ++x) {
      first[static_cast<std::size_t>(y) * width + x] = output[x];
    }
  }
  for (std::uint32_t x = 0; x < width; ++x) {
    for (std::uint32_t y = 0; y < height; ++y) {
      input[y] = first[static_cast<std::size_t>(y) * width + x];
    }
    DistanceTransform1D(input.data(), output.data(), height, locations, boundaries);
    for (std::uint32_t y = 0; y < height; ++y) {
      result[static_cast<std::size_t>(y) * width + x] = output[y];
    }
  }
  return result;
}

}  // namespace

SignedDistanceFieldBuilder::SignedDistanceFieldBuilder(ObstacleConfig config)
: config_(config)
{
  if (!(config_.grid_resolution_m > 0.0F) || !(config_.grid_width_m > 0.0F) ||
    !(config_.grid_height_m > 0.0F) || !(config_.maximum_distance_m > 0.0F) ||
    config_.obstacle_inflation_radius_m < 0.0F)
  {
    throw std::invalid_argument("invalid signed-distance-field configuration");
  }
  width_ = static_cast<std::uint32_t>(std::ceil(
      config_.grid_width_m / config_.grid_resolution_m));
  height_ = static_cast<std::uint32_t>(std::ceil(
      config_.grid_height_m / config_.grid_resolution_m));
  if (width_ < 2U || height_ < 2U) {
    throw std::invalid_argument("signed-distance-field grid is too small");
  }
}

ObstacleField SignedDistanceFieldBuilder::Build(
  const std::vector<Point2D> & obstacle_points, const Pose2D & center,
  const std::int64_t stamp_ns, const std::uint64_t generation) const
{
  ObstacleField field;
  field.stamp_ns = stamp_ns;
  field.generation = generation;
  field.resolution_m = config_.grid_resolution_m;
  field.width = width_;
  field.height = height_;
  field.origin_east_m = center.east_m - 0.5F * static_cast<float>(width_) * field.resolution_m;
  field.origin_north_m = center.north_m - 0.5F * static_cast<float>(height_) * field.resolution_m;
  const std::size_t cells = static_cast<std::size_t>(width_) * height_;
  std::vector<std::uint8_t> occupied(cells, 0U);
  const int inflation_cells = static_cast<int>(std::ceil(
      config_.obstacle_inflation_radius_m / field.resolution_m));
  const float inflation_squared = config_.obstacle_inflation_radius_m *
    config_.obstacle_inflation_radius_m;
  for (const auto & point : obstacle_points) {
    if (!std::isfinite(point.east_m) || !std::isfinite(point.north_m)) {
      continue;
    }
    const int center_x = static_cast<int>(std::floor(
        (point.east_m - field.origin_east_m) / field.resolution_m));
    const int center_y = static_cast<int>(std::floor(
        (point.north_m - field.origin_north_m) / field.resolution_m));
    for (int dy = -inflation_cells; dy <= inflation_cells; ++dy) {
      for (int dx = -inflation_cells; dx <= inflation_cells; ++dx) {
        const int x = center_x + dx;
        const int y = center_y + dy;
        if (x < 0 || y < 0 || x >= static_cast<int>(width_) ||
          y >= static_cast<int>(height_))
        {
          continue;
        }
        const float offset_x = static_cast<float>(dx) * field.resolution_m;
        const float offset_y = static_cast<float>(dy) * field.resolution_m;
        if (inflation_cells == 0 ||
          offset_x * offset_x + offset_y * offset_y <= inflation_squared)
        {
          occupied[static_cast<std::size_t>(y) * width_ + static_cast<std::uint32_t>(x)] = 1U;
        }
      }
    }
  }

  const bool has_obstacle = std::any_of(
    occupied.begin(), occupied.end(), [](const std::uint8_t value) {return value != 0U;});
  field.signed_distance_m.assign(cells, config_.maximum_distance_m);
  if (!has_obstacle) {
    return field;
  }
  std::vector<std::uint8_t> free_space(cells);
  std::transform(
    occupied.begin(), occupied.end(), free_space.begin(),
    [](const std::uint8_t value) {return value == 0U ? 1U : 0U;});
  const auto distance_to_obstacle = SquaredDistanceTransform(occupied, width_, height_);
  const auto distance_to_free = SquaredDistanceTransform(free_space, width_, height_);
  for (std::size_t index = 0; index < cells; ++index) {
    const float cells_distance = std::sqrt(
      occupied[index] ? distance_to_free[index] : distance_to_obstacle[index]);
    const float signed_distance = cells_distance * field.resolution_m *
      (occupied[index] ? -1.0F : 1.0F);
    field.signed_distance_m[index] = std::clamp(
      signed_distance, -config_.maximum_distance_m, config_.maximum_distance_m);
  }
  return field;
}

float SampleSignedDistance(
  const ObstacleField & field, const float east_m, const float north_m,
  const float outside_value_m)
{
  if (!field.valid()) {
    return outside_value_m;
  }
  const float grid_x = (east_m - field.origin_east_m) / field.resolution_m - 0.5F;
  const float grid_y = (north_m - field.origin_north_m) / field.resolution_m - 0.5F;
  if (grid_x < 0.0F || grid_y < 0.0F ||
    grid_x >= static_cast<float>(field.width - 1U) ||
    grid_y >= static_cast<float>(field.height - 1U))
  {
    return outside_value_m;
  }
  const auto x0 = static_cast<std::uint32_t>(std::floor(grid_x));
  const auto y0 = static_cast<std::uint32_t>(std::floor(grid_y));
  const float tx = grid_x - static_cast<float>(x0);
  const float ty = grid_y - static_cast<float>(y0);
  const auto at = [&field](const std::uint32_t x, const std::uint32_t y) {
      return field.signed_distance_m[static_cast<std::size_t>(y) * field.width + x];
    };
  const float lower = at(x0, y0) + tx * (at(x0 + 1U, y0) - at(x0, y0));
  const float upper = at(x0, y0 + 1U) + tx *
    (at(x0 + 1U, y0 + 1U) - at(x0, y0 + 1U));
  return lower + ty * (upper - lower);
}

}  // namespace xxcar::mppi
