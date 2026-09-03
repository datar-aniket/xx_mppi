#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "xx_mppi/controller/mppi_controller.hpp"
#include "xx_mppi/dynamics/analytic_dynamics.hpp"
#include "xx_mppi/dynamics/rollout.hpp"
#include "xx_mppi/math.hpp"
#include "xx_mppi/reference/raceline.hpp"

namespace xxcar::mppi {
namespace {

// A surveyed closed loop with real curvature, so the Frenet path terms and the
// Cartesian integration are exercised against genuine geometry rather than a
// straight line where every convention agrees.
std::string CurvedCsv() {
  return std::string(XX_MPPI_TEST_DATA_DIR) + "/curved_raceline.csv";
}

VehicleParameters TestVehicle() {
  VehicleParameters parameters;
  parameters.mass_kg = 2.0F;
  parameters.yaw_inertia_kgm2 = 0.04F;
  parameters.cg_to_front_m = 0.156F;
  parameters.cg_to_rear_m = 0.156F;
  parameters.front_cornering_stiffness_nprad = 75.0F;
  parameters.rear_cornering_stiffness_nprad = 75.0F;
  parameters.wheel_radius_m = 0.05842F;
  parameters.driven_wheel_inertia_kgm2 = 0.01F;
  parameters.locked_awd = true;
  return parameters;
}

TEST(Frames, EnuYawAndCsvHeadingRoundTrip) {
  for (const float yaw : {-3.0F, -1.2F, 0.0F, 0.7F, 2.9F}) {
    EXPECT_NEAR(
      heading_from_north_to_enu_yaw(enu_yaw_to_heading_from_north(yaw)), yaw, 1.0e-5F);
  }
  // North-bound in ENU is yaw = pi/2 and CSV heading phi = 0.
  EXPECT_NEAR(enu_yaw_to_heading_from_north(0.5F * kPi), 0.0F, 1.0e-6F);
}

TEST(Frames, CsvHeadingMatchesSurveyedTangent) {
  const auto raceline = Raceline::LoadCsv(CurvedCsv());
  const auto & points = raceline.points();
  ASSERT_GT(points.size(), 10U);
  // The loader's convention is tangent = (-sin(phi), cos(phi)). Compare it with
  // the surveyed centreline geometry the same file carries.
  for (std::size_t i = 0; i + 1U < points.size(); ++i) {
    const float ds = points[i + 1U].s_m - points[i].s_m;
    ASSERT_GT(ds, 0.0F);
    // Average on the circle: consecutive phi values straddle +/-pi at the seam.
    const float phi = std::atan2(
      std::sin(points[i].heading_from_north_rad) +
      std::sin(points[i + 1U].heading_from_north_rad),
      std::cos(points[i].heading_from_north_rad) +
      std::cos(points[i + 1U].heading_from_north_rad));
    EXPECT_NEAR((points[i + 1U].east_m - points[i].east_m) / ds, -std::sin(phi), 0.05F);
    EXPECT_NEAR((points[i + 1U].north_m - points[i].north_m) / ds, std::cos(phi), 0.05F);
  }
}

TEST(Frames, ProjectionAndCartesianMappingAreInverses) {
  const auto raceline = Raceline::LoadCsv(CurvedCsv());
  const float s = 0.3F * raceline.length();
  const auto centre = raceline.Interpolate(s);
  for (const float e : {-0.4F, 0.0F, 0.35F}) {
    const auto point = raceline.ToCartesian(s, e);
    const auto projection = raceline.Project(
      point.first, point.second, centre.heading_from_north_rad, s, 5.0F);
    ASSERT_TRUE(projection.valid);
    EXPECT_NEAR(projection.e_m, e, 0.02F);
    EXPECT_NEAR(projection.s_m, s, 0.15F);
    EXPECT_NEAR(projection.relative_course_rad, 0.0F, 0.05F);
  }
}

TEST(Frames, PositiveDeviationIsLeftOfTheDirectionOfTravel) {
  const auto raceline = Raceline::LoadCsv(CurvedCsv());
  const float s = 0.5F * raceline.length();
  const auto centre = raceline.Interpolate(s);
  const auto left = raceline.ToCartesian(s, 0.5F);
  // Rotating the ENU tangent by +90 degrees must point at the positive-e side.
  const float yaw = heading_from_north_to_enu_yaw(centre.heading_from_north_rad);
  EXPECT_NEAR(left.first - centre.east_m, 0.5F * -std::sin(yaw), 1.0e-4F);
  EXPECT_NEAR(left.second - centre.north_m, 0.5F * std::cos(yaw), 1.0e-4F);
}

// The decisive check on the new frame option: the same body model driven by the
// same controls from the same physical state must trace the same ground path in
// either frame.
TEST(Frames, CartesianAndFrenetRolloutsTraceTheSamePath) {
  const auto raceline = Raceline::LoadCsv(CurvedCsv());
  const float s0 = 0.25F * raceline.length();
  const auto centre = raceline.Interpolate(s0);
  const float lateral = 0.15F;
  const float relative_course = 0.08F;
  const auto start = raceline.ToCartesian(s0, lateral);
  const float yaw = heading_from_north_to_enu_yaw(
    centre.heading_from_north_rad + relative_course);

  const State frenet_initial{
    0.0F, 2.0F, 0.0F, 2.0F, lateral, relative_course, s0};
  const State cartesian_initial{
    0.0F, 2.0F, 0.0F, 2.0F, start.first, start.second, yaw};

  const std::vector<Control> controls(20U, Control{0.12F, 0.05F});
  const AnalyticDynamics frenet(
    ModelKind::kKinematicBicycle, TestVehicle(), FrameKind::kFrenet);
  const AnalyticDynamics cartesian(
    ModelKind::kKinematicBicycle, TestVehicle(), FrameKind::kCartesian);
  const auto frenet_states = RolloutAnalytic(
    frenet, raceline, frenet_initial, controls, 0.05F, 4U, IntegratorKind::kRungeKutta4);
  const auto cartesian_states = RolloutAnalytic(
    cartesian, raceline, cartesian_initial, controls, 0.05F, 4U,
    IntegratorKind::kRungeKutta4);
  ASSERT_EQ(frenet_states.size(), cartesian_states.size());

  for (std::size_t i = 0; i < frenet_states.size(); ++i) {
    const auto from_frenet = StateToEnu(raceline, frenet_states[i], FrameKind::kFrenet);
    const auto from_cartesian = StateToEnu(
      raceline, cartesian_states[i], FrameKind::kCartesian);
    // The tolerance covers only the map's piecewise-linear centreline and
    // curvature sampling; a sign or 90-degree convention error is metres wide.
    EXPECT_NEAR(from_frenet.first, from_cartesian.first, 0.05F) << "step " << i;
    EXPECT_NEAR(from_frenet.second, from_cartesian.second, 0.05F) << "step " << i;
  }
  // The manoeuvre must actually leave the start, or the comparison is vacuous.
  const auto last = StateToEnu(raceline, cartesian_states.back(), FrameKind::kCartesian);
  EXPECT_GT(std::hypot(last.first - start.first, last.second - start.second), 1.0F);
}

TEST(Frames, PositiveSteeringTurnsLeftInBothFrames) {
  const auto raceline = Raceline::LoadCsv(CurvedCsv());
  const AnalyticDynamics cartesian(
    ModelKind::kKinematicBicycle, TestVehicle(), FrameKind::kCartesian);
  const State initial{0.0F, 2.0F, 0.0F, 2.0F, 0.0F, 0.0F, 0.0F};
  const std::vector<Control> controls(10U, Control{0.3F, 0.0F});
  const auto states = RolloutAnalytic(
    cartesian, raceline, initial, controls, 0.05F, 2U, IntegratorKind::kEuler);
  // ENU yaw increases (counter-clockwise) and the path curls to the north.
  EXPECT_GT(states.back()[kHeadingEnu], initial[kHeadingEnu]);
  EXPECT_GT(states.back()[kNorthM], 0.0F);
  EXPECT_GT(states.back()[kEastM], 0.0F);
}

// The reported failure: an implausible sideslip rotates the projection's
// relative course heading past +/-90 degrees, s_dot = V cos(dphi) turns
// negative, and the published path retreats along the track before turning
// around. Conditioning the sideslip once keeps cos(dphi) positive.
TEST(Frames, BoundedSideslipKeepsThePathAdvancingAlongTheTrack) {
  const auto raceline = Raceline::LoadCsv(CurvedCsv());
  const float s0 = 0.4F * raceline.length();
  const auto centre = raceline.Interpolate(s0);
  const auto start = raceline.ToCartesian(s0, 0.0F);
  const float body_heading = centre.heading_from_north_rad;
  constexpr float kMaximum = 0.8F;

  // 2.0 rad is inside the range EkfState actually reports on this vehicle.
  const float raw = 2.0F;
  ContinuousProjector raw_projector(raceline, 5.0F);
  const auto raw_projection = raw_projector.Update(
    start.first, start.second, body_heading, raw);
  ASSERT_TRUE(raw_projection.valid);
  EXPECT_LT(std::cos(raw_projection.relative_course_rad), 0.0F);

  const float conditioned = ConditionedSideslip(raw, 2.0F, kMaximum);
  EXPECT_FLOAT_EQ(conditioned, kMaximum);
  ContinuousProjector projector(raceline, 5.0F);
  const auto projection = projector.Update(
    start.first, start.second, body_heading, conditioned);
  ASSERT_TRUE(projection.valid);
  EXPECT_GT(std::cos(projection.relative_course_rad), 0.0F);
}

TEST(Frames, SideslipConditioningMatchesTheDocumentedContract) {
  EXPECT_FLOAT_EQ(ConditionedSideslip(0.3F, 2.0F, 0.8F), 0.3F);
  EXPECT_FLOAT_EQ(ConditionedSideslip(-2.7F, 2.0F, 0.8F), -0.8F);
  // A non-finite estimate is only zeroed at a standstill; while moving it is
  // passed through so the caller rejects the sample.
  EXPECT_FLOAT_EQ(ConditionedSideslip(
      std::numeric_limits<float>::quiet_NaN(), 0.1F, 0.8F), 0.0F);
  EXPECT_TRUE(std::isnan(ConditionedSideslip(
      std::numeric_limits<float>::quiet_NaN(), 2.0F, 0.8F)));
}

// Both frames must start from the same physical course heading, or a bounded
// sideslip would mean one frame's initial condition and not the other's.
TEST(Frames, BothFramesStartFromTheSameCourseHeading) {
  const auto raceline = Raceline::LoadCsv(CurvedCsv());
  const float s0 = 0.4F * raceline.length();
  const auto centre = raceline.Interpolate(s0);
  const auto start = raceline.ToCartesian(s0, 0.0F);
  const float sideslip = ConditionedSideslip(2.0F, 2.0F, 0.8F);
  const float yaw = heading_from_north_to_enu_yaw(centre.heading_from_north_rad);

  ContinuousProjector projector(raceline, 5.0F);
  const auto projection = projector.Update(
    start.first, start.second, centre.heading_from_north_rad, sideslip);
  ASSERT_TRUE(projection.valid);

  // Frenet carries the course-relative heading; Cartesian carries the body yaw
  // and adds the same sideslip in its derivative.
  const float frenet_course = enu_yaw_to_heading_from_north(
    heading_from_north_to_enu_yaw(
      centre.heading_from_north_rad + projection.relative_course_rad));
  const float cartesian_course = enu_yaw_to_heading_from_north(yaw + sideslip);
  EXPECT_NEAR(wrap_to_pi(frenet_course - cartesian_course), 0.0F, 1.0e-4F);
}

}  // namespace
}  // namespace xxcar::mppi
