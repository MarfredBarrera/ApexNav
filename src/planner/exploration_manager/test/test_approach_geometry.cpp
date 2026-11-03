// approach::candidates (perception gate state 3): bearings, radii, ranks and the new-view rule.
//   colcon test --packages-select exploration_manager
#include <gtest/gtest.h>
#include <exploration_manager/approach_geometry.h>
#include <cmath>
#include <set>

using apexnav_planner::approach::Candidate;
using apexnav_planner::approach::candidates;
using apexnav_planner::approach::maxPath;
using apexnav_planner::approach::radii;
using Eigen::Vector2d;

static double angleFrom(const Vector2d& object, const Vector2d& p, double reference)
{
  double a = std::atan2(p(1) - object(1), p(0) - object(0)) - reference;
  return std::remainder(a, 2 * M_PI) * 180.0 / M_PI;
}

static std::set<int> jitters(const std::vector<Candidate>& cs)
{
  std::set<int> out;
  for (const auto& c : cs) out.insert((int)std::lround(c.jitter_deg));
  return out;
}

TEST(ApproachGeometry, Radii)
{
  std::vector<double> r = radii();
  ASSERT_EQ(r.size(), 4u);
  EXPECT_DOUBLE_EQ(r.front(), 0.8);
  EXPECT_NEAR(r[1], 1.05, 1e-9);
  EXPECT_NEAR(r[2], 1.3, 1e-9);
  EXPECT_DOUBLE_EQ(r.back(), 1.5);
}

TEST(ApproachGeometry, FarCameraUsesLineOfSight)
{
  Vector2d object(1.0, 2.0), camera(1.0, 5.0);  // 3 m away, line of sight +90 deg
  auto cs = candidates(object, camera, false);
  EXPECT_EQ(jitters(cs), (std::set<int>{ -40, -20, 0, 20, 40 }));
  // Rank order: straight-on first, then +-20, then +-40
  EXPECT_EQ(cs.front().rank, 0);
  EXPECT_DOUBLE_EQ(cs.front().jitter_deg, 0.0);
  EXPECT_EQ(cs.back().rank, 2);
  for (const auto& c : cs) {
    EXPECT_NEAR((c.point - object).norm(), c.radius, 1e-9);
    EXPECT_NEAR(angleFrom(object, c.point, M_PI / 2), c.jitter_deg, 1e-6);
    EXPECT_EQ(c.rank, (int)std::lround(std::fabs(c.jitter_deg) / 20.0));
    EXPECT_GE(c.radius, 0.8 - 1e-9);
    EXPECT_LE(c.radius, 1.5 + 1e-9);
  }
  // Straight-on: all four radii are >= 0.5 m closer than 3 m
  int straight = 0;
  for (const auto& c : cs) straight += c.rank == 0;
  EXPECT_EQ(straight, 4);
}

TEST(ApproachGeometry, NewViewRule)
{
  // Camera 1.8 m away: straight-on and +-20 need r <= 1.8 - 0.5 - 0.45 = 0.85 (only 0.8);
  // +-40 is around enough (>= 30 + 10 deg) at any radius
  Vector2d object(0.0, 0.0), camera(1.8, 0.0);
  for (const auto& c : candidates(object, camera, false)) {
    if (std::fabs(c.jitter_deg) < 30.0) {
      EXPECT_DOUBLE_EQ(c.radius, 0.8);
    }
  }
  int side = 0;
  for (const auto& c : candidates(object, camera, false)) side += c.rank == 2;
  EXPECT_EQ(side, 8);
}

TEST(ApproachGeometry, NearCameraSideOnly)
{
  // Within 1.5 m: no straight-on bearing; +-20 cannot be 0.5 m closer either, so +-40 remain
  Vector2d object(0.0, 0.0), camera(-1.2, 0.0);
  auto cs = candidates(object, camera, false);
  EXPECT_EQ(jitters(cs), (std::set<int>{ -40, 40 }));
  for (const auto& c : cs) EXPECT_NEAR(std::fabs(angleFrom(object, c.point, M_PI)), 40.0, 1e-6);
}

TEST(ApproachGeometry, NearCameraArrivalSlack)
{
  // 1.45 m: +-20 at 0.8 m is only 0.65 m closer, under 0.5 m + the 0.45 m arrival slack
  auto cs = candidates(Vector2d(0, 0), Vector2d(0.0, 1.45), false);
  EXPECT_EQ(jitters(cs), (std::set<int>{ -40, 40 }));
}

TEST(ApproachGeometry, ArrivalStillNew)
{
  // Candidates that rely on being closer stay 0.5 m closer from anywhere within the 0.45 m
  // soft-reach distance of them
  Vector2d object(0.0, 0.0), camera(3.0, 1.0);
  double range = camera.norm();
  for (const auto& c : candidates(object, camera, false)) {
    for (int k = 0; k < 16; ++k) {
      Vector2d p = c.point + 0.45 * Vector2d(std::cos(k * M_PI / 8), std::sin(k * M_PI / 8));
      bool closer = p.norm() <= range - 0.5;
      // (side candidates rely on the angle; the gate judges an arrival without thresholds)
      if (std::fabs(c.jitter_deg) < 40.0) {
        EXPECT_TRUE(closer) << c.jitter_deg << " " << c.radius;
      }
    }
  }
}

TEST(ApproachGeometry, MinRangeOfAllJudgedViews)
{
  // The latest camera is 3 m away, but an earlier judged view was 1.6 m away: straight-on
  // would need r <= 0.65, so only the side bearings remain
  auto cs = candidates(Vector2d(0, 0), Vector2d(3.0, 0.0), false, 1.6);
  EXPECT_EQ(jitters(cs), (std::set<int>{ -40, 40 }));
  // min_range <= 0: the camera's distance
  EXPECT_EQ(jitters(candidates(Vector2d(0, 0), Vector2d(3.0, 0.0), false, 0.0)),
      (std::set<int>{ -40, -20, 0, 20, 40 }));
}

TEST(ApproachGeometry, StopBackstopSideOnly)
{
  auto cs = candidates(Vector2d(0, 0), Vector2d(4.0, 0.0), true);
  EXPECT_EQ(jitters(cs), (std::set<int>{ -40, 40 }));
  EXPECT_EQ(cs.size(), 8u);
}

TEST(ApproachGeometry, Deterministic)
{
  auto a = candidates(Vector2d(0.3, -1.0), Vector2d(2.0, 2.0), false);
  auto b = candidates(Vector2d(0.3, -1.0), Vector2d(2.0, 2.0), false);
  ASSERT_EQ(a.size(), b.size());
  for (size_t i = 0; i < a.size(); ++i) EXPECT_EQ(a[i].point, b[i].point);
}

TEST(ApproachGeometry, MaxPath)
{
  EXPECT_DOUBLE_EQ(maxPath(40), 5.0);
  EXPECT_DOUBLE_EQ(maxPath(-3), 0.0);
}
