#ifndef _APPROACH_GEOMETRY_H_
#define _APPROACH_GEOMETRY_H_

// Perception gate approach (VerificationResult state 3): candidate viewpoints around the
// object point. Pure geometry (no map), so it is unit-tested on its own
// (test/test_approach_geometry.cpp); ExplorationManager::searchApproachPath checks each
// candidate against the map.

#include <Eigen/Eigen>
#include <algorithm>
#include <cmath>
#include <vector>

namespace apexnav_planner {
namespace approach {

constexpr double MIN_RADIUS = 0.8;     // m from the object point
constexpr double MAX_RADIUS = 1.5;
constexpr double RADIUS_STEP = 0.25;
constexpr double JITTER_STEP_DEG = 20.0;  // bearings: line of sight + 0, +-20, +-40 deg
constexpr int MAX_JITTER_RANK = 2;
constexpr double NEAR_DISTANCE = 1.5;  // camera already this close: side bearings only
constexpr double RANK_COST = 1.0;      // m of path one jitter rank is worth
// A viewpoint must give the gate a new view: this much closer to the object than the
// insufficient view, or this far around it (perception_gate approach_min_closer_m / _angle_deg)
constexpr double MIN_CLOSER = 0.5;
constexpr double MIN_ANGLE_DEG = 30.0;
// The planner counts a viewpoint as reached this far off (FSMConstants::
// VIEWPOINT_SOFT_REACH_DISTANCE), so a candidate keeps that much (and some angle) in reserve
constexpr double ARRIVAL_SLACK = 0.45;
constexpr double ANGLE_SLACK_DEG = 10.0;

struct Candidate {
  Eigen::Vector2d point;
  double jitter_deg;  // signed, from the line of sight object -> camera
  int rank;           // |jitter| / 20 deg
  double radius;
};

// Radii MIN_RADIUS, +RADIUS_STEP, ... up to MAX_RADIUS (inclusive: 0.8, 1.05, 1.3, 1.5)
inline std::vector<double> radii()
{
  std::vector<double> out;
  for (double r = MIN_RADIUS; r < MAX_RADIUS - 1e-6; r += RADIUS_STEP) out.push_back(r);
  out.push_back(MAX_RADIUS);
  return out;
}

// Candidates object + r (cos b, sin b), b = bearing(object -> camera) + jitter, ordered by
// rank, then jitter (+ before -), then radius. The straight-on bearing (rank 0) is dropped
// when the camera is already within NEAR_DISTANCE of the object (the view was likely cut
// off or occluded, not far); side_only (stop backstop) keeps only the +-40 deg bearings.
// Candidates that would not give a new view even ARRIVAL_SLACK off are dropped: neither
// MIN_CLOSER + ARRIVAL_SLACK closer than min_range (the closest view the gate judged; <= 0:
// the camera's distance) nor MIN_ANGLE_DEG + ANGLE_SLACK_DEG around the object.
inline std::vector<Candidate> candidates(
    const Eigen::Vector2d& object, const Eigen::Vector2d& camera, bool side_only, double min_range = 0.0)
{
  std::vector<Candidate> out;
  const double range = (camera - object).norm();
  const double closest = min_range > 0.0 ? std::min(min_range, range) : range;
  const double sight = range > 1e-6 ? std::atan2(camera(1) - object(1), camera(0) - object(0)) : 0.0;
  const bool near = range <= NEAR_DISTANCE;
  for (int rank = 0; rank <= MAX_JITTER_RANK; ++rank) {
    if (rank == 0 && (near || side_only))
      continue;
    if (side_only && rank != MAX_JITTER_RANK)
      continue;
    for (int sign : { +1, -1 }) {
      if (rank == 0 && sign < 0)
        continue;
      double jitter = sign * rank * JITTER_STEP_DEG;
      double bearing = sight + jitter * M_PI / 180.0;
      for (double r : radii()) {
        bool closer = r <= closest - MIN_CLOSER - ARRIVAL_SLACK + 1e-9;
        bool around = std::fabs(jitter) >= MIN_ANGLE_DEG + ANGLE_SLACK_DEG - 1e-9;
        if (!closer && !around)
          continue;
        out.push_back({ object + r * Eigen::Vector2d(std::cos(bearing), std::sin(bearing)), jitter, rank, r });
      }
    }
  }
  return out;
}

// Longest path an approach of max_steps actions may plan: half of them forward (0.25 m
// each), the rest for turns and the final turn to the object
inline double maxPath(int max_steps)
{
  return 0.5 * std::max(max_steps, 0) * 0.25;
}

}  // namespace approach
}  // namespace apexnav_planner

#endif
