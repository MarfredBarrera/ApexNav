# Last-mile navigation benchmark (2026-11-11)

Offline benchmark for ApexNav ObjectNav episodes. **Start** = the agent pose at the moment the perception gate first
ACCEPTS the target (first `verification/decisions.jsonl` row with `state == 1`; pose = that step's entry in
`records/*/steps.jsonl`). **Task** = drive to the accepted object and STOP within 1 m (horizontal) of it.
Used to compare navigators (A*, DD-PPO PointNav, MPPI, ...) on exactly the part of the episode the gate hands over.

Code: `~/codes/ApexNav/lastmile/` (`bench.py` harness, `oracle.py`, `build.py`, `gt.py`). Data is built in
`lastmile/bench_data/` (visible inside the container) and mirrored here by `python -m lastmile.build --publish`.

## Files
- `episodes.tsv`: one row per episode: set, ep (index into the habitat episode iterator = `epiN` dir), tag, scene, target,
  cat, accept_step, remaining_steps (500 - accept_step), start_x/y/z (habitat world), start_yaw (rad, ccw about +y,
  0 looks along -z), start_pitch (rad), accept_dist (horizontal m to nearest GT point of the accepted object),
  centroid_dist, goal_source (`approach_cloud` = the gate's perceived object cloud, else `cluster_centroid` = a single
  point), gt_instance (HM3D semantic instance id:name used for scoring), gt_is_goal, gt_vs_perceived_m, native_bench
  (bench success of the pipeline run), native_eye (eye label of the pipeline run: S/FS/FR/explore), baseline_eye (eye label
  of the baseline run), cls (postaccept class), pose_replay_err_m, navmesh_snap_m, note, plus the oracle columns.
- `episodes/<set>_<ep>.npz`: `position`, `yaw`, `pitch`, `goal_cloud` (what the agent knows, world frame),
  `gt_points` (scoring only), `gt_instance`.
- `results/<nav>_<budget>.tsv`: per-episode outcomes of a navigator run.

## Tags
- FAIL (23): `lastmile.tsv` classes A_stuck / A_stepout / A_wrongside: the gate accepted an eye-verified real object, then native ApexNav failed.
- SUCCESS (30): gate accepted a real goal object and the pipeline succeeded (eye S, bench success); chosen spread over accept
  distance (includes 8 episodes with accept > 3 m). A new navigator must not lose these.
- benchonly_1m (8), budget (2): the other `lastmile.tsv` rows (S under the 1 m rule / accept at step 496-499).

- SUCCESS_capstop (7): gate accepted and ApexNav was forced to STOP (step cap / stucking) within 1 m: extra SUCCESS cases (test2 173, heldout 70, 597, tuning 179, 761, 809, 459). Note test2 173: the chosen GT flower instance is 6.6 m from the perceived centroid, GT assignment unverified.
- test2 701 is tagged SUCCESS (corrected 1 m scorer: stopped 0.5 m from the bed); original tag in `orig_tag`.
- fresh400 (set `fresh400`, **TEST ONLY**: tune navigators on the other sets, evaluate once here): FAIL = 16 post-accept failures whose accepted cluster overlaps a goal
  (36, 172, 230, 231, 235, 249, 540, 561, 661, 665, 669, 700, 748, 867, 959, 973; 101/184/254 are non-goal clusters, 435/719 are S), SUCCESS = 20 post-accept pipeline successes spread over accept distance.
- offfloor: all targets on another floor (none found: floor_ok = 1 for every episode); would carry `failure_class = exploration/other` (not navigator-recoverable, still in overall SR tables).

## Fine actions (`--actions fine [--small-m 0.10|0.125]`)
Adds ids FORWARD_SMALL=4, LEFT_15=5, RIGHT_15=6 next to 0-3 (default `std` unchanged). Results include per-type counts (`n_fwd`, `n_left`, `n_right`, `n_fwd_small`, `n_left15`, `n_right15`).
Habitat's no-sliding step is partial near walls (moves up to the obstacle), so blocked-forward counts in `collisions` include partial moves.

## Start pose provenance
steps.jsonl logs odometry in the planner frame relative to the episode start (x forward, y left, yaw ccw). We map it to the
world frame with the dataset start pose and also replay the logged actions in habitat-sim (navmesh recomputed exactly as
habitat-lab does for ApexNav: radius 0.18, height 0.88, max_climb 0.2, slope 45, no sliding); the two agree to 0.000 m on
all episodes (`pose_replay_err_m`). The first attempt without the recomputed navmesh diverged on 11 episodes. The accept
pose is the observation the verifier saw. RGB was not re-rendered (GPU 3 had < 3 GB free).

## Scoring
success = navigator issues STOP while the horizontal distance to the nearest point of the accepted (GT) object is <= 1.0 m
and the nearest such point is within 2 m vertically (same storey). Also logged: within1m (at any time), final_dist,
min_dist, actions, collisions (blocked FORWARD). Budgets: `remaining` (rest of the 500-step episode) and `150`.
GT object = the semantic instance (goal ids or same category) nearest the perceived goal estimate on the agent's floor;
overrides: heldout 601 -> instance 616 (sitting bench), heldout 322 / test2 312 (loveseat, no instance) -> GT is the dilated
perceived cloud (`gt_proxy`). Perceived goal for direct accepts is only the fused cluster centroid (no cloud was saved);
approach-accepted episodes carry the gate's object cloud.

## Run a navigator
```
docker exec apexnav bash -c "source /opt/apexnav/apexnav-env.sh; use_core; cd /workspace/ApexNav; \
  OMP_NUM_THREADS=2 PYTHONPATH=. python -m lastmile.bench --nav <name> --budget remaining|150 [--subset FAIL,SUCCESS | test2:382] [--render] [--gpu 3]"
python -m lastmile.build --publish   # (in the container) mirror results to this directory
```
`--nav NAME` imports `lastmile/nav_NAME.py` and calls `make()`. Without `--render` obs rgb/depth are None (no GPU memory).
With `--render`: 640x480 RGB + raw-metre depth, hfov 79, camera 0.88 m. Actions: STOP=0, FORWARD=1 (0.25 m), LEFT=2 (30 deg),
RIGHT=3 (30 deg), no sliding.

## Add a navigator
Create `lastmile/nav_<name>.py` with `class X(bench.Navigator)` (`reset(obs, start_info)`, `act(obs) -> int`) and
`def make(): return X()`. obs: `rgb`, `depth`, `pose {position, yaw, pitch}`, `goal_cloud` (world frame, perceived). `start_info`:
scene, ep, camera, goal_cloud, goal_estimate, budget_steps, start pose. If the class defines `set_sim(sim)` it receives
the habitat-sim (for navmesh-based planners; do not read GT). Never use `episodes/*.npz:gt_points`.
