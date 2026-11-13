# Running ApexNav with the perception gate

When the ApexNav planner selects a target object, and again before it stops at one, the
gate sends the detection to `perception.Verifier` (SAM3 -> SAM3D -> view pick -> SAM3) in the
3DAwareVerification container. A rejected object counts as a non-detection: the planner drops it
and keeps exploring. The gate is off by default, and with it off ApexNav behaves as before.
Design: `3DAwareVerification/docs/apexnav_integration_plan.md`.

Two containers on one GPU, talking over HTTP:

| container | image | runs |
|---|---|---|
| `3dawareverification-dev-1` | `3d-aware:latest` | `perception.server` on port 8770 (~24 GB idle, ~30 GB peak) |
| `apexnav` | `apexnav:jazzy` | VLM servers, Habitat, ROS 2 planner (~10-12 GB) |

Both run under docker compose, so all container flags live in the compose files:
`~/codes/3DAwareVerification/compose.yaml` (service `dev`) and `docker/docker-compose.yml`
(service `apexnav`: GPUs, shm, data mount, thread caps, and the Verifier's network). You need
2-3 terminals on the host, and you work inside the containers with `docker compose exec -it`.

## 0. Pick a GPU (every time)

The server is shared. Check GPU and memory before starting:

```bash
nvidia-smi
uptime
free -g
```

Pick a GPU with at least 45 GB free and low utilization (one run peaks at about 41 GB). Use it
for everything below; `GPU=0` is the example. Don't start if host RAM is nearly full: the
memory-pressure killer (oomd) kills whole containers.

One-time setup, if `docker images` doesn't list `apexnav:jazzy`, load it rather than letting
compose rebuild it: `docker load -i /ICONAS/users/marfred/ApexNav/apexnav-jazzy-image-2026-10-17.tar`.

## 1. Terminal 1: start the Verifier server

```bash
cd ~/codes/3DAwareVerification
docker compose up -d dev          # no-op if it's already running
docker compose exec -it dev bash
# inside the container (repo at /workspace):
sam3d-env core python -m perception.server --host 0.0.0.0 --port 8770 --gpu 0 \
    --selector vqf --threshold 0.15 --log-dir outputs/apexnav_gate/server
```

Keep it in the foreground; it is ready when it prints `perception server on http://0.0.0.0:8770 {...settings}`.

Options (fixed for the whole server run; restart it to compare):
- `--selector vqf|top2gap`: which view-selection rule to use.
- `--no-ensemble`: turn the SAM3D ensemble filter off.
- `--threshold 0.15`: maximum ensemble disagreement for a reconstruction to count as trusted.

Check it from another host terminal:

```bash
cd ~/codes/3DAwareVerification
docker compose exec -it dev sam3d-env core python -m perception.server ping http://localhost:8770
```

The server exits with code 1 if a model worker dies. Every request is logged in
`outputs/apexnav_gate/server/requests.jsonl`.

## 2. Terminal 2: start the ApexNav container

Start the Verifier's container (step 1) first: the apexnav service joins its network
(`3dawareverification_default`), and `up` fails if that network doesn't exist.

```bash
cd ~/codes/ApexNav/docker
docker compose up -d              # creates the container, or starts the stopped one
docker compose exec -it apexnav bash
```

Inside the container:

```bash
source /opt/apexnav/apexnav-env.sh && use_core
nvidia-smi -L            # GPU indices should match the host's
./scripts/build_ws.sh    # first time, and after any C++/msg change
python -c "from perception_gate.client import ping; print(ping('http://3dawareverification-dev-1:8770'))"
```

The `ping` should print the server settings. To use a different data folder, set
`APEXNAV_DATA=/path` before `docker compose up` (default `/ICONAS/users/marfred/apexnav_data`).
After editing the compose file, run `docker compose up -d --force-recreate` to apply it.

## 3. Still in terminal 2: start the VLM servers and planner

```bash
VLM_GPU=0 VERIFICATION_GATE=true ./scripts/start_session.sh
tmux attach -t apexnav
```

This tmux session has three windows (switch with `Ctrl-b` then the window number; detach with
`Ctrl-b d`):
- `0 vlm`: grounding_dino, blip2itm, sam, yolov7. Wait 1-2 min until all four have loaded.
- `1 ros`: Foxglove (you can `Ctrl-c` it to save CPU) and the planner.
- `2 eval`: a shell ready to run episodes, with `CUDA_VISIBLE_DEVICES` set.

ollama isn't needed: the cached LLM answers in `llm/answers` cover the HM3D targets.

## 4a. Run one episode by hand (eval window)

```bash
python habitat_evaluation.py --dataset hm3dv2 test_epi_num=529 \
    verification.enabled=true video_output_path=videos/mine/epi529
```

`verification.enabled` must match the planner's `verification_gate` (set by
`VERIFICATION_GATE` in step 3). Otherwise the run aborts at startup instead of hanging. To run
the baseline by hand, restart the planner in the `ros` window with `verification_gate:=false`
and pass `verification.enabled=false`.

## 4b. Run a batch, gate off vs on (eval window)

`scripts/run_gate_e2e.sh` restarts the planner with the right flag for each mode, then runs the
listed episodes one after another:

```bash
# the known false-positive / goal-class records (run_index from
# 3DAwareVerification/data/apexnav/records/epi<N>_*/meta.json; true FPs are in verified.txt)
SIM_GPU=0 ./scripts/run_gate_e2e.sh baseline 213 229 529 508 511
SIM_GPU=0 ./scripts/run_gate_e2e.sh gate     213 229 529 508 511

# ordinary val episodes, for the success-rate cost
SIM_GPU=0 ./scripts/run_gate_e2e.sh baseline 0 1 2 3 4 5 6 7 8 9
SIM_GPU=0 ./scripts/run_gate_e2e.sh gate     0 1 2 3 4 5 6 7 8 9
```

To repeat runs, set `OUT_ROOT=videos/gate_e2e_seed2` (any new folder) so earlier results aren't
overwritten. In the original mode runs are not deterministic: under host load, two runs of the
same episode can diverge before the first verification.

**Deterministic runs (`sync_step`).** `SYNC_STEP=true ./scripts/run_gate_e2e.sh ...` launches the
planner with `sync_step:=true` and runs `habitat_evaluation.py sync_step=true` (both must
agree; the evaluator checks). Each executed step goes to the planner once, as one
`plan_env/StepObservation` (step id, depth, camera pose, odom, ITM, object clouds, gate
reopens; reliable QoS), integrated in a fixed order with a synchronous ESDF update; the FSM
plans only after `/habitat/action_finish_step` k and step k is integrated, map and FSM
callbacks share one MutuallyExclusive group, and actions come back as `StepAction` (step,
fusion snapshot seq, FSM verdict). TSP waits without the 5 s cutoff, A* time budgets become
node-expansion budgets (`ASTAR_EXPANSIONS_PER_S` in `algorithm.launch.py`, measured on the host
in wall-clock mode), the planner is restarted per episode, and a gate error or timeout aborts
instead of failing open. Two sync runs of the same episodes give identical steps.jsonl
(`videos/sync_runs/compare_runs.py`). Not changed: the gate rules (section 7), detectors and
planner decision rules. Sync and original mode are different map dynamics (no repeated
integration), so compare strategies only within one mode.

## 5. Terminal 3 (optional): GPU guard

On the host:

```bash
cd ~/codes/ApexNav && ./scripts/gpu_guard.sh 0
```

It prints memory and load every 5 s. If the GPU passes 44 GB or another user's process
appears on it, it touches `videos/gate_e2e/ABORT` and kills the current episode; the batch
script then stops. Delete `videos/gate_e2e/ABORT` before the next batch.

## 6. Results

Per episode, in `videos/gate_e2e/<baseline|gate>/epi<N>/` (or your `video_output_path`):
- `eval.log`: Habitat result (success, SPL, steps).
- `records/`: the episode recording (frames, object fusion state, verification per step).
- `verification/decisions.jsonl`: one line per gate decision. It includes the cluster,
  the evidence view and its match distance, the camera elevation and intrinsics, the server
  settings, the verdict, the latency, any error, and the ground truth (whether the object
  was really a goal instance).
- `verification/epi<N>_<label>/c<id>_s<step>/`: input image, mask, chosen SAM3D view,
  decision.json.

Quick summary of verdicts:

```bash
cat videos/gate_e2e/gate/epi*/verification/decisions.jsonl | python -c "
import sys, json
for line in sys.stdin:
    d = json.loads(line); print(d.get('request_id'), d.get('target'), d.get('verdict', {}).get('accepted'), d.get('latency_s'))"
```

## 7. Gate settings (`config/habitat_eval_*.yaml`, `verification:`)

Override any of them on the command line, e.g. `verification.max_checks=1`.

| key | default | meaning |
|---|---|---|
| `enabled` | false | must match the planner's `verification_gate` |
| `url` | `http://3dawareverification-dev-1:8770` | Verifier server |
| `timeout_s` | 120 | per-request HTTP timeout |
| `on_error` / `on_no_evidence` | accept | fail open: errors and missing views keep the detection, and are logged |
| `on_no_evidence: defer` | | no stored view: the planner goes on to the cluster and the stop re-verifies it by position; still no view at the stop rejects |
| `on_view_reject` | reject | the raw frame accepts but the SAM3D view says no: `reject`, or `defer` (the planner may approach the cluster but not stop at it; the stop backstop verifies it again, and a view "no" there rejects). Pair `defer` with server flags `--judge-input --view-when unsure` |
| `max_consecutive_errors` | 3 | abort the run instead of silently running ungated |
| `max_checks` | 3 | checks per cluster, including re-checks |
| `recheck_cooldown_steps` | 25 | steps before a rejected cluster can be re-checked |
| `recheck_min_distance_m` / `recheck_min_angle_deg` | 1.0 / 45 | the re-check needs a view this far from / this different to earlier views |
| `match_distance_m` | 0.75 | how close a stored view must be to the cluster / stop point |
| `approach` | false | server `--selector d1` (three-way verdicts): "insufficient" sends the agent closer for a new view (state 3). Must match the planner's `verification_approach` and needs `on_error: reject`; `scripts/run_strategy.sh d1` sets all three |
| `approach_max_per_cluster` / `approach_max_steps` / `approach_episode_steps` | 2 / 40 / 120 | approach budgets (also capped by the steps left in the episode); a failed approach (no object path, target lost, budget used up, no view on arrival) or "insufficient" on arrival rejects |
| `approach_min_closer_m` / `approach_min_angle_deg` | 0.5 / 30 | a view on the way is judged at once when it is this much closer than, or this far around the object from, every view already judged (insufficient: keep approaching); on arrival the newest view since the approach began is judged without these thresholds |

Approach (d1): the insufficient cluster becomes the planner's forced target and ApexNav's
own object navigation (approach point, per-step replanning) drives toward it. A new view
on the way is judged at once; at the object the agent turns to face it and the fresh view
is judged; "insufficient" there (or at any stop) rejects: the agent never stops on it.

Give-up stops (no frontier left, stuck, step limit) are not verified, by design.

## 8. Clean up

```bash
# terminal 2: leave tmux (Ctrl-b d), exit the shell, then
cd ~/codes/ApexNav/docker && docker compose down
# terminal 1: Ctrl-c the server (leave the dev container running for other work)
nvidia-smi    # check the GPU is free again
```

## Troubleshooting

- **`network 3dawareverification_default not found`** on `docker compose up`: start the Verifier's
  container first (`docker compose up -d dev` in ~/codes/3DAwareVerification).
- **`ping` fails from apexnav**: check that the server is running in terminal 1. If apexnav was
  created before the network was added to its compose file, run `docker compose up -d --force-recreate`.
- **`RuntimeError: verification.enabled is ... here but ... in the planner`**: the planner's
  `verification_gate` and `verification.enabled` disagree. Restart one of them.
- **A container disappears (exit 137)**: oomd killed it because host RAM was full. Check
  `free -g` and wait for a quieter time, then `docker compose up -d` again.
- **Workspace not built / missing `local_setup.bash`**: run `./scripts/build_ws.sh`.
