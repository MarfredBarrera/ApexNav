# ApexNav in Docker — ROS 2 Jazzy, headless, Foxglove

Setup for running ApexNav on a headless GPU server: no ROS on the host, no display,
no RViz. Everything lives in one container; visualization goes through
`foxglove_bridge` over a websocket instead of RViz2.

Built on the `ros2-jazzy` branch. `apexnav_environment.yaml` is **not** used — see
[Why not the conda env](#why-not-the-conda-env) for the reason.

---

## Quick start

All commands run from the repo root unless noted. `DC` below is shorthand for
`docker compose -f docker/docker-compose.yml`.

```bash
cd ~/codes/ApexNav

# 1. once: data tree (~7 GB -> ~/nas/apexnav_data). Idempotent; skips what exists.
./scripts/setup_data.sh

# 2. once: build the image (~25 min; habitat-sim compiles from source)
docker compose -f docker/docker-compose.yml build

# 3. start the container
docker compose -f docker/docker-compose.yml up -d

# 4. after any C++ change: build the ROS workspace (~2.5 min)
docker compose -f docker/docker-compose.yml exec apexnav ./scripts/build_ws.sh

# 5. sanity check - ends with "All checks passed."
docker compose -f docker/docker-compose.yml exec apexnav ./scripts/verify_install.sh

# 6. launch the pipeline: 4 VLM servers + foxglove_bridge + planner + eval shell
#    add -e VLM_GPU=<n> to put it on a GPU other than 0 - see "Which GPU each process uses"
docker compose -f docker/docker-compose.yml exec apexnav ./scripts/start_session.sh

# 7. attach (Ctrl-b d to detach and leave everything running)
docker compose -f docker/docker-compose.yml exec apexnav tmux attach -t apexnav
```

Give step 6 about 90 seconds before evaluating - the VLM servers are loading weights.
Readiness check (want `4`):

```bash
docker compose -f docker/docker-compose.yml exec apexnav bash -c 'ss -ltn | grep -cE ":1218[1-4]"'
```

tmux windows: `Ctrl-b 0` vlm (4 panes) · `Ctrl-b 1` ros · `Ctrl-b 2` eval.

### Run an evaluation

In the `eval` window (or any `use_core` shell):

```bash
python habitat_evaluation.py --dataset hm3dv2                  # full benchmark, resumable
python habitat_evaluation.py --dataset hm3dv1
python habitat_evaluation.py --dataset hm3dv2 test_epi_num=10  # single episode
python habitat_evaluation.py --dataset hm3dv2 need_video=true  # videos sorted by outcome
```

Results land in `videos/test_hm3dv2_val/`: `record.txt` per episode, `continue.txt` for
running totals. A full run resumes from `continue.txt` after Ctrl-C - delete that file to
start over. `--dataset mp3d` will not run: the MP3D scene meshes need their own
Matterport application.

### Episode records and failure forensics

Every episode appends one structured line to `videos/test_hm3dv2_val/episodes.jsonl` -
per-episode success, SPL, result category, and the planner's fusion configuration. This is
the only place per-episode metrics exist; `record.txt` stores running averages.

Episodes whose result category is listed in `record.flush_on` (default `["false positive",
"last mile nav failure"]`)
additionally get a full artifact directory, enough to reconstruct what the agent saw before
it stopped at the wrong object. The category keeps its space as a *value* - that is what
`flush_on` matches and what `episodes.jsonl` stores - but on disk it is written
`false_positive` / `last_mile_nav_failure`, so record paths need no shell quoting
(`params.RESULT_DIRNAMES`):

```
videos/test_hm3dv2_val/records/false_positive/epi<N>_<scene>_<episode_id>_<label>/
    meta.json        # episode identity, LLM answer + threshold, detector and fusion
                     # config, final metrics, replay commands
    steps.jsonl      # per step: action, pose, camera pitch, ITM score,
                     # distance to goal, per-detection confidences, the FSM
                     # verdict, fused per-cluster confidence, and semantic
                     # instance IDs/pixel counts under each detector mask
    trajectory.npy   # (T,3) float32 pose array
    rgb/             # raw observations; 000000.jpg is the reset frame,
                     # 00000N.jpg is step N in steps.jsonl
    rgb_annot/       # the same frames with detector boxes and masks drawn
    masks/           # detector masks, {step}_{detection}.png, lossless 0/1 uint8
```

`trajectory.npy` follows the same indexing: row 0 is the pose after `env.reset()`, row N
the pose after step N, so it has one more row than `steps.jsonl` has lines.

To create a forensic GIF while replaying, enable the visualization override:

```
python -m basic_utils.record_episode.replay_episode <record_dir> \
    visualization.gif.save=true
```

The GIF uses replay-generated RGBs, overlays the lossless detector masks stored
with the record, labels target and non-target candidates, and marks the first
FSM `SEARCH_OBJECT` frame as `TARGET COMMIT`. If available, the earlier fused
confidence-gate crossing is marked separately.

At a target commit, the recorder compares Habitat semantic instance IDs under
the selected detector mask with `episode.goals[*].object_id`. This exact
instance evidence, rather than a map-centroid distance, assigns `false positive`
(a different instance) or `last mile nav failure` (the goal instance was selected
but success radius was not reached). Offline replay repeats and prints this check
for old records whose masks were saved.

To render MP4 instead, disable GIF output and enable MP4 output:

```
python -m basic_utils.record_episode.replay_episode <record_dir> \
    visualization.gif.save=false visualization.mp4.save=true
```

MP4 uses baseline H.264 (`avc1`)/yuv420p for compatibility with Electron-based
VS Code and web players. Its speed comes from `visualization.mp4.fps`; it is
written as `episode.mp4` by default.

For false-positive and last-mile-navigation-failure records, replay also saves the annotated target-decision
frame separately as `decision_frame.jpg` beside the GIF and RGB directory. Set
`visualization.decision_frame.only_false_positive=false` to save it for every
record with a decision signal.

Each replay also writes `committed_target_confidence.png`: the fused-confidence
history of the specific cluster chosen at the first target commit. The green
line marks that cluster becoming confident and the magenta line marks the
planner commit.

To replay every false-positive record in one batch, pass the category directory:

```
python -m basic_utils.record_episode.replay_episode \
    videos/test_hm3dv2_val/records/false_positive
```

By default, an empty `visualization.gif.target_episodes` selects every episode.
Filter to specific run indices, episode IDs, or record directory names with:

```
python -m basic_utils.record_episode.replay_episode \
    videos/test_hm3dv2_val/records/false_positive \
    visualization.gif.target_episodes=[12,18]
```

#### Pinpointing where the algorithm committed

Two per-step fields answer "when did it decide?", and they measure different things:

- `detections[].score` is the **raw per-frame detector confidence** (YOLO or GroundingDINO).
  It is *not* what the planner commits on.
- `object_fusion` is the **fused confidence** accumulated in the planner's object map -
  one entry per cluster, with `confidence` (target class), `observation_num`, and
  `is_confident`, alongside the `min_confidence` / `min_observation_num` gate in force.
  This is what [`isConfidenceObject`](../src/planner/plan_env/src/object_map2d.cpp) reads.
  It arrives on `/object/fusion_state`; `update_seq` lets you tell whether a step saw a
  stale snapshot, since the planner's cycle and the simulator's steps are not 1:1.
- `final_state` / `expl_result` are the FSM's verdict for that step
  (`params.FINAL_RESULT` / `params.EXPL_RESULT`).

`summarize_run.py` reports both signals per episode:

```
gate crossed at step 17: cluster 1 confidence 0.815 >= 0.650 after 2 observations (min 2) at (-0.02, 1.49)
FSM committed at step 34 via SEARCH_BEST_OBJECT
```

They can disagree, and the disagreement is the point. A gate crossing with no FSM
commitment means the object passed the confidence test but no path to it was found. An
FSM commitment via `SEARCH_SUSPICIOUS_OBJECT` or `SEARCH_EXTREME` means the run committed
through a low-confidence fallback branch and never passed the gate at all - a different
failure mode from a confidently-wrong detection, and the summarizer flags it.

Everything is buffered in RAM and dropped for episodes that do not match, so successful
episodes cost only the summary line. A worst-case 500-step episode is about 52 MB on disk
and roughly the same peak in RAM, so `record.flush_on=all` costs tens of GB over a full
HM3D-v2 val sweep. `record.enabled=false` turns it off; `record.capture_annotated_rgb=false`
roughly halves the size.

Value maps are deliberately not captured; use `need_video=true` if you want the planner's
top-down view.

To review a finished run:

```bash
python -m basic_utils.failure_check.summarize_run videos/test_hm3dv2_val
```

That prints the result breakdown and lists each false positive with two ways to reproduce
it: `test_epi_num=<N>` for a fresh live run, or

```bash
python -m basic_utils.record_episode.replay_episode '<record_dir>'
```

to re-drive the recorded action sequence through Habitat alone - no planner and no VLM
servers needed. The offline replay is exact; a fresh live run may diverge, because the
planner's A* search is bounded by wall-clock time and the ROS loop cadence is not
deterministic.

#### What is stored vs regenerated

Habitat is deterministic: replaying the recorded actions reproduces the poses, and
therefore the RGB and depth observations, bit for bit (verified across processes). So the
rule is **store what cannot be regenerated, regenerate everything else**:

| data | where it comes from | cost |
|---|---|---|
| detector masks | **stored** - reproducing them means re-running GroundingDINO / YOLO / MobileSAM, which is not bit-reproducible across GPUs | ~4 KB each, ~1.6 MB per 500-step episode |
| depth | **regenerated** by replay | would be ~590 MB per 500-step episode if stored |
| world-frame point clouds | **regenerated** by replay | ~0.9 MB per frame at pixel stride 2 |

Replay settings live in [`config/replay.yaml`](../config/replay.yaml) and take hydra-style
overrides, the same convention `habitat_evaluation.py` uses. By default a replay
regenerates the depth images and nothing else - the projection is opt-in:

```bash
# depth only (default)
python -m basic_utils.record_episode.replay_episode '<record_dir>'

# depth in metres, plus world-frame clouds and RGB, every 5th step
python -m basic_utils.record_episode.replay_episode '<record_dir>' \
    depth.metric=true clouds.save=true rgb.save=true step_stride=5

# somewhere other than replays/<record name>/
python -m basic_utils.record_episode.replay_episode '<record_dir>' output_dir=/tmp/myreplay
```

Depth is written as float32 `.npy`, normalized to `[0,1]` exactly as habitat produced it
unless `depth.metric=true`. Clouds are `(N,3)` float32 in the planner's world frame
(x = -gps[2], y = -gps[0], z up, floor near 0).

The unprojection is **pitch-aware**, unlike the planner's own
`get_object_point_cloud`, which uses yaw only - which is why `map_ros.cpp` discards frames
where the camera is tilted. Since `steps.jsonl` records `camera_pitch`, look-down frames
are usable here. Measured against the floor plane, tilted frames land at +0.002 m +/- 0.008;
ignoring pitch puts them off by ~0.39 m.

To pull one object out, pair its stored mask with the regenerated depth for the same step -
`steps.jsonl` gives the mask filename per detection.

### Single-frame ablation

`exploration.launch.py` takes a `multiview_fusion` argument that switches the planner
between ApexNav's multi-view confidence fusion and a single-frame baseline:

```bash
ros2 launch exploration_manager exploration.launch.py                        # fusion on
ros2 launch exploration_manager exploration.launch.py multiview_fusion:=false
```

It sets three coupled parameters together, since changing only one does not give a clean
baseline:

| parameter | `:=true` | `:=false` |
|---|---|---|
| `object.fusion_type` | `1` (weighted) | `0` (latest frame wins) |
| `object.min_observation_num` | `2` | `1` |
| `object.use_observation` | `true` | `false` |

With fusion off, an object cluster's confidence is whatever the detector reported in the
current frame, one sighting is enough to accept it, and negative-evidence decay is
disabled. Point clouds still accumulate across frames - navigation targets depend on the
cluster geometry - so this isolates the *confidence* signal, not all multi-view behaviour.
The per-category LLM acceptance threshold arriving on `/detector/confidence_threshold` is
unchanged, so both arms share the same gate.

`habitat_evaluation.py` reads these parameters back from the running planner and stamps
them into every record, so an arm is never mislabelled. Verify with:

```bash
ros2 param get /exploration_node object.fusion_type
```

### Running components by hand

One shell per component (`... exec apexnav bash`). The environment matters: the servers
only run under `use_vlm`, and anything importing rclpy only runs under `use_core`. The
prompt shows which you are in - `[apexnav:core]` or `[apexnav:vlm]`.

```bash
use_vlm                                          # python 3.9
python -m vlm.detector.grounding_dino --port 12181
python -m vlm.itm.blip2itm            --port 12182
python -m vlm.segmentor.sam           --port 12183
python -m vlm.detector.yolov7         --port 12184

use_core                                         # python 3.12, the default
ros2 launch exploration_manager foxglove.launch.py     # replaces rviz.launch.py
ros2 launch exploration_manager exploration.launch.py
python habitat_evaluation.py --dataset hm3dv2
```

### Connect Foxglove

From your laptop:

```bash
# host-side port is 8766 here (8765 is taken on this machine) - see "Host specifics"
ssh -L 8765:localhost:8766 <this-host>
```

Open Foxglove → **Open connection** → `ws://localhost:8765`, and import
`src/planner/exploration_manager/config/apexnav_foxglove_layout.json` as a layout.

### Stopping

```bash
docker compose -f docker/docker-compose.yml exec apexnav tmux kill-session -t apexnav
docker compose -f docker/docker-compose.yml down
```

Nothing is lost on `down`: code, `build/`, `install/` and results are all bind mounts.

---

## What runs where

The four VLM servers bind `localhost` (`vlm/server_wrapper.py`) and every client
hardcodes `http://localhost:<port>` (`vlm/segmentor/sam.py` and friends), so all
processes share one network namespace — one container, not a compose fan-out.

Inside it there are two Python environments:

| | Env A | Env B |
|---|---|---|
| activate | `use_core` (default) | `use_vlm` |
| path | `/opt/venv/apexnav` | `/opt/conda/envs/apexnav_vlm` |
| python | 3.12 | 3.9 |
| holds | ROS 2 Jazzy, habitat-sim 0.3.4, habitat-lab, CUDA torch | salesforce-lavis, GroundingDINO + CUDA ext, MobileSAM, YOLOv7 |
| runs | `habitat_evaluation.py`, the C++ nodes, `foxglove_bridge` | the four `python -m vlm.*` servers |

Both see the repo through `PYTHONPATH=/workspace/ApexNav` — the source tree is used
directly rather than pip-installed, so an edit to `vlm/` is live in both immediately,
with nothing to reinstall and no second copy to keep in sync.

`use_core` and `use_vlm` are shell functions from `/opt/apexnav/apexnav-env.sh`,
available in any shell in the container including `docker exec`.

## Which GPU each process uses

The container sees **every** GPU (`gpus: all`, and `CUDA_VISIBLE_DEVICES` is not set in
`docker-compose.yml`), so the choice is per session rather than per container — no
recreate needed to move work to another card.

None of the four servers takes a `--device` flag: each asks for `torch.device("cuda")`,
which is the first device *its own process* can see. `start_session.sh` exploits that by
giving each pane its own `CUDA_VISIBLE_DEVICES`:

```bash
DC="docker compose -f docker/docker-compose.yml"

# everything on GPU 3
$DC exec -e VLM_GPU=3 apexnav ./scripts/start_session.sh

# servers on 3, habitat-sim's EGL context on 4
$DC exec -e VLM_GPU=3 -e SIM_GPU=4 apexnav ./scripts/start_session.sh

# blip2itm is the heaviest of the four; give it a card of its own
$DC exec -e VLM_GPU=3 -e GPU_BLIP2ITM=4 apexnav ./scripts/start_session.sh
```

`exec` does not inherit your host shell's environment, hence `-e` on each variable —
`VLM_GPU=3 $DC exec apexnav ...` would silently start on GPU 0. From a shell already
inside the container the plain prefix form works: `VLM_GPU=3 ./scripts/start_session.sh`.

| variable | default | applies to |
|---|---|---|
| `VLM_GPU` | `0` | all four servers, and `SIM_GPU` if unset |
| `GPU_GROUNDING_DINO`, `GPU_BLIP2ITM`, `GPU_SAM`, `GPU_YOLOV7` | `VLM_GPU` | that one server |
| `SIM_GPU` | `VLM_GPU` | the eval window — habitat-sim and anything else run there |

Indices are host GPU indices as `nvidia-smi` reports them, *not* offsets into some
already-masked list: a process-level `CUDA_VISIBLE_DEVICES` replaces the inherited value
outright rather than nesting inside it. The script checks the indices against
`nvidia-smi -L` and refuses to start on a bad one, because the failure is otherwise quiet
— `sam` and `blip2itm` fall back to CPU and just look slow. The started session prints
its assignment; `nvidia-smi` confirms it once the weights load.

The `ros` window gets no `CUDA_VISIBLE_DEVICES` at all — `foxglove_bridge` and the
planner are C++ nodes that never touch CUDA.

Two sessions cannot split GPUs inside one container: the ports (12181–12184) are
hardcoded in the clients, so only one set of servers can run at a time. Use a second
container for that.

## Why two environments

Jazzy's `rclpy` is a **Python 3.12** C extension. Every `habitat-sim` conda build ever
published — 0.1.4 through 0.3.3 — is **Python 3.9 only**. `habitat_evaluation.py`
imports both, so with published binaries there is no interpreter where it runs. That
is upstream issue
[#36](https://github.com/Robotics-STAR-Lab/ApexNav/issues/36), still open:

```
ModuleNotFoundError: No module named 'rclpy._rclpy_pybind11'
The C extension '/opt/ros/.../_rclpy_pybind11.cpython-39-x86_64-linux-gnu.so' isn't present
```

The fix, which the maintainer gives in that thread, is to build habitat-sim from
source on a newer Python. This image builds **v0.3.4**, the first release whose build
matrix targets 3.10–3.12 (its own README says `conda create -n habitat python=3.12`).
It was never published to the `aihabitat` channel, which is why the branch is stuck
on 0.3.1/py3.9.

Env B then exists for one reason only: `salesforce-lavis==1.0.2` carries py3.9-era
pins (`opencv-python-headless==4.5.5.64`, `decord`) with no cp312 wheels. The VLM
servers need neither ROS nor Habitat, so isolating them costs nothing.

### Why not the conda env

`apexnav_environment.yaml` still reads `name: apexnav` / `python=3.9.21` /
`habitat-sim=0.3.1=py3.9_*` on every branch, including `ros2-jazzy`. The ROS 2 port
([PR #33](https://github.com/Robotics-STAR-Lab/ApexNav/pull/33)) updated the README to
say `conda activate apexnav_ros2` but never regenerated the env file, so following the
branch's install steps literally produces the ImportError above. The port itself is
sound — the contributor tested build, both control modes, and all four servers — it is
the install doc that is stale.

### Deviations from the README, and why

Several README steps no longer work at all in 2026 — not because the instructions were
wrong, but because pinned artifacts were withdrawn and defaults changed underneath them.
Each row below is a thing that fails outright if you follow the README literally.

| README | here | why |
|---|---|---|
| `python=3.9` | 3.12 (Env A) | Jazzy `rclpy` is cpython-312 |
| `habitat-sim 0.3.1` conda | v0.3.4 from source | only source builds support 3.12 |
| `python setup.py install --headless --bullet` | `HABITAT_BUILD_GUI_VIEWERS=OFF pip install .` | **fails**: v0.3.4 moved to scikit-build-core, and `setup.py` is now a shim that exits 1. Bullet is on by default. |
| `torch==2.5.0 --index-url .../cu124` | `torch==2.8.0`, `torchvision==0.23.0`, cu126 | **fails**: 2.5.0 pins `nvidia-cudnn-cu12==9.1.0.70`, which has been pulled from the index (versions jump 9.0.0.312 → 9.1.1.17). 2.8.0/0.23.0 is the newest pair with both cp312 and cp39 wheels, so both envs share one version. |
| `pip install tf-transformations` | `ros-jazzy-tf-transformations` (apt) | **fails**: no such project on PyPI; it is a ROS package |
| `pip install salesforce-lavis==1.0.2` | same, but `--no-deps` + a curated dep set | **fails**: it requires an *unpinned* `spacy`, which now resolves to a version needing `thinc>=8.3.12` / Python ≥3.10, unsatisfiable in a py3.9 env. Pinned to `spacy==3.7.5`. |
| Miniconda | Miniforge | **fails**: Miniconda's `defaults` channels now refuse non-interactive use pending Anaconda ToS acceptance (`CondaToSNonInteractiveError`), and those terms carry commercial conditions. Miniforge is the same conda on conda-forge. |
| `numpy==1.23.5` | 1.26.4 (Env A) | no cp312 wheels for 1.23.5; 1.26.4 is habitat-lab v0.3.4's own pin. Safe: ApexNav's Python uses no numpy aliases removed in 1.24+, and no numba JIT despite the `numba` pin. Env B keeps 1.23.5 exactly. |
| `opencv-python==4.6.0.66` | 4.10.0.84 (Env A) | no cp312 wheels for 4.6. Env B keeps 4.6.0.66. |
| `open3d`, `moviepy` pins | dropped | nothing in the repo imports either |
| `roslaunch ... rviz.launch` | `ros2 launch exploration_manager foxglove.launch.py` | no display on this host |
| `ros-jazzy-desktop` | `ros-jazzy-ros-base` + explicit deps | RViz2's Qt stack is unused here |

Two upstream source patches are applied in the Dockerfile, both consequences of the
forced torch upgrade:

- **GroundingDINO** — `Tensor::type()` no longer compiles. The ros2-jazzy README says to
  replace `value.type()` with `value.scalar_type()`, which is right for
  `AT_DISPATCH_FLOATING_TYPES(value.type(), ...)` but wrong for the eleven
  `X.type().is_cuda()` device assertions; applying it as a blanket substitution yields
  `error: expression must have class type but it has type "c10::ScalarType"`. The two
  cases get different rewrites.
- **yolov7** — torch 2.6 flipped `torch.load`'s default to `weights_only=True`, so
  loading `yolov7-e6e.pt` raises `Weights only load failed ... models.yolo.Model`.
  yolov7 passes no flag, so `weights_only=False` is patched into `attempt_load`. The
  checkpoint is the official release artifact, so this is not a new trust boundary.

The `transformers` conflict from upstream issues #22/#23 (lavis pins `<4.27`, ApexNav
pins `4.43.2`) does not arise here: with lavis installed `--no-deps` there is nothing to
resolve against, and 4.43.2 is what runs either way.

---

## Data

`scripts/setup_data.sh` writes to `~/nas/apexnav_data` (override with `APEXNAV_DATA`),
bind-mounted at `/workspace/ApexNav/data`:

```
scene_datasets/hm3d/val/…      100 HM3D-v0.2 val scenes, copied from an existing
scene_datasets/hm3d_v0.2 -> hm3d   local tree rather than re-downloaded
datasets/objectnav/hm3d/{v1,v2}/
datasets/objectnav/mp3d/v1/
groundingdino_swint_ogc.pth, mobile_sam.pt, yolov7-e6e.pt
ollama/                         qwen3:8b, if you pull it
```

Two things worth knowing:

- **MP3D episodes are not optional.** `habitat_evaluation.py` opens
  `data/datasets/objectnav/mp3d/v1/val/val.json.gz` for its category-name mapping on
  every run, HM3D included. The MP3D *scene* dataset needs a separate Matterport
  application and is not installed, so `--dataset mp3d` will not run.
- The MP3D episode URL contains `objectnav/m3d/v1/`. That is not a typo to fix — the
  `mp3d` spelling returns 403; `m3d` is the path that serves.

## Foxglove instead of RViz

`foxglove.launch.py` mirrors `rviz.launch.py`: it starts `foxglove_bridge` on port
8765 and keeps the same `world → navigation` static transform, which the Foxglove 3D
panel needs as a frame to render the world-framed clouds against.

`apexnav_foxglove_layout.json` is `ApexNav.rviz` translated panel-for-panel — same
topics, same colors, same defaults for which layers start visible. Point clouds that
RViz drew with `FlatColor` use flat colors here; `value_map` and `confidence_map`,
which RViz colored by `Intensity`, use a turbo colormap on the intensity field.

Custom messages need no setup: the bridge sends each topic's schema from the ROS 2
typesupport, so `plan_env/MultipleMasksWithConfidence` and `trajectory_manager/PolyTraj`
decode in Foxglove as long as the bridge runs from the sourced workspace, which it does.

## Extending the perception stack

Three places, depending on what you are changing:

- **Detection fusion / accept–reject** — `vlm/utils/get_object_utils.py::get_object`.
  Runs in Env A, in the habitat process, not in any server. It calls the four clients
  and decides what counts as a detection.
- **Semantic fusion, confidence accumulation** — C++,
  `src/planner/plan_env/src/object_map2d.cpp` and `value_map2d.cpp`, consuming
  `/detector/clouds_with_scores` and `/detector/confidence_threshold`.
- **A new model** — either a new HTTP server following
  `vlm/server_wrapper.py::host_model` on a free port, or inline on GPU in Env A, which
  is what the CUDA torch in Env A is for.

If the split ever gets in the way: lavis is the *only* thing pinning Env B to 3.9.
Rewriting `vlm/itm/blip2itm.py` against `transformers.Blip2ForImageTextRetrieval` with
the same `Salesforce/blip2-itm-vit-g` checkpoint (~20 lines) collapses everything into
one 3.12 env. Reproduce the paper's numbers first — that cosine score feeds the value
map's thresholds.

## Host specifics worth knowing

**Docker here is rootless.** That inverts the usual UID story: your host files appear
inside the container as uid 0, and container root maps back to your host uid. So the
container runs as **root**, which is what keeps `build/`, `install/`, `log/`, `videos/`
and `outputs/` owned by you on the bind mount — and it is unprivileged on the host. A
non-root container user maps into the subuid range and gets `EACCES` on your own files
(`colcon build` fails outright). Under rootful Docker, invert it: `APEX_USER=1018:1020`.

**Ports.** Host 8765 and 11434 are already taken by other processes on this machine, so
the compose file publishes Foxglove on **host 8766** (container side unchanged at 8765)
and does not publish ollama at all. Tunnel accordingly:
`ssh -L 8765:localhost:8766 <host>`. Override with `APEX_FOXGLOVE_PORT`.

## Known upstream quirk: record.txt in single-episode mode

`habitat_evaluation.py` prints `Episode N data written to .../record.txt`, but the
`if flag_once: break` returns *before* the `write_record` calls. So when you pass
`test_epi_num=<id>`, the message appears and no file is written. Metrics still print to
stdout. Multi-episode runs (no `test_epi_num`) write the record normally. Not something
this container setup changes.

The episode recorder is deliberately flushed *before* that break, so `episodes.jsonl` and
the artifact directories are written in single-episode mode too - which is what makes
`test_epi_num=<N>` usable for re-running a recorded false positive.

## Troubleshooting

**`ImportError: ... _rclpy_pybind11`** — you are in Env B. Run `use_core`.

**habitat-sim fails to create a context** — the container needs
`NVIDIA_DRIVER_CAPABILITIES=all`; the `graphics` capability, not just `compute`, is
what EGL needs. Already set in `docker-compose.yml`.

**`from plan_env.msg import ...` fails after a successful build** — the workspace was
built against the wrong interpreter. Rebuild with `./scripts/build_ws.sh`, which passes
`-DPython3_EXECUTABLE=/opt/venv/apexnav/bin/python`.

**Foxglove connects but no topics** — check the bridge is running
(`ros2 node list | grep foxglove`) and that both sides share `ROS_DOMAIN_ID=42`.

**Nodes see no data from each other** — Fast-DDS shared memory needs a large `/dev/shm`;
the compose file sets `shm_size: 8gb`.

**Slow episode transitions** — scene GLBs load from NFS. If it is painful, copy the
val subset you use to local disk and point `APEXNAV_DATA` at it.
