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

`habitat_evaluation.py` prints `Episode N data written to .../record.txt` at line 543,
but line 553 (`if flag_once: break`) returns *before* the `write_record` calls at
557/569. So when you pass `test_epi_num=<id>`, the message appears and no file is
written. Metrics still print to stdout. Multi-episode runs (no `test_epi_num`) write the
record normally. Not something this container setup changes.

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
