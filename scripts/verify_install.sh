#!/usr/bin/env bash
#
# Post-install self-check. Run inside the container:
#   docker compose exec apexnav ./scripts/verify_install.sh
#
# Each check gates the next one conceptually, but all of them run so you get the
# full picture in one pass. Exit code is non-zero if anything failed.

# No `set -u`: ROS 2's setup.bash dereferences unset variables.
set -o pipefail

# shellcheck disable=SC1091
source /opt/apexnav/apexnav-env.sh

WS=/workspace/ApexNav
cd "${WS}"
failures=0

section() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
pass()    { printf '  \033[32mok\033[0m      %s\n' "$*"; }
fail()    { printf '  \033[31mFAILED\033[0m  %s\n' "$*"; failures=$((failures + 1)); }

check() {  # label  command...
    local label="$1"; shift
    local out
    if out=$("$@" 2>&1); then
        pass "${label}${out:+ - ${out}}"
    else
        fail "${label}"
        printf '%s\n' "${out}" | sed 's/^/          /' | tail -15
    fi
}

# ---------------------------------------------------------------------------
section "Env A (python 3.12) - ROS 2 + Habitat in one interpreter"
# ---------------------------------------------------------------------------
# This is the check that matters most: upstream issue #36 is exactly the claim that
# rclpy and habitat_sim cannot be imported together with the published packages.
use_core
check "python version" python -c "import sys; print('.'.join(map(str, sys.version_info[:3])))"
check "rclpy + habitat_sim together" python -c "
import rclpy, habitat_sim, habitat, numpy, torch
print(f'habitat_sim={habitat_sim.__version__} numpy={numpy.__version__} torch={torch.__version__} cuda={torch.cuda.is_available()}')
"
check "vlm client import chain" python -c "
import vlm.utils.get_object_utils, vlm.utils.get_itm_message
print('get_object + itm clients import')
"
check "habitat2ros" python -c "import habitat2ros; print('ok')"

# ---------------------------------------------------------------------------
section "Env B (python 3.9) - VLM model stack"
# ---------------------------------------------------------------------------
use_vlm
check "python version" python -c "import sys; print('.'.join(map(str, sys.version_info[:3])))"
# Importing _C proves the CUDA extension actually compiled, not just installed.
check "torch + lavis + groundingdino._C" python -c "
import torch, lavis
from groundingdino import _C
print(f'torch={torch.__version__} cuda={torch.cuda.is_available()}')
"
check "mobile_sam + yolov7 deps" python -c "
import mobile_sam, ultralytics, cv2, numpy
print(f'cv2={cv2.__version__} numpy={numpy.__version__}')
"
use_core

# ---------------------------------------------------------------------------
section "Headless rendering (EGL, no X server)"
# ---------------------------------------------------------------------------
scene=$(find "${WS}/data/scene_datasets/hm3d/val" -maxdepth 2 -name '*.basis.glb' 2>/dev/null | head -1)
if [ -z "${scene}" ]; then
    fail "no HM3D scene found - run scripts/setup_data.sh first"
else
    check "habitat_sim steps a real scene" python - "${scene}" <<'PY'
import sys
import habitat_sim

scene = sys.argv[1]
cfg = habitat_sim.SimulatorConfiguration()
cfg.scene_id = scene
cfg.enable_physics = False

rgb = habitat_sim.CameraSensorSpec()
rgb.uuid, rgb.sensor_type, rgb.resolution = "rgb", habitat_sim.SensorType.COLOR, [480, 640]
depth = habitat_sim.CameraSensorSpec()
depth.uuid, depth.sensor_type, depth.resolution = "depth", habitat_sim.SensorType.DEPTH, [480, 640]

agent = habitat_sim.agent.AgentConfiguration()
agent.sensor_specifications = [rgb, depth]

with habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent])) as sim:
    obs = sim.step("move_forward")
    print(f"rgb={obs['rgb'].shape} depth={obs['depth'].shape}")
PY
fi

# ---------------------------------------------------------------------------
section "ROS 2 workspace"
# ---------------------------------------------------------------------------
if [ ! -f "${WS}/install/setup.bash" ]; then
    fail "workspace not built - run ./scripts/build_ws.sh"
else
    # `ros2` on PATH is itself worth asserting: a PATH-ordering bug in apexnav-env.sh
    # once removed it from every shell while the msg-import check below still passed.
    check "ros2 on PATH" bash -c "command -v ros2"
    check "packages present" bash -c "set -o pipefail; n=\$(ros2 pkg list | grep -cE '^(exploration_manager|plan_env|path_searching|trajectory_manager|vis_utils|lkh_mtsp_solver)\$'); echo \"\${n}/6 apexnav packages\"; [ \"\${n}\" -eq 6 ]"
    # Generated message modules must be importable from Env A's interpreter, which is
    # what -DPython3_EXECUTABLE in build_ws.sh is for.
    check "custom msgs importable from Env A" python -c "
from plan_env.msg import MultipleMasksWithConfidence
from trajectory_manager.msg import PolyTraj
print('MultipleMasksWithConfidence, PolyTraj')
"
    check "foxglove_bridge available" bash -c "set -o pipefail; ros2 pkg executables foxglove_bridge | head -1"
    check "foxglove launch installed" bash -c "set -o pipefail; ls \$(ros2 pkg prefix exploration_manager)/share/exploration_manager/launch/foxglove.launch.py"
    check "foxglove layout installed" bash -c "set -o pipefail; ls \$(ros2 pkg prefix exploration_manager)/share/exploration_manager/config/apexnav_foxglove_layout.json"
fi

# ---------------------------------------------------------------------------
section "Data"
# ---------------------------------------------------------------------------
for p in data/scene_datasets/hm3d/val data/datasets/objectnav/hm3d/v2/val \
         data/datasets/objectnav/mp3d/v1/val data/groundingdino_swint_ogc.pth \
         data/mobile_sam.pt data/yolov7-e6e.pt; do
    if [ -e "${WS}/${p}" ]; then pass "${p}"; else fail "${p}"; fi
done

# ---------------------------------------------------------------------------
if [ "${failures}" -eq 0 ]; then
    printf '\n\033[32mAll checks passed.\033[0m\n'
else
    printf '\n\033[31m%d check(s) failed.\033[0m\n' "${failures}"
fi
exit "${failures}"
