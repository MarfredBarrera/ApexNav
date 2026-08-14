# Shell environment for the ApexNav container.
# Sourced by the entrypoint and by ~/.bashrc, so `docker exec ... bash` works too.
# shellcheck shell=bash

export APEX_WS=/workspace/ApexNav
export APEX_CORE_VENV=${APEX_CORE_VENV:-/opt/venv/apexnav}
export APEX_VLM_ENV=${APEX_VLM_ENV:-/opt/conda/envs/apexnav_vlm}

# --- ROS 2 --------------------------------------------------------------------
# Sourced BEFORE APEX_BASE_PATH is captured: setup.bash puts /opt/ros/jazzy/bin on
# PATH, and use_core/use_vlm rebuild PATH from APEX_BASE_PATH. Capturing first would
# silently drop `ros2` from every shell.
# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
if [ -f "${APEX_WS}/install/setup.bash" ]; then
    # shellcheck disable=SC1091
    source "${APEX_WS}/install/setup.bash"
fi

# PATH with ROS but without either Python env prepended, so use_core/use_vlm can
# switch cleanly instead of stacking.
if [ -z "${APEX_BASE_PATH:-}" ]; then
    export APEX_BASE_PATH="$PATH"
fi

# ROS 2 setup.bash prepends to PYTHONPATH, so re-assert the workspace root.
export PYTHONPATH="${APEX_WS}:${PYTHONPATH}"
export APEX_ROS_PYTHONPATH="$PYTHONPATH"

# Env A: ROS 2 + Habitat + your code. The default.
use_core() {
    export PATH="${APEX_CORE_VENV}/bin:${APEX_BASE_PATH}"
    export VIRTUAL_ENV="${APEX_CORE_VENV}"
    export PYTHONPATH="${APEX_ROS_PYTHONPATH}"
    export APEX_ENV=core
}

# Env B: the four VLM servers. Python 3.9, no ROS - the ROS entries are stripped
# from PYTHONPATH because they point at python3.12 trees this interpreter can't use.
use_vlm() {
    export PATH="${APEX_VLM_ENV}/bin:${APEX_BASE_PATH}"
    unset VIRTUAL_ENV
    export PYTHONPATH="${APEX_WS}"
    export APEX_ENV=vlm
}

use_core

# Short prompt marker so it is obvious which interpreter a pane is using.
if [ -n "${PS1:-}" ]; then
    PS1='[apexnav:${APEX_ENV}] \w\$ '
fi
