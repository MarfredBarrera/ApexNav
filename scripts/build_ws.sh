#!/usr/bin/env bash
#
# Build the colcon workspace. Run inside the container:
#   docker compose exec apexnav ./scripts/build_ws.sh
#
# Pass extra colcon arguments through, e.g.
#   ./scripts/build_ws.sh --packages-select plan_env

# No `set -u`: ROS 2's setup.bash dereferences unset variables.
set -eo pipefail

if [ ! -d /opt/ros/jazzy ]; then
    echo "error: this script runs inside the apexnav container." >&2
    exit 1
fi

cd /workspace/ApexNav
source /opt/ros/jazzy/setup.bash

# -DPython3_EXECUTABLE is the important flag: rosidl_generator_py has to emit the
# plan_env / trajectory_manager message modules for the same 3.12 interpreter Env A
# uses, otherwise habitat_evaluation.py's
#   from plan_env.msg import MultipleMasksWithConfidence
# fails at runtime even though the build succeeded.
colcon build \
    --symlink-install \
    --cmake-args \
        -DCMAKE_BUILD_TYPE=Release \
        -DPython3_EXECUTABLE="${APEX_CORE_VENV:-/opt/venv/apexnav}/bin/python" \
    "$@"

echo
echo "Built. Source it with:  source /workspace/ApexNav/install/setup.bash"
echo "(new shells pick it up automatically via /opt/apexnav/apexnav-env.sh)"
