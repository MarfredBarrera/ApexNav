#!/usr/bin/env bash
# Container entrypoint: set up the shell environment, then run whatever was asked for.
# No `set -u`: ROS 2's setup.bash dereferences unset variables.
set -eo pipefail

# shellcheck disable=SC1091
source /opt/apexnav/apexnav-env.sh

# vlm/detector/yolov7.py does sys.path.insert(0, "yolov7/") and grounding_dino.py
# reads "GroundingDINO/groundingdino/config/...", both relative to the CWD. The
# checkouts live in the image, so link them into the (bind-mounted) workspace.
# Both names are already in .gitignore.
for repo in GroundingDINO yolov7; do
    if [ ! -e "${APEX_WS}/${repo}" ]; then
        ln -s "/opt/external/${repo}" "${APEX_WS}/${repo}" 2>/dev/null || true
    fi
done

if [ ! -d "${APEX_WS}/data" ]; then
    echo "warning: ${APEX_WS}/data is not mounted - run scripts/setup_data.sh on the host first." >&2
fi

exec "$@"
