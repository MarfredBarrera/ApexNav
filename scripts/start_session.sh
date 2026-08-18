#!/usr/bin/env bash
#
# Bring up the whole pipeline in a tmux session. Run inside the container:
#   docker compose exec apexnav ./scripts/start_session.sh
#   docker compose exec apexnav tmux attach -t apexnav
#
# The run sequence spans both Python environments, which is why this exists rather
# than a single command: the four VLM servers are Env B (python 3.9) and everything
# else is Env A (python 3.12).
#
#   window 0  vlm         4 panes: grounding_dino, blip2itm, sam, yolov7
#   window 1  ros         2 panes: foxglove_bridge, exploration
#   window 2  eval        interactive shell in Env A, ready for habitat_evaluation.py
#
# GPU selection is per session, not per container - the container sees every GPU:
#   VLM_GPU=3 ./scripts/start_session.sh              all four servers + habitat on GPU 3
#   VLM_GPU=3 SIM_GPU=4 ./scripts/start_session.sh    servers on 3, habitat-sim on 4
#   VLM_GPU=3 GPU_BLIP2ITM=4 ./scripts/start_session.sh   blip2 (the big one) on its own
# Indices are host GPU indices as `nvidia-smi` reports them. From the host, pass them
# through explicitly - `exec` does not forward your shell's environment:
#   docker compose exec -e VLM_GPU=3 apexnav ./scripts/start_session.sh

set -euo pipefail

SESSION="${SESSION:-apexnav}"
WS=/workspace/ApexNav
DATASET="${DATASET:-hm3dv2}"

# None of the four servers takes a --device flag; each one asks for torch.device("cuda"),
# i.e. the first device its process can see. So the assignment happens here, by giving
# each pane its own CUDA_VISIBLE_DEVICES.
VLM_GPU="${VLM_GPU:-0}"
GPU_GROUNDING_DINO="${GPU_GROUNDING_DINO:-${VLM_GPU}}"
GPU_BLIP2ITM="${GPU_BLIP2ITM:-${VLM_GPU}}"
GPU_SAM="${GPU_SAM:-${VLM_GPU}}"
GPU_YOLOV7="${GPU_YOLOV7:-${VLM_GPU}}"
# habitat-sim's EGL context and anything else run from the eval window.
SIM_GPU="${SIM_GPU:-${VLM_GPU}}"

cd "${WS}"

if [ ! -f "${WS}/install/setup.bash" ]; then
    echo "error: workspace not built - run ./scripts/build_ws.sh first." >&2
    exit 1
fi
for w in data/groundingdino_swint_ogc.pth data/mobile_sam.pt data/yolov7-e6e.pt; do
    if [ ! -s "${WS}/${w}" ]; then
        echo "error: missing ${w} - run scripts/setup_data.sh on the host first." >&2
        exit 1
    fi
done

if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "session '${SESSION}' already running; attach with: tmux attach -t ${SESSION}"
    exit 0
fi

# A bad index is worth catching here: torch just reports no CUDA, and sam/blip2itm fall
# back to CPU silently (grounding_dino dies instead), which reads as "the models are slow".
if command -v nvidia-smi >/dev/null 2>&1; then
    ngpu=$(nvidia-smi -L | wc -l)
    for spec in "${GPU_GROUNDING_DINO}" "${GPU_BLIP2ITM}" "${GPU_SAM}" "${GPU_YOLOV7}" "${SIM_GPU}"; do
        for idx in ${spec//,/ }; do
            case "${idx}" in
                ''|*[!0-9]*) echo "error: GPU index '${idx}' is not a number." >&2; exit 1 ;;
            esac
            if [ "${idx}" -ge "${ngpu}" ]; then
                echo "error: GPU ${idx} does not exist - this host has ${ngpu} (0-$((ngpu - 1)))." >&2
                exit 1
            fi
        done
    done
fi

# Every pane sources the env file, then picks its interpreter with use_core/use_vlm.
# Pass '-' as the GPU for panes that never touch CUDA, so they inherit nothing.
run() {  # window.pane  env  gpu  command
    local gpu_prefix=""
    if [ "$3" != "-" ]; then
        gpu_prefix="CUDA_VISIBLE_DEVICES=$3 "
    fi
    tmux send-keys -t "$1" "source /opt/apexnav/apexnav-env.sh && $2 && cd ${WS} && ${gpu_prefix}$4" C-m
}

# --- window 0: the VLM servers (Env B) ---------------------------------------
tmux new-session  -d -s "${SESSION}" -n vlm -c "${WS}"
tmux split-window -t "${SESSION}:vlm" -h -c "${WS}"
tmux split-window -t "${SESSION}:vlm.0" -v -c "${WS}"
tmux split-window -t "${SESSION}:vlm.2" -v -c "${WS}"

run "${SESSION}:vlm.0" use_vlm "${GPU_GROUNDING_DINO}" "python -m vlm.detector.grounding_dino --port 12181"
run "${SESSION}:vlm.1" use_vlm "${GPU_BLIP2ITM}"       "python -m vlm.itm.blip2itm         --port 12182"
run "${SESSION}:vlm.2" use_vlm "${GPU_SAM}"            "python -m vlm.segmentor.sam        --port 12183"
run "${SESSION}:vlm.3" use_vlm "${GPU_YOLOV7}"         "python -m vlm.detector.yolov7      --port 12184"

# --- window 1: ROS (Env A) ----------------------------------------------------
tmux new-window   -t "${SESSION}" -n ros -c "${WS}"
tmux split-window -t "${SESSION}:ros" -v -c "${WS}"

# Foxglove replaces rviz.launch.py here. Both panes are C++ nodes - no CUDA.
run "${SESSION}:ros.0" use_core - "ros2 launch exploration_manager foxglove.launch.py"
# Give the models a head start; the planner is cheap to restart if you need to.
run "${SESSION}:ros.1" use_core - "sleep 20 && ros2 launch exploration_manager exploration.launch.py"

# --- window 2: the evaluation shell (Env A) -----------------------------------
# Exported rather than prefixed, so re-runs and any other command typed in this pane
# land on the same GPU as the pre-typed one.
tmux new-window -t "${SESSION}" -n eval -c "${WS}"
tmux send-keys -t "${SESSION}:eval" \
    "source /opt/apexnav/apexnav-env.sh && use_core && export CUDA_VISIBLE_DEVICES=${SIM_GPU} && cd ${WS}" C-m
tmux send-keys -t "${SESSION}:eval" \
    "# ready. The VLM servers take ~1-2 min to load their weights, then run:" C-m
tmux send-keys -t "${SESSION}:eval" \
    "python habitat_evaluation.py --dataset ${DATASET} test_epi_num=10"

tmux select-window -t "${SESSION}:eval"

cat <<EOF

tmux session '${SESSION}' started.

  GPUs:      grounding_dino=${GPU_GROUNDING_DINO}  blip2itm=${GPU_BLIP2ITM}  sam=${GPU_SAM}  yolov7=${GPU_YOLOV7}  habitat=${SIM_GPU}
             (set VLM_GPU / SIM_GPU / GPU_<MODEL> when starting the session)
  attach:    docker compose exec apexnav tmux attach -t ${SESSION}
  foxglove:  ssh -L 8765:localhost:8766 \$(hostname)   then connect to ws://localhost:8765
             and import src/planner/exploration_manager/config/apexnav_foxglove_layout.json

The eval window has the command pre-typed but NOT submitted - wait for the four
VLM servers in window 0 to finish loading weights, then press Enter there.
EOF
