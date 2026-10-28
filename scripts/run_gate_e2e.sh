#!/usr/bin/env bash
# Run hm3dv2 episodes with the perception gate off or on, inside the apexnav container.
#
#   ./scripts/run_gate_e2e.sh <baseline|gate> <episode>...
#   SIM_GPU=0 OUT_ROOT=videos/gate_e2e ./scripts/run_gate_e2e.sh gate 213 229 529
#   SYNC_STEP=true ./scripts/run_gate_e2e.sh baseline 213     # deterministic step
#   EVAL_ARGS="verification.on_view_reject=defer" ./scripts/run_gate_e2e.sh gate 213
#
# Needs the four VLM servers running (scripts/start_session.sh) and, for "gate", the
# perception server reachable at verification.url. The planner is (re)started here with
# the matching verification_gate and sync_step flags. Touch $OUT_ROOT/ABORT to stop after
# the current episode (scripts/gpu_guard.sh does this automatically).
#
# SYNC_STEP=true (sync_step mode, see habitat_evaluation.py) also restarts the planner for
# every episode (PLANNER_PER_EPISODE, default on with SYNC_STEP), so no planner or TSP
# solver state carries over from the previous episode; its log is $OUT/planner.log.
MODE=${1:?usage: run_gate_e2e.sh <baseline|gate> <episode>...}; shift
case "$MODE" in baseline) GATE=false ;; gate) GATE=true ;; *) echo "mode must be baseline or gate"; exit 2 ;; esac
WS=/workspace/ApexNav
OUT_ROOT=${OUT_ROOT:-videos/gate_e2e}
DATASET=${DATASET:-hm3dv2}
SYNC_STEP=${SYNC_STEP:-false}
case "$SYNC_STEP" in true|false) ;; *) echo "SYNC_STEP must be true or false"; exit 2 ;; esac
PLANNER_PER_EPISODE=${PLANNER_PER_EPISODE:-$SYNC_STEP}

# ROS's setup.bash reads unset variables, so set -u only after sourcing it.
source /opt/apexnav/apexnav-env.sh >/dev/null 2>&1
use_core
set -u
cd "$WS"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=${SIM_GPU:-0}

# Every episode needs the four VLM servers (scripts/start_session.sh). Without them
# habitat_evaluation.py logs connection errors and still exits 0, so check first.
vlm_up() {
  local port
  for port in 12181 12182 12183 12184; do
    (exec 3<>/dev/tcp/127.0.0.1/$port) 2>/dev/null || { echo "VLM server on port $port is down"; return 1; }
  done
}

stop_planner() { pkill -f exploration.launch.py; pkill -f exploration_node; pkill -f tsp_node; sleep 3; }
start_planner() {  # log file, seconds to wait
  ros2 launch exploration_manager exploration.launch.py verification_gate:=$GATE sync_step:=$SYNC_STEP > "$1" 2>&1 &
  sleep "$2"
}
planner_up() { pgrep -f exploration_node >/dev/null && pgrep -f tsp_node >/dev/null; }

stop_planner
mkdir -p "$OUT_ROOT"
if [ "$PLANNER_PER_EPISODE" != true ]; then
  start_planner "$OUT_ROOT/planner_$MODE.log" 25
fi

for N in "$@"; do
  if [ -f "$OUT_ROOT/ABORT" ]; then echo "$OUT_ROOT/ABORT exists, stopping"; break; fi
  if ! vlm_up; then echo "stopping before epi$N"; stop_planner; exit 1; fi
  OUT=$OUT_ROOT/$MODE/epi$N
  mkdir -p "$OUT"
  if [ "$PLANNER_PER_EPISODE" = true ]; then
    stop_planner
    start_planner "$OUT/planner.log" 5
  fi
  if ! planner_up; then echo "planner is down before epi$N"; exit 1; fi
  echo "$(date +%T) start $MODE epi$N"
  timeout 3600 python habitat_evaluation.py --dataset "$DATASET" test_epi_num=$N video_output_path=$OUT \
      verification.enabled=$GATE sync_step=$SYNC_STEP "record.flush_on=[all]" ${EVAL_ARGS:-} > "$OUT/eval.log" 2>&1 < /dev/null
  rc=$?
  echo "$(date +%T) done $MODE epi$N rc=$rc"
done
stop_planner
