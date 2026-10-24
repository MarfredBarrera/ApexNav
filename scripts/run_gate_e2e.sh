#!/usr/bin/env bash
# Run hm3dv2 episodes with the perception gate off or on, inside the apexnav container.
#
#   ./scripts/run_gate_e2e.sh <baseline|gate> <episode>...
#   SIM_GPU=0 OUT_ROOT=videos/gate_e2e ./scripts/run_gate_e2e.sh gate 213 229 529
#
# Needs the four VLM servers running (scripts/start_session.sh) and, for "gate", the
# perception server reachable at verification.url. The planner is (re)started here with
# the matching verification_gate flag. Touch $OUT_ROOT/ABORT to stop after the current
# episode (scripts/gpu_guard.sh does this automatically).
MODE=${1:?usage: run_gate_e2e.sh <baseline|gate> <episode>...}; shift
case "$MODE" in baseline) GATE=false ;; gate) GATE=true ;; *) echo "mode must be baseline or gate"; exit 2 ;; esac
WS=/workspace/ApexNav
OUT_ROOT=${OUT_ROOT:-videos/gate_e2e}
DATASET=${DATASET:-hm3dv2}

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
stop_planner
mkdir -p "$OUT_ROOT"
ros2 launch exploration_manager exploration.launch.py verification_gate:=$GATE > "$OUT_ROOT/planner_$MODE.log" 2>&1 &
sleep 25

for N in "$@"; do
  if [ -f "$OUT_ROOT/ABORT" ]; then echo "$OUT_ROOT/ABORT exists, stopping"; break; fi
  if ! vlm_up; then echo "stopping before epi$N"; stop_planner; exit 1; fi
  OUT=$OUT_ROOT/$MODE/epi$N
  mkdir -p "$OUT"
  echo "$(date +%T) start $MODE epi$N"
  timeout 3600 python habitat_evaluation.py --dataset "$DATASET" test_epi_num=$N video_output_path=$OUT \
      verification.enabled=$GATE "record.flush_on=[all]" > "$OUT/eval.log" 2>&1 < /dev/null
  echo "$(date +%T) done $MODE epi$N rc=$?"
done
stop_planner
