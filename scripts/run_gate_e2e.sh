#!/usr/bin/env bash
# Run hm3dv2 episodes with the perception gate off or on, inside the apexnav container.
#
#   ./scripts/run_gate_e2e.sh <baseline|gate> <episode>...
#   SIM_GPU=0 OUT_ROOT=videos/gate_e2e ./scripts/run_gate_e2e.sh gate 213 229 529
#   SYNC_STEP=true ./scripts/run_gate_e2e.sh baseline 213     # deterministic step
#   EVAL_ARGS="verification.on_view_reject=defer" ./scripts/run_gate_e2e.sh gate 213
#   APPROACH=true EVAL_ARGS="verification.approach=true verification.on_error=reject" ./scripts/run_gate_e2e.sh gate 213  # d1 server
#
# Needs the four VLM servers running (scripts/start_session.sh) and, for "gate", the
# perception server reachable at verification.url. The planner is (re)started here with
# the matching verification_gate and sync_step flags. Touch $OUT_ROOT/ABORT to stop after
# the current episode (scripts/gpu_guard.sh does this automatically).
#
# SYNC_STEP=true (sync_step mode, see habitat_evaluation.py) also restarts the planner for
# every episode (PLANNER_PER_EPISODE, default on with SYNC_STEP), so no planner or TSP
# solver state carries over from the previous episode; its log is $OUT/planner.log.
#
# Parallel workers (scripts/run_parallel.sh -> run_strategy.sh WORKERS=N): WORKER=<k> makes this one of several
# copies in the container. Worker k talks to its own planner on ROS_DOMAIN_ID DOMAIN_BASE+k (default 50+k), the
# planner writes its LKH files to /tmp/apexnav_tsp_w<k> (APEXNAV_TSP_DIR, read by algorithm.launch.py), and it
# never touches another worker's planner (without WORKER the start also clears any planner left in the container,
# e.g. the one start_session.sh launches). QUEUE=<file> takes the episodes from a shared queue file (one per
# line, popped under flock) instead of the arguments. TERM stops the running episode and this worker's planner.
#
# The episodes run on a snapshot of perception_gate/ and active_approach/ in $OUT_ROOT/code (taken once per OUT_ROOT).
MODE=${1:?usage: run_gate_e2e.sh <baseline|gate> <episode>...}; shift
case "$MODE" in baseline) GATE=false ;; gate) GATE=true ;; *) echo "mode must be baseline or gate"; exit 2 ;; esac
WS=/workspace/ApexNav
OUT_ROOT=${OUT_ROOT:-videos/gate_e2e}
DATASET=${DATASET:-hm3dv2}
SYNC_STEP=${SYNC_STEP:-false}
case "$SYNC_STEP" in true|false) ;; *) echo "SYNC_STEP must be true or false"; exit 2 ;; esac
PLANNER_PER_EPISODE=${PLANNER_PER_EPISODE:-$SYNC_STEP}
APPROACH=${APPROACH:-false}  # planner verification_approach (gate state 3, server selector d1)
case "$APPROACH" in true|false) ;; *) echo "APPROACH must be true or false"; exit 2 ;; esac
WORKER=${WORKER:-}
QUEUE=${QUEUE:-}

# ROS's setup.bash reads unset variables, so set -u only after sourcing it.
source /opt/apexnav/apexnav-env.sh >/dev/null 2>&1
use_core
set -u
cd "$WS"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=${SIM_GPU:-0}
export MPPI_GPU=${MPPI_GPU:-0}   # mppi_v15 batched view scoring on the sim GPU (index 0 under CUDA_VISIBLE_DEVICES)
[ "$MPPI_GPU" = cpu ] && unset MPPI_GPU   # MPPI_GPU=cpu: score on the CPU (no ~0.7 GiB CUDA context per simulator)
TAG=
if [ -n "$WORKER" ]; then
  export ROS_DOMAIN_ID=$(( ${DOMAIN_BASE:-50} + WORKER ))
  export APEXNAV_TSP_DIR=/tmp/apexnav_tsp_w$WORKER
  mkdir -p "$APEXNAV_TSP_DIR"
  TAG="w$WORKER "
  mkdir -p "$OUT_ROOT" && echo $$ > "$OUT_ROOT/.worker$WORKER.pid"   # run_strategy.sh's cleanup TERMs this PID
fi

# Every episode needs the four VLM servers (scripts/start_session.sh). Without them
# habitat_evaluation.py logs connection errors and still exits 0, so check first.
vlm_up() {
  local port
  for port in 12181 12182 12183 12184; do
    (exec 3<>/dev/tcp/127.0.0.1/$port) 2>/dev/null || { echo "${TAG}VLM server on port $port is down"; return 1; }
  done
}

# The planner (ros2 launch + exploration_node + tsp_node) runs in its own session, so it is stopped by that
# session id: SIGINT lets ros2 launch shut its nodes down and reap them, SIGKILL whatever is left after 10 s.
PLANNER=; EVAL=
planner_procs() { [ -n "$PLANNER" ] && ps -s "$PLANNER" -o stat=,args= 2>/dev/null | grep -v '^Z'; }
stop_planner() {
  [ -n "$PLANNER" ] || return 0
  kill -INT -- "-$PLANNER" 2>/dev/null
  for _ in $(seq 20); do planner_procs >/dev/null || break; sleep 0.5; done
  kill -KILL -- "-$PLANNER" 2>/dev/null
  wait "$PLANNER" 2>/dev/null
  PLANNER=
}
start_planner() {  # log file, seconds to wait
  setsid ros2 launch exploration_manager exploration.launch.py verification_gate:=$GATE verification_approach:=$APPROACH \
      sync_step:=$SYNC_STEP > "$1" 2>&1 < /dev/null &
  PLANNER=$!
  sleep "$2"
}
planner_up() { planner_procs | grep -q exploration_node && planner_procs | grep -q tsp_node; }
next_episode() {  # pop the first line of $QUEUE (empty output: queue done)
  flock "$QUEUE.lock" bash -c 'head -n 1 "$1"; sed -i 1d "$1"' _ "$QUEUE"
}
trap '[ -n "$EVAL" ] && kill "$EVAL" 2>/dev/null; stop_planner; [ -n "$WORKER" ] && rm -f "$OUT_ROOT/.worker$WORKER.pid"' EXIT
trap 'exit 143' TERM INT

if [ -z "$WORKER" ]; then  # serial: clear any planner in the container
  pkill -f exploration.launch.py; pkill -f exploration_node; pkill -f tsp_node; sleep 3
fi
mkdir -p "$OUT_ROOT"
# Code snapshot: the episodes import perception_gate/ and active_approach/ (with active_approach.belief's
# replay_deps, found relative to the package) from $OUT_ROOT/code, so edits made while the run goes on do not leak
# into its later episodes. Taken once per OUT_ROOT (later blocks and resumes reuse it; delete $OUT_ROOT/code for a
# new one), under a lock so parallel workers take one. PYTHONSAFEPATH keeps the script's directory ($WS) off the
# front of sys.path, so the snapshot (first on PYTHONPATH) wins over $WS (still on PYTHONPATH for everything else).
CODE=$WS/$OUT_ROOT/code
(
  flock 9
  if [ ! -d "$CODE" ]; then
    rm -rf "$CODE.tmp"
    for pkg in perception_gate active_approach; do mkdir -p "$CODE.tmp/$pkg" && cp $pkg/*.py "$CODE.tmp/$pkg/"; done
    deps=videos/active_approach/replay_deps
    mkdir -p "$CODE.tmp/$deps/perception" && cp $deps/jev_gate.json "$CODE.tmp/$deps/" \
        && cp $deps/perception/*.py "$CODE.tmp/$deps/perception/"
    mv "$CODE.tmp" "$CODE" && echo "$(date +%T) code snapshot in $CODE"
  fi
) 9> "$OUT_ROOT/.code.lock"
export PYTHONSAFEPATH=1 PYTHONPATH="$CODE:$PYTHONPATH"
if [ "$PLANNER_PER_EPISODE" != true ]; then
  start_planner "$OUT_ROOT/planner_$MODE${WORKER:+_w$WORKER}.log" 25
fi

FAILED=0
run_episode() {  # return 1 stops the loop (ABORT, or a server/planner down: FAILED=1)
  local N=$1 OUT=$OUT_ROOT/$MODE/epi$1 rc
  if [ -f "$OUT_ROOT/ABORT" ]; then echo "${TAG}$OUT_ROOT/ABORT exists, stopping"; return 1; fi
  if ! vlm_up; then echo "${TAG}stopping before epi$N"; FAILED=1; return 1; fi
  mkdir -p "$OUT"
  if [ "$PLANNER_PER_EPISODE" = true ]; then
    stop_planner
    [ -n "$WORKER" ] && rm -f "$APEXNAV_TSP_DIR"/atsp_tour.*  # no LKH files from this worker's last episode
    start_planner "$OUT/planner.log" 5
  fi
  if ! planner_up; then echo "${TAG}planner is down before epi$N"; FAILED=1; return 1; fi
  echo "$(date +%T) ${TAG}start $MODE epi$N"
  timeout 3600 python habitat_evaluation.py --dataset "$DATASET" test_epi_num=$N video_output_path=$OUT \
      verification.enabled=$GATE sync_step=$SYNC_STEP "record.flush_on=[all]" ${EVAL_ARGS:-} > "$OUT/eval.log" 2>&1 < /dev/null &
  EVAL=$!
  wait $EVAL  # in the background so a TERM stops the episode now
  rc=$?
  EVAL=
  echo "$(date +%T) ${TAG}done $MODE epi$N rc=$rc"
}

if [ -n "$QUEUE" ]; then
  while N=$(next_episode) && [ -n "$N" ]; do run_episode "$N" || break; done
else
  for N in "$@"; do run_episode "$N" || break; done
fi
exit $FAILED
