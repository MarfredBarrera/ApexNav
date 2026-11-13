#!/usr/bin/env bash
# Closed-loop run of one strategy with WORKERS episodes in parallel on GPU $GPU (host side): the whole run in one go.
#
#   WORKERS=3 GPU=6 NAME=<name> EPISODES="63 445 95" ./scripts/run_parallel.sh <strategy> [sync, default true]
#   WORKERS=3 GPU=6 NAME=sync_baseline EPISODES_FILE=videos/active_approach/tmp/sc/gate_v15_episodes.txt \
#       ./scripts/run_parallel.sh baseline
#   WORKERS=3 GPU=6 NAME=sync_jev_sam3_mppi_v15_x APPROACH_PLANNER=mppi_v15 EPISODES="..." ./scripts/run_parallel.sh jev_sam3
#
# Starts the four VLM servers (scripts/start_session.sh, all on $GPU) unless they are up, closes the session's own
# planner and foxglove window (each worker launches its own planner), then runs scripts/run_strategy.sh once with
# WORKERS=$WORKERS and REUSE_SERVER=1: the episodes go on a shared queue that WORKERS copies of run_gate_e2e.sh
# drain, each with its own simulator, planner and ROS_DOMAIN_ID, all sharing the VLM servers and one perception
# server. Strategy options (APPROACH_PLANNER, LAST_MILE, MV_ARGS, DATASET, ...) pass through to run_strategy.sh.
# Output as for run_strategy.sh: videos/<name>/<baseline|gate>/epiN, log videos/<name>.log (this script) and
# videos/<name>/run_gate.log (the workers). Resume: rerun the same line; finished episodes (episodes.jsonl) are
# skipped and a half-run one is moved aside. Touch videos/<name>/ABORT to stop after the running episodes.
# GPU memory sets the worker count: the stack holds ~39 GiB of the 44.4 GiB, each simulator ~0.7 GiB, and an MPPI
# approach on the GPU another ~0.7 GiB CUDA context (4 workers with GPU MPPI ran out of memory, 2026-11-10). So with
# WORKERS >= 3 the MPPI scoring runs on the CPU (MPPI_GPU=cpu; same decisions on the 12-episode check); set MPPI_GPU=0
# to force the GPU. WORKERS=3 is the recommended setting (4 peaked at 44.9 of 45.0 GB).
# On exit (done, failure, Ctrl-C/TERM) it stops the perception server and the VLM servers, so the GPU is not left
# held; KEEP_SERVERS=1 leaves them up for a next run (stop them yourself after).
set -u
STRAT=${1:?usage: WORKERS=N GPU=g NAME=name EPISODES="..." run_parallel.sh <strategy> [sync]}
SYNC=${2:-true}
GPU=${GPU:?set GPU}
NAME=${NAME:?set NAME}
WORKERS=${WORKERS:-3}
[ "$WORKERS" -ge 3 ] && export MPPI_GPU=${MPPI_GPU:-cpu}
DEV=3dawareverification-dev-1
VPORT=${VLM_PORT_BASE:-12181}  # VLM servers on VPORT..VPORT+3; a second concurrent run sets VLM_PORT_BASE (e.g. 12281),
PPORT=${PERCEPTION_PORT:-8770} # PERCEPTION_PORT (e.g. 8771), DOMAIN_BASE (e.g. 60) and SESSION (tmux, e.g. p2vlm)
SESSION=${SESSION:-apexnav}
export SESSION
cd /home/marfred/codes/ApexNav
EPIS=(${EPISODES:-$(cat ${EPISODES_FILE:?set EPISODES or EPISODES_FILE})})
[ ${#EPIS[@]} -gt 0 ] || { echo "no episodes"; exit 2; }

stop_servers() {
  [ "${KEEP_SERVERS:-0}" = 1 ] && { echo "$(date +%T) servers left running (KEEP_SERVERS=1)"; return; }
  # only the servers on THIS run's ports, by PID (the server's workers are its children)
  for p in $(docker exec $DEV pgrep -f "^python -m perception.server --host 0.0.0.0 --port $PPORT "); do
    w=$(docker exec $DEV pgrep -P $p); docker exec $DEV kill $p
    sleep 20  # the server stops its workers
    for q in $w; do docker exec $DEV kill $q 2>/dev/null; done
  done
  docker exec apexnav tmux kill-session -t "=$SESSION" 2>/dev/null
  echo "$(date +%T) servers stopped"
}
child=
trap stop_servers EXIT
trap '[ -n "$child" ] && kill $child 2>/dev/null && wait $child; exit 130' INT TERM

vlm_up() { docker exec apexnav bash -c "for p in \$(seq $VPORT $((VPORT + 3))); do (exec 3<>/dev/tcp/127.0.0.1/\$p) 2>/dev/null || exit 1; done"; }
if ! vlm_up; then
  echo "$(date +%T) starting the VLM servers on GPU $GPU"
  docker exec -e VLM_GPU=$GPU -e SIM_GPU=$GPU -e OMP_NUM_THREADS=2 -e SESSION=$SESSION -e VLM_PORT_BASE=$VPORT apexnav bash -c \
      "cd /workspace/ApexNav && ./scripts/start_session.sh" > /dev/null
  for _ in $(seq 60); do vlm_up && break; sleep 5; done
  vlm_up || { echo "VLM servers did not come up"; exit 1; }
fi
docker exec apexnav tmux kill-window -t "=$SESSION:ros" 2>/dev/null   # its planner/foxglove are not used by the workers

echo "$(date +%T) $NAME: $STRAT, ${#EPIS[@]} episodes, $WORKERS workers, MPPI_GPU=${MPPI_GPU:-0}"
VLM_PORT_BASE=$VPORT REUSE_SERVER=1 WORKERS=$WORKERS GPU=$GPU ./scripts/run_strategy.sh $STRAT $NAME $SYNC "${EPIS[@]}" >> videos/$NAME.log 2>&1 &
child=$!
wait $child   # in the background so a TERM stops the run now (run_strategy.sh's cleanup stops the workers)
rc=$?
echo "$(date +%T) $NAME: run_strategy.sh exit $rc"
[ $rc -eq 0 ] && echo "$(date +%T) all done"
exit $rc
