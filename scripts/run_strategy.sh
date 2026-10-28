#!/usr/bin/env bash
# Closed-loop ApexNav runs of one verification strategy on GPU $GPU (host side).
#   ./scripts/run_strategy.sh <baseline|sam3|naive|vqf|top2gap|defer> <name> <sync true|false> <episode>...
#   GPU=2 DATASET=ovon ./scripts/run_strategy.sh defer ovon_defer true 2044 2965
# Output: ~/codes/ApexNav/videos/<name>/<baseline|gate>/epiN; perception server logs:
# ~/codes/3DAwareVerification/outputs/apexnav_online/server_<name>/. Episodes that already
# have an episodes.jsonl are skipped (resume). Exits non-zero on ABORT, a server that does
# not come up, or a run that finished fewer episodes than asked.
#   sam3:    --selector input                         (SAM3 on the raw frame only)
#   naive:   --selector naive                         (target prompt alone, score >= 0.5)
#   vqf:     --selector vqf --no-ensemble --judge-input
#   top2gap: --selector top2gap --no-ensemble --judge-input
#   defer:   --selector top2gap --no-ensemble --judge-input --view-when unsure, and the gate
#            defers (verification.on_view_reject=defer) instead of rejecting on a view "no"
# DATASET (default hm3dv2) is passed to habitat_evaluation.py --dataset.
# Needs the four VLM servers (scripts/start_session.sh); does not start or stop them.
# On any exit it stops the perception server/workers, its gpu_guard and any episode or
# planner still running. Each start appends the code versions to videos/<name>/versions.txt.
set -u
STRAT=${1:?strategy}; NAME=${2:?name}; SYNC=${3:?sync}; shift 3
GPU=${GPU:-0}
DATASET=${DATASET:-hm3dv2}
eval_args=
DEV=3dawareverification-dev-1
APEX=/home/marfred/codes/ApexNav
AV=/home/marfred/codes/3DAwareVerification
case $STRAT in
  baseline) mode=baseline; flags= ;;
  sam3) mode=gate; flags="--selector input" ;;
  naive) mode=gate; flags="--selector naive" ;;
  vqf) mode=gate; flags="--selector vqf --no-ensemble --judge-input" ;;
  top2gap) mode=gate; flags="--selector top2gap --no-ensemble --judge-input" ;;
  defer) mode=gate; flags="--selector top2gap --no-ensemble --judge-input --view-when unsure"
         eval_args="verification.on_view_reject=defer" ;;
  *) echo "unknown strategy $STRAT"; exit 2 ;;
esac
root=$APEX/videos/$NAME
mkdir -p $root
if [ -f $root/ABORT ]; then echo "$root/ABORT exists"; exit 1; fi
todo=()
for n in "$@"; do [ -f $root/$mode/epi$n/episodes.jsonl ] || todo+=($n); done
if [ ${#todo[@]} -eq 0 ]; then echo "$(date +%T) $NAME: nothing to do"; exit 0; fi

guard=
cleanup() {
  [ -n "$guard" ] && kill $guard 2>/dev/null
  pkill -f "gpu_guard.sh $GPU" 2>/dev/null
  docker exec $DEV pkill -f "perception.server"
  docker exec $DEV pkill -f "perception.worker"
  docker exec apexnav pkill -f habitat_evaluation.py
  docker exec apexnav pkill -f 'exploration.launch.py|exploration_node|tsp_node'
}
trap cleanup EXIT
trap 'exit 130' INT TERM

{
  echo "== $(date -Is) $STRAT ($mode) dataset=$DATASET sync=$SYNC gpu=$GPU flags='$flags' eval_args='$eval_args'"
  echo "episodes: ${todo[*]}"
  echo "ApexNav $(git -C $APEX rev-parse --abbrev-ref HEAD) $(git -C $APEX rev-parse HEAD)"
  git -C $APEX status --short
  echo "3DAwareVerification $(git -C $AV rev-parse --abbrev-ref HEAD) $(git -C $AV rev-parse HEAD)"
  git -C $AV status --short
  echo "planner $(stat -c '%y' $APEX/build/exploration_manager/exploration_node)"
} >> $root/versions.txt

docker exec $DEV pkill -f "perception.server"; docker exec $DEV pkill -f "perception.worker"; sleep 5
if [ $mode = gate ]; then
  log=outputs/apexnav_online/server_$NAME
  echo "$(date +%T) $NAME: starting server ($flags)"
  docker exec $DEV mkdir -p /workspace/$log
  docker exec -d $DEV bash -lc "cd /workspace && OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 sam3d-env core python -m perception.server --host 0.0.0.0 --port 8770 --gpu $GPU $flags --log-dir $log >> $log/server.out 2>&1"
  for i in $(seq 60); do
    docker exec $DEV sam3d-env core python -m perception.server ping http://localhost:8770 >/dev/null 2>&1 && break
    sleep 10
  done
  docker exec $DEV sam3d-env core python -m perception.server ping http://localhost:8770 >/dev/null 2>&1 || { echo "$NAME: server did not come up"; exit 1; }
else
  echo "$(date +%T) $NAME: no gate"
fi
(cd $APEX && ABORT=$root/ABORT ./scripts/gpu_guard.sh $GPU 44000 8 >> $root/gpu_guard.log 2>&1) &
guard=$!
docker exec apexnav bash -c "cd /workspace/ApexNav && DATASET=$DATASET SIM_GPU=$GPU SYNC_STEP=$SYNC OUT_ROOT=videos/$NAME EVAL_ARGS='$eval_args' ./scripts/run_gate_e2e.sh $mode ${todo[*]} >> videos/$NAME/run_gate.log 2>&1"
done_n=0; for n in "${todo[@]}"; do [ -f $root/$mode/epi$n/episodes.jsonl ] && done_n=$((done_n+1)); done
echo "$(date +%T) $NAME: finished $done_n/${#todo[@]} episodes"
[ -f $root/ABORT ] && { echo "$NAME aborted"; exit 1; }
[ $done_n -eq ${#todo[@]} ] || exit 1
