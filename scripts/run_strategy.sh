#!/usr/bin/env bash
# Closed-loop ApexNav runs of one verification strategy on GPU $GPU (host side).
#   ./scripts/run_strategy.sh <baseline|sam3|naive|vqf|top2gap|defer|d1|d1mv|jev_sam3> <name> <sync true|false> <episode>...
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
#   d1:      --selector d1 (sufficiency check -> SAM3, else Qwen + SAM3D grid, three-way), and
#            the gate approaches on "insufficient" (verification.approach=true; the planner
#            is launched with verification_approach:=true); server errors reject
#            (verification.on_error=reject), as every other approach failure does
#            D1_EVIDENCE=input: Qwen judges the frame alone (no SAM3D grid; fits one GPU)
#            D1_ACCEPT_MIN=p: Qwen accepts only with p(correct) >= p, else insufficient (approach)
#   d1mv:    d1 with Qwen on the frame alone for first requests (--d1-evidence input) and the
#            close-range decision over the bank of the object's views (verification.multiview=true):
#            Qwen sees up to 6 outlined views plus the SAM3D grid of the best close view
#            (verification.multiview_grid; server --multiview-grid), SAM3 over the views if insufficient
#            MV_GRID=false: images only (no SAM3D at all); MV_VARIANT=crop_small|forced2: prompt variant
#            MV_ARGS="+verification.multiview_vote=0.6 ...": more gate settings (new keys need "+")
#   jev_sam3: --selector jev_sam3 (JEV-27B-VL int8 + SAM3 workers, perception.jev): JEV only judges the
#            uncertainty (frozen jev_gate.json: naming entropy + look-alike contention; unsure -> approach
#            with the d1mv view bank, multiview_grid=false; unsure views on the way keep approaching);
#            SAM3 decides: its raw label on a sure first frame, and over up to 6 banked views on arrival,
#            failure and sure views on the way (d1mv rule: M >= multiview_arrival/failure_margin,
#            V >= multiview_vote; multiview_min_views=1: one view decides, an empty bank rejects).
#            JEV loads from the pre-quantized local copy (/data/models/JEV-27B-VL-int8), slimmed (answer-token
#            head rows on the GPU, embedding table on CPU), so fp32 SAM3 fits next to it on one GPU.
#            APPROACH_PLANNER=mppi|closer: the gate drives the approach itself (perception_gate.approach:
#            active_approach's footprint MPPI or the straight-line closer on the agent's own local map) with the
#            decide.py flow (JEV naming belief -> P(consistent) >= 0.8 -> fused SAM3 over the nearest views);
#            needs sync true
#            LAST_MILE=0 (with APPROACH_PLANNER): after an accept the gate does not drive the last mile itself;
#            ApexNav's own navigation goes on to the accepted cluster (not forced). Default 1 (last mile on).
# NO_EVIDENCE=accept|reject|defer (any gate strategy): verification.on_no_evidence, the verdict on a cluster without a
#            stored view; defer: ApexNav goes on to it and the stop re-verifies it. Unset: the yaml (accept).
# CLOSE_IN=1 [CLOSE_IN_BUDGET=12] (any gate strategy): verification.close_in, the terminal close-in: when the planner
#            STOPs or ends stucking at an object the gate accepted, up to 12 more actions toward the object's points,
#            then STOP (perception_gate/close_in.py). Off by default; needs LAST_MILE=0 for a clean comparison.
# REUSE_SERVER=1: keep a running perception server with the same selector (and start one if none runs),
#            and leave it running on exit, so a driver running blocks of episodes loads the models once;
#            the driver stops it (lib.sh cleanup).
# DATASET (default hm3dv2) is passed to habitat_evaluation.py --dataset.
# WORKERS=N (default 1): N episode workers in parallel (run_gate_e2e.sh WORKER=1..N, each with its own simulator,
#            planner and ROS_DOMAIN_ID) take the episodes from the shared queue videos/<name>/queue; all share the four
#            VLM servers and the perception server. scripts/run_parallel.sh is the driver.
# Needs the four VLM servers (scripts/start_session.sh); does not start or stop them.
# On any exit it stops the perception server/workers, its gpu_guard and any episode or
# planner still running. Each start appends the code versions to videos/<name>/versions.txt.
set -u
STRAT=${1:?strategy}; NAME=${2:?name}; SYNC=${3:?sync}; shift 3
GPU=${GPU:-0}
WORKERS=${WORKERS:-1}
DATASET=${DATASET:-hm3dv2}
eval_args=
approach=false
case ${LAST_MILE:-1} in   # LAST_MILE=0: no gate last mile after an accept (needs APPROACH_PLANNER)
  1|true) last_mile= ;;
  0|false) last_mile=" +verification.last_mile=false" ;;
  *) echo "LAST_MILE must be 0 or 1"; exit 2 ;;
esac
case ${NO_EVIDENCE:-} in   # NO_EVIDENCE=defer: a cluster without a stored view is re-verified at the stop
  '') no_evidence= ;;
  accept|reject|defer) no_evidence=" verification.on_no_evidence=$NO_EVIDENCE" ;;
  *) echo "NO_EVIDENCE must be accept, reject or defer"; exit 2 ;;
esac
case ${CLOSE_IN:-0} in   # CLOSE_IN=1: terminal close-in at an accepted object (any gate strategy)
  0|false) close_in= ;;
  1|true) close_in=" +verification.close_in=true${CLOSE_IN_BUDGET:+ +verification.close_in_budget=$CLOSE_IN_BUDGET}" ;;
  *) echo "CLOSE_IN must be 0 or 1"; exit 2 ;;
esac
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
  d1mv) mode=gate; grid=${MV_GRID:-true}
        flags="--selector d1 --d1-evidence input${D1_ACCEPT_MIN:+ --d1-accept-min $D1_ACCEPT_MIN}"
        [ "$grid" = true ] && flags="$flags --multiview-grid"
        # the multiview keys are not in the habitat_eval yaml: Hydra adds them with "+"
        eval_args="verification.approach=true verification.on_error=reject +verification.multiview=true +verification.multiview_grid=$grid${MV_VARIANT:+ +verification.multiview_variant=$MV_VARIANT}${MV_ARGS:+ $MV_ARGS}"
        approach=true ;;
  jev_sam3) mode=gate; flags="--selector jev_sam3"
       eval_args="verification.approach=true verification.on_error=reject +verification.multiview=true +verification.multiview_grid=false +verification.multiview_sam3=never +verification.multiview_min_views=1${APPROACH_PLANNER:+ +verification.approach_planner=$APPROACH_PLANNER}$last_mile${MV_ARGS:+ $MV_ARGS}"
       approach=true ;;
  d1) mode=gate; flags="--selector d1 --d1-evidence ${D1_EVIDENCE:-grid}${D1_ACCEPT_MIN:+ --d1-accept-min $D1_ACCEPT_MIN}${D1_STAGGER:+ --stagger}"; eval_args="verification.approach=true verification.on_error=reject"; approach=true ;;
  *) echo "unknown strategy $STRAT"; exit 2 ;;
esac
[ $mode = gate ] && eval_args="$eval_args$no_evidence$close_in"
root=$APEX/videos/$NAME
mkdir -p $root
if [ -f $root/ABORT ]; then echo "$root/ABORT exists"; exit 1; fi
todo=()
for n in "$@"; do [ -f $root/$mode/epi$n/episodes.jsonl ] || todo+=($n); done
if [ ${#todo[@]} -eq 0 ]; then echo "$(date +%T) $NAME: nothing to do"; exit 0; fi
# A killed attempt leaves a directory without episodes.jsonl; the gate appends to its
# decisions.jsonl, so move it aside instead of mixing two attempts
for n in "${todo[@]}"; do
  [ -d $root/$mode/epi$n ] && mv $root/$mode/epi$n $root/$mode/epi$n.partial.$(date +%Y%m%d%H%M%S)
done

guard=; child=
cleanup() {
  # the episode loop runs inside the container and outlives its docker exec client; parallel workers get a TERM by
  # PID first (each then stops its own episode and planner), the pkills below are the backstop
  for f in $root/.worker*.pid; do
    [ -f "$f" ] && docker exec apexnav kill -TERM "$(cat "$f")" 2>/dev/null
    rm -f "$f"
  done
  docker exec apexnav pkill -f run_gate_e2e.sh
  [ -n "$child" ] && kill $child 2>/dev/null
  [ -n "$guard" ] && kill $guard 2>/dev/null
  pkill -f "gpu_guard.sh $GPU" 2>/dev/null
  if [ "${REUSE_SERVER:-0}" != 1 ]; then
    docker exec $DEV pkill -f "perception.server"
    docker exec $DEV pkill -f "perception.worker"
  fi
  docker exec apexnav pkill -f habitat_evaluation.py
  docker exec apexnav pkill -f 'exploration.launch.py|exploration_node|tsp_node'
}
trap cleanup EXIT
trap 'exit 130' INT TERM

{
  echo "== $(date -Is) $STRAT ($mode) dataset=$DATASET sync=$SYNC gpu=$GPU workers=$WORKERS flags='$flags' eval_args='$eval_args'"
  echo "episodes: ${todo[*]}"
  echo "ApexNav $(git -C $APEX rev-parse --abbrev-ref HEAD) $(git -C $APEX rev-parse HEAD)"
  git -C $APEX status --short
  echo "3DAwareVerification $(git -C $AV rev-parse --abbrev-ref HEAD) $(git -C $AV rev-parse HEAD)"
  git -C $AV status --short
  echo "planner $(stat -c '%y' $APEX/build/exploration_manager/exploration_node)"
} >> $root/versions.txt

ping_server() { docker exec $DEV sam3d-env core python -m perception.server ping http://localhost:8770 2>/dev/null; }
selector=$(echo "$flags" | awk '{print $2}')  # flags start with --selector <name>
if [ $mode = gate ] && [ "${REUSE_SERVER:-0}" = 1 ] && ping_server | grep -q "\"selector\": \"$selector\""; then
  echo "$(date +%T) $NAME: reusing the running server ($flags)"
elif [ $mode = gate ]; then
  docker exec $DEV pkill -f "perception.server"; docker exec $DEV pkill -f "perception.worker"; sleep 5
  log=outputs/apexnav_online/server_$NAME
  echo "$(date +%T) $NAME: starting server ($flags)"
  docker exec $DEV mkdir -p /workspace/$log
  docker exec -d ${ALLOC_CONF:+-e PYTORCH_CUDA_ALLOC_CONF=$ALLOC_CONF} $DEV bash -lc "cd /workspace && OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 sam3d-env core python -m perception.server --host 0.0.0.0 --port 8770 --gpu $GPU $flags --log-dir $log >> $log/server.out 2>&1"
  for i in $(seq 90); do  # jev_sam3: JEV int8 from local disk
    ping_server >/dev/null && break
    sleep 10
  done
  ping_server >/dev/null || { echo "$NAME: server did not come up"; exit 1; }
else
  docker exec $DEV pkill -f "perception.server"; docker exec $DEV pkill -f "perception.worker"; sleep 5
  echo "$(date +%T) $NAME: no gate"
fi
# compute apps: 4 VLM servers + 2 perception workers + one per simulator that runs the MPPI approach on the GPU
(cd $APEX && ABORT=$root/ABORT ./scripts/gpu_guard.sh $GPU ${GUARD_MAX_MIB:-45500} ${GUARD_MAX_APPS:-$((7 + WORKERS))} >> $root/gpu_guard.log 2>&1) &
guard=$!
env="DATASET=$DATASET SIM_GPU=$GPU SYNC_STEP=$SYNC APPROACH=$approach OUT_ROOT=videos/$NAME EVAL_ARGS='$eval_args'${MPPI_GPU:+ MPPI_GPU=$MPPI_GPU}"
if [ "$WORKERS" -le 1 ]; then
  docker exec apexnav bash -c "cd /workspace/ApexNav && $env ./scripts/run_gate_e2e.sh $mode ${todo[*]} >> videos/$NAME/run_gate.log 2>&1" &
  child=$!
else
  printf '%s\n' "${todo[@]}" > $root/queue
  rm -f $root/queue.lock $root/.worker*.pid
  for k in $(seq $WORKERS); do
    docker exec apexnav bash -c "cd /workspace/ApexNav && WORKER=$k QUEUE=videos/$NAME/queue $env ./scripts/run_gate_e2e.sh $mode >> videos/$NAME/run_gate.log 2>&1" &
    child="$child $!"
    sleep ${WORKER_STAGGER:-3}   # spread the simulator starts
  done
fi
wait $child  # in the background so a TERM runs cleanup now, not after the run
done_n=0; for n in "${todo[@]}"; do [ -f $root/$mode/epi$n/episodes.jsonl ] && done_n=$((done_n+1)); done
echo "$(date +%T) $NAME: finished $done_n/${#todo[@]} episodes"
[ -f $root/ABORT ] && { echo "$NAME aborted"; exit 1; }
[ $done_n -eq ${#todo[@]} ] || exit 1
