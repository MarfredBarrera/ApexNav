#!/usr/bin/env bash
# Host-side guard for scripts/run_gate_e2e.sh on a shared GPU: stops the run when GPU memory
# nears the limit or another user's process appears on the GPU.
#
#   ./scripts/gpu_guard.sh <gpu index> [max MiB, default 44000] [max processes, default 8]
#
# On trip it touches videos/gate_e2e/ABORT and kills the running episode. Ctrl-C to stop.
GPU=${1:?usage: gpu_guard.sh <gpu index> [max MiB] [max processes]}
MAX_MIB=${2:-44000}
MAX_APPS=${3:-8}
ABORT=${ABORT:-$(cd "$(dirname "$0")/.." && pwd)/videos/gate_e2e/ABORT}
UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU")
while true; do
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
  apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$UUID")
  echo "$(date +%T) gpu$GPU mem=${mem}MiB apps=$apps load=$(cut -d' ' -f1 /proc/loadavg)"
  if [ "$mem" -gt "$MAX_MIB" ] || [ "$apps" -gt "$MAX_APPS" ]; then
    echo "$(date +%T) GUARD TRIPPED"
    mkdir -p "$(dirname "$ABORT")" && touch "$ABORT"
    docker exec apexnav pkill -f habitat_evaluation.py
    exit 1
  fi
  sleep 5
done
