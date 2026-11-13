#!/usr/bin/env bash
# In the apexnav container: signal only the processes of ONE run (OUT_ROOT=videos/<name> in their environment),
# so a second run in the same container is left alone.   ./scripts/kill_run.sh videos/<name> [signal, default TERM]
OUT=${1:?usage: kill_run.sh videos/<name> [signal]}; SIG=${2:-TERM}
for p in $(pgrep -f 'run_gate_e2e.sh|habitat_evaluation.py|exploration.launch.py|exploration_node|tsp_node'); do
  [ "$p" = "$$" ] && continue
  tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx "OUT_ROOT=$OUT" && kill -$SIG $p 2>/dev/null
done
exit 0
