#!/usr/bin/env bash
# Horizon-sensitivity sweep: does the model's plan RANKING survive short rollouts?
# Re-runs experiment #3 (probe-vs-model) at H = 1,2,3,5,8.
set -euo pipefail
cd "$(dirname "$0")"
export SUMOCFG_COLOGNE8=/home/fuisloy/projects/cair/resco/resco_benchmark/environments/cologne8/cologne8.sumocfg
PY=.venv/bin/python
mkdir -p logs results/horizon_sweep
# --iters 2: keep candidate diversity comparable across H (full 4-iter CEM collapses
# the elite at short H). NOTE: cem_rollout's first predicted step is action-independent
# by construction, so H here = 1 + (effective action-conditioned horizon). H>=2 needed
# for any candidate signal; we sweep H in {2,3,4,6,9} = effective {1,2,3,5,8}.
for H in 2 3 4 6 9; do
  echo "=== H=$H (effective $((H-1))) ==="
  $PY diag_probe_vs_model.py --run L05ar --seeds 777 101 --n_steps 25 --horizon "$H" \
      --iters 2 --S 96 \
      > "logs/horizon_H${H}.log" 2>&1
  cp traffic_data_cologne8/diag_probe_vs_model.json      "results/horizon_sweep/H${H}.json"
  cp traffic_data_cologne8/diag_probe_vs_model_rows.csv  "results/horizon_sweep/H${H}_rows.csv"
done
echo "HORIZON SWEEP DONE"
grep -H "rank_\|ratio_BA" results/horizon_sweep/H*.json
