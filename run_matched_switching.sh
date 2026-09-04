#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export SUMOCFG_COLOGNE8=/home/fuisloy/projects/cair/resco/resco_benchmark/environments/cologne8/cologne8.sumocfg
mkdir -p logs results
.venv/bin/python diag_matched_switching.py --run L05ar --seeds 777 101 202 \
    --n_anchors 20 --out results/diag_matched_switching_L05ar.json \
    > logs/matched_switching_L05ar.log 2>&1
echo "MATCHED-SWITCHING DONE"
tail -25 logs/matched_switching_L05ar.log
