#!/usr/bin/env bash
# #4-proper: K=4 seed-only bootstrap ensemble.
# Identical to L05ar (level 0.5, rollout_k 4, 80 epochs, lr 3e-4) except --seed.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
mkdir -p logs
for S in 1001 2002 3003 4004; do
  echo "=== training L05ar_bs$S (seed $S) ==="
  $PY train_multi_sumo_ar.py --data_dir traffic_data_cologne8 --level 0.5 \
      --tag "L05ar_bs$S" --rollout_k 4 --epochs 80 --seed "$S" \
      > "logs/train_L05ar_bs$S.log" 2>&1
done
echo "ALL BOOTSTRAP RUNS DONE"
