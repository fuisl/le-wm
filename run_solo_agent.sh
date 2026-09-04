#!/usr/bin/env bash
# #6(a) MA-specificity control: single-agent large-action-space planning.
# One intersection plans its H-step phase sequence (space P^H ~= the 8-agent
# per-step joint space); all others hold. Re-run the #4-proper detectability
# analysis. If per-plan error is detectable HERE but not in the full joint
# search (#4-proper), the FACTORED JOINT structure is what defeats detection.
set -uo pipefail
cd "$(dirname "$0")"
export SUMOCFG_COLOGNE8=/home/fuisloy/projects/cair/resco/resco_benchmark/environments/cologne8/cologne8.sumocfg
PY=.venv/bin/python
BS="L05ar_bs1001 L05ar_bs2002 L05ar_bs3003 L05ar_bs4004"
mkdir -p logs results
for A in 0 3 7; do          # the three 4-phase intersections in cologne8
  echo "=== solo_agent $A ==="
  $PY diag_detectability.py --primary L05ar --ensemble $BS \
      --seeds 777 101 202 303 404 --n_steps 25 --solo_agent "$A" --tag "solo$A" \
      > "logs/det_solo$A.log" 2>&1 || echo "solo$A FAILED"
  cp traffic_data_cologne8/diag_detectability_solo$A.json results/ 2>/dev/null || true
done
echo "SOLO-AGENT CONTROL DONE"
grep -h "best\|NEGATIVE\|DETECTABILITY SIGNAL" logs/det_solo*.log
