#!/usr/bin/env bash
# Master pipeline for the remaining full-paper experiments. Self-sequences:
#   A. wait for the 4 seed-only bootstrap checkpoints
#   B. #4-proper + #6 MA-specificity: detectability re-run with the bootstrap ensemble
#   C. wait for the augmented corpus
#   D. #7: AC post-training v1 (freeze+coverage) and v2 (freeze+coverage+displacement)
#   E. re-run the headline diagnostics on the AC models:
#      oracle-vs-greedy, probe-vs-model (#3), plan-vs-beh (#1), matched-switching, detectability
set -uo pipefail
cd "$(dirname "$0")"
export SUMOCFG_COLOGNE8=/home/fuisloy/projects/cair/resco/resco_benchmark/environments/cologne8/cologne8.sumocfg
PY=.venv/bin/python
mkdir -p logs results
log(){ echo "[$(date +%H:%M:%S)] $*"; }

BS="L05ar_bs1001 L05ar_bs2002 L05ar_bs3003 L05ar_bs4004"

# ---------- A. wait for bootstrap checkpoints ----------
log "A. waiting for bootstrap checkpoints ..."
for r in $BS; do
  until [ -f "traffic_runs_sumo/$r/weights_epoch_80.pt" ]; do sleep 30; done
  log "   $r ready"
done

# ---------- B. #4-proper + #6 MA-specificity ----------
log "B. detectability re-run: seed-only bootstrap ensemble + MA split (5 seeds)"
$PY diag_detectability.py --primary L05ar --ensemble $BS \
    --seeds 777 101 202 303 404 --n_steps 25 --tag proper \
    > logs/diag_detectability_proper.log 2>&1
log "   -> traffic_data_cologne8/diag_detectability_proper.json"
cp traffic_data_cologne8/diag_detectability_proper*.json results/ 2>/dev/null || true
cp traffic_data_cologne8/diag_detectability_proper*rows.csv results/ 2>/dev/null || true

# ---------- C. wait for augmented corpus ----------
log "C. waiting for augmented corpus ..."
until [ -f "traffic_data_cologne8_aug/train.pt" ]; do sleep 30; done
until ! pgrep -f "generate_augment_sumo" >/dev/null; do sleep 20; done
log "   augmented corpus ready"; tail -3 logs/gen_augment.log

# ---------- D. #7 AC post-training ----------
log "D. #7-v1: freeze encoder + coverage data"
$PY train_multi_sumo_ar.py --data_dir traffic_data_cologne8_aug --level 0.5 \
    --tag L05ar_ac_v1 --rollout_k 4 --epochs 80 --seed 7001 \
    --init_from L05ar --freeze_encoder --displacement_w 0.0 \
    > logs/train_L05ar_ac_v1.log 2>&1
log "   v1 done"; tail -2 logs/train_L05ar_ac_v1.log

log "D. #7-v2: freeze encoder + coverage data + Delta-JEPA displacement loss"
$PY train_multi_sumo_ar.py --data_dir traffic_data_cologne8_aug --level 0.5 \
    --tag L05ar_ac_v2 --rollout_k 4 --epochs 80 --seed 7002 \
    --init_from L05ar --freeze_encoder --displacement_w 1.0 \
    > logs/train_L05ar_ac_v2.log 2>&1
log "   v2 done"; tail -2 logs/train_L05ar_ac_v2.log

# ---------- E. re-run headline diagnostics on the AC models ----------
for M in L05ar L05ar_ac_v1 L05ar_ac_v2; do
  log "E. [$M] closed-loop latent-CEM vs MaxPressure vs random (#7 pivotal number)"
  $PY diag_compounding.py --run "$M" --seeds 777 101 202 303 404 --n_steps 40 \
      > "logs/compounding_${M}.log" 2>&1 || log "   compounding $M FAILED"
  cp traffic_data_cologne8/diag_compounding.json "results/compounding_${M}.json" 2>/dev/null || true
  cp traffic_data_cologne8/diag_compounding_rows.csv "results/compounding_${M}_rows.csv" 2>/dev/null || true

  log "E. [$M] probe-vs-model (#3)"
  $PY diag_probe_vs_model.py --run "$M" --seeds 777 101 202 --n_steps 25 --horizon 5 \
      > "logs/pvm_${M}.log" 2>&1 || log "   pvm $M FAILED"
  cp traffic_data_cologne8/diag_probe_vs_model.json "results/probe_vs_model_${M}.json" 2>/dev/null || true

  log "E. [$M] plan-vs-beh (#1)"
  $PY diag_plan_vs_beh.py --run "$M" --seed 777 --n_anchors 20 \
      > "logs/pvb_${M}.log" 2>&1 || log "   pvb $M FAILED"
  cp traffic_data_cologne8/diag_plan_vs_beh.json "results/plan_vs_beh_${M}.json" 2>/dev/null || true

  log "E. [$M] matched-switching (#1 control)"
  $PY diag_matched_switching.py --run "$M" --seeds 777 101 202 --n_anchors 20 \
      --out "results/matched_switching_${M}.json" > "logs/ms_${M}.log" 2>&1 || log "   ms $M FAILED"

  log "E. [$M] detectability (with bootstrap ensemble)"
  $PY diag_detectability.py --primary "$M" --ensemble $BS \
      --seeds 777 101 202 303 404 --n_steps 25 --tag "$M" \
      > "logs/det_${M}.log" 2>&1 || log "   det $M FAILED"
  cp traffic_data_cologne8/diag_detectability_${M}*.json results/ 2>/dev/null || true
done

log "PIPELINE DONE"
