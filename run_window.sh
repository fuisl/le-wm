#!/bin/bash
# T7 window-length test: same corpus/model as L05ar, predictor context 6 and 12 steps.
cd "$(dirname "$0")"
PY=.venv/bin/python
until grep -q OBS_MODELS_DONE logs/obs_models.log 2>/dev/null; do sleep 60; done
for H in 6 12; do
  export WM_HISTORY=$H
  TAG=L05ar_h$H
  echo "=== train $TAG  $(date)"
  $PY train_multi_sumo_ar.py --data_dir traffic_data_cologne8 --level 0.5 --rollout_k 4 --epochs 80 --history $H --tag $TAG > logs/train_$TAG.log 2>&1
  echo "=== diag $TAG  $(date)"
  $PY diag_probe_vs_model.py --run $TAG --seeds 777 101 202 --n_steps 25 > logs/pvm_$TAG.log 2>&1
  cp traffic_data_cologne8/diag_probe_vs_model.json results/probe_vs_model_$TAG.json
  $PY diag_compounding.py --run $TAG --seeds 777 101 202 303 404 --n_steps 40 > logs/compounding_$TAG.log 2>&1
  cp traffic_data_cologne8/diag_compounding.json results/compounding_$TAG.json
  $PY diag_matched_switching.py --run $TAG --seeds 777 101 202 --out results/matched_switching_$TAG.json > logs/ms_$TAG.log 2>&1
  echo "=== done $TAG  $(date)"
done
echo WINDOW_DONE
