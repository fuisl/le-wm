#!/bin/bash
# T2/T3: train richer-observation world models and run the core diagnostics.
# Each model sees ONLY its observation mode (SUMO_OBS_MODE) end-to-end: corpus,
# probe fit, closed-loop env. Everything else identical to L05ar.
cd "$(dirname "$0")"
PY=.venv/bin/python
for M in link raster; do
  export SUMO_OBS_MODE=$M DATA_DIR=traffic_data_cologne8_$M
  TAG=L05ar_$M
  echo "=== train $TAG  $(date)"
  $PY train_multi_sumo_ar.py --data_dir $DATA_DIR --level 0.5 --rollout_k 4 --epochs 80 --tag $TAG > logs/train_$TAG.log 2>&1
  echo "=== diag $TAG  $(date)"
  $PY diag_probe_vs_model.py --run $TAG --seeds 777 101 202 --n_steps 25 > logs/pvm_$TAG.log 2>&1
  cp $DATA_DIR/diag_probe_vs_model.json results/probe_vs_model_$TAG.json
  $PY diag_compounding.py --run $TAG --data_dir $DATA_DIR --seeds 777 101 202 303 404 --n_steps 40 > logs/compounding_$TAG.log 2>&1
  cp $DATA_DIR/diag_compounding.json results/compounding_$TAG.json; cp $DATA_DIR/diag_compounding_rows.csv results/compounding_${TAG}_rows.csv
  $PY diag_plan_vs_beh.py --run $TAG --data_dir $DATA_DIR --seed 777 > logs/pvb_$TAG.log 2>&1
  cp $DATA_DIR/diag_plan_vs_beh.json results/plan_vs_beh_$TAG.json
  $PY diag_matched_switching.py --run $TAG --data_dir $DATA_DIR --seeds 777 101 202 --out results/matched_switching_$TAG.json > logs/ms_$TAG.log 2>&1
  echo "=== done $TAG  $(date)"
done
echo OBS_MODELS_DONE
