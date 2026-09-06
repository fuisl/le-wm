#!/bin/bash
# T5-lite: same models, same latents, non-linear (MLP) read-out instead of ridge.
cd "$(dirname "$0")"
PY=.venv/bin/python
export PROBE=mlp
for spec in "L05ar:traffic_data_cologne8" "L05ar_link:traffic_data_cologne8_link" "L05ar_raster:traffic_data_cologne8_raster"; do
  TAG=${spec%%:*}; DD=${spec##*:}
  case $TAG in *_link) export SUMO_OBS_MODE=link;; *_raster) export SUMO_OBS_MODE=raster;; *) export SUMO_OBS_MODE=base;; esac
  export DATA_DIR=$DD
  echo "=== mlp-probe $TAG  $(date)"
  $PY diag_probe_vs_model.py --run $TAG --seeds 777 101 202 --n_steps 25 > logs/pvm_mlp_$TAG.log 2>&1
  cp $DD/diag_probe_vs_model.json results/probe_vs_model_mlp_$TAG.json
  $PY diag_compounding.py --run $TAG --data_dir $DD --seeds 777 101 202 303 404 --n_steps 40 > logs/compounding_mlp_$TAG.log 2>&1
  cp $DD/diag_compounding.json results/compounding_mlp_$TAG.json
  echo "=== done mlp-probe $TAG  $(date)"
done
echo MLPPROBE_DONE
