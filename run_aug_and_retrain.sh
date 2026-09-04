#!/usr/bin/env bash
# #7: build the planner-matched corpus, then run 3 AC post-training variants:
#   v1  freeze encoder + coverage data (no displacement loss)
#   v2  freeze encoder + coverage data + Delta-JEPA displacement loss
#   v3  full fine-tune (encoder unfrozen) + coverage data + displacement loss
set -euo pipefail
cd "$(dirname "$0")"
export SUMOCFG_COLOGNE8=/home/fuisloy/projects/cair/resco/resco_benchmark/environments/cologne8/cologne8.sumocfg
PY=.venv/bin/python
mkdir -p logs

echo "=== [1/4] generating augmented corpus ==="
$PY -m traffic.generate_augment_sumo --out_dir traffic_data_cologne8_aug \
    --base_dir traffic_data_cologne8 --n_random_joint 6 --n_eps 6 --n_cem 3 \
    --cem_model L05ar --T 300 --cem_T 150 > logs/gen_augment.log 2>&1
tail -4 logs/gen_augment.log

echo "=== [2/4] #7-v1: freeze encoder + coverage data ==="
$PY train_multi_sumo_ar.py --data_dir traffic_data_cologne8_aug --level 0.5 \
    --tag L05ar_ac_v1 --rollout_k 4 --epochs 80 --seed 7001 \
    --init_from L05ar --freeze_encoder --displacement_w 0.0 \
    > logs/train_L05ar_ac_v1.log 2>&1
tail -3 logs/train_L05ar_ac_v1.log

echo "=== [3/4] #7-v2: freeze encoder + coverage data + displacement loss ==="
$PY train_multi_sumo_ar.py --data_dir traffic_data_cologne8_aug --level 0.5 \
    --tag L05ar_ac_v2 --rollout_k 4 --epochs 80 --seed 7002 \
    --init_from L05ar --freeze_encoder --displacement_w 1.0 \
    > logs/train_L05ar_ac_v2.log 2>&1
tail -3 logs/train_L05ar_ac_v2.log

echo "=== [4/4] #7-v3: full fine-tune + coverage data + displacement loss ==="
$PY train_multi_sumo_ar.py --data_dir traffic_data_cologne8_aug --level 0.5 \
    --tag L05ar_ac_v3 --rollout_k 4 --epochs 80 --seed 7003 \
    --init_from L05ar --displacement_w 1.0 \
    > logs/train_L05ar_ac_v3.log 2>&1
tail -3 logs/train_L05ar_ac_v3.log

echo "#7 RETRAIN PIPELINE DONE"
