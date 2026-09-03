#!/bin/bash
set -e
cd /home/fuisloy/projects/cair/le-wm
source .venv/bin/activate
echo "### JOB A: switch-penalty (Step 0.5) ###"
python diag_switch_penalty.py --run L05ar --n_anchors 15 --repeats 5 --seed 777
echo "### JOB A also on L05ar_v2 (coverage-trained) for comparison ###"
python diag_switch_penalty.py --run L05ar_v2 --data_dir traffic_data_cologne8_v2 --n_anchors 15 --repeats 5 --seed 777
cp traffic_data_cologne8_v2/diag_switch_penalty.json traffic_data_cologne8_v2/diag_switch_penalty_v2.json
echo "### JOB A DONE ###"
