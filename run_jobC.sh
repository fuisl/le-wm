#!/bin/bash
set -e
cd /home/fuisloy/projects/cair/le-wm
source .venv/bin/activate
echo "### JOB C: exploitation diag on ingolstadt21 (L05ar cologne8-trained, ZERO-SHOT) ###"
CFG=/home/fuisloy/projects/HMARL-TSC/environments/ingolstadt21/ingolstadt21.sumocfg
for s in 777 101; do
  echo "--- ingolstadt21 seed $s ---"
  python diag_exploitation_coverage.py --run L05ar --data_dir traffic_data_ingolstadt21 \
    --sumocfg $CFG --begin 57600 --seed $s --n_steps 20 --num_samples 64
  cp traffic_data_ingolstadt21/diag_exploitation_coverage_rows.csv traffic_data_ingolstadt21/diag_rows_ing_$s.csv
  cp traffic_data_ingolstadt21/diag_exploitation_coverage.json traffic_data_ingolstadt21/diag_ing_$s.json
done
echo "### JOB C DONE ###"
