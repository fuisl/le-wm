#!/bin/bash
set -e
cd /home/fuisloy/projects/cair/le-wm
source .venv/bin/activate
echo "### JOB B: exploitation diag with L05ar_v2 (trained WITH random_joint coverage) ###"
for s in 777 101 202; do
  echo "--- L05ar_v2 seed $s ---"
  python diag_exploitation_coverage.py --run L05ar_v2 --data_dir traffic_data_cologne8_v2 \
    --seed $s --n_steps 30 --num_samples 64
  cp traffic_data_cologne8_v2/diag_exploitation_coverage_rows.csv traffic_data_cologne8_v2/diag_rows_v2_$s.csv
  cp traffic_data_cologne8_v2/diag_exploitation_coverage.json traffic_data_cologne8_v2/diag_v2_$s.json
done
echo "### JOB B DONE ###"
