#!/bin/bash
# T4 planning-fix study, batch 1 (L05ar, 5 seeds, 40 decision steps, oracle arm on)
cd "$(dirname "$0")"
PY=.venv/bin/python
COMMON="--run L05ar --seeds 777 101 202 303 404 --n_compare 40 --warmup 10 --oracle"
$PY plan_cem_multi.py $COMMON --horizon 5 --tag pf_base_H5            > logs/pf_base_H5.log 2>&1 &
$PY plan_cem_multi.py $COMMON --horizon 5 --hold_prior 0.7 --tag pf_hold_H5 > logs/pf_hold_H5.log 2>&1 &
$PY plan_cem_multi.py $COMMON --horizon 5 --hold_prior 0.7 --warm_start --tag pf_holdwarm_H5 > logs/pf_holdwarm_H5.log 2>&1 &
$PY plan_cem_multi.py $COMMON --horizon 5 --min_hold 2 --tag pf_mh2_H5   > logs/pf_mh2_H5.log 2>&1 &
wait
echo BATCH1_DONE
