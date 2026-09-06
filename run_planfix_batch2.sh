#!/bin/bash
# T4 batch 2: legal-phase mask on, then the proposal / action-regime variants on top.
cd "$(dirname "$0")"
PY=.venv/bin/python
COMMON="--run L05ar --seeds 777 101 202 303 404 --n_compare 40 --warmup 10 --oracle --legal_only"
$PY plan_cem_multi.py $COMMON --horizon 5 --tag pf_legal_H5                       > logs/pf_legal_H5.log 2>&1 &
$PY plan_cem_multi.py $COMMON --horizon 5 --hold_prior 0.7 --tag pf_legal_hold_H5  > logs/pf_legal_hold_H5.log 2>&1 &
$PY plan_cem_multi.py $COMMON --horizon 5 --hold_prior 0.7 --warm_start --tag pf_legal_holdwarm_H5 > logs/pf_legal_holdwarm_H5.log 2>&1 &
$PY plan_cem_multi.py $COMMON --horizon 5 --min_hold 2 --tag pf_legal_mh2_H5      > logs/pf_legal_mh2_H5.log 2>&1 &
$PY plan_cem_multi.py $COMMON --horizon 8 --min_hold 4 --tag pf_legal_mh4_H8      > logs/pf_legal_mh4_H8.log 2>&1 &
$PY plan_cem_multi.py $COMMON --horizon 5 --per_agent --tag pf_legal_peragent_H5  > logs/pf_legal_peragent_H5.log 2>&1 &
wait
echo BATCH2_DONE
