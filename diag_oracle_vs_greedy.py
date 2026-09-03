"""Experiment #5: is the search itself sound?  oracle-CEM vs MaxPressure, many seeds.

The load-bearing positive control for Paper 1: "the identical factored CEM search,
with candidates scored by the TRUE simulator instead of the learned model, beats
a strong greedy baseline (MaxPressure)".  If oracle-CEM does NOT reliably beat
MaxPressure, the whole "the planner is fine, the model is the problem" story
collapses.  Addendum 4 had it on 2/3 seeds at cologne8 -- too thin.

Runs oracle-CEM / MaxPressure / fixed-time closed-loop for N seeds (and optionally
several networks), each from a shared MaxPressure warm-up, and reports the tail
cumulative halting + RESCO metrics, the oracle/MP ratio per seed, and a sign test.

Oracle-CEM deliberately gets a SMALLER search budget than the latent planner
(24 samples / 3 iters vs 64 / 4) so a win can't be blamed on extra search.

Usage:
  python diag_oracle_vs_greedy.py --data_dir traffic_data_cologne8 \
      --sumocfg /home/fuisloy/projects/HMARL-TSC/environments/cologne8/cologne8.sumocfg \
      --begin 25200 --seeds 777 101 202 303 404 505 --n_compare 70
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from plan_cem_multi import OracleFactoredCEM, run_episode, HS
from traffic.sumo_multi_env import (
    SumoMultiEnv, controller_max_pressure, controller_fixed_time,
)


def make_oracle_decision(n_nodes, P_max, horizon, S, topk, iters, seed, act_steps=1):
    oc = OracleFactoredCEM(n_nodes, P_max, horizon, S, topk, iters,
                           rng=np.random.default_rng(seed))
    cache = {"queue": []}

    def decision(env, t, states, actions):
        if cache["queue"]:
            return cache["queue"].pop(0)
        plan = oc.plan_env(env, f"_ovg_snap_{os.getpid()}_{seed}.xml", act_steps)
        cache["queue"] = [plan[k] for k in range(1, len(plan))]
        return plan[0]
    return decision


def mp_decision():
    cache = {}
    def d(env, t, states, actions):
        cache.setdefault("p", controller_max_pressure(env))
        return cache["p"](t)
    return d


def ft_decision():
    cache = {}
    def d(env, t, states, actions):
        cache.setdefault("p", controller_fixed_time(env, np.random.default_rng(0)))
        return cache["p"](t)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="traffic_data_cologne8")
    ap.add_argument("--sumocfg",
                    default="/home/fuisloy/projects/HMARL-TSC/environments/cologne8/cologne8.sumocfg")
    ap.add_argument("--begin", type=int, default=25200)
    ap.add_argument("--label", default="cologne8")
    ap.add_argument("--seeds", type=int, nargs="+", default=[777, 101, 202, 303, 404, 505])
    ap.add_argument("--n_compare", type=int, default=70)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--oracle_S", type=int, default=24)
    ap.add_argument("--oracle_topk", type=int, default=6)
    ap.add_argument("--oracle_iters", type=int, default=3)
    args = ap.parse_args()

    import torch
    meta = torch.load(f"{args.data_dir}/val.pt", weights_only=False)
    n_nodes, P_max = meta["n_nodes"], meta["P_max"]
    print(f"{args.label}: N={n_nodes}  oracle budget {args.oracle_S}x{args.oracle_iters} "
          f"(vs latent-CEM 64x4)  n_compare={args.n_compare}  seeds={args.seeds}")

    rows = []
    for seed in args.seeds:
        res = {}
        for name, dfn in (
            ("oracle-CEM", make_oracle_decision(n_nodes, P_max, args.horizon,
                                                args.oracle_S, args.oracle_topk,
                                                args.oracle_iters, seed)),
            ("max_pressure", mp_decision()),
            ("fixed_time", ft_decision()),
        ):
            env = SumoMultiEnv(args.sumocfg, seed=seed, warmup=0, metrics=True, begin=args.begin)
            tail, wall, _, m = run_episode(env, args.n_compare, args.warmup, dfn)
            res[name] = dict(tail=tail, **{k: m[k] for k in
                                          ("duration", "delay", "wait", "queue", "throughput")},
                             ms=wall * 1000)
        r = res["oracle-CEM"]["tail"] / res["max_pressure"]["tail"]
        rows.append((seed, res, r))
        print(f"  seed {seed:>4}: tail  oracle {res['oracle-CEM']['tail']:>7.0f}  "
              f"MP {res['max_pressure']['tail']:>7.0f}  FT {res['fixed_time']['tail']:>7.0f}  "
              f"| oracle/MP {r:.2f}  ({'WIN' if r < 1 else 'lose'})  "
              f"[{res['oracle-CEM']['ms']:.0f} ms/dec]")

    ratios = np.array([r for _, _, r in rows])
    wins = int((ratios < 1).sum())
    # bootstrap CI on the mean ratio
    bs = np.array([np.mean(np.random.choice(ratios, len(ratios), replace=True)) for _ in range(5000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"\n=== oracle-CEM / MaxPressure  (tail halting) ===")
    print(f"  per-seed ratios: {np.round(ratios, 2).tolist()}")
    print(f"  mean {ratios.mean():.2f}  (95% CI {lo:.2f}–{hi:.2f})   wins {wins}/{len(ratios)}")

    # RESCO metrics, mean over seeds
    print(f"\n=== RESCO metrics (mean over {len(rows)} seeds; lower better except throughput) ===")
    print(f"  ⚠️ for oracle-CEM, duration/delay/wait/throughput are tripinfo-contaminated by "
          f"save/restore branch rollouts; only `queue` and `tail` are clean (see Exp Log Add. 4b)")
    print(f"{'controller':>13} {'duration':>9} {'delay':>8} {'wait':>8} {'queue':>7} {'thru':>6}")
    agg = {}
    for name in ("oracle-CEM", "max_pressure", "fixed_time"):
        a = {k: float(np.mean([res[name][k] for _, res, _ in rows]))
             for k in ("duration", "delay", "wait", "queue", "throughput")}
        agg[name] = a
        print(f"{name:>13} {a['duration']:>9.1f} {a['delay']:>8.1f} {a['wait']:>8.1f} "
              f"{a['queue']:>7.1f} {a['throughput']:>6.0f}")

    if wins >= 0.8 * len(ratios) and hi < 1.0:
        verdict = (f"SEARCH IS SOUND: oracle-CEM beats MaxPressure on {wins}/{len(ratios)} seeds, "
                   f"mean ratio {ratios.mean():.2f} (95% CI upper {hi:.2f} < 1). The identical "
                   f"factored search with true dynamics beats greedy -> the planner/cost design "
                   f"is fine; the model is the problem.")
    elif wins >= 0.5 * len(ratios):
        verdict = (f"MIXED: oracle-CEM beats MaxPressure on {wins}/{len(ratios)} seeds, mean "
                   f"{ratios.mean():.2f} (CI {lo:.2f}–{hi:.2f}). Positive control is real but "
                   f"not overwhelming -> report the margin honestly.")
    else:
        verdict = (f"POSITIVE CONTROL FAILS: oracle-CEM only beats MaxPressure on {wins}/{len(ratios)} "
                   f"seeds -> cannot claim 'the search is sound'; the planner/cost design needs work "
                   f"before blaming the model.")
    print(f"\n>>> VERDICT ({args.label}): {verdict}")

    Path(f"{args.data_dir}/diag_oracle_vs_greedy_{args.label}.json").write_text(json.dumps(dict(
        label=args.label, n_nodes=n_nodes, seeds=args.seeds,
        oracle_budget=[args.oracle_S, args.oracle_iters],
        ratios=ratios.tolist(), mean_ratio=float(ratios.mean()),
        ci=[float(lo), float(hi)], wins=wins, n=len(ratios),
        resco=agg, verdict=verdict), indent=2))
    print(f"wrote {args.data_dir}/diag_oracle_vs_greedy_{args.label}.json")


if __name__ == "__main__":
    main()
