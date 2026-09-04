"""Consolidate results/*.json from the full-paper experiment pipeline into one
readable table. Run after run_pipeline.sh finishes (or any time for partial results).

    python summarize_results.py
"""
import glob
import json
import os

R = "results"


def load(p):
    try:
        return json.load(open(p))
    except Exception as e:
        return {"_error": str(e)}


def sec(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


# ---------- horizon sweep ----------
sec("Horizon sweep  (probe-vs-model rank agreement vs effective horizon)")
rows = []
for f in sorted(glob.glob(f"{R}/horizon_sweep/H*.json")):
    d = load(f)
    if "rank_A" not in d:
        continue
    Hrep = os.path.basename(f)[1:-5]
    eff = int(Hrep) - 1 if Hrep.isdigit() else "?"
    rows.append((eff, d["rank_A"], d["rank_B"], d.get("top1_B", float("nan")),
                 d.get("ratio_BA", float("nan"))))
print(f"{'effH':>5} {'probe rank':>11} {'model rank':>11} {'model top1':>11} {'errB/errA':>10}")
for eff, a, b, t1, rba in sorted(rows):
    print(f"{eff:>5} {a:>+11.2f} {b:>+11.2f} {t1:>11.2f} {rba:>10.2f}")

# ---------- matched switching ----------
sec("Matched-switching control  (#1 de-confound: OOD-combination vs switching rate)")
for f in sorted(glob.glob(f"{R}/*matched_switching*.json")):
    d = load(f)
    if "mean_lat_ratio" not in d:
        continue
    print(f"\n[{os.path.basename(f)}]  run={d.get('run')} seeds={d.get('seeds')}")
    print(f"  {'group':>20} {'lat-err ratio':>14} {'joint switch rate':>18}")
    for g in ["beh", "plan", "rand_matched", "plan_downsampled"]:
        if g in d["mean_lat_ratio"]:
            print(f"  {g:>20} {d['mean_lat_ratio'][g]:>14.2f} {d['joint_switch_rate'][g]:>18.3f}")
    print(f"  switch_share={d.get('switch_share', float('nan')):.2f}  "
          f"planner_share={d.get('planner_share', float('nan')):.2f}")
    print(f"  verdict: {d.get('verdict','')[:300]}")

# ---------- detectability (#4-proper and per-model) ----------
sec("Detectability  (#4-proper: within-step Spearman of runtime signals vs per-plan regret)")
for f in sorted(glob.glob(f"{R}/diag_detectability_*.json")):
    d = load(f)
    if "signals" not in d:
        continue
    print(f"\n[{os.path.basename(f)}]  primary={d.get('primary')} "
          f"ensemble={d.get('ensemble')} seeds={d.get('seeds')} "
          f"decision_steps={d.get('n_decision_steps','?')}")
    print(f"  {'signal':>14} {'within r(regret)':>16} {'95% CI':>20}")
    for k, v in d["signals"].items():
        ci = v.get("within_spearman_regret_ci", [float('nan')] * 2)
        print(f"  {k:>14} {v['within_spearman_regret']:>+16.3f}   [{ci[0]:+.2f}, {ci[1]:+.2f}]")
    ma = d.get("ma_specificity")
    if ma:
        print(f"  -- MA-specificity: cov_agent {ma['cov_agent_r']:+.3f} "
              f"{tuple(round(x,2) for x in ma['cov_agent_ci'])}  |  "
              f"cov_joint {ma['cov_joint_r']:+.3f} {tuple(round(x,2) for x in ma['cov_joint_ci'])}")
        print(f"     joint-agent diff {ma['joint_minus_agent']:+.3f} "
              f"{tuple(round(x,2) for x in ma['joint_minus_agent_ci'])}")
        print(f"     {ma['verdict'][:280]}")
    print(f"  verdict: {d.get('verdict','')[:240]}")

# ---------- compounding / closed-loop (#7 pivotal) ----------
sec("Closed-loop latent-CEM vs MaxPressure  (#7 pivotal number)")
for f in sorted(glob.glob(f"{R}/compounding_*.json")):
    d = load(f)
    print(f"\n[{os.path.basename(f)}]")
    for k in ("run", "tail_latent", "tail_random", "tail_mp",
              "ratio_latent_mp", "ratio_latent_random", "regret_model_over_random",
              "compounding_pct", "verdict"):
        if k in d:
            v = d[k]
            print(f"  {k:>24}: {v if not isinstance(v,float) else round(v,3)}")

# ---------- probe-vs-model / plan-vs-beh on AC models ----------
sec("AC-model re-runs  (probe-vs-model #3, plan-vs-beh #1)")
for f in sorted(glob.glob(f"{R}/probe_vs_model_*.json")) + sorted(glob.glob(f"{R}/plan_vs_beh_*.json")):
    d = load(f)
    print(f"\n[{os.path.basename(f)}]")
    for k in ("run", "rank_A", "rank_B", "ratio_BA", "top1_A", "top1_B",
              "mean_latent_ratio", "ratio_growth", "cov_beh", "cov_plan", "verdict"):
        if k in d:
            v = d[k]
            print(f"  {k:>20}: {v if not isinstance(v,float) else round(v,3)}")


print()
