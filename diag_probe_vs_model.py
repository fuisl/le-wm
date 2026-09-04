"""Experiment #3: is the control gap the world MODEL, or the linear PROBE?

The plan cost is `probe-decoded halting summed over the rollout`. Oracle-CEM
removes the probe entirely (true simulator cost) but does not separate
model-dynamics error from probe-readout error. This 3-way score does.

For one candidate set per decision step, score every plan three ways:
  A  probe-only  : encode the TRUE SUMO rollout states, decode via the probe,
                   sum halting.  Error(A) vs true = pure probe error.
  B  model+probe : roll the model H steps, decode via the SAME probe, sum halting.
                   Error(B) vs true = model+probe error (what latent-CEM uses).
  C  true        : the real simulator halting (oracle).

Decomposition per plan:  err_A = |A - C|      (probe)
                         err_B = |B - C|      (model + probe)
                         model_contribution = err_B - err_A
Also: rank agreement of each scorer's plan ordering with the true ordering
(Spearman), and top-1 pick match.

Read:  err_B >> err_A  and rank(B) << rank(A)  -> the MODEL is the problem, not
       the probe.  err_B ~ err_A                -> the probe is carrying the blame.

Usage:  python diag_probe_vs_model.py --run L05ar --seeds 777 101 --n_steps 25
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from eval_multi_sumo import load, encode_batch
from visualize_rollout_cologne8 import fit_pressure_probe, decode
from plan_cem_multi import encode_context, cem_rollout, SUMOCFG, HS
from traffic.sumo_multi_env import SumoMultiEnv, controller_max_pressure

DATA_DIR = "traffic_data_cologne8"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="L05ar")
    ap.add_argument("--weights", default="weights_epoch_80.pt")
    ap.add_argument("--seeds", type=int, nargs="+", default=[777, 101])
    ap.add_argument("--n_steps", type=int, default=25)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--S", type=int, default=64)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--topk", type=int, default=8)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = torch.load(f"{DATA_DIR}/val.pt", weights_only=False)
    n, P, F = meta["n_nodes"], meta["P_max"], meta["node_feature_dim"]
    ni = torch.from_numpy(np.asarray(meta["neighbor_idx"]))
    nm = torch.from_numpy(np.asarray(meta["neighbor_mask"]))
    H, S = args.horizon, args.S

    model, _ = load(args.run, args.weights, device)
    pr = fit_pressure_probe(model, f"{DATA_DIR}/train.pt", n, P, F, ni, nm, device)
    print(f"model {args.run}  probe MSE {pr['mse']:.1f}")

    rows = []
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        env = SumoMultiEnv(SUMOCFG, seed=seed, warmup=0, metrics=False, begin=25200)
        env.reset()
        mp = controller_max_pressure(env)
        states, actions = [env.state()], []
        for t in range(args.warmup):
            ph = mp(t)
            actions.append(env.encode_action(np.clip(ph, 0, P - 1)))
            states.append(env.step(ph))

        for step in range(args.n_steps):
            sh = np.stack(states[-HS:]).astype(np.float32)
            ah = np.stack(actions[-HS:]).astype(np.float32)
            z_ctx, a_ctx = encode_context(model, sh, ah, n, ni, nm, device)

            # CEM
            probs = np.full((n, H, P), 1.0 / P)
            phases = None
            for _ in range(args.iters):
                phases = np.empty((S, n, H), np.int64)
                for i in range(n):
                    for h in range(H):
                        phases[:, i, h] = rng.choice(P, size=S, p=probs[i, h])
                pe = cem_rollout(model, z_ctx, a_ctx, phases, n, P, ni, nm, device)
                mc = decode(pr, pe).clamp(min=0).sum(dim=(1, 2, 3)).cpu().numpy()
                elite = phases[np.argsort(mc)[: args.topk]]
                for i in range(n):
                    for h in range(H):
                        c = np.bincount(elite[:, i, h], minlength=P)
                        probs[i, h] = (c + 1e-3) / (c.sum() + P * 1e-3)

            # B: model + probe
            pe = cem_rollout(model, z_ctx, a_ctx, phases, n, P, ni, nm, device)   # (S,N,H,d)
            costB = decode(pr, pe).clamp(min=0).sum(dim=(1, 2, 3)).cpu().numpy()

            # A: probe-only (encode the TRUE rollout) and C: true simulator
            snap = env.save_state(f"_pvm_snap_{os.getpid()}_{seed}.xml")
            true_states = np.zeros((S, HS + H, n * F), np.float32)
            true_states[:, :HS] = sh[None]
            costC = np.zeros(S)
            for s in range(S):
                env.load_state(snap)
                tot = 0.0
                for h in range(H):
                    st = env.step(phases[s, :, h])
                    true_states[s, HS + h] = st
                    tot += float(st.reshape(n, F)[:, :P].sum())
                costC[s] = tot
            env.load_state(snap)
            with torch.no_grad():
                o = encode_batch(model, torch.from_numpy(true_states).float(),
                                 torch.zeros(S, HS + H, n * P), n, ni, nm, device)
            zt = o["emb"].reshape(S, n, HS + H, -1)[:, :, HS:, :]                 # (S,N,H,d)
            costA = decode(pr, zt).clamp(min=0).sum(dim=(1, 2, 3)).cpu().numpy()

            errA = np.abs(costA - costC)
            errB = np.abs(costB - costC)

            def _sp(x, y):
                # spearman is undefined if either side is (near-)constant, e.g. when
                # a short-horizon CEM collapses the candidate set. Report nan then.
                if np.unique(np.round(x, 6)).size < 3 or np.unique(np.round(y, 6)).size < 3:
                    return np.nan
                return spearmanr(x, y)[0]

            spA = _sp(costA, costC)
            spB = _sp(costB, costC)
            top1A = int(np.argmin(costA) == np.argmin(costC))
            top1B = int(np.argmin(costB) == np.argmin(costC))
            for s in range(S):
                rows.append(dict(seed=seed, step=step, cand=s,
                                 costA=float(costA[s]), costB=float(costB[s]), costC=float(costC[s]),
                                 errA=float(errA[s]), errB=float(errB[s])))
            rows[-1].update(dict(spearmanA=float(spA), spearmanB=float(spB),
                                 top1A=top1A, top1B=top1B))  # attach step-level to last row
            print(f"  seed {seed} step {step:>2}  errA {errA.mean():6.1f}  errB {errB.mean():6.1f}  "
                  f"| rank(true): probe {spA:+.2f}  model+probe {spB:+.2f}")
            exec_phase = probs[:, 0].argmax(1)
            actions.append(env.encode_action(np.clip(exec_phase, 0, P - 1)))
            states.append(env.step(exec_phase))
        env.close()

    # aggregate
    A = np.array([r["errA"] for r in rows]); B = np.array([r["errB"] for r in rows])
    steps = [(r["seed"], r["step"]) for r in rows]
    spA = [r["spearmanA"] for r in rows if "spearmanA" in r]
    spB = [r["spearmanB"] for r in rows if "spearmanB" in r]
    t1A = [r["top1A"] for r in rows if "top1A" in r]
    t1B = [r["top1B"] for r in rows if "top1B" in r]
    print(f"\n=== probe-only (A) vs model+probe (B), {len(A)} plans over {len(spA)} decision steps ===")
    print(f"  mean |cost - true|        A (probe)  {A.mean():7.1f}     B (model+probe) {B.mean():7.1f}     "
          f"B/A {B.mean()/A.mean():.2f}")
    print(f"  model's own contribution  (errB - errA) mean {np.mean(B - A):7.1f}  "
          f"({100*np.mean(B - A)/B.mean():.0f}% of total model+probe error)")
    n_valid = int(np.sum(~np.isnan(spB)))
    print(f"  rank agreement w/ true    A {np.nanmean(spA):+.2f}     B {np.nanmean(spB):+.2f}   "
          f"({n_valid}/{len(spB)} steps with a well-defined model ranking)")
    print(f"  top-1 pick matches true   A {np.mean(t1A):.2f}     B {np.mean(t1B):.2f}   (chance 1/{S})")

    frac_model = float(np.mean(B - A) / B.mean())
    rA, rB = float(np.nanmean(spA)), float(np.nanmean(spB))
    # CEM only needs the RANKING right, so rank agreement is the decisive metric.
    if rA > 0.4 and rB < rA - 0.3:
        verdict = (f"the MODEL is the problem, NOT the probe: given TRUE dynamics the probe "
                   f"ranks plans well (Spearman {rA:+.2f}); the model DESTROYS the ranking "
                   f"({rA:+.2f} -> {rB:+.2f}). The probe's absolute error is larger ({A.mean():.0f} "
                   f"vs {B.mean():.0f}) but rank-preserving; the model's error is rank-destroying.")
    elif rB >= rA - 0.15:
        verdict = ("the model does NOT clearly destroy the ranking -> the control failure "
                   "may be more about the probe / cost design than the world model")
    else:
        verdict = (f"model degrades ranking ({rA:+.2f} -> {rB:+.2f}) but the probe alone is also "
                   f"weak ({rA:+.2f}) -> both contribute; report the split")
    print(f"\n>>> VERDICT: {verdict}")

    Path(f"{DATA_DIR}/diag_probe_vs_model.json").write_text(json.dumps(dict(
        run=args.run, seeds=args.seeds, n_steps=args.n_steps, n_plans=len(A),
        mean_errA=float(A.mean()), mean_errB=float(B.mean()), ratio_BA=float(B.mean() / A.mean()),
        model_contribution_frac=frac_model,
        rank_A=float(np.nanmean(spA)), rank_B=float(np.nanmean(spB)),
        rank_B_valid_steps=int(np.sum(~np.isnan(np.array(spB, dtype=float)))),
        rank_B_total_steps=len(spB),
        horizon=args.horizon,
        top1_A=float(np.mean(t1A)), top1_B=float(np.mean(t1B)),
        verdict=verdict), indent=2))
    import csv
    with open(f"{DATA_DIR}/diag_probe_vs_model_rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "step", "cand", "costA", "costB", "costC", "errA", "errB"])
        w.writeheader()
        w.writerows([{k: r[k] for k in w.fieldnames} for r in rows])
    print(f"wrote {DATA_DIR}/diag_probe_vs_model.json (+ _rows.csv)")


if __name__ == "__main__":
    main()
