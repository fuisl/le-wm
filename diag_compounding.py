"""Experiment #2 (Exploitation Experiment Plan): random-pick baseline + compounding.

Two questions:
 (a) Is latent-CEM's plan pick actually WORSE than picking a random candidate?
     If not, "exploitation" (adversarial inversion) is the wrong word - it's a
     weakly-informative model.
 (b) Does the per-decision regret GROW over a closed-loop episode, tracking drift
     of the trajectory into OOD latent regions? If yes, the story is "mild
     per-decision optimism bias that COMPOUNDS under closed-loop feedback".

Method: run latent-CEM closed-loop for N steps. At every step, from the frozen
pre-step SUMO state, score S sampled candidates with the REAL simulator (oracle),
and record:
  regret_model  = oracle_cost(model's pick)   - min_c oracle_cost(c)
  regret_random = mean_c oracle_cost(c)        - min_c oracle_cost(c)
  cov_now       = kNN dist of the current context latent to the training bank
  eps_plan_now  = latent rollout error of the model on its own chosen plan
Then regress regret_model and cov_now on the closed-loop step index.
Also run MaxPressure and random-candidate closed-loop for the tail-cost picture.

Usage:  python diag_compounding.py --run L05ar --n_steps 40 --seeds 777 101 202
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from eval_multi_sumo import load, encode_batch
from visualize_rollout_cologne8 import fit_pressure_probe, decode
from plan_cem_multi import encode_context, cem_rollout, SUMOCFG, HS
from diag_exploitation_coverage import build_coverage_bank, knn_dist
from traffic.sumo_multi_env import SumoMultiEnv, controller_max_pressure


def cov_of_context(z_last, phase, Zb_by_phase, MU, SD, k=8):
    """z_last (N,d) current context latent, phase (N,) -> mean kNN dist."""
    zt = ((z_last.cpu() - MU) / SD)
    d = torch.full((z_last.shape[0],), float("nan"))
    for p, Zb in Zb_by_phase.items():
        m = torch.from_numpy(phase) == p
        if m.any():
            d[m] = knn_dist(zt[m], Zb, k=k)
    return float(d.nanmean())


def run_seed(args, model, pr, ni, nm, n_nodes, P_max, F, Zb_by_phase, MU, SD, device, seed):
    H, S, ITERS, TOPK = args.horizon, args.cem_S, args.cem_iters, args.topk
    rng = np.random.default_rng(seed)
    env = SumoMultiEnv(args.sumocfg, seed=seed, warmup=0, metrics=False, begin=args.begin)
    env.reset()
    mp = controller_max_pressure(env)
    states, actions = [env.state()], []
    for t in range(args.warmup):
        ph = mp(t)
        actions.append(env.encode_action(np.clip(ph, 0, P_max - 1)))
        states.append(env.step(ph))

    rows = []
    tail_latent = tail_random = tail_mp = 0.0
    # snapshot to fork MaxPressure / random tails from the SAME warmed state
    for step in range(args.n_steps):
        sh = np.stack(states[-HS:]).astype(np.float32)
        ah = np.stack(actions[-HS:]).astype(np.float32)
        z_ctx, a_ctx = encode_context(model, sh, ah, n_nodes, ni, nm, device)
        cur = states[-1].reshape(n_nodes, F)[:, P_max:2 * P_max].argmax(1)
        cov_now = cov_of_context(z_ctx[:, -1, :], cur, Zb_by_phase, MU, SD, k=args.k)

        # CEM
        probs = np.full((n_nodes, H, P_max), 1.0 / P_max)
        phases = None
        for _ in range(ITERS):
            phases = np.empty((S, n_nodes, H), dtype=np.int64)
            for i in range(n_nodes):
                for h in range(H):
                    phases[:, i, h] = rng.choice(P_max, size=S, p=probs[i, h])
            pe = cem_rollout(model, z_ctx, a_ctx, phases, n_nodes, P_max, ni, nm, device)
            mc = decode(pr, pe).clamp(min=0).sum(dim=(1, 2, 3)).cpu().numpy()
            elite = phases[np.argsort(mc)[:TOPK]]
            for i in range(n_nodes):
                for h in range(H):
                    c = np.bincount(elite[:, i, h], minlength=P_max)
                    probs[i, h] = (c + 1e-3) / (c.sum() + P_max * 1e-3)
        pe = cem_rollout(model, z_ctx, a_ctx, phases, n_nodes, P_max, ni, nm, device)
        mc = decode(pr, pe).clamp(min=0).sum(dim=(1, 2, 3)).cpu().numpy()
        exec_phase = probs[:, 0].argmax(1)
        m_pick = int(np.argmin(mc))
        rand_pick = int(rng.integers(S))

        # oracle-score every candidate from the frozen state
        snap = env.save_state(f"_cmp_snap_{os.getpid()}_{seed}.xml")
        oc = np.zeros(S)
        for s in range(S):
            env.load_state(snap)
            tot = 0.0
            for h in range(H):
                st = env.step(phases[s, :, h])
                tot += float(st.reshape(n_nodes, F)[:, :P_max].sum())
            oc[s] = tot
        env.load_state(snap)
        o_best = oc.min()
        regret_model = float(oc[m_pick] - o_best)
        regret_random = float(oc.mean() - o_best)

        # eps_plan on the model's chosen plan (vs true SUMO rollout of it)
        env.load_state(snap)
        tp = np.zeros((H, n_nodes * F), dtype=np.float32)
        for h in range(H):
            tp[h] = env.step(phases[m_pick, :, h])
        env.load_state(snap)
        full = np.concatenate([sh, tp], 0)[None]
        with torch.no_grad():
            o = encode_batch(model, torch.from_numpy(full).float(),
                             torch.zeros(1, HS + H, n_nodes * P_max), n_nodes, ni, nm, device)
        zt = o["emb"].reshape(1, n_nodes, HS + H, -1)[0, :, HS:, :]
        eps_plan = float(((pe[m_pick] - zt) ** 2).mean().item())

        rows.append(dict(step=step, regret_model=regret_model, regret_random=regret_random,
                         cov_now=cov_now, eps_plan=eps_plan,
                         oracle_mean=float(oc.mean()), oracle_min=float(o_best)))
        # advance the real (latent-CEM) trajectory
        actions.append(env.encode_action(np.clip(exec_phase, 0, P_max - 1)))
        s_next = env.step(exec_phase)
        states.append(s_next)
        tail_latent += float(s_next.reshape(n_nodes, F)[:, :P_max].sum())
    env.close()

    # MaxPressure and random-candidate tails from the same warm-up
    for name in ("mp", "rand"):
        e2 = SumoMultiEnv(args.sumocfg, seed=seed, warmup=0, metrics=False, begin=args.begin)
        e2.reset()
        m2 = controller_max_pressure(e2)
        st2, ac2 = [e2.state()], []
        for t in range(args.warmup):
            ph = m2(t)
            ac2.append(e2.encode_action(np.clip(ph, 0, P_max - 1)))
            st2.append(e2.step(ph))
        tot = 0.0
        r2 = np.random.default_rng(seed + 1)
        for step in range(args.n_steps):
            if name == "mp":
                ph = m2(args.warmup + step)
            else:
                ph = r2.integers(0, P_max, size=n_nodes)
            s_n = e2.step(np.clip(ph, 0, P_max - 1))
            tot += float(s_n.reshape(n_nodes, F)[:, :P_max].sum())
        e2.close()
        if name == "mp":
            tail_mp = tot
        else:
            tail_random = tot
    return rows, dict(tail_latent=tail_latent, tail_random=tail_random, tail_mp=tail_mp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="L05ar")
    ap.add_argument("--weights", default="weights_epoch_80.pt")
    ap.add_argument("--data_dir", default="traffic_data_cologne8")
    ap.add_argument("--sumocfg", default=SUMOCFG)
    ap.add_argument("--begin", type=int, default=25200)
    ap.add_argument("--seeds", type=int, nargs="+", default=[777, 101, 202])
    ap.add_argument("--n_steps", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--cem_S", type=int, default=64)
    ap.add_argument("--cem_iters", type=int, default=4)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--burn", type=int, default=8, help="drop the first N (post-warmup transient) steps from trend fits")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = torch.load(f"{args.data_dir}/val.pt", weights_only=False)
    n_nodes, P_max, F = meta["n_nodes"], meta["P_max"], meta["node_feature_dim"]
    ni = torch.from_numpy(np.asarray(meta["neighbor_idx"]))
    nm = torch.from_numpy(np.asarray(meta["neighbor_mask"]))
    model, cfg = load(args.run, args.weights, device)
    pr = fit_pressure_probe(model, f"{args.data_dir}/train.pt", n_nodes, P_max, F, ni, nm, device)
    Zb, PHb, MU, SD, _ = build_coverage_bank(model, f"{args.data_dir}/train.pt",
                                             n_nodes, F, P_max, ni, nm, device)
    Zb_by_phase = {p: Zb[PHb == p].to(device) for p in range(P_max)}
    print(f"model {args.run}  probe MSE {pr['mse']:.1f}")

    all_rows, tails = [], []
    for seed in args.seeds:
        rows, tail = run_seed(args, model, pr, ni, nm, n_nodes, P_max, F,
                              Zb_by_phase, MU, SD, device, seed)
        for r in rows:
            r["seed"] = seed
        all_rows += rows
        tails.append(tail)
        print(f"  seed {seed}: tail halting  latent-CEM {tail['tail_latent']:.0f}  "
              f"random {tail['tail_random']:.0f}  MaxPressure {tail['tail_mp']:.0f}")

    analyse(all_rows, tails, args)


def analyse(rows, tails, args):
    import numpy as np
    from scipy.stats import pearsonr, spearmanr
    keep = [r for r in rows if r["step"] >= args.burn]   # drop post-warmup transient from trends
    print(f"(trend analysis on steps >= {args.burn}: {len(keep)}/{len(rows)} rows)")
    step = np.array([r["step"] for r in keep], float)
    rm = np.array([r["regret_model"] for r in keep], float)
    rr = np.array([r["regret_random"] for r in keep], float)
    cov = np.array([r["cov_now"] for r in keep], float)
    eps = np.array([r["eps_plan"] for r in keep], float)

    print("\n=== (a) is the model's pick worse than random? ===")
    print(f"  mean regret_model  = {rm.mean():7.1f}")
    print(f"  mean regret_random = {rr.mean():7.1f}")
    print(f"  model / random     = {rm.mean() / (rr.mean() + 1e-9):.2f}   "
          f"({'WORSE than random -> exploitation' if rm.mean() > rr.mean() else 'better than random -> weakly-informative model, not adversarial'})")
    frac_worse = float((rm > rr).mean())
    print(f"  fraction of steps where model pick is worse than the mean plan: {frac_worse:.2f}")

    print("\n=== (b) does per-decision regret COMPOUND over the closed loop? ===")
    def trend(y):
        b = np.polyfit(step, y, 1)
        r = pearsonr(step, y)[0]
        # normalise slope by episode span
        return b[0], b[0] * (step.max() - step.min()), r
    for name, y in (("regret_model", rm), ("regret_random", rr), ("cov_now", cov), ("eps_plan", eps)):
        sl, span, r = trend(y)
        print(f"  {name:>13}: slope/step {sl:+8.3f}   total drift {span:+8.1f}   pearson(step) {r:+.2f}")
    rc = spearmanr(cov, rm)[0]
    re_ = spearmanr(eps, rm)[0]
    print(f"  spearman(cov_now, regret_model) = {rc:+.2f}    spearman(eps_plan, regret_model) = {re_:+.2f}")

    tl = np.mean([t["tail_latent"] for t in tails])
    tr = np.mean([t["tail_random"] for t in tails])
    tm = np.mean([t["tail_mp"] for t in tails])
    print(f"\n=== closed-loop tail halting (mean over {len(tails)} seeds) ===")
    print(f"  latent-CEM {tl:.0f}   random {tr:.0f}   MaxPressure {tm:.0f}"
          f"   (latent/MP {tl/tm:.2f}, latent/random {tl/tr:.2f})")

    sl_m = trend(rm)[0]
    sl_c = trend(cov)[0]
    if rm.mean() < rr.mean() and sl_m > 0 and sl_c > 0 and rc > 0.15:
        verdict = ("COMPOUNDING OPTIMISM BIAS: per-decision the model beats random, but "
                   "regret and OOD-drift both grow over the closed loop and co-vary "
                   "-> reframe from 'adversarial exploitation' to 'mild bias that compounds'")
    elif rm.mean() >= rr.mean():
        verdict = "ADVERSARIAL EXPLOITATION: model pick is worse than random"
    else:
        verdict = ("WEAKLY-INFORMATIVE MODEL, NO CLEAR COMPOUNDING: model beats random, "
                   "regret does not trend up -> the closed-loop gap is not a compounding story")
    print(f"\n>>> VERDICT: {verdict}")

    Path(f"{args.data_dir}/diag_compounding.json").write_text(json.dumps(dict(
        run=args.run, seeds=args.seeds, n_steps=args.n_steps,
        mean_regret_model=float(rm.mean()), mean_regret_random=float(rr.mean()),
        model_over_random=float(rm.mean() / (rr.mean() + 1e-9)),
        regret_slope=float(trend(rm)[0]), regret_total_drift=float(trend(rm)[1]),
        cov_slope=float(trend(cov)[0]), eps_slope=float(trend(eps)[0]),
        spearman_cov_regret=float(rc), spearman_eps_regret=float(re_),
        tail_latent=float(tl), tail_random=float(tr), tail_mp=float(tm),
        verdict=verdict), indent=2))
    import csv
    with open(f"{args.data_dir}/diag_compounding_rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {args.data_dir}/diag_compounding.json (+ _rows.csv)")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        seed_arr = np.array([r["seed"] for r in keep])
        for s in sorted(set(seed_arr)):
            m = seed_arr == s
            ax[0].plot(step[m], rm[m], ".-", alpha=.6, label=f"seed {s}")
        ax[0].axhline(rr.mean(), color="k", ls="--", label="mean random regret")
        b = np.polyfit(step, rm, 1); ax[0].plot(step, np.polyval(b, step), "r-", lw=2, label="trend")
        ax[0].set_xlabel("closed-loop step"); ax[0].set_ylabel("per-decision regret (oracle units)")
        ax[0].set_title("regret vs step (compounding?)"); ax[0].legend(fontsize=7)
        ax[1].scatter(cov, rm, s=10, alpha=.4)
        ax[1].set_xlabel("cov_now (context latent kNN dist)"); ax[1].set_ylabel("regret_model")
        ax[1].set_title(f"regret vs OOD-drift  spearman {rc:+.2f}")
        ax[2].plot(step, np.polyval(np.polyfit(step, cov, 1), step), label="cov_now trend")
        ax[2].plot(step, np.polyval(np.polyfit(step, eps, 1), step), label="eps_plan trend")
        ax[2].set_xlabel("closed-loop step"); ax[2].set_title("OOD-drift over the episode"); ax[2].legend(fontsize=8)
        fig.suptitle(f"{args.run} / cologne8 — compounding check", fontsize=10)
        fig.tight_layout(); fig.savefig(f"{args.data_dir}/diag_compounding.png", dpi=110)
        print(f"wrote {args.data_dir}/diag_compounding.png")
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
