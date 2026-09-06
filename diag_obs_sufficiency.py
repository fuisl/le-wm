"""T1 (2026-09-06): is the observation sufficient for the dynamics the planner needs?

Model-free test. For each observation mode (column subsets of a corpus generated
with --obs_mode full) fit simple regressors that predict, per signal,
  (k=1) halting at t+1,
  (k=5) cumulative halting over t+1..t+5   (= the planner's cost),
from: own obs history (HS steps), neighbour-mean obs at t, own + neighbour-mean
one-hot actions over t..t+k-1. Report val R^2 / RMSE.

Then the ranking test that matters for planning: on counterfactual.pt (30 anchors x
16 held-phase branches, H=5) predict each branch's network cost and report the
within-anchor Spearman with the true SUMO cost and the top-1 hit rate. A regressor
on a *sufficient* observation should rank well; if base cannot and raster can, the
observation is the bottleneck (H_obs). Compare with the world model's rank_B.

Usage:
    python diag_obs_sufficiency.py --src traffic_data_cologne8_full --modes base link raster full
"""
import argparse
import json
import time

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from traffic.slice_obs import block_cols
from traffic.multi_agent import masked_neighbor_mean

HS = 3


def nbr_mean_np(x, ni, nm):
    """x: (T, N, D) -> (T, N, D) mean over real neighbours."""
    xt = torch.from_numpy(x).float().permute(1, 0, 2).unsqueeze(0)      # (1,N,T,D)
    out = masked_neighbor_mean(xt, torch.from_numpy(ni), torch.from_numpy(nm))
    return out[0].permute(1, 0, 2).numpy()


def build_rows(episodes, N, F_full, cols, P, ni, nm, k):
    X, Y1, Yk = [], [], []
    for ep in episodes:
        s = ep["state"].reshape(-1, N, F_full)[:, :, cols]               # (T,N,Fm)
        a = ep["action"].reshape(-1, N, P)                                # (T,N,P)
        T = s.shape[0]
        sn = nbr_mean_np(s, ni, nm)
        an = nbr_mean_np(a, ni, nm)
        halt = s[:, :, :P].sum(-1)                                        # (T,N) pressure-block sum
        for t in range(HS - 1, T - k):
            own_hist = s[t - HS + 1:t + 1].transpose(1, 0, 2).reshape(N, -1)   # (N, HS*Fm)
            acts = a[t:t + k].transpose(1, 0, 2).reshape(N, -1)                 # (N, k*P)
            actn = an[t:t + k].transpose(1, 0, 2).reshape(N, -1)
            feat = np.concatenate([own_hist, sn[t], acts, actn], 1)
            X.append(feat)
            Y1.append(halt[t + 1])
            Yk.append(halt[t + 1:t + k + 1].sum(0))
    return np.concatenate(X), np.concatenate(Y1), np.concatenate(Yk)


def cf_rows(cf, N, F_full, cols, P, ni, nm, k):
    """Per anchor: (16, N, feat), true per-branch network cost (16,)."""
    out = []
    for smp in cf["samples"]:
        s0 = smp["anchor_state"].reshape(N, F_full)[:, cols]
        hist = np.repeat(s0[None], HS, 0)                                 # no history in CF: replicate
        sn = nbr_mean_np(s0[None], ni, nm)[0]
        feats, costs = [], []
        for b in smp["branches"]:
            a = np.repeat(b["action"].reshape(1, N, P), k, 0)             # held phase
            an = nbr_mean_np(a, ni, nm)
            own_hist = hist.transpose(1, 0, 2).reshape(N, -1)
            feat = np.concatenate([own_hist, sn, a.transpose(1, 0, 2).reshape(N, -1),
                                   an.transpose(1, 0, 2).reshape(N, -1)], 1)
            st = b["states"].reshape(-1, N, F_full)[1:k + 1, :, cols]
            feats.append(feat); costs.append(st[:, :, :P].sum())
        out.append((np.stack(feats), np.array(costs)))
    return out


def switching_branches(sumocfg, n_anchors, k, N, P, n_green, seed, n_branch=16, obs_mode="full"):
    """Fresh anchors (random-controller warm-up, as in generate_counterfactual) with
    per-step RANDOM joint sequences over LEGAL codes -- the switch-heavy population a
    uniform CEM proposes at iteration 0 -- plus true SUMO costs. Returns list of
    (anchor_state (N*F), hist (HS,N*F), action_seqs (n_branch,k,N*P), costs (n_branch,))."""
    import os
    from traffic.sumo_multi_env import SumoMultiEnv, controller_random
    rng = np.random.default_rng(seed)
    out = []
    for a in range(n_anchors):
        env = SumoMultiEnv(sumocfg, seed=int(rng.integers(0, 1_000_000)), warmup=10, obs_mode=obs_mode)
        env.reset()
        pol = controller_random(env, np.random.default_rng(int(rng.integers(0, 1_000_000))), switch_prob=0.25)
        hist = []
        for t in range(int(rng.integers(20, 70))):
            hist.append(env.step(pol(t)))
        hist = np.stack(hist[-HS:])
        snap = env.save_state(f"/tmp/claude-1000/_suf_{os.getpid()}_{a}.xml")
        seqs, costs = [], []
        for b in range(n_branch):
            ph = np.stack([rng.integers(0, n_green) for _ in range(k)])          # (k,N) legal per-step random
            env.load_state(snap)
            tot = 0.0
            for h in range(k):
                st = env.step(ph[h])
                tot += float(st.reshape(N, -1)[:, :P].sum())
            seqs.append(np.stack([env.encode_action(ph[h]) for h in range(k)]).reshape(k, N, P))
            costs.append(tot)
        env.close()
        out.append((hist, np.stack(seqs), np.array(costs)))
    return out


def sw_rows(sw_set, N, F_full, cols, P, ni, nm, k):
    out = []
    for hist_full, seqs, costs in sw_set:
        hist = hist_full.reshape(HS, N, F_full)[:, :, cols]
        sn = nbr_mean_np(hist[-1:], ni, nm)[0]
        own_hist = hist.transpose(1, 0, 2).reshape(N, -1)
        feats = []
        for a in seqs:                                                     # (k,N,P)
            an = nbr_mean_np(a, ni, nm)
            feats.append(np.concatenate([own_hist, sn, a.transpose(1, 0, 2).reshape(N, -1),
                                         an.transpose(1, 0, 2).reshape(N, -1)], 1))
        out.append((np.stack(feats), costs))
    return out


def rank_eval(model, cf_set):
    rs, top1 = [], 0
    for feats, true_cost in cf_set:
        B, N, D = feats.shape
        pred = model.predict(feats.reshape(B * N, D)).reshape(B, N).sum(1)
        r = spearmanr(pred, true_cost).correlation
        if np.isfinite(r):
            rs.append(r)
        top1 += int(np.argmin(pred) == np.argmin(true_cost))
    return float(np.mean(rs)), top1 / len(cf_set)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="traffic_data_cologne8_full")
    ap.add_argument("--modes", nargs="+", default=["base", "link", "raster", "full"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--gbm_iters", type=int, default=300)
    ap.add_argument("--out", default="results/diag_obs_sufficiency.json")
    ap.add_argument("--switching", type=int, default=0,
                    help="also rank this many fresh anchors x 16 switch-heavy random branches (SUMO)")
    ap.add_argument("--sumocfg", default=None)
    args = ap.parse_args()
    sw_set = None
    if args.switching > 0:
        from plan_cem_multi import SUMOCFG
        meta0 = torch.load(f"{args.src}/val.pt", weights_only=False)
        sw_set = switching_branches(args.sumocfg or SUMOCFG, args.switching, args.k, meta0["n_nodes"],
                                    meta0["P_max"], np.asarray(meta0["n_green_phases"]), seed=4242)
        print(f"built {len(sw_set)} switching anchors x 16 branches", flush=True)

    tr = torch.load(f"{args.src}/train.pt", weights_only=False)
    va = torch.load(f"{args.src}/val.pt", weights_only=False)
    cf = torch.load(f"{args.src}/counterfactual.pt", weights_only=False)
    N, F_full, P = tr["n_nodes"], tr["node_feature_dim"], tr["P_max"]
    ni, nm = np.asarray(tr["neighbor_idx"]), np.asarray(tr["neighbor_mask"])
    k = args.k
    res = {"k": k, "modes": {}}
    for mode in args.modes:
        cols = block_cols(P, mode)
        t0 = time.time()
        Xtr, Y1tr, Yktr = build_rows(tr["episodes"], N, F_full, cols, P, ni, nm, k)
        Xva, Y1va, Ykva = build_rows(va["episodes"], N, F_full, cols, P, ni, nm, k)
        cfs = cf_rows(cf, N, F_full, cols, P, ni, nm, k)
        r = {"F": len(cols), "n_train": int(len(Xtr)), "n_val": int(len(Xva))}
        for name, mk in (("ridge", lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
                         ("gbm", lambda: HistGradientBoostingRegressor(max_iter=args.gbm_iters,
                                                                       learning_rate=0.08, random_state=0))):
            for tgt, Ytr, Yva in (("h1", Y1tr, Y1va), (f"cum{k}", Yktr, Ykva)):
                m = mk().fit(Xtr, Ytr)
                p = m.predict(Xva)
                r2 = 1 - ((p - Yva) ** 2).mean() / Yva.var()
                rmse = float(np.sqrt(((p - Yva) ** 2).mean()))
                r[f"{name}_{tgt}_r2"] = float(r2); r[f"{name}_{tgt}_rmse"] = rmse
                if tgt.startswith("cum"):
                    rho, top1 = rank_eval(m, cfs)
                    r[f"{name}_cf_spearman"] = rho; r[f"{name}_cf_top1"] = top1
                    if sw_set is not None:
                        rho_s, top1_s = rank_eval(m, sw_rows(sw_set, N, F_full, cols, P, ni, nm, k))
                        r[f"{name}_sw_spearman"] = rho_s; r[f"{name}_sw_top1"] = top1_s
        r["sec"] = round(time.time() - t0, 1)
        res["modes"][mode] = r
        print(f"[{mode:6s}] F={r['F']:3d}  ridge R2 h1 {r['ridge_h1_r2']:.3f} cum{k} {r[f'ridge_cum{k}_r2']:.3f}  "
              f"gbm R2 h1 {r['gbm_h1_r2']:.3f} cum{k} {r[f'gbm_cum{k}_r2']:.3f}  |  CF rank "
              f"ridge {r['ridge_cf_spearman']:+.2f} (top1 {r['ridge_cf_top1']:.2f})  "
              f"gbm {r['gbm_cf_spearman']:+.2f} (top1 {r['gbm_cf_top1']:.2f})"
              + (f"  |  SWITCHING rank ridge {r['ridge_sw_spearman']:+.2f} gbm {r['gbm_sw_spearman']:+.2f} "
                 f"(top1 {r['gbm_sw_top1']:.2f})" if sw_set is not None else "")
              + f"   {r['sec']}s", flush=True)
    # true-cost sanity: spearman of true vs true = 1; baseline = random ranking ~0
    json.dump(res, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
