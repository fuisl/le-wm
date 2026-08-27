"""Does re-anchoring the belief on real observations rescue the world model?

Motivated by LeWM's own note that flat models collapse over long horizons, and
by the observation (2026-08-27) that our open-loop 297-step imagined rollout on
cologne8 diverges hard. A planner never rolls open-loop: it imagines H steps,
acts, re-observes, re-plans. This script measures how much that helps.

Experiment 1 - trust horizon from a fresh belief:
    from many start points, encode 3 real steps, imagine N steps with the true
    action sequence, decode, error vs offset h. vs persistence(last real obs).

Experiment 2 - receding-horizon sweep:
    run the whole episode closed-loop with re-anchor interval K in
    {1,2,3,5,10,20,inf}. Every K steps, re-encode from the real last-3 obs,
    imagine K steps, score against truth, then jump the belief back to reality.
    K=inf is the current open-loop rollout. Reports mean decoded MSE over the
    episode vs K, against a persistence-with-same-K baseline.

Experiment 3 - plan-ranking vs scoring horizon:
    for the counterfactual anchors, does scoring branches on a *shorter* imagined
    horizon rank them better? (the planning-relevant version of "short = trusted")

Usage: python test_receding_horizon.py --runs L0 L05 L05perm
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from eval_multi_sumo import encode_batch, load, node_rollout
from visualize_rollout_cologne8 import decode, fit_pressure_probe

DATA_DIR = "traffic_data_cologne8"
WEIGHTS = "weights_epoch_80.pt"
HS = 3


def total_halting(state_flat, n_nodes, P, F):
    return state_flat.reshape(-1, n_nodes, F)[..., :P].sum(-1)   # (..., N)


def exp1_trust_horizon(model, pr, episodes, n_nodes, P, F, ni, nm, device, N=20):
    me = np.zeros(N); pe = np.zeros(N); cnt = 0
    for ep in episodes:
        T = ep["state"].shape[0]
        span = T - HS - N
        if span <= 0:
            continue
        for s in range(0, span, max(1, span // 25)):
            sh = torch.from_numpy(ep["state"][s:s + HS])
            ah = torch.from_numpy(ep["action"][s:s + HS])
            af = torch.from_numpy(ep["action"][s + HS:s + HS + N])
            true = total_halting(torch.from_numpy(ep["state"][s + HS:s + HS + N]).to(device), n_nodes, P, F)
            eh = torch.from_numpy(ep["edge_feat"][s:s + HS]) if "edge_feat" in ep else None
            pe_emb = node_rollout(model, sh, ah, af, n_nodes, ni, nm, HS, device, edge_hist=eh)
            dec = decode(pr, pe_emb).clamp(min=0).sum(-1)               # (N, n_nodes)
            per = total_halting(torch.from_numpy(ep["state"][s + HS - 1]).to(device), n_nodes, P, F).reshape(n_nodes)
            per = per.unsqueeze(0).expand(N, n_nodes)
            me += ((dec - true) ** 2).mean(1).cpu().numpy()
            pe += ((per - true) ** 2).mean(1).cpu().numpy()
            cnt += 1
    return me / cnt, pe / cnt, cnt


def exp2_receding(model, pr, ep, n_nodes, P, F, ni, nm, device, K):
    """One episode, re-anchor every K steps (K=None -> open loop). Returns
    (mean model MSE, mean persistence MSE) over all scored steps."""
    T = ep["state"].shape[0]
    states = ep["state"]; actions = ep["action"]
    Kb = (T - HS) if K is None else K
    m_errs, p_errs = [], []
    for t0 in range(HS, T, Kb):
        h = min(Kb, T - t0)
        if h <= 0:
            break
        sh = torch.from_numpy(states[t0 - HS:t0])
        ah = torch.from_numpy(actions[t0 - HS:t0])
        af = torch.from_numpy(actions[t0:t0 + h])
        true = total_halting(torch.from_numpy(states[t0:t0 + h]).to(device), n_nodes, P, F)
        eh = torch.from_numpy(ep["edge_feat"][t0 - HS:t0]) if "edge_feat" in ep else None
        pe_emb = node_rollout(model, sh, ah, af, n_nodes, ni, nm, HS, device, edge_hist=eh)
        dec = decode(pr, pe_emb).clamp(min=0).sum(-1)                  # (h, n_nodes)
        per = total_halting(torch.from_numpy(states[t0 - 1]).to(device), n_nodes, P, F).reshape(n_nodes)
        per = per.unsqueeze(0).expand(h, n_nodes)
        m_errs.append(((dec - true) ** 2).mean(1).cpu().numpy())
        p_errs.append(((per - true) ** 2).mean(1).cpu().numpy())
    return float(np.concatenate(m_errs).mean()), float(np.concatenate(p_errs).mean())


def exp3_ranking_vs_horizon(model, pr, cf_path, n_nodes, P, F, ni, nm, device):
    data = torch.load(cf_path, weights_only=False)
    samples = data["samples"]
    Hmax = data["samples"][0]["branches"][0]["states"].shape[0] - 1
    top1 = {h: [] for h in range(1, Hmax + 1)}
    sp = {h: [] for h in range(1, Hmax + 1)}
    for smp in samples:
        a_state = torch.from_numpy(smp["anchor_state"])
        a_act = torch.from_numpy(smp["anchor_action"])
        sh = a_state.unsqueeze(0).expand(HS, -1).contiguous()
        ah = a_act.unsqueeze(0).expand(HS, -1).contiguous()
        # predicted / true running cost per branch, cumulative over horizon
        P_cost = []  # (n_branches, Hmax)
        T_cost = []
        for br in smp["branches"]:
            af = torch.from_numpy(br["action"]).unsqueeze(0).expand(Hmax, -1).contiguous()
            eh = (torch.from_numpy(smp["anchor_edge"]).unsqueeze(0).expand(HS, -1).contiguous()
                  if "anchor_edge" in smp else None)
            pe = node_rollout(model, sh, ah, af, n_nodes, ni, nm, HS, device, edge_hist=eh)
            dec = decode(pr, pe).clamp(min=0).sum(-1)                  # (Hmax, n_nodes)
            P_cost.append(np.cumsum(dec.sum(1).cpu().numpy()))
            tr = total_halting(torch.from_numpy(br["states"][1:]), n_nodes, P, F)
            T_cost.append(np.cumsum(tr.sum(1).numpy()))
        P_cost = np.array(P_cost); T_cost = np.array(T_cost)
        for h in range(1, Hmax + 1):
            cp, ct = P_cost[:, h - 1], T_cost[:, h - 1]
            top1[h].append(int(np.argmin(cp) == np.argmin(ct)))
            r, _ = spearmanr(cp, ct)
            sp[h].append(r)
    return {h: (float(np.mean(top1[h])), float(np.nanmean(sp[h]))) for h in top1}, len(samples[0]["branches"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["L0", "L05", "L05perm"])
    ap.add_argument("--episode", type=int, default=3)
    ap.add_argument("--data_dir", default=DATA_DIR)
    ap.add_argument("--weights", default=WEIGHTS)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dd = Path(args.data_dir)
    meta = torch.load(dd / "val.pt", weights_only=False)
    n_nodes, P, F = meta["n_nodes"], meta["P_max"], meta["node_feature_dim"]
    ni = torch.from_numpy(np.asarray(meta["neighbor_idx"]))
    nm = torch.from_numpy(np.asarray(meta["neighbor_mask"]))
    val = torch.load(dd / "val.pt", weights_only=False)
    ep = val["episodes"][args.episode]

    Ks = [1, 2, 3, 5, 10, 20, None]
    for run in args.runs:
        print(f"\n================ {run} ================")
        model, _ = load(run, args.weights, device)
        pr = fit_pressure_probe(model, dd / "train.pt", n_nodes, P, F, ni, nm, device)

        me, pe, cnt = exp1_trust_horizon(model, pr, val["episodes"], n_nodes, P, F, ni, nm, device, N=20)
        print(f"\n[exp1] trust horizon from a fresh 3-step belief  ({cnt} windows)")
        print(f"  {'h':>3} {'model MSE':>11} {'persist MSE':>12} {'ratio':>7}")
        for h in [1, 2, 3, 5, 8, 12, 16, 20]:
            print(f"  {h:>3} {me[h-1]:>11.1f} {pe[h-1]:>12.1f} {me[h-1]/pe[h-1]:>7.2f}")

        print(f"\n[exp2] receding-horizon sweep on val ep #{args.episode} ({ep['controller']})")
        print(f"  {'K':>5} {'model MSE':>11} {'persist MSE':>12} {'ratio':>7}")
        for K in Ks:
            m, p = exp2_receding(model, pr, ep, n_nodes, P, F, ni, nm, device, K)
            tag = "open" if K is None else str(K)
            print(f"  {tag:>5} {m:>11.1f} {p:>12.1f} {m/p:>7.2f}")

        rank, nbr = exp3_ranking_vs_horizon(model, pr, dd / "counterfactual.pt", n_nodes, P, F, ni, nm, device)
        print(f"\n[exp3] plan-ranking vs scoring horizon  (chance top-1 = {1/nbr:.3f})")
        print(f"  {'h':>3} {'top-1':>7} {'spearman':>9}")
        for h in sorted(rank):
            print(f"  {h:>3} {rank[h][0]:>7.3f} {rank[h][1]:>9.3f}")


if __name__ == "__main__":
    main()
