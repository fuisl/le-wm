"""Full-episode rollout under the new settings: open-loop vs receding-horizon
re-anchoring, for the AR-trained (and PNA) cologne8 models. Dumps JSON for the
artifact.

For one held-out val episode and each model:
  ground truth  - real per-node total halting
  teacher-forced - encode each true state, decode (no drift; the floor)
  open loop      - 3-step context, 297 steps of pure imagination (K=inf)
  receding K=3/5 - re-encode from the real last-3 obs every K steps, imagine K,
                   then jump the belief back to reality (an MPC-style loop)
plus the re-anchor sweep (mean decoded MSE vs K, model vs persistence).

Usage:
    python visualize_receding_cologne8.py --runs L05 L05ar L05pnaar L05permar L05pnapermar
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from eval_multi_sumo import encode_batch, load, node_rollout
from visualize_rollout_cologne8 import decode, fit_pressure_probe

DATA_DIR = "traffic_data_cologne8"
WEIGHTS = "weights_epoch_80.pt"
HS = 3
EPISODE_IDX = 3


def total_halting(flat, n, P, F):
    return flat.reshape(-1, n, F)[..., :P].sum(-1)


def receding(model, pr, ep, n, P, F, ni, nm, K, device):
    """(n_nodes, T-HS) decoded total-halting trace, re-anchored every K steps.
    K=None -> open loop."""
    T = ep["state"].shape[0]
    st, ac = ep["state"], ep["action"]
    Kb = (T - HS) if K is None else K
    out = np.zeros((T - HS, n))
    for t0 in range(HS, T, Kb):
        h = min(Kb, T - t0)
        if h <= 0:
            break
        sh = torch.from_numpy(st[t0 - HS:t0]); ah = torch.from_numpy(ac[t0 - HS:t0])
        af = torch.from_numpy(ac[t0:t0 + h])
        pe = node_rollout(model, sh, ah, af, n, ni, nm, HS, device)
        dec = decode(pr, pe).clamp(min=0).sum(-1).cpu().numpy()      # (h, n)
        out[t0 - HS:t0 - HS + h] = dec
    return out.T                                                    # (n, T-HS)


def sweep(model, pr, ep, n, P, F, ni, nm, device, Ks=(1, 2, 3, 5, 10, 20)):
    T = ep["state"].shape[0]
    st = ep["state"]
    m_out, p_out = [], []
    for K in list(Ks) + [None]:
        Kb = (T - HS) if K is None else K
        me, pe = [], []
        for t0 in range(HS, T, Kb):
            h = min(Kb, T - t0)
            if h <= 0:
                break
            sh = torch.from_numpy(st[t0 - HS:t0])
            ah = torch.from_numpy(ep["action"][t0 - HS:t0])
            af = torch.from_numpy(ep["action"][t0:t0 + h])
            true = total_halting(torch.from_numpy(st[t0:t0 + h]).to(device), n, P, F)
            pe_emb = node_rollout(model, sh, ah, af, n, ni, nm, HS, device)
            dec = decode(pr, pe_emb).clamp(min=0).sum(-1)
            per = total_halting(torch.from_numpy(st[t0 - 1]).to(device), n, P, F).reshape(n)
            per = per.unsqueeze(0).expand(h, n)
            me.append(((dec - true) ** 2).mean(1).cpu().numpy())
            pe.append(((per - true) ** 2).mean(1).cpu().numpy())
        m_out.append(float(np.concatenate(me).mean()))
        p_out.append(float(np.concatenate(pe).mean()))
    return list(Ks), m_out[:-1], p_out[:-1], m_out[-1], p_out[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+",
                    default=["L05", "L05ar", "L05pnaar", "L05permar", "L05pnapermar"])
    ap.add_argument("--out", default="traffic_data_cologne8/receding_viz.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dd = Path(DATA_DIR)
    meta = torch.load(dd / "val.pt", weights_only=False)
    n, P, Fdim = meta["n_nodes"], meta["P_max"], meta["node_feature_dim"]
    ni = torch.from_numpy(np.asarray(meta["neighbor_idx"]))
    nm = torch.from_numpy(np.asarray(meta["neighbor_mask"]))
    ep = torch.load(dd / "val.pt", weights_only=False)["episodes"][EPISODE_IDX]
    T = ep["state"].shape[0]
    gt = total_halting(ep["state"], n, P, Fdim)                     # (T, n)

    LABELS = {
        "L05": "mean-pool · 1-step loss",
        "L05ar": "mean-pool · +AR loss",
        "L05pnaar": "PNA-pool · +AR loss",
        "L05permar": "mean-pool · +AR · wrong-nbr",
        "L05pnapermar": "PNA-pool · +AR · wrong-nbr",
        "L0ar": "no coupling · +AR loss",
    }
    models = {}
    for run in args.runs:
        try:
            model, cfg = load(run, WEIGHTS, device)
        except FileNotFoundError:
            print(f"skip {run} (no checkpoint)"); continue
        pr = fit_pressure_probe(model, dd / "train.pt", n, P, Fdim, ni, nm, device)
        with torch.no_grad():
            o = encode_batch(model, torch.from_numpy(ep["state"]).unsqueeze(0),
                             torch.from_numpy(ep["action"]).unsqueeze(0), n, ni, nm, device)
            emb_tf = o["emb"].reshape(1, n, T, -1).transpose(1, 2)[0]
            tf = decode(pr, emb_tf).clamp(min=0).sum(-1).cpu().numpy()      # (T, n)
        open_ = receding(model, pr, ep, n, P, Fdim, ni, nm, None, device)   # (n, T-HS)
        r3 = receding(model, pr, ep, n, P, Fdim, ni, nm, 3, device)
        r5 = receding(model, pr, ep, n, P, Fdim, ni, nm, 5, device)
        Ks, mmse, pmse, om, op = sweep(model, pr, ep, n, P, Fdim, ni, nm, device)
        models[run] = {
            "label": LABELS.get(run, run),
            "probe_mse": pr["mse"],
            "network_total": {
                "teacher_forced": tf.sum(1).tolist(),
                "open": open_.sum(0).tolist(),
                "recede_k3": r3.sum(0).tolist(),
                "recede_k5": r5.sum(0).tolist(),
            },
            "per_node_total": {
                "teacher_forced": tf.T.tolist(),
                "open": open_.tolist(),
                "recede_k3": r3.tolist(),
                "recede_k5": r5.tolist(),
            },
            "sweep": {"K": Ks, "model_mse": mmse, "persist_mse": pmse,
                      "open_model": om, "open_persist": op},
        }
        print(f"{run:<14} probe {pr['mse']:5.1f} | open MSE {om:7.0f} | K5 MSE "
              f"{mmse[3]:7.1f} vs persist {pmse[3]:7.1f}")

    out = {
        "meta": {
            "scenario": "cologne8 - 8 signals", "controller": ep["controller"],
            "episode_idx": EPISODE_IDX, "episode_len": T, "step_length_s": 5,
            "history_size": HS, "n_nodes": n, "P_max": P,
            "neighbor_degrees": nm.sum(1).tolist(), "tl_ids": meta["tl_ids"],
            "checkpoint": WEIGHTS,
        },
        "ground_truth_total": gt.T.tolist(),                # (n, T)
        "network_total_ground_truth": gt.sum(1).tolist(),   # (T,)
        "models": models,
    }
    Path(args.out).write_text(json.dumps(out))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
