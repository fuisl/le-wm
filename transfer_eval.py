"""Zero-shot transfer eval: run one trained model on several scenario data dirs
without retraining, dump the scenario-normalised comparison (model/persistence
ratio vs re-anchor K and vs horizon, plan-ranking vs scoring horizon) for the
artifact.

Usage:
    python transfer_eval.py --run L05ar \
        --dirs traffic_data_cologne8:cologne8 traffic_data_ingolstadt21:ingolstadt21
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from eval_multi_sumo import load
from visualize_rollout_cologne8 import fit_pressure_probe
from test_receding_horizon import (
    exp1_trust_horizon, exp2_receding, exp3_ranking_vs_horizon, HS,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="L05ar")
    ap.add_argument("--weights", default="weights_epoch_80.pt")
    ap.add_argument("--dirs", nargs="+",
                    default=["traffic_data_cologne8:cologne8",
                             "traffic_data_ingolstadt21:ingolstadt21"])
    ap.add_argument("--episode", type=int, default=3)
    ap.add_argument("--out", default="traffic_data_ingolstadt21/transfer_viz.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load(args.run, args.weights, device)

    Ks = [1, 2, 3, 5, 10, 20]
    scenarios = {}
    for spec in args.dirs:
        dd, label = spec.split(":")
        dd = Path(dd)
        meta = torch.load(dd / "val.pt", weights_only=False)
        n, P, F = meta["n_nodes"], meta["P_max"], meta["node_feature_dim"]
        ni = torch.from_numpy(np.asarray(meta["neighbor_idx"]))
        nm = torch.from_numpy(np.asarray(meta["neighbor_mask"]))
        deg = np.asarray(meta["neighbor_mask"]).sum(1)
        val = torch.load(dd / "val.pt", weights_only=False)
        ep = val["episodes"][min(args.episode, len(val["episodes"]) - 1)]
        pr = fit_pressure_probe(model, dd / "train.pt", n, P, F, ni, nm, device)

        me, pe, _ = exp1_trust_horizon(model, pr, val["episodes"], n, P, F, ni, nm, device, N=20)
        k_m, k_p = [], []
        for K in Ks:
            m, p = exp2_receding(model, pr, ep, n, P, F, ni, nm, device, K)
            k_m.append(m); k_p.append(p)
        om, op = exp2_receding(model, pr, ep, n, P, F, ni, nm, device, None)
        rank, nbr = exp3_ranking_vs_horizon(model, pr, dd / "counterfactual.pt", n, P, F, ni, nm, device)

        scenarios[label] = {
            "n_nodes": int(n), "deg_min": int(deg.min()), "deg_mean": float(deg.mean()),
            "deg_max": int(deg.max()), "probe_mse": pr["mse"], "controller": ep["controller"],
            "trust_h": list(range(1, 21)),
            "trust_ratio": [float(me[i] / pe[i]) for i in range(20)],
            "K": Ks,
            "recede_ratio": [float(a / b) for a, b in zip(k_m, k_p)],
            "recede_model_mse": [float(x) for x in k_m],
            "recede_persist_mse": [float(x) for x in k_p],
            "open_ratio": float(om / op),
            "rank_h": sorted(rank),
            "rank_top1": [rank[h][0] for h in sorted(rank)],
            "rank_spearman": [float(rank[h][1]) for h in sorted(rank)],
            "chance_top1": 1.0 / nbr,
        }
        print(f"{label:>14}  N={n:>2} deg {deg.min()}-{deg.max()} (mean {deg.mean():.1f})  "
              f"K5 ratio {scenarios[label]['recede_ratio'][3]:.2f}  open {scenarios[label]['open_ratio']:.2f}  "
              f"rank@5 top1 {rank[5][0]:.2f} sp {rank[5][1]:.2f}")

    out = {"model": args.run, "trained_on": "cologne8", "scenarios": scenarios}
    Path(args.out).write_text(json.dumps(out))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
