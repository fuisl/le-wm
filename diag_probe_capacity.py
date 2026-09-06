"""Is the halting cost still IN the latent, or merely not linearly decodable?

For each model: fit (a) the ridge probe used by the planner and (b) a 2-layer MLP probe
on train latents -> per-phase halting block; report val MSE in physical units.
If ridge >> MLP the read-out is the bottleneck (value-equivalence: a reward-free latent on a
richer observation does not keep the cost linearly accessible); if both are high the
information is gone.

Usage: python diag_probe_capacity.py --runs L05ar:traffic_data_cologne8 L05ar_link:traffic_data_cologne8_link ...
"""
import argparse
import json

import numpy as np
import torch

from eval_multi_sumo import load, encode_batch
from traffic.dataset import TrafficDataset
from visualize_rollout_cologne8 import RIDGE_LAMBDA


def latents(model, path, n_nodes, P, ni, nm, device):
    ds = TrafficDataset(path, window=4)
    ld = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False)
    X, Y = [], []
    with torch.no_grad():
        for b in ld:
            o = encode_batch(model, b["state"], b["action"], n_nodes, ni, nm, device)
            B, T = b["state"].shape[:2]
            F = b["state"].shape[-1] // n_nodes
            st = b["state"].view(B, T, n_nodes, F)[..., :P]
            z = o["emb"].reshape(B, n_nodes, T, -1).transpose(1, 2)
            X.append(z.reshape(-1, z.shape[-1]).cpu()); Y.append(st.reshape(-1, P).cpu())
    return torch.cat(X), torch.cat(Y)


def ridge(Xtr, Ytr, Xva, Yva):
    ymu, ysd = Ytr.mean(0), Ytr.std(0).clamp(min=1e-6)
    X1 = torch.cat([Xtr, torch.ones(len(Xtr), 1)], 1).double()
    A = X1.T @ X1 + RIDGE_LAMBDA * torch.eye(X1.shape[1], dtype=torch.double)
    W = torch.linalg.solve(A, X1.T @ ((Ytr - ymu) / ysd).double())
    Xv = torch.cat([Xva, torch.ones(len(Xva), 1)], 1).double()
    pred = (Xv @ W).float() * ysd + ymu
    return float(((pred - Yva) ** 2).mean())


def mlp(Xtr, Ytr, Xva, Yva, device, epochs=40):
    ymu, ysd = Ytr.mean(0), Ytr.std(0).clamp(min=1e-6)
    xmu, xsd = Xtr.mean(0), Xtr.std(0).clamp(min=1e-6)
    net = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], 256), torch.nn.GELU(),
                              torch.nn.Linear(256, 256), torch.nn.GELU(),
                              torch.nn.Linear(256, Ytr.shape[1])).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    Xt = ((Xtr - xmu) / xsd).to(device); Yt = ((Ytr - ymu) / ysd).to(device)
    n = len(Xt)
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, 512):
            idx = perm[i:i + 512]
            loss = torch.nn.functional.mse_loss(net(Xt[idx]), Yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = net(((Xva - xmu) / xsd).to(device)).cpu() * ysd + ymu
    return float(((pred - Yva) ** 2).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["L05ar:traffic_data_cologne8"])
    ap.add_argument("--out", default="results/diag_probe_capacity.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    res = {}
    for spec in args.runs:
        run, dd = spec.split(":")
        meta = torch.load(f"{dd}/val.pt", weights_only=False)
        n_nodes, P = meta["n_nodes"], meta["P_max"]
        ni = torch.from_numpy(np.asarray(meta["neighbor_idx"])); nm = torch.from_numpy(np.asarray(meta["neighbor_mask"]))
        model, cfg = load(run, "weights_epoch_80.pt", device)
        Xtr, Ytr = latents(model, f"{dd}/train.pt", n_nodes, P, ni, nm, device)
        Xva, Yva = latents(model, f"{dd}/val.pt", n_nodes, P, ni, nm, device)
        var = float(Yva.var(0).mean())
        r = dict(F=cfg["node_F"], ridge_val_mse=ridge(Xtr, Ytr, Xva, Yva), mlp_val_mse=mlp(Xtr, Ytr, Xva, Yva, device),
                 target_var=var)
        r["ridge_r2"] = 1 - r["ridge_val_mse"] / var; r["mlp_r2"] = 1 - r["mlp_val_mse"] / var
        res[run] = r
        print(f"{run:14s} F={r['F']:3d}  ridge val MSE {r['ridge_val_mse']:6.1f} (R2 {r['ridge_r2']:.3f})   "
              f"MLP val MSE {r['mlp_val_mse']:6.1f} (R2 {r['mlp_r2']:.3f})   target var {var:.1f}", flush=True)
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
