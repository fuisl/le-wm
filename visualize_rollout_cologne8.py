"""Full-episode decoded-rollout visualization for the cologne8 multi-agent JEPA.

For one held-out val episode:
  - ground truth   : real SUMO per-node halting-vehicle counts
  - teacher-forced : encode each true state, decode via probe (1-step, no drift)
  - imagined       : autoregressive latent rollout from a 3-step context using
                     the true action sequence, decoded via probe (compounding)
plus the rollout-error-vs-horizon curve (imagined vs persistence), all three
models (L0 / L05 / L05perm).

Probe upgrade vs eval_multi_sumo: z-scored target + ridge (lambda) instead of
plain lstsq on the raw 0..100+ feature vector, and it targets the per-phase
*pressure* block (P_max dims) whose sum = total halting per node - the
control-relevant quantity - not the phase one-hot / elapsed dims that a
dynamics latent cannot linearly decode anyway.

Usage: python visualize_rollout_cologne8.py
"""

import os
import json
from pathlib import Path

import numpy as np
import torch

from eval_multi_sumo import encode_batch, load, node_rollout
from traffic.dataset import TrafficDataset

DATA_DIR = "traffic_data_cologne8"
RUNS = {"L0": "L0", "L05": "L05", "L05perm": "L05perm"}
WEIGHTS = "weights_epoch_80.pt"
HISTORY = int(os.environ.get("WM_HISTORY", 3))   # follows the model under test
EPISODE_IDX = 3          # val ep 3 = fixed_time (periodic -> real structure to predict)
RIDGE_LAMBDA = 10.0


def fit_pressure_probe(model, path, n_nodes, P, F, nbr_idx, nbr_mask, device):
    """emb -> per-phase pressure block (P dims). z-score target + ridge."""
    ds = TrafficDataset(path, window=HISTORY + 1)
    ld = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False)
    X, Y = [], []
    with torch.no_grad():
        for b in ld:
            o = encode_batch(model, b["state"], b["action"], n_nodes, nbr_idx, nbr_mask, device, edge_feat=b.get("edge_feat"))
            B, T = b["state"].shape[:2]
            st = b["state"].view(B, T, n_nodes, F)[..., :P]
            z = o["emb"].reshape(B, n_nodes, T, -1).transpose(1, 2)
            X.append(z.reshape(-1, z.shape[-1]).cpu())
            Y.append(st.reshape(-1, P).cpu())
    X, Y = torch.cat(X).double(), torch.cat(Y).double()
    ymu, ysd = Y.mean(0), Y.std(0).clamp(min=1e-6)
    Yz = (Y - ymu) / ysd
    if os.environ.get("PROBE", "ridge") == "mlp":
        # T5-lite (2026-09-06): non-linear post-hoc read-out. Same latents, same
        # target; a 2-layer MLP instead of ridge. diag_probe_capacity.py shows the
        # ridge probe loses most of the halting signal (val R2 0.85-0.94) while an
        # MLP recovers it (0.999) on every model.
        torch.manual_seed(0)
        xmu, xsd = X.mean(0), X.std(0).clamp(min=1e-6)
        net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 256), torch.nn.GELU(),
                                  torch.nn.Linear(256, 256), torch.nn.GELU(),
                                  torch.nn.Linear(256, P)).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
        Xt = ((X - xmu) / xsd).float().to(device); Yt = Yz.float().to(device)
        n = len(Xt)
        for _ in range(40):
            perm = torch.randperm(n, device=device)
            for i in range(0, n, 512):
                idx = perm[i:i + 512]
                loss = torch.nn.functional.mse_loss(net(Xt[idx]), Yt[idx])
                opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pred = net(Xt).cpu().double()
        mse_phys = (((pred * ysd + ymu) - Y) ** 2).mean().item()
        return {"net": net, "xmu": xmu.float(), "xsd": xsd.float(), "ymu": ymu.float(),
                "ysd": ysd.float(), "mse": mse_phys, "kind": "mlp"}
    X1 = torch.cat([X, torch.ones(len(X), 1, dtype=torch.double)], 1)
    d = X1.shape[1]
    A = X1.T @ X1 + RIDGE_LAMBDA * torch.eye(d, dtype=torch.double)
    W = torch.linalg.solve(A, X1.T @ Yz)                     # (d, P) maps emb -> z-scored
    pred = X1 @ W
    mse_phys = (((pred * ysd + ymu) - Y) ** 2).mean().item()
    return {"W": W.float(), "ymu": ymu.float(), "ysd": ysd.float(), "mse": mse_phys}


def decode(pr, emb):
    if pr.get("kind") == "mlp":
        dev = emb.device
        with torch.no_grad():
            z = pr["net"].to(dev)((emb - pr["xmu"].to(dev)) / pr["xsd"].to(dev))
        return z * pr["ysd"].to(dev) + pr["ymu"].to(dev)
    ones = torch.ones(*emb.shape[:-1], 1, device=emb.device, dtype=emb.dtype)
    z = torch.cat([emb, ones], -1) @ pr["W"].to(emb.device)
    return z * pr["ysd"].to(emb.device) + pr["ymu"].to(emb.device)


def horizon_curve(model, pr, episodes, n_nodes, P, F, nbr_idx, nbr_mask, HS, n_steps, device):
    me = np.zeros(n_steps); pe = np.zeros(n_steps); cnt = 0
    for ep in episodes:
        T = ep["state"].shape[0]
        span = T - HS - n_steps
        if span <= 0:
            continue
        for s in range(0, span, max(1, span // 20)):
            sh = torch.from_numpy(ep["state"][s:s + HS])
            ah = torch.from_numpy(ep["action"][s:s + HS])
            af = torch.from_numpy(ep["action"][s + HS:s + HS + n_steps])
            true = torch.from_numpy(ep["state"][s + HS:s + HS + n_steps]).to(device).view(n_steps, n_nodes, F)[..., :P].sum(-1)
            eh = torch.from_numpy(ep["edge_feat"][s:s + HS]) if "edge_feat" in ep else None
            pe_emb = node_rollout(model, sh, ah, af, n_nodes, nbr_idx, nbr_mask, HS, device, edge_hist=eh)
            dec = decode(pr, pe_emb).clamp(min=0).sum(-1)     # (n_steps, N) total halting/node
            per = torch.from_numpy(ep["state"][s + HS - 1]).to(device).view(n_nodes, F)[..., :P].sum(-1)
            per = per.unsqueeze(0).expand(n_steps, n_nodes)
            me += ((dec - true) ** 2).mean(1).cpu().numpy()
            pe += ((per - true) ** 2).mean(1).cpu().numpy()
            cnt += 1
    return (me / max(cnt, 1)).tolist(), (pe / max(cnt, 1)).tolist(), cnt


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dd = Path(DATA_DIR)
    meta = torch.load(dd / "val.pt", weights_only=False)
    n_nodes, P, F = meta["n_nodes"], meta["P_max"], meta["node_feature_dim"]
    nbr_idx = torch.from_numpy(np.asarray(meta["neighbor_idx"]))
    nbr_mask = torch.from_numpy(np.asarray(meta["neighbor_mask"]))
    tl_ids = meta["tl_ids"]
    val = torch.load(dd / "val.pt", weights_only=False)
    ep = val["episodes"][EPISODE_IDX]
    states = ep["state"]                       # (T, N*F)
    actions = ep["action"]                     # (T, N*A)
    T = states.shape[0]
    states_n = states.reshape(T, n_nodes, F)
    gt_pressure = states_n[..., :P]            # (T, N, P)
    gt_total = gt_pressure.sum(-1)             # (T, N)

    models_out = {}
    horizon = {}
    for key, run in RUNS.items():
        model, _ = load(run, WEIGHTS, device)
        pr = fit_pressure_probe(model, dd / "train.pt", n_nodes, P, F, nbr_idx, nbr_mask, device)

        # teacher-forced: encode every true state (Embedder is pointwise in time)
        with torch.no_grad():
            ef_full = torch.from_numpy(ep["edge_feat"]).unsqueeze(0) if "edge_feat" in ep else None
            o = encode_batch(model, torch.from_numpy(states).unsqueeze(0),
                             torch.from_numpy(actions).unsqueeze(0), n_nodes,
                             nbr_idx, nbr_mask, device, edge_feat=ef_full)
            emb_tf = o["emb"].reshape(1, n_nodes, T, -1).transpose(1, 2)[0]   # (T, N, d)
            tf_pressure = decode(pr, emb_tf).clamp(min=0).cpu().numpy()       # (T, N, P)

        # imagined: 3-step context, autoregressive for the rest of the episode
        sh = torch.from_numpy(states[:HISTORY])
        ah = torch.from_numpy(actions[:HISTORY])
        af = torch.from_numpy(actions[HISTORY:])
        eh0 = torch.from_numpy(ep["edge_feat"][:HISTORY]) if "edge_feat" in ep else None
        pe_emb = node_rollout(model, sh, ah, af, n_nodes, nbr_idx, nbr_mask, HISTORY, device, edge_hist=eh0)  # (T-H, N, d)
        im_pressure = decode(pr, pe_emb).clamp(min=0).detach().cpu().numpy()  # (T-H, N, P)

        models_out[key] = {
            "probe_mse": pr["mse"],
            "teacher_forced_total": tf_pressure.sum(-1).T.tolist(),   # (N, T)
            "imagined_total": im_pressure.sum(-1).T.tolist(),         # (N, T-H)
            "network_total_teacher_forced": tf_pressure.sum(-1).sum(-1).tolist(),  # (T,)
            "network_total_imagined": im_pressure.sum(-1).sum(-1).tolist(),        # (T-H,)
            "teacher_forced_pressure_node0": tf_pressure[:, 0, :].tolist(),
            "imagined_pressure_node0": im_pressure[:, 0, :].tolist(),
        }
        me, pe_, cnt = horizon_curve(model, pr, val["episodes"], n_nodes, P, F,
                                     nbr_idx, nbr_mask, HISTORY, 12, device)
        horizon[key] = me
        horizon["_persistence"] = pe_
        horizon["_n_windows"] = cnt

    out = {
        "meta": {
            "scenario": "cologne8 (RESCO, TAPAS-Cologne demand) - 8 signals",
            "controller": ep["controller"],
            "episode_idx": EPISODE_IDX,
            "step_length_s": 5,
            "history_size": HISTORY,
            "episode_len": T,
            "n_nodes": n_nodes,
            "P_max": P,
            "tl_ids": tl_ids,
            "neighbor_degrees": nbr_mask.sum(1).tolist(),
            "checkpoint": WEIGHTS,
            "ridge_lambda": RIDGE_LAMBDA,
        },
        "ground_truth_total": gt_total.T.tolist(),        # (N, T)
        "ground_truth_pressure_node0": gt_pressure[:, 0, :].tolist(),  # (T, P)
        "network_total_ground_truth": gt_total.sum(1).tolist(),        # (T,)
        "models": models_out,
        "horizon_curves": horizon,
    }
    out_path = dd / "rollout_viz_cologne8.json"
    out_path.write_text(json.dumps(out))
    print(f"wrote {out_path}")
    print("probe phys MSE (total-halting units^2, per phase-dim):",
          {k: round(v["probe_mse"], 2) for k, v in models_out.items()})
    print("horizon curve (imagined total-halting/node MSE) h1/h6/h12:")
    for k in RUNS:
        c = horizon[k]
        print(f"  {k:<8} {c[0]:.1f} / {c[5]:.1f} / {c[11]:.1f}   (persist {horizon['_persistence'][0]:.1f} / "
              f"{horizon['_persistence'][5]:.1f} / {horizon['_persistence'][11]:.1f})")


if __name__ == "__main__":
    main()
