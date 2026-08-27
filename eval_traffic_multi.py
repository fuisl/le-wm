"""Compare Level 0 / Level 0.5 / Level 0.5+permute-control checkpoints
(traffic/multi_agent.py) against a raw-state persistence baseline.

Mirrors eval_traffic.py's methodology (linear probe from frozen embedding
back to raw state, rollout error vs horizon, model vs persistence) but
node-aware: the probe and rollout operate per-node, with masked neighbor
pooling recomputed at every autoregressive step for level 0.5 (each step's
pooled input depends on that step's own embeddings, which change as the
rollout unrolls).

Latent-space MSE is not reported as a headline number here - Level 0 and
Level 0.5 learn different latent spaces (different predictor input dims),
so raw pred_loss values are not comparable across levels. Everything that
IS compared across levels goes through the probe back to raw queue counts,
same units, same ground truth, matching eval_traffic.py's convention.

Usage:
    python eval_traffic_multi.py --run_name cmp-level0
    python eval_traffic_multi.py --run_name cmp-level05
    python eval_traffic_multi.py --run_name cmp-level05-permute
"""

import argparse

import numpy as np
import torch

from eval_traffic import apply_probe, load_model
from traffic.ctm_env import CTMGridEnv
from traffic.dataset import TrafficDataset
from traffic.multi_agent import masked_neighbor_mean


def fit_node_probe(model, val_path, window, n_nodes, neighbor_idx, neighbor_mask, device):
    """Linear probe: own per-node embedding -> own raw per-node state (n*F ->
    pooled over the (node, time) axis, one shared probe for all nodes - same
    spirit as the model itself: nothing node-specific)."""
    ds = TrafficDataset(val_path, window=window)
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False)
    embs, states = [], []
    with torch.no_grad():
        for batch in loader:
            info = {
                "state": batch["state"].to(device),
                "action": batch["action"].to(device),
                "n_nodes": n_nodes,
                "neighbor_idx": neighbor_idx.to(device),
                "neighbor_mask": neighbor_mask.to(device),
            }
            out = model.encode(info)
            B, T = batch["state"].shape[:2]
            F = batch["state"].shape[-1] // n_nodes
            state_n = batch["state"].view(B, T, n_nodes, F)
            own_emb = out["emb"].reshape(B, n_nodes, T, -1).transpose(1, 2)  # (B,T,N,d)
            embs.append(own_emb.reshape(-1, own_emb.shape[-1]).cpu())
            states.append(state_n.reshape(-1, F).cpu())
    X = torch.cat(embs)
    Y = torch.cat(states)
    X1 = torch.cat([X, torch.ones(X.shape[0], 1)], dim=1)
    W, *_ = torch.linalg.lstsq(X1, Y)
    probe_mse = torch.nn.functional.mse_loss(X1 @ W, Y).item()
    print(f"[probe] fit on {X.shape[0]} (node-emb, node-state) pairs, train MSE={probe_mse:.4f}")
    return W


def node_rollout(model, state_hist, action_hist, action_future, n_nodes, neighbor_idx, neighbor_mask,
                  history_size, device):
    """Autoregressive rollout of own per-node embeddings.

    state_hist / action_hist: (H, n*F) / (H, n*A) context, flat like the dataset.
    action_future: (n_steps, n*A).
    returns: (n_steps, N, d) predicted own embeddings.
    """
    neighbor_idx = neighbor_idx.to(device)
    neighbor_mask = neighbor_mask.to(device)

    with torch.no_grad():
        info = {
            "state": state_hist.unsqueeze(0).to(device),
            "action": action_hist.unsqueeze(0).to(device),
            "n_nodes": n_nodes,
            "neighbor_idx": neighbor_idx,
            "neighbor_mask": neighbor_mask,
        }
        info = model.encode(info)
        H = state_hist.shape[0]
        z = info["emb"].reshape(1, n_nodes, H, -1).transpose(1, 2)          # (1,H,N,d) own
        act = info["act_emb"].reshape(1, n_nodes, H, -1).transpose(1, 2)    # (1,H,N,a_emb)

        preds = []
        for t in range(action_future.shape[0]):
            HS = history_size
            z_ctx = z[:, -HS:]    # (1,HS,N,d)
            act_ctx = act[:, -HS:]

            if model.level == "0.5":
                z_bn = z_ctx.transpose(1, 2)      # (1,N,HS,d)
                act_bn = act_ctx.transpose(1, 2)  # (1,N,HS,a_emb)
                z_pool = masked_neighbor_mean(z_bn, neighbor_idx, neighbor_mask)
                act_pool = masked_neighbor_mean(act_bn, neighbor_idx, neighbor_mask)
                if model.permute_control:
                    perm = torch.randperm(n_nodes, device=device)
                    z_pool = z_pool[:, perm]
                    act_pool = act_pool[:, perm]
                pred_in_emb = torch.cat([z_bn, z_pool], dim=-1)
                pred_in_act = torch.cat([act_bn, act_pool], dim=-1)
            else:
                pred_in_emb = z_ctx.transpose(1, 2)
                pred_in_act = act_ctx.transpose(1, 2)

            pred_in_emb = pred_in_emb.reshape(n_nodes, HS, -1)
            pred_in_act = pred_in_act.reshape(n_nodes, HS, -1)
            pred_next = model.predict(pred_in_emb, pred_in_act)[:, -1:]  # (N,1,d)
            pred_next = pred_next.reshape(1, 1, n_nodes, -1)

            preds.append(pred_next)
            z = torch.cat([z, pred_next], dim=1)

            next_action = action_future[t : t + 1].unsqueeze(0).to(device)  # (1,1,n*A)
            A = next_action.shape[-1] // n_nodes
            next_action_n = next_action.view(1, 1, n_nodes, A).reshape(n_nodes, 1, A)
            next_act_emb = model.action_encoder(next_action_n)  # (N,1,a_emb)
            act = torch.cat([act, next_act_emb.reshape(1, 1, n_nodes, -1)], dim=1)

        return torch.cat(preds, dim=1).squeeze(0)  # (n_steps, N, d)


def eval_rollout_vs_horizon(model, W, val_path, rows, cols, history_size, n_steps, n_episodes, device):
    n_nodes = rows * cols
    neighbor_idx_np, neighbor_mask_np = CTMGridEnv(rows=rows, cols=cols).neighbor_table()
    neighbor_idx = torch.from_numpy(neighbor_idx_np)
    neighbor_mask = torch.from_numpy(neighbor_mask_np)

    data = torch.load(val_path, weights_only=False)
    episodes = data["episodes"][:n_episodes]

    probe_model_err = np.zeros(n_steps)
    raw_persist_err = np.zeros(n_steps)
    count = 0

    with torch.no_grad():
        for ep in episodes:
            T = ep["state"].shape[0]
            span = T - history_size - n_steps
            if span <= 0:
                continue
            for start in range(0, span, max(1, span // 4 or 1)):
                state_hist = torch.from_numpy(ep["state"][start : start + history_size])
                action_hist = torch.from_numpy(ep["action"][start : start + history_size])
                action_future = torch.from_numpy(ep["action"][start + history_size : start + history_size + n_steps])
                true_states = torch.from_numpy(
                    ep["state"][start + history_size : start + history_size + n_steps]
                ).to(device).view(n_steps, n_nodes, -1)

                pred_emb = node_rollout(
                    model, state_hist, action_hist, action_future, n_nodes,
                    neighbor_idx, neighbor_mask, history_size, device,
                )  # (n_steps, N, d)

                probe_pred_state = apply_probe(W, pred_emb)  # (n_steps, N, F)
                probe_model_err += ((probe_pred_state - true_states) ** 2).mean(dim=(1, 2)).cpu().numpy()

                raw_persist = state_hist[-1].to(device).view(1, n_nodes, -1).expand_as(true_states)
                raw_persist_err += ((raw_persist - true_states) ** 2).mean(dim=(1, 2)).cpu().numpy()

                count += 1

    probe_model_err /= max(count, 1)
    raw_persist_err /= max(count, 1)

    print(f"\n=== rollout error vs horizon ({count} windows, n_nodes={n_nodes}) ===")
    print(f"{'h':>3} {'probe(model)':>13} {'raw(persist)':>13} {'model < persist?':>18}")
    for h in range(n_steps):
        beats = "yes" if probe_model_err[h] < raw_persist_err[h] else "NO"
        print(f"{h+1:>3} {probe_model_err[h]:>13.4f} {raw_persist_err[h]:>13.4f} {beats:>18}")
    return probe_model_err, raw_persist_err


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", required=True)
    p.add_argument("--weights", default="weights_epoch_60.pt")
    p.add_argument("--data_dir", default="traffic_data")
    p.add_argument("--history_size", type=int, default=3)
    p.add_argument("--rollout_steps", type=int, default=10)
    p.add_argument("--n_episodes", type=int, default=40)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.run_name, args.weights, device)

    train_path = f"{args.data_dir}/train.pt"
    val_path = f"{args.data_dir}/val.pt"
    val_meta = torch.load(val_path, weights_only=False)
    rows, cols = val_meta["rows"], val_meta["cols"]
    n_nodes = rows * cols
    window = args.history_size + 1

    neighbor_idx_np, neighbor_mask_np = CTMGridEnv(rows=rows, cols=cols).neighbor_table()
    neighbor_idx = torch.from_numpy(neighbor_idx_np)
    neighbor_mask = torch.from_numpy(neighbor_mask_np)

    # probe fit on train, rollout scored on val - fitting and scoring on the
    # same split would let differences in latent geometry (not dynamics)
    # leak into the cross-model comparison.
    W = fit_node_probe(model, train_path, window, n_nodes, neighbor_idx, neighbor_mask, device)
    eval_rollout_vs_horizon(model, W, val_path, rows, cols, args.history_size,
                             args.rollout_steps, args.n_episodes, device)


if __name__ == "__main__":
    main()
