"""Evaluate a trained traffic-LeWM checkpoint against the two checks that
distinguish "learned traffic dynamics" from "learned that traffic is smooth":

1. Rollout error vs horizon, model vs a persistence baseline (latent space,
   and via a linear probe back to raw queue counts).
2. Counterfactual accuracy: from an identical anchor state, does the model's
   predicted embedding change across different signal-phase choices in a way
   that tracks the true divergence in raw state? (Pre-Pitch Demo figure 1.)

Usage:
    python eval_traffic.py --run_name traffic-lewm --weights weights_epoch_60.pt
"""

import argparse

import numpy as np
import torch

import stable_worldmodel as swm
from traffic.dataset import TrafficDataset


def load_model(run_name, filename, device):
    model = swm.wm.utils.load_pretrained(f"{run_name}/{filename}")
    model.to(device).eval()
    return model


def fit_linear_probe(model, val_path, window, device):
    """Least-squares probe: frozen embedding -> raw state, fit on val data."""
    ds = TrafficDataset(val_path, window=window)
    loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False)
    embs, states = [], []
    with torch.no_grad():
        for batch in loader:
            info = {"state": batch["state"].to(device)}
            out = model.encode(info)
            embs.append(out["emb"].reshape(-1, out["emb"].shape[-1]).cpu())
            states.append(batch["state"].reshape(-1, batch["state"].shape[-1]).cpu())
    X = torch.cat(embs)
    Y = torch.cat(states)
    X1 = torch.cat([X, torch.ones(X.shape[0], 1)], dim=1)
    W, *_ = torch.linalg.lstsq(X1, Y)
    probe_mse = torch.nn.functional.mse_loss(X1 @ W, Y).item()
    print(f"[probe] fit on {X.shape[0]} (emb, state) pairs, train MSE={probe_mse:.4f}")
    return W


def apply_probe(W, emb):
    ones = torch.ones(*emb.shape[:-1], 1, device=emb.device, dtype=emb.dtype)
    emb1 = torch.cat([emb, ones], dim=-1)
    return emb1 @ W.to(emb.device)


def latent_rollout(model, state_hist, action_hist, action_future, history_size, device):
    """Autoregressive latent rollout, mirroring JEPA.rollout() but keyed on 'state'.

    state_hist / action_hist: (H, D) / (H, A) context.
    action_future: (n_steps, A) actions to imagine forward.
    returns predicted embeddings for the n_steps imagined future frames: (n_steps, d)
    """
    with torch.no_grad():
        info = {"state": state_hist.unsqueeze(0).to(device), "action": action_hist.unsqueeze(0).to(device)}
        info = model.encode(info)
        emb = info["emb"]  # (1, H, d)
        act = action_hist.unsqueeze(0).to(device)

        preds = []
        for t in range(action_future.shape[0]):
            act_emb = model.action_encoder(act)
            pred = model.predict(emb[:, -history_size:], act_emb[:, -history_size:])[:, -1:]
            preds.append(pred)
            emb = torch.cat([emb, pred], dim=1)
            next_a = action_future[t : t + 1].unsqueeze(0).to(device)
            act = torch.cat([act, next_a], dim=1)
        return torch.cat(preds, dim=1).squeeze(0)  # (n_steps, d)


def eval_rollout_vs_horizon(model, W, val_path, history_size, n_steps, n_episodes, device):
    data = torch.load(val_path, weights_only=False)
    episodes = data["episodes"][:n_episodes]

    lat_model_err = np.zeros(n_steps)
    lat_persist_err = np.zeros(n_steps)
    probe_model_err = np.zeros(n_steps)
    raw_persist_err = np.zeros(n_steps)
    count = 0

    with torch.no_grad():
        for ep in episodes:
            T = ep["state"].shape[0]
            for start in range(0, T - history_size - n_steps, max(1, (T - history_size - n_steps) // 4 or 1)):
                state_hist = torch.from_numpy(ep["state"][start : start + history_size])
                action_hist = torch.from_numpy(ep["action"][start : start + history_size])
                action_future = torch.from_numpy(ep["action"][start + history_size : start + history_size + n_steps])
                true_states = torch.from_numpy(ep["state"][start + history_size : start + history_size + n_steps]).to(device)

                true_emb = model.encode({"state": true_states.unsqueeze(0)})["emb"].squeeze(0)  # (n_steps, d)

                pred_emb = latent_rollout(model, state_hist, action_hist, action_future, history_size, device)
                last_ctx_emb = model.encode({"state": state_hist.unsqueeze(0).to(device)})["emb"][0, -1]  # (d,)
                persist_emb = last_ctx_emb.unsqueeze(0).expand_as(pred_emb)

                lat_model_err += ((pred_emb - true_emb) ** 2).mean(-1).cpu().numpy()
                lat_persist_err += ((persist_emb - true_emb) ** 2).mean(-1).cpu().numpy()

                probe_pred_state = apply_probe(W, pred_emb)
                probe_model_err += ((probe_pred_state - true_states) ** 2).mean(-1).cpu().numpy()

                raw_persist = state_hist[-1].to(device).unsqueeze(0).expand_as(true_states)
                raw_persist_err += ((raw_persist - true_states) ** 2).mean(-1).cpu().numpy()

                count += 1

    for arr in (lat_model_err, lat_persist_err, probe_model_err, raw_persist_err):
        arr /= max(count, 1)

    print(f"\n=== rollout error vs horizon ({count} windows) ===")
    print(f"{'h':>3} {'latent(model)':>14} {'latent(persist)':>16} {'probe(model)':>13} {'raw(persist)':>13}")
    for h in range(n_steps):
        print(f"{h+1:>3} {lat_model_err[h]:>14.4f} {lat_persist_err[h]:>16.4f} "
              f"{probe_model_err[h]:>13.4f} {raw_persist_err[h]:>13.4f}")


def eval_counterfactual(model, cf_path, history_size, device):
    data = torch.load(cf_path, weights_only=False)
    samples = data["samples"]

    true_deltas, pred_deltas = [], []
    for sample in samples:
        anchor_state = torch.from_numpy(sample["anchor_state"])
        anchor_action = torch.from_numpy(sample["anchor_action"])
        state_hist = anchor_state.unsqueeze(0).expand(history_size, -1).contiguous()
        action_hist = anchor_action.unsqueeze(0).expand(history_size, -1).contiguous()

        branch_pred_embs, branch_true_states = [], []
        for br in sample["branches"]:
            horizon = br["states"].shape[0] - 1
            action_future = torch.from_numpy(br["action"]).unsqueeze(0).expand(horizon, -1).contiguous()
            pred_emb = latent_rollout(model, state_hist, action_hist, action_future, history_size, device)
            branch_pred_embs.append(pred_emb[-1])  # embedding at final horizon step
            branch_true_states.append(torch.from_numpy(br["states"][-1]).to(device))  # true final state

        # all pairwise branch differences, since actions are a joint one-hot per intersection
        n_branches = len(branch_pred_embs)
        for i in range(n_branches):
            for j in range(i + 1, n_branches):
                pred_delta = (branch_pred_embs[i] - branch_pred_embs[j]).norm().item()
                true_delta = (branch_true_states[i] - branch_true_states[j]).norm().item()
                pred_deltas.append(pred_delta)
                true_deltas.append(true_delta)

    pred_deltas = np.array(pred_deltas)
    true_deltas = np.array(true_deltas)
    corr = np.corrcoef(pred_deltas, true_deltas)[0, 1]

    print(f"\n=== counterfactual accuracy ({len(samples)} anchors, {len(pred_deltas)} action-pairs) ===")
    print(f"correlation(||Δ predicted embedding||, ||Δ true final state||) = {corr:.3f}")
    print(f"mean true delta (branches genuinely diverge if >0): {true_deltas.mean():.3f}")
    print(f"mean predicted delta (model is action-sensitive if >0): {pred_deltas.mean():.3f}")
    zero_action_pairs = (true_deltas < 1e-6).mean()
    print(f"fraction of anchor pairs with ~zero true divergence: {zero_action_pairs:.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", default="traffic-lewm")
    p.add_argument("--weights", default="weights_epoch_60.pt")
    p.add_argument("--data_dir", default="traffic_data")
    p.add_argument("--history_size", type=int, default=3)
    p.add_argument("--rollout_steps", type=int, default=10)
    p.add_argument("--n_episodes", type=int, default=20)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.run_name, args.weights, device)

    val_path = f"{args.data_dir}/val.pt"
    cf_path = f"{args.data_dir}/counterfactual.pt"
    window = args.history_size + 1

    W = fit_linear_probe(model, val_path, window, device)
    eval_rollout_vs_horizon(model, W, val_path, args.history_size, args.rollout_steps, args.n_episodes, device)
    eval_counterfactual(model, cf_path, args.history_size, device)


if __name__ == "__main__":
    main()
