"""M3.5 kill test: is the model action-blind?

Three independent checks, per Implementation Guide M3.5:
  A. Controller holdout - does the model trained without max_pressure survive
     max_pressure action sequences it never saw?
  B. Zeroed-action control - how much worse is a model that can never see the
     action at all, trained/evaluated the same way otherwise?
  C. Action-sensitivity - from the same context, does the prediction actually
     move when the future action sequence is shuffled?

Reuses eval_traffic.py's probe/rollout machinery throughout - no new model
code, only new comparisons.
"""

import json

import numpy as np
import torch

from eval_traffic import apply_probe, fit_linear_probe, latent_rollout, load_model
from traffic.dataset import TrafficDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HISTORY_SIZE = 3
HORIZON = 10


def rollout_error_vs_horizon(model, W, episodes, n_episodes=None, zero_actions=False, shuffle_actions=False, rng=None):
    """Same windowed-rollout evaluation as eval_traffic.eval_rollout_vs_horizon,
    generalized with zero/shuffle switches for the ablations here."""
    episodes = episodes[:n_episodes] if n_episodes else episodes
    lat_err = np.zeros(HORIZON)
    probe_err = np.zeros(HORIZON)
    persist_err = np.zeros(HORIZON)
    count = 0

    with torch.no_grad():
        for ep in episodes:
            T = ep["state"].shape[0]
            span = T - HISTORY_SIZE - HORIZON
            if span <= 0:
                continue
            for start in range(0, span, max(1, span // 4 or 1)):
                state_hist = torch.from_numpy(ep["state"][start:start + HISTORY_SIZE])
                action_hist = torch.from_numpy(ep["action"][start:start + HISTORY_SIZE])
                action_future = torch.from_numpy(
                    ep["action"][start + HISTORY_SIZE:start + HISTORY_SIZE + HORIZON])
                true_states = torch.from_numpy(
                    ep["state"][start + HISTORY_SIZE:start + HISTORY_SIZE + HORIZON]).to(DEVICE)

                if zero_actions:
                    action_hist = torch.zeros_like(action_hist)
                    action_future = torch.zeros_like(action_future)
                if shuffle_actions:
                    perm = rng.permutation(action_future.shape[0])
                    action_future = action_future[perm]

                true_emb = model.encode({"state": true_states.unsqueeze(0)})["emb"].squeeze(0)
                pred_emb = latent_rollout(model, state_hist, action_hist, action_future, HISTORY_SIZE, DEVICE)

                lat_err += ((pred_emb - true_emb) ** 2).mean(-1).cpu().numpy()
                probe_pred = apply_probe(W, pred_emb)
                probe_err += ((probe_pred - true_states) ** 2).mean(-1).cpu().numpy()
                raw_persist = state_hist[-1].to(DEVICE).unsqueeze(0).expand_as(true_states)
                persist_err += ((raw_persist - true_states) ** 2).mean(-1).cpu().numpy()
                count += 1

    for arr in (lat_err, probe_err, persist_err):
        arr /= max(count, 1)
    return {"latent": lat_err.tolist(), "probe_raw_state": probe_err.tolist(),
            "persistence": persist_err.tolist(), "n_windows": count}


def check_a_controller_holdout():
    print("\n=== A. Controller holdout (max_pressure never trained on) ===")
    holdout_model = load_model("traffic-lewm-sumo-holdout", "weights_epoch_60.pt", DEVICE)
    full_model = load_model("traffic-lewm-sumo", "weights_epoch_60.pt", DEVICE)

    W_holdout = fit_linear_probe(holdout_model, "traffic_data_sumo/holdout/val.pt", HISTORY_SIZE + 1, DEVICE)
    W_full = fit_linear_probe(full_model, "traffic_data_sumo/val.pt", HISTORY_SIZE + 1, DEVICE)

    test_data = torch.load("traffic_data_sumo/holdout/test_maxpressure.pt", weights_only=False)
    episodes = test_data["episodes"]

    holdout_result = rollout_error_vs_horizon(holdout_model, W_holdout, episodes)
    full_result = rollout_error_vs_horizon(full_model, W_full, episodes)

    print(f"{'h':>3} {'holdout model':>14} {'full-mix model':>16} {'persistence':>13}")
    for h in range(HORIZON):
        print(f"{h+1:>3} {holdout_result['probe_raw_state'][h]:>14.3f} "
              f"{full_result['probe_raw_state'][h]:>16.3f} {holdout_result['persistence'][h]:>13.3f}")

    return {"holdout_model": holdout_result, "full_mix_model": full_result}


def check_b_zeroed_action():
    print("\n=== B. Zeroed-action control ===")
    real_model = load_model("traffic-lewm-sumo", "weights_epoch_60.pt", DEVICE)
    zero_model = load_model("traffic-lewm-sumo-zeroaction", "weights_epoch_60.pt", DEVICE)

    W_real = fit_linear_probe(real_model, "traffic_data_sumo/val.pt", HISTORY_SIZE + 1, DEVICE)
    W_zero = fit_linear_probe(zero_model, "traffic_data_sumo/val.pt", HISTORY_SIZE + 1, DEVICE)

    val_data = torch.load("traffic_data_sumo/val.pt", weights_only=False)
    episodes = val_data["episodes"]

    real_result = rollout_error_vs_horizon(real_model, W_real, episodes)
    zero_result = rollout_error_vs_horizon(zero_model, W_zero, episodes, zero_actions=True)

    print(f"{'h':>3} {'real model':>12} {'zeroed-action model':>20} {'persistence':>13}")
    for h in range(HORIZON):
        print(f"{h+1:>3} {real_result['probe_raw_state'][h]:>12.3f} "
              f"{zero_result['probe_raw_state'][h]:>20.3f} {real_result['persistence'][h]:>13.3f}")

    return {"real_model": real_result, "zeroed_action_model": zero_result}


def check_c_action_sensitivity():
    print("\n=== C. Action-sensitivity (shuffled action sequence) ===")
    model = load_model("traffic-lewm-sumo", "weights_epoch_60.pt", DEVICE)
    val_data = torch.load("traffic_data_sumo/val.pt", weights_only=False)
    episodes = val_data["episodes"]
    rng = np.random.default_rng(0)

    deltas = []
    with torch.no_grad():
        for ep in episodes:
            T = ep["state"].shape[0]
            span = T - HISTORY_SIZE - HORIZON
            if span <= 0:
                continue
            for start in range(0, span, max(1, span // 3 or 1)):
                state_hist = torch.from_numpy(ep["state"][start:start + HISTORY_SIZE])
                action_hist = torch.from_numpy(ep["action"][start:start + HISTORY_SIZE])
                action_future = torch.from_numpy(
                    ep["action"][start + HISTORY_SIZE:start + HISTORY_SIZE + HORIZON])

                pred_real = latent_rollout(model, state_hist, action_hist, action_future, HISTORY_SIZE, DEVICE)

                perm = rng.permutation(action_future.shape[0])
                pred_shuffled = latent_rollout(
                    model, state_hist, action_hist, action_future[perm], HISTORY_SIZE, DEVICE)

                delta = (pred_real - pred_shuffled).norm(dim=-1).mean().item()
                deltas.append(delta)

    mean_delta = float(np.mean(deltas))
    print(f"mean ||pred(true actions) - pred(shuffled actions)|| = {mean_delta:.3f}  (n={len(deltas)} windows)")
    print("(> 0 means the rollout is action-sensitive; near 0 would mean the predictor ignores the action input)")
    return {"mean_embedding_delta_shuffled_vs_true": mean_delta, "n_windows": len(deltas)}


def main():
    results = {
        "a_controller_holdout": check_a_controller_holdout(),
        "b_zeroed_action": check_b_zeroed_action(),
        "c_action_sensitivity": check_c_action_sensitivity(),
    }
    with open("traffic_data_sumo/m35_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote traffic_data_sumo/m35_results.json")


if __name__ == "__main__":
    main()
