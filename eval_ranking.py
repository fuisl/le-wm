"""Plan-ranking accuracy: the go/no-go gate for building a CEM planner.

Counterfactual correlation (eval_traffic.py's eval_counterfactual) measures
whether predicted-embedding divergence tracks true divergence in *magnitude*.
That is not what a planner needs. A CEM planner only has to rank candidate
action branches correctly and execute the best-ranked one - so the number
that actually predicts planner viability is ranking accuracy, not magnitude
correlation ("Sign agreement is the number that matters - MPC only needs the
ordering right", per the Implementation Guide's M4.5).

For each of the 20 counterfactual anchors already on disk (8-way full phase
enumeration, 5-step/25s horizon), this computes a running queue cost -
summed decoded halting-vehicle count over the branch horizon, the same
quantity sumo_env.py's own controller_max_pressure heuristic uses per step -
from both the model's imagined rollout (via the frozen linear probe) and the
true SUMO trajectory. Symlog normalisation (Final Architecture v1's v1 spec)
is skipped here: it's a monotonic transform and does not change rank order,
so it's irrelevant to this specific test even though it matters once this
cost is used inside an actual CEM planner.

Reuses eval_traffic.py's load_model / fit_linear_probe / apply_probe /
latent_rollout - no new model code.

Usage:
    python eval_ranking.py --run_name traffic-lewm-sumo --weights weights_epoch_60.pt
"""

import argparse

import numpy as np
import torch
from scipy.stats import spearmanr

from eval_traffic import apply_probe, fit_linear_probe, latent_rollout, load_model


def branch_costs(model, W, sample, history_size, device):
    """Predicted and true running cost (summed decoded halting count over the
    branch horizon) for every action branch from one counterfactual anchor.
    Lower cost = less congestion = the action a planner should prefer."""
    anchor_state = torch.from_numpy(sample["anchor_state"])
    anchor_action = torch.from_numpy(sample["anchor_action"])
    state_hist = anchor_state.unsqueeze(0).expand(history_size, -1).contiguous()
    action_hist = anchor_action.unsqueeze(0).expand(history_size, -1).contiguous()

    pred_costs, true_costs = [], []
    for br in sample["branches"]:
        horizon = br["states"].shape[0] - 1
        action_future = torch.from_numpy(br["action"]).unsqueeze(0).expand(horizon, -1).contiguous()

        pred_emb = latent_rollout(model, state_hist, action_hist, action_future, history_size, device)
        pred_state = apply_probe(W, pred_emb)  # (horizon, state_dim)
        pred_costs.append(pred_state.sum().item())

        true_state = torch.from_numpy(br["states"][1:])  # (horizon, state_dim), drop anchor step
        true_costs.append(true_state.sum().item())

    return np.array(pred_costs), np.array(true_costs)


def eval_ranking(model, W, cf_path, history_size, device):
    data = torch.load(cf_path, weights_only=False)
    samples = data["samples"]

    top1_hits, spearmans, n_branches = [], [], None
    for sample in samples:
        pred_costs, true_costs = branch_costs(model, W, sample, history_size, device)
        n_branches = len(pred_costs)

        top1_hits.append(int(np.argmin(pred_costs) == np.argmin(true_costs)))
        rho, _ = spearmanr(pred_costs, true_costs)
        spearmans.append(0.0 if np.isnan(rho) else rho)

    top1_hits = np.array(top1_hits)
    spearmans = np.array(spearmans)
    chance = 1.0 / n_branches

    print(f"\n=== plan-ranking accuracy ({len(samples)} anchors, {n_branches}-way branch comparison) ===")
    print(f"top-1 agreement (argmin predicted cost == argmin true cost): "
          f"{top1_hits.mean():.3f} ({top1_hits.sum()}/{len(top1_hits)}); chance = {chance:.3f}")
    print(f"mean Spearman rank correlation, predicted vs true branch ordering: "
          f"{spearmans.mean():.3f} (std {spearmans.std():.3f})")
    print("\nper-anchor: top1_hit, spearman")
    for i, (h, r) in enumerate(zip(top1_hits, spearmans)):
        print(f"  anchor {i:>2}: {int(h)}  {r:+.3f}")

    return top1_hits, spearmans


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", default="traffic-lewm-sumo")
    p.add_argument("--weights", default="weights_epoch_60.pt")
    p.add_argument("--data_dir", default="traffic_data_sumo")
    p.add_argument("--history_size", type=int, default=3)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.run_name, args.weights, device)

    val_path = f"{args.data_dir}/val.pt"
    cf_path = f"{args.data_dir}/counterfactual.pt"
    window = args.history_size + 1

    W = fit_linear_probe(model, val_path, window, device)
    eval_ranking(model, W, cf_path, args.history_size, device)


if __name__ == "__main__":
    main()
