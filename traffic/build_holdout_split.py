"""M3.5 kill test, part 1: controller holdout.

Pools traffic_data_sumo/{train,val}.pt and re-splits by controller identity
rather than randomly, so max_pressure is never seen during training at all -
"do the learned dynamics survive an action distribution they never saw?"
(Implementation Guide M3.5).

Usage:
    python -m traffic.build_holdout_split
"""

from pathlib import Path

import torch

IN_DIR = Path("traffic_data_sumo")
OUT_DIR = Path("traffic_data_sumo/holdout")
HELD_OUT_CONTROLLER = "max_pressure"
VAL_FRACTION = 0.15  # of the non-held-out pool, for loss monitoring only


def main():
    train = torch.load(IN_DIR / "train.pt", weights_only=False)
    val = torch.load(IN_DIR / "val.pt", weights_only=False)
    all_episodes = train["episodes"] + val["episodes"]
    meta = {"state_dim": train["state_dim"], "action_dim": train["action_dim"]}

    seen = [e for e in all_episodes if e["controller"] != HELD_OUT_CONTROLLER]
    held_out = [e for e in all_episodes if e["controller"] == HELD_OUT_CONTROLLER]

    n_val = max(1, int(len(seen) * VAL_FRACTION))
    seen_val, seen_train = seen[:n_val], seen[n_val:]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"episodes": seen_train, **meta}, OUT_DIR / "train.pt")
    torch.save({"episodes": seen_val, **meta}, OUT_DIR / "val.pt")
    torch.save({"episodes": held_out, **meta}, OUT_DIR / "test_maxpressure.pt")

    print(f"train (fixed_time + random only): {len(seen_train)} episodes")
    print(f"val   (fixed_time + random only): {len(seen_val)} episodes")
    print(f"test  ({HELD_OUT_CONTROLLER}, never trained on): {len(held_out)} episodes")


if __name__ == "__main__":
    main()
