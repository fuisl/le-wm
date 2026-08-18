"""Generate CTM traffic trajectories for LeWM training, and a separate
counterfactual branch set for the "does the model answer control questions"
eval (same anchor state, different actions, different true futures).

Usage:
    python -m traffic.generate_dataset --out_dir traffic_data
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import torch

from traffic.ctm_env import (
    CTMGridEnv,
    controller_fixed_time,
    controller_max_pressure,
    controller_random,
)

CONTROLLERS = ["fixed_time", "random", "max_pressure"]


def build_policy(env, name, rng):
    if name == "fixed_time":
        cycle = int(rng.integers(6, 16))
        offset = rng.integers(0, cycle, size=env.n)
        return controller_fixed_time(env, cycle=cycle, offset=offset)
    if name == "random":
        return controller_random(env, switch_prob=float(rng.uniform(0.05, 0.3)), rng=rng)
    if name == "max_pressure":
        return controller_max_pressure(env)
    raise ValueError(name)


def rollout_episode(controller_name, T, seed, rows, cols):
    env = CTMGridEnv(rows=rows, cols=cols, seed=seed)
    env.reset()
    rng = np.random.default_rng(seed + 1)  # separate stream from env dynamics noise
    policy = build_policy(env, controller_name, rng)

    states, actions = [env.state()], []
    for t in range(T):
        phases = policy(t)
        actions.append(env.encode_action(phases))
        states.append(env.step(phases))

    return {
        "state": np.stack(states[:-1]).astype(np.float32),
        "action": np.stack(actions).astype(np.float32),
        "next_state": np.stack(states[1:]).astype(np.float32),
        "controller": controller_name,
    }


def make_split(n_episodes, T, rows, cols, seed_offset):
    episodes = []
    for i in range(n_episodes):
        ctrl = CONTROLLERS[i % len(CONTROLLERS)]
        episodes.append(rollout_episode(ctrl, T, seed=seed_offset + i, rows=rows, cols=cols))
    return episodes


def generate_trajectories(out_dir, n_train, n_val, T, rows, cols, seed):
    episodes_train = make_split(n_train, T, rows, cols, seed_offset=seed)
    episodes_val = make_split(n_val, T, rows, cols, seed_offset=seed + 1_000_000)

    meta = {"state_dim": rows * cols * 4, "action_dim": rows * cols * 2, "rows": rows, "cols": cols}
    torch.save({"episodes": episodes_train, **meta}, out_dir / "train.pt")
    torch.save({"episodes": episodes_val, **meta}, out_dir / "val.pt")
    print(f"saved {n_train} train / {n_val} val episodes (T={T}) to {out_dir}")


def generate_counterfactual(out_dir, n_anchors, horizon, rows, cols, seed):
    n = rows * cols
    rng = np.random.default_rng(seed)
    all_actions = list(itertools.product([0, 1], repeat=n))

    samples = []
    for _ in range(n_anchors):
        env = CTMGridEnv(rows=rows, cols=cols, seed=int(rng.integers(0, 1_000_000)))
        env.reset()
        warm_rng = np.random.default_rng(int(rng.integers(0, 1_000_000)))
        policy = controller_random(env, switch_prob=0.15, rng=warm_rng)
        for t in range(int(rng.integers(20, 80))):
            env.step(policy(t))

        anchor_state = env.state()
        anchor_action = env.encode_action(env.phase)  # phase active going into the anchor state
        snapshot = env.clone_state()

        branches = []
        for phases in all_actions:
            env.set_state(snapshot)
            phases_arr = np.array(phases)
            traj_states = [env.state()]
            for _h in range(horizon):
                traj_states.append(env.step(phases_arr))
            branches.append({
                "phases": phases,
                "action": env.encode_action(phases_arr),
                "states": np.stack(traj_states).astype(np.float32),  # (horizon+1, D)
            })

        samples.append({
            "anchor_state": anchor_state.astype(np.float32),
            "anchor_action": anchor_action.astype(np.float32),
            "branches": branches,
        })

    torch.save(
        {"samples": samples, "state_dim": n * 4, "action_dim": n * 2,
         "horizon": horizon, "rows": rows, "cols": cols},
        out_dir / "counterfactual.pt",
    )
    print(f"saved {n_anchors} counterfactual anchors x {len(all_actions)} action branches "
          f"(horizon={horizon}) to {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="traffic_data")
    p.add_argument("--rows", type=int, default=2)
    p.add_argument("--cols", type=int, default=2)
    p.add_argument("--n_train", type=int, default=300)
    p.add_argument("--n_val", type=int, default=40)
    p.add_argument("--T", type=int, default=120)
    p.add_argument("--n_anchors", type=int, default=60)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    generate_trajectories(out_dir, args.n_train, args.n_val, args.T, args.rows, args.cols, args.seed)
    generate_counterfactual(out_dir, args.n_anchors, args.horizon, args.rows, args.cols, args.seed + 42)


if __name__ == "__main__":
    main()
