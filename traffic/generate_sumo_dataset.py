"""Generate LeWM training trajectories + counterfactual branches from a real
RESCO/SUMO scenario, in the same .pt format traffic/generate_dataset.py
produces for the synthetic CTM generator — so dataset.py, train_traffic.py
and eval_traffic.py need no changes to run on real data.

Usage:
    python -m traffic.generate_sumo_dataset \
        --sumocfg /path/to/resco/environments/cologne1/cologne1.sumocfg \
        --out_dir traffic_data_sumo
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch

from traffic.sumo_env import (
    STEP_LENGTH,
    SumoTLEnv,
    controller_fixed_time,
    controller_max_pressure,
    controller_random,
)

CONTROLLERS = ["fixed_time", "random", "max_pressure"]


def read_n_steps(sumocfg_path):
    """Number of STEP_LENGTH control decisions in the scenario's configured
    [begin, end) window, read directly rather than relying on SUMO to signal
    when it runs out (avoids masking real bugs behind a broad except)."""
    root = ET.parse(sumocfg_path).getroot()
    time = root.find("time")
    begin = float(time.find("begin").get("value")) if time.find("begin") is not None else 0.0
    end = float(time.find("end").get("value"))
    return int((end - begin) // STEP_LENGTH)


def build_policy(env, name, rng):
    if name == "fixed_time":
        cycle = int(rng.integers(4, 10))  # x STEP_LENGTH(5s) = 20-45s green per phase
        return controller_fixed_time(env, cycle=cycle)
    if name == "random":
        return controller_random(env, switch_prob=float(rng.uniform(0.05, 0.25)), rng=rng)
    if name == "max_pressure":
        return controller_max_pressure(env)
    raise ValueError(name)


def rollout_episode(sumocfg, controller_name, seed, n_steps):
    env = SumoTLEnv(sumocfg, seed=seed)
    env.reset()
    rng = np.random.default_rng(seed + 1)
    policy = build_policy(env, controller_name, rng)

    states, actions = [env.state()], []
    for t in range(n_steps):
        phase = policy(t)
        actions.append(env.encode_action(phase))
        states.append(env.step(phase))
    env.close()

    return {
        "state": np.stack(states[:-1]).astype(np.float32),
        "action": np.stack(actions).astype(np.float32),
        "next_state": np.stack(states[1:]).astype(np.float32),
        "controller": controller_name,
    }


def make_split(sumocfg, n_episodes, n_steps, seed_offset):
    episodes = []
    for i in range(n_episodes):
        ctrl = CONTROLLERS[i % len(CONTROLLERS)]
        ep = rollout_episode(sumocfg, ctrl, seed=seed_offset + i, n_steps=n_steps)
        episodes.append(ep)
        print(f"  episode {i + 1}/{n_episodes} ({ctrl}, seed={seed_offset + i}): "
              f"{ep['state'].shape[0]} steps")
    return episodes


def generate_trajectories(sumocfg, out_dir, n_train, n_val, seed):
    n_steps = read_n_steps(sumocfg)
    probe_env = SumoTLEnv(sumocfg, seed=seed)
    probe_env.reset()
    meta = {"state_dim": probe_env.state_dim(), "action_dim": probe_env.action_dim()}
    probe_env.close()

    print(f"state_dim={meta['state_dim']} action_dim={meta['action_dim']} n_steps={n_steps}")
    print("generating train episodes...")
    episodes_train = make_split(sumocfg, n_train, n_steps, seed_offset=seed)
    print("generating val episodes...")
    episodes_val = make_split(sumocfg, n_val, n_steps, seed_offset=seed + 1000)

    torch.save({"episodes": episodes_train, **meta}, out_dir / "train.pt")
    torch.save({"episodes": episodes_val, **meta}, out_dir / "val.pt")
    print(f"saved {n_train} train / {n_val} val episodes to {out_dir}")


def generate_counterfactual(sumocfg, out_dir, n_anchors, horizon, seed):
    # counterfactual branches consume the scenario's own step budget too (each
    # anchor needs a warmup segment on the *same* continuous trajectory before
    # branching) - stop early rather than run past the configured time window.
    step_budget = read_n_steps(sumocfg)

    env = SumoTLEnv(sumocfg, seed=seed)
    env.reset()
    rng = np.random.default_rng(seed + 42)
    warmup_policy = controller_random(env, switch_prob=0.15, rng=rng)

    state_path = str(out_dir / "_cf_snapshot.xml")
    samples = []
    t = 0
    for k in range(n_anchors):
        warmup = int(rng.integers(10, 40))
        if t + warmup >= step_budget:
            print(f"  stopping at {k}/{n_anchors} anchors: scenario step budget "
                  f"({step_budget}) exhausted")
            break
        for _ in range(warmup):
            env.step(warmup_policy(t))
            t += 1

        anchor_state = env.state()
        anchor_action = env.encode_action(env.current_phase)
        snapshot = env.save_state(state_path)

        branches = []
        for phase in range(env.action_dim()):
            env.load_state(snapshot)
            traj_states = [env.state()]
            for _h in range(horizon):
                traj_states.append(env.step(phase))
            branches.append({
                "phases": (phase,),
                "action": env.encode_action(phase),
                "states": np.stack(traj_states).astype(np.float32),
            })

        # continue the underlying trajectory from one branch so later anchors
        # aren't all clustered near t=0
        env.load_state(snapshot)
        env.step(env.current_phase)

        samples.append({
            "anchor_state": anchor_state.astype(np.float32),
            "anchor_action": anchor_action.astype(np.float32),
            "branches": branches,
        })
        print(f"  anchor {k + 1}/{n_anchors} done")

    env.close()
    Path(state_path).unlink(missing_ok=True)

    torch.save(
        {"samples": samples, "state_dim": env.state_dim(), "action_dim": env.action_dim(),
         "horizon": horizon},
        out_dir / "counterfactual.pt",
    )
    print(f"saved {n_anchors} counterfactual anchors x {env.action_dim()} action branches "
          f"(horizon={horizon}) to {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sumocfg", required=True)
    p.add_argument("--out_dir", default="traffic_data_sumo")
    p.add_argument("--n_train", type=int, default=30)
    p.add_argument("--n_val", type=int, default=8)
    p.add_argument("--n_anchors", type=int, default=20)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    generate_trajectories(args.sumocfg, out_dir, args.n_train, args.n_val, args.seed)
    generate_counterfactual(args.sumocfg, out_dir, args.n_anchors, args.horizon, args.seed + 500)


if __name__ == "__main__":
    main()
