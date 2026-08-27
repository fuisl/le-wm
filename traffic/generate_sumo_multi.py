"""Generate multi-signal SUMO trajectories (cologne8) for the topology-free
multi-agent JEPA, plus a counterfactual branch set for plan-ranking eval.

Mirrors traffic/generate_dataset.py (CTM) and traffic/generate_sumo_dataset.py
(single-TL), generalised to N signals with the fixed-F feature contract of
traffic/sumo_multi_env.py.

Usage:
    python -m traffic.generate_sumo_multi --out_dir traffic_data_cologne8 \
        --sumocfg /home/fuisloy/projects/HMARL-TSC/environments/cologne8/cologne8.sumocfg
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import torch

from traffic.sumo_multi_env import (
    SumoMultiEnv,
    controller_fixed_time,
    controller_max_pressure,
    controller_random,
)

CONTROLLERS = ["fixed_time", "random", "max_pressure"]


def build_policy(env, name, rng):
    if name == "fixed_time":
        return controller_fixed_time(env, rng)
    if name == "random":
        return controller_random(env, rng, switch_prob=float(rng.uniform(0.15, 0.4)))
    if name == "max_pressure":
        return controller_max_pressure(env)
    raise ValueError(name)


def rollout_episode(sumocfg, controller_name, T, seed, begin, warmup):
    env = SumoMultiEnv(sumocfg, begin=begin, seed=seed, warmup=warmup)
    env.reset()
    rng = np.random.default_rng(seed + 7)
    policy = build_policy(env, controller_name, rng)

    states, actions = [env.state()], []
    for t in range(T):
        phases = policy(t)
        actions.append(env.encode_action(np.clip(phases, 0, env.P_max - 1)))
        states.append(env.step(phases))
    env.close()

    return {
        "state": np.stack(states[:-1]).astype(np.float32),
        "action": np.stack(actions).astype(np.float32),
        "controller": controller_name,
        "seed": seed,
    }


def make_env_meta(sumocfg, begin, warmup):
    env = SumoMultiEnv(sumocfg, begin=begin, seed=0, warmup=0)
    env.reset()
    idx, mask = env.neighbor_table()
    meta = {
        "n_nodes": env.n_nodes(),
        "node_feature_dim": env.node_feature_dim(),
        "node_action_dim": env.node_action_dim(),
        "state_dim": env.state_dim(),
        "action_dim": env.action_dim(),
        "P_max": env.P_max,
        "neighbor_idx": idx,
        "neighbor_mask": mask,
        "tl_ids": list(env.tl_ids),
        "n_green_phases": [len(env.green_phases[t]) for t in env.tl_ids],
    }
    env.close()
    return meta


def generate_counterfactual(sumocfg, out_dir, n_anchors, horizon, begin, warmup, seed):
    rng = np.random.default_rng(seed)
    samples = []
    for a in range(n_anchors):
        env = SumoMultiEnv(sumocfg, begin=begin, seed=int(rng.integers(0, 1_000_000)),
                           warmup=warmup)
        env.reset()
        N = env.n_nodes()
        # warm to a non-trivial mid-episode state with a random controller
        wrng = np.random.default_rng(int(rng.integers(0, 1_000_000)))
        pol = controller_random(env, wrng, switch_prob=0.25)
        for t in range(int(rng.integers(20, 70))):
            env.step(pol(t))

        anchor_state = env.state().copy()
        anchor_phases = np.array([env.current_phase[t] for t in env.tl_ids])
        anchor_action = env.encode_action(anchor_phases)
        snap = env.save_state(f"/tmp/claude-1000/_cf_{a}.xml")

        # branch set: 4 coordinated (all TLs -> phase p) + 12 random joint vectors
        branch_phase_sets = [np.full(N, p) for p in range(env.P_max)]
        for _ in range(12):
            branch_phase_sets.append(rng.integers(0, env.P_max, size=N))

        branches = []
        for ph in branch_phase_sets:
            env.load_state(snap)
            ph = np.clip(ph, 0, env.P_max - 1)
            traj = [env.state().copy()]
            for _h in range(horizon):
                traj.append(env.step(ph).copy())
            branches.append({
                "phases": tuple(int(x) for x in ph),
                "action": env.encode_action(ph),
                "states": np.stack(traj).astype(np.float32),
            })
        samples.append({
            "anchor_state": anchor_state.astype(np.float32),
            "anchor_action": anchor_action.astype(np.float32),
            "branches": branches,
        })
        env.close()

    meta = make_env_meta(sumocfg, begin, warmup)
    torch.save({"samples": samples, "horizon": horizon, **meta},
               Path(out_dir) / "counterfactual.pt")
    print(f"saved {n_anchors} CF anchors x {len(branch_phase_sets)} branches (H={horizon})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sumocfg", required=True)
    p.add_argument("--out_dir", default="traffic_data_cologne8")
    p.add_argument("--n_train", type=int, default=24)
    p.add_argument("--n_val", type=int, default=6)
    p.add_argument("--T", type=int, default=300)
    p.add_argument("--begin", type=int, default=25200)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--n_anchors", type=int, default=30)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = make_env_meta(args.sumocfg, args.begin, args.warmup)
    print(f"cologne8: N={meta['n_nodes']} F={meta['node_feature_dim']} "
          f"A={meta['node_action_dim']} green_phases={meta['n_green_phases']}")
    print(f"neighbor degrees: {meta['neighbor_mask'].sum(1).tolist()}")

    def make_split(n, seed_off):
        eps = []
        for i in range(n):
            ctrl = CONTROLLERS[i % len(CONTROLLERS)]
            eps.append(rollout_episode(args.sumocfg, ctrl, args.T,
                                       seed=seed_off + i, begin=args.begin,
                                       warmup=args.warmup))
            s = eps[-1]["state"].reshape(args.T, meta["n_nodes"], -1)
            th = s[:, :, :meta["P_max"]].sum(-1).sum(-1).mean()
            print(f"  [{seed_off + i:>3}] {ctrl:<11} mean total-halting/step = {th:.1f}")
        return eps

    train = make_split(args.n_train, args.seed)
    val = make_split(args.n_val, args.seed + 100_000)

    torch.save({"episodes": train, **meta}, out_dir / "train.pt")
    torch.save({"episodes": val, **meta}, out_dir / "val.pt")
    print(f"saved {args.n_train} train / {args.n_val} val (T={args.T}) to {out_dir}")

    generate_counterfactual(args.sumocfg, out_dir, args.n_anchors, args.horizon,
                            args.begin, args.warmup, args.seed + 42)


if __name__ == "__main__":
    main()
