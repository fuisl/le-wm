"""CTM-grid data in the same shape contract as generate_sumo_multi, so the
standalone train_multi_sumo_ar / eval_multi_sumo / test_receding_horizon path
runs on the synthetic 4x4 - the clean coupling testbed where mean-pool coupling
already beat its permute control (2026-08-20). Adds edge_feat + random_joint.

Usage:
    python -m traffic.generate_ctm_multi --out_dir traffic_data_4x4_v2 --rows 4 --cols 4
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import torch

from traffic.ctm_env import (
    CTMGridEnv, controller_fixed_time, controller_max_pressure, controller_random,
)

CONTROLLERS = ["fixed_time", "random", "max_pressure", "random_joint"]


def build_policy(env, name, rng):
    if name == "fixed_time":
        cyc = int(rng.integers(6, 16)); off = rng.integers(0, cyc, size=env.n)
        return controller_fixed_time(env, cycle=cyc, offset=off)
    if name == "random":
        return controller_random(env, switch_prob=float(rng.uniform(0.05, 0.3)), rng=rng)
    if name == "max_pressure":
        return controller_max_pressure(env)
    if name == "random_joint":
        return lambda t: rng.integers(0, 2, size=env.n)
    raise ValueError(name)


def rollout(ctrl, T, seed, rows, cols):
    env = CTMGridEnv(rows=rows, cols=cols, seed=seed)
    env.reset()
    rng = np.random.default_rng(seed + 7)
    pol = build_policy(env, ctrl, rng)
    S, A, E = [env.state()], [], [env.edge_features().reshape(-1)]
    for t in range(T):
        ph = np.asarray(pol(t)).astype(int)
        A.append(env.encode_action(ph))
        S.append(env.step(ph))
        E.append(env.edge_features().reshape(-1))
    return {"state": np.stack(S[:-1]).astype(np.float32),
            "action": np.stack(A).astype(np.float32),
            "edge_feat": np.stack(E[:-1]).astype(np.float32),
            "controller": ctrl, "seed": seed}


def meta_of(rows, cols):
    env = CTMGridEnv(rows=rows, cols=cols, seed=0); env.reset()
    idx, mask = env.neighbor_table()
    return {"n_nodes": env.n, "node_feature_dim": 4, "node_action_dim": 2,
            "state_dim": env.state_dim(), "action_dim": env.action_dim(), "P_max": 2,
            "neighbor_idx": idx, "neighbor_mask": mask, "deg": 4,
            "edge_pair_dim": env.edge_pair_dim(), "rows": rows, "cols": cols}


def gen_cf(out_dir, n_anchors, horizon, rows, cols, seed):
    rng = np.random.default_rng(seed)
    n = rows * cols
    coord = [np.full(n, p) for p in (0, 1)]
    samples = []
    for _ in range(n_anchors):
        env = CTMGridEnv(rows=rows, cols=cols, seed=int(rng.integers(0, 1_000_000)))
        env.reset()
        wr = np.random.default_rng(int(rng.integers(0, 1_000_000)))
        pol = controller_random(env, switch_prob=0.15, rng=wr)
        for t in range(int(rng.integers(20, 80))):
            env.step(pol(t))
        anchor_state = env.state().copy()
        anchor_edge = env.edge_features().reshape(-1).copy()
        snap = env.clone_state()
        branch_sets = coord + [rng.integers(0, 2, size=n) for _ in range(14)]
        branches = []
        for ph in branch_sets:
            env.set_state(snap); ph = np.asarray(ph).astype(int)
            st = [env.state().copy()]; ef = [env.edge_features().reshape(-1).copy()]
            for _h in range(horizon):
                st.append(env.step(ph).copy()); ef.append(env.edge_features().reshape(-1).copy())
            branches.append({"phases": tuple(int(x) for x in ph),
                             "action": env.encode_action(ph),
                             "states": np.stack(st).astype(np.float32),
                             "edge_feat": np.stack(ef).astype(np.float32)})
        samples.append({"anchor_state": anchor_state.astype(np.float32),
                        "anchor_action": env.encode_action(np.zeros(n, int)).astype(np.float32),
                        "anchor_edge": anchor_edge.astype(np.float32), "branches": branches})
    torch.save({"samples": samples, "horizon": horizon, **meta_of(rows, cols)},
               Path(out_dir) / "counterfactual.pt")
    print(f"saved {n_anchors} CF anchors x {len(branch_sets)} branches")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="traffic_data_4x4_v2")
    p.add_argument("--rows", type=int, default=4)
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--n_train", type=int, default=32)
    p.add_argument("--n_val", type=int, default=8)
    p.add_argument("--T", type=int, default=200)
    p.add_argument("--n_anchors", type=int, default=40)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    m = meta_of(a.rows, a.cols)
    print(f"CTM {a.rows}x{a.cols}: N={m['n_nodes']} F=4 A=2 deg=4 edge_pair_dim={m['edge_pair_dim']}")

    def split(n, off):
        eps = []
        for i in range(n):
            ctrl = CONTROLLERS[i % len(CONTROLLERS)]
            eps.append(rollout(ctrl, a.T, off + i, a.rows, a.cols))
            th = eps[-1]["state"].reshape(a.T, m["n_nodes"], 4).sum((1, 2)).mean()
            print(f"  [{off+i:>3}] {ctrl:<12} mean total queue/step = {th:.0f}")
        return eps

    torch.save({"episodes": split(a.n_train, a.seed), **m}, out / "train.pt")
    torch.save({"episodes": split(a.n_val, a.seed + 100_000), **m}, out / "val.pt")
    print(f"saved {a.n_train} train / {a.n_val} val (T={a.T}) to {out}")
    gen_cf(a.out_dir, a.n_anchors, a.horizon, a.rows, a.cols, a.seed + 42)


if __name__ == "__main__":
    main()
