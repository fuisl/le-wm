"""#7 AC post-training corpus: planner-matched trajectories the base L05ar model
never saw (24 episodes of fixed_time / random / max_pressure only).

Adds three episode types, appended to the ORIGINAL train/val episodes so the model
still sees scripted behaviour:

  random_joint : every signal draws an independent uniform phase EVERY step
                 = the factored CEM iteration-0 action distribution.
  eps_phase    : max_pressure base, but each signal's phase is resampled to a
                 uniform-random phase with prob `eps` each step
                 = an intermediate joint-switching regime between scripted and CEM.
  cem_elite    : closed-loop latent-CEM with the base model; the CEM's chosen joint
                 phase is EXECUTED in real SUMO each step and the trajectory recorded
                 = on-distribution DAgger data for the planner's own query pattern.

Usage:
  SUMOCFG_COLOGNE8=/path/cologne8.sumocfg \
  python -m traffic.generate_augment_sumo --out_dir traffic_data_cologne8_aug \
      --base_dir traffic_data_cologne8 --n_random_joint 6 --n_eps 6 --n_cem 3 \
      --cem_model L05ar --T 300 --cem_T 150
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch

from traffic.sumo_multi_env import SumoMultiEnv, controller_max_pressure


def _sumocfg(cli):
    for c in (cli, os.environ.get("SUMOCFG_COLOGNE8"),
              "/home/fuisloy/projects/cair/resco/resco_benchmark/environments/cologne8/cologne8.sumocfg"):
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError("no cologne8.sumocfg found; pass --sumocfg or set SUMOCFG_COLOGNE8")


def rollout_random_joint(cfg, T, seed, begin, warmup):
    env = SumoMultiEnv(cfg, begin=begin, seed=seed, warmup=warmup)
    env.reset()
    rng = np.random.default_rng(seed + 7)
    P = env.P_max
    states, actions = [env.state()], []
    for _ in range(T):
        ph = rng.integers(0, P, size=len(env.tl_ids))
        actions.append(env.encode_action(np.clip(ph, 0, P - 1)))
        states.append(env.step(ph))
    env.close()
    return dict(state=np.stack(states[:-1]).astype(np.float32),
                action=np.stack(actions).astype(np.float32),
                controller="random_joint", seed=seed)


def rollout_eps_phase(cfg, T, seed, begin, warmup, eps):
    env = SumoMultiEnv(cfg, begin=begin, seed=seed, warmup=warmup)
    env.reset()
    rng = np.random.default_rng(seed + 7)
    mp = controller_max_pressure(env)
    P, N = env.P_max, len(env.tl_ids)
    states, actions = [env.state()], []
    for t in range(T):
        ph = np.asarray(mp(t)).copy()
        flip = rng.random(N) < eps
        ph[flip] = rng.integers(0, P, size=int(flip.sum()))
        actions.append(env.encode_action(np.clip(ph, 0, P - 1)))
        states.append(env.step(ph))
    env.close()
    return dict(state=np.stack(states[:-1]).astype(np.float32),
                action=np.stack(actions).astype(np.float32),
                controller=f"eps_phase_{eps:.2f}", seed=seed)


def rollout_cem_elite(cfg, T, seed, begin, warmup, model_run, base_dir,
                      cem_S=24, cem_iters=2, H=4):
    """closed-loop latent-CEM; execute the chosen joint phase in real SUMO each step."""
    from eval_multi_sumo import load
    from visualize_rollout_cologne8 import fit_pressure_probe, decode
    from plan_cem_multi import encode_context, cem_rollout, HS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = torch.load(f"{base_dir}/val.pt", weights_only=False)
    n, P, F = meta["n_nodes"], meta["P_max"], meta["node_feature_dim"]
    ni = torch.from_numpy(np.asarray(meta["neighbor_idx"]))
    nm = torch.from_numpy(np.asarray(meta["neighbor_mask"]))
    model, _ = load(model_run, "weights_epoch_80.pt", device)
    pr = fit_pressure_probe(model, f"{base_dir}/train.pt", n, P, F, ni, nm, device)

    env = SumoMultiEnv(cfg, begin=begin, seed=seed, warmup=warmup)
    env.reset()
    mp = controller_max_pressure(env)
    rng = np.random.default_rng(seed + 7)
    states, actions = [env.state()], []
    for t in range(warmup):
        ph = mp(t)
        actions.append(env.encode_action(np.clip(ph, 0, P - 1)))
        states.append(env.step(ph))

    for _ in range(T):
        sh = np.stack(states[-HS:]).astype(np.float32)
        ah = np.stack(actions[-HS:]).astype(np.float32)
        z_ctx, a_ctx = encode_context(model, sh, ah, n, ni, nm, device)
        probs = np.full((n, H, P), 1.0 / P)
        for _ in range(cem_iters):
            phases = np.empty((cem_S, n, H), np.int64)
            for i in range(n):
                for h in range(H):
                    phases[:, i, h] = rng.choice(P, size=cem_S, p=probs[i, h])
            pe = cem_rollout(model, z_ctx, a_ctx, phases, n, P, ni, nm, device)
            mc = decode(pr, pe).clamp(min=0).sum(dim=(1, 2, 3)).cpu().numpy()
            elite = phases[np.argsort(mc)[:8]]
            for i in range(n):
                for h in range(H):
                    c = np.bincount(elite[:, i, h], minlength=P)
                    probs[i, h] = (c + 1e-3) / (c.sum() + P * 1e-3)
        ph = probs[:, 0].argmax(1)
        actions.append(env.encode_action(np.clip(ph, 0, P - 1)))
        states.append(env.step(ph))
    env.close()
    return dict(state=np.stack(states[warmup:-1]).astype(np.float32),
                action=np.stack(actions[warmup:]).astype(np.float32),
                controller="cem_elite", seed=seed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sumocfg", default=None)
    p.add_argument("--out_dir", default="traffic_data_cologne8_aug")
    p.add_argument("--base_dir", default="traffic_data_cologne8")
    p.add_argument("--n_random_joint", type=int, default=6)
    p.add_argument("--n_eps", type=int, default=6)
    p.add_argument("--n_cem", type=int, default=3)
    p.add_argument("--cem_model", default="L05ar")
    p.add_argument("--T", type=int, default=300)
    p.add_argument("--cem_T", type=int, default=150)
    p.add_argument("--begin", type=int, default=25200)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--seed", type=int, default=90000)
    p.add_argument("--val_frac", type=float, default=0.2)
    args = p.parse_args()

    cfg = _sumocfg(args.sumocfg)
    print(f"sumocfg {cfg}")
    base_tr = torch.load(f"{args.base_dir}/train.pt", weights_only=False)
    base_va = torch.load(f"{args.base_dir}/val.pt", weights_only=False)
    meta = {k: v for k, v in base_tr.items() if k != "episodes"}

    new_eps = []
    s = args.seed
    for i in range(args.n_random_joint):
        print(f"random_joint {i}"); new_eps.append(rollout_random_joint(cfg, args.T, s, args.begin, args.warmup)); s += 1
    for i in range(args.n_eps):
        eps = 0.3 if i % 2 == 0 else 0.5
        print(f"eps_phase {i} (eps={eps})"); new_eps.append(rollout_eps_phase(cfg, args.T, s, args.begin, args.warmup, eps)); s += 1
    for i in range(args.n_cem):
        print(f"cem_elite {i}")
        new_eps.append(rollout_cem_elite(cfg, args.cem_T, s, args.begin, args.warmup, args.cem_model, args.base_dir)); s += 1

    rng = np.random.default_rng(args.seed)
    rng.shuffle(new_eps)
    n_val = max(1, int(len(new_eps) * args.val_frac))
    aug_va, aug_tr = new_eps[:n_val], new_eps[n_val:]

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    torch.save({"episodes": base_tr["episodes"] + aug_tr, **meta}, out / "train.pt")
    torch.save({"episodes": base_va["episodes"] + aug_va, **meta}, out / "val.pt")
    # keep the counterfactual set from the base dir for eval compatibility
    cf = Path(args.base_dir) / "counterfactual.pt"
    if cf.exists():
        torch.save(torch.load(cf, weights_only=False), out / "counterfactual.pt")
    print(f"\nwrote {out}/train.pt  ({len(base_tr['episodes'])} base + {len(aug_tr)} aug = "
          f"{len(base_tr['episodes']) + len(aug_tr)} episodes)")
    print(f"wrote {out}/val.pt    ({len(base_va['episodes'])} base + {len(aug_va)} aug)")
    import collections
    print("aug controllers:", collections.Counter(e["controller"] for e in new_eps))


if __name__ == "__main__":
    main()
