"""Minimal closed-loop latent-MPC controller for a single SUMO intersection
(cologne1), plus the baseline sweep needed to judge it. This is the natural
next step after eval_ranking.py's plan-ranking-accuracy gate passed clearly
better than chance.

Design per D6 Planner / Final Architecture v1's frozen v1 spec:

- **Factored CEM** (trivial here - one agent, nothing to factor with a single
  intersection; this is D4 variant A's degenerate case, before cologne3 makes
  factorisation across neighbours meaningful).
- **Plan cost = running queue cost**, decoded through the frozen linear probe
  fit_linear_probe() already produces - the operational form of "MaxPressure-
  based cost" this codebase uses (sumo_env.py's own controller_max_pressure
  scores a phase by summed halting count on the lanes it serves; this sums
  that same quantity over the imagined rollout). Symlog normalisation is
  skipped for this single-intersection, single-regime v0 - it matters once
  costs must be compared across free-flow and gridlock regimes, not yet here.
- **No goal embedding.** Per Quasimetric Cost-to-Go, scoring plans by
  ‖ẑ_{t+H} - z_g‖ is a type error for an irreversible system like traffic
  (queues aren't reversible, so no single latent target is "the goal"). This
  planner instead scores an integrated running cost over the rollout, which
  needs no goal state at all.

**The win condition, fixed in advance** (Implementation Guide discipline):
because the plan cost *is* a MaxPressure-shaped quantity, direct MaxPressure
running in the simulator - not persistence, not fixed-time - is the bar to
beat. If latent-MPC only matches it, that reads as "an expensive
approximation of MaxPressure, learned instead of computed." The result that
would justify the model is multi-step anticipation MaxPressure cannot do -
holding a green because a platoon is a few seconds out - which only shows up
over a horizon, not a single greedy step.

All three controllers compared here share an identical MaxPressure warm-up
(same seed => same demand => same traffic state at the start of the compared
window), then diverge only at decision time, so the tail comparison isolates
controller quality rather than warm-up luck.

Usage:
    python plan_cem.py --sumocfg /path/to/RESCO/environments/cologne1/cologne1.sumocfg \
        --run_name traffic-lewm-sumo --weights weights_epoch_60.pt
"""

import argparse
import time

import numpy as np
import torch

from eval_traffic import apply_probe, fit_linear_probe, load_model
from traffic.sumo_env import SumoTLEnv, controller_fixed_time, controller_max_pressure


def rollout_batch(model, state_hist, action_hist, action_seqs, history_size, device):
    """Batched autoregressive latent rollout, the CEM-friendly sibling of
    eval_traffic.py's latent_rollout (same recurrence, vectorised over S
    candidate action sequences instead of one).

    state_hist / action_hist: (H, D) / (H, A) shared context.
    action_seqs: (S, n_steps, A) one-hot candidate action sequences.
    returns predicted embeddings for all candidates: (S, n_steps, d)
    """
    S, n_steps, _ = action_seqs.shape
    with torch.no_grad():
        ctx = {
            "state": state_hist.unsqueeze(0).expand(S, -1, -1).to(device),
            "action": action_hist.unsqueeze(0).expand(S, -1, -1).to(device),
        }
        ctx = model.encode(ctx)
        emb = ctx["emb"]  # (S, H, d)
        act = ctx["action"]  # (S, H, A), kept raw and re-embedded each step

        preds = []
        for t in range(n_steps):
            act_emb = model.action_encoder(act)
            pred = model.predict(emb[:, -history_size:], act_emb[:, -history_size:])[:, -1:]
            preds.append(pred)
            emb = torch.cat([emb, pred], dim=1)
            next_a = action_seqs[:, t : t + 1, :].to(device)
            act = torch.cat([act, next_a], dim=1)
        return torch.cat(preds, dim=1)  # (S, n_steps, d)


def running_cost(W, pred_emb):
    """Sum of decoded halting-vehicle count over the rollout, per candidate -
    lower is less congestion. pred_emb: (S, n_steps, d) -> (S,)"""
    pred_state = apply_probe(W, pred_emb)  # (S, n_steps, state_dim)
    return pred_state.sum(dim=(1, 2))


class CEM:
    """Generic categorical CEM over a fixed per-step action alphabet - D6's
    default planner, single-agent case of the factored search (nothing to
    factor with one intersection). Takes a cost_fn(phases: (S, horizon) int
    array) -> (S,) cost array, so the exact same search loop drives both the
    learned-model planner and the oracle-simulator ablation below; only the
    cost source differs, which is the point of the ablation."""

    def __init__(self, action_dim, horizon, num_samples=64, topk=8, n_iters=4):
        self.action_dim, self.horizon = action_dim, horizon
        self.num_samples, self.topk, self.n_iters = num_samples, topk, n_iters

    def plan(self, cost_fn):
        probs = np.full((self.horizon, self.action_dim), 1.0 / self.action_dim)

        for _ in range(self.n_iters):
            phases = np.array([
                [np.random.choice(self.action_dim, p=probs[h]) for h in range(self.horizon)]
                for _ in range(self.num_samples)
            ])  # (S, horizon)

            cost = cost_fn(phases)

            elite_idx = np.argsort(cost)[: self.topk]
            elite_phases = phases[elite_idx]  # (topk, horizon)

            for h in range(self.horizon):
                counts = np.bincount(elite_phases[:, h], minlength=self.action_dim)
                probs[h] = (counts + 1e-3) / (counts.sum() + self.action_dim * 1e-3)

        return int(np.argmax(probs[0]))


class LatentCEMPlanner:
    """CEM scored by the learned world model's imagined rollout, decoded
    through the frozen linear probe - the actual planner under test."""

    def __init__(self, model, W, action_dim, history_size, horizon=5,
                 num_samples=64, topk=8, n_iters=4, device="cuda"):
        self.model, self.W = model, W
        self.history_size, self.device = history_size, device
        self.cem = CEM(action_dim, horizon, num_samples, topk, n_iters)

    def plan(self, state_hist, action_hist):
        def cost_fn(phases):
            action_seqs = torch.zeros(phases.shape[0], self.cem.horizon, self.cem.action_dim)
            for s in range(phases.shape[0]):
                action_seqs[s, np.arange(self.cem.horizon), phases[s]] = 1.0
            pred_emb = rollout_batch(self.model, state_hist, action_hist, action_seqs,
                                      self.history_size, self.device)
            return running_cost(self.W, pred_emb).cpu().numpy()

        return self.cem.plan(cost_fn)


class OracleCEMPlanner:
    """Same CEM search as LatentCEMPlanner, but each candidate branch is
    scored by rolling the *real* SUMO simulator forward (save_state/
    load_state) instead of the learned model's imagined rollout. Isolates
    whether latent-CEM's shortfall vs. direct MaxPressure (2026-08-19,
    Experiment Log) comes from model/probe imagination error or from the CEM
    planning logic itself - the same attribution split M3.5 used to keep
    "more data" and "different architecture" from being conflated, applied
    here to "better model" vs. "better search". This is also the
    "simulator-in-the-loop oracle" model source Benchmarks and Baselines
    already names for the killer experiment's three-way comparison."""

    def __init__(self, action_dim, horizon=5, num_samples=24, topk=6, n_iters=3,
                 snapshot_path="_oracle_cem_snapshot.xml"):
        self.cem = CEM(action_dim, horizon, num_samples, topk, n_iters)
        self.snapshot_path = snapshot_path

    def plan(self, env):
        snapshot = env.save_state(self.snapshot_path)

        def cost_fn(phases):
            costs = np.zeros(phases.shape[0])
            for s in range(phases.shape[0]):
                env.load_state(snapshot)
                total = 0.0
                for h in range(phases.shape[1]):
                    total += float(env.step(int(phases[s, h])).sum())
                costs[s] = total
            return costs

        best = self.cem.plan(cost_fn)
        env.load_state(snapshot)  # restore the true decision-point state
        return best


def run_episode(sumocfg, seed, n_steps, warmup_steps, decision_fn):
    """decision_fn(env, t, states, actions) -> phase, called only for
    t >= warmup_steps. Steps [0, warmup_steps) always use MaxPressure,
    identical across every controller being compared, so the compared window
    starts from the same traffic state (same seed => same demand)."""
    env = SumoTLEnv(sumocfg, seed=seed)
    env.reset()
    warmup_policy = controller_max_pressure(env)

    states, actions = [env.state()], []
    for t in range(warmup_steps):
        phase = warmup_policy(t)
        actions.append(env.encode_action(phase))
        states.append(env.step(phase))

    tail_halting, wall_clock = 0.0, []
    for t in range(warmup_steps, n_steps):
        t0 = time.time()
        phase = decision_fn(env, t, states, actions)
        wall_clock.append(time.time() - t0)

        actions.append(env.encode_action(phase))
        s = env.step(phase)
        states.append(s)
        tail_halting += float(s.sum())

    env.close()
    return tail_halting, (float(np.mean(wall_clock)) if wall_clock else 0.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sumocfg", required=True)
    p.add_argument("--run_name", default="traffic-lewm-sumo")
    p.add_argument("--weights", default="weights_epoch_60.pt")
    p.add_argument("--data_dir", default="traffic_data_sumo")
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--n_steps", type=int, default=120, help="total control decisions, incl. warm-up")
    p.add_argument("--warmup_steps", type=int, default=10)
    p.add_argument("--history_size", type=int, default=3)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--num_samples", type=int, default=64)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--n_iters", type=int, default=4)
    p.add_argument("--fixed_time_cycle", type=int, default=6)
    p.add_argument("--skip_oracle", action="store_true",
                    help="oracle-CEM re-simulates SUMO per candidate branch - slow; "
                         "skip it for quick iteration on the other three controllers")
    p.add_argument("--oracle_num_samples", type=int, default=24)
    p.add_argument("--oracle_topk", type=int, default=6)
    p.add_argument("--oracle_n_iters", type=int, default=3)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.run_name, args.weights, device)
    val_meta = torch.load(f"{args.data_dir}/val.pt", weights_only=False)
    action_dim = val_meta["action_dim"]  # W.shape[1] is state_dim, not action_dim - don't reuse it
    W = fit_linear_probe(model, f"{args.data_dir}/val.pt", args.history_size + 1, device)

    planner = LatentCEMPlanner(model, W, action_dim=action_dim, history_size=args.history_size,
                                horizon=args.horizon, num_samples=args.num_samples,
                                topk=args.topk, n_iters=args.n_iters, device=device)

    def cem_decision(env, t, states, actions):
        state_hist = torch.from_numpy(np.stack(states[-args.history_size:])).float()
        action_hist = torch.from_numpy(np.stack(actions[-args.history_size:])).float()
        return planner.plan(state_hist, action_hist)

    oracle_planner = OracleCEMPlanner(action_dim=action_dim, horizon=args.horizon,
                                       num_samples=args.oracle_num_samples,
                                       topk=args.oracle_topk, n_iters=args.oracle_n_iters)

    def oracle_decision(env, t, states, actions):
        return oracle_planner.plan(env)

    def maxpressure_decision_factory():
        # built once per episode so its internal state (none needed here,
        # but matches sumo_env.py's own convention) stays consistent
        cache = {}
        def decision(env, t, states, actions):
            if "policy" not in cache:
                cache["policy"] = controller_max_pressure(env)
            return cache["policy"](t)
        return decision

    def fixed_time_decision_factory(cycle):
        cache = {}
        def decision(env, t, states, actions):
            if "policy" not in cache:
                cache["policy"] = controller_fixed_time(env, cycle=cycle)
            return cache["policy"](t)
        return decision

    controllers = {
        "latent-CEM": cem_decision,
        "max_pressure (direct)": maxpressure_decision_factory(),
        "fixed_time": fixed_time_decision_factory(args.fixed_time_cycle),
    }
    if not args.skip_oracle:
        controllers["oracle-CEM"] = oracle_decision

    print(f"\n=== closed-loop comparison: cologne1, seed={args.seed}, "
          f"{args.warmup_steps} warm-up + {args.n_steps - args.warmup_steps} compared steps ===")
    results = {}
    for name, decision_fn in controllers.items():
        halting, wall_clock = run_episode(args.sumocfg, args.seed, args.n_steps,
                                           args.warmup_steps, decision_fn)
        results[name] = halting
        print(f"{name:>24}: tail cumulative halting-count = {halting:>10.1f}"
              + (f"   ({wall_clock * 1000:.1f} ms/decision)" if wall_clock else ""))

    best = min(results, key=results.get)
    print(f"\nbest: {best}")
    print("win condition (per plan_cem.py's docstring): latent-CEM should beat "
          "'max_pressure (direct)', not just fixed_time/persistence.")
    if "oracle-CEM" in results:
        print("oracle-CEM (same CEM search, real-simulator cost instead of the learned "
              "model) separates model/probe error from CEM search-logic quality: if "
              "oracle-CEM beats max_pressure (direct) but latent-CEM doesn't, the gap is "
              "model/probe error, not the planner; if oracle-CEM also loses, CEM's search "
              "or the raw-halting-count cost itself needs work before blaming the model.")


if __name__ == "__main__":
    main()
