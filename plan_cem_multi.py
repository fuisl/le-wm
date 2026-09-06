"""Closed-loop factored latent-MPC controller for multi-signal SUMO (cologne8).

The multi-intersection sibling of plan_cem.py. Per D4 / D6:

- **Factored CEM**: per-signal, per-horizon-step categorical distributions
  probs[i, h, phase] (N x H x P_max). Each iteration samples S *joint* candidate
  sequences (S, N, H), scores each with the ONE shared world model (which sees
  neighbours through the pool - coupling is in the scorer, not the search), keeps
  the elite by total cost, refits each probs[i, h] from the elite. Never
  enumerates the 4^8-per-step joint space.
- **Receding horizon**: plan H steps, execute `act_steps` (1 by default),
  re-observe, re-plan - the MPC split.
- **Cost** = summed probe-decoded halting over the imagined H-step rollout, over
  all N signals (the MaxPressure-shaped quantity; no goal embedding).
- **Win condition** (fixed in advance): beat *direct MaxPressure* running in the
  simulator, not just fixed-time. Matching MaxPressure = "a learned, expensive
  approximation of MaxPressure". Beating it needs multi-step anticipation.

Shared MaxPressure warm-up (same seed => same demand => same start state), then
the tail cumulative halting over the compared window is the metric.

Usage:
    python plan_cem_multi.py --run L05ar --seeds 777 101 202 --act_steps 1
"""

import argparse
import os
import time

import numpy as np
import torch

from eval_multi_sumo import load
from visualize_rollout_cologne8 import decode, fit_pressure_probe
from traffic.multi_agent import masked_neighbor_mean, masked_neighbor_pna
from traffic.sumo_multi_env import (
    SumoMultiEnv,
    controller_fixed_time,
    controller_max_pressure,
)

def _resolve_sumocfg():
    """env SUMOCFG_COLOGNE8 > original HMARL path > RESCO benchmark copy in this repo."""
    import os
    for c in (os.environ.get("SUMOCFG_COLOGNE8"),
              "/home/fuisloy/projects/HMARL-TSC/environments/cologne8/cologne8.sumocfg",
              "/home/fuisloy/projects/cair/resco/resco_benchmark/environments/cologne8/cologne8.sumocfg"):
        if c and os.path.exists(c):
            return c
    return "/home/fuisloy/projects/HMARL-TSC/environments/cologne8/cologne8.sumocfg"


SUMOCFG = _resolve_sumocfg()
DATA_DIR = "traffic_data_cologne8"
HS = int(os.environ.get("WM_HISTORY", 3))   # context window; set WM_HISTORY to match the checkpoint cfg["history"]


def encode_context(model, state_hist, action_hist, n_nodes, ni, nm, device):
    """state_hist/action_hist: (HS, N*F)/(HS, N*A) -> z (N,HS,d), act_emb (N,HS,a)."""
    info = {"state": torch.from_numpy(state_hist).unsqueeze(0).float().to(device),
            "action": torch.from_numpy(action_hist).unsqueeze(0).float().to(device),
            "n_nodes": n_nodes}
    if model.level == "0.5":
        info["neighbor_idx"] = ni.to(device); info["neighbor_mask"] = nm.to(device)
    with torch.no_grad():
        out = model.encode(info)
    z = out["emb"].reshape(1, n_nodes, HS, -1).transpose(1, 2)[0].transpose(0, 1)   # (N,HS,d)
    a = out["act_emb"].reshape(1, n_nodes, HS, -1).transpose(1, 2)[0].transpose(0, 1)
    return z, a


def cem_rollout(model, z_ctx, a_ctx, action_seqs, n_nodes, P_max, ni, nm, device):
    """Batched factored rollout. z_ctx/a_ctx: (N,HS,·). action_seqs: (S,N,H) int.
    Returns predicted own embeddings (S, N, H, d)."""
    S, N, H = action_seqs.shape
    ni, nm = ni.to(device), nm.to(device)
    z = z_ctx.unsqueeze(0).expand(S, -1, -1, -1).contiguous().to(device)   # (S,N,HS,d)
    a = a_ctx.unsqueeze(0).expand(S, -1, -1, -1).contiguous().to(device)
    lvl05 = model.level == "0.5"
    with torch.no_grad():
        preds = []
        for h in range(H):
            zc, ac = z[:, :, -HS:], a[:, :, -HS:]
            if lvl05:
                if model.neighbor_agg == "pna":
                    zp = model.nbr_proj(masked_neighbor_pna(zc, ni, nm))
                else:
                    zp = masked_neighbor_mean(zc, ni, nm)
                ap = masked_neighbor_mean(ac, ni, nm)
                if model.permute_control:
                    perm = model._fixed_derangement(n_nodes, device)
                    zp, ap = zp[:, perm], ap[:, perm]
                pin_e = torch.cat([zc, zp], -1).reshape(S * N, HS, -1)
                pin_a = torch.cat([ac, ap], -1).reshape(S * N, HS, -1)
            else:
                pin_e = zc.reshape(S * N, HS, -1)
                pin_a = ac.reshape(S * N, HS, -1)
            nxt = model.predict(pin_e, pin_a)[:, -1].reshape(S, N, 1, -1)
            preds.append(nxt)
            z = torch.cat([z, nxt], dim=2)
            oh = torch.zeros(S, N, P_max, device=device)
            idx = torch.from_numpy(action_seqs[:, :, h]).long().to(device)
            oh.scatter_(2, idx.unsqueeze(-1), 1.0)
            ae = model.action_encoder(oh.reshape(S * N, 1, P_max)).reshape(S, N, 1, -1)
            a = torch.cat([a, ae], dim=2)
    return torch.cat(preds, dim=2)   # (S, N, H, d)


class FactoredCEM:
    """Factored categorical CEM over joint phase sequences.

    Proposal-distribution options (2026-09-06, planning-fix study; all off by
    default => the original uniform, memoryless search):
      hold_prior  p : initial categorical puts mass p on each signal's CURRENT
                      phase at every horizon step (V-JEPA-2-AC / PlaNet style
                      "start from do-nothing"); the all-hold sequence is also
                      injected as candidate 0 of iteration 0.
      warm_start    : initial categorical = previous decision's fitted
                      categorical shifted one step (TD-MPC2 style), averaged
                      with the hold prior if both are on.
      min_hold    k : each signal's phase is sampled in blocks of k steps
                      (block-constant), AND a signal that has been green for
                      fewer than k steps is forced to hold for the remainder
                      -- a real minimum-green constraint at execution.
      per_agent     : AMM-style per-intersection search: each signal optimises
                      its own sequence with every other signal held at its
                      current phase; the shared model still scores jointly.
    """

    def __init__(self, n_nodes, P_max, horizon, num_samples=64, topk=8, n_iters=4, rng=None,
                 hold_prior=0.0, warm_start=False, min_hold=1, per_agent=False,
                 n_valid=None, legal_only=False):
        self.N, self.P, self.H = n_nodes, P_max, horizon
        self.S, self.topk, self.iters = num_samples, topk, n_iters
        self.rng = rng or np.random.default_rng(0)
        self.hold_prior, self.warm_start = float(hold_prior), bool(warm_start)
        self.min_hold, self.per_agent = max(1, int(min_hold)), bool(per_agent)
        # legal_only: signals with fewer than P_max green phases only ever get
        # codes < n_valid[i]. Without it (the ORIGINAL behaviour) the search
        # feeds the model action one-hots that never occur in training data
        # (cologne8: 5 of 8 signals have 2 or 3 phases) -- SUMO silently clips
        # them to the last legal phase, the model does not.
        self.n_valid = None if n_valid is None else np.asarray(n_valid, dtype=int)
        self.legal_only = bool(legal_only) and self.n_valid is not None
        self.prev_probs = None

    def _mask(self, probs):
        if self.legal_only:
            for i in range(self.N):
                probs[i, :, self.n_valid[i]:] = 0.0
            probs = probs / probs.sum(-1, keepdims=True)
        return probs

    # -- proposal distribution --------------------------------------------
    def _init_probs(self, cur_phase):
        N, H, P = self.N, self.H, self.P
        probs = np.full((N, H, P), 1.0 / P)
        used_prev = False
        if self.warm_start and self.prev_probs is not None:
            probs[:, :-1] = self.prev_probs[:, 1:]
            probs[:, -1] = 1.0 / P
            used_prev = True
        if self.hold_prior > 0 and cur_phase is not None:
            hp = np.full((N, H, P), (1.0 - self.hold_prior) / max(P - 1, 1))
            hp[np.arange(N), :, cur_phase] = self.hold_prior
            probs = 0.5 * (probs + hp) if used_prev else hp
        return self._mask(probs)

    def _sample(self, probs, cur_phase, elapsed):
        S, N, H, P = self.S, self.N, self.H, self.P
        phases = np.empty((S, N, H), dtype=np.int64)
        for i in range(N):
            h = 0
            while h < H:
                blk = min(self.min_hold, H - h)
                phases[:, i, h:h + blk] = self.rng.choice(P, size=S, p=probs[i, h])[:, None]
                h += blk
        if self.min_hold > 1 and elapsed is not None and cur_phase is not None:
            for i in range(N):
                rem = int(self.min_hold - elapsed[i])
                if rem > 0:
                    phases[:, i, :min(rem, H)] = cur_phase[i]
        return phases

    def _refit(self, probs, elite, nodes=None):
        for i in (range(self.N) if nodes is None else nodes):
            for h in range(self.H):
                c = np.bincount(elite[:, i, h], minlength=self.P)
                probs[i, h] = (c + 1e-3) / (c.sum() + self.P * 1e-3)
        return self._mask(probs)

    # -- search -------------------------------------------------------------
    def plan(self, cost_fn, cur_phase=None, elapsed=None):
        return self.plan_multi(cost_fn, 1, cur_phase, elapsed)[0]

    def plan_multi(self, cost_fn, act_steps, cur_phase=None, elapsed=None):
        """Return argmax phases for the first `act_steps` horizon steps as an
        (act_steps, N) array. act_steps=1 => plan-H / act-1 MPC."""
        if self.per_agent:
            return self._plan_per_agent(cost_fn, act_steps, cur_phase, elapsed)
        probs = self._init_probs(cur_phase)
        for it in range(self.iters):
            phases = self._sample(probs, cur_phase, elapsed)
            if it == 0 and self.hold_prior > 0 and cur_phase is not None:
                phases[0] = np.asarray(cur_phase)[:, None]          # seeded all-hold candidate
            cost = cost_fn(phases)
            elite = phases[np.argsort(cost)[: self.topk]]
            probs = self._refit(probs, elite)
        self.prev_probs = probs
        k = max(1, min(act_steps, self.H))
        out = probs[:, :k].argmax(axis=2).T   # (k, N)
        if self.min_hold > 1 and elapsed is not None and cur_phase is not None:
            for i in range(self.N):
                rem = int(self.min_hold - elapsed[i])
                if rem > 0:
                    out[:min(rem, k), i] = cur_phase[i]
        return out

    def _plan_per_agent(self, cost_fn, act_steps, cur_phase, elapsed):
        assert cur_phase is not None, "per_agent search needs the current joint phase"
        base = np.asarray(cur_phase)
        probs = self._init_probs(cur_phase)
        final = np.tile(base[:, None], (1, self.H))                # (N,H)
        for i in range(self.N):
            for it in range(self.iters):
                smp = self._sample(probs, cur_phase, elapsed)      # (S,N,H)
                phases = np.tile(base[None, :, None], (self.S, 1, self.H))
                phases[:, i, :] = smp[:, i, :]
                if it == 0 and self.hold_prior > 0:
                    phases[0, i, :] = base[i]
                cost = cost_fn(phases)
                elite = phases[np.argsort(cost)[: self.topk]]
                probs = self._refit(probs, elite, nodes=[i])
            final[i] = probs[i].argmax(axis=1)
        self.prev_probs = probs
        k = max(1, min(act_steps, self.H))
        return final[:, :k].T


class OracleFactoredCEM(FactoredCEM):
    """Same factored search, each joint candidate scored by rolling the real
    SUMO env forward (save_state/load_state) - isolates model/probe error from
    the CEM logic + cost design, like plan_cem.py's OracleCEMPlanner."""

    def plan_env(self, env, snap_path, act_steps, cur_phase=None, elapsed=None):
        snap = env.save_state(snap_path)
        F, P = env.node_feature_dim(), env.P_max

        def cost_fn(phases):                     # phases: (S,N,H)
            out = np.zeros(len(phases))
            for s in range(len(phases)):
                env.load_state(snap)
                tot = 0.0
                for h in range(phases.shape[2]):
                    st = env.step(phases[s, :, h])
                    tot += float(st.reshape(env.n_nodes(), F)[:, :P].sum())
                out[s] = tot
            return out

        plan = self.plan_multi(cost_fn, act_steps, cur_phase, elapsed)
        env.load_state(snap)
        return plan


def env_phase_state(env):
    """Current joint phase (N,) and steps-in-green (N,) for min-hold / hold-prior."""
    cur = np.array([env.current_phase[t] for t in env.tl_ids])
    el = np.array([env.elapsed[t] // env.step_length for t in env.tl_ids])
    return cur, el


def run_episode(env, n_compare, warmup, decision_fn):
    env.reset()
    warm = controller_max_pressure(env)
    states = [env.state()]
    actions = []
    for t in range(warmup):
        ph = warm(t)
        actions.append(env.encode_action(np.clip(ph, 0, env.P_max - 1)))
        states.append(env.step(ph))
    tail, wall, trace = 0.0, [], []
    n_switch, n_sig = 0, 0
    n_green = np.array([len(env.green_phases[x]) for x in env.tl_ids])
    for t in range(n_compare):
        t0 = time.time()
        ph = decision_fn(env, t, states, actions)
        wall.append(time.time() - t0)
        ph = np.clip(np.asarray(ph), 0, env.P_max - 1)
        cur = np.array([env.current_phase[x] for x in env.tl_ids])
        eff = np.minimum(ph, n_green - 1)                 # what SUMO actually does with the code
        n_switch += int((eff != cur).sum()); n_sig += len(cur)
        actions.append(env.encode_action(ph))
        s = env.step(ph)
        env.metrics_tick()
        states.append(s)
        step_halt = float(s.reshape(env.n_nodes(), env.node_feature_dim())[:, :env.P_max].sum())
        tail += step_halt
        trace.append(step_halt)
    env.close()                       # flushes --tripinfo-output; parse AFTER this
    m = env.episode_metrics()
    m["switch_rate"] = n_switch / max(n_sig, 1)
    return tail, (float(np.mean(wall)) if wall else 0.0), trace, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="L05ar")
    ap.add_argument("--weights", default="weights_epoch_80.pt")
    ap.add_argument("--seeds", type=int, nargs="+", default=[777, 101, 202])
    ap.add_argument("--n_compare", type=int, default=70)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--act_steps", type=int, default=1, help="execute this many planned steps before re-planning")
    ap.add_argument("--num_samples", type=int, default=64)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--n_iters", type=int, default=4)
    ap.add_argument("--fixed_cycle", type=int, default=6)
    ap.add_argument("--oracle", action="store_true", help="add oracle-CEM (real-SUMO cost) - slow, 1 seed recommended")
    ap.add_argument("--dump_traces", action="store_true")
    ap.add_argument("--wandb", action="store_true", help="log RESCO metrics to wandb (offline if no creds)")
    # planning-fix study (2026-09-06)
    ap.add_argument("--hold_prior", type=float, default=0.0, help="CEM init mass on the current phase")
    ap.add_argument("--warm_start", action="store_true", help="shift previous fitted categorical")
    ap.add_argument("--min_hold", type=int, default=1, help="minimum green in control steps (block sampling + execution constraint)")
    ap.add_argument("--per_agent", action="store_true", help="per-intersection search, others held")
    ap.add_argument("--legal_only", action="store_true", help="mask CEM categoricals to each signal's legal phase codes")
    ap.add_argument("--oracle_samples", type=int, default=24)
    ap.add_argument("--oracle_iters", type=int, default=3)
    ap.add_argument("--oracle_topk", type=int, default=6)
    ap.add_argument("--tag", default=None, help="results/control_<tag>.json")
    ap.add_argument("--data_dir", default=None, help="override DATA_DIR (obs-mode corpora)")
    args = ap.parse_args()
    global DATA_DIR
    if args.data_dir:
        DATA_DIR = args.data_dir

    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = torch.load(f"{DATA_DIR}/val.pt", weights_only=False)
    n_nodes, P_max = meta["n_nodes"], meta["P_max"]
    ni = torch.from_numpy(np.asarray(meta["neighbor_idx"]))
    nm = torch.from_numpy(np.asarray(meta["neighbor_mask"]))
    F = meta["node_feature_dim"]

    model, cfg = load(args.run, args.weights, device)
    pr = fit_pressure_probe(model, f"{DATA_DIR}/train.pt", n_nodes, P_max, F, ni, nm, device)
    print(f"model {args.run} ({cfg.get('neighbor_agg','mean')}, level {cfg['level']}, "
          f"rollout_k {cfg.get('rollout_k','-')})  probe MSE {pr['mse']:.1f}")

    def make_cem_decision(seed):
        cem = FactoredCEM(n_nodes, P_max, args.horizon, args.num_samples, args.topk,
                          args.n_iters, rng=np.random.default_rng(seed),
                          hold_prior=args.hold_prior, warm_start=args.warm_start,
                          min_hold=args.min_hold, per_agent=args.per_agent,
                          n_valid=meta.get("n_green_phases"), legal_only=args.legal_only)
        cache = {"queue": []}

        def decision(env, t, states, actions):
            if cache["queue"]:                       # committing a previously-planned step
                return cache["queue"].pop(0)
            sh = np.stack(states[-HS:]).astype(np.float32)
            ah = np.stack(actions[-HS:]).astype(np.float32)
            z_ctx, a_ctx = encode_context(model, sh, ah, n_nodes, ni, nm, device)
            cur, el = env_phase_state(env)

            def cost_fn(phases):
                pe = cem_rollout(model, z_ctx, a_ctx, phases, n_nodes, P_max, ni, nm, device)
                dec = decode(pr, pe).clamp(min=0)    # (S,N,H,P)
                return dec.sum(dim=(1, 2, 3)).cpu().numpy()

            plan = cem.plan_multi(cost_fn, args.act_steps, cur, el)   # (act_steps, N)
            cache["queue"] = [plan[k] for k in range(1, len(plan))]
            return plan[0]
        return decision

    def mp_decision_factory():
        cache = {}
        def d(env, t, states, actions):
            cache.setdefault("p", controller_max_pressure(env))
            return cache["p"](t)
        return d

    def ft_decision_factory(cyc):
        cache = {}
        def d(env, t, states, actions):
            cache.setdefault("p", controller_fixed_time(env, np.random.default_rng(0)))
            return cache["p"](t)
        return d

    import wandb_utils
    wb = wandb_utils.init(args.wandb, project="cair-traffic-cologne8",
                          name=f"control-{args.run}",
                          config={"model": args.run, "weights": args.weights,
                                  "horizon": args.horizon, "act_steps": args.act_steps,
                                  "num_samples": args.num_samples, "n_iters": args.n_iters,
                                  "n_compare": args.n_compare, "seeds": args.seeds,
                                  "probe_mse": pr["mse"], **{f"cfg/{k}": v for k, v in cfg.items()}},
                          group="closed-loop-control")

    def make_oracle_decision(seed):
        oc = OracleFactoredCEM(n_nodes, P_max, args.horizon, args.oracle_samples, args.oracle_topk,
                               args.oracle_iters, rng=np.random.default_rng(seed),
                               hold_prior=args.hold_prior, warm_start=args.warm_start,
                               min_hold=args.min_hold, per_agent=False,
                               n_valid=meta.get("n_green_phases"), legal_only=args.legal_only)
        cache = {"queue": []}
        def decision(env, t, states, actions):
            if cache["queue"]:
                return cache["queue"].pop(0)
            cur, el = env_phase_state(env)
            plan = oc.plan_env(env, f"_oracle_multi_snap_{os.getpid()}.xml", args.act_steps, cur, el)
            cache["queue"] = [plan[k] for k in range(1, len(plan))]
            return plan[0]
        return decision

    rows = []            # (seed, {name: tail_halting})
    metric_rows = []     # (seed, {name: resco_metrics_dict})
    for seed in args.seeds:
        res, mres, traces = {}, {}, {}
        pairs = [
            ("latent-CEM", make_cem_decision(seed)),
            ("max_pressure", mp_decision_factory()),
            ("fixed_time", ft_decision_factory(args.fixed_cycle)),
        ]
        if args.oracle:
            pairs.append(("oracle-CEM", make_oracle_decision(seed)))
        for name, dfn in pairs:
            env2 = SumoMultiEnv(SUMOCFG, seed=seed, warmup=0, metrics=True)
            tail, wall, tr, m = run_episode(env2, args.n_compare, args.warmup, dfn)  # closes env2 itself
            res[name], mres[name], traces[name] = tail, m, tr
            print(f"  seed {seed:>4}  {name:>14}:  tail {tail:6.0f}  dur {m['duration']:7.1f}  delay {m['delay']:7.1f}  "
                  f"wait {m['wait']:7.1f}  queue {m['queue']:6.1f}  thru {m['throughput']:>4}  sw {m['switch_rate']:.2f}"
                  + (f"   ({wall*1000:.0f} ms/dec)" if wall > 1e-4 else ""), flush=True)
            wb.log({f"{name}/duration": m["duration"], f"{name}/delay": m["delay"],
                    f"{name}/wait": m["wait"], f"{name}/queue": m["queue"],
                    f"{name}/throughput": m["throughput"], f"{name}/tail_halting": tail,
                    "seed": seed})
        rows.append((seed, res)); metric_rows.append((seed, mres))
        if args.dump_traces and seed == args.seeds[0]:
            import json
            json.dump({"seed": seed, "warmup": args.warmup, "n_compare": args.n_compare,
                       "step_s": 5, "model": args.run, "tails": res, "traces": traces,
                       "metrics": mres},
                      open(f"{DATA_DIR}/control_traces.json", "w"))
            print(f"  wrote {DATA_DIR}/control_traces.json")

    if args.tag:
        import json
        os.makedirs("results", exist_ok=True)
        json.dump({"args": vars(args), "model": args.run, "tails": [(s, r) for s, r in rows],
                   "metrics": [(s, m) for s, m in metric_rows],
                   "mean_ratio_cem_mp": float(np.mean([r["latent-CEM"] / r["max_pressure"] for _, r in rows])),
                   "mean_ratio_oracle_mp": (float(np.mean([r["oracle-CEM"] / r["max_pressure"] for _, r in rows]))
                                            if "oracle-CEM" in rows[0][1] else None)},
                  open(f"results/control_{args.tag}.json", "w"), indent=1)
        print(f"  wrote results/control_{args.tag}.json")

    print("\n=== summary (tail cumulative halting, lower = better) ===")
    has_oracle = "oracle-CEM" in rows[0][1]
    hdr = f"{'seed':>6} {'latent-CEM':>12} {'max_pressure':>13} {'fixed_time':>11}"
    if has_oracle:
        hdr += f" {'oracle-CEM':>11}"
    print(hdr + f"  {'CEM/MP':>7}")
    for seed, r in rows:
        line = f"{seed:>6} {r['latent-CEM']:>12.0f} {r['max_pressure']:>13.0f} {r['fixed_time']:>11.0f}"
        if has_oracle:
            line += f" {r['oracle-CEM']:>11.0f}"
        print(line + f"  {r['latent-CEM']/r['max_pressure']:>7.2f}")
    mean_ratio = np.mean([r['latent-CEM'] / r['max_pressure'] for _, r in rows])
    print(f"\nmean latent-CEM / max_pressure = {mean_ratio:.2f}  "
          f"({'BEATS' if mean_ratio < 1 else 'loses to'} direct MaxPressure)")
    if has_oracle:
        omr = np.mean([r['oracle-CEM'] / r['max_pressure'] for _, r in rows])
        print(f"mean oracle-CEM / max_pressure = {omr:.2f}  "
              f"-> gap is {'MODEL/PROBE error' if omr < mean_ratio * 0.9 else 'the CEM search / cost design'}")

    # RESCO-style metric table (mean over seeds), the numbers a RESCO eval reports
    print("\n=== RESCO metrics (mean over seeds; lower = better except throughput) ===")
    names = [n for n, _ in metric_rows[0][1].items()]
    print(f"{'controller':>14} {'duration':>10} {'delay':>9} {'wait':>9} {'queue':>8} {'throughput':>11}")
    summary = {}
    for name in names:
        agg = {k: float(np.nanmean([mr[name][k] for _, mr in metric_rows]))
               for k in ("duration", "delay", "wait", "queue", "throughput")}
        summary[name] = agg
        print(f"{name:>14} {agg['duration']:>10.1f} {agg['delay']:>9.1f} {agg['wait']:>9.1f} "
              f"{agg['queue']:>8.1f} {agg['throughput']:>11.0f}")
        for k, v in agg.items():
            wb.log({f"mean/{name}/{k}": v})
    wb.finish()


if __name__ == "__main__":
    main()
