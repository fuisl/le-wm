"""Step 0.5: is the world model over-optimistic about high-switching plans because
it never SAW them (coverage), or because it structurally UNDER-WEIGHTS the real
per-switch penalty (each phase change = 3 s yellow + 2 s all-red = 5 s lost)?

Walk a MaxPressure trajectory; at ~15 anchors compare, from the SAME frozen state:
  hold  = keep every signal's current phase for H steps
  swk   = flip k signals to a random different phase (then hold), k in {1,2,4,8}
against BOTH the world model's predicted H-step halting and real SUMO.

  d_true(k)  = oracle_cost(swk) - oracle_cost(hold)
  d_model(k) = model_cost(swk)  - model_cost(hold)

Read:
  slope(d_model) ~ slope(d_true)  but d_model noisier/biased  -> coverage problem
  slope(d_model) << slope(d_true)                              -> model is blind to
                                                                  the switch penalty
                                                                  (structural fix:
                                                                   make transition
                                                                   cost explicit)

Usage:  python diag_switch_penalty.py --run L05ar --n_anchors 15 --repeats 5
"""
import argparse
import os
import numpy as np
import torch

from eval_multi_sumo import load
from visualize_rollout_cologne8 import fit_pressure_probe, decode
from plan_cem_multi import encode_context, cem_rollout, SUMOCFG, HS
from traffic.sumo_multi_env import SumoMultiEnv, controller_max_pressure

DATA_DIR = "traffic_data_cologne8"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="L05ar")
    ap.add_argument("--weights", default="weights_epoch_80.pt")
    ap.add_argument("--data_dir", default="traffic_data_cologne8")
    ap.add_argument("--sumocfg", default=SUMOCFG)
    ap.add_argument("--begin", type=int, default=25200)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--n_anchors", type=int, default=15)
    ap.add_argument("--anchor_every", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = torch.load(f"{args.data_dir}/val.pt", weights_only=False)
    n_nodes, P_max, F = meta["n_nodes"], meta["P_max"], meta["node_feature_dim"]
    ni = torch.from_numpy(np.asarray(meta["neighbor_idx"]))
    nm = torch.from_numpy(np.asarray(meta["neighbor_mask"]))

    model, cfg = load(args.run, args.weights, device)
    pr = fit_pressure_probe(model, f"{args.data_dir}/train.pt", n_nodes, P_max, F, ni, nm, device)
    print(f"model {args.run} (level {cfg['level']}, agg {cfg.get('neighbor_agg','mean')})  probe MSE {pr['mse']:.1f}")

    ks = [0, 1, 2, 4, min(8, n_nodes)]
    rng = np.random.default_rng(args.seed)
    env = SumoMultiEnv(args.sumocfg, seed=args.seed, warmup=0, metrics=False, begin=args.begin)
    env.reset()
    warm = controller_max_pressure(env)
    states, actions = [env.state()], []
    for t in range(args.warmup):
        ph = warm(t)
        actions.append(env.encode_action(np.clip(ph, 0, P_max - 1)))
        states.append(env.step(ph))

    d_true = {k: [] for k in ks}
    d_model = {k: [] for k in ks}
    lvl_true, lvl_model = [], []          # absolute hold-cost, for scale
    H = args.horizon

    for a in range(args.n_anchors):
        for t in range(args.anchor_every):          # advance the trajectory
            ph = warm(len(actions))
            actions.append(env.encode_action(np.clip(ph, 0, P_max - 1)))
            states.append(env.step(ph))
        sh = np.stack(states[-HS:]).astype(np.float32)
        ah = np.stack(actions[-HS:]).astype(np.float32)
        z_ctx, a_ctx = encode_context(model, sh, ah, n_nodes, ni, nm, device)
        cur = states[-1].reshape(n_nodes, F)[:, P_max:2 * P_max].argmax(1)     # (N,)
        snap = env.save_state(f"_switchpen_snap_{os.getpid()}_{args.seed}.xml")

        # build the candidate set: hold + repeats of each k
        plans, tags = [], []
        plans.append(np.tile(cur[:, None], (1, H)))          # hold
        tags.append(0)
        for k in ks[1:]:
            for _ in range(args.repeats):
                p = cur.copy()
                flip = rng.choice(n_nodes, size=k, replace=False)
                for j in flip:
                    p[j] = (cur[j] + rng.integers(1, P_max)) % P_max
                plans.append(np.tile(p[:, None], (1, H)))
                tags.append(k)
        plans = np.stack(plans)                               # (C,N,H)
        tags = np.array(tags)

        # model cost for all candidates
        pe = cem_rollout(model, z_ctx, a_ctx, plans, n_nodes, P_max, ni, nm, device)
        mc = decode(pr, pe).clamp(min=0).sum(dim=(1, 2, 3)).cpu().numpy()      # (C,)

        # oracle cost for all candidates
        oc = np.zeros(len(plans))
        for c in range(len(plans)):
            env.load_state(snap)
            tot = 0.0
            for h in range(H):
                st = env.step(plans[c, :, h])
                tot += float(st.reshape(n_nodes, F)[:, :P_max].sum())
            oc[c] = tot
        env.load_state(snap)

        hold_m, hold_o = mc[0], oc[0]
        lvl_model.append(hold_m); lvl_true.append(hold_o)
        for k in ks[1:]:
            sel = tags == k
            d_model[k].append(float(np.mean(mc[sel] - hold_m)))
            d_true[k].append(float(np.mean(oc[sel] - hold_o)))
        print(f"  anchor {a:>2}  hold: model {hold_m:7.0f} / true {hold_o:7.0f}   "
              + "  ".join(f"k{k}:Δm{np.mean(d_model[k][-1]):+6.0f}/Δt{np.mean(d_true[k][-1]):+6.0f}" for k in ks[1:]))
    env.close()

    # ---- fit slopes: Δcost vs k ----
    kk = np.array(ks[1:], float)
    Dm = np.array([np.mean(d_model[k]) for k in ks[1:]])
    Dt = np.array([np.mean(d_true[k]) for k in ks[1:]])
    Sm = np.array([np.std(d_model[k]) for k in ks[1:]])
    St = np.array([np.std(d_true[k]) for k in ks[1:]])
    sl_m = np.polyfit(kk, Dm, 1)[0]
    sl_t = np.polyfit(kk, Dt, 1)[0]

    print(f"\n=== switch penalty: mean Δcost vs #signals flipped (vs 'hold all') ===")
    print(f"{'k':>3} {'Δmodel':>10} {'±sd':>7} {'Δtrue(SUMO)':>13} {'±sd':>7} {'model/true':>11}")
    for i, k in enumerate(ks[1:]):
        ratio = Dm[i] / Dt[i] if abs(Dt[i]) > 1e-6 else float("nan")
        print(f"{k:>3} {Dm[i]:>10.1f} {Sm[i]:>7.1f} {Dt[i]:>13.1f} {St[i]:>7.1f} {ratio:>11.2f}")
    ratio_sl = sl_m / sl_t if abs(sl_t) > 1e-6 else float("nan")
    print(f"\nslope Δcost/flip :  model {sl_m:+.1f}   true {sl_t:+.1f}   model/true slope = {ratio_sl:.2f}")
    print(f"hold-cost scale  :  model {np.mean(lvl_model):.0f}   true {np.mean(lvl_true):.0f}  "
          f"(H={H} steps, {n_nodes} signals)")

    if sl_t <= 0:
        verdict = ("true switching does NOT reliably raise cost here (congested regime: "
                   "flipping can help) -> 'switch penalty' framing doesn't apply cleanly")
    elif sl_m / sl_t < 0.5:
        verdict = ("model UNDER-WEIGHTS the switch penalty (slope < 50% of true) -> "
                   "STRUCTURAL fix: make transition/elapsed cost explicit, not just coverage")
    elif sl_m / sl_t > 1.5:
        verdict = "model OVER-weights switching -> not the exploitation direction here"
    else:
        verdict = ("model tracks the mean switch penalty (slope ~matches) -> the "
                   "over-optimism is about VARIANCE / specific plans -> COVERAGE fix")
    print(f"\n>>> VERDICT: {verdict}")

    import json
    from pathlib import Path
    Path(f"{args.data_dir}/diag_switch_penalty.json").write_text(json.dumps(dict(
        run=args.run, ks=ks, d_model_mean=Dm.tolist(), d_true_mean=Dt.tolist(),
        d_model_sd=Sm.tolist(), d_true_sd=St.tolist(),
        slope_model=float(sl_m), slope_true=float(sl_t),
        hold_cost_model=float(np.mean(lvl_model)), hold_cost_true=float(np.mean(lvl_true)),
        verdict=verdict), indent=2))
    print(f"wrote {args.data_dir}/diag_switch_penalty.json")


if __name__ == "__main__":
    main()
