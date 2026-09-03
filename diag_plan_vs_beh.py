"""Experiment #1 (Exploitation Experiment Plan): epsilon_plan vs epsilon_beh.

Is world-model rollout error *concentrated on the planner's own action sequences*
(distribution shift), or is the model just uniformly bad?

Walk a MaxPressure trajectory; at M anchors, from the SAME frozen SUMO state, roll
the model H steps under:
  BEH   : action sequences a scripted controller produces (max_pressure, fixed_time,
          eps-random) - these are ~ rho_beh
  PLAN  : elite action sequences from a short CEM run - these are ~ rho_plan
For every sequence, also roll REAL SUMO H steps (saveState/loadState) and encode
the true states, so error is measured against ground truth, not against the
model's own imagined target.

  err(h) = mean_over_seqs || z_model[h] - z_true[h] ||   (latent, per-node)
  also decoded (probe -> total halting) MSE per horizon.

Read:  err_plan(h) >> err_beh(h)  and growing with h  -> distribution shift confirmed
       err_plan ~ err_beh                              -> model is bad everywhere;
                                                          "planner-induced" is wrong

Usage:  python diag_plan_vs_beh.py --run L05ar --n_anchors 20 --seed 777
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from eval_multi_sumo import load
from visualize_rollout_cologne8 import fit_pressure_probe, decode
from plan_cem_multi import encode_context, cem_rollout, SUMOCFG, HS
from diag_exploitation_coverage import build_coverage_bank, coverage_for_plans
from traffic.sumo_multi_env import (
    SumoMultiEnv, controller_max_pressure, controller_fixed_time,
)


def true_rollout(env, snap, phases, n_nodes, F, P_max):
    """phases (N,H) int -> (H, N*F) true states after each step, from snap."""
    env.load_state(snap)
    out = np.zeros((phases.shape[1], n_nodes * F), dtype=np.float32)
    halt = np.zeros(phases.shape[1])
    for h in range(phases.shape[1]):
        st = env.step(phases[:, h])
        out[h] = st
        halt[h] = float(st.reshape(n_nodes, F)[:, :P_max].sum())
    env.load_state(snap)
    return out, halt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="L05ar")
    ap.add_argument("--weights", default="weights_epoch_80.pt")
    ap.add_argument("--data_dir", default="traffic_data_cologne8")
    ap.add_argument("--sumocfg", default=SUMOCFG)
    ap.add_argument("--begin", type=int, default=25200)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--n_anchors", type=int, default=20)
    ap.add_argument("--anchor_every", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--n_plan", type=int, default=12, help="elite planner seqs per anchor")
    ap.add_argument("--cem_S", type=int, default=64)
    ap.add_argument("--cem_iters", type=int, default=3)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = torch.load(f"{args.data_dir}/val.pt", weights_only=False)
    n_nodes, P_max, F = meta["n_nodes"], meta["P_max"], meta["node_feature_dim"]
    ni = torch.from_numpy(np.asarray(meta["neighbor_idx"]))
    nm = torch.from_numpy(np.asarray(meta["neighbor_mask"]))
    H = args.H

    model, cfg = load(args.run, args.weights, device)
    pr = fit_pressure_probe(model, f"{args.data_dir}/train.pt", n_nodes, P_max, F, ni, nm, device)
    Zb, PHb, MU, SD, PHASE_LOGP = build_coverage_bank(
        model, f"{args.data_dir}/train.pt", n_nodes, F, P_max, ni, nm, device)
    Zb_by_phase = {p: Zb[PHb == p].to(device) for p in range(P_max)}
    print(f"model {args.run}  probe MSE {pr['mse']:.1f}")

    env = SumoMultiEnv(args.sumocfg, seed=args.seed, warmup=0, metrics=False, begin=args.begin)
    env.reset()
    mp = controller_max_pressure(env)
    ft = controller_fixed_time(env, np.random.default_rng(args.seed))
    rng = np.random.default_rng(args.seed)
    states, actions = [env.state()], []
    for t in range(args.warmup):
        ph = mp(t)
        actions.append(env.encode_action(np.clip(ph, 0, P_max - 1)))
        states.append(env.step(ph))

    # accumulators, keyed by group
    lat = {"beh": np.zeros(H), "plan": np.zeros(H)}
    dec = {"beh": np.zeros(H), "plan": np.zeros(H)}
    cnt = {"beh": 0, "plan": 0}
    cov = {"beh": [], "plan": []}     # cov_seq_true of each seq, for context
    per_anchor = []

    for a in range(args.n_anchors):
        for _ in range(args.anchor_every):
            ph = mp(len(actions))
            actions.append(env.encode_action(np.clip(ph, 0, P_max - 1)))
            states.append(env.step(ph))
        sh = np.stack(states[-HS:]).astype(np.float32)
        ah = np.stack(actions[-HS:]).astype(np.float32)
        z_ctx, a_ctx = encode_context(model, sh, ah, n_nodes, ni, nm, device)
        cur = states[-1].reshape(n_nodes, F)[:, P_max:2 * P_max].argmax(1)
        snap = env.save_state(f"_pvb_snap_{os.getpid()}_{args.seed}.xml")

        # ---- BEH action sequences ----
        beh_seqs = []
        # max_pressure for H steps from the frozen state
        env.load_state(snap)
        mp_seq = np.stack([np.clip(mp(1000 + h), 0, P_max - 1) for h in range(H)], 1)
        env.load_state(snap)
        beh_seqs.append(mp_seq)
        # fixed_time for H steps
        env.load_state(snap)
        ft_seq = np.stack([np.clip(ft(1000 + h), 0, P_max - 1) for h in range(H)], 1)
        env.load_state(snap)
        beh_seqs.append(ft_seq)
        # eps-random durations: hold current phase, flip one signal occasionally
        for _ in range(2):
            s = np.tile(cur[:, None], (1, H))
            for h in range(1, H):
                s[:, h] = s[:, h - 1]
                if rng.random() < 0.15:
                    j = rng.integers(n_nodes)
                    s[j, h] = (s[j, h - 1] + rng.integers(1, P_max)) % P_max
            beh_seqs.append(s)

        # ---- PLAN action sequences: short CEM, take the elite ----
        probs = np.full((n_nodes, H, P_max), 1.0 / P_max)
        for _ in range(args.cem_iters):
            ph = np.empty((args.cem_S, n_nodes, H), dtype=np.int64)
            for i in range(n_nodes):
                for h in range(H):
                    ph[:, i, h] = rng.choice(P_max, size=args.cem_S, p=probs[i, h])
            pe = cem_rollout(model, z_ctx, a_ctx, ph, n_nodes, P_max, ni, nm, device)
            mc = decode(pr, pe).clamp(min=0).sum(dim=(1, 2, 3)).cpu().numpy()
            elite = ph[np.argsort(mc)[: max(args.n_plan, 8)]]
            for i in range(n_nodes):
                for h in range(H):
                    c = np.bincount(elite[:, i, h], minlength=P_max)
                    probs[i, h] = (c + 1e-3) / (c.sum() + P_max * 1e-3)
        plan_seqs = list(elite[: args.n_plan])

        # ---- roll model + true SUMO for every sequence ----
        for grp, seqs in (("beh", beh_seqs), ("plan", plan_seqs)):
            seqs_arr = np.stack(seqs)                         # (K,N,H)
            pe = cem_rollout(model, z_ctx, a_ctx, seqs_arr, n_nodes, P_max, ni, nm, device)  # (K,N,H,d)
            _, cs = coverage_for_plans(pe, seqs_arr, Zb_by_phase, MU, SD, k=8)
            for k in range(len(seqs)):
                tp, _ = true_rollout(env, snap, seqs_arr[k], n_nodes, F, P_max)
                # encode the true H states with the shared HS context prefix
                pref = np.stack(states[-HS:]).astype(np.float32)
                full = np.concatenate([pref, tp], 0)[None]     # (1, HS+H, N*F)
                with torch.no_grad():
                    from eval_multi_sumo import encode_batch
                    o = encode_batch(model, torch.from_numpy(full).float(),
                                     torch.zeros(1, HS + H, n_nodes * P_max),
                                     n_nodes, ni, nm, device)
                zt = o["emb"].reshape(1, n_nodes, HS + H, -1)[0, :, HS:, :]   # (N,H,d)
                zm = pe[k]                                                    # (N,H,d)
                lat[grp] += ((zm - zt) ** 2).mean(-1).mean(0).cpu().numpy()   # per-h latent MSE
                dm = decode(pr, zm).clamp(min=0)[..., :P_max].sum(-1)         # (N,H)
                dtt = torch.from_numpy(tp).to(device).view(H, n_nodes, F)[..., :P_max].sum(-1).T
                dec[grp] += ((dm - dtt) ** 2).mean(0).cpu().numpy()
                cnt[grp] += 1
            cov[grp].append(float(np.mean(cs)))

        la = lat["plan"] / max(cnt["plan"], 1)
        lb = lat["beh"] / max(cnt["beh"], 1)
        per_anchor.append(dict(anchor=a, ratio_h1=float(la[0] / (lb[0] + 1e-9)),
                               ratio_h5=float(la[min(4, H - 1)] / (lb[min(4, H - 1)] + 1e-9))))
        print(f"  anchor {a:>2}  latent MSE ratio plan/beh  h1 {la[0]/(lb[0]+1e-9):.2f}  "
              f"h{min(5,H)} {la[min(4,H-1)]/(lb[min(4,H-1)]+1e-9):.2f}")
    env.close()

    for g in ("beh", "plan"):
        lat[g] /= max(cnt[g], 1)
        dec[g] /= max(cnt[g], 1)

    print(f"\n=== epsilon_plan vs epsilon_beh  ({cnt['beh']} beh seqs, {cnt['plan']} plan seqs) ===")
    print(f"{'h':>3} {'lat_beh':>9} {'lat_plan':>9} {'ratio':>7}   {'dec_beh':>9} {'dec_plan':>9} {'ratio':>7}")
    for h in range(H):
        print(f"{h+1:>3} {lat['beh'][h]:>9.4f} {lat['plan'][h]:>9.4f} "
              f"{lat['plan'][h]/(lat['beh'][h]+1e-9):>7.2f}   "
              f"{dec['beh'][h]:>9.1f} {dec['plan'][h]:>9.1f} "
              f"{dec['plan'][h]/(dec['beh'][h]+1e-9):>7.2f}")
    r_lat = float(np.mean(lat['plan'] / (lat['beh'] + 1e-9)))
    grow = float((lat['plan'] / (lat['beh'] + 1e-9))[-1] - (lat['plan'] / (lat['beh'] + 1e-9))[0])
    print(f"\nmean latent ratio plan/beh = {r_lat:.2f}   (grows by {grow:+.2f} h1->h{H})")
    print(f"cov_seq_true: beh {np.mean(cov['beh']):.2f}   plan {np.mean(cov['plan']):.2f}")
    verdict = ("distribution shift CONFIRMED: model error is markedly larger on planner "
               "actions" if r_lat > 1.3 else
               "weak/no distribution shift: model error is similar on beh and plan actions "
               "-> 'planner-induced' framing is not supported")
    print(f"\n>>> VERDICT: {verdict}")

    Path(f"{args.data_dir}/diag_plan_vs_beh.json").write_text(json.dumps(dict(
        run=args.run, seed=args.seed, H=H,
        lat_beh=lat['beh'].tolist(), lat_plan=lat['plan'].tolist(),
        dec_beh=dec['beh'].tolist(), dec_plan=dec['plan'].tolist(),
        mean_latent_ratio=r_lat, ratio_growth=grow,
        cov_beh=float(np.mean(cov['beh'])), cov_plan=float(np.mean(cov['plan'])),
        per_anchor=per_anchor, verdict=verdict), indent=2))
    print(f"wrote {args.data_dir}/diag_plan_vs_beh.json")


if __name__ == "__main__":
    main()
