"""Experiment #1 matched-switching control.

#1 (diag_plan_vs_beh) found model latent-rollout error ~4.3x larger on CEM-elite
(PLAN) action sequences than on scripted-controller (BEH) sequences. But PLAN and
BEH differ in TWO ways at once:
  (a) PLAN sequences are drawn from the planner's search distribution (a specific
      OOD *joint* phase pattern), and
  (b) PLAN sequences SWITCH phases far more often than any scripted controller.
So the 4.3x is a compound effect. This script de-conflates (a) from (b) with two
extra groups measured from the SAME frozen states, against the SAME ground truth:

  beh            : scripted (max_pressure / fixed_time / eps-hold)         -- low switch rate, in-dist
  plan           : CEM elite                                              -- high switch rate, planner-selected
  rand_matched   : RANDOM joint phase sequences, switch count per signal
                   resampled to match each PLAN elite (switch TIMING + target
                   phase random)                                          -- high switch rate, NOT planner-selected
  plan_downsampled: PLAN elites with switches thinned (hold phases longer)
                   until the per-signal switch count matches the scripted mean -- low switch rate, planner-selected

Reads:
  eps(rand_matched) ~ eps(plan)      -> the 4.3x is JUST switching frequency; drop "planner-induced".
  eps(plan_downsampled) ~ eps(beh)   -> same conclusion from the other side.
  eps(rand_matched) ~ eps(beh) AND eps(plan_downsampled) ~ eps(plan)
                                     -> switching frequency is NOT the driver; the
                                        planner's OOD joint *combination* is. #1 stands.
  intermediate                       -> switching explains part; report the split.

Usage:
  SUMOCFG_COLOGNE8=/path/to/cologne8.sumocfg \
  python diag_matched_switching.py --run L05ar --seeds 777 101 202 --n_anchors 20
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from eval_multi_sumo import load, encode_batch
from visualize_rollout_cologne8 import fit_pressure_probe, decode
from plan_cem_multi import encode_context, cem_rollout, SUMOCFG, HS
from traffic.sumo_multi_env import (
    SumoMultiEnv, controller_max_pressure, controller_fixed_time,
)


def switch_counts(seq):
    """seq (N,H) int -> per-signal number of steps where the phase changes (length H-1)."""
    return (seq[:, 1:] != seq[:, :-1]).sum(1)  # (N,)


def joint_switch_rate(seq):
    return float((seq[:, 1:] != seq[:, :-1]).mean())


def resample_random_matched(target_seq, cur, P, H, rng):
    """A random joint sequence with the SAME per-signal switch count as target_seq,
    switch times uniform-random, switch targets uniform-random (!= current phase)."""
    N = target_seq.shape[0]
    sc = switch_counts(target_seq)                       # (N,)
    out = np.tile(cur[:, None], (1, H)).astype(np.int64)
    for i in range(N):
        k = int(min(sc[i], H - 1))
        times = np.sort(rng.choice(np.arange(1, H), size=k, replace=False)) if k > 0 else []
        ph = int(cur[i])
        for h in range(1, H):
            if h in times:
                ph = (ph + int(rng.integers(1, P))) % P
            out[i, h] = ph
    return out


def downsample_switches(plan_seq, target_count, cur):
    """plan_seq (N,H) -> keep only the FIRST `target_count` switches per signal,
    hold the phase otherwise. target_count is a scalar (scripted mean)."""
    N, H = plan_seq.shape
    out = np.tile(cur[:, None], (1, H)).astype(np.int64)
    for i in range(N):
        kept = 0
        ph = int(cur[i])
        for h in range(1, H):
            want = plan_seq[i, h] != plan_seq[i, h - 1]
            if want and kept < target_count:
                ph = int(plan_seq[i, h])
                kept += 1
            out[i, h] = ph
    return out


def true_rollout(env, snap, phases, n_nodes, F, P_max):
    env.load_state(snap)
    out = np.zeros((phases.shape[1], n_nodes * F), dtype=np.float32)
    for h in range(phases.shape[1]):
        out[h] = env.step(phases[:, h])
    env.load_state(snap)
    return out


def seq_latent_err(model, pr, env, snap, seqs_arr, z_ctx, a_ctx, states,
                   n_nodes, P_max, F, ni, nm, device, H):
    """seqs_arr (K,N,H) -> (per-h latent MSE summed over K, count K, per-h decoded MSE summed)."""
    pe = cem_rollout(model, z_ctx, a_ctx, seqs_arr, n_nodes, P_max, ni, nm, device)  # (K,N,H,d)
    lat = np.zeros(H); dec = np.zeros(H); k_ok = 0
    pref = np.stack(states[-HS:]).astype(np.float32)
    for k in range(seqs_arr.shape[0]):
        tp = true_rollout(env, snap, seqs_arr[k], n_nodes, F, P_max)
        full = np.concatenate([pref, tp], 0)[None]
        with torch.no_grad():
            o = encode_batch(model, torch.from_numpy(full).float(),
                             torch.zeros(1, HS + H, n_nodes * P_max), n_nodes, ni, nm, device)
        zt = o["emb"].reshape(1, n_nodes, HS + H, -1)[0, :, HS:, :]      # (N,H,d)
        zm = pe[k]
        lat += ((zm - zt) ** 2).mean(-1).mean(0).cpu().numpy()
        dm = decode(pr, zm).clamp(min=0)[..., :P_max].sum(-1)
        dtt = torch.from_numpy(tp).to(device).view(H, n_nodes, F)[..., :P_max].sum(-1).T
        dec += ((dm - dtt) ** 2).mean(0).cpu().numpy()
        k_ok += 1
    return lat, k_ok, dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="L05ar")
    ap.add_argument("--weights", default="weights_epoch_80.pt")
    ap.add_argument("--data_dir", default="traffic_data_cologne8")
    ap.add_argument("--sumocfg", default=SUMOCFG)
    ap.add_argument("--begin", type=int, default=25200)
    ap.add_argument("--seeds", type=int, nargs="+", default=[777, 101, 202])
    ap.add_argument("--n_anchors", type=int, default=20)
    ap.add_argument("--anchor_every", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--n_plan", type=int, default=12)
    ap.add_argument("--cem_S", type=int, default=64)
    ap.add_argument("--cem_iters", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = torch.load(f"{args.data_dir}/val.pt", weights_only=False)
    n_nodes, P_max, F = meta["n_nodes"], meta["P_max"], meta["node_feature_dim"]
    ni = torch.from_numpy(np.asarray(meta["neighbor_idx"]))
    nm = torch.from_numpy(np.asarray(meta["neighbor_mask"]))
    H = args.H
    model, _ = load(args.run, args.weights, device)
    pr = fit_pressure_probe(model, f"{args.data_dir}/train.pt", n_nodes, P_max, F, ni, nm, device)
    print(f"model {args.run}  probe MSE {pr['mse']:.1f}  sumocfg {args.sumocfg}")

    GROUPS = ["beh", "plan", "rand_matched", "plan_downsampled"]
    lat = {g: np.zeros(H) for g in GROUPS}
    dec = {g: np.zeros(H) for g in GROUPS}
    cnt = {g: 0 for g in GROUPS}
    swr = {g: [] for g in GROUPS}          # joint switch rate of the sequences actually used

    for seed in args.seeds:
        env = SumoMultiEnv(args.sumocfg, seed=seed, warmup=0, metrics=False, begin=args.begin)
        env.reset()
        mp = controller_max_pressure(env)
        ft = controller_fixed_time(env, np.random.default_rng(seed))
        rng = np.random.default_rng(seed)
        states, actions = [env.state()], []
        for t in range(args.warmup):
            ph = mp(t)
            actions.append(env.encode_action(np.clip(ph, 0, P_max - 1)))
            states.append(env.step(ph))

        for a in range(args.n_anchors):
            for _ in range(args.anchor_every):
                ph = mp(len(actions))
                actions.append(env.encode_action(np.clip(ph, 0, P_max - 1)))
                states.append(env.step(ph))
            sh = np.stack(states[-HS:]).astype(np.float32)
            ah = np.stack(actions[-HS:]).astype(np.float32)
            z_ctx, a_ctx = encode_context(model, sh, ah, n_nodes, ni, nm, device)
            cur = states[-1].reshape(n_nodes, F)[:, P_max:2 * P_max].argmax(1)
            snap = env.save_state(f"_ms_snap_{os.getpid()}_{seed}.xml")

            # ---- BEH ----
            beh_seqs = []
            env.load_state(snap)
            beh_seqs.append(np.stack([np.clip(mp(1000 + h), 0, P_max - 1) for h in range(H)], 1))
            env.load_state(snap)
            beh_seqs.append(np.stack([np.clip(ft(1000 + h), 0, P_max - 1) for h in range(H)], 1))
            env.load_state(snap)
            for _ in range(2):
                s = np.tile(cur[:, None], (1, H)).astype(np.int64)
                for h in range(1, H):
                    s[:, h] = s[:, h - 1]
                    if rng.random() < 0.15:
                        j = rng.integers(n_nodes)
                        s[j, h] = (s[j, h - 1] + rng.integers(1, P_max)) % P_max
                beh_seqs.append(s)

            # ---- PLAN (CEM elite) ----
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

            # scripted mean per-signal switch count (for down-sampling target)
            beh_sc = np.mean([switch_counts(s).mean() for s in beh_seqs])
            tgt_count = int(round(beh_sc))

            # ---- rand_matched + plan_downsampled, one per plan elite ----
            rand_seqs = [resample_random_matched(ps, cur, P_max, H, rng) for ps in plan_seqs]
            down_seqs = [downsample_switches(ps, tgt_count, cur) for ps in plan_seqs]

            for grp, seqs in (("beh", beh_seqs), ("plan", plan_seqs),
                              ("rand_matched", rand_seqs), ("plan_downsampled", down_seqs)):
                arr = np.stack(seqs)
                l, k, d = seq_latent_err(model, pr, env, snap, arr, z_ctx, a_ctx, states,
                                         n_nodes, P_max, F, ni, nm, device, H)
                lat[grp] += l; dec[grp] += d; cnt[grp] += k
                swr[grp].append(float(np.mean([joint_switch_rate(s) for s in seqs])))

            r_now = (lat["plan"] / max(cnt["plan"], 1)).mean() / ((lat["beh"] / max(cnt["beh"], 1)).mean() + 1e-9)
            rm_now = (lat["rand_matched"] / max(cnt["rand_matched"], 1)).mean() / ((lat["beh"] / max(cnt["beh"], 1)).mean() + 1e-9)
            print(f"  seed {seed} anchor {a:>2}  running ratio plan/beh {r_now:.2f}  rand_matched/beh {rm_now:.2f}")
        env.close()

    for g in GROUPS:
        lat[g] /= max(cnt[g], 1)
        dec[g] /= max(cnt[g], 1)

    base = lat["beh"] + 1e-9
    ratios = {g: float(np.mean(lat[g] / base)) for g in GROUPS}
    print(f"\n=== matched-switching control  ({args.run}, {len(args.seeds)} seeds, {args.n_anchors} anchors/seed) ===")
    print(f"{'group':>18} {'meanLatRatio':>13} {'jointSwitchRate':>16} {'nseq':>6}")
    for g in GROUPS:
        print(f"{g:>18} {ratios[g]:>13.2f} {np.mean(swr[g]):>16.3f} {cnt[g]:>6}")
    print(f"\n{'h':>3} " + " ".join(f"{g[:10]:>11}" for g in GROUPS))
    for h in range(H):
        print(f"{h+1:>3} " + " ".join(f"{lat[g][h]:>11.4f}" for g in GROUPS))

    # attribution
    r_plan = ratios["plan"]; r_rand = ratios["rand_matched"]; r_down = ratios["plan_downsampled"]
    switch_share = (r_rand - 1.0) / (r_plan - 1.0 + 1e-9)          # fraction of the excess explained by switching alone
    planner_share = 1.0 - switch_share
    if r_rand >= 0.8 * r_plan:
        verdict = (f"the {r_plan:.1f}x is ~ENTIRELY switching frequency: random joint sequences "
                   f"matched to the planner's switch rate are already {r_rand:.1f}x. "
                   f"'Planner-induced OOD combination' is NOT supported by #1 as stated -- reframe to "
                   f"'model error grows with joint switching frequency, which the planner drives up'.")
    elif r_rand <= 1.3 and r_down >= 0.7 * r_plan:
        verdict = (f"switching frequency is NOT the driver: rand_matched stays near beh ({r_rand:.1f}x) "
                   f"while plan_downsampled stays near plan ({r_down:.1f}x). The planner's OOD *joint "
                   f"combination* is what the model gets wrong. #1's 4.3x stands as a distribution-shift result.")
    else:
        verdict = (f"SPLIT: switching frequency explains ~{100*switch_share:.0f}% of the excess error "
                   f"(rand_matched {r_rand:.1f}x vs plan {r_plan:.1f}x), planner-selection ~{100*planner_share:.0f}% "
                   f"(plan_downsampled {r_down:.1f}x). Report both components.")
    print(f"\n>>> VERDICT: {verdict}")

    out = args.out or f"{args.data_dir}/diag_matched_switching.json"
    Path(out).write_text(json.dumps(dict(
        run=args.run, seeds=args.seeds, n_anchors=args.n_anchors, H=H,
        mean_lat_ratio=ratios,
        joint_switch_rate={g: float(np.mean(swr[g])) for g in GROUPS},
        lat_by_h={g: lat[g].tolist() for g in GROUPS},
        dec_by_h={g: dec[g].tolist() for g in GROUPS},
        switch_share=float(switch_share), planner_share=float(planner_share),
        verdict=verdict), indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
