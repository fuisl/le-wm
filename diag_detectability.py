"""Experiment #4 (fast cut): can ANY cheap runtime signal predict per-plan regret?

The whole Paper 1 headline (CB2) is that planner-induced model error is NOT
detectable at the plan level. Step 0 tested one crude signal (latent kNN). This
tests the standard ones:

  ens_disagree  : std over a 4-checkpoint ensemble of the z-scored plan cost
                  (L05ar / L0ar / L0ar_v2 / L05ar_v2 - zero extra training)
  gmm_ood       : negative log-likelihood of the plan's (z, action) rollout under
                  a GMM fit on training (z, action)  -- a proper density model
  disp_norm     : mean latent displacement per rollout step, / training 1-step std
                  (Delta-JEPA-style "is the model making big moves")
  frac_clipped  : fraction of decoded rollout values that needed clamping to >= 0
                  (the model predicting physically impossible states)
  n_switch0     : agents flipping phase at h=0 (the Step-0 action-structure signal)
  cov_seq       : mean kNN dist of the imagined rollout latents to training bank

against per-candidate regret (oracle_cost - min oracle_cost) and |over-optimism|,
within each decision step (rank corr) and pooled.

Verdict: any |within-step Spearman| > 0.35 -> the "undetectable" claim is in
trouble; pivot the framing.  All weak -> the negative holds, stress-tested.

Usage:  python diag_detectability.py --seeds 777 101 --n_steps 25
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr
from sklearn.mixture import GaussianMixture

from eval_multi_sumo import load, encode_batch
from visualize_rollout_cologne8 import fit_pressure_probe, decode
from plan_cem_multi import encode_context, cem_rollout, SUMOCFG, HS
from diag_exploitation_coverage import build_coverage_bank, knn_dist
from traffic.sumo_multi_env import SumoMultiEnv, controller_max_pressure

DATA_DIR = "traffic_data_cologne8"
# default = the old heterogeneous-checkpoint set (#4 fast cut). Pass --ensemble
# L05ar_bs1001 L05ar_bs2002 ... for the #4-proper seed-only bootstrap ensemble.
ENSEMBLE = ["L05ar", "L0ar", "L0ar_v2", "L05ar_v2"]


def build_joint_bank(model, path, n_nodes, ni, nm, device):
    """One row per training timestep: the JOINT latent = per-node latents concatenated
    in node order (N*d), per-dim standardised. Used to score whether a candidate plan's
    joint configuration (not each agent in isolation) is covered by training."""
    data = torch.load(path, weights_only=False)
    Zs = []
    with torch.no_grad():
        for ep in data["episodes"]:
            st = torch.from_numpy(ep["state"]).unsqueeze(0).float()
            ac = torch.from_numpy(ep["action"]).unsqueeze(0).float()
            out = encode_batch(model, st, ac, n_nodes, ni, nm, device)
            T = st.shape[1]
            z = out["emb"].reshape(1, n_nodes, T, -1)[0].permute(1, 0, 2)   # (T,N,d)
            Zs.append(z.reshape(T, -1).cpu())                                # (T, N*d)
    Z = torch.cat(Zs).float()
    mu, sd = Z.mean(0), Z.std(0).clamp(min=1e-6)
    Z = (Z - mu) / sd
    print(f"  [joint bank] {Z.shape[0]} timestep rows, dim={Z.shape[1]} (= N*d)")
    return Z.to(device), mu, sd


def bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=0):
    """percentile bootstrap CI of the mean of `values` (a 1-D array)."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def onehot(idx, P):
    o = np.zeros((*idx.shape, P), np.float32)
    np.put_along_axis(o, idx[..., None], 1.0, -1)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", default="L05ar")
    ap.add_argument("--weights", default="weights_epoch_80.pt")
    ap.add_argument("--ensemble", nargs="+", default=ENSEMBLE,
                    help="run names for the disagreement ensemble; pass the L05ar_bs* "
                         "seed-only runs for the #4-proper bootstrap ensemble")
    ap.add_argument("--tag", default="", help="suffix for output filenames")
    ap.add_argument("--solo_agent", type=int, default=-1,
                    help="MA-specificity control: if >=0, only this intersection's H-step "
                         "phase sequence is planned (space P^H, ~matched to the 8-agent joint "
                         "space); all other signals hold their current phase. Detectability "
                         "here vs the full joint search isolates whether it is the FACTORED "
                         "joint structure or just a large action space that defeats detection.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[777, 101])
    ap.add_argument("--n_steps", type=int, default=25)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--S", type=int, default=64)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = torch.load(f"{DATA_DIR}/val.pt", weights_only=False)
    n, P, F = meta["n_nodes"], meta["P_max"], meta["node_feature_dim"]
    ni = torch.from_numpy(np.asarray(meta["neighbor_idx"]))
    nm = torch.from_numpy(np.asarray(meta["neighbor_mask"]))
    H, S = args.horizon, args.S

    prim, _ = load(args.primary, args.weights, device)
    pr = fit_pressure_probe(prim, f"{DATA_DIR}/train.pt", n, P, F, ni, nm, device)
    print(f"primary {args.primary}  probe MSE {pr['mse']:.1f}")

    # ensemble members + their probes
    ens = []
    for name in args.ensemble:
        m, _ = load(name, args.weights, device)
        p = fit_pressure_probe(m, f"{DATA_DIR}/train.pt", n, P, F, ni, nm, device)
        ens.append((name, m, p))
    print(f"ensemble ({len(ens)}): {[e[0] for e in ens]}")

    # joint-configuration bank (primary) for the MA-specificity signal
    Zj, MUj, SDj = build_joint_bank(prim, f"{DATA_DIR}/train.pt", n, ni, nm, device)

    # coverage bank (primary) + GMM on (z, onehot(action)) + 1-step displacement stats
    Zb, PHb, MU, SD, _ = build_coverage_bank(prim, f"{DATA_DIR}/train.pt", n, F, P, ni, nm, device)
    Zb_by_phase = {ph: Zb[PHb == ph].to(device) for ph in range(P)}
    Xg = np.concatenate([((Zb - MU) / SD).numpy(), onehot(PHb.numpy().astype(int), P)], 1)
    gmm = GaussianMixture(n_components=16, covariance_type="diag", random_state=0,
                          reg_covar=1e-4).fit(Xg[np.random.default_rng(0).permutation(len(Xg))[:20000]])
    # training 1-step latent displacement std (primary): from consecutive bank rows in an episode
    tr = torch.load(f"{DATA_DIR}/train.pt", weights_only=False)
    disps = []
    with torch.no_grad():
        for ep in tr["episodes"][:8]:
            st = torch.from_numpy(ep["state"]).unsqueeze(0).float()
            ac = torch.from_numpy(ep["action"]).unsqueeze(0).float()
            o = encode_batch(prim, st, ac, n, ni, nm, device)
            z = o["emb"].reshape(1, n, -1, prim.encoder.emb_dim if hasattr(prim.encoder, "emb_dim") else 64)[0]
            disps.append((z[:, 1:] - z[:, :-1]).norm(dim=-1).reshape(-1).cpu())
    disp_std = float(torch.cat(disps).mean())
    print(f"GMM fit on {min(len(Xg),20000)} rows; training 1-step |Δz| mean {disp_std:.3f}")

    def roll_member(m, z_ctx, a_ctx, phases):
        return cem_rollout(m, z_ctx, a_ctx, phases, n, P, ni, nm, device)   # (S,N,H,d)

    rows = []
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        env = SumoMultiEnv(SUMOCFG, seed=seed, warmup=0, metrics=False, begin=25200)
        env.reset()
        mp = controller_max_pressure(env)
        states, actions = [env.state()], []
        for t in range(args.warmup):
            ph = mp(t)
            actions.append(env.encode_action(np.clip(ph, 0, P - 1)))
            states.append(env.step(ph))

        for step in range(args.n_steps):
            sh = np.stack(states[-HS:]).astype(np.float32)
            ah = np.stack(actions[-HS:]).astype(np.float32)
            z_ctx, a_ctx = encode_context(prim, sh, ah, n, ni, nm, device)
            cur = states[-1].reshape(n, F)[:, P:2 * P].argmax(1)

            # CEM with the primary
            probs = np.full((n, H, P), 1.0 / P)
            phases = None
            for _ in range(args.iters):
                phases = np.empty((S, n, H), np.int64)
                for i in range(n):
                    for h in range(H):
                        phases[:, i, h] = rng.choice(P, size=S, p=probs[i, h])
                if args.solo_agent >= 0:
                    for i in range(n):
                        if i != args.solo_agent:
                            phases[:, i, :] = cur[i]          # non-planned signals hold
                pe = roll_member(prim, z_ctx, a_ctx, phases)
                mc = decode(pr, pe).clamp(min=0).sum(dim=(1, 2, 3)).cpu().numpy()
                elite = phases[np.argsort(mc)[: args.topk]]
                for i in range(n):
                    for h in range(H):
                        c = np.bincount(elite[:, i, h], minlength=P)
                        probs[i, h] = (c + 1e-3) / (c.sum() + P * 1e-3)
            pe = roll_member(prim, z_ctx, a_ctx, phases)                 # (S,N,H,d) primary
            dec = decode(pr, pe)
            mc = dec.clamp(min=0).sum(dim=(1, 2, 3)).cpu().numpy()

            # --- signals ---
            # ensemble disagreement: z-scored plan cost per member, std over members
            costs = []
            for _, m, p in ens:
                pk = roll_member(m, z_ctx, a_ctx, phases)
                ck = decode(p, pk).clamp(min=0).sum(dim=(1, 2, 3)).cpu().numpy()
                costs.append((ck - ck.mean()) / (ck.std() + 1e-9))
            ens_dis = np.stack(costs).std(0)                            # (S,)
            # gmm ood: mean neg-log-lik of (z, onehot(a)) along the rollout
            zt = ((pe.reshape(-1, pe.shape[-1]).cpu().numpy() - MU.numpy()) / SD.numpy())
            at = onehot(phases.reshape(S, n, H).transpose(0, 1, 2).reshape(-1), P)
            gm = -gmm.score_samples(np.concatenate([zt, at], 1)).reshape(S, n, H).mean((1, 2))
            # displacement norm / training std
            disp = (pe[:, :, 1:] - pe[:, :, :-1]).norm(dim=-1).mean(dim=(1, 2)).cpu().numpy() / disp_std
            # frac clipped (decoded negatives)
            frac_clip = (dec < 0).float().mean(dim=(1, 2, 3)).cpu().numpy()
            # n_switch0
            nsw0 = (phases[:, :, 0] != cur[None, :]).sum(1)
            # cov_seq (imagined latents)
            zc = ((pe.reshape(-1, pe.shape[-1]).cpu()))
            pc = torch.from_numpy(phases).reshape(-1)
            dd = torch.full((S * n * H,), float("nan"))
            for ph in range(P):
                mk = pc == ph
                if mk.any():
                    dd[mk] = knn_dist(((zc[mk] - MU) / SD), Zb_by_phase[ph], k=args.k)
            cov_seq = dd.reshape(S, n, H).mean((1, 2)).numpy()
            # cov_joint: kNN dist of the JOINT rollout config (concat over agents) to the
            # joint training bank. cov_seq above averages per-agent coverage; this asks
            # whether the *combination* of agent states/phases is jointly covered.
            zj = pe.permute(0, 2, 1, 3).reshape(S, H, n * pe.shape[-1])       # (S,H,N*d)
            zj = ((zj.reshape(S * H, -1).cpu() - MUj.cpu()) / SDj.cpu())
            cov_joint = knn_dist(zj, Zj, k=args.k).reshape(S, H).mean(1).numpy()

            # --- oracle ---
            snap = env.save_state(f"_det_snap_{os.getpid()}_{seed}.xml")
            oc = np.zeros(S)
            for s in range(S):
                env.load_state(snap)
                tot = 0.0
                for h in range(H):
                    stt = env.step(phases[s, :, h])
                    tot += float(stt.reshape(n, F)[:, :P].sum())
                oc[s] = tot
            env.load_state(snap)
            regret = oc - oc.min()
            mz = (mc - mc.mean()) / (mc.std() + 1e-9)
            oz = (oc - oc.mean()) / (oc.std() + 1e-9)
            optimism = np.abs(oz - mz)

            for s in range(S):
                rows.append(dict(seed=seed, step=step, cand=s,
                                 regret=float(regret[s]), optimism=float(optimism[s]),
                                 ens_disagree=float(ens_dis[s]), gmm_ood=float(gm[s]),
                                 disp_norm=float(disp[s]), frac_clipped=float(frac_clip[s]),
                                 n_switch0=int(nsw0[s]), cov_seq=float(cov_seq[s]),
                                 cov_agent=float(cov_seq[s]), cov_joint=float(cov_joint[s])))

            exec_phase = probs[:, 0].argmax(1)
            actions.append(env.encode_action(np.clip(exec_phase, 0, P - 1)))
            states.append(env.step(exec_phase))
            print(f"  seed {seed} step {step:>2}  regret mean {regret.mean():.0f}  "
                  f"ens_dis {ens_dis.mean():.2f}")
        env.close()

    analyse(rows, args)


def analyse(rows, args):
    import csv
    sigs = ["ens_disagree", "gmm_ood", "disp_norm", "frac_clipped", "n_switch0",
            "cov_seq", "cov_agent", "cov_joint"]
    arr = {k: np.array([r[k] for r in rows], float) for k in sigs + ["regret", "optimism", "step", "seed"]}
    step_keys = sorted(set(zip(arr["seed"].tolist(), arr["step"].tolist())))

    def within_per_step(x, y):
        """list of per-decision-step Spearman(x, y) (one value per step)."""
        rs = []
        for (sd, st) in step_keys:
            m = (arr["seed"] == sd) & (arr["step"] == st)
            if m.sum() >= 10:
                r = spearmanr(x[m], y[m])[0]
                if np.isfinite(r):
                    rs.append(r)
        return np.array(rs)

    def within(x, y):
        v = within_per_step(x, y)
        return float(v.mean()) if len(v) else float("nan")

    print(f"\n=== per-plan regret / |optimism| vs runtime signals  "
          f"({len(rows)} plans, {len(step_keys)} decision steps, {len(args.seeds)} seeds) ===")
    print(f"{'signal':>14} {'pool r(regret)':>14} {'within r(regret)':>16} {'95% CI':>18} {'within r(|opt|)':>15}")
    out = {}
    best = ("", 0.0)
    for k in sigs:
        pr_ = pearsonr(arr[k], arr["regret"])[0]
        per_step = within_per_step(arr[k], arr["regret"])
        wr = float(per_step.mean()) if len(per_step) else float("nan")
        lo, hi = bootstrap_ci(per_step)
        wo = within(arr[k], arr["optimism"])
        out[k] = dict(pool_pearson_regret=float(pr_), within_spearman_regret=wr,
                      within_spearman_regret_ci=[lo, hi], n_steps=int(len(per_step)),
                      within_spearman_optimism=wo)
        print(f"{k:>14} {pr_:>14.3f} {wr:>16.3f}   [{lo:+.2f},{hi:+.2f}] {wo:>15.3f}")
        if k not in ("cov_agent",) and abs(wr) > abs(best[1]):   # cov_agent duplicates cov_seq
            best = (k, wr)

    # --- MA-specificity: per-agent-mean coverage vs joint-config coverage ---
    ca = within_per_step(arr["cov_agent"], arr["regret"])
    cj = within_per_step(arr["cov_joint"], arr["regret"])
    d_per_step = cj[:min(len(ca), len(cj))] - ca[:min(len(ca), len(cj))]
    ca_ci, cj_ci = bootstrap_ci(ca), bootstrap_ci(cj)
    d_ci = bootstrap_ci(d_per_step)
    print(f"\n=== MA-specificity: does the JOINT configuration predict regret better than "
          f"per-agent coverage? ===")
    print(f"  cov_agent (per-agent-mean)  within r(regret) = {ca.mean():+.3f}  95% CI [{ca_ci[0]:+.2f}, {ca_ci[1]:+.2f}]")
    print(f"  cov_joint (joint config)    within r(regret) = {cj.mean():+.3f}  95% CI [{cj_ci[0]:+.2f}, {cj_ci[1]:+.2f}]")
    print(f"  paired difference (joint - agent)          = {d_per_step.mean():+.3f}  95% CI [{d_ci[0]:+.2f}, {d_ci[1]:+.2f}]")
    if d_ci[0] > 0.0:
        ma_verdict = ("joint-configuration coverage predicts per-plan regret better than per-agent "
                      "coverage (paired-difference 95% CI excludes 0) -> concrete factored-structure "
                      "result: the model's failure is about the joint config, not any agent in isolation. "
                      f"(joint {cj.mean():+.2f} vs agent {ca.mean():+.2f}; both still below a gateable ~0.5)")
    elif d_ci[1] < 0.0:
        ma_verdict = ("per-agent-mean coverage predicts per-plan regret better than the joint config "
                      "(paired-difference 95% CI excludes 0) -> the 'the joint structure is what the "
                      "model gets wrong' framing is NOT supported on this axis.")
    else:
        ma_verdict = ("joint and per-agent coverage predict regret about equally (paired-difference "
                      "95% CI spans 0). No isolable factored-structure effect on this axis; neither is "
                      "gateable anyway.")
    print(f"  >>> {ma_verdict}")
    ma_out = dict(cov_agent_r=float(ca.mean()), cov_agent_ci=list(ca_ci),
                  cov_joint_r=float(cj.mean()), cov_joint_ci=list(cj_ci),
                  joint_minus_agent=float(d_per_step.mean()), joint_minus_agent_ci=list(d_ci),
                  verdict=ma_verdict)

    # can we gate? fraction of oracle-advantage recoverable by picking the
    # lowest-signal candidate instead of the model's argmin, per step
    print(f"\n=== gate simulation: pick argmin(signal) instead of argmin(model_cost) ===")
    gate = {}
    for k in sigs:
        rec = []
        for (sd, st) in set(zip(arr["seed"], arr["step"])):
            m = (arr["seed"] == sd) & (arr["step"] == st)
            if m.sum() < 10:
                continue
            reg = arr["regret"][m]
            g_pick = np.argmin(arr[k][m])          # candidate the gate would keep
            # model's pick ~ argmin cost is not stored here; use mean regret as the "no-gate" ref
            rec.append((reg.mean() - reg[g_pick]) / (reg.mean() + 1e-9))
        gate[k] = float(np.mean(rec))
        print(f"{k:>14}  mean regret reduction vs a random/mean plan: {gate[k]*100:+.0f}%")

    thresh = 0.35
    hit = [k for k in sigs if abs(out[k]["within_spearman_regret"]) > thresh
           or abs(out[k]["within_spearman_optimism"]) > thresh]
    if hit:
        verdict = (f"DETECTABILITY SIGNAL FOUND: {hit} exceed |within-step Spearman| {thresh} "
                   f"-> the 'undetectable' claim needs revision; pivot toward a gate")
    else:
        verdict = (f"NEGATIVE HOLDS: no signal exceeds |within-step Spearman| {thresh} "
                   f"(best {best[0]} = {best[1]:+.2f}) -> plan-level error is not runtime-detectable, "
                   f"stress-tested against ensemble + density + displacement + plausibility")
    print(f"\n>>> VERDICT: {verdict}")

    tag = f"_{args.tag}" if args.tag else ""
    Path(f"{DATA_DIR}/diag_detectability{tag}.json").write_text(json.dumps(dict(
        primary=args.primary, ensemble=args.ensemble, seeds=args.seeds, n_steps=args.n_steps,
        solo_agent=args.solo_agent, n_decision_steps=len(step_keys),
        signals=out, gate_sim=gate, best=dict(name=best[0], within_spearman=best[1]),
        ma_specificity=ma_out, verdict=verdict), indent=2))
    with open(f"{DATA_DIR}/diag_detectability{tag}_rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {DATA_DIR}/diag_detectability{tag}.json (+ _rows.csv)")


if __name__ == "__main__":
    main()
