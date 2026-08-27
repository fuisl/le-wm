"""Evaluate topology-free multi-agent JEPA checkpoints on cologne8.

Checks (all through a frozen linear probe back to raw per-node features, so
Level 0 / 0.5 / permute are compared in the same units):

 1. rollout error vs horizon: model vs persistence (full feature + pressure block)
 2. counterfactual accuracy: corr(||Δ pred emb||, ||Δ true state||) across branches
 3. plan-ranking: does the model's argmin-cost branch match SUMO's? (top-1 +
    mean Spearman), cost = summed total halting-pressure over the branch horizon
    - the MaxPressure-shaped quantity a CEM planner would minimise.

Usage:
    python eval_multi_sumo.py --data_dir traffic_data_cologne8 --runs L0 L05 L05perm --weights weights_epoch_80.pt
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from module import MLP, ARPredictor, Embedder
from traffic.dataset import TrafficDataset
from traffic.multi_agent import MultiAgentJEPA, masked_neighbor_mean, masked_neighbor_pna

EMBED_DIM = 64


def build_model(cfg):
    pred_in = EMBED_DIM if str(cfg["level"]) == "0" else 2 * EMBED_DIM
    bn = torch.nn.BatchNorm1d
    return MultiAgentJEPA(
        encoder=Embedder(input_dim=cfg["node_F"], smoothed_dim=EMBED_DIM, emb_dim=EMBED_DIM),
        predictor=ARPredictor(num_frames=cfg["history"], depth=4, heads=4, mlp_dim=256,
                              input_dim=pred_in, hidden_dim=EMBED_DIM, output_dim=EMBED_DIM,
                              dim_head=32, dropout=0.1, emb_dropout=0.0),
        action_encoder=Embedder(input_dim=cfg["node_A"], smoothed_dim=EMBED_DIM, emb_dim=EMBED_DIM),
        level=cfg["level"], permute_control=cfg["permute_control"],
        neighbor_agg=cfg.get("neighbor_agg", "mean"), emb_dim=EMBED_DIM,
        edge_dim=cfg.get("edge_dim", 0),
        projector=MLP(EMBED_DIM, 256, EMBED_DIM, norm_fn=bn),
        pred_proj=MLP(EMBED_DIM, 256, EMBED_DIM, norm_fn=bn),
    )


def load(run, weights, device):
    ck = torch.load(Path("traffic_runs_sumo", run, weights), weights_only=False)
    m = build_model(ck["cfg"]).to(device)
    m.load_state_dict(ck["model_state"])
    m.eval()
    return m, ck["cfg"]


def encode_batch(model, state, action, n_nodes, nbr_idx, nbr_mask, device, edge_feat=None):
    info = {"state": state.to(device), "action": action.to(device), "n_nodes": n_nodes}
    if model.level == "0.5":
        info["neighbor_idx"] = nbr_idx.to(device)
        info["neighbor_mask"] = nbr_mask.to(device)
    if model.edge_dim > 0 and edge_feat is not None:
        info["edge_feat"] = edge_feat.to(device)
    return model.encode(info)


def fit_probe(model, path, window, n_nodes, nbr_idx, nbr_mask, device):
    ds = TrafficDataset(path, window=window)
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False)
    X, Y = [], []
    with torch.no_grad():
        for b in loader:
            out = encode_batch(model, b["state"], b["action"], n_nodes, nbr_idx, nbr_mask, device, edge_feat=b.get("edge_feat"))
            B, T = b["state"].shape[:2]
            F = b["state"].shape[-1] // n_nodes
            st = b["state"].view(B, T, n_nodes, F)
            z = out["emb"].reshape(B, n_nodes, T, -1).transpose(1, 2)  # (B,T,N,d)
            X.append(z.reshape(-1, z.shape[-1]).cpu())
            Y.append(st.reshape(-1, F).cpu())
    X, Y = torch.cat(X), torch.cat(Y)
    X1 = torch.cat([X, torch.ones(X.shape[0], 1)], 1)
    W, *_ = torch.linalg.lstsq(X1, Y)
    mse = torch.nn.functional.mse_loss(X1 @ W, Y).item()
    print(f"  [probe] {X.shape[0]} pairs, train MSE={mse:.3f}")
    return W


def apply_probe(W, emb):
    ones = torch.ones(*emb.shape[:-1], 1, device=emb.device, dtype=emb.dtype)
    return torch.cat([emb, ones], -1) @ W.to(emb.device)


def node_rollout(model, s_hist, a_hist, a_future, n_nodes, nbr_idx, nbr_mask, HS, device,
                 edge_hist=None):
    """s_hist/a_hist: (H, N*F)/(H, N*A). a_future: (n_steps, N*A). -> (n_steps, N, d).
    edge_hist: (H, N*deg*EF) context edge features; held STALE for every rolled step
    (we have no future edge observations at inference - the CoDreamer stale-graph
    approximation, documented in D2)."""
    from traffic.multi_agent import edge_message_pool
    nbr_idx, nbr_mask = nbr_idx.to(device), nbr_mask.to(device)
    edge_stale = None
    if model.edge_dim > 0 and edge_hist is not None:
        deg = nbr_idx.shape[1]
        eh = edge_hist.to(device).view(HS, n_nodes, deg, model.edge_dim)[-HS:]      # (HS,N,deg,EF)
        edge_stale = eh.permute(1, 2, 0, 3).unsqueeze(0)                            # (1,N,deg,HS,EF)
    with torch.no_grad():
        out = encode_batch(model, s_hist.unsqueeze(0), a_hist.unsqueeze(0),
                           n_nodes, nbr_idx, nbr_mask, device,
                           edge_feat=edge_hist.unsqueeze(0) if edge_hist is not None else None)
        H = s_hist.shape[0]
        z = out["emb"].reshape(1, n_nodes, H, -1).transpose(1, 2)      # (1,H,N,d)
        act = out["act_emb"].reshape(1, n_nodes, H, -1).transpose(1, 2)
        preds = []
        for t in range(a_future.shape[0]):
            z_ctx, a_ctx = z[:, -HS:], act[:, -HS:]
            if model.level == "0.5":
                z_bn, a_bn = z_ctx.transpose(1, 2), a_ctx.transpose(1, 2)  # (1,N,HS,·)
                if model.edge_dim > 0:
                    z_pool = edge_message_pool(z_bn, edge_stale, nbr_idx, nbr_mask, model.msg_mlp)
                elif model.neighbor_agg == "pna":
                    z_pool = model.nbr_proj(masked_neighbor_pna(z_bn, nbr_idx, nbr_mask))
                else:
                    z_pool = masked_neighbor_mean(z_bn, nbr_idx, nbr_mask)
                a_pool = masked_neighbor_mean(a_bn, nbr_idx, nbr_mask)
                if model.permute_control:
                    perm = model._fixed_derangement(n_nodes, device)
                    z_pool, a_pool = z_pool[:, perm], a_pool[:, perm]
                pin_e = torch.cat([z_bn, z_pool], -1).reshape(n_nodes, HS, -1)
                pin_a = torch.cat([a_bn, a_pool], -1).reshape(n_nodes, HS, -1)
            else:
                pin_e = z_ctx.transpose(1, 2).reshape(n_nodes, HS, -1)
                pin_a = a_ctx.transpose(1, 2).reshape(n_nodes, HS, -1)
            nxt = model.predict(pin_e, pin_a)[:, -1:]           # (N,1,d)
            nxt = nxt.reshape(1, 1, n_nodes, -1)
            preds.append(nxt)
            z = torch.cat([z, nxt], 1)
            na = a_future[t:t + 1].unsqueeze(0).to(device)
            A = na.shape[-1] // n_nodes
            na_emb = model.action_encoder(na.view(1, 1, n_nodes, A).reshape(n_nodes, 1, A))
            act = torch.cat([act, na_emb.reshape(1, 1, n_nodes, -1)], 1)
        return torch.cat(preds, 1).squeeze(0)  # (n_steps, N, d)


def eval_rollout(model, W, val_path, n_nodes, P_max, nbr_idx, nbr_mask, HS, n_steps, device):
    data = torch.load(val_path, weights_only=False)
    F = data["node_feature_dim"]
    me_full = np.zeros(n_steps); pe_full = np.zeros(n_steps)
    me_pr = np.zeros(n_steps); pe_pr = np.zeros(n_steps)
    cnt = 0
    for ep in data["episodes"]:
        T = ep["state"].shape[0]
        span = T - HS - n_steps
        if span <= 0:
            continue
        for start in range(0, span, max(1, span // 6)):
            s_hist = torch.from_numpy(ep["state"][start:start + HS])
            a_hist = torch.from_numpy(ep["action"][start:start + HS])
            a_fut = torch.from_numpy(ep["action"][start + HS:start + HS + n_steps])
            true = torch.from_numpy(ep["state"][start + HS:start + HS + n_steps]).to(device).view(n_steps, n_nodes, F)
            eh = torch.from_numpy(ep["edge_feat"][start:start + HS]) if "edge_feat" in ep else None
            pe = node_rollout(model, s_hist, a_hist, a_fut, n_nodes, nbr_idx, nbr_mask, HS, device, edge_hist=eh)
            dec = apply_probe(W, pe)                         # (n_steps, N, F)
            per = s_hist[-1].to(device).view(1, n_nodes, F).expand_as(true)
            me_full += ((dec - true) ** 2).mean((1, 2)).cpu().numpy()
            pe_full += ((per - true) ** 2).mean((1, 2)).cpu().numpy()
            me_pr += ((dec[..., :P_max].sum(-1) - true[..., :P_max].sum(-1)) ** 2).mean(1).cpu().numpy()
            pe_pr += ((per[..., :P_max].sum(-1) - true[..., :P_max].sum(-1)) ** 2).mean(1).cpu().numpy()
            cnt += 1
    for a in (me_full, pe_full, me_pr, pe_pr):
        a /= max(cnt, 1)
    print(f"  rollout ({cnt} windows)   h: full-feat MSE model/persist   |   total-halting SE model/persist")
    for h in range(n_steps):
        b1 = "y" if me_full[h] < pe_full[h] else "N"
        b2 = "y" if me_pr[h] < pe_pr[h] else "N"
        print(f"    h{h+1:<2} {me_full[h]:8.3f} / {pe_full[h]:8.3f} [{b1}]   |   {me_pr[h]:9.1f} / {pe_pr[h]:9.1f} [{b2}]")
    return dict(me_full=me_full, pe_full=pe_full, me_pr=me_pr, pe_pr=pe_pr)


def eval_cf(model, W, cf_path, n_nodes, P_max, nbr_idx, nbr_mask, HS, device):
    data = torch.load(cf_path, weights_only=False)
    F = data["node_feature_dim"]
    samples = data["samples"]
    pred_d, true_d = [], []
    top1, sp = [], []
    for smp in samples:
        a_state = torch.from_numpy(smp["anchor_state"])
        a_act = torch.from_numpy(smp["anchor_action"])
        s_hist = a_state.unsqueeze(0).expand(HS, -1).contiguous()
        a_hist = a_act.unsqueeze(0).expand(HS, -1).contiguous()
        embs, costs_pred, costs_true = [], [], []
        for br in smp["branches"]:
            H = br["states"].shape[0] - 1
            a_fut = torch.from_numpy(br["action"]).unsqueeze(0).expand(H, -1).contiguous()
            eh = (torch.from_numpy(smp["anchor_edge"]).unsqueeze(0).expand(HS, -1).contiguous()
                  if "anchor_edge" in smp else None)
            pe = node_rollout(model, s_hist, a_hist, a_fut, n_nodes, nbr_idx, nbr_mask, HS, device, edge_hist=eh)  # (H,N,d)
            dec = apply_probe(W, pe).view(H, n_nodes, F)
            embs.append(pe[-1].reshape(-1))
            costs_pred.append(dec[..., :P_max].clamp(min=0).sum().item())
            tr = torch.from_numpy(br["states"][1:]).view(H, n_nodes, F)
            costs_true.append(tr[..., :P_max].sum().item())
        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                pred_d.append((embs[i] - embs[j]).norm().item())
                si = torch.from_numpy(smp["branches"][i]["states"][-1])
                sj = torch.from_numpy(smp["branches"][j]["states"][-1])
                true_d.append((si - sj).norm().item())
        cp, ct = np.array(costs_pred), np.array(costs_true)
        top1.append(int(np.argmin(cp) == np.argmin(ct)))
        r, _ = spearmanr(cp, ct)
        sp.append(r)
    corr = np.corrcoef(pred_d, true_d)[0, 1]
    print(f"  counterfactual: corr(||Δemb||,||Δstate||)={corr:.3f}  "
          f"(mean true Δ {np.mean(true_d):.1f}, pred Δ {np.mean(pred_d):.2f})")
    print(f"  plan-ranking:  top-1 argmin match {np.mean(top1):.3f} "
          f"(chance {1/len(samples[0]['branches']):.3f})   mean Spearman {np.nanmean(sp):.3f}")
    return dict(cf_corr=corr, top1=float(np.mean(top1)), spearman=float(np.nanmean(sp)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="traffic_data_cologne8")
    p.add_argument("--runs", nargs="+", default=["L0", "L05", "L05perm"])
    p.add_argument("--weights", default="weights_epoch_80.pt")
    p.add_argument("--history", type=int, default=3)
    p.add_argument("--rollout_steps", type=int, default=10)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dd = Path(args.data_dir)
    meta = torch.load(dd / "val.pt", weights_only=False)
    n_nodes, P_max = meta["n_nodes"], meta["P_max"]
    nbr_idx = torch.from_numpy(np.asarray(meta["neighbor_idx"]))
    nbr_mask = torch.from_numpy(np.asarray(meta["neighbor_mask"]))
    HS = args.history
    window = HS + 1

    results = {}
    for run in args.runs:
        print(f"\n===== {run} =====")
        model, cfg = load(run, args.weights, device)
        W = fit_probe(model, dd / "train.pt", window, n_nodes, nbr_idx, nbr_mask, device)
        r1 = eval_rollout(model, W, dd / "val.pt", n_nodes, P_max, nbr_idx, nbr_mask,
                          HS, args.rollout_steps, device)
        r2 = eval_cf(model, W, dd / "counterfactual.pt", n_nodes, P_max, nbr_idx, nbr_mask, HS, device)
        results[run] = {**r1, **r2}

    print("\n===== summary =====")
    print(f"{'run':<10} {'roll h5 full':>12} {'persist h5':>11} {'roll h5 halt':>13} {'persist':>9} "
          f"{'cf corr':>8} {'top1':>6} {'spearman':>9}")
    for run, r in results.items():
        print(f"{run:<10} {r['me_full'][4]:>12.3f} {r['pe_full'][4]:>11.3f} "
              f"{r['me_pr'][4]:>13.1f} {r['pe_pr'][4]:>9.1f} "
              f"{r['cf_corr']:>8.3f} {r['top1']:>6.3f} {r['spearman']:>9.3f}")


if __name__ == "__main__":
    main()
