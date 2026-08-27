"""train_multi_sumo.py + a short autoregressive rollout loss.

The base trainer only ever asks the predictor for one step from real context
(num_preds=1). LeWM / V-JEPA 2-AC both add a short teacher-forced-then-unrolled
loss so the model is trained on the distribution of its *own* predictions - the
fix for the open-loop compounding this project's 2026-08-27 cologne8 run showed.

This trainer keeps the 1-step term and adds:  sum_{k=1..K}  w_k * || z_hat_k - z_true_k ||^2
where z_hat_k is produced by feeding the predictor its own previous output
(pooling neighbours from the rolled-forward embeddings each step, exactly as
inference does). w_k = 1/k (nearer steps matter more).

Usage:
    python train_multi_sumo_ar.py --level 0.5 --tag L05ar --rollout_k 4
    python train_multi_sumo_ar.py --level 0   --tag L0ar  --rollout_k 4
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from einops import rearrange

from module import MLP, ARPredictor, Embedder, SIGReg
from traffic.dataset import TrafficDataset
from traffic.multi_agent import MultiAgentJEPA, masked_neighbor_mean, masked_neighbor_pna

EMBED_DIM = 64
HISTORY = 3
SIGREG_W = 0.09


def build_model(node_F, node_A, level, permute_control, neighbor_agg="mean"):
    pred_in = EMBED_DIM if str(level) == "0" else 2 * EMBED_DIM
    bn = torch.nn.BatchNorm1d
    return MultiAgentJEPA(
        encoder=Embedder(input_dim=node_F, smoothed_dim=EMBED_DIM, emb_dim=EMBED_DIM),
        predictor=ARPredictor(num_frames=HISTORY, depth=4, heads=4, mlp_dim=256,
                              input_dim=pred_in, hidden_dim=EMBED_DIM, output_dim=EMBED_DIM,
                              dim_head=32, dropout=0.1, emb_dropout=0.0),
        action_encoder=Embedder(input_dim=node_A, smoothed_dim=EMBED_DIM, emb_dim=EMBED_DIM),
        level=level, permute_control=permute_control,
        neighbor_agg=neighbor_agg, emb_dim=EMBED_DIM,
        projector=MLP(EMBED_DIM, 256, EMBED_DIM, norm_fn=bn),
        pred_proj=MLP(EMBED_DIM, 256, EMBED_DIM, norm_fn=bn),
    )


def ar_forward(model, sigreg, batch, n_nodes, ni, nm, device, K):
    """window = HISTORY + K. 1-step teacher-forced loss + K-step AR rollout loss."""
    state = batch["state"].to(device).float()
    action = batch["action"].to(device).float()
    B, W = state.shape[:2]
    F = state.shape[-1] // n_nodes
    A = action.shape[-1] // n_nodes
    lvl05 = model.level == "0.5"

    info = {"state": state, "action": action, "n_nodes": n_nodes}
    if lvl05:
        info["neighbor_idx"] = ni.to(device); info["neighbor_mask"] = nm.to(device)
    out = model.encode(info)
    emb = out["emb"]                                   # (B*N, W, d) own
    act_emb = out["act_emb"]                           # (B*N, W, a)
    d = emb.shape[-1]
    emb_bn = rearrange(emb, "(b n) t d -> b n t d", b=B)          # (B,N,W,d)
    act_bn = rearrange(act_emb, "(b n) t d -> b n t d", b=B)

    ni_d = ni.to(device); nm_d = nm.to(device)

    def pred_in(z_bn, a_bn):
        """z_bn/a_bn: (B,N,ctx,·) -> predictor inputs (B*N,ctx,·)."""
        if not lvl05:
            return (rearrange(z_bn, "b n t d -> (b n) t d"),
                    rearrange(a_bn, "b n t d -> (b n) t d"))
        if model.neighbor_agg == "pna":
            zp = model.nbr_proj(masked_neighbor_pna(z_bn, ni_d, nm_d))
        else:
            zp = masked_neighbor_mean(z_bn, ni_d, nm_d)
        ap = masked_neighbor_mean(a_bn, ni_d, nm_d)
        if model.permute_control:
            perm = model._fixed_derangement(n_nodes, device)
            zp, ap = zp[:, perm], ap[:, perm]
        return (rearrange(torch.cat([z_bn, zp], -1), "b n t d -> (b n) t d"),
                rearrange(torch.cat([a_bn, ap], -1), "b n t d -> (b n) t d"))

    # ---- 1-step teacher-forced: predict step HISTORY from real steps [0,HISTORY) ----
    pe, pa = pred_in(emb_bn[:, :, :HISTORY], act_bn[:, :, :HISTORY])
    tf_pred = model.predict(pe, pa)[:, -1]                        # (B*N, d) ~ emb[:,HISTORY]
    tf_loss = (tf_pred - emb[:, HISTORY]).pow(2).mean()

    # ---- K-step autoregressive rollout ----
    z_roll = emb_bn[:, :, :HISTORY].clone()                       # (B,N,HISTORY,d) own
    a_roll = act_bn[:, :, :HISTORY].clone()
    roll_loss = 0.0
    wsum = 0.0
    for k in range(1, K + 1):
        pe, pa = pred_in(z_roll[:, :, -HISTORY:], a_roll[:, :, -HISTORY:])
        nxt = model.predict(pe, pa)[:, -1]                        # (B*N, d)
        nxt_bn = rearrange(nxt, "(b n) d -> b n 1 d", b=B)
        w = 1.0 / k
        roll_loss = roll_loss + w * (nxt - emb[:, HISTORY + k - 1]).pow(2).mean()
        wsum += w
        z_roll = torch.cat([z_roll, nxt_bn], dim=2)
        # true action for the next step (teacher-forced actions, as at inference)
        a_next = act_bn[:, :, HISTORY + k - 1: HISTORY + k]
        a_roll = torch.cat([a_roll, a_next], dim=2)
    roll_loss = roll_loss / wsum

    sig_loss = sigreg(emb.transpose(0, 1))
    loss = tf_loss + roll_loss + SIGREG_W * sig_loss
    rl_log = roll_loss.detach().item() if torch.is_tensor(roll_loss) else roll_loss
    return loss, tf_loss.item(), rl_log, sig_loss.item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="traffic_data_cologne8")
    p.add_argument("--level", default="0.5")
    p.add_argument("--permute_control", action="store_true")
    p.add_argument("--tag", default="L05ar")
    p.add_argument("--rollout_k", type=int, default=4)
    p.add_argument("--neighbor_agg", default="mean", choices=["mean", "pna"])
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=3072)
    p.add_argument("--save_every", type=int, default=40)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dd = Path(args.data_dir)
    meta = torch.load(dd / "train.pt", weights_only=False)
    n_nodes, node_F, node_A = meta["n_nodes"], meta["node_feature_dim"], meta["node_action_dim"]
    ni = torch.from_numpy(np.asarray(meta["neighbor_idx"]))
    nm = torch.from_numpy(np.asarray(meta["neighbor_mask"]))
    K = args.rollout_k
    window = HISTORY + K
    print(f"cologne8 N={n_nodes} F={node_F} A={node_A}  level={args.level} "
          f"permute={args.permute_control}  agg={args.neighbor_agg}  rollout_k={K}  window={window}")

    tr_set = TrafficDataset(dd / "train.pt", window=window)
    va_set = TrafficDataset(dd / "val.pt", window=window)
    tr = torch.utils.data.DataLoader(tr_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    va = torch.utils.data.DataLoader(va_set, batch_size=args.batch_size, shuffle=False)
    print(f"train windows {len(tr_set)}  val windows {len(va_set)}")

    model = build_model(node_F, node_A, args.level, args.permute_control, args.neighbor_agg).to(device)
    print(f"params {sum(x.numel() for x in model.parameters())/1e3:.0f}K")
    sigreg = SIGReg(knots=17, num_proj=1024).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    run_dir = Path("traffic_runs_sumo", args.tag)
    run_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(1, args.epochs + 1):
        model.train(); t0 = time.time(); acc = []
        for b in tr:
            opt.zero_grad()
            loss, tf, rl, sl = ar_forward(model, sigreg, b, n_nodes, ni, nm, device, K)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            acc.append((loss.item(), tf, rl, sl))
        sched.step()
        model.eval(); vacc = []
        with torch.no_grad():
            for b in va:
                loss, tf, rl, sl = ar_forward(model, sigreg, b, n_nodes, ni, nm, device, K)
                vacc.append((loss.item(), tf, rl, sl))
        acc, vacc = np.array(acc), np.array(vacc)
        if ep == 1 or ep % 10 == 0 or ep == args.epochs:
            print(f"ep {ep:>3} | train {acc[:,0].mean():.4f} (tf {acc[:,1].mean():.4f} roll {acc[:,2].mean():.4f}) "
                  f"| val (tf {vacc[:,1].mean():.4f} roll {vacc[:,2].mean():.4f}) | {time.time()-t0:.1f}s")
        if ep % args.save_every == 0 or ep == args.epochs:
            torch.save({"model_state": model.state_dict(),
                        "cfg": {"node_F": node_F, "node_A": node_A, "level": args.level,
                                "permute_control": args.permute_control, "embed_dim": EMBED_DIM,
                                "history": HISTORY, "rollout_k": K, "neighbor_agg": args.neighbor_agg},
                        "epoch": ep}, run_dir / f"weights_epoch_{ep}.pt")
    print(f"done -> {run_dir}")


if __name__ == "__main__":
    main()
