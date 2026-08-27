"""Standalone trainer for the topology-free multi-agent JEPA on real multi-signal
SUMO data (cologne8). No hydra / stable-pretraining / stable-worldmodel: plain
PyTorch, reusing the real model code (traffic.multi_agent.MultiAgentJEPA +
module.*) and the same 2-term LeWM objective (next-emb MSE + SIGReg) and
lejepa_forward_multi wiring as train_traffic_multi.py.

Neighbour table comes from the dataset .pt (built by generate_sumo_multi from the
real network), not from CTMGridEnv - that is the whole point of "topology-free".

Usage:
    python train_multi_sumo.py --data_dir traffic_data_cologne8 --level 0   --tag L0
    python train_multi_sumo.py --data_dir traffic_data_cologne8 --level 0.5 --tag L05
    python train_multi_sumo.py --data_dir traffic_data_cologne8 --level 0.5 --permute_control --tag L05perm
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from module import MLP, ARPredictor, Embedder, SIGReg
from traffic.dataset import TrafficDataset
from traffic.multi_agent import MultiAgentJEPA

EMBED_DIM = 64
HISTORY = 3
NUM_PREDS = 1
SIGREG_W = 0.09


def build_model(node_F, node_A, level, permute_control):
    pred_in = EMBED_DIM if str(level) == "0" else 2 * EMBED_DIM
    bn = torch.nn.BatchNorm1d
    return MultiAgentJEPA(
        encoder=Embedder(input_dim=node_F, smoothed_dim=EMBED_DIM, emb_dim=EMBED_DIM),
        predictor=ARPredictor(
            num_frames=HISTORY, depth=4, heads=4, mlp_dim=256,
            input_dim=pred_in, hidden_dim=EMBED_DIM, output_dim=EMBED_DIM,
            dim_head=32, dropout=0.1, emb_dropout=0.0,
        ),
        action_encoder=Embedder(input_dim=node_A, smoothed_dim=EMBED_DIM, emb_dim=EMBED_DIM),
        level=level,
        permute_control=permute_control,
        projector=MLP(EMBED_DIM, 256, EMBED_DIM, norm_fn=bn),
        pred_proj=MLP(EMBED_DIM, 256, EMBED_DIM, norm_fn=bn),
    )


def forward_loss(model, sigreg, batch, n_nodes, nbr_idx, nbr_mask, device):
    info = {
        "state": batch["state"].to(device),
        "action": batch["action"].to(device),
        "n_nodes": n_nodes,
    }
    if model.level == "0.5":
        info["neighbor_idx"] = nbr_idx.to(device)
        info["neighbor_mask"] = nbr_mask.to(device)
    out = model.encode(info)
    emb = out["emb"]                                   # (B*N, W, d)  -- target + sigreg
    pred_in_emb = out["pred_in_emb"][:, :HISTORY]
    pred_in_act = out["pred_in_act_emb"][:, :HISTORY]
    tgt = emb[:, NUM_PREDS:]
    pred = model.predict(pred_in_emb, pred_in_act)
    pred_loss = (pred - tgt).pow(2).mean()
    sig_loss = sigreg(emb.transpose(0, 1))
    return pred_loss + SIGREG_W * sig_loss, pred_loss.item(), sig_loss.item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="traffic_data_cologne8")
    p.add_argument("--level", default="0")
    p.add_argument("--permute_control", action="store_true")
    p.add_argument("--tag", default="L0")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=3072)
    p.add_argument("--save_every", type=int, default=20)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dd = Path(args.data_dir)
    meta = torch.load(dd / "train.pt", weights_only=False)
    n_nodes = meta["n_nodes"]
    node_F = meta["node_feature_dim"]
    node_A = meta["node_action_dim"]
    nbr_idx = torch.from_numpy(np.asarray(meta["neighbor_idx"]))
    nbr_mask = torch.from_numpy(np.asarray(meta["neighbor_mask"]))
    print(f"cologne8: N={n_nodes} F={node_F} A={node_A} "
          f"deg={nbr_mask.sum(1).tolist()}  level={args.level} permute={args.permute_control}")

    window = HISTORY + NUM_PREDS
    train_set = TrafficDataset(dd / "train.pt", window=window)
    val_set = TrafficDataset(dd / "val.pt", window=window)
    train = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                        shuffle=True, drop_last=True)
    val = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    print(f"train windows: {len(train_set)}  val windows: {len(val_set)}")

    model = build_model(node_F, node_A, args.level, args.permute_control).to(device)
    n_params = sum(x.numel() for x in model.parameters())
    print(f"model params: {n_params/1e3:.0f}K")
    sigreg = SIGReg(knots=17, num_proj=1024).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    run_dir = Path("traffic_runs_sumo", args.tag)
    run_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        tr = []
        for batch in train:
            opt.zero_grad()
            loss, pl, sl = forward_loss(model, sigreg, batch, n_nodes, nbr_idx, nbr_mask, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr.append((loss.item(), pl, sl))
        sched.step()

        model.eval()
        va = []
        with torch.no_grad():
            for batch in val:
                loss, pl, sl = forward_loss(model, sigreg, batch, n_nodes, nbr_idx, nbr_mask, device)
                va.append((loss.item(), pl, sl))
        tr, va = np.array(tr), np.array(va)
        # latent health: per-dim std of a val batch's embeddings
        with torch.no_grad():
            b = next(iter(val))
            info = {"state": b["state"].to(device), "action": b["action"].to(device), "n_nodes": n_nodes}
            if model.level == "0.5":
                info["neighbor_idx"] = nbr_idx.to(device); info["neighbor_mask"] = nbr_mask.to(device)
            z = model.encode(info)["emb"]
            std = z.reshape(-1, z.shape[-1]).std(0).mean().item()
        if ep == 1 or ep % 5 == 0 or ep == args.epochs:
            print(f"ep {ep:>3} | train {tr[:,0].mean():.4f} (pred {tr[:,1].mean():.4f} sig {tr[:,2].mean():.3f}) "
                  f"| val {va[:,0].mean():.4f} (pred {va[:,1].mean():.4f}) | z-std {std:.3f} "
                  f"| {time.time()-t0:.1f}s")

        if ep % args.save_every == 0 or ep == args.epochs:
            torch.save({
                "model_state": model.state_dict(),
                "cfg": {"node_F": node_F, "node_A": node_A, "level": args.level,
                        "permute_control": args.permute_control, "embed_dim": EMBED_DIM,
                        "history": HISTORY},
                "epoch": ep,
            }, run_dir / f"weights_epoch_{ep}.pt")

    print(f"done -> {run_dir}")


if __name__ == "__main__":
    main()
