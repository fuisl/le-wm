"""Generate the data behind the multi-node decoded-rollout visualization: for
one held-out CTM-grid episode, compute ground truth, teacher-forced probe
decode, and imagined-rollout probe decode, per selected node, and dump to
JSON for the HTML artifact to render.

Mirrors visualize_rollout.py's cologne1 machinery exactly, generalized to
the node-structured multi-agent path (traffic/multi_agent.py). Also dumps
the rollout-error-vs-horizon comparison across Level 0 / 0.5 / 0.5+permute,
computed the same way as eval_traffic_multi.py.

Usage: python visualize_rollout_multi.py
"""

import json
from pathlib import Path

import numpy as np
import torch

from eval_traffic import apply_probe, load_model
from eval_traffic_multi import eval_rollout_vs_horizon, fit_node_probe, node_rollout
from traffic.ctm_env import CTMGridEnv

DATA_DIR = "traffic_data_4x4"
ROWS, COLS = 4, 4
HISTORY_SIZE = 3
HORIZON = 20
EPISODE_IDX = 3
WINDOW_START = 20
RUNS = {
    "level0": "cmp4x4-level0",
    "level05": "cmp4x4-level05",
    "level05_permute": "cmp4x4-level05-permute",
}
WEIGHTS = "weights_epoch_30.pt"

# corner (degree 2), edge (degree 3), interior (degree 4) in a 4x4 grid
SELECTED_NODES = {"corner (0,0), degree 2": 0, "edge (0,1), degree 3": 1, "interior (1,1), degree 4": 5}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_nodes = ROWS * COLS
    neighbor_idx_np, neighbor_mask_np = CTMGridEnv(rows=ROWS, cols=COLS).neighbor_table()
    neighbor_idx = torch.from_numpy(neighbor_idx_np)
    neighbor_mask = torch.from_numpy(neighbor_mask_np)

    train_path = f"{DATA_DIR}/train.pt"
    val_path = f"{DATA_DIR}/val.pt"
    window = HISTORY_SIZE + 1

    # ---- headline chart: rollout error vs horizon, all three arms ----
    horizon_curves = {}
    for key, run_name in RUNS.items():
        model = load_model(run_name, WEIGHTS, device)
        W = fit_node_probe(model, train_path, window, n_nodes, neighbor_idx, neighbor_mask, device)
        probe_err, persist_err = eval_rollout_vs_horizon(
            model, W, val_path, ROWS, COLS, HISTORY_SIZE, 10, 40, device
        )
        horizon_curves[key] = probe_err.tolist()
    horizon_curves["persistence"] = persist_err.tolist()  # same for all arms, same episodes

    # ---- imagined rollout figure: Level 0.5 (real neighbors), one window, 3 nodes ----
    model = load_model(RUNS["level05"], WEIGHTS, device)
    W = fit_node_probe(model, train_path, window, n_nodes, neighbor_idx, neighbor_mask, device)

    data = torch.load(val_path, weights_only=False)
    ep = data["episodes"][EPISODE_IDX]
    controller = ep["controller"]

    total_len = HISTORY_SIZE + HORIZON
    s0, s1 = WINDOW_START, WINDOW_START + total_len
    states = ep["state"][s0:s1]    # (total_len, n*4)
    actions = ep["action"][s0:s1]  # (total_len, n*2)
    assert states.shape[0] == total_len, "window runs past episode end"

    with torch.no_grad():
        info = {
            "state": torch.from_numpy(states).unsqueeze(0).to(device),
            "action": torch.from_numpy(actions).unsqueeze(0).to(device),
            "n_nodes": n_nodes,
            "neighbor_idx": neighbor_idx.to(device),
            "neighbor_mask": neighbor_mask.to(device),
        }
        out = model.encode(info)
        own_emb = out["emb"].reshape(1, n_nodes, total_len, -1).transpose(1, 2)[0]  # (T, N, d)
        teacher_forced = apply_probe(W, own_emb).cpu().numpy()  # (T, N, 4)

    state_hist = torch.from_numpy(states[:HISTORY_SIZE])
    action_hist = torch.from_numpy(actions[:HISTORY_SIZE])
    action_future = torch.from_numpy(actions[HISTORY_SIZE:])
    pred_emb = node_rollout(
        model, state_hist, action_hist, action_future, n_nodes,
        neighbor_idx, neighbor_mask, HISTORY_SIZE, device,
    )  # (HORIZON, N, d)
    imagined = apply_probe(W, pred_emb).detach().cpu().numpy()  # (HORIZON, N, 4)

    states_n = states.reshape(total_len, n_nodes, 4)
    phase_idx_per_step = [int(np.argmax(a)) for a in actions.reshape(total_len, n_nodes, 2)[:, 0]]
    # phase is per-node; recompute per node below instead of using node 0 globally
    phases_n = np.argmax(actions.reshape(total_len, n_nodes, 2), axis=-1)  # (T, N)

    node_frames = {}
    for label, node_id in SELECTED_NODES.items():
        frames = []
        for t in range(total_len):
            is_context = t < HISTORY_SIZE
            is_imagined = t >= HISTORY_SIZE
            phase = int(phases_n[t, node_id])
            green = [phase == 0, phase == 0, phase == 1, phase == 1]  # N,S,E,W per DIRS order
            frames.append({
                "t": t,
                "region": "context" if is_context else "imagined",
                "phase": phase,
                "green": green,
                "ground_truth": states_n[t, node_id].tolist(),
                "teacher_forced": teacher_forced[t, node_id].tolist(),
                "imagined": imagined[t - HISTORY_SIZE, node_id].tolist() if is_imagined else None,
            })
        node_frames[label] = frames

    out = {
        "meta": {
            "scenario": f"synthetic CTM grid, {ROWS}x{COLS} ({n_nodes} nodes)",
            "controller": controller,
            "episode_idx": EPISODE_IDX,
            "window_start_step": WINDOW_START,
            "history_size": HISTORY_SIZE,
            "horizon": HORIZON,
            "checkpoint": f"{RUNS['level05']}/{WEIGHTS}",
            "queue_labels": ["N", "S", "E", "W"],
        },
        "horizon_curves": horizon_curves,
        "node_frames": node_frames,
    }

    out_path = Path("traffic_data_4x4/rollout_viz_multi.json")
    out_path.write_text(json.dumps(out))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
