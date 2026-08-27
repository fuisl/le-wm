"""Acceptance test for the fixed-F-per-node prerequisite (M1).

The claim under test: a MultiAgentJEPA built once, for one grid size, runs
forward on a *different* grid size with no config change and no shape
error - because encoder/action_encoder/predictor input dims are set from
the fixed per-node feature width F and action width A, never from n.

This is the actual pass/fail for "scale it up, not scenario-specific" as a
prerequisite; everything else (Level 0 vs 0.5 comparison) is downstream of
this passing. Run: python -m traffic.test_shape_invariance
"""

import numpy as np
import torch

from module import ARPredictor, Embedder
from traffic.ctm_env import CTMGridEnv
from traffic.multi_agent import MultiAgentJEPA

EMBED_DIM = 16
HISTORY = 3


def build_model(level):
    F, A = 4, 2  # fixed per-node widths (CTMGridEnv.node_feature_dim() / node_action width)
    pred_in_dim = EMBED_DIM if level == "0" else 2 * EMBED_DIM
    encoder = Embedder(input_dim=F, smoothed_dim=EMBED_DIM, emb_dim=EMBED_DIM)
    action_encoder = Embedder(input_dim=A, smoothed_dim=EMBED_DIM, emb_dim=EMBED_DIM)
    predictor = ARPredictor(
        num_frames=HISTORY, depth=2, heads=2, mlp_dim=64,
        input_dim=pred_in_dim, hidden_dim=EMBED_DIM, output_dim=EMBED_DIM, dim_head=16,
    )
    return MultiAgentJEPA(encoder, predictor, action_encoder, level=level)


def make_batch(rows, cols, batch_size=5, T=HISTORY):
    env = CTMGridEnv(rows=rows, cols=cols, seed=0)
    env.reset()
    n = env.n
    neighbor_idx, neighbor_mask = env.neighbor_table()

    states, actions = [], []
    for _ in range(batch_size):
        env.reset()
        s, a = [], []
        for t in range(T):
            phases = env.rng.integers(0, 2, size=n)
            s.append(env.node_features().reshape(-1))
            a.append(env.node_action(phases).reshape(-1))
            env.step(phases)
        states.append(np.stack(s))
        actions.append(np.stack(a))

    info = {
        "state": torch.from_numpy(np.stack(states)).float(),   # (B, T, n*F)
        "action": torch.from_numpy(np.stack(actions)).float(), # (B, T, n*A)
        "n_nodes": n,
        "neighbor_idx": torch.from_numpy(neighbor_idx),
        "neighbor_mask": torch.from_numpy(neighbor_mask),
    }
    return info, n


def check(level):
    print(f"--- level {level} ---")
    model = build_model(level)

    info_small, n_small = make_batch(rows=2, cols=2)
    out_small = model.encode(info_small)
    pred_small = model.predict(out_small["pred_in_emb"], out_small["pred_in_act_emb"])
    assert pred_small.shape == (5 * n_small, HISTORY, EMBED_DIM), pred_small.shape
    print(f"  rows=2,cols=2 (n={n_small}): pred {tuple(pred_small.shape)} OK")

    # same model instance, no config change, larger and non-square grid
    info_big, n_big = make_batch(rows=4, cols=3)
    out_big = model.encode(info_big)
    pred_big = model.predict(out_big["pred_in_emb"], out_big["pred_in_act_emb"])
    assert pred_big.shape == (5 * n_big, HISTORY, EMBED_DIM), pred_big.shape
    print(f"  rows=4,cols=3 (n={n_big}): pred {tuple(pred_big.shape)} OK")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  shared weights: {n_params} params, used at both n={n_small} and n={n_big}")


def check_boundary_pooling_no_degree_leak():
    """Regression guard for the zero-pad-then-/4 trap: an interior node (degree
    4) and a corner node (degree 2) pooling the same neighbor values must get
    the same pooled value, not a value scaled down by missing neighbors."""
    from traffic.multi_agent import masked_neighbor_mean

    rows, cols = 3, 3
    env = CTMGridEnv(rows=rows, cols=cols, seed=0)
    neighbor_idx, neighbor_mask = env.neighbor_table()
    n = env.n

    corner = env.idx(0, 0)   # degree 2
    interior = env.idx(1, 1) # degree 4
    assert neighbor_mask[corner].sum() == 2
    assert neighbor_mask[interior].sum() == 4

    x = torch.full((1, n, 1, 1), 3.0)  # every node's feature is the constant 3.0
    pooled = masked_neighbor_mean(x, torch.from_numpy(neighbor_idx), torch.from_numpy(neighbor_mask))
    assert torch.allclose(pooled[0, corner], torch.tensor([3.0]))
    assert torch.allclose(pooled[0, interior], torch.tensor([3.0]))
    print("--- boundary pooling ---")
    print(f"  corner (degree 2) pooled == interior (degree 4) pooled == 3.0: OK")


if __name__ == "__main__":
    check("0")
    check("0.5")
    check_boundary_pooling_no_degree_leak()
    print("\nPASS: fixed-F per-node prerequisite holds - shared weights run on unseen grid sizes.")
