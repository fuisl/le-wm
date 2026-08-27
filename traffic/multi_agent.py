"""Shape-agnostic, topology-generalizing wrapper around jepa.JEPA.

Prerequisite this answers: cologne1's encoder input_dim is hard-set from one
scenario's flattened state (sumo_env.py state_dim() = len(self.lanes);
train_traffic.py sets cfg.model.encoder.input_dim = train_set.state_dim at
runtime). That model cannot run on an intersection with a different lane
count, independent of any GNN question.

Fix: the encoder/predictor/action_encoder (module.py, unchanged) already
operate on a flat (B, T, D) tensor with shared weights per timestep - they
have no opinion about what B is. Folding the node axis into the batch axis
gets a per-node, node-count-agnostic model for free (D4 Variant A / "Level
0 - no coupling"). Level 0.5 adds a masked mean over each node's actual
graph neighbors (Deep-Sets-style pooling, no adjacency-restricted attention,
no edge features) before the fold, per D4's boxed predictor equation - which
includes neighbors' actions {a^j_t} as well as their embeddings {z^j_t},
because in CTMGridEnv neighbor j's phase choice at t (not just its queue
state) gates whether j discharges into i this step.

jepa.py and module.py are shared with the pixel/robot upstream code and are
deliberately left untouched; this file only adds a node-structured encode()
on top of the existing submodules.
"""

import torch
from einops import rearrange

from jepa import JEPA


def masked_neighbor_mean(x, neighbor_idx, neighbor_mask):
    """Mean-pool x over each node's actual neighbors (not zero-padded / deg).

    x: (B, N, T, D)
    neighbor_idx: (N, deg) int64, -1 where no neighbor exists
    neighbor_mask: (N, deg) bool, True where that slot is a real neighbor
    returns: (B, N, T, D) - pooled[b, i, t] = mean_{j in neighbors(i)} x[b, j, t]
    Isolated nodes (no neighbors at all) pool to zero.

    `deg` is read from neighbor_idx.shape[1] - 4 for the CTM compass grid, but
    arbitrary for a real network (cologne8: up to 6). Same code path either way.
    """
    B, N, T, D = x.shape
    deg = neighbor_idx.shape[1]
    safe_idx = neighbor_idx.clamp(min=0)  # -1 -> 0, masked out below; avoids gather OOB
    gathered = x[:, safe_idx]  # (B, N, deg, T, D)
    mask = neighbor_mask.to(x.dtype).view(1, N, deg, 1, 1)
    summed = (gathered * mask).sum(dim=2)  # (B, N, T, D)
    degree = mask.sum(dim=2).clamp(min=1.0)  # (1, N, 1, 1), avoid /0 for isolated nodes
    return summed / degree


class MultiAgentJEPA(JEPA):
    """Level 0 ("no coupling") / Level 0.5 ("pooled neighbor summary") JEPA.

    Same weights run on any node count / any grid shape - node count is read
    off the batch at forward time, never baked into a layer width. Nothing
    in the inputs is scenario-specific (no per-node ID embedding, no
    positional embedding over the node axis): every node sees only its own
    state/action and, at level 0.5, a permutation-invariant summary of its
    neighbors'.
    """

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        level="0",
        permute_control=False,
        projector=None,
        pred_proj=None,
    ):
        super().__init__(encoder, predictor, action_encoder, projector, pred_proj)
        level = str(level)  # hydra CLI overrides (model.level=0) parse unquoted "0" as int
        assert level in ("0", "0.5")
        self.level = level
        self.permute_control = permute_control
        self.permute_seed = 0
        self._perm_cache = {}  # n_nodes -> fixed derangement, drawn once per model instance

    def _fixed_derangement(self, n, device):
        """One permutation, fixed for the lifetime of this model instance (per
        n_nodes), with no fixed points. This is the attribution control: the
        model trains against a *consistently wrong* neighbor wiring, so any
        gain it shows is attributable to "pooling adds capacity/regularizes",
        not "pooling the right neighbors helps". Re-drawing the permutation
        every forward pass (the earlier bug here) makes it noise injection
        instead - a different, unintended experiment that plausibly explains
        gains *growing* with rollout horizon (regularization compounding with
        drift) rather than answering the attribution question at all.
        """
        if n not in self._perm_cache:
            g = torch.Generator().manual_seed(self.permute_seed * 1_000_003 + n)
            if n > 1:
                while True:
                    perm = torch.randperm(n, generator=g)
                    if (perm == torch.arange(n)).sum() == 0:
                        break
            else:
                perm = torch.randperm(n, generator=g)
            self._perm_cache[n] = perm
        return self._perm_cache[n].to(device)

    def encode(self, info):
        """info must have "state": (B, T, N*F), "action": (B, T, N*A), "n_nodes": int,
        and, at level 0.5, "neighbor_idx": (N, 4) int64 / "neighbor_mask": (N, 4) bool.

        Sets info["emb"]/"act_emb" to the *own* per-node embeddings (B*N, T, d) -
        used as the JEPA prediction target and for SIGReg, exactly like the
        single-node path. Sets info["pred_in_emb"]/"pred_in_act_emb" to what
        the predictor actually conditions on: identical to emb/act_emb at
        level 0, or own-concat-pooled-neighbor at level 0.5.
        """
        n = info["n_nodes"]
        state = info["state"].float()
        action = info["action"].float()
        B, T = state.shape[:2]
        F = state.shape[-1] // n
        A = action.shape[-1] // n

        state_n = rearrange(state.view(B, T, n, F), "b t n f -> (b n) t f")
        action_n = rearrange(action.view(B, T, n, A), "b t n a -> (b n) t a")

        z = self.encoder(state_n)  # (B*N, T, d)
        z = self.projector(rearrange(z, "bn t d -> (bn t) d"))
        z = rearrange(z, "(bn t) d -> bn t d", t=T)
        act_emb = self.action_encoder(action_n)  # (B*N, T, a_emb)

        info["emb"] = z
        info["act_emb"] = act_emb

        if self.level == "0":
            info["pred_in_emb"] = z
            info["pred_in_act_emb"] = act_emb
            return info

        neighbor_idx = info["neighbor_idx"]
        neighbor_mask = info["neighbor_mask"]
        z_bn = rearrange(z, "(b n) t d -> b n t d", b=B)
        act_bn = rearrange(act_emb, "(b n) t d -> b n t d", b=B)

        z_pool = masked_neighbor_mean(z_bn, neighbor_idx, neighbor_mask)
        act_pool = masked_neighbor_mean(act_bn, neighbor_idx, neighbor_mask)

        if self.permute_control:
            # attribution control: same shapes/magnitudes, correspondence to
            # the *right* neighbor destroyed - isolates "does pooling help"
            # from "does pooling *this node's actual neighbors* help". Fixed
            # per model instance (see _fixed_derangement), not redrawn per
            # call, so the model learns one consistently-wrong topology
            # rather than being regularized by fresh noise every step.
            perm = self._fixed_derangement(n, z.device)
            z_pool = z_pool[:, perm]
            act_pool = act_pool[:, perm]

        pred_in_emb = torch.cat([z_bn, z_pool], dim=-1)
        pred_in_act = torch.cat([act_bn, act_pool], dim=-1)

        info["pred_in_emb"] = rearrange(pred_in_emb, "b n t d -> (b n) t d")
        info["pred_in_act_emb"] = rearrange(pred_in_act, "b n t d -> (b n) t d")
        return info
