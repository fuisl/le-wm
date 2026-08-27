"""Train the multi-agent JEPA (Level 0 / Level 0.5, traffic/multi_agent.py)
on the synthetic CTM grid.

Mirrors train_traffic.py's structure exactly; only the encode/target wiring
differs, because the predictor's input (own state, or own+pooled-neighbor at
level 0.5) and the JEPA target (always the node's own next embedding) are no
longer the same tensor once neighbor pooling is in the loop.

Usage:
    python train_traffic_multi.py model.level=0
    python train_traffic_multi.py model.level=0.5
    python train_traffic_multi.py model.level=0.5 model.permute_control=true  # attribution control
"""



from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from module import SIGReg
from traffic.ctm_env import CTMGridEnv
from traffic.dataset import TrafficDataset
from utils import SaveCkptCallback


def lejepa_forward_multi(self, batch, stage, cfg, n_nodes, neighbor_idx, neighbor_mask):
    """Node-structured JEPA step.

    Own embedding (info["emb"]) is always the loss target and the SIGReg
    input, regardless of level - it's the thing the model is actually
    representing. The predictor's input (info["pred_in_emb"/"pred_in_act_emb"])
    is level-dependent: identical to the own embedding at level 0, or
    own-concat-pooled-neighbor at level 0.5. Conflating the two would let
    level 0.5 "predict" a target that leaks into its own input.
    """
    ctx_len = cfg.history_size
    n_preds = cfg.num_preds
    lambd = cfg.loss.sigreg.weight

    batch = {**batch, "n_nodes": n_nodes}
    if str(cfg.model.level) == "0.5":
        device = batch["state"].device
        batch["neighbor_idx"] = neighbor_idx.to(device)
        batch["neighbor_mask"] = neighbor_mask.to(device)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B*N, T, d) - own embedding, prediction target
    pred_in_emb = output["pred_in_emb"][:, :ctx_len]
    pred_in_act = output["pred_in_act_emb"][:, :ctx_len]
    tgt_emb = emb[:, n_preds:]

    pred_emb = self.model.predict(pred_in_emb, pred_in_act)

    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


@hydra.main(version_base=None, config_path="./config/train", config_name="traffic_multi")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    window = cfg.history_size + cfg.num_preds
    train_set = TrafficDataset(cfg.data.train_path, window=window)
    val_set = TrafficDataset(cfg.data.val_path, window=window)

    assert train_set.rows is not None, (
        f"{cfg.data.train_path} has no rows/cols metadata - regenerate with "
        "traffic.generate_dataset (grid data only, this script is CTM-grid-specific)"
    )
    n_nodes = train_set.rows * train_set.cols
    node_feature_dim = train_set.state_dim // n_nodes
    node_action_dim = train_set.action_dim // n_nodes

    neighbor_idx, neighbor_mask = CTMGridEnv(rows=train_set.rows, cols=train_set.cols).neighbor_table()
    neighbor_idx = torch.from_numpy(neighbor_idx)
    neighbor_mask = torch.from_numpy(neighbor_mask)

    with open_dict(cfg):
        cfg.model.encoder.input_dim = node_feature_dim
        cfg.model.action_encoder.input_dim = node_action_dim
        cfg.model.predictor.input_dim = (
            cfg.embed_dim if str(cfg.model.level) == "0" else 2 * cfg.embed_dim
        )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader, shuffle=True, drop_last=True)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)

    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(
            lejepa_forward_multi,
            cfg=cfg,
            n_nodes=n_nodes,
            neighbor_idx=neighbor_idx,
            neighbor_mask=neighbor_mask,
        ),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_dir = Path("traffic_runs", cfg.subdir)
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name, cfg=cfg.model, epoch_interval=5,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
        default_root_dir=str(run_dir),
    )

    ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    manager()
    return


if __name__ == "__main__":
    run()
