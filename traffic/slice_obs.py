"""Slice a corpus generated with --obs_mode full into base / link / raster corpora
(same episodes, same seeds, column subsets), so that observation richness is
the ONLY difference between the models trained on them.

Usage:
    python -m traffic.slice_obs --src traffic_data_cologne8_full --dst_prefix traffic_data_cologne8
      -> traffic_data_cologne8_base / _link / _raster  (train.pt, val.pt, counterfactual.pt)
    optionally --check_against traffic_data_cologne8  (old base corpus; asserts equality of
    the base block episode-by-episode, so L05ar stays a valid baseline)
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from traffic.sumo_multi_env import RASTER_CELLS


def block_cols(P, mode):
    base = list(range(0, 2 * P + 1))
    link = list(range(2 * P + 1, 2 * P + 1 + 10))
    r0 = 2 * P + 1 + 10
    ras = list(range(r0, r0 + P * RASTER_CELLS * 2))
    return {"base": base, "link": base + link, "raster": base + ras, "full": base + link + ras}[mode]


def slice_state(x, N, F_full, cols):
    lead = x.shape[:-1]
    return x.reshape(*lead, N, F_full)[..., cols].reshape(*lead, N * len(cols)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="traffic_data_cologne8_full")
    ap.add_argument("--dst_prefix", default="traffic_data_cologne8")
    ap.add_argument("--modes", nargs="+", default=["base", "link", "raster"])
    ap.add_argument("--check_against", default=None)
    args = ap.parse_args()
    src = Path(args.src)
    for mode in args.modes:
        dst = Path(f"{args.dst_prefix}_{mode}")
        dst.mkdir(exist_ok=True)
        for split in ("train", "val"):
            d = torch.load(src / f"{split}.pt", weights_only=False)
            N, F_full, P = d["n_nodes"], d["node_feature_dim"], d["P_max"]
            cols = block_cols(P, mode)
            eps = []
            for ep in d["episodes"]:
                e = dict(ep)
                e["state"] = slice_state(ep["state"], N, F_full, cols)
                eps.append(e)
            out = dict(d); out["episodes"] = eps
            out["node_feature_dim"] = len(cols); out["state_dim"] = N * len(cols); out["obs_mode"] = mode
            torch.save(out, dst / f"{split}.pt")
            if args.check_against and mode == "base" and split == "train":
                old = torch.load(Path(args.check_against) / "train.pt", weights_only=False)
                diffs = [float(np.abs(a["state"] - b["state"]).max()) for a, b in zip(eps, old["episodes"])]
                print(f"[check] base slice vs {args.check_against}: max |diff| per episode = "
                      f"{np.round(diffs, 3).tolist()}")
        cf = torch.load(src / "counterfactual.pt", weights_only=False)
        N, F_full, P = cf["n_nodes"], cf["node_feature_dim"], cf["P_max"]
        cols = block_cols(P, mode)
        samples = []
        for s in cf["samples"]:
            s2 = dict(s)
            s2["anchor_state"] = slice_state(s["anchor_state"], N, F_full, cols)
            s2["branches"] = [dict(b, states=slice_state(b["states"], N, F_full, cols)) for b in s["branches"]]
            samples.append(s2)
        out = dict(cf); out["samples"] = samples
        out["node_feature_dim"] = len(cols); out["state_dim"] = N * len(cols); out["obs_mode"] = mode
        torch.save(out, dst / "counterfactual.pt")
        print(f"wrote {dst}: F={len(cols)}")


if __name__ == "__main__":
    main()
