"""torch Dataset over CTM episodes saved by generate_dataset.py.

Each item is a fixed-length window of consecutive (state, action) steps,
matching the shape lejepa_forward expects: (window, state_dim) / (window,
action_dim) with window == history_size + num_preds.
"""

import torch
from torch.utils.data import Dataset


class TrafficDataset(Dataset):
    def __init__(self, path, window):
        data = torch.load(path, weights_only=False)
        self.episodes = data["episodes"]
        self.state_dim = data["state_dim"]
        self.action_dim = data["action_dim"]
        self.rows = data.get("rows")  # None for non-grid (e.g. SUMO) datasets
        self.cols = data.get("cols")
        self.window = window

        self.index = []
        for ei, ep in enumerate(self.episodes):
            T = ep["state"].shape[0]
            for start in range(T - window + 1):
                self.index.append((ei, start))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ei, start = self.index[idx]
        ep = self.episodes[ei]
        end = start + self.window
        item = {
            "state": torch.from_numpy(ep["state"][start:end]),
            "action": torch.from_numpy(ep["action"][start:end]),
        }
        if "edge_feat" in ep:
            item["edge_feat"] = torch.from_numpy(ep["edge_feat"][start:end])
        return item
