import os
import json
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

class TraceDataset(Dataset):
    def __init__(self, mapping_json_path, snapshot_dir):
        with open(mapping_json_path, 'r') as f:
            self.trace_map = json.load(f)  # {trace_id: {'label': 0, 'snapshots': [...]}}
        self.snapshot_dir = snapshot_dir
        self.trace_ids = list(self.trace_map.keys())

    def __len__(self):
        return len(self.trace_ids)

    def __getitem__(self, idx):
        trace_id = self.trace_ids[idx]
        trace_info = self.trace_map[trace_id]
        snapshot_ids = trace_info["snapshots"]
        label = torch.tensor([trace_info["label"]], dtype=torch.long)

        snapshots = []
        for snap_id in snapshot_ids:
            snap_path = os.path.join(self.snapshot_dir, f"{snap_id}.pt")
            snap = torch.load(snap_path)
            if not hasattr(snap, 'batch'):
                snap.batch = torch.zeros(snap.num_nodes, dtype=torch.long)
            snapshots.append(snap)

        return snapshots, label
