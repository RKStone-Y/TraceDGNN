import os
import os.path as osp
import json
import torch.utils.data
import numpy as np
import torch
from tqdm import tqdm
from collections.abc import Mapping, Sequence
from typing import List, Optional, Union
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Batch, Dataset
from torch_geometric.data.data import BaseData, Data


class TraceGraphDataset(Dataset):
    def __init__(self, root = None, transform = None, pre_transform = None, pre_filter = None, log = True, force_reload = False):
        super().__init__(root, transform, pre_transform, pre_filter, log, force_reload)
    
    def __getitem__(self, idx):
        return super().__getitem__(idx)
    
    @property
    def raw_file_names(self):
        """trace_file_dict: trace_id : filename"""
        file_dict = {}
        # list all snapshots json files and sort them
        # each file contains 50 snapshots in our experiment
        
        for file in os.listdir(self.raw_dir):
            if file.endswith('.json') and file.startswith('snapshots'):
                file_dict[file[file.index('_') + 1: -5]] = file
        file_list = [file_dict[k] for k in sorted(file_dict.keys())]
        # print(file_list)
        return file_list

    @property
    def raw_dir(self) -> str:
        dir = f"{self.root}/preprocessed/snapshots_train" 
        # dir = self.root + '/data/metric_' + str(self.metric_length)
        return dir

    @property
    def processed_dir(self) -> str:
        dir = self.root + '/processed'
        os.makedirs(dir, exist_ok=True)
        return dir

    @property
    def processed_file_names(self):
        file_list = []
        for file in os.listdir(self.processed_dir):
            if file in ['pre_filter.pt', 'pre_transform.pt']:
                continue
            if file.startswith(f'snapshot_'):
                file_list.append(file)
        return sorted(file_list)


    def len(self) -> int:
        return len(self.processed_file_names)


    def get(self, idx: int):
        data = torch.load(
            osp.join(self.processed_dir, f'snapshot_{idx}.pt'))
        return data



    def sub_process(self, file, idx):
        with open(file, 'r') as f:
            trace_data = json.load(f)

        filename = file.split('/')[-1]

        # 提取 "snapshots_" 之后和 ".json" 之前的部分
        trace_id = filename.replace("snapshots_", "").replace(".json", "")
        snapshot_id_list = []
        trace_anomaly = 0
        if trace_data['anomaly']:
            trace_anomaly = 0
        else:
            trace_anomaly = 1

        snapshots = trace_data['snapshots']
        # print(f"len={len(snapshots)}")

        for it, snapshot in enumerate(snapshots):
            
            data = Data(
                x=torch.tensor(
                        np.asarray(snapshot['span_embeddings']),  # 从 span_embeddings 字段读取 50 维向量
                        dtype=torch.float
                    ),
                edge_index=torch.tensor(
                    np.asarray(snapshot['alert_link_alert_edges']), dtype=torch.long),
                trace_id=trace_id,
                names = snapshot['alert_names'],
                is_new = snapshot['alert_is_new'],
                # anomaly=trace_anomaly,
                y=trace_anomaly
            )


            if self.pre_filter is not None and not self.pre_filter(graph_data):
                return

            if self.pre_transform is not None:
                graph_data = self.pre_transform(graph_data)
            # print(f"snapshots info:{graph_data['alert']}")
            
            snapshot_id_list.append(idx + it)
            torch.save(data, osp.join(self.processed_dir,
                                            f'snapshot_{idx + it}.pt'))
            
            return trace_id, snapshot_id_list, idx + len(snapshots)
        
    def process(self):

        idx = 0
        # results = []
        trace_id_list = {}

        for file in tqdm(self.raw_paths):
            trace_id, snap_list, idx = self.sub_process(file, idx)
            trace_id_list[trace_id] = snap_list