import os
import os.path as osp
import json
import numpy as np
import torch
from tqdm import tqdm
from torch_geometric.data import Dataset
from torch_geometric.data.data import BaseData, Data



"""
    This is a data initialization for TraceDyGNN
"""

class TraceGraphDataset(Dataset):
    def __init__(self, root = None, transform = None, pre_transform = None, pre_filter = None, log = True):
        map_file = "/home/fdse/yzc/TraceDyGNN/TraceRAG_DATA_Preprocess/trace_to_snapshot_map.json"
        self.trace_sanpshot_map = {}
        if os.path.exists(map_file):
        
            with open(map_file, 'r', encoding='utf-8') as f:
                self.trace_sanpshot_map = json.load(f)  # 读取现有数据
            print(f"已加载现有文件: {map_file}")
        else:
            print(f"文件 {map_file} 不存在，将创建新文件")
            self.trace_sanpshot_map = {}
        super().__init__(root, transform, pre_transform, pre_filter, log)

        
    
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
        dir = f"/home/fdse/yzc/TraceDyGNN/DyAlert/preprocessed/snapshots_embedding/train"
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
                edge_index=torch.tensor(np.asarray(snapshot['edge_index']).astype(np.int64), dtype=torch.long),
                trace_id=trace_id,
                names = snapshot['span_names'],
                is_new = snapshot['is_new'],
                y=trace_anomaly
            )



            if self.pre_filter is not None and not self.pre_filter(graph_data):
                return

            if self.pre_transform is not None:
                graph_data = self.pre_transform(graph_data)
            # print(f"snapshots info:{graph_data['alert']}")
            
            snapshot_id_list.append(idx + it)
            
            torch.save(data, osp.join(self.processed_dir, f'snapshot_{idx + it}.pt'))
            
        
        return trace_id, trace_anomaly, snapshot_id_list, idx + len(snapshots)
        

    def process(self):

        idx = 0
        # results = []
        # self.trace_sanpshot_map = {}

        for file in tqdm(self.raw_paths):
            print(f"file: {file}")

            trace_id, trace_anomaly, snap_list, idx = self.sub_process(file, idx)
            self.trace_sanpshot_map[trace_id] = {
                "label": trace_anomaly,
                "snapshot_list": snap_list
            }
            if idx > 1000:
                print("数据处理完成，已达到1000个快照，停止处理")
                break
            # break

        map_path = self.processed_dir + '/trace_to_snapshot_map.json'
        with open(map_path, 'w', encoding='utf-8') as f:
            json.dump(self.trace_sanpshot_map, f, indent=4, ensure_ascii=False)
            
        print("数据已成功写入文件")



if __name__ == '__main__':
    dataset = TraceGraphDataset('/home/fdse/yzc/TraceDyGNN/TraceRAG_DATA_Preprocess')
    print(dataset)
    print(dataset[0])
    # print(dataset[186].edge_index)
    # print(dataset[0].names)