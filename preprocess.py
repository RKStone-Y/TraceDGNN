import json
import os
from tqdm import tqdm


def reindex_snapshots(input_path):
    with open(input_path, 'r') as f:
        data = json.load(f)

    max_id = 0  # 当前使用的最大编号
    old_to_new_map = {}  # 映射 span_name 原编号 -> 新编号

    for snapshot in data['snapshots']:
        new_span_names = []
        updated_map = {}

        for i, name in enumerate(snapshot['span_names']):
            is_new = snapshot['is_new'][i]
            if is_new == 1:
                new_id = str(max_id)
                max_id += 1
            else:
                # 如果之前出现过该节点，保持其编号
                if name in old_to_new_map:
                    new_id = old_to_new_map[name]
                else:
                    new_id = str(max_id)
                    max_id += 1
            new_span_names.append(new_id)
            updated_map[name] = new_id

        # 更新 span_names 和记录映射
        snapshot['span_names'] = new_span_names
        old_to_new_map.update(updated_map)

        # 更新 edge_index
        if 'edge_index' in snapshot:
            new_edge_index = [[], []]
            for i, src in enumerate(snapshot['edge_index'][0]):
                dst = snapshot['edge_index'][1][i]
                new_edge_index[0].append(updated_map.get(src, src))
                new_edge_index[1].append(updated_map.get(dst, dst))
            snapshot['edge_index'] = new_edge_index

    with open(input_path, 'w') as f:
        json.dump(data, f, indent=2)

# 示例用法
if __name__ == "__main__":
    # 替换为实际的输入和输出路径
    for file in tqdm(os.listdir('/home/fdse/yzc/TraceDyGNN/DyAlert/preprocessed/snapshots_embedding/train'), desc="Processing files"):
        if file.endswith('.json'):
            input_path = os.path.join('/home/fdse/yzc/TraceDyGNN/DyAlert/preprocessed/snapshots_embedding/train', file)
            reindex_snapshots(input_path)

            # reindex_snapshots('/home/fdse/yzc/TraceDyGNN/DyAlert/preprocessed/snapshots_embedding/train/snapshots_1a5fbf9d87a018475c8797b78a600894.json')
