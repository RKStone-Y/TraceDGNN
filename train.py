import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from trace_dataset import TraceDataset
from model import TraceModel

# 参数设定
in_channels = 50            # snapshot 的 x 特征维度
gnn_hidden_dim = 64
gnn_out_dim = 64            # GNN 输出作为 GRU 的输入
rnn_hidden_dim = 128        # GRU 输出维度
num_classes = 2             # 二分类
num_epochs = 10
learning_rate = 1e-3
batch_size = 1              # 注意：batch_size > 1 时需调整处理逻辑

# 数据集和数据加载
dataset = TraceDataset(
    "/home/fdse/yzc/TraceDyGNN/TraceRAG_DATA_Preprocess/trace_to_snapshot_map.json",
    "/home/fdse/yzc/TraceDyGNN/TraceRAG_DATA_Preprocess/processed"
)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)

# 模型、损失、优化器
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1)

model = TraceModel(
    in_channels=in_channels,
    gnn_hidden=gnn_hidden_dim,
    gnn_out=gnn_out_dim,
    rnn_hidden=rnn_hidden_dim,
    num_classes=num_classes,
    bidirectional=False
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练过程
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for batch in dataloader:
        # batch_size = 1，所以 batch 是一个长度为 1 的列表
        snapshots, label = batch[0]

        # label 是标量 tensor，如 tensor(0)，转换为 1D 张量 [0] 以符合 CrossEntropyLoss
        label = label.unsqueeze(0).to(device)

        # snapshots 是一个 list[Data]，每个 snapshot 是一个图
        snapshots = [data.to(device) for data in snapshots]

        optimizer.zero_grad()
        output = model(snapshots)  # output shape: [1, num_classes]
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# 保存模型
torch.save(model.state_dict(), "trace_model.pth")
