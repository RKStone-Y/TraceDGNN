import torch
from torch.utils.data import DataLoader
from trace_dataset import TraceDataset
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from model import TraceModel

# ----------- 参数配置 -----------
in_channels = 50
gnn_hidden_dim = 64
gnn_out_dim = 64
rnn_hidden_dim = 128
num_classes = 2
batch_size = 1  # 必须为 1，当前模型结构不支持批量 trace 序列处理

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# ----------- 加载测试集 -----------
test_dataset = TraceDataset(
    "/home/fdse/yzc/TraceDyGNN/TraceRAG_DATA_Preprocess/trace_to_snapshot_map.json",
    "/home/fdse/yzc/TraceDyGNN/TraceRAG_DATA_Preprocess/processed"
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=lambda x: x
)

# ----------- 初始化模型并加载训练好的参数 -----------
model = TraceModel(
    in_channels=in_channels,
    gnn_hidden=gnn_hidden_dim,
    gnn_out=gnn_out_dim,
    rnn_hidden=rnn_hidden_dim,
    num_classes=num_classes,
    bidirectional=False
).to(device)

model.load_state_dict(torch.load("/home/fdse/yzc/TraceDyGNN/TraceRAG_DATA_Preprocess/trace_model.pth", map_location=device))
model.eval()

# ----------- 推理并评估准确率 -----------
# correct = 0
# total = 0
y_true = []
y_pred = []

with torch.no_grad():
    for batch in test_loader:
        # snapshots, label = batch[0]
        # snapshots = [graph.to(device) for graph in snapshots]
        # label = label.to(device).unsqueeze(0)  # [1]

        # output = model(snapshots)              # [1, num_classes]
        # predicted = torch.argmax(output, dim=1)  # [1]

        # if predicted.item() == label.item():
        #     correct += 1
        # total += 1
        # print(f"True label: {label.item()}, Predicted: {predicted.item()}")

        snapshots, label = batch[0]
        snapshots = [graph.to(device) for graph in snapshots]
        label = label.to(device)

        output = model(snapshots)  # [1, num_classes]
        pred = torch.argmax(output, dim=1).item()

        y_true.append(label.item())
        y_pred.append(pred)

# accuracy = correct / total if total > 0 else 0
# print(f"\n✅ Test Accuracy: {accuracy * 100:.2f}%  ({correct}/{total})")

precision = precision_score(y_true, y_pred, average='binary')
recall = recall_score(y_true, y_pred, average='binary')
f1score = f1_score(y_true, y_pred, average='binary')
accuracy = accuracy_score(y_true, y_pred)

print(f"✅ Test Results:")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1score:.4f}")
print(f"Accuracy:  {accuracy:.4f}")
