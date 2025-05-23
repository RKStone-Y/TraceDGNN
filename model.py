import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1)

# ----- 1. GNN Encoder -----
class GNNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        return x

# ----- 2. GRU Encoder -----
class GRUEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, seq_tensor):
        _, h_n = self.gru(seq_tensor)
        return h_n[-1]  # Last layer's hidden state

# ----- 3. Complete Trace Model -----
class TraceModel(nn.Module):
    def __init__(self, gnn_in, gnn_hidden, gnn_out, gru_hidden, num_classes):
        super().__init__()
        self.gnn = GNNEncoder(gnn_in, gnn_hidden, gnn_out)
        self.gru = GRUEncoder(gnn_out, gru_hidden)
        self.classifier = nn.Linear(gru_hidden, num_classes)

    def forward(self, trace_snapshots):
        graph_embeddings = []
        for snap in trace_snapshots:
            snap.batch = torch.zeros(snap.num_nodes, dtype=torch.long)
            emb = self.gnn(snap)  # shape: [1, gnn_out]
            graph_embeddings.append(emb)
        seq = torch.stack(graph_embeddings, dim=1)  # shape: [1, seq_len, gnn_out]
        trace_rep = self.gru(seq)  # shape: [1, gru_hidden]
        out = self.classifier(trace_rep)  # shape: [1, num_classes]
        return out

# ----- 4. Custom Dataset -----
class TraceDataset(Dataset):
    def __init__(self, trace_list):
        self.trace_list = trace_list  # List of (snapshots, label) tuples

    def __len__(self):
        return len(self.trace_list)

    def __getitem__(self, idx):
        return self.trace_list[idx]  # (List[Data], label)

# ----- 5. Training Loop -----
def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for trace_snapshots, label in dataloader:
        label = label.to(device)
        trace_snapshots = [g.to(device) for g in trace_snapshots[0]]  # batch size = 1

        optimizer.zero_grad()
        out = model(trace_snapshots)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# Example Usage:
model = TraceModel(gnn_in=50, gnn_hidden=64, gnn_out=64, gru_hidden=64, num_classes=2).to(device)
# dataset = TraceDataset(trace_data_list)  # List of (List[Data], label)
dataset = TraceDataset("trace_map.json", "snapshots/")
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_model(model, dataloader, criterion, optimizer, device)
torch.save(model.state_dict(), "trace_model.pth")