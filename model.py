import torch
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool


# GNN Encoder 
class GNNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if edge_index.max().item() >= x.size(0):
            print("❌ edge_index 越界:")
            print(f"x.shape: {x.shape}")
            print(f"edge_index.max(): {edge_index.max().item()}, x.size(0): {x.size(0)}")
            print(f"edge_index:\n{edge_index}")
            raise ValueError("edge_index 中包含无效节点索引")
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = global_mean_pool(x, batch)  # [num_graphs, out_channels]
        return x


# GRU Encoder 
class GRUEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, bidirectional=False):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, seq_tensor):  # [batch_size=1, seq_len, input_size]
        _, h_n = self.gru(seq_tensor)  # h_n: [num_layers * num_directions, batch, hidden_size]
        if self.bidirectional:
            # Concatenate last layer's forward and backward hidden state
            forward = h_n[-2]
            backward = h_n[-1]
            return torch.cat((forward, backward), dim=-1)  # [batch, hidden_size*2]
        else:
            return h_n[-1]  # [batch, hidden_size]


class TraceModel(nn.Module):
    def __init__(self,
                 in_channels,          # dim of node features
                 gnn_hidden,           # hidden dim in GNN
                 gnn_out,              # output dim of GNN
                 rnn_hidden,           # hidden dim in GRU
                 num_classes,          # number of output classes
                 bidirectional=False): # use BiGRU or not
        super().__init__()
        self.gnn_encoder = GNNEncoder(in_channels, gnn_hidden, gnn_out)
        self.gru_encoder = GRUEncoder(
            input_size=gnn_out,
            hidden_size=rnn_hidden,
            num_layers=2,
            bidirectional=bidirectional
        )
        self.final_hidden = rnn_hidden * 2 if bidirectional else rnn_hidden
        self.classifier = nn.Linear(self.final_hidden, num_classes)

    def encode_trace(self, snapshots):
        """Encodes a trace from its snapshots to a single vector representation."""
        embeddings = []
        for snapshot in snapshots:
            if not hasattr(snapshot, 'batch'):
                snapshot.batch = torch.zeros(snapshot.num_nodes, dtype=torch.long, device=snapshot.x.device)
            emb = self.gnn_encoder(snapshot)  # [1, gnn_out]
            embeddings.append(emb)

        seq = torch.stack(embeddings, dim=1)  # [1, T, gnn_out]
        trace_repr = self.gru_encoder(seq)    # [1, final_hidden]
        return trace_repr

    def forward(self, snapshots):
        trace_repr = self.encode_trace(snapshots)
        out = self.classifier(trace_repr)     # [1, num_classes]
        return out
