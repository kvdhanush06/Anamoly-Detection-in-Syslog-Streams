"""
Baseline BertGCN Architecture as defined in Section 4.4 & 4.5 of Alsalmi et al.:
- Graph Convolution Layer:
  H^(l+1) = sigma( D_tilde^(-1/2) * A_tilde * D_tilde^(-1/2) * H^(l) * W^(l) )
  where A_tilde = A + I, D_tilde is diagonal degree matrix.
- Readout: Global pooling R(.) = mean, sum, or max pooling to form h_G.
- Classifier: y_hat = Softmax(W_c * h_G + b_c) with Cross-Entropy Loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(GraphConvLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x: [N, in_features]
        adj: [N, N] (weighted adjacency matrix)
        Computes symmetric normalized GCN convolution:
        A_tilde = adj + I
        D_tilde = diag(sum(A_tilde, dim=1))
        norm_adj = D_tilde^(-1/2) * A_tilde * D_tilde^(-1/2)
        out = norm_adj * (x * W) + b
        """
        N = adj.size(0)
        # Add self-loops
        A_tilde = adj + torch.eye(N, device=adj.device)
        
        # Degree matrix
        degrees = torch.sum(A_tilde, dim=1)
        deg_inv_sqrt = torch.pow(torch.clamp(degrees, min=1e-8), -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        D_inv_sqrt = torch.diag(deg_inv_sqrt)

        # Normalized adjacency
        norm_adj = torch.mm(torch.mm(D_inv_sqrt, A_tilde), D_inv_sqrt)

        # Message passing & feature projection
        support = torch.mm(x, self.weight)
        output = torch.mm(norm_adj, support) + self.bias
        return output

class BertGCN(nn.Module):
    def __init__(self, in_dim: int = 768, hidden_dim: int = 128, num_classes: int = 2, num_layers: int = 2, readout: str = "mean", dropout: float = 0.2):
        super(BertGCN, self).__init__()
        self.readout = readout
        self.dropout = dropout

        self.convs = nn.ModuleList()
        # Layer 1
        self.convs.append(GraphConvLayer(in_dim, hidden_dim))
        # Subsequent layers
        for _ in range(num_layers - 1):
            self.convs.append(GraphConvLayer(hidden_dim, hidden_dim))

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Processes a single graph or batched graph.
        x: [N, in_dim]
        adj: [N, N]
        """
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, adj)
            if i < len(self.convs) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)

        # Graph Readout Pooling
        if self.readout == "mean":
            h_g = torch.mean(h, dim=0, keepdim=True)
        elif self.readout == "max":
            h_g = torch.max(h, dim=0, keepdim=True)[0]
        elif self.readout == "sum":
            h_g = torch.sum(h, dim=0, keepdim=True)
        else:
            h_g = torch.mean(h, dim=0, keepdim=True)

        logits = self.classifier(h_g) # [1, num_classes]
        return logits
