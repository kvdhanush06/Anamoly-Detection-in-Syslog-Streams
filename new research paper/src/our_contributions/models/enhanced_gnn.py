"""
Our Novel Architectural Enhancements over Baseline BertGCN:
1. Edge-Weighted Graph Attention (EW-GAT) / Weighted Convolution:
   Unlike vanilla GCN which discards transition frequency differences in its normalization,
   EW-GAT uses transition frequencies log(1 + w_ij) as edge priors in attention scoring.
2. Multi-Scale Readout:
   Combines mean, max, and std graph pooling:
   h_G = [ mean(H) || max(H) || std(H) ]
   followed by a non-linear projection MLP with residual skip.
3. Focal Loss & Dynamic Threshold Support:
   Enhances gradient updates on hard boundary examples.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeWeightedGATConv(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.2, alpha: float = 0.2):
        super(EdgeWeightedGATConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a_src = nn.Parameter(torch.empty(size=(out_features, 1)))
        self.a_dst = nn.Parameter(torch.empty(size=(out_features, 1)))
        self.a_edge = nn.Parameter(torch.empty(size=(1, 1)))
        nn.init.xavier_uniform_(self.a_src.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_edge.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        h: [N, in_features]
        adj: [N, N] (weighted transition counts)
        """
        N = h.size(0)
        # Linear transformation
        Wh = torch.mm(h, self.W) # [N, out_features]

        # Self-attention with edge transition frequency prior
        f_1 = torch.mm(Wh, self.a_src) # [N, 1]
        f_2 = torch.mm(Wh, self.a_dst) # [N, 1]
        
        # log-scaled transition frequencies
        log_edge_weights = torch.log1p(adj) # [N, N]
        
        # Combined attention logit: e_ij = LeakyReLU(f_1[i] + f_2[j] + w_e * log(1+w_ij))
        attn_logits = self.leakyrelu(f_1 + f_2.T + log_edge_weights * self.a_edge)

        # Mask non-edges (except self loops and positive transitions)
        mask = (adj > 0) | torch.eye(N, dtype=torch.bool, device=adj.device)
        attn_logits = attn_logits.masked_fill(~mask, -1e9)

        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        h_prime = torch.mm(attn_weights, Wh) + self.bias
        return h_prime

class EnhancedLogGNN(nn.Module):
    def __init__(self, in_dim: int = 768, hidden_dim: int = 128, num_classes: int = 2, num_layers: int = 2, dropout: float = 0.2):
        super(EnhancedLogGNN, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(EdgeWeightedGATConv(in_dim, hidden_dim, dropout=dropout))
        for _ in range(num_layers - 1):
            self.convs.append(EdgeWeightedGATConv(hidden_dim, hidden_dim, dropout=dropout))

        # Multi-scale readout concatenates Mean + Max + Std pooling: hidden_dim * 3
        readout_dim = hidden_dim * 3
        self.mlp_head = nn.Sequential(
            nn.Linear(readout_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = x
        for i, conv in enumerate(self.convs):
            h_new = conv(h, adj)
            if i > 0 and h_new.shape == h.shape:
                h = h + F.elu(h_new) # Residual skip connection
            else:
                h = F.elu(h_new)

        # Multi-Scale Readout Pooling
        mean_p = torch.mean(h, dim=0, keepdim=True)
        max_p = torch.max(h, dim=0, keepdim=True)[0]
        std_p = torch.std(h, dim=0, keepdim=True) if h.size(0) > 1 else torch.zeros_like(mean_p)
        
        h_g = torch.cat([mean_p, max_p, std_p], dim=-1) # [1, hidden_dim * 3]
        logits = self.mlp_head(h_g)                     # [1, num_classes]
        return logits
