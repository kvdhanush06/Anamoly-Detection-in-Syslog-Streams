"""
Model Trainer with Minority Oversampling, Optimization, and Evaluation.
Section 4.5 & 5.1 of Alsalmi et al.:
- Oversampling: Random duplication of minority anomaly class in training set
- Optimizer: Adam
- Loss: Cross-Entropy Loss (or Focal Loss for Enhanced Model)
- 70:30 Train-Test split
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from typing import List, Tuple, Dict, Any
from tqdm import tqdm
try:
    from .evaluate import compute_metrics
    from .data.graph_builder import LogGraph
except ImportError:
    from evaluate import compute_metrics
    from data.graph_builder import LogGraph

def oversample_graphs(graphs: List[LogGraph], random_state: int = 42) -> List[LogGraph]:
    """
    Applies random minority oversampling to balance normal (0) and anomalous (1) graphs.
    """
    normals = [g for g in graphs if g.label == 0]
    anomalies = [g for g in graphs if g.label == 1]
    
    if not anomalies or not normals:
        return graphs
        
    num_normals = len(normals)
    num_anomalies = len(anomalies)

    if num_anomalies < num_normals:
        rng = random.Random(random_state)
        # Duplicate anomalies with replacement until counts match
        sampled_anomalies = [rng.choice(anomalies) for _ in range(num_normals)]
        combined = normals + sampled_anomalies
        rng.shuffle(combined)
        return combined
    return graphs

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

def train_model(
    model: nn.Module,
    train_graphs: List[LogGraph],
    val_graphs: List[LogGraph] = None,
    epochs: int = 15,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    use_focal_loss: bool = False,
    device: str = None
) -> Dict[str, Any]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Compute class counts for weighted cross entropy if oversampling isn't enough
    labels = [g.label for g in train_graphs]
    num_norm = sum(1 for l in labels if l == 0)
    num_anom = sum(1 for l in labels if l == 1)
    
    if use_focal_loss:
        criterion = FocalLoss().to(device)
    elif num_anom > 0 and num_norm > 0:
        pos_weight = torch.tensor([1.0, num_norm / num_anom], device=device)
        criterion = nn.CrossEntropyLoss(weight=pos_weight).to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)

    best_val_f1 = -1.0
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        random.shuffle(train_graphs)

        for g in train_graphs:
            x = torch.tensor(g.node_features, dtype=torch.float32, device=device)
            adj = torch.tensor(g.adj_matrix, dtype=torch.float32, device=device)
            target = torch.tensor([g.label], dtype=torch.long, device=device)

            optimizer.zero_grad()
            logits = model(x, adj) # [1, 2]
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_graphs))

        # Validation check if val set is provided
        if val_graphs:
            val_metrics = evaluate_model(model, val_graphs, device=device)
            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_loss:.4f} | Val F1: {val_metrics['f1']*100:.2f}% | Val Rec: {val_metrics['recall']*100:.2f}%")
        else:
            if epoch % 5 == 0 or epoch == epochs:
                print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_loss:.4f}")

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return {"model": model, "best_val_f1": best_val_f1}

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    test_graphs: List[LogGraph],
    threshold: float = 0.5,
    device: str = None
) -> Dict[str, float]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model.eval()
    y_true = []
    y_pred = []

    for g in test_graphs:
        x = torch.tensor(g.node_features, dtype=torch.float32, device=device)
        adj = torch.tensor(g.adj_matrix, dtype=torch.float32, device=device)
        
        logits = model(x, adj) # [1, 2]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        anomaly_prob = probs[1]

        pred = 1 if anomaly_prob >= threshold else 0
        y_true.append(g.label)
        y_pred.append(pred)

    return compute_metrics(np.array(y_true), np.array(y_pred))
