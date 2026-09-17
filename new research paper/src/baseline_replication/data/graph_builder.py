"""
Log Session Graph Builder for BertGCN.
Section 4.3 of Alsalmi et al. (CMC 2025/2026):

Graph Construction:
  - Each LOG SESSION (HDFS block or BGL sliding window) → one directed weighted graph G.
  - NODES: unique log events/messages within the session; node feature = BERT embedding x_i ∈ R^768.
  - DIRECTED EDGES: (i → j) if log event j appears directly after event i in the sequence.
    Edge weight w_ij = number of times event j follows event i (transition frequency).
  - SELF-LOOPS: Added on the FIRST event of the session (starting node), as described in Sec 4.3.
    "Self-loops are added on the starting events to preserve their importance."
  - ADJACENCY MATRIX: Weighted, directed N×N matrix where A[i,j] = w_ij (transition count).

The GraphBuilder embeds all unique messages in a session using BertEmbedder and constructs
the adjacency matrix and node feature tensor as numpy arrays (consumed directly by PyTorch).
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict


@dataclass
class LogGraph:
    """
    Represents a single labelled log session as a directed weighted graph.

    Attributes:
        node_features: np.ndarray of shape [N, 768] — BERT embeddings per unique event.
        adj_matrix:    np.ndarray of shape [N, N]   — directed weighted adjacency matrix.
        label:         int — 0 (Normal) or 1 (Anomaly).
        num_nodes:     int — number of unique events N.
        raw_sequence:  list of original log messages (for debugging).
    """
    node_features: np.ndarray       # [N, 768]
    adj_matrix: np.ndarray          # [N, N] — directed, weighted
    label: int                      # 0 = Normal, 1 = Anomaly
    num_nodes: int = field(init=False)
    raw_sequence: List[str] = field(default_factory=list, repr=False)

    def __post_init__(self):
        self.num_nodes = self.node_features.shape[0]


class GraphBuilder:
    """
    Builds directed weighted LogGraph objects from cleaned log sequences.

    Implements the graph construction described in Section 4.3 of Alsalmi et al.
    """

    def __init__(self, embedder):
        """
        Args:
            embedder: A BertEmbedder instance used to obtain node features.
        """
        self.embedder = embedder

    def build_graph_from_sequence(
        self,
        messages: List[str],
        label: int,
        raw_messages: List[str] = None,
    ) -> LogGraph:
        """
        Converts a list of cleaned log messages (a session) into a LogGraph.

        Steps:
          1. Deduplicate messages → nodes (preserving first-occurrence order).
          2. Map message → node index.
          3. Compute BERT embeddings for all unique messages (cached by BertEmbedder).
          4. Build directed weighted adjacency matrix from consecutive transitions.
          5. Add self-loop on the first/starting event node (Sec 4.3).
          6. Return a LogGraph with node_features, adj_matrix, and label.

        Args:
            messages:     List of cleaned log messages for this session.
            label:        Ground-truth label (0 = Normal, 1 = Anomaly).
            raw_messages: Optional original (uncleaned) messages for debugging.

        Returns:
            A LogGraph instance.
        """
        if not messages:
            # Edge case: empty session → single dummy node
            dummy_emb = np.zeros((1, 768), dtype=np.float32)
            adj = np.zeros((1, 1), dtype=np.float32)
            adj[0, 0] = 1.0  # self-loop
            return LogGraph(
                node_features=dummy_emb,
                adj_matrix=adj,
                label=label,
                raw_sequence=raw_messages or messages,
            )

        # ── 1. Deduplicate events → ordered unique node list ──────────────────
        seen: Dict[str, int] = {}
        unique_messages: List[str] = []
        for msg in messages:
            if msg not in seen:
                seen[msg] = len(unique_messages)
                unique_messages.append(msg)

        N = len(unique_messages)
        msg_to_idx: Dict[str, int] = {m: i for i, m in enumerate(unique_messages)}

        # ── 2. BERT Embeddings for all unique messages ────────────────────────
        node_features = self.embedder.embed_texts(unique_messages)  # [N, 768]

        # ── 3. Build directed weighted adjacency matrix ───────────────────────
        adj = np.zeros((N, N), dtype=np.float32)

        for i in range(len(messages) - 1):
            src = msg_to_idx[messages[i]]
            dst = msg_to_idx[messages[i + 1]]
            adj[src, dst] += 1.0

        # ── 4. Self-loop on the FIRST/STARTING event (Sec 4.3) ───────────────
        # The paper states: "self-loops are added on the starting events to
        # preserve their importance in the graph."
        first_event_idx = msg_to_idx[messages[0]]
        adj[first_event_idx, first_event_idx] += 1.0

        return LogGraph(
            node_features=node_features,
            adj_matrix=adj,
            label=label,
            raw_sequence=raw_messages or messages,
        )
