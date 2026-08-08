"""
Phase 2 - Step 2: Graph Neural Network Encoder

GAT-based encoder that converts variable-size UAST graphs
into fixed-dimensional embeddings (768-dim) for downstream tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, GINConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch


class UASTGraphEncoder(nn.Module):
    """
    Graph Attention Network for UAST encoding.

    Architecture:
        Input: Variable-size UAST graph (node features + edge index)
        → 3x GAT layers with batch normalization
        → Global mean pooling
        Output: Fixed 768-dim embedding
    """

    def __init__(
        self,
        node_feature_dim: int = 6,
        hidden_dim: int = 256,
        output_dim: int = 768,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_layers: int = 3,
    ):
        super().__init__()

        self.dropout = dropout
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Linear(node_feature_dim, hidden_dim)

        # GAT layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            if i == num_layers - 1:
                # Last layer: single head, output dim
                self.convs.append(
                    GATConv(
                        hidden_dim * num_heads if i > 0 else hidden_dim,
                        output_dim,
                        heads=1,
                        concat=False,
                        dropout=dropout,
                    )
                )
                self.batch_norms.append(nn.BatchNorm1d(output_dim))
            else:
                in_dim = hidden_dim if i == 0 else hidden_dim * num_heads
                self.convs.append(
                    GATConv(
                        in_dim,
                        hidden_dim,
                        heads=num_heads,
                        dropout=dropout,
                    )
                )
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim * num_heads))

    def forward(self, x, edge_index, batch):
        """
        Args:
            x: [num_nodes, node_feature_dim]
            edge_index: [2, num_edges]
            batch: [num_nodes] graph assignment vector

        Returns:
            graph_embedding: [batch_size, output_dim]
        """
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)

        # GAT layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling
        graph_emb = global_mean_pool(x, batch)

        return graph_emb


class UASTClassifier(nn.Module):
    """
    UAST GNN Encoder + Domain Classification Head.

    Used for Phase 2 domain classification task.
    """

    def __init__(
        self,
        num_classes: int,
        node_feature_dim: int = 6,
        hidden_dim: int = 256,
        embedding_dim: int = 768,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder = UASTGraphEncoder(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x, edge_index, batch):
        """Returns (logits, embeddings)."""
        embeddings = self.encoder(x, edge_index, batch)
        logits = self.classifier(embeddings)
        return logits, embeddings

    def get_embeddings(self, x, edge_index, batch):
        """Get graph embeddings without classification."""
        return self.encoder(x, edge_index, batch)
