"""
SymAlign-UAST Classifier: End-to-end Neuro-Symbolic Model.

Architecture:
    UAST Graph → GNN Encoder (768-dim)
                    ↓
    Hyperbolic Contrastive Alignment (Poincaré Ball)
                    ↓
    logmap₀ → Tangent Space → MLP Classifier → logits
                    ↓
    softmax(logits) + symbolic_evidence → NeuroSymbolicLoss

Total Loss:
    L = L_CE + λ_contrast * L_contrastive + λ_logic * L_logic
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .poincare_ops import logmap0
from .hyperbolic_aligner import (
    HyperbolicContrastiveAligner,
)
from ..losses.neurosymbolic_loss import NeuroSymbolicLoss


class SymAlignClassifier(nn.Module):
    """
    SymAlign-UAST: GNN + Hyperbolic Alignment + Symbolic Logic Losses.

    Extends HyperbolicAlignedClassifier with neuro-symbolic loss pathway.
    """

    def __init__(
        self,
        gnn_encoder: nn.Module,
        aligner: HyperbolicContrastiveAligner,
        neurosymbolic_loss: NeuroSymbolicLoss,
        num_classes: int,
        projection_dim: int = 128,
        lambda_contrast: float = 0.5,
        lambda_logic: float = 0.3,
    ):
        super().__init__()
        self.gnn_encoder = gnn_encoder
        self.aligner = aligner
        self.neurosymbolic_loss = neurosymbolic_loss
        self.lambda_contrast = lambda_contrast
        self.lambda_logic = lambda_logic

        # Classifier in tangent space (same arch as HyperbolicAlignedClassifier)
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(projection_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        semantic_emb: torch.Tensor,
        symbolic_evidence: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Full forward pass.

        Args:
            x: node features [total_nodes, 6]
            edge_index: graph edges [2, total_edges]
            batch: node-to-graph assignment
            semantic_emb: [B, 768] library semantic embeddings
            symbolic_evidence: [B, 14] pre-computed evidence vectors

        Returns:
            logits: [B, num_classes] raw classification logits
            hyp_graph: [B, projection_dim] hyperbolic graph embedding
            contrast_loss: scalar hyperbolic contrastive loss
            logic_loss: scalar neuro-symbolic logic loss
            logic_components: dict of individual loss values
        """
        # GNN encoding
        graph_emb = self.gnn_encoder(x, edge_index, batch)

        # Hyperbolic contrastive loss
        contrast_loss, _ = self.aligner(graph_emb, semantic_emb)

        # Project to Poincaré ball, then back to tangent space for classification
        hyp_graph = self.aligner.project_graph(graph_emb)
        tangent_emb = logmap0(hyp_graph, self.aligner.curvature)
        logits = self.classifier(tangent_emb)

        # Symbolic logic loss on softmax probabilities
        pred_probs = F.softmax(logits, dim=-1)
        logic_loss, logic_components = self.neurosymbolic_loss(
            pred_probs, symbolic_evidence
        )

        return logits, hyp_graph, contrast_loss, logic_loss, logic_components

    def compute_total_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        contrast_loss: torch.Tensor,
        logic_loss: torch.Tensor,
        ce_criterion: nn.Module,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the total training loss.

        L_total = L_CE + λ_contrast * L_contrastive + λ_logic * L_logic
        """
        ce_loss = ce_criterion(logits, labels)
        total = (
            ce_loss
            + self.lambda_contrast * contrast_loss
            + self.lambda_logic * logic_loss
        )

        loss_dict = {
            "ce_loss": ce_loss.item(),
            "contrast_loss": contrast_loss.item(),
            "logic_loss": logic_loss.item(),
            "total_loss": total.item(),
        }

        return total, loss_dict

    def get_aligned_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        semantic_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get aligned embeddings on Poincaré ball (for zero-shot eval)."""
        graph_emb = self.gnn_encoder(x, edge_index, batch)
        return self.aligner.get_aligned_embeddings(graph_emb, semantic_emb)
