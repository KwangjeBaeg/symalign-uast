"""
Neuro-Symbolic Logic Loss for SymAlign-UAST.

Three differentiable symbolic logic losses using Lukasiewicz T-norm
and Reichenbach Implication to enforce logical consistency between
GNN predictions and symbolic evidence extracted from UAST.

Loss 1 (L_struct): Structural Integrity
  - "High-confidence predictions must be supported by structural evidence"
  - Penalizes max(0, P(d) - structural_support)

Loss 2 (L_behav): Behavioral Consistency
  - "Predicted domain must match observed behavioral patterns"
  - Penalizes max(0, P(d) - behavior_match(d))
  - behavior_match = evidence @ domain_behavior_map.T / expected_count

Loss 3 (L_excl): Domain Exclusion
  - "Mutually exclusive domains should not be co-predicted"
  - Penalizes P(d_i) * P(d_j) for exclusive pairs (i,j)

All losses are differentiable and operate on softmax probabilities.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn

from ..knowledge.symbolic_taxonomy import NUM_BEHAVIORAL_CATEGORIES
from ..data.symbolic_evidence import NUM_STRUCTURAL_FEATURES


class NeuroSymbolicLoss(nn.Module):
    """
    Three differentiable symbolic logic losses.

    Args:
        domain_behavior_map: [num_classes, 7] binary matrix of expected
            behavioral categories per domain
        exclusion_matrix: [num_classes, num_classes] symmetric binary matrix
            where 1 = mutually exclusive domain pair
        lambda_struct: weight for structural integrity loss
        lambda_behav: weight for behavioral consistency loss
        lambda_excl: weight for domain exclusion loss
    """

    def __init__(
        self,
        domain_behavior_map: torch.Tensor,
        exclusion_matrix: torch.Tensor,
        lambda_struct: float = 1.0,
        lambda_behav: float = 1.0,
        lambda_excl: float = 1.0,
    ):
        super().__init__()
        self.register_buffer("domain_behavior_map", domain_behavior_map)
        self.register_buffer("exclusion_matrix", exclusion_matrix)
        self.lambda_struct = lambda_struct
        self.lambda_behav = lambda_behav
        self.lambda_excl = lambda_excl

        # Pre-compute expected behavior counts per domain (for normalization)
        expected_counts = domain_behavior_map.sum(dim=1).clamp(min=1.0)
        self.register_buffer("expected_counts", expected_counts)

        # Upper-triangular mask for exclusion (avoid double-counting)
        num_classes = exclusion_matrix.size(0)
        triu_mask = torch.triu(torch.ones(num_classes, num_classes), diagonal=1)
        excl_triu = exclusion_matrix * triu_mask
        self.register_buffer("excl_triu", excl_triu)

    def forward(
        self,
        pred_probs: torch.Tensor,
        symbolic_evidence: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the 3 symbolic logic losses.

        Args:
            pred_probs: [B, num_classes] softmax probabilities
            symbolic_evidence: [B, 14] (7 behavioral + 7 structural)

        Returns:
            total_loss: scalar (weighted sum of 3 losses)
            components: dict with individual loss values for logging
        """
        behavioral = symbolic_evidence[:, :NUM_BEHAVIORAL_CATEGORIES]  # [B, 7]
        structural = symbolic_evidence[:, NUM_BEHAVIORAL_CATEGORIES:]  # [B, 7]

        # =====================================================================
        # Rule 1: Structural Integrity (L_struct)
        # =====================================================================
        # "If the model predicts domain d with high probability, there should
        #  be sufficient structural complexity in the code to support it."
        #
        # structural_support is the mean of normalized structural stats,
        # representing overall code complexity. If a prediction is confident
        # but the code has low structural complexity, penalize.
        #
        # Lukasiewicz implication: P(d) → S ≈ min(1, 1 - P(d) + S)
        # Violation = max(0, P(d) - S)

        structural_support = structural.mean(dim=1, keepdim=True)  # [B, 1]
        structural_support = structural_support.expand_as(pred_probs)  # [B, C]
        l_struct = torch.clamp(pred_probs - structural_support, min=0).mean()

        # =====================================================================
        # Rule 2: Behavioral Consistency (L_behav)
        # =====================================================================
        # "If the model predicts domain d, the behavioral patterns expected
        #  for that domain should be present in the code's evidence."
        #
        # behavior_match[b, d] = (behavioral[b] · domain_behavior_map[d]) /
        #                         expected_count[d]
        # This is the fraction of expected behaviors actually observed.
        #
        # Violation = max(0, P(d) - behavior_match(d))

        # [B, C] = [B, 7] @ [7, C]
        behavior_match = behavioral @ self.domain_behavior_map.T
        # Normalize by expected count per domain: [B, C] / [1, C]
        behavior_match = behavior_match / self.expected_counts.unsqueeze(0)
        # Clamp to [0, 1]
        behavior_match = behavior_match.clamp(0.0, 1.0)

        l_behav = torch.clamp(pred_probs - behavior_match, min=0).mean()

        # =====================================================================
        # Rule 3: Domain Exclusion (L_excl)
        # =====================================================================
        # "Mutually exclusive domains should not be co-predicted."
        #
        # For each exclusive pair (i, j):
        #   penalty = P(d_i) * P(d_j)
        # This is the product T-norm: T(a, b) = a * b

        # [B, C, 1] * [B, 1, C] = [B, C, C]
        pred_outer = pred_probs.unsqueeze(2) * pred_probs.unsqueeze(1)
        # Apply upper-triangular exclusion mask [1, C, C]
        l_excl = (pred_outer * self.excl_triu.unsqueeze(0)).sum(dim=(1, 2)).mean()

        # =====================================================================
        # Total
        # =====================================================================
        total = (
            self.lambda_struct * l_struct
            + self.lambda_behav * l_behav
            + self.lambda_excl * l_excl
        )

        components = {
            "l_struct": l_struct.item(),
            "l_behav": l_behav.item(),
            "l_excl": l_excl.item(),
            "l_logic_total": total.item(),
        }

        return total, components
