"""
Symbolic Loss Variants for Experiment 03.

Implements:
  - S1 (Current): Product T-norm exclusion + Reichenbach implication (as in paper)
  - S2 (Łukasiewicz-Consistent): All rules under Łukasiewicz semantics
  - Single-rule ablation helpers for S6
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn

from ..knowledge.symbolic_taxonomy import NUM_BEHAVIORAL_CATEGORIES
from ..data.symbolic_evidence import NUM_STRUCTURAL_FEATURES


class CurrentSymbolicLoss(nn.Module):
    """
    S1: Current paper implementation (same as NeuroSymbolicLoss).

    - Structural: max(0, P(d) - structural_support)  [Reichenbach implication]
    - Behavioral: max(0, P(d) - behavior_match(d))  [Reichenbach implication]
    - Exclusion:  P(d_i) * P(d_j)  [Product T-norm]
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

        expected_counts = domain_behavior_map.sum(dim=1).clamp(min=1.0)
        self.register_buffer("expected_counts", expected_counts)

        num_classes = exclusion_matrix.size(0)
        triu_mask = torch.triu(torch.ones(num_classes, num_classes), diagonal=1)
        excl_triu = exclusion_matrix * triu_mask
        self.register_buffer("excl_triu", excl_triu)

    def forward(
        self,
        pred_probs: torch.Tensor,
        symbolic_evidence: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        behavioral = symbolic_evidence[:, :NUM_BEHAVIORAL_CATEGORIES]
        structural = symbolic_evidence[:, NUM_BEHAVIORAL_CATEGORIES:]

        # Rule 1: Structural Integrity
        structural_support = structural.mean(dim=1, keepdim=True).expand_as(pred_probs)
        l_struct = torch.clamp(pred_probs - structural_support, min=0).mean()

        # Rule 2: Behavioral Consistency
        behavior_match = behavioral @ self.domain_behavior_map.T
        behavior_match = behavior_match / self.expected_counts.unsqueeze(0)
        behavior_match = behavior_match.clamp(0.0, 1.0)
        l_behav = torch.clamp(pred_probs - behavior_match, min=0).mean()

        # Rule 3: Domain Exclusion (Product T-norm)
        pred_outer = pred_probs.unsqueeze(2) * pred_probs.unsqueeze(1)
        l_excl = (pred_outer * self.excl_triu.unsqueeze(0)).sum(dim=(1, 2)).mean()

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


class LukasiewiczSymbolicLoss(nn.Module):
    """
    S2: Łukasiewicz-Consistent formulation.

    All three rules derived from Łukasiewicz T-norm/implication:
      T_L(a, b) = max(0, a + b - 1)
      I_L(a, b) = min(1, 1 - a + b)

    - Structural: L = E[1 - I_L(p_y, h_struct)]
                    = E[max(0, p_y - h_struct)]  (same as S1 in this case)
    - Behavioral: L = E[1 - I_L(p_y, min(1, sum_{b: M_yb=1} s_b))]
                    The disjunction uses bounded-sum (Łukasiewicz t-conorm)
    - Exclusion:  L = sum_{(i,j)} T_L(p_i, p_j) = sum max(0, p_i + p_j - 1)
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

        num_classes = exclusion_matrix.size(0)
        triu_mask = torch.triu(torch.ones(num_classes, num_classes), diagonal=1)
        excl_triu = exclusion_matrix * triu_mask
        self.register_buffer("excl_triu", excl_triu)

    def forward(
        self,
        pred_probs: torch.Tensor,
        symbolic_evidence: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        behavioral = symbolic_evidence[:, :NUM_BEHAVIORAL_CATEGORIES]
        structural = symbolic_evidence[:, NUM_BEHAVIORAL_CATEGORIES:]

        # Rule 1: Structural Integrity — I_L(p, s) = min(1, 1 - p + s)
        # Loss = 1 - I_L(p, s) = max(0, p - s)
        structural_support = structural.mean(dim=1, keepdim=True).expand_as(pred_probs)
        l_struct = torch.clamp(pred_probs - structural_support, min=0).mean()

        # Rule 2: Behavioral Consistency — Łukasiewicz disjunction (bounded sum)
        # For each domain d: consequent = min(1, sum_{b: M_db=1} s_b)
        # This is the Łukasiewicz t-conorm (bounded sum) of the matching behaviors
        # Loss = max(0, p_d - consequent_d)
        # [B, C] = [B, 7] @ [C, 7].T = behavioral dot domain_behavior_map
        behavior_sum = behavioral @ self.domain_behavior_map.T  # [B, C]
        # Bounded sum: min(1, sum) — Łukasiewicz t-conorm
        behavior_consequent = torch.clamp(behavior_sum, min=0.0, max=1.0)
        l_behav = torch.clamp(pred_probs - behavior_consequent, min=0).mean()

        # Rule 3: Exclusion — Łukasiewicz T-norm: T_L(a, b) = max(0, a + b - 1)
        # [B, C, 1] + [B, 1, C] - 1 = [B, C, C]
        pred_sum = pred_probs.unsqueeze(2) + pred_probs.unsqueeze(1) - 1.0
        pred_t_norm = torch.clamp(pred_sum, min=0)
        l_excl = (pred_t_norm * self.excl_triu.unsqueeze(0)).sum(dim=(1, 2)).mean()

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


def get_symbolic_loss(
    variant: str,
    domain_behavior_map: torch.Tensor,
    exclusion_matrix: torch.Tensor,
    lambda_struct: float = 1.0,
    lambda_behav: float = 1.0,
    lambda_excl: float = 1.0,
) -> nn.Module:
    """
    Factory function to create the appropriate symbolic loss for each variant.

    Args:
        variant: One of "S0", "S1", "S2", "S3", "S4", "S5",
                 "S6_struct", "S6_behav", "S6_excl"

    Returns:
        Loss module (or None for S0)
    """
    if variant == "S0":
        # No symbolic loss — return a dummy that always returns 0
        return _NoSymbolicLoss()

    if variant in ("S1", "S3", "S4", "S5"):
        # S1 = current, S3 uses S1 for training, S4/S5 use current loss with corrupted ontology
        return CurrentSymbolicLoss(
            domain_behavior_map, exclusion_matrix,
            lambda_struct, lambda_behav, lambda_excl,
        )

    if variant == "S2":
        return LukasiewiczSymbolicLoss(
            domain_behavior_map, exclusion_matrix,
            lambda_struct, lambda_behav, lambda_excl,
        )

    # S6 single-rule ablation
    if variant == "S6_struct":
        return CurrentSymbolicLoss(
            domain_behavior_map, exclusion_matrix,
            lambda_struct=1.0, lambda_behav=0.0, lambda_excl=0.0,
        )
    if variant == "S6_behav":
        return CurrentSymbolicLoss(
            domain_behavior_map, exclusion_matrix,
            lambda_struct=0.0, lambda_behav=1.0, lambda_excl=0.0,
        )
    if variant == "S6_excl":
        return CurrentSymbolicLoss(
            domain_behavior_map, exclusion_matrix,
            lambda_struct=0.0, lambda_behav=0.0, lambda_excl=1.0,
        )

    raise ValueError(f"Unknown symbolic variant: {variant}")


class _NoSymbolicLoss(nn.Module):
    """Dummy loss that always returns zero (for S0 no-symbolic baseline)."""

    def forward(self, pred_probs, symbolic_evidence):
        zero = pred_probs.sum() * 0.0
        return zero, {"l_struct": 0.0, "l_behav": 0.0, "l_excl": 0.0, "l_logic_total": 0.0}
