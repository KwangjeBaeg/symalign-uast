"""
Hard Post-Hoc Correction for Experiment 03 (S3 variant).

Inference-time rule-based veto:
  1. Behavior veto: if top-1 class has low behavior match, swap to next valid
  2. Exclusion veto: if top-2 are mutually exclusive, drop the second

Training uses S1 (current symbolic loss). Only inference differs.
"""

from typing import Dict, Optional

import numpy as np
import torch


def apply_hard_posthoc(
    logits: torch.Tensor,
    symbolic_evidence: torch.Tensor,
    domain_behavior_map: torch.Tensor,
    exclusion_matrix: torch.Tensor,
    tau_b: float = 0.5,
) -> torch.Tensor:
    """
    Apply hard post-hoc correction to model predictions.

    Args:
        logits: [B, C] raw logits
        symbolic_evidence: [B, 14] (7 behavioral + 7 structural)
        domain_behavior_map: [C, 7] binary behavior expectations
        exclusion_matrix: [C, C] symmetric binary exclusion
        tau_b: behavior match threshold for veto

    Returns:
        corrected_logits: [B, C] with vetoed classes set to -inf
    """
    num_behav = domain_behavior_map.shape[1]
    behavioral = symbolic_evidence[:, :num_behav]  # [B, 7]

    # Compute behavior match score for all classes: [B, C]
    # behavior_match[b, d] = behavioral[b] · M[d] / sum(M[d])
    expected_counts = domain_behavior_map.sum(dim=1).clamp(min=1.0)  # [C]
    behavior_match = behavioral @ domain_behavior_map.T  # [B, C]
    behavior_match = behavior_match / expected_counts.unsqueeze(0)

    corrected = logits.clone()
    B, C = logits.shape

    for b in range(B):
        ranking = torch.argsort(corrected[b], descending=True)

        # --- Behavior veto ---
        top1 = ranking[0].item()
        if behavior_match[b, top1] < tau_b:
            # Find first rank that passes threshold
            for r in range(1, C):
                candidate = ranking[r].item()
                if behavior_match[b, candidate] >= tau_b:
                    # Veto top-1: set its logit below candidate
                    corrected[b, top1] = corrected[b, candidate] - 1.0
                    break

        # --- Exclusion veto ---
        # Re-rank after behavior veto
        ranking = torch.argsort(corrected[b], descending=True)
        top1 = ranking[0].item()
        top2 = ranking[1].item()

        if exclusion_matrix[top1, top2] > 0:
            # Drop top-2 (keep top-1, demote top-2)
            corrected[b, top2] = corrected[b].min() - 1.0

    return corrected


def apply_hard_posthoc_batch(
    logits: np.ndarray,
    symbolic_evidence: np.ndarray,
    domain_behavior_map: np.ndarray,
    exclusion_matrix: np.ndarray,
    tau_b: float = 0.5,
) -> np.ndarray:
    """NumPy version for evaluation pipeline (non-differentiable)."""
    corrected = apply_hard_posthoc(
        torch.tensor(logits, dtype=torch.float32),
        torch.tensor(symbolic_evidence, dtype=torch.float32),
        torch.tensor(domain_behavior_map, dtype=torch.float32),
        torch.tensor(exclusion_matrix, dtype=torch.float32),
        tau_b=tau_b,
    )
    return corrected.numpy()
