"""
Ontology Corruption for Experiment 03.

S4: Shuffled-Ontology — random permute behavior map rows + resample exclusion pairs
S5: Noisy-Ontology — flip-direction stratified corruption of behavior mask and exclusion pairs
"""

import copy
from typing import List, Tuple

import numpy as np
import torch


def shuffle_ontology(
    domain_behavior_map: torch.Tensor,
    exclusion_matrix: torch.Tensor,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    S4: Shuffled ontology — preserves density but randomizes assignment.

    - Behavior map: permute rows (domain labels) randomly
    - Exclusion matrix: resample same number of pairs randomly

    This preserves the same mask density and pair count, so any effect
    from S1 that doesn't appear in S4 is attributable to *correct* ontology,
    not just to having a regularization penalty.
    """
    rng = np.random.default_rng(seed)

    num_classes = domain_behavior_map.shape[0]

    # Shuffle behavior map rows
    perm = rng.permutation(num_classes)
    shuffled_behavior = domain_behavior_map[perm].clone()

    # Resample exclusion pairs with same count
    orig_excl = exclusion_matrix.numpy()
    num_pairs = int(np.triu(orig_excl, k=1).sum())

    shuffled_excl = np.zeros_like(orig_excl)
    all_pairs = [(i, j) for i in range(num_classes) for j in range(i + 1, num_classes)]
    selected = rng.choice(len(all_pairs), size=num_pairs, replace=False)
    for idx in selected:
        i, j = all_pairs[idx]
        shuffled_excl[i, j] = 1.0
        shuffled_excl[j, i] = 1.0

    return shuffled_behavior, torch.tensor(shuffled_excl, dtype=torch.float32)


def corrupt_ontology(
    domain_behavior_map: torch.Tensor,
    exclusion_matrix: torch.Tensor,
    flip_1to0_ratio: float = 0.2,
    flip_0to1_ratio: float = 0.2,
    excl_add_ratio: float = 0.2,
    excl_drop_ratio: float = 0.2,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    S5: Noisy ontology — flip-direction stratified corruption.

    Args:
        flip_1to0_ratio: fraction of 1-entries in behavior map to flip to 0
        flip_0to1_ratio: fraction of 0-entries in behavior map to flip to 1
        excl_add_ratio: fraction of non-exclusive pairs to add as exclusive
        excl_drop_ratio: fraction of existing exclusive pairs to drop
        seed: random seed for reproducibility
    """
    rng = np.random.default_rng(seed)

    # --- Behavior map corruption ---
    bmap = domain_behavior_map.clone().numpy()

    # Flip 1→0
    ones_idx = np.argwhere(bmap == 1.0)
    n_flip_1to0 = int(len(ones_idx) * flip_1to0_ratio)
    if n_flip_1to0 > 0:
        chosen = rng.choice(len(ones_idx), size=n_flip_1to0, replace=False)
        for idx in chosen:
            r, c = ones_idx[idx]
            bmap[r, c] = 0.0

    # Flip 0→1
    zeros_idx = np.argwhere(bmap == 0.0)
    n_flip_0to1 = int(len(zeros_idx) * flip_0to1_ratio)
    if n_flip_0to1 > 0:
        chosen = rng.choice(len(zeros_idx), size=n_flip_0to1, replace=False)
        for idx in chosen:
            r, c = zeros_idx[idx]
            bmap[r, c] = 1.0

    # --- Exclusion matrix corruption ---
    excl = exclusion_matrix.clone().numpy()
    num_classes = excl.shape[0]

    # Get current pairs (upper triangle)
    current_pairs = []
    non_pairs = []
    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            if excl[i, j] == 1.0:
                current_pairs.append((i, j))
            else:
                non_pairs.append((i, j))

    # Drop existing pairs
    n_drop = int(len(current_pairs) * excl_drop_ratio)
    if n_drop > 0 and current_pairs:
        drop_idx = rng.choice(len(current_pairs), size=min(n_drop, len(current_pairs)), replace=False)
        for idx in drop_idx:
            i, j = current_pairs[idx]
            excl[i, j] = 0.0
            excl[j, i] = 0.0

    # Add new pairs
    n_add = int(len(non_pairs) * excl_add_ratio)
    if n_add > 0 and non_pairs:
        add_idx = rng.choice(len(non_pairs), size=min(n_add, len(non_pairs)), replace=False)
        for idx in add_idx:
            i, j = non_pairs[idx]
            excl[i, j] = 1.0
            excl[j, i] = 1.0

    return torch.tensor(bmap, dtype=torch.float32), torch.tensor(excl, dtype=torch.float32)


def get_corrupted_ontology(
    variant: str,
    domain_behavior_map: torch.Tensor,
    exclusion_matrix: torch.Tensor,
    noise_level: float = 0.2,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get ontology tensors for a given variant.

    Returns (behavior_map, exclusion_matrix) — possibly corrupted.
    """
    if variant == "S4":
        return shuffle_ontology(domain_behavior_map, exclusion_matrix, seed=seed)

    if variant.startswith("S5"):
        # Parse noise level from variant name like "S5_20"
        return corrupt_ontology(
            domain_behavior_map, exclusion_matrix,
            flip_1to0_ratio=noise_level,
            flip_0to1_ratio=noise_level,
            excl_add_ratio=noise_level,
            excl_drop_ratio=noise_level,
            seed=seed,
        )

    # S0, S1, S2, S3, S6_* — use original ontology
    return domain_behavior_map.clone(), exclusion_matrix.clone()
