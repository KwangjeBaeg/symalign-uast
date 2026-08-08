"""
Domain Ontology Loader for SymAlign-UAST.

Loads pre-defined domain relationships from domain_ontology.json:
  - domain_behavior_map: which behavioral categories each domain expects
  - exclusion_matrix: which domain pairs are mutually exclusive
  - compatibility_groups: which domains tend to co-occur

Converts these to torch tensors aligned with a LabelEncoder's class ordering.
"""

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

from .symbolic_taxonomy import (
    BEHAVIORAL_CATEGORIES,
    NUM_BEHAVIORAL_CATEGORIES,
)


DEFAULT_ONTOLOGY_PATH = os.path.join(
    os.path.dirname(__file__), "domain_ontology.json"
)


class DomainOntology:
    """
    Loads and serves domain ontology as tensors for the symbolic loss.
    """

    def __init__(self, ontology_path: str = DEFAULT_ONTOLOGY_PATH):
        with open(ontology_path) as f:
            self.data = json.load(f)

        self.domains = self.data["domains"]
        self.behavior_map_raw = self.data["domain_behavior_map"]
        self.exclusion_pairs_raw = self.data["exclusion_pairs"]

    def get_domain_behavior_map(
        self, label_encoder: LabelEncoder
    ) -> torch.Tensor:
        """
        Build [num_classes, 7] binary matrix of expected behaviors per domain.

        Rows are ordered according to label_encoder.classes_.
        Domains not in the ontology get all-zeros (no expected behavior).
        """
        num_classes = len(label_encoder.classes_)
        behavior_map = np.zeros(
            (num_classes, NUM_BEHAVIORAL_CATEGORIES), dtype=np.float32
        )

        for idx, domain_name in enumerate(label_encoder.classes_):
            if domain_name in self.behavior_map_raw:
                vec = self.behavior_map_raw[domain_name]
                behavior_map[idx, :len(vec)] = vec

        return torch.tensor(behavior_map, dtype=torch.float32)

    def get_exclusion_matrix(
        self, label_encoder: LabelEncoder
    ) -> torch.Tensor:
        """
        Build [num_classes, num_classes] symmetric binary matrix.
        exclusion_matrix[i,j] = 1 means domains i and j are mutually exclusive.
        """
        num_classes = len(label_encoder.classes_)
        domain_to_idx = {
            name: idx for idx, name in enumerate(label_encoder.classes_)
        }

        matrix = np.zeros((num_classes, num_classes), dtype=np.float32)

        for d1, d2 in self.exclusion_pairs_raw:
            i = domain_to_idx.get(d1)
            j = domain_to_idx.get(d2)
            if i is not None and j is not None:
                matrix[i, j] = 1.0
                matrix[j, i] = 1.0

        return torch.tensor(matrix, dtype=torch.float32)

    def print_summary(self, label_encoder: LabelEncoder):
        """Print ontology summary aligned with label encoder."""
        behavior_map = self.get_domain_behavior_map(label_encoder)
        exclusion_matrix = self.get_exclusion_matrix(label_encoder)

        print("\nDomain-Behavior Map:")
        header = "Domain".ljust(25) + "  ".join(
            cat[:6] for cat in BEHAVIORAL_CATEGORIES
        )
        print(header)
        print("-" * len(header))
        for idx, domain in enumerate(label_encoder.classes_):
            row = domain.ljust(25) + "  ".join(
                f"  {int(v)}" for v in behavior_map[idx]
            )
            print(row)

        num_excl = int(exclusion_matrix.sum() / 2)
        print(f"\nExclusion pairs: {num_excl}")
        for i in range(len(label_encoder.classes_)):
            for j in range(i + 1, len(label_encoder.classes_)):
                if exclusion_matrix[i, j] > 0:
                    print(
                        f"  {label_encoder.classes_[i]} <-> "
                        f"{label_encoder.classes_[j]}"
                    )
