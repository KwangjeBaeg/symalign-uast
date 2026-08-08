"""
Symbolic Evidence Extractor for UAST files.

Extracts a 14-dimensional evidence vector per UAST file:
  - behavioral_vector[7]: binary activation of 7 abstract behavioral categories
  - structural_stats[7]: normalized node-type statistics

The evidence is pre-computed and cached before training, then attached to
PyG Data objects as `data.symbolic_evidence`.
"""

import json
import os
import pickle
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

from ..knowledge.symbolic_taxonomy import (
    BEHAVIORAL_CATEGORIES,
    NUM_BEHAVIORAL_CATEGORIES,
    COMPILED_PATTERNS,
)
from .subtree_extractor import UAST_NODE_TYPES


NUM_STRUCTURAL_FEATURES = 7
EVIDENCE_DIM = NUM_BEHAVIORAL_CATEGORIES + NUM_STRUCTURAL_FEATURES  # 14


class SymbolicEvidenceExtractor:
    """
    Extracts symbolic evidence vectors from raw UAST JSON files.

    For each file produces a 14-dim vector:
      [0:7]  - behavioral categories (binary: 1 if pattern detected)
      [7:14] - structural statistics (normalized to [0,1])
    """

    def extract_from_uast(self, uast_data: dict) -> np.ndarray:
        """
        Extract symbolic evidence from a single UAST JSON.

        Args:
            uast_data: Parsed UAST JSON with 'nodes' and 'edges'

        Returns:
            evidence: [14] float32 numpy array
        """
        nodes = uast_data.get("nodes", {})
        total_nodes = len(nodes)
        if total_nodes == 0:
            return np.zeros(EVIDENCE_DIM, dtype=np.float32)

        # --- Behavioral evidence ---
        behavioral = np.zeros(NUM_BEHAVIORAL_CATEGORIES, dtype=np.float32)
        # --- Structural counts ---
        type_counts = {t: 0 for t in UAST_NODE_TYPES}
        total_children = 0

        for node_id_str, node in nodes.items():
            node_type = node.get("node_type", "")
            code_snippet = node.get("code_snippet", "")
            original_code = node.get("metadata", {}).get("original_code", "")

            # Count node types
            if node_type in type_counts:
                type_counts[node_type] += 1

            # Count children for avg_children stat
            total_children += len(node.get("children", []))

            # Match behavioral categories
            if node_type == "uast_call":
                text = f"{code_snippet} {original_code}"
                for i, category in enumerate(BEHAVIORAL_CATEGORIES):
                    if behavioral[i] == 0:  # skip if already activated
                        pattern = COMPILED_PATTERNS[category]["call_pattern"]
                        if pattern.search(text):
                            behavioral[i] = 1.0

            elif node_type == "uast_package":
                text = original_code or code_snippet
                for i, category in enumerate(BEHAVIORAL_CATEGORIES):
                    if behavioral[i] == 0:
                        pattern = COMPILED_PATTERNS[category]["pkg_pattern"]
                        if pattern.search(text):
                            behavioral[i] = 1.0

        # --- Structural statistics (normalized) ---
        func_count = type_counts.get("uast_function", 0)
        class_count = type_counts.get("uast_class", 0)
        call_count = type_counts.get("uast_call", 0)
        pkg_count = type_counts.get("uast_package", 0)
        comment_count = type_counts.get("uast_comment", 0)

        func_ratio = func_count / total_nodes
        class_ratio = class_count / total_nodes
        call_ratio = call_count / total_nodes
        pkg_ratio = pkg_count / total_nodes
        comment_ratio = comment_count / total_nodes
        call_to_func_ratio = (call_count / func_count) if func_count > 0 else 0.0
        avg_children = total_children / total_nodes

        # Normalize call_to_func_ratio and avg_children to [0,1] with soft cap
        call_to_func_ratio = min(call_to_func_ratio / 50.0, 1.0)
        avg_children = min(avg_children / 10.0, 1.0)

        structural = np.array([
            func_ratio,
            class_ratio,
            call_ratio,
            pkg_ratio,
            comment_ratio,
            call_to_func_ratio,
            avg_children,
        ], dtype=np.float32)

        return np.concatenate([behavioral, structural])

    def process_dataset(
        self,
        dataset: list,
        uast_dir: str,
        cache_path: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Extract symbolic evidence for an entire dataset.

        Args:
            dataset: List of dicts from UASTDataLoader (must have 'file_id')
            uast_dir: Path to Conversion/UAST_files/
            cache_path: Optional path to cache results (pickle)

        Returns:
            Dict mapping file_id -> evidence vector [14]
        """
        if cache_path and os.path.exists(cache_path):
            print(f"Loading cached symbolic evidence from {cache_path}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        evidence_map: Dict[str, np.ndarray] = {}
        skipped = 0

        for item in tqdm(dataset, desc="Extracting symbolic evidence"):
            file_id = item["file_id"]
            uast_path = os.path.join(uast_dir, f"UAST_{file_id}.json")

            if not os.path.exists(uast_path):
                skipped += 1
                continue

            try:
                with open(uast_path) as f:
                    uast_data = json.load(f)
                evidence_map[file_id] = self.extract_from_uast(uast_data)
            except Exception:
                skipped += 1
                continue

        print(f"Extracted evidence for {len(evidence_map)} files (skipped {skipped})")

        # Print statistics
        if evidence_map:
            all_evidence = np.stack(list(evidence_map.values()))
            print("\nBehavioral category activation rates:")
            for i, cat in enumerate(BEHAVIORAL_CATEGORIES):
                rate = all_evidence[:, i].mean()
                print(f"  {cat:20s}: {rate:.1%}")
            print("\nStructural statistics (mean):")
            stat_names = [
                "func_ratio", "class_ratio", "call_ratio",
                "pkg_ratio", "comment_ratio", "call_to_func",
                "avg_children",
            ]
            for i, name in enumerate(stat_names):
                val = all_evidence[:, NUM_BEHAVIORAL_CATEGORIES + i].mean()
                print(f"  {name:20s}: {val:.4f}")

        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(evidence_map, f)
            print(f"Cached symbolic evidence to {cache_path}")

        return evidence_map
