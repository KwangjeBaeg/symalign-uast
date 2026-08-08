"""
Phase 2 - Step 1: UAST Subtree Extraction

Extracts library-specific subtrees from full UAST graphs.
Each subtree captures the k-hop neighborhood around library-related nodes
(imports, function calls, etc.)

Converts UAST subtrees to PyTorch Geometric Data objects for GNN training.
"""

import json
import os
import pickle
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm


# All known UAST node types (discovered from data)
UAST_NODE_TYPES = [
    "uast_root",
    "uast_function",
    "uast_class",
    "uast_call",
    "uast_package",
    "uast_comment",
]

NODE_TYPE_TO_IDX = {t: i for i, t in enumerate(UAST_NODE_TYPES)}


class UASTSubtreeExtractor:
    """
    Extracts library-specific subtrees from UAST JSON files.

    For each file, identifies nodes related to library imports and calls,
    then extracts their k-hop neighborhood as a subgraph.
    """

    def __init__(self, k_hop: int = 3):
        self.k_hop = k_hop

    def load_uast_json(self, uast_path: str) -> dict:
        """Load raw UAST JSON (nodes + edges)."""
        with open(uast_path) as f:
            return json.load(f)

    def find_library_nodes(self, uast_data: dict, library: str) -> List[int]:
        """
        Find all nodes in the UAST that reference a given library.

        Searches:
        - uast_package nodes (import statements)
        - uast_call nodes (function calls using the library)
        """
        seed_nodes = []

        for node_id_str, node in uast_data["nodes"].items():
            node_id = int(node_id_str)
            node_type = node.get("node_type", "")
            code_snippet = node.get("code_snippet", "")
            original_code = node.get("metadata", {}).get("original_code", "")

            # Check import statements
            if node_type == "uast_package":
                if library in original_code or library in code_snippet:
                    seed_nodes.append(node_id)

            # Check function calls
            elif node_type == "uast_call":
                if library in code_snippet or library in original_code:
                    seed_nodes.append(node_id)

        return seed_nodes

    def build_adjacency(self, uast_data: dict) -> Dict[int, List[int]]:
        """Build bidirectional adjacency list from edges."""
        adj = defaultdict(list)
        for edge in uast_data["edges"]:
            s, e = edge["start_id"], edge["end_id"]
            adj[s].append(e)
            adj[e].append(s)
        return adj

    def extract_khop_subgraph(
        self,
        uast_data: dict,
        seed_nodes: List[int],
        k: int,
    ) -> Tuple[List[int], List[Tuple[int, int]]]:
        """
        Extract k-hop neighborhood subgraph around seed nodes.

        Returns:
            subgraph_nodes: List of node IDs in the subgraph
            subgraph_edges: List of (src, dst) edges in the subgraph
        """
        if not seed_nodes:
            return [], []

        adj = self.build_adjacency(uast_data)

        # BFS to find k-hop neighborhood
        visited = set(seed_nodes)
        queue = deque([(n, 0) for n in seed_nodes])

        while queue:
            node_id, depth = queue.popleft()
            if depth >= k:
                continue
            for neighbor in adj.get(node_id, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))

        subgraph_nodes = sorted(visited)
        node_set = set(subgraph_nodes)

        # Extract edges within subgraph
        subgraph_edges = []
        for edge in uast_data["edges"]:
            s, e = edge["start_id"], edge["end_id"]
            if s in node_set and e in node_set:
                subgraph_edges.append((s, e))

        return subgraph_nodes, subgraph_edges

    def extract_full_graph(self, uast_data: dict) -> Tuple[List[int], List[Tuple[int, int]]]:
        """Extract the entire UAST graph (no subtree filtering)."""
        nodes = sorted(int(k) for k in uast_data["nodes"].keys())
        edges = [(e["start_id"], e["end_id"]) for e in uast_data["edges"]]
        return nodes, edges

    def encode_node_features(self, uast_data: dict, node_ids: List[int]) -> np.ndarray:
        """
        Encode node features as one-hot vectors of node type.

        Args:
            uast_data: Raw UAST JSON
            node_ids: List of node IDs to encode

        Returns:
            features: [num_nodes, num_node_types] numpy array
        """
        num_types = len(UAST_NODE_TYPES)
        features = np.zeros((len(node_ids), num_types), dtype=np.float32)

        for i, nid in enumerate(node_ids):
            node = uast_data["nodes"].get(str(nid), {})
            node_type = node.get("node_type", "")
            if node_type in NODE_TYPE_TO_IDX:
                features[i, NODE_TYPE_TO_IDX[node_type]] = 1.0
            else:
                # Unknown node type - use all zeros (will be handled by GNN)
                pass

        return features

    def uast_to_graph_data(
        self,
        uast_data: dict,
        node_ids: List[int],
        edges: List[Tuple[int, int]],
    ) -> dict:
        """
        Convert UAST subgraph to a dict suitable for PyTorch Geometric.

        Returns dict with:
            x: node features [num_nodes, num_features]
            edge_index: [2, num_edges]
            num_nodes: int
        """
        if not node_ids:
            return None

        # Create node ID mapping (original -> 0-indexed)
        id_map = {nid: i for i, nid in enumerate(node_ids)}

        # Encode features
        x = self.encode_node_features(uast_data, node_ids)

        # Remap edges
        edge_index = np.array(
            [[id_map[s], id_map[e]] for s, e in edges if s in id_map and e in id_map],
            dtype=np.int64,
        )

        if len(edge_index) == 0:
            edge_index = np.zeros((0, 2), dtype=np.int64)

        return {
            "x": x,
            "edge_index": edge_index.T if len(edge_index) > 0 else np.zeros((2, 0), dtype=np.int64),
            "num_nodes": len(node_ids),
        }

    def process_file(
        self,
        uast_path: str,
        libraries: List[str],
        use_subtree: bool = True,
    ) -> Optional[dict]:
        """
        Process a single UAST file into a graph data dict.

        Args:
            uast_path: Path to UAST_*.json file
            libraries: Libraries found in this file
            use_subtree: If True, extract library-specific subtrees.
                         If False, use full graph.

        Returns:
            graph_data dict or None if the file is too small
        """
        try:
            uast_data = self.load_uast_json(uast_path)
        except Exception:
            return None

        if use_subtree:
            # Collect seed nodes for all libraries
            all_seeds = []
            for lib in libraries:
                all_seeds.extend(self.find_library_nodes(uast_data, lib))

            # If no seeds found, use root children as fallback
            if not all_seeds:
                root = uast_data["nodes"].get("0", {})
                all_seeds = root.get("children", [])[:5]

            node_ids, edges = self.extract_khop_subgraph(
                uast_data, all_seeds, self.k_hop
            )
        else:
            node_ids, edges = self.extract_full_graph(uast_data)

        if len(node_ids) < 2:
            return None

        return self.uast_to_graph_data(uast_data, node_ids, edges)

    def process_dataset(
        self,
        dataset: list,
        uast_dir: str,
        use_subtree: bool = True,
        max_nodes: int = 500,
        cache_path: Optional[str] = None,
    ) -> List[dict]:
        """
        Process entire dataset of UAST files into graph data.

        Args:
            dataset: List of dicts from UASTDataLoader.load_dataset()
            uast_dir: Path to Conversion/UAST_files/
            use_subtree: Whether to use subtree extraction
            max_nodes: Maximum nodes per graph (truncate larger ones)
            cache_path: Optional path to cache processed graphs

        Returns:
            List of graph data dicts with additional 'label' field
        """
        if cache_path and os.path.exists(cache_path):
            print(f"Loading cached graph data from {cache_path}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        graphs = []
        skipped = 0

        for item in tqdm(dataset, desc="Processing UAST graphs"):
            file_id = item["file_id"]
            uast_path = os.path.join(uast_dir, f"UAST_{file_id}.json")

            if not os.path.exists(uast_path):
                skipped += 1
                continue

            graph_data = self.process_file(
                uast_path, item["libraries"], use_subtree
            )

            if graph_data is None:
                skipped += 1
                continue

            # Truncate large graphs
            if graph_data["num_nodes"] > max_nodes:
                graph_data["x"] = graph_data["x"][:max_nodes]
                # Filter edges to only include nodes within truncation
                mask = (graph_data["edge_index"][0] < max_nodes) & (
                    graph_data["edge_index"][1] < max_nodes
                )
                graph_data["edge_index"] = graph_data["edge_index"][:, mask]
                graph_data["num_nodes"] = max_nodes

            # Add metadata
            graph_data["file_id"] = file_id
            graph_data["libraries"] = item["libraries"]

            graphs.append(graph_data)

        print(f"Processed {len(graphs)} graphs (skipped {skipped})")

        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(graphs, f)
            print(f"Cached graph data to {cache_path}")

        return graphs
