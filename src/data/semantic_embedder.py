"""
Phase 3 - Step 1: LLM Semantic Embeddings

Generates semantic embeddings for library concepts using Sentence-BERT.
Maps library descriptions to 768-dim vectors in semantic space.
"""

import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch


class LibrarySemanticEmbedder:
    """
    Embeds library concept descriptions into semantic vectors
    using Sentence-BERT (all-mpnet-base-v2).

    Output: 768-dim embeddings for each library.
    """

    def __init__(self, model_name: str = "all-mpnet-base-v2", device: str = "cuda"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)
        self.device = device
        self.library_embeddings = {}

    def create_description(self, concept: dict) -> str:
        """
        Create a natural language description from library concept dict.

        Example output:
        "numpy is a library for Array Processing in the Scientific Computing domain.
         It is an External dependency that works on Cross-platform."
        """
        name = concept.get("Library Name", "unknown")
        functionality = concept.get("Functionality") or "general purpose"
        domain = concept.get("Domain") or "general"
        dep_type = concept.get("Dependency Type") or "external"
        platform = concept.get("Platform") or "cross-platform"
        framework = concept.get("Framework") or "none"

        parts = [f"{name} is a library for {functionality} in the {domain} domain."]

        if dep_type and dep_type != "None":
            parts.append(f"It is a {dep_type} dependency.")

        if platform and platform != "None":
            parts.append(f"It works on {platform}.")

        if framework and framework not in ["None", "none"]:
            parts.append(f"It is part of the {framework} framework.")

        return " ".join(parts)

    def embed_libraries(
        self,
        library_concepts: List[dict],
        batch_size: int = 32,
    ) -> Dict[str, np.ndarray]:
        """
        Batch-embed all library concept descriptions.

        Args:
            library_concepts: List of concept dicts

        Returns:
            Dict mapping library name -> 768-dim embedding
        """
        descriptions = []
        lib_names = []

        for concept in library_concepts:
            name = concept.get("Library Name")
            if not name:
                continue
            desc = self.create_description(concept)
            descriptions.append(desc)
            lib_names.append(name)

        if not descriptions:
            return {}

        print(f"Embedding {len(descriptions)} library descriptions...")
        embeddings = self.model.encode(
            descriptions,
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=batch_size,
        )

        self.library_embeddings = {
            name: emb for name, emb in zip(lib_names, embeddings)
        }

        print(f"Generated embeddings: {embeddings.shape}")
        return self.library_embeddings

    def embed_domains(
        self,
        domains: List[str],
    ) -> Dict[str, np.ndarray]:
        """
        Embed domain descriptions for alignment targets.
        """
        domain_descriptions = {
            "Scientific Computing": "Scientific computing, numerical analysis, mathematical computation",
            "Quantum Computing": "Quantum computing, quantum circuits, quantum algorithms",
            "Data Processing": "Data processing, data manipulation, data transformation",
            "Systems Programming": "Systems programming, operating systems, low-level computing",
            "Testing": "Software testing, unit testing, test automation",
            "Security": "Computer security, cryptography, authentication",
            "DevOps": "DevOps, deployment, continuous integration, infrastructure",
            "Networking": "Computer networking, HTTP, sockets, network protocols",
            "Web Development": "Web development, web frameworks, HTTP servers",
            "Web": "Web development, web applications",
            "Machine Learning": "Machine learning, deep learning, artificial intelligence",
            "GUI": "Graphical user interface, desktop applications",
            "Cloud Computing": "Cloud computing, cloud services, distributed systems",
            "Type Checking": "Type checking, static analysis, type annotations",
            "Science": "Scientific research, scientific computing",
            "Unknown": "General purpose programming, miscellaneous",
        }

        descriptions = []
        domain_names = []
        for d in domains:
            desc = domain_descriptions.get(d, f"{d} programming domain")
            descriptions.append(desc)
            domain_names.append(d)

        embeddings = self.model.encode(
            descriptions,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

        return {name: emb for name, emb in zip(domain_names, embeddings)}

    def get_embedding(self, library_name: str) -> Optional[np.ndarray]:
        """Get embedding for a single library."""
        return self.library_embeddings.get(library_name)

    def get_embedding_matrix(self, library_names: List[str]) -> np.ndarray:
        """
        Get embedding matrix for a list of libraries.
        Returns zeros for unknown libraries.
        """
        dim = 768
        matrix = np.zeros((len(library_names), dim), dtype=np.float32)
        for i, name in enumerate(library_names):
            emb = self.library_embeddings.get(name)
            if emb is not None:
                matrix[i] = emb
        return matrix

    def save_embeddings(self, path: str):
        """Save embeddings to npz file."""
        names = list(self.library_embeddings.keys())
        embeddings = np.stack([self.library_embeddings[n] for n in names])
        np.savez(path, names=names, embeddings=embeddings)
        print(f"Saved {len(names)} embeddings to {path}")

    def load_embeddings(self, path: str):
        """Load embeddings from npz file."""
        data = np.load(path, allow_pickle=True)
        names = data["names"]
        embeddings = data["embeddings"]
        self.library_embeddings = {
            str(n): emb for n, emb in zip(names, embeddings)
        }
        print(f"Loaded {len(self.library_embeddings)} embeddings from {path}")
