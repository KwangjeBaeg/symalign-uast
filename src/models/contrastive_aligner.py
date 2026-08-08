"""
Phase 3 - Step 2 & 3: Contrastive Alignment Model + Joint Training

Aligns UAST-GNN structural embeddings with LLM semantic embeddings
using InfoNCE (CLIP-style) contrastive loss.
"""

import json
import os
import time
import pickle
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from torch_geometric.data import Data, DataLoader


class ContrastiveAligner(nn.Module):
    """
    Aligns UAST structure embeddings with LLM semantic embeddings
    via InfoNCE contrastive loss (CLIP-style).
    """

    def __init__(
        self,
        graph_dim: int = 768,
        semantic_dim: int = 768,
        projection_dim: int = 256,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.temperature = temperature

        # Projection heads
        self.graph_proj = nn.Sequential(
            nn.Linear(graph_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, projection_dim),
            nn.LayerNorm(projection_dim),
        )

        self.semantic_proj = nn.Sequential(
            nn.Linear(semantic_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, projection_dim),
            nn.LayerNorm(projection_dim),
        )

    def forward(
        self, graph_emb: torch.Tensor, semantic_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute InfoNCE contrastive loss.

        Args:
            graph_emb: [batch, graph_dim] from UAST-GNN
            semantic_emb: [batch, semantic_dim] from LLM

        Returns:
            loss: scalar InfoNCE loss
            logits: [batch, batch] similarity matrix
        """
        z = F.normalize(self.graph_proj(graph_emb), dim=-1)
        s = F.normalize(self.semantic_proj(semantic_emb), dim=-1)

        # Cosine similarity matrix
        logits = torch.matmul(z, s.T) / self.temperature

        # Labels: diagonal (positive pairs)
        labels = torch.arange(logits.size(0), device=logits.device)

        # Symmetric loss
        loss_z2s = F.cross_entropy(logits, labels)
        loss_s2z = F.cross_entropy(logits.T, labels)
        loss = (loss_z2s + loss_s2z) / 2

        return loss, logits

    def get_aligned_embeddings(
        self, graph_emb: torch.Tensor, semantic_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get projected and normalized embeddings."""
        z = F.normalize(self.graph_proj(graph_emb), dim=-1)
        s = F.normalize(self.semantic_proj(semantic_emb), dim=-1)
        return z, s


class AlignedClassifier(nn.Module):
    """
    End-to-end model: UAST-GNN + Contrastive Alignment + Classification.

    Jointly optimizes:
    1. Classification loss (cross-entropy)
    2. Contrastive alignment loss (InfoNCE)
    """

    def __init__(
        self,
        gnn_encoder: nn.Module,
        aligner: ContrastiveAligner,
        num_classes: int,
        projection_dim: int = 256,
        lambda_contrast: float = 0.5,
    ):
        super().__init__()
        self.gnn_encoder = gnn_encoder
        self.aligner = aligner
        self.lambda_contrast = lambda_contrast

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            logits: classification logits
            contrast_loss: InfoNCE loss
            aligned_graph_emb: projected graph embeddings
        """
        # GNN encoding
        graph_emb = self.gnn_encoder(x, edge_index, batch)

        # Contrastive alignment
        contrast_loss, _ = self.aligner(graph_emb, semantic_emb)

        # Classification on aligned graph embeddings
        aligned_graph, _ = self.aligner.get_aligned_embeddings(graph_emb, semantic_emb)
        logits = self.classifier(aligned_graph)

        return logits, contrast_loss, aligned_graph

    def get_aligned_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        semantic_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get aligned embeddings for downstream tasks."""
        graph_emb = self.gnn_encoder(x, edge_index, batch)
        return self.aligner.get_aligned_embeddings(graph_emb, semantic_emb)


def run_phase3_experiment(
    dataset: list,
    library_to_concept: dict,
    library_concepts: list,
    uast_dir: str,
    results_dir: str,
    phase2_model_path: str,
    phase2_label_encoder_path: str,
    device: str = "cuda",
    num_epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-4,
    lambda_contrast: float = 0.5,
    temperature: float = 0.07,
    projection_dim: int = 256,
):
    """
    Run complete Phase 3 contrastive alignment experiment.
    """
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 80)
    print("PHASE 3: Contrastive Structure-Semantic Alignment")
    print("=" * 80)

    # Step 1: Load Phase 2 artifacts
    print("\n--- Step 1: Loading Phase 2 Artifacts ---")
    from experiments.phase2.gnn_encoder import UASTGraphEncoder, UASTClassifier
    from experiments.phase2.subtree_extractor import UASTSubtreeExtractor, UAST_NODE_TYPES
    from experiments.phase2.train_gnn import prepare_labels, graphs_to_pyg

    with open(phase2_label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    num_classes = len(label_encoder.classes_)
    node_feature_dim = len(UAST_NODE_TYPES)

    # Load GNN encoder from Phase 2
    phase2_model = UASTClassifier(
        num_classes=num_classes,
        node_feature_dim=node_feature_dim,
        hidden_dim=256,
        embedding_dim=768,
    )
    phase2_model.load_state_dict(torch.load(phase2_model_path, weights_only=True, map_location=device))
    gnn_encoder = phase2_model.encoder

    # Step 2: Generate semantic embeddings
    print("\n--- Step 2: Generating Semantic Embeddings ---")
    from experiments.phase3.semantic_embedder import LibrarySemanticEmbedder

    embedder = LibrarySemanticEmbedder(device=device)
    lib_embeddings = embedder.embed_libraries(library_concepts)
    domain_embeddings = embedder.embed_domains(list(label_encoder.classes_))

    # Save semantic embeddings
    embedder.save_embeddings(os.path.join(results_dir, "semantic_embeddings.npz"))

    # Step 3: Prepare data (reuse Phase 2 graph cache if available)
    print("\n--- Step 3: Preparing Data ---")
    phase2_cache = os.path.join(os.path.dirname(phase2_model_path), "graph_cache_k3_subTrue.pkl")
    cache_path = os.path.join(results_dir, "graph_cache_phase3.pkl")
    if os.path.exists(phase2_cache):
        print(f"Reusing Phase 2 graph cache: {phase2_cache}")
        cache_path = phase2_cache
    extractor = UASTSubtreeExtractor(k_hop=3)
    graphs = extractor.process_dataset(dataset, uast_dir, cache_path=cache_path)

    labels = prepare_labels(dataset, library_to_concept)

    # Match graphs to labels
    file_to_label = {}
    file_to_libs = {}
    for i, item in enumerate(dataset):
        file_to_label[item["file_id"]] = labels[i]
        file_to_libs[item["file_id"]] = item["libraries"]

    valid_graphs = []
    valid_labels = []
    valid_lib_lists = []
    for g in graphs:
        fid = g.get("file_id")
        if fid and fid in file_to_label and file_to_label[fid] != "Unknown":
            valid_graphs.append(g)
            valid_labels.append(file_to_label[fid])
            valid_lib_lists.append(file_to_libs.get(fid, []))

    print(f"Valid graphs: {len(valid_graphs)}")

    # Convert to PyG
    pyg_data = graphs_to_pyg(valid_graphs, valid_labels, label_encoder)

    # Compute semantic embedding for each graph (average of its library embeddings)
    semantic_targets = []
    for libs in valid_lib_lists:
        lib_embs = []
        for lib in set(libs):
            emb = lib_embeddings.get(lib)
            if emb is not None:
                lib_embs.append(emb)
        if lib_embs:
            semantic_targets.append(np.mean(lib_embs, axis=0))
        else:
            semantic_targets.append(np.zeros(768, dtype=np.float32))

    semantic_targets = np.array(semantic_targets, dtype=np.float32)

    # Attach semantic targets to PyG data
    for i, data in enumerate(pyg_data):
        if i < len(semantic_targets):
            data.semantic_emb = torch.tensor(semantic_targets[i], dtype=torch.float32)

    # Split data
    indices = list(range(len(pyg_data)))
    encoded_labels = label_encoder.transform(valid_labels)

    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=encoded_labels
    )
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=0.15,
        random_state=42,
        stratify=encoded_labels[train_idx],
    )

    train_data = [pyg_data[i] for i in train_idx]
    val_data = [pyg_data[i] for i in val_idx]
    test_data = [pyg_data[i] for i in test_idx]

    loader_kwargs = {"pin_memory": True, "num_workers": 4} if device == "cuda" else {}
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_data, batch_size=batch_size, **loader_kwargs)
    test_loader = DataLoader(test_data, batch_size=batch_size, **loader_kwargs)

    # Step 4: Build aligned model
    print("\n--- Step 4: Building Aligned Model ---")
    aligner = ContrastiveAligner(
        graph_dim=768,
        semantic_dim=768,
        projection_dim=projection_dim,
        temperature=temperature,
    )

    model = AlignedClassifier(
        gnn_encoder=gnn_encoder,
        aligner=aligner,
        num_classes=num_classes,
        projection_dim=projection_dim,
        lambda_contrast=lambda_contrast,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    ce_criterion = nn.CrossEntropyLoss()

    # Mixed-precision training for GPU (T4 Tensor Cores support FP16)
    use_amp = device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Step 5: Training loop
    print("\n--- Step 5: Training ---")
    if use_amp:
        print("  Using mixed-precision (FP16) training")
    best_val_acc = 0
    patience_counter = 0
    history = {"train_loss": [], "val_acc": [], "contrast_loss": []}
    save_path = os.path.join(results_dir, "best_aligned_model.pt")

    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_ce = 0
        total_contrast = 0
        total_samples = 0

        for batch in train_loader:
            batch = batch.to(device)
            semantic_emb = batch.semantic_emb.view(-1, 768) if hasattr(batch, "semantic_emb") else torch.zeros(batch.y.size(0), 768, device=device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits, contrast_loss, _ = model(
                    batch.x, batch.edge_index, batch.batch, semantic_emb
                )
                ce_loss = ce_criterion(logits, batch.y)
                loss = ce_loss + lambda_contrast * contrast_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            bs = batch.y.size(0)
            total_loss += loss.item() * bs
            total_ce += ce_loss.item() * bs
            total_contrast += contrast_loss.item() * bs
            total_samples += bs

        scheduler.step()

        # Validation
        model.eval()
        val_preds = []
        val_labels_list = []
        val_probs = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                semantic_emb = batch.semantic_emb.view(-1, 768) if hasattr(batch, "semantic_emb") else torch.zeros(batch.y.size(0), 768, device=device)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits, _, _ = model(
                        batch.x, batch.edge_index, batch.batch, semantic_emb
                    )
                preds = logits.argmax(dim=-1)
                val_preds.append(preds.cpu().numpy())
                val_labels_list.append(batch.y.cpu().numpy())
                val_probs.append(F.softmax(logits, dim=-1).cpu().numpy())

        val_preds = np.concatenate(val_preds)
        val_labels_arr = np.concatenate(val_labels_list)
        val_acc = accuracy_score(val_labels_arr, val_preds)

        history["train_loss"].append(total_loss / total_samples)
        history["val_acc"].append(val_acc)
        history["contrast_loss"].append(total_contrast / total_samples)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:3d} | "
                f"Loss: {total_loss/total_samples:.4f} | "
                f"CE: {total_ce/total_samples:.4f} | "
                f"Contrast: {total_contrast/total_samples:.4f} | "
                f"Val Acc: {val_acc:.2%} | "
                f"Best: {best_val_acc:.2%}"
            )

        if patience_counter >= 15:
            print(f"Early stopping at epoch {epoch+1}")
            break

    train_time = time.time() - start_time

    # Load best model
    model.load_state_dict(torch.load(save_path, weights_only=True))

    # Step 6: Test evaluation
    print("\n--- Step 6: Test Evaluation ---")
    model.eval()
    test_preds = []
    test_labels_list = []
    test_probs = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            semantic_emb = batch.semantic_emb.view(-1, 768) if hasattr(batch, "semantic_emb") else torch.zeros(batch.y.size(0), 768, device=device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits, _, _ = model(
                    batch.x, batch.edge_index, batch.batch, semantic_emb
                )
            test_preds.append(logits.argmax(dim=-1).cpu().numpy())
            test_labels_list.append(batch.y.cpu().numpy())
            test_probs.append(F.softmax(logits, dim=-1).cpu().numpy())

    test_preds = np.concatenate(test_preds)
    test_labels_arr = np.concatenate(test_labels_list)
    test_probs = np.concatenate(test_probs)

    top1 = accuracy_score(test_labels_arr, test_preds)

    k = min(5, test_probs.shape[1])
    top5_preds = np.argsort(test_probs, axis=1)[:, -k:]
    top5 = np.mean([l in t for l, t in zip(test_labels_arr, top5_preds)])

    mrr_scores = []
    for i in range(len(test_labels_arr)):
        ranked = np.argsort(test_probs[i])[::-1]
        rank = np.where(ranked == test_labels_arr[i])[0][0] + 1
        mrr_scores.append(1.0 / rank)
    mrr = np.mean(mrr_scores)

    print(f"\nTest Top-1: {top1:.2%}")
    print(f"Test Top-5: {top5:.2%}")
    print(f"Test MRR:   {mrr:.4f}")

    # Step 7: Zero-shot retrieval evaluation
    print("\n--- Step 7: Zero-Shot Retrieval ---")
    zero_shot_results = evaluate_zero_shot(
        model, test_loader, embedder, label_encoder, device
    )

    # Save results
    results = {
        "phase": "phase3",
        "config": {
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "lr": lr,
            "lambda_contrast": lambda_contrast,
            "temperature": temperature,
            "projection_dim": projection_dim,
        },
        "training": {
            "train_time_seconds": train_time,
            "best_val_acc": best_val_acc,
        },
        "test_results": {
            "top1": float(top1),
            "top5": float(top5),
            "mrr": float(mrr),
        },
        "zero_shot": zero_shot_results,
    }

    results_path = os.path.join(results_dir, "phase3_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results


def evaluate_zero_shot(model, test_loader, embedder, label_encoder, device):
    """
    Evaluate zero-shot retrieval:
    Given a code's graph embedding, retrieve the most similar domain
    using only the aligned semantic space (no classification head).
    """
    model.eval()
    all_aligned_graph = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            semantic_emb = batch.semantic_emb.view(-1, 768) if hasattr(batch, "semantic_emb") else torch.zeros(batch.y.size(0), 768, device=device)

            aligned_graph, _ = model.get_aligned_embeddings(
                batch.x, batch.edge_index, batch.batch, semantic_emb
            )
            all_aligned_graph.append(aligned_graph.cpu().numpy())
            all_labels.append(batch.y.cpu().numpy())

    all_aligned_graph = np.concatenate(all_aligned_graph)
    all_labels = np.concatenate(all_labels)

    # Get domain embeddings in aligned space
    domain_names = list(label_encoder.classes_)
    domain_embs = embedder.embed_domains(domain_names)

    # Project domain embeddings through semantic projection
    domain_emb_matrix = np.stack([domain_embs[d] for d in domain_names])
    domain_emb_tensor = torch.tensor(domain_emb_matrix, dtype=torch.float32).to(device)

    with torch.no_grad():
        _, aligned_domains = model.aligner.get_aligned_embeddings(
            torch.zeros(len(domain_names), 768, device=device),  # dummy
            domain_emb_tensor,
        )
        aligned_domains = aligned_domains.cpu().numpy()

    # Cosine similarity retrieval
    from sklearn.metrics.pairwise import cosine_similarity

    sims = cosine_similarity(all_aligned_graph, aligned_domains)
    retrieval_preds = sims.argmax(axis=1)

    retrieval_acc = accuracy_score(all_labels, retrieval_preds)

    # Top-k retrieval
    top5_preds = np.argsort(sims, axis=1)[:, -5:]
    top5_acc = np.mean([l in t for l, t in zip(all_labels, top5_preds)])

    top10_preds = np.argsort(sims, axis=1)[:, -10:]
    top10_acc = np.mean([l in t for l, t in zip(all_labels, top10_preds)])

    print(f"Zero-Shot Retrieval Top-1: {retrieval_acc:.2%}")
    print(f"Zero-Shot Retrieval Top-5: {top5_acc:.2%}")
    print(f"Zero-Shot Retrieval Top-10: {top10_acc:.2%}")

    return {
        "retrieval_top1": float(retrieval_acc),
        "retrieval_top5": float(top5_acc),
        "retrieval_top10": float(top10_acc),
    }
