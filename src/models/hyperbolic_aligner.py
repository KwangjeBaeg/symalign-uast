"""
Hyperbolic Contrastive Alignment Experiment

Replaces the Euclidean contrastive alignment (Phase 3) with
Poincaré Ball hyperbolic alignment to better preserve the
hierarchical tree structure of UASTs.

Key idea: UAST is a tree → trees have exponential node growth →
Euclidean space cannot faithfully embed trees without distortion.
Hyperbolic space (constant negative curvature) naturally matches
tree topology and can embed them with low distortion.

Architecture:
    GNN encoder (768-dim Euclidean)
    → Tangent space projection (768 → projection_dim)
    → Exponential map to Poincaré ball
    → Hyperbolic contrastive loss (negative distance-based)

Reference:
    - Nickel & Kiela, "Poincaré Embeddings" (NeurIPS 2017)
    - Chami et al., "Hyperbolic Graph Neural Networks" (NeurIPS 2019)
"""

import json
import os
import time
import pickle
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from torch_geometric.data import Data, DataLoader

from .poincare_ops import (
    expmap0, logmap0, project, hyp_distance, hyp_distance_batch,
    poincare_mean, PoincareLinear, MIN_NORM, BALL_EPS,
)


class HyperbolicContrastiveAligner(nn.Module):
    """
    Contrastive alignment in Poincaré Ball hyperbolic space.

    Projects both graph and semantic embeddings onto the Poincaré ball,
    then uses hyperbolic distance for the contrastive loss.
    """

    def __init__(
        self,
        graph_dim: int = 768,
        semantic_dim: int = 768,
        projection_dim: int = 128,
        curvature: float = 1.0,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.curvature = curvature
        self.temperature = temperature
        self.projection_dim = projection_dim

        # Euclidean projection to tangent space at origin
        self.graph_proj = nn.Sequential(
            nn.Linear(graph_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, projection_dim),
        )

        self.semantic_proj = nn.Sequential(
            nn.Linear(semantic_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, projection_dim),
        )

        # Scale init for stability in hyperbolic space
        for module in [self.graph_proj, self.semantic_proj]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.1)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def project_graph(self, graph_emb: torch.Tensor) -> torch.Tensor:
        """Project graph embeddings onto Poincaré ball."""
        h = self.graph_proj(graph_emb)
        return project(expmap0(h, self.curvature), self.curvature)

    def project_semantic(self, semantic_emb: torch.Tensor) -> torch.Tensor:
        """Project semantic embeddings onto Poincaré ball."""
        h = self.semantic_proj(semantic_emb)
        return project(expmap0(h, self.curvature), self.curvature)

    def forward(
        self, graph_emb: torch.Tensor, semantic_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute hyperbolic contrastive loss.

        Uses negative hyperbolic distance as similarity
        (closer points = higher similarity).

        Returns:
            loss: scalar contrastive loss
            dist_matrix: [B, B] distance matrix
        """
        z = self.project_graph(graph_emb)
        s = self.project_semantic(semantic_emb)

        # Pairwise hyperbolic distances [B, B]
        dist_matrix = hyp_distance_batch(z, s, self.curvature)

        # Convert distances to logits (negative distance / temperature)
        logits = -dist_matrix / self.temperature

        # Labels: diagonal (positive pairs)
        labels = torch.arange(logits.size(0), device=logits.device)

        # Symmetric loss
        loss_z2s = F.cross_entropy(logits, labels)
        loss_s2z = F.cross_entropy(logits.T, labels)
        loss = (loss_z2s + loss_s2z) / 2

        return loss, dist_matrix

    def get_aligned_embeddings(
        self, graph_emb: torch.Tensor, semantic_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get projected embeddings on Poincaré ball."""
        z = self.project_graph(graph_emb)
        s = self.project_semantic(semantic_emb)
        return z, s


class HyperbolicAlignedClassifier(nn.Module):
    """
    End-to-end: UAST-GNN → Hyperbolic Alignment → Classification.

    Classification is done in tangent space (log map back to Euclidean).
    """

    def __init__(
        self,
        gnn_encoder: nn.Module,
        aligner: HyperbolicContrastiveAligner,
        num_classes: int,
        projection_dim: int = 128,
        lambda_contrast: float = 0.5,
    ):
        super().__init__()
        self.gnn_encoder = gnn_encoder
        self.aligner = aligner
        self.lambda_contrast = lambda_contrast

        # Classify in tangent space (log map → MLP)
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
        Returns:
            logits: classification logits
            contrast_loss: hyperbolic contrastive loss
            hyp_graph_emb: graph embedding on Poincaré ball
        """
        graph_emb = self.gnn_encoder(x, edge_index, batch)
        contrast_loss, _ = self.aligner(graph_emb, semantic_emb)

        # Get hyperbolic graph embedding
        hyp_graph = self.aligner.project_graph(graph_emb)

        # Map back to tangent space for classification
        tangent_emb = logmap0(hyp_graph, self.aligner.curvature)
        logits = self.classifier(tangent_emb)

        return logits, contrast_loss, hyp_graph

    def get_aligned_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        semantic_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get aligned embeddings on Poincaré ball."""
        graph_emb = self.gnn_encoder(x, edge_index, batch)
        return self.aligner.get_aligned_embeddings(graph_emb, semantic_emb)


def evaluate_zero_shot_hyperbolic(
    model: HyperbolicAlignedClassifier,
    test_loader: DataLoader,
    embedder,
    label_encoder: LabelEncoder,
    device: str,
    use_amp: bool = True,
):
    """
    Zero-shot retrieval using hyperbolic distance.

    Instead of cosine similarity in Euclidean space, we use negative
    hyperbolic distance to find the closest domain embedding.
    """
    model.eval()
    all_hyp_graph = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            semantic_emb = (
                batch.semantic_emb.view(-1, 768)
                if hasattr(batch, "semantic_emb")
                else torch.zeros(batch.y.size(0), 768, device=device)
            )
            with torch.cuda.amp.autocast(enabled=use_amp):
                hyp_graph, _ = model.get_aligned_embeddings(
                    batch.x, batch.edge_index, batch.batch, semantic_emb
                )
            all_hyp_graph.append(hyp_graph.float().cpu())
            all_labels.append(batch.y.cpu().numpy())

    all_hyp_graph = torch.cat(all_hyp_graph)
    all_labels = np.concatenate(all_labels)

    # Get domain embeddings on Poincaré ball
    domain_names = list(label_encoder.classes_)
    domain_embs = embedder.embed_domains(domain_names)
    domain_emb_matrix = np.stack([domain_embs[d] for d in domain_names])
    domain_emb_tensor = torch.tensor(domain_emb_matrix, dtype=torch.float32).to(device)

    with torch.no_grad():
        _, hyp_domains = model.aligner.get_aligned_embeddings(
            torch.zeros(len(domain_names), 768, device=device),
            domain_emb_tensor,
        )
        hyp_domains = hyp_domains.float().cpu()

    # Hyperbolic distance retrieval
    curvature = model.aligner.curvature
    dist_matrix = hyp_distance_batch(all_hyp_graph, hyp_domains, curvature)

    # Nearest (smallest distance) = best match
    retrieval_preds = dist_matrix.argmin(dim=1).numpy()
    retrieval_acc = accuracy_score(all_labels, retrieval_preds)

    # Top-k (smallest k distances)
    top5_preds = dist_matrix.argsort(dim=1)[:, :5].numpy()
    top5_acc = np.mean([l in t for l, t in zip(all_labels, top5_preds)])

    top10_preds = dist_matrix.argsort(dim=1)[:, :10].numpy()
    top10_acc = np.mean([l in t for l, t in zip(all_labels, top10_preds)])

    print(f"Hyperbolic Zero-Shot Top-1: {retrieval_acc:.2%}")
    print(f"Hyperbolic Zero-Shot Top-5: {top5_acc:.2%}")
    print(f"Hyperbolic Zero-Shot Top-10: {top10_acc:.2%}")

    return {
        "retrieval_top1": float(retrieval_acc),
        "retrieval_top5": float(top5_acc),
        "retrieval_top10": float(top10_acc),
    }


def run_hyperbolic_experiment(
    dataset: list,
    library_to_concept: dict,
    library_concepts: list,
    uast_dir: str,
    results_dir: str,
    phase2_model_path: str,
    phase2_label_encoder_path: str,
    device: str = "cuda",
    num_epochs: int = 100,
    batch_size: int = 256,
    lr: float = 5e-5,
    lambda_contrast: float = 0.5,
    temperature: float = 0.1,
    projection_dim: int = 128,
    curvature: float = 1.0,
):
    """
    Run Hyperbolic Contrastive Alignment experiment.

    This is the hyperbolic counterpart to Phase 3's Euclidean contrastive alignment.
    """
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 80)
    print("EXPERIMENT: Hyperbolic Contrastive Alignment (Poincaré Ball)")
    print(f"  curvature={curvature}, projection_dim={projection_dim}")
    print(f"  temperature={temperature}, lr={lr}")
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

    phase2_model = UASTClassifier(
        num_classes=num_classes,
        node_feature_dim=node_feature_dim,
        hidden_dim=256,
        embedding_dim=768,
    )
    phase2_model.load_state_dict(
        torch.load(phase2_model_path, weights_only=True, map_location=device)
    )
    gnn_encoder = phase2_model.encoder

    # Step 2: Generate semantic embeddings
    print("\n--- Step 2: Generating Semantic Embeddings ---")
    from experiments.phase3.semantic_embedder import LibrarySemanticEmbedder

    embedder = LibrarySemanticEmbedder(device=device)
    lib_embeddings = embedder.embed_libraries(library_concepts)
    domain_embeddings = embedder.embed_domains(list(label_encoder.classes_))

    # Step 3: Prepare data (reuse Phase 2 graph cache)
    print("\n--- Step 3: Preparing Data ---")
    phase2_cache = os.path.join(
        os.path.dirname(phase2_model_path), "graph_cache_k3_subTrue.pkl"
    )
    cache_path = os.path.join(results_dir, "graph_cache_hyperbolic.pkl")
    if os.path.exists(phase2_cache):
        print(f"Reusing Phase 2 graph cache: {phase2_cache}")
        cache_path = phase2_cache

    extractor = UASTSubtreeExtractor(k_hop=3)
    graphs = extractor.process_dataset(dataset, uast_dir, cache_path=cache_path)

    labels = prepare_labels(dataset, library_to_concept)

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

    pyg_data = graphs_to_pyg(valid_graphs, valid_labels, label_encoder)

    # Compute semantic targets
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

    for i, data in enumerate(pyg_data):
        if i < len(semantic_targets):
            data.semantic_emb = torch.tensor(semantic_targets[i], dtype=torch.float32)

    # Split
    indices = list(range(len(pyg_data)))
    encoded_labels = label_encoder.transform(valid_labels)

    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=encoded_labels
    )
    train_idx, val_idx = train_test_split(
        train_idx, test_size=0.15, random_state=42,
        stratify=encoded_labels[train_idx],
    )

    train_data = [pyg_data[i] for i in train_idx]
    val_data = [pyg_data[i] for i in val_idx]
    test_data = [pyg_data[i] for i in test_idx]

    loader_kwargs = {"pin_memory": True, "num_workers": 4} if device == "cuda" else {}
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_data, batch_size=batch_size, **loader_kwargs)
    test_loader = DataLoader(test_data, batch_size=batch_size, **loader_kwargs)

    # Step 4: Build hyperbolic model
    print("\n--- Step 4: Building Hyperbolic Model ---")
    aligner = HyperbolicContrastiveAligner(
        graph_dim=768,
        semantic_dim=768,
        projection_dim=projection_dim,
        curvature=curvature,
        temperature=temperature,
    )

    model = HyperbolicAlignedClassifier(
        gnn_encoder=gnn_encoder,
        aligner=aligner,
        num_classes=num_classes,
        projection_dim=projection_dim,
        lambda_contrast=lambda_contrast,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Use RSGD-friendly optimizer with lower LR for hyperbolic stability
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    ce_criterion = nn.CrossEntropyLoss()

    use_amp = device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Step 5: Training
    print("\n--- Step 5: Training ---")
    if use_amp:
        print("  Using mixed-precision (FP16) training")
    best_val_acc = 0
    patience_counter = 0
    history = {"train_loss": [], "val_acc": [], "contrast_loss": []}
    save_path = os.path.join(results_dir, "best_hyperbolic_model.pt")

    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_ce = 0
        total_contrast = 0
        total_samples = 0

        for batch_data in train_loader:
            batch_data = batch_data.to(device)
            semantic_emb = (
                batch_data.semantic_emb.view(-1, 768)
                if hasattr(batch_data, "semantic_emb")
                else torch.zeros(batch_data.y.size(0), 768, device=device)
            )

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits, contrast_loss, _ = model(
                    batch_data.x, batch_data.edge_index, batch_data.batch, semantic_emb
                )
                ce_loss = ce_criterion(logits, batch_data.y)
                loss = ce_loss + lambda_contrast * contrast_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            bs = batch_data.y.size(0)
            total_loss += loss.item() * bs
            total_ce += ce_loss.item() * bs
            total_contrast += contrast_loss.item() * bs
            total_samples += bs

        scheduler.step()

        # Validation
        model.eval()
        val_preds = []
        val_labels_list = []

        with torch.no_grad():
            for batch_data in val_loader:
                batch_data = batch_data.to(device)
                semantic_emb = (
                    batch_data.semantic_emb.view(-1, 768)
                    if hasattr(batch_data, "semantic_emb")
                    else torch.zeros(batch_data.y.size(0), 768, device=device)
                )
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits, _, _ = model(
                        batch_data.x, batch_data.edge_index, batch_data.batch, semantic_emb
                    )
                val_preds.append(logits.argmax(dim=-1).cpu().numpy())
                val_labels_list.append(batch_data.y.cpu().numpy())

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
        for batch_data in test_loader:
            batch_data = batch_data.to(device)
            semantic_emb = (
                batch_data.semantic_emb.view(-1, 768)
                if hasattr(batch_data, "semantic_emb")
                else torch.zeros(batch_data.y.size(0), 768, device=device)
            )
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits, _, _ = model(
                    batch_data.x, batch_data.edge_index, batch_data.batch, semantic_emb
                )
            test_preds.append(logits.argmax(dim=-1).cpu().numpy())
            test_labels_list.append(batch_data.y.cpu().numpy())
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

    # Step 7: Zero-shot retrieval (hyperbolic)
    print("\n--- Step 7: Zero-Shot Retrieval (Hyperbolic) ---")
    zero_shot_results = evaluate_zero_shot_hyperbolic(
        model, test_loader, embedder, label_encoder, device, use_amp
    )

    # Save results
    results = {
        "experiment": "hyperbolic_alignment",
        "config": {
            "curvature": curvature,
            "projection_dim": projection_dim,
            "temperature": temperature,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "lr": lr,
            "lambda_contrast": lambda_contrast,
        },
        "training": {
            "train_time_seconds": train_time,
            "best_val_acc": best_val_acc,
            "epochs_trained": epoch + 1,
        },
        "test_results": {
            "top1": float(top1),
            "top5": float(top5),
            "mrr": float(mrr),
        },
        "zero_shot": zero_shot_results,
    }

    results_path = os.path.join(results_dir, "hyperbolic_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results
