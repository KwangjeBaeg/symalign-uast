"""
Geometry Ablation Experiment Runner (Experiment 01).

Unified runner for E1–E4 with identical data, GAT backbone, projector
architecture, and evaluation — the ONLY variable is geometry.

Variants:
  E1: CE-only           (GAT + CE head, no prototype retrieval)
  E2: Euclidean Contr.  (GAT + Euclidean contrastive + cosine retrieval)
  E3: Hybrid Hyperbolic (GAT + Poincaré contrastive + hyperbolic retrieval)
  E4: Fully Hyperbolic  (Hyperbolic GAT + Poincaré contrastive + hyp retrieval)
"""

import json
import os
import pickle
import time
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

from torch_geometric.data import Data, DataLoader

from experiments.phase2.gnn_encoder import UASTGraphEncoder, UASTClassifier
from experiments.phase2.subtree_extractor import UASTSubtreeExtractor, UAST_NODE_TYPES
from experiments.phase2.train_gnn import prepare_labels, graphs_to_pyg
from experiments.phase3.semantic_embedder import LibrarySemanticEmbedder
from experiments.hyperbolic.poincare_ops import (
    expmap0, logmap0, project, hyp_distance_batch,
)
from experiments.geometry_ablation.unified_evaluator import (
    compute_indist_metrics,
    compute_zeroshot_metrics,
    compute_knn_accuracy,
    compute_geometry_diagnostics,
)


# ============================================================================
# Unified projector (shared architecture for E2/E3/E4)
# ============================================================================

PROJECTION_DIM = 128  # Shared across all contrastive variants


class UnifiedProjector(nn.Module):
    """Identical projector for fair geometry comparison."""

    def __init__(self, input_dim: int = 768, projection_dim: int = PROJECTION_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, projection_dim),
        )
        # Small-scale init for hyperbolic stability (safe for Euclidean too)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================================
# E1: CE-only classifier
# ============================================================================

class E1_CEOnly(nn.Module):
    """GAT encoder + CE classifier head. No contrastive, no retrieval."""

    def __init__(self, gnn_encoder: UASTGraphEncoder, num_classes: int):
        super().__init__()
        self.gnn_encoder = gnn_encoder
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

    def forward(self, x, edge_index, batch, semantic_emb=None):
        graph_emb = self.gnn_encoder(x, edge_index, batch)
        logits = self.classifier(graph_emb)
        dummy_loss = torch.tensor(0.0, device=logits.device)
        return logits, dummy_loss, graph_emb

    def get_aligned_embeddings(self, x, edge_index, batch, semantic_emb=None):
        graph_emb = self.gnn_encoder(x, edge_index, batch)
        return graph_emb, None


# ============================================================================
# E2: Euclidean Contrastive
# ============================================================================

class E2_EuclideanContrastive(nn.Module):
    """GAT encoder + Euclidean contrastive (cosine) alignment + CE head."""

    def __init__(
        self,
        gnn_encoder: UASTGraphEncoder,
        num_classes: int,
        temperature: float = 0.1,
        lambda_contrast: float = 0.5,
    ):
        super().__init__()
        self.gnn_encoder = gnn_encoder
        self.lambda_contrast = lambda_contrast
        self.temperature = temperature

        self.graph_proj = UnifiedProjector(768, PROJECTION_DIM)
        self.semantic_proj = UnifiedProjector(768, PROJECTION_DIM)

        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(PROJECTION_DIM, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )

    def forward(self, x, edge_index, batch, semantic_emb):
        graph_emb = self.gnn_encoder(x, edge_index, batch)

        z = F.normalize(self.graph_proj(graph_emb), dim=-1)
        s = F.normalize(self.semantic_proj(semantic_emb), dim=-1)

        # InfoNCE (cosine similarity)
        logits_contrast = torch.matmul(z, s.T) / self.temperature
        labels_contrast = torch.arange(z.size(0), device=z.device)
        loss_z2s = F.cross_entropy(logits_contrast, labels_contrast)
        loss_s2z = F.cross_entropy(logits_contrast.T, labels_contrast)
        contrast_loss = (loss_z2s + loss_s2z) / 2

        logits = self.classifier(z)
        return logits, contrast_loss, z

    def get_aligned_embeddings(self, x, edge_index, batch, semantic_emb):
        graph_emb = self.gnn_encoder(x, edge_index, batch)
        z = F.normalize(self.graph_proj(graph_emb), dim=-1)
        s = F.normalize(self.semantic_proj(semantic_emb), dim=-1)
        return z, s


# ============================================================================
# E3: Hybrid Hyperbolic (current method)
# ============================================================================

class E3_HybridHyperbolic(nn.Module):
    """GAT encoder + Poincaré contrastive alignment + CE head (tangent space)."""

    def __init__(
        self,
        gnn_encoder: UASTGraphEncoder,
        num_classes: int,
        curvature: float = 1.0,
        temperature: float = 0.1,
        lambda_contrast: float = 0.5,
    ):
        super().__init__()
        self.gnn_encoder = gnn_encoder
        self.lambda_contrast = lambda_contrast
        self.temperature = temperature
        self.curvature = curvature

        self.graph_proj = UnifiedProjector(768, PROJECTION_DIM)
        self.semantic_proj = UnifiedProjector(768, PROJECTION_DIM)

        # Classify from tangent space (logmap0)
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(PROJECTION_DIM, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )

    def _to_poincare(self, h: torch.Tensor) -> torch.Tensor:
        return project(expmap0(h, self.curvature), self.curvature)

    def forward(self, x, edge_index, batch, semantic_emb):
        graph_emb = self.gnn_encoder(x, edge_index, batch)

        z_tang = self.graph_proj(graph_emb)
        s_tang = self.semantic_proj(semantic_emb)

        z_hyp = self._to_poincare(z_tang)
        s_hyp = self._to_poincare(s_tang)

        # Hyperbolic contrastive: -distance / temperature
        dist_matrix = hyp_distance_batch(z_hyp, s_hyp, self.curvature)
        logits_contrast = -dist_matrix / self.temperature
        labels_contrast = torch.arange(z_hyp.size(0), device=z_hyp.device)
        loss_z2s = F.cross_entropy(logits_contrast, labels_contrast)
        loss_s2z = F.cross_entropy(logits_contrast.T, labels_contrast)
        contrast_loss = (loss_z2s + loss_s2z) / 2

        # Classify in tangent space
        tangent_emb = logmap0(z_hyp, self.curvature)
        logits = self.classifier(tangent_emb)
        return logits, contrast_loss, z_hyp

    def get_aligned_embeddings(self, x, edge_index, batch, semantic_emb):
        graph_emb = self.gnn_encoder(x, edge_index, batch)
        z_hyp = self._to_poincare(self.graph_proj(graph_emb))
        s_hyp = self._to_poincare(self.semantic_proj(semantic_emb))
        return z_hyp, s_hyp


# ============================================================================
# E4: Fully Hyperbolic (exploratory pilot)
# ============================================================================

class HyperbolicGATLayer(nn.Module):
    """
    Simplified hyperbolic graph attention layer.

    Operates in tangent space at origin for message passing,
    then projects back to Poincaré ball. This avoids full
    Riemannian optimization while still keeping the encoder
    in hyperbolic space.
    """

    def __init__(self, in_dim: int, out_dim: int, curvature: float = 1.0,
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.curvature = curvature
        from torch_geometric.nn import GATConv
        self.gat = GATConv(in_dim, out_dim, heads=num_heads,
                           concat=False, dropout=dropout)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x_hyp, edge_index):
        # logmap to tangent space
        x_tang = logmap0(x_hyp, self.curvature)
        # message passing in tangent space
        x_tang = self.gat(x_tang, edge_index)
        x_tang = self.bn(x_tang)
        x_tang = F.relu(x_tang)
        # expmap back to Poincaré ball
        x_hyp = project(expmap0(x_tang, self.curvature), self.curvature)
        return x_hyp


class HyperbolicGATEncoder(nn.Module):
    """
    Fully hyperbolic GAT encoder: all message passing happens via
    tangent-space GAT interleaved with exp/log maps.
    """

    def __init__(self, node_feature_dim: int = 6, hidden_dim: int = 256,
                 output_dim: int = 768, curvature: float = 1.0,
                 num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.curvature = curvature
        # Input: Euclidean features → Poincaré ball
        self.input_proj = nn.Linear(node_feature_dim, hidden_dim)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_d = hidden_dim if i == 0 else hidden_dim
            out_d = output_dim if i == num_layers - 1 else hidden_dim
            self.layers.append(
                HyperbolicGATLayer(in_d, out_d, curvature, dropout=dropout)
            )

    def forward(self, x, edge_index, batch):
        from torch_geometric.nn import global_mean_pool
        # Project features to tangent space, then to Poincaré ball
        h = self.input_proj(x)
        h = project(expmap0(h, self.curvature), self.curvature)

        for layer in self.layers:
            h = layer(h, edge_index)

        # Pool in tangent space
        h_tang = logmap0(h, self.curvature)
        graph_emb_tang = global_mean_pool(h_tang, batch)
        # Back to Poincaré ball
        return project(expmap0(graph_emb_tang, self.curvature), self.curvature)


class E4_FullyHyperbolic(nn.Module):
    """Fully hyperbolic encoder + Poincaré contrastive + CE (tangent)."""

    def __init__(
        self,
        num_classes: int,
        node_feature_dim: int = 6,
        curvature: float = 1.0,
        temperature: float = 0.1,
        lambda_contrast: float = 0.5,
    ):
        super().__init__()
        self.curvature = curvature
        self.temperature = temperature
        self.lambda_contrast = lambda_contrast

        self.hyp_encoder = HyperbolicGATEncoder(
            node_feature_dim=node_feature_dim,
            hidden_dim=256,
            output_dim=768,
            curvature=curvature,
        )
        # Projector: tangent space → tangent space → Poincaré
        self.graph_proj = UnifiedProjector(768, PROJECTION_DIM)
        self.semantic_proj = UnifiedProjector(768, PROJECTION_DIM)

        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(PROJECTION_DIM, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )

    def _to_poincare(self, h: torch.Tensor) -> torch.Tensor:
        return project(expmap0(h, self.curvature), self.curvature)

    def forward(self, x, edge_index, batch, semantic_emb):
        # Encoder output already on Poincaré ball
        z_hyp_raw = self.hyp_encoder(x, edge_index, batch)
        # Project to alignment space via tangent
        z_tang = logmap0(z_hyp_raw, self.curvature)
        z_proj = self.graph_proj(z_tang)
        z_hyp = self._to_poincare(z_proj)

        s_proj = self.semantic_proj(semantic_emb)
        s_hyp = self._to_poincare(s_proj)

        dist_matrix = hyp_distance_batch(z_hyp, s_hyp, self.curvature)
        logits_contrast = -dist_matrix / self.temperature
        labels_contrast = torch.arange(z_hyp.size(0), device=z_hyp.device)
        loss_z2s = F.cross_entropy(logits_contrast, labels_contrast)
        loss_s2z = F.cross_entropy(logits_contrast.T, labels_contrast)
        contrast_loss = (loss_z2s + loss_s2z) / 2

        tangent_emb = logmap0(z_hyp, self.curvature)
        logits = self.classifier(tangent_emb)
        return logits, contrast_loss, z_hyp

    def get_aligned_embeddings(self, x, edge_index, batch, semantic_emb):
        z_hyp_raw = self.hyp_encoder(x, edge_index, batch)
        z_tang = logmap0(z_hyp_raw, self.curvature)
        z_proj = self.graph_proj(z_tang)
        z_hyp = self._to_poincare(z_proj)
        s_hyp = self._to_poincare(self.semantic_proj(semantic_emb))
        return z_hyp, s_hyp


# ============================================================================
# Unified training & evaluation loop
# ============================================================================

def _set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(
    variant: str,
    gnn_encoder: UASTGraphEncoder,
    num_classes: int,
    node_feature_dim: int = 6,
    curvature: float = 1.0,
    temperature: float = 0.1,
    lambda_contrast: float = 0.5,
) -> nn.Module:
    """Factory for E1–E4."""
    if variant == "E1":
        return E1_CEOnly(gnn_encoder, num_classes)
    elif variant == "E2":
        return E2_EuclideanContrastive(
            gnn_encoder, num_classes, temperature, lambda_contrast,
        )
    elif variant == "E3":
        return E3_HybridHyperbolic(
            gnn_encoder, num_classes, curvature, temperature, lambda_contrast,
        )
    elif variant == "E4":
        return E4_FullyHyperbolic(
            num_classes, node_feature_dim, curvature, temperature, lambda_contrast,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")


# ============================================================================
# Balanced validation helper
# ============================================================================

def _compute_val_zs_top1(
    model,
    val_loader,
    domain_emb_tensor: "torch.Tensor",
    device: str,
    use_amp: bool,
    variant: str,
) -> float:
    """
    Compute ZS Top-1 on the validation set using prototype retrieval.
    Used for the balanced early stopping criterion.
    """
    import torch
    import torch.nn.functional as F
    from experiments.hyperbolic.poincare_ops import expmap0, project, hyp_distance_batch

    model.eval()
    val_aligned, val_lbls_list = [], []
    with torch.no_grad():
        for batch_data in val_loader:
            batch_data = batch_data.to(device)
            sem = (
                batch_data.semantic_emb.view(-1, 768)
                if hasattr(batch_data, "semantic_emb")
                else torch.zeros(batch_data.y.size(0), 768, device=device)
            )
            with torch.cuda.amp.autocast(enabled=use_amp):
                g_aligned, _ = model.get_aligned_embeddings(
                    batch_data.x, batch_data.edge_index, batch_data.batch, sem,
                )
            val_aligned.append(g_aligned.float().cpu())
            val_lbls_list.append(batch_data.y.cpu().numpy())

    val_aligned = torch.cat(val_aligned)
    val_labels = np.concatenate(val_lbls_list)

    with torch.no_grad():
        if variant == "E2":
            d_aligned = F.normalize(
                model.semantic_proj(domain_emb_tensor), dim=-1
            ).cpu()
            from sklearn.metrics.pairwise import cosine_similarity
            sim = cosine_similarity(val_aligned.numpy(), d_aligned.numpy())
            ranked = np.argsort(-sim, axis=1)
        else:  # E3, E4
            s_tang = model.semantic_proj(domain_emb_tensor)
            d_aligned = project(
                expmap0(s_tang, model.curvature), model.curvature
            ).cpu()
            dist_np = hyp_distance_batch(
                val_aligned, d_aligned, model.curvature
            ).numpy()
            ranked = np.argsort(dist_np, axis=1)

    top1 = float(np.mean(ranked[:, 0] == val_labels))
    return top1


def run_geometry_experiment(
    variant: str,
    dataset: list,
    library_to_concept: dict,
    library_concepts: list,
    uast_dir: str,
    results_dir: str,
    phase2_model_path: str,
    phase2_label_encoder_path: str,
    graph_cache_path: str,
    device: str = "cuda",
    seed: int = 42,
    num_epochs: int = 100,
    batch_size: int = 256,
    lr: float = 5e-5,
    lambda_contrast: float = 0.5,
    temperature: float = 0.1,
    curvature: float = 1.0,
    patience: int = 15,
    val_criterion: str = "accuracy",  # "accuracy" or "balanced" (harmonic mean of InDist+ZS)
) -> Dict:
    """
    Run one geometry ablation configuration.

    Returns dict with all three tiers of metrics + training metadata.
    """
    _set_seed(seed)
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 80)
    print(f"GEOMETRY ABLATION: {variant}  (seed={seed})")
    print("=" * 80)

    # ── 1. Load Phase 2 artifacts ──
    with open(phase2_label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)
    num_classes = len(label_encoder.classes_)
    node_feature_dim = len(UAST_NODE_TYPES)

    # Load pretrained GAT encoder (shared for E1–E3)
    if variant != "E4":
        phase2_model = UASTClassifier(
            num_classes=num_classes,
            node_feature_dim=node_feature_dim,
            hidden_dim=256,
            embedding_dim=768,
        )
        phase2_model.load_state_dict(
            torch.load(phase2_model_path, weights_only=True, map_location="cpu")
        )
        gnn_encoder = phase2_model.encoder
    else:
        gnn_encoder = None  # E4 has its own encoder

    # ── 2. Semantic embeddings ──
    embedder = LibrarySemanticEmbedder(device=device)
    lib_embeddings = embedder.embed_libraries(library_concepts)
    domain_embeddings = embedder.embed_domains(list(label_encoder.classes_))

    # ── 3. Prepare graph data ──
    extractor = UASTSubtreeExtractor(k_hop=3)
    graphs = extractor.process_dataset(dataset, uast_dir, cache_path=graph_cache_path)
    labels = prepare_labels(dataset, library_to_concept)

    file_to_label = {}
    file_to_libs = {}
    for i, item in enumerate(dataset):
        file_to_label[item["file_id"]] = labels[i]
        file_to_libs[item["file_id"]] = item["libraries"]

    valid_graphs, valid_labels, valid_lib_lists = [], [], []
    for g in graphs:
        fid = g.get("file_id")
        if fid and fid in file_to_label and file_to_label[fid] != "Unknown":
            valid_graphs.append(g)
            valid_labels.append(file_to_label[fid])
            valid_lib_lists.append(file_to_libs.get(fid, []))

    print(f"Valid graphs: {len(valid_graphs)}")

    pyg_data = graphs_to_pyg(valid_graphs, valid_labels, label_encoder)

    # Semantic targets (mean library embedding per file)
    semantic_targets = []
    for libs in valid_lib_lists:
        lib_embs = [lib_embeddings[lib] for lib in set(libs) if lib in lib_embeddings]
        if lib_embs:
            semantic_targets.append(np.mean(lib_embs, axis=0))
        else:
            semantic_targets.append(np.zeros(768, dtype=np.float32))
    semantic_targets = np.array(semantic_targets, dtype=np.float32)

    for i, data in enumerate(pyg_data):
        if i < len(semantic_targets):
            data.semantic_emb = torch.tensor(semantic_targets[i], dtype=torch.float32)

    # ── 4. Split (seed-controlled) ──
    indices = list(range(len(pyg_data)))
    encoded_labels = label_encoder.transform(valid_labels)

    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=seed, stratify=encoded_labels,
    )
    train_idx, val_idx = train_test_split(
        train_idx, test_size=0.15, random_state=seed,
        stratify=encoded_labels[train_idx],
    )

    train_data = [pyg_data[i] for i in train_idx]
    val_data = [pyg_data[i] for i in val_idx]
    test_data = [pyg_data[i] for i in test_idx]

    lkw = {"pin_memory": True, "num_workers": 4} if device == "cuda" else {}
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, **lkw)
    val_loader = DataLoader(val_data, batch_size=batch_size, **lkw)
    test_loader = DataLoader(test_data, batch_size=batch_size, **lkw)

    # ── 5. Build model ──
    _set_seed(seed)  # re-seed for model init
    model = build_model(
        variant, gnn_encoder, num_classes, node_feature_dim,
        curvature, temperature, lambda_contrast,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    ce_criterion = nn.CrossEntropyLoss()

    use_amp = device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ── 5b. Pre-project domain prototypes for balanced val criterion ──
    domain_emb_tensor_val = None
    if val_criterion == "balanced" and variant in ("E2", "E3", "E4"):
        domain_names_val = list(label_encoder.classes_)
        domain_emb_matrix_val = np.stack([domain_embeddings[d] for d in domain_names_val])
        domain_emb_tensor_val = torch.tensor(
            domain_emb_matrix_val, dtype=torch.float32
        ).to(device)

    # ── 6. Training ──
    best_val_score = 0.0
    patience_counter = 0
    save_path = os.path.join(results_dir, f"best_{variant}_seed{seed}.pt")
    converged = True
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        total_loss, total_ce, total_con, total_n = 0.0, 0.0, 0.0, 0

        for batch_data in train_loader:
            batch_data = batch_data.to(device)
            sem = (
                batch_data.semantic_emb.view(-1, 768)
                if hasattr(batch_data, "semantic_emb")
                else torch.zeros(batch_data.y.size(0), 768, device=device)
            )

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits, contrast_loss, _ = model(
                    batch_data.x, batch_data.edge_index, batch_data.batch, sem,
                )
                ce_loss = ce_criterion(logits, batch_data.y)
                lc = lambda_contrast if variant != "E1" else 0.0
                loss = ce_loss + lc * contrast_loss

            if not torch.isfinite(loss):
                print(f"  [WARN] Non-finite loss at epoch {epoch+1}, skipping batch")
                converged = False
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            bs = batch_data.y.size(0)
            total_loss += loss.item() * bs
            total_ce += ce_loss.item() * bs
            total_con += contrast_loss.item() * bs
            total_n += bs

        scheduler.step()

        # Validation
        model.eval()
        val_preds, val_lbls = [], []
        with torch.no_grad():
            for batch_data in val_loader:
                batch_data = batch_data.to(device)
                sem = (
                    batch_data.semantic_emb.view(-1, 768)
                    if hasattr(batch_data, "semantic_emb")
                    else torch.zeros(batch_data.y.size(0), 768, device=device)
                )
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits, _, _ = model(
                        batch_data.x, batch_data.edge_index, batch_data.batch, sem,
                    )
                val_preds.append(logits.argmax(dim=-1).cpu().numpy())
                val_lbls.append(batch_data.y.cpu().numpy())

        val_preds = np.concatenate(val_preds)
        val_lbls = np.concatenate(val_lbls)
        val_acc = accuracy_score(val_lbls, val_preds)

        # Balanced criterion: harmonic mean of in-dist accuracy and ZS Top-1 on val set
        if val_criterion == "balanced" and domain_emb_tensor_val is not None:
            val_zs_top1 = _compute_val_zs_top1(
                model, val_loader, domain_emb_tensor_val,
                device, use_amp, variant,
            )
            if val_acc > 0 and val_zs_top1 > 0:
                val_score = 2 * val_acc * val_zs_top1 / (val_acc + val_zs_top1)
            else:
                val_score = 0.0
        else:
            val_score = val_acc
            val_zs_top1 = 0.0

        if val_score > best_val_score:
            best_val_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg_loss = total_loss / max(total_n, 1)
            avg_ce = total_ce / max(total_n, 1)
            avg_con = total_con / max(total_n, 1)
            if val_criterion == "balanced" and variant not in ("E1",):
                print(
                    f"  Epoch {epoch+1:3d} | Loss {avg_loss:.4f} | "
                    f"CE {avg_ce:.4f} | Con {avg_con:.4f} | "
                    f"InDist {val_acc:.2%} | ZS {val_zs_top1:.2%} | "
                    f"Bal {val_score:.4f} | Best {best_val_score:.4f}"
                )
            else:
                print(
                    f"  Epoch {epoch+1:3d} | Loss {avg_loss:.4f} | "
                    f"CE {avg_ce:.4f} | Con {avg_con:.4f} | "
                    f"Val {val_acc:.2%} | Best {best_val_score:.2%}"
                )

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    train_time = time.time() - start_time
    epochs_trained = epoch + 1

    # Load best checkpoint
    model.load_state_dict(torch.load(save_path, weights_only=True, map_location=device))
    best_val_acc = best_val_score  # for results dict

    # ── 7. Evaluation ──
    print(f"\n--- Evaluation ({variant}, seed={seed}) ---")
    model.eval()

    # 7a. In-distribution metrics
    test_probs_list, test_labels_list = [], []
    test_embs_list = []
    with torch.no_grad():
        for batch_data in test_loader:
            batch_data = batch_data.to(device)
            sem = (
                batch_data.semantic_emb.view(-1, 768)
                if hasattr(batch_data, "semantic_emb")
                else torch.zeros(batch_data.y.size(0), 768, device=device)
            )
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits, _, embs = model(
                    batch_data.x, batch_data.edge_index, batch_data.batch, sem,
                )
            test_probs_list.append(F.softmax(logits.float(), dim=-1).cpu().numpy())
            test_labels_list.append(batch_data.y.cpu().numpy())
            test_embs_list.append(embs.float().cpu().numpy())

    test_probs = np.concatenate(test_probs_list)
    test_labels = np.concatenate(test_labels_list)
    test_embs = np.concatenate(test_embs_list)

    indist = compute_indist_metrics(test_labels, test_probs)
    print(f"  In-Dist Top-1: {indist['top1']:.2%}  MRR: {indist['mrr']:.4f}  "
          f"Macro-F1: {indist['macro_f1']:.4f}")

    # 7b. Zero-shot retrieval
    zs_results = {"zs_top1": 0.0, "zs_top5": 0.0, "zs_top10": 0.0,
                  "zs_mrr": 0.0, "map5": 0.0, "map10": 0.0, "knn10": 0.0}

    if variant != "E1":
        # Get domain prototypes in aligned space
        domain_names = list(label_encoder.classes_)
        domain_emb_matrix = np.stack([domain_embeddings[d] for d in domain_names])
        domain_emb_tensor = torch.tensor(domain_emb_matrix, dtype=torch.float32).to(device)

        with torch.no_grad():
            # Get all test embeddings in aligned space
            all_graph_aligned, all_domain_aligned = [], None
            for batch_data in test_loader:
                batch_data = batch_data.to(device)
                sem = (
                    batch_data.semantic_emb.view(-1, 768)
                    if hasattr(batch_data, "semantic_emb")
                    else torch.zeros(batch_data.y.size(0), 768, device=device)
                )
                with torch.cuda.amp.autocast(enabled=use_amp):
                    g_aligned, _ = model.get_aligned_embeddings(
                        batch_data.x, batch_data.edge_index, batch_data.batch, sem,
                    )
                all_graph_aligned.append(g_aligned.float().cpu())

            all_graph_aligned = torch.cat(all_graph_aligned)

            # Project domain prototypes through semantic_proj
            if variant == "E2":
                d_aligned = F.normalize(
                    model.semantic_proj(domain_emb_tensor), dim=-1
                ).cpu()
                # Cosine distance → similarity (higher = better)
                from sklearn.metrics.pairwise import cosine_similarity
                dist_matrix = cosine_similarity(
                    all_graph_aligned.numpy(), d_aligned.numpy()
                )
                zs_results = compute_zeroshot_metrics(
                    dist_matrix, test_labels, smaller_is_better=False,
                )
            elif variant in ("E3", "E4"):
                s_tang = model.semantic_proj(domain_emb_tensor)
                d_aligned = project(
                    expmap0(s_tang, model.curvature), model.curvature
                ).cpu()
                # Hyperbolic distance (smaller = better)
                dist_np = hyp_distance_batch(
                    all_graph_aligned, d_aligned, model.curvature
                ).numpy()
                zs_results = compute_zeroshot_metrics(
                    dist_np, test_labels, smaller_is_better=True,
                )

        # kNN-10
        knn_embs = all_graph_aligned.numpy()
        zs_results["knn10"] = compute_knn_accuracy(
            knn_embs, test_labels, k=10, metric="euclidean",
        )

        print(f"  ZS Top-1: {zs_results['zs_top1']:.2%}  MAP@5: {zs_results['map5']:.4f}  "
              f"kNN-10: {zs_results['knn10']:.4f}")

        # F3: Retrieval confusion — per-sample top-1 retrieved domain
        if variant == "E2":
            ranked_all = np.argsort(-dist_matrix, axis=1)
        else:
            ranked_all = np.argsort(dist_np, axis=1)

        # Map: true_domain → {retrieved_domain: count}
        retrieval_confusion = {}
        for true_idx, ranked_row in zip(test_labels, ranked_all):
            true_name = domain_names[true_idx]
            retrieved_name = domain_names[ranked_row[0]]
            if true_name not in retrieval_confusion:
                retrieval_confusion[true_name] = {}
            retrieval_confusion[true_name][retrieved_name] = (
                retrieval_confusion[true_name].get(retrieved_name, 0) + 1
            )
        zs_results["retrieval_confusion"] = retrieval_confusion

    # 7c. Geometry diagnostics
    # Use tangent-space embeddings for diagnostics (fair across E2/E3)
    if variant == "E1":
        diag_embs = test_embs  # raw graph embeddings
    elif variant == "E2":
        diag_embs = all_graph_aligned.numpy()
    else:
        diag_embs = logmap0(
            all_graph_aligned, model.curvature
        ).numpy() if variant in ("E3", "E4") else test_embs

    # Behavioral generality per domain (number of active behavior categories)
    behavioral_generality = {
        "Scientific Computing": 5, "Quantum Computing": 4,
        "Data Processing": 5, "Systems Programming": 3,
        "Testing": 2, "Security": 4, "DevOps": 3,
        "Networking": 3, "Web Development": 3, "Web": 3,
        "Machine Learning": 5, "GUI": 2, "Cloud Computing": 3,
        "Science": 5, "Type Checking": 2, "Unknown": 1,
    }

    geo_diag = compute_geometry_diagnostics(
        diag_embs, test_labels,
        domain_names=list(label_encoder.classes_),
        behavioral_generality=behavioral_generality,
    )
    print(f"  Sep. ratio: {geo_diag['sep_ratio']:.4f}  "
          f"Silhouette: {geo_diag['silhouette']:.4f}  "
          f"Norm-depth ρ: {geo_diag['norm_depth_corr']:.4f}")

    # ── 8. Assemble results ──
    result = {
        "variant": variant,
        "seed": seed,
        "training": {
            "train_time_seconds": train_time,
            "epochs_trained": epochs_trained,
            "best_val_acc": best_val_acc,
            "val_criterion": val_criterion,
            "converged": converged,
            "n_params": n_params,
        },
        "indist": indist,
        "zero_shot": zs_results,
        "geometry": geo_diag,
    }

    out_path = os.path.join(results_dir, f"{variant}_seed{seed}_results.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Results saved to {out_path}")

    return result
