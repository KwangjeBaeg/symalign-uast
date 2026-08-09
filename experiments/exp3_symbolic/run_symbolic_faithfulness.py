"""
Experiment 03: Symbolic Faithfulness — Main Runner.

Runs S0–S6 variants with fixed E3 backbone, evaluating symbolic validity,
reliability, and performance. Each variant trains a SymAlign model with
different symbolic loss/ontology configurations.

Usage:
    from experiments.exp3_symbolic.run_symbolic_faithfulness import (
        run_symbolic_faithfulness_experiment,
    )
    result = run_symbolic_faithfulness_experiment(variant="S1", seed=42, ...)
"""

import json
import os
import pickle
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from torch_geometric.data import DataLoader

from src.models.poincare_ops import hyp_distance_batch
from src.models.hyperbolic_aligner import (
    HyperbolicContrastiveAligner,
)
from src.data.symbolic_evidence import (
    SymbolicEvidenceExtractor,
    EVIDENCE_DIM,
)
from src.knowledge.domain_ontology import DomainOntology
from src.models.symalign_classifier import SymAlignClassifier
from src.losses.symbolic_losses import get_symbolic_loss
from .ontology_corruption import get_corrupted_ontology
from .hard_posthoc import apply_hard_posthoc
from src.evaluation.evaluation_metrics import evaluate_all


def _set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_symbolic_faithfulness_experiment(
    dataset: list,
    library_to_concept: dict,
    library_concepts: list,
    uast_dir: str,
    results_dir: str,
    phase2_model_path: str,
    phase2_label_encoder_path: str,
    graph_cache_path: str,
    variant: str = "S1",
    device: str = "cuda",
    seed: int = 42,
    num_epochs: int = 100,
    batch_size: int = 256,
    lr: float = 5e-5,
    lambda_contrast: float = 0.5,
    lambda_logic: float = 0.3,
    temperature: float = 0.1,
    projection_dim: int = 128,
    curvature: float = 1.0,
    noise_level: float = 0.2,
    ontology_path: Optional[str] = None,
    cache_dir: str = "",
) -> Dict:
    """
    Run a single symbolic faithfulness experiment variant.

    Args:
        variant: "S0", "S1", "S2", "S3", "S4", "S5",
                 "S6_struct", "S6_behav", "S6_excl"
        noise_level: for S5 variants (0.1, 0.2, 0.3)
        Other args: same as run_symalign_experiment
    """
    _set_seed(seed)
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 80)
    print(f"EXPERIMENT 03: Symbolic Faithfulness — {variant} (seed={seed})")
    print(f"  device={device}, epochs={num_epochs}, lr={lr}")
    print(f"  lambda_contrast={lambda_contrast}, lambda_logic={lambda_logic}")
    if variant == "S5":
        print(f"  noise_level={noise_level}")
    print("=" * 80)

    # =========================================================================
    # Step 0: Pre-compute Symbolic Evidence
    # =========================================================================
    print("\n--- Step 0: Extracting Symbolic Evidence ---")
    sym_extractor = SymbolicEvidenceExtractor()

    evidence_cache = os.path.join(cache_dir, "symbolic_evidence_cache.pkl") if cache_dir else ""
    if evidence_cache and os.path.exists(evidence_cache):
        print(f"Loading evidence from cache: {evidence_cache}")
        with open(evidence_cache, "rb") as f:
            evidence_map = pickle.load(f)
    else:
        evidence_map = sym_extractor.process_dataset(
            dataset, uast_dir,
            cache_path=evidence_cache if evidence_cache else None,
        )

    # =========================================================================
    # Step 1: Load Phase 2 Artifacts (GNN encoder)
    # =========================================================================
    print("\n--- Step 1: Loading Phase 2 Artifacts ---")
    from src.models.gnn_encoder import UASTGraphEncoder, UASTClassifier
    from src.data.subtree_extractor import (
        UASTSubtreeExtractor,
        UAST_NODE_TYPES,
    )
    from experiments.train_gnn import prepare_labels, graphs_to_pyg

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
    print(f"Loaded GNN encoder ({sum(p.numel() for p in gnn_encoder.parameters()):,} params)")

    # =========================================================================
    # Step 2: Generate Semantic Embeddings
    # =========================================================================
    print("\n--- Step 2: Generating Semantic Embeddings ---")
    from src.data.semantic_embedder import LibrarySemanticEmbedder

    embedder = LibrarySemanticEmbedder(device=device)
    lib_embeddings = embedder.embed_libraries(library_concepts)

    # =========================================================================
    # Step 3: Prepare Data
    # =========================================================================
    print("\n--- Step 3: Preparing Data ---")
    extractor = UASTSubtreeExtractor(k_hop=3)
    graphs = extractor.process_dataset(dataset, uast_dir, cache_path=graph_cache_path)

    labels = prepare_labels(dataset, library_to_concept)

    file_to_label = {}
    file_to_libs = {}
    for i, item in enumerate(dataset):
        file_to_label[item["file_id"]] = labels[i]
        file_to_libs[item["file_id"]] = item["libraries"]

    valid_graphs = []
    valid_labels = []
    valid_lib_lists = []
    valid_file_ids = []
    for g in graphs:
        fid = g.get("file_id")
        if fid and fid in file_to_label and file_to_label[fid] != "Unknown":
            valid_graphs.append(g)
            valid_labels.append(file_to_label[fid])
            valid_lib_lists.append(file_to_libs.get(fid, []))
            valid_file_ids.append(fid)

    print(f"Valid graphs: {len(valid_graphs)}")

    pyg_data = graphs_to_pyg(valid_graphs, valid_labels, label_encoder)

    # Attach semantic embeddings
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

    # Attach symbolic evidence
    evidence_attached = 0
    for i, data in enumerate(pyg_data):
        if i < len(semantic_targets):
            data.semantic_emb = torch.tensor(semantic_targets[i], dtype=torch.float32)

        fid = valid_file_ids[i] if i < len(valid_file_ids) else None
        if fid and fid in evidence_map:
            data.symbolic_evidence = torch.tensor(
                evidence_map[fid], dtype=torch.float32
            )
            evidence_attached += 1
        else:
            data.symbolic_evidence = torch.zeros(EVIDENCE_DIM, dtype=torch.float32)

    print(f"Symbolic evidence attached: {evidence_attached}/{len(pyg_data)}")

    # Split (fixed random_state=42 for reproducibility across variants)
    indices = list(range(len(pyg_data)))
    encoded_labels = label_encoder.transform(valid_labels)

    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=encoded_labels,
    )
    train_idx, val_idx = train_test_split(
        train_idx, test_size=0.15, random_state=42,
        stratify=encoded_labels[train_idx],
    )

    train_data = [pyg_data[i] for i in train_idx]
    val_data = [pyg_data[i] for i in val_idx]
    test_data = [pyg_data[i] for i in test_idx]

    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    loader_kwargs = {"pin_memory": True, "num_workers": 4} if device == "cuda" else {}
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_data, batch_size=batch_size, **loader_kwargs)
    test_loader = DataLoader(test_data, batch_size=batch_size, **loader_kwargs)

    # =========================================================================
    # Step 4: Build Model with variant-specific symbolic loss
    # =========================================================================
    print(f"\n--- Step 4: Building Model (variant={variant}) ---")

    # Load original ontology
    ontology = DomainOntology(ontology_path) if ontology_path else DomainOntology()
    orig_behavior_map = ontology.get_domain_behavior_map(label_encoder)
    orig_exclusion_matrix = ontology.get_exclusion_matrix(label_encoder)

    # Get (possibly corrupted) ontology for this variant
    behavior_map, exclusion_matrix = get_corrupted_ontology(
        variant, orig_behavior_map, orig_exclusion_matrix,
        noise_level=noise_level, seed=seed,
    )

    # Create the symbolic loss module
    sym_loss_module = get_symbolic_loss(
        variant, behavior_map, exclusion_matrix,
    )

    # For S0: lambda_logic = 0 (no symbolic loss contribution)
    effective_lambda_logic = 0.0 if variant == "S0" else lambda_logic

    # Hyperbolic aligner (E3 backbone)
    aligner = HyperbolicContrastiveAligner(
        graph_dim=768,
        semantic_dim=768,
        projection_dim=projection_dim,
        curvature=curvature,
        temperature=temperature,
    )

    # Full model
    model = SymAlignClassifier(
        gnn_encoder=gnn_encoder,
        aligner=aligner,
        neurosymbolic_loss=sym_loss_module,
        num_classes=num_classes,
        projection_dim=projection_dim,
        lambda_contrast=lambda_contrast,
        lambda_logic=effective_lambda_logic,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"Ontology: {'original' if variant not in ('S4', 'S5') else 'corrupted'}")
    print(f"Effective lambda_logic: {effective_lambda_logic}")

    # =========================================================================
    # Step 5: Training
    # =========================================================================
    print("\n--- Step 5: Training ---")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    ce_criterion = nn.CrossEntropyLoss()

    use_amp = device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_acc = 0
    patience_counter = 0
    patience = 15
    save_path = os.path.join(results_dir, f"best_model_{variant}_s{seed}.pt")

    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_samples = 0

        for batch_data in train_loader:
            batch_data = batch_data.to(device)
            semantic_emb = (
                batch_data.semantic_emb.view(-1, 768)
                if hasattr(batch_data, "semantic_emb")
                else torch.zeros(batch_data.y.size(0), 768, device=device)
            )
            symbolic_evidence = (
                batch_data.symbolic_evidence.view(-1, EVIDENCE_DIM)
                if hasattr(batch_data, "symbolic_evidence")
                else torch.zeros(batch_data.y.size(0), EVIDENCE_DIM, device=device)
            )

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits, _, contrast_loss, logic_loss, _ = model(
                    batch_data.x, batch_data.edge_index, batch_data.batch,
                    semantic_emb, symbolic_evidence,
                )
                total_loss, _ = model.compute_total_loss(
                    logits, batch_data.y, contrast_loss, logic_loss, ce_criterion,
                )

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            bs = batch_data.y.size(0)
            epoch_loss += total_loss.item() * bs
            epoch_samples += bs

        scheduler.step()

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_data in val_loader:
                batch_data = batch_data.to(device)
                semantic_emb = (
                    batch_data.semantic_emb.view(-1, 768)
                    if hasattr(batch_data, "semantic_emb")
                    else torch.zeros(batch_data.y.size(0), 768, device=device)
                )
                symbolic_evidence = (
                    batch_data.symbolic_evidence.view(-1, EVIDENCE_DIM)
                    if hasattr(batch_data, "symbolic_evidence")
                    else torch.zeros(batch_data.y.size(0), EVIDENCE_DIM, device=device)
                )
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits, _, _, _, _ = model(
                        batch_data.x, batch_data.edge_index, batch_data.batch,
                        semantic_emb, symbolic_evidence,
                    )
                preds = logits.argmax(dim=-1)
                val_correct += (preds == batch_data.y).sum().item()
                val_total += batch_data.y.size(0)

        val_acc = val_correct / val_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch+1:3d} | Loss: {epoch_loss/epoch_samples:.4f} | "
                f"Val: {val_acc:.2%} | Best: {best_val_acc:.2%}"
            )

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    train_time = time.time() - start_time
    print(f"  Training time: {train_time:.1f}s, best val acc: {best_val_acc:.2%}")

    # Load best model
    model.load_state_dict(torch.load(save_path, weights_only=True))

    # =========================================================================
    # Step 6: Comprehensive Evaluation
    # =========================================================================
    print("\n--- Step 6: Evaluation ---")
    model.eval()

    test_logits_all = []
    test_labels_all = []
    test_evidence_all = []

    with torch.no_grad():
        for batch_data in test_loader:
            batch_data = batch_data.to(device)
            semantic_emb = (
                batch_data.semantic_emb.view(-1, 768)
                if hasattr(batch_data, "semantic_emb")
                else torch.zeros(batch_data.y.size(0), 768, device=device)
            )
            symbolic_evidence = (
                batch_data.symbolic_evidence.view(-1, EVIDENCE_DIM)
                if hasattr(batch_data, "symbolic_evidence")
                else torch.zeros(batch_data.y.size(0), EVIDENCE_DIM, device=device)
            )
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits, _, _, _, _ = model(
                    batch_data.x, batch_data.edge_index, batch_data.batch,
                    semantic_emb, symbolic_evidence,
                )
            test_logits_all.append(logits.float().cpu())
            test_labels_all.append(batch_data.y.cpu().numpy())
            test_evidence_all.append(symbolic_evidence.float().cpu().numpy())

    test_logits = torch.cat(test_logits_all).numpy()
    test_labels = np.concatenate(test_labels_all)
    test_evidence = np.concatenate(test_evidence_all)
    test_probs = _softmax(test_logits)

    # For S3: apply hard post-hoc correction
    if variant == "S3":
        print("  Applying hard post-hoc correction...")
        corrected_logits = apply_hard_posthoc(
            torch.tensor(test_logits),
            torch.tensor(test_evidence),
            orig_behavior_map,
            orig_exclusion_matrix,
            tau_b=0.5,
        ).numpy()
        test_probs_corrected = _softmax(corrected_logits)
    else:
        test_probs_corrected = test_probs

    # Use ORIGINAL ontology for evaluation (we always measure violations
    # against the true constraints, even if training used corrupted ones)
    eval_behavior_map = orig_behavior_map.numpy()
    eval_exclusion_matrix = orig_exclusion_matrix.numpy()

    # Identify minority classes (bottom 25% by count)
    class_counts = np.bincount(test_labels, minlength=num_classes)
    threshold = np.percentile(class_counts[class_counts > 0], 25)
    minority_classes = [c for c in range(num_classes) if 0 < class_counts[c] <= threshold]

    # Run full evaluation
    metrics = evaluate_all(
        test_labels, test_probs_corrected, test_evidence,
        eval_behavior_map, eval_exclusion_matrix,
        minority_classes=minority_classes,
    )

    # =========================================================================
    # Step 7: Zero-Shot Retrieval
    # =========================================================================
    print("\n--- Step 7: Zero-Shot Retrieval ---")
    zs_results = _evaluate_zero_shot(
        model, test_loader, embedder, label_encoder, device, use_amp,
    )
    metrics.update({f"zs_{k}": v for k, v in zs_results.items()})

    # =========================================================================
    # Compile results
    # =========================================================================
    print(f"\n  {variant} seed={seed}: Top-1={metrics['top1']:.4f} "
          f"Macro-F1={metrics['macro_f1']:.4f} ECE={metrics['ece']:.4f} "
          f"ViolRate={metrics['total_viol_rate']:.4f} "
          f"EntCorr={metrics['entropy_violation_corr']:.4f}")

    result = {
        "variant": variant,
        "seed": seed,
        "noise_level": noise_level if variant == "S5" else None,
        "training": {
            "train_time_seconds": train_time,
            "best_val_acc": best_val_acc,
            "epochs_trained": epoch + 1,
        },
        "metrics": metrics,
    }

    # Save per-run results
    out_path = os.path.join(results_dir, f"result_{variant}_s{seed}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {out_path}")

    return result


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def _evaluate_zero_shot(model, test_loader, embedder, label_encoder, device, use_amp):
    """Zero-shot retrieval via hyperbolic distance."""
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
                    batch.x, batch.edge_index, batch.batch, semantic_emb,
                )
            all_hyp_graph.append(hyp_graph.float().cpu())
            all_labels.append(batch.y.cpu().numpy())

    all_hyp_graph = torch.cat(all_hyp_graph)
    all_labels = np.concatenate(all_labels)

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

    curvature = model.aligner.curvature
    dist_matrix = hyp_distance_batch(all_hyp_graph, hyp_domains, curvature)

    top1_preds = dist_matrix.argmin(dim=1).numpy()
    top1 = accuracy_score(all_labels, top1_preds)

    top5_preds = dist_matrix.argsort(dim=1)[:, :5].numpy()
    top5 = np.mean([l in t for l, t in zip(all_labels, top5_preds)])

    top10_preds = dist_matrix.argsort(dim=1)[:, :10].numpy()
    top10 = np.mean([l in t for l, t in zip(all_labels, top10_preds)])

    return {"top1": float(top1), "top5": float(top5), "top10": float(top10)}
