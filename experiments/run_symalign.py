"""
SymAlign-UAST Experiment Runner.

Mirrors run_hyperbolic_experiment() from experiments/hyperbolic/hyperbolic_alignment.py
with added symbolic evidence extraction and neuro-symbolic logic losses.

Usage (via Modal):
    modal run modal_symalign.py
"""

import json
import os
import pickle
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from torch_geometric.data import DataLoader

from experiments.hyperbolic.poincare_ops import hyp_distance_batch
from experiments.hyperbolic.hyperbolic_alignment import (
    HyperbolicContrastiveAligner,
)
from experiments.symalign.symbolic_evidence import (
    SymbolicEvidenceExtractor,
    EVIDENCE_DIM,
)
from experiments.symalign.domain_ontology import DomainOntology
from experiments.symalign.neurosymbolic_loss import NeuroSymbolicLoss
from experiments.symalign.symalign_classifier import SymAlignClassifier


def evaluate_zero_shot_symalign(
    model: SymAlignClassifier,
    test_loader: DataLoader,
    embedder,
    label_encoder: LabelEncoder,
    device: str,
    use_amp: bool = True,
):
    """
    Zero-shot retrieval using hyperbolic distance (same as hyperbolic experiment).
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
    domain_emb_tensor = torch.tensor(
        domain_emb_matrix, dtype=torch.float32
    ).to(device)

    with torch.no_grad():
        _, hyp_domains = model.aligner.get_aligned_embeddings(
            torch.zeros(len(domain_names), 768, device=device),
            domain_emb_tensor,
        )
        hyp_domains = hyp_domains.float().cpu()

    curvature = model.aligner.curvature
    dist_matrix = hyp_distance_batch(all_hyp_graph, hyp_domains, curvature)

    retrieval_preds = dist_matrix.argmin(dim=1).numpy()
    retrieval_acc = accuracy_score(all_labels, retrieval_preds)

    top5_preds = dist_matrix.argsort(dim=1)[:, :5].numpy()
    top5_acc = np.mean([l in t for l, t in zip(all_labels, top5_preds)])

    top10_preds = dist_matrix.argsort(dim=1)[:, :10].numpy()
    top10_acc = np.mean([l in t for l, t in zip(all_labels, top10_preds)])

    print(f"Zero-Shot Top-1:  {retrieval_acc:.2%}")
    print(f"Zero-Shot Top-5:  {top5_acc:.2%}")
    print(f"Zero-Shot Top-10: {top10_acc:.2%}")

    return {
        "retrieval_top1": float(retrieval_acc),
        "retrieval_top5": float(top5_acc),
        "retrieval_top10": float(top10_acc),
    }


def run_symalign_experiment(
    dataset: list,
    library_to_concept: dict,
    library_concepts: list,
    uast_dir: str,
    results_dir: str,
    phase2_model_path: str,
    phase2_label_encoder_path: str,
    ontology_path: Optional[str] = None,
    device: str = "cuda",
    num_epochs: int = 100,
    batch_size: int = 256,
    lr: float = 5e-5,
    lambda_contrast: float = 0.5,
    lambda_logic: float = 0.3,
    lambda_struct: float = 1.0,
    lambda_behav: float = 1.0,
    lambda_excl: float = 1.0,
    temperature: float = 0.1,
    projection_dim: int = 128,
    curvature: float = 1.0,
):
    """
    Run SymAlign-UAST experiment: GNN + Hyperbolic + Symbolic Logic Losses.
    """
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 80)
    print("EXPERIMENT: SymAlign-UAST (Neuro-Symbolic Logic-Constrained)")
    print(f"  curvature={curvature}, projection_dim={projection_dim}")
    print(f"  temperature={temperature}, lr={lr}")
    print(f"  lambda_contrast={lambda_contrast}, lambda_logic={lambda_logic}")
    print(f"  lambda_struct={lambda_struct}, lambda_behav={lambda_behav}, lambda_excl={lambda_excl}")
    print("=" * 80)

    # =========================================================================
    # Step 0: Pre-compute Symbolic Evidence
    # =========================================================================
    print("\n--- Step 0: Extracting Symbolic Evidence ---")
    sym_extractor = SymbolicEvidenceExtractor()
    evidence_cache = os.path.join(results_dir, "symbolic_evidence_cache.pkl")
    evidence_map = sym_extractor.process_dataset(
        dataset, uast_dir, cache_path=evidence_cache
    )

    # =========================================================================
    # Step 1: Load Phase 2 Artifacts
    # =========================================================================
    print("\n--- Step 1: Loading Phase 2 Artifacts ---")
    from experiments.phase2.gnn_encoder import UASTGraphEncoder, UASTClassifier
    from experiments.phase2.subtree_extractor import (
        UASTSubtreeExtractor,
        UAST_NODE_TYPES,
    )
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
    print(f"Loaded GNN encoder ({sum(p.numel() for p in gnn_encoder.parameters()):,} params)")

    # =========================================================================
    # Step 2: Generate Semantic Embeddings
    # =========================================================================
    print("\n--- Step 2: Generating Semantic Embeddings ---")
    from experiments.phase3.semantic_embedder import LibrarySemanticEmbedder

    embedder = LibrarySemanticEmbedder(device=device)
    lib_embeddings = embedder.embed_libraries(library_concepts)
    domain_embeddings = embedder.embed_domains(list(label_encoder.classes_))

    # =========================================================================
    # Step 3: Prepare Data (graphs + semantic + symbolic)
    # =========================================================================
    print("\n--- Step 3: Preparing Data ---")
    phase2_cache = os.path.join(
        os.path.dirname(phase2_model_path), "graph_cache_k3_subTrue.pkl"
    )
    cache_path = os.path.join(results_dir, "graph_cache_symalign.pkl")
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
            data.semantic_emb = torch.tensor(
                semantic_targets[i], dtype=torch.float32
            )

        fid = valid_file_ids[i] if i < len(valid_file_ids) else None
        if fid and fid in evidence_map:
            data.symbolic_evidence = torch.tensor(
                evidence_map[fid], dtype=torch.float32
            )
            evidence_attached += 1
        else:
            data.symbolic_evidence = torch.zeros(EVIDENCE_DIM, dtype=torch.float32)

    print(f"Symbolic evidence attached: {evidence_attached}/{len(pyg_data)}")

    # Split
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

    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    loader_kwargs = (
        {"pin_memory": True, "num_workers": 4} if device == "cuda" else {}
    )
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, **loader_kwargs
    )
    val_loader = DataLoader(val_data, batch_size=batch_size, **loader_kwargs)
    test_loader = DataLoader(test_data, batch_size=batch_size, **loader_kwargs)

    # =========================================================================
    # Step 4: Build SymAlign Model
    # =========================================================================
    print("\n--- Step 4: Building SymAlign Model ---")

    # Load domain ontology
    if ontology_path is None:
        ontology_path = os.path.join(
            os.path.dirname(__file__), "domain_ontology.json"
        )
    ontology = DomainOntology(ontology_path)
    ontology.print_summary(label_encoder)

    domain_behavior_map = ontology.get_domain_behavior_map(label_encoder)
    exclusion_matrix = ontology.get_exclusion_matrix(label_encoder)

    # Hyperbolic aligner
    aligner = HyperbolicContrastiveAligner(
        graph_dim=768,
        semantic_dim=768,
        projection_dim=projection_dim,
        curvature=curvature,
        temperature=temperature,
    )

    # Neuro-symbolic loss (individual rule weights for ablation)
    ns_loss = NeuroSymbolicLoss(
        domain_behavior_map=domain_behavior_map,
        exclusion_matrix=exclusion_matrix,
        lambda_struct=lambda_struct,
        lambda_behav=lambda_behav,
        lambda_excl=lambda_excl,
    )

    # Full model
    model = SymAlignClassifier(
        gnn_encoder=gnn_encoder,
        aligner=aligner,
        neurosymbolic_loss=ns_loss,
        num_classes=num_classes,
        projection_dim=projection_dim,
        lambda_contrast=lambda_contrast,
        lambda_logic=lambda_logic,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # =========================================================================
    # Step 5: Training
    # =========================================================================
    print("\n--- Step 5: Training ---")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    ce_criterion = nn.CrossEntropyLoss()

    use_amp = device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if use_amp:
        print("  Using mixed-precision (FP16) training")

    best_val_acc = 0
    patience_counter = 0
    patience = 15
    history = {
        "train_loss": [],
        "val_acc": [],
        "ce_loss": [],
        "contrast_loss": [],
        "logic_loss": [],
        "l_struct": [],
        "l_behav": [],
        "l_excl": [],
    }
    save_path = os.path.join(results_dir, "best_symalign_model.pt")

    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        epoch_totals = {
            "loss": 0, "ce": 0, "contrast": 0, "logic": 0,
            "struct": 0, "behav": 0, "excl": 0, "samples": 0,
        }

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
                else torch.zeros(
                    batch_data.y.size(0), EVIDENCE_DIM, device=device
                )
            )

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits, _, contrast_loss, logic_loss, logic_comps = model(
                    batch_data.x,
                    batch_data.edge_index,
                    batch_data.batch,
                    semantic_emb,
                    symbolic_evidence,
                )
                total_loss, loss_dict = model.compute_total_loss(
                    logits, batch_data.y, contrast_loss, logic_loss, ce_criterion
                )

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            bs = batch_data.y.size(0)
            epoch_totals["loss"] += total_loss.item() * bs
            epoch_totals["ce"] += loss_dict["ce_loss"] * bs
            epoch_totals["contrast"] += loss_dict["contrast_loss"] * bs
            epoch_totals["logic"] += loss_dict["logic_loss"] * bs
            epoch_totals["struct"] += logic_comps["l_struct"] * bs
            epoch_totals["behav"] += logic_comps["l_behav"] * bs
            epoch_totals["excl"] += logic_comps["l_excl"] * bs
            epoch_totals["samples"] += bs

        scheduler.step()
        n = epoch_totals["samples"]

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
                symbolic_evidence = (
                    batch_data.symbolic_evidence.view(-1, EVIDENCE_DIM)
                    if hasattr(batch_data, "symbolic_evidence")
                    else torch.zeros(
                        batch_data.y.size(0), EVIDENCE_DIM, device=device
                    )
                )
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits, _, _, _, _ = model(
                        batch_data.x,
                        batch_data.edge_index,
                        batch_data.batch,
                        semantic_emb,
                        symbolic_evidence,
                    )
                val_preds.append(logits.argmax(dim=-1).cpu().numpy())
                val_labels_list.append(batch_data.y.cpu().numpy())

        val_preds = np.concatenate(val_preds)
        val_labels_arr = np.concatenate(val_labels_list)
        val_acc = accuracy_score(val_labels_arr, val_preds)

        # Record history
        history["train_loss"].append(epoch_totals["loss"] / n)
        history["val_acc"].append(val_acc)
        history["ce_loss"].append(epoch_totals["ce"] / n)
        history["contrast_loss"].append(epoch_totals["contrast"] / n)
        history["logic_loss"].append(epoch_totals["logic"] / n)
        history["l_struct"].append(epoch_totals["struct"] / n)
        history["l_behav"].append(epoch_totals["behav"] / n)
        history["l_excl"].append(epoch_totals["excl"] / n)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:3d} | "
                f"Loss: {epoch_totals['loss']/n:.4f} | "
                f"CE: {epoch_totals['ce']/n:.4f} | "
                f"Contr: {epoch_totals['contrast']/n:.4f} | "
                f"Logic: {epoch_totals['logic']/n:.4f} "
                f"(S:{epoch_totals['struct']/n:.4f} "
                f"B:{epoch_totals['behav']/n:.4f} "
                f"E:{epoch_totals['excl']/n:.6f}) | "
                f"Val: {val_acc:.2%} | Best: {best_val_acc:.2%}"
            )

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    train_time = time.time() - start_time

    # Load best model
    model.load_state_dict(torch.load(save_path, weights_only=True))

    # =========================================================================
    # Step 6: Test Evaluation
    # =========================================================================
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
            symbolic_evidence = (
                batch_data.symbolic_evidence.view(-1, EVIDENCE_DIM)
                if hasattr(batch_data, "symbolic_evidence")
                else torch.zeros(
                    batch_data.y.size(0), EVIDENCE_DIM, device=device
                )
            )
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits, _, _, _, _ = model(
                    batch_data.x,
                    batch_data.edge_index,
                    batch_data.batch,
                    semantic_emb,
                    symbolic_evidence,
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

    # =========================================================================
    # Step 7: Zero-Shot Retrieval
    # =========================================================================
    print("\n--- Step 7: Zero-Shot Retrieval (Hyperbolic) ---")
    zero_shot_results = evaluate_zero_shot_symalign(
        model, test_loader, embedder, label_encoder, device, use_amp
    )

    # =========================================================================
    # Save Results
    # =========================================================================
    results = {
        "experiment": "symalign_uast",
        "config": {
            "curvature": curvature,
            "projection_dim": projection_dim,
            "temperature": temperature,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "lr": lr,
            "lambda_contrast": lambda_contrast,
            "lambda_logic": lambda_logic,
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
        "logic_loss_history": {
            "l_struct": history["l_struct"],
            "l_behav": history["l_behav"],
            "l_excl": history["l_excl"],
        },
    }

    results_path = os.path.join(results_dir, "symalign_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results
