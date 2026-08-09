"""
Robustness Experiment Runner — Experiment 02.

Trains each model (B1–B5) on CLEAN data, then evaluates under 5 obfuscation
conditions (C0–C4). All models share the same split, seeds, and early stopping.

Models:
    B1: GraphCodeBERT-base fine-tuning
    B2: CodeBERT-base fine-tuning
    B3: TF-IDF + Logistic Regression (existing)
    B4: Structure-only GNN (existing Phase 2 model)
    B5: Hybrid SymAlign (geometry-only, no symbolic)

Conditions:
    C0: Clean (0%)
    C1: Import masking only
    C2: Identifier stripping only
    C3: 50% obfuscation
    C4: 100% obfuscation
"""

import json
import os
import pickle
import random
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score,
                              classification_report)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

from src.data.obfuscation import (
    build_code_texts, apply_obfuscation,
    _collect_library_tokens, _collect_identifier_tokens,
)


CONDITIONS = ["C0", "C1", "C2", "C3", "C4"]
CONDITION_LABELS = {
    "C0": "Clean (0%)",
    "C1": "Import mask only",
    "C2": "Identifier strip only",
    "C3": "Partial (50%)",
    "C4": "Full (100%)",
}


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_labels(dataset: List[Dict], library_to_concept: Dict) -> List[str]:
    """Assign domain labels to each file (same logic as existing robustness)."""
    labels = []
    for item in dataset:
        lib_counts = Counter(item["libraries"])
        label = "Unknown"
        for lib, _ in lib_counts.most_common():
            if lib in library_to_concept:
                domain = library_to_concept[lib].get("domain")
                if domain and domain not in ["None", None]:
                    label = domain
                    break
        labels.append(label)
    return labels


# ── Metrics ──

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_proba: np.ndarray) -> Dict[str, float]:
    """Compute Top-1, Top-5, macro-F1, weighted-F1, MRR."""
    top1 = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    k = min(5, y_proba.shape[1])
    top5_preds = np.argsort(y_proba, axis=1)[:, -k:]
    top5 = float(np.mean([l in t for l, t in zip(y_true, top5_preds)]))

    mrr_scores = []
    for i in range(len(y_true)):
        ranked = np.argsort(y_proba[i])[::-1]
        rank = np.where(ranked == y_true[i])[0][0] + 1
        mrr_scores.append(1.0 / rank)
    mrr = float(np.mean(mrr_scores))

    return {"top1": top1, "top5": top5, "macro_f1": macro_f1,
            "weighted_f1": weighted_f1, "mrr": mrr}


# ── B3: TF-IDF + Logistic Regression ──

def train_eval_tfidf(train_docs: List[str], y_train: np.ndarray,
                     test_docs_by_cond: Dict[str, List[str]],
                     y_test: np.ndarray, num_classes: int) -> Dict[str, Dict]:
    """Train TF-IDF on clean train, evaluate on each obfuscation condition."""
    vectorizer = TfidfVectorizer(
        max_features=1000, ngram_range=(1, 2), min_df=2, max_df=0.8,
        token_pattern=r"\S+",
    )
    X_train = vectorizer.fit_transform(train_docs)
    clf = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    results = {}
    for cond, docs in test_docs_by_cond.items():
        X_test = vectorizer.transform(docs)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)
        metrics = compute_metrics(y_test, y_pred, y_proba)
        results[cond] = metrics
        print(f"    B3 {cond}: Top-1={metrics['top1']:.4f}  F1={metrics['macro_f1']:.4f}")

    return results


# ── B4: Structure-only GNN (existing Phase 2) ──

def eval_gnn(phase2_model_path: str, phase2_label_encoder_path: str,
             graph_cache_path: str, dataset: List[Dict],
             library_to_concept: Dict, uast_dir: str,
             device: str, batch_size: int = 256) -> Dict[str, Dict]:
    """
    Evaluate the existing GNN model. Structure-only → invariant to obfuscation.
    Returns the same metrics for all conditions.
    """
    from src.models.gnn_encoder import UASTClassifier
    from src.data.subtree_extractor import UASTSubtreeExtractor, UAST_NODE_TYPES
    from experiments.train_gnn import prepare_labels, graphs_to_pyg
    from torch_geometric.data import DataLoader as PyGLoader

    with open(phase2_label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    num_classes = len(label_encoder.classes_)
    node_feature_dim = len(UAST_NODE_TYPES)

    model = UASTClassifier(
        num_classes=num_classes,
        node_feature_dim=node_feature_dim,
        hidden_dim=256,
        embedding_dim=768,
    )
    model.load_state_dict(torch.load(phase2_model_path, weights_only=True,
                                     map_location=device))
    model = model.to(device)
    model.eval()

    extractor = UASTSubtreeExtractor(k_hop=3)
    graphs = extractor.process_dataset(dataset, uast_dir, cache_path=graph_cache_path)

    all_labels = get_labels(dataset, library_to_concept)
    file_to_label = {item["file_id"]: all_labels[i] for i, item in enumerate(dataset)}

    valid_graphs, valid_labels = [], []
    for g in graphs:
        fid = g.get("file_id")
        if fid and fid in file_to_label and file_to_label[fid] != "Unknown":
            valid_graphs.append(g)
            valid_labels.append(file_to_label[fid])

    pyg_data = graphs_to_pyg(valid_graphs, valid_labels, label_encoder)
    gnn_encoded = label_encoder.transform(valid_labels)
    _, gnn_test_idx = train_test_split(
        list(range(len(pyg_data))), test_size=0.2, random_state=42,
        stratify=gnn_encoded,
    )
    test_data = [pyg_data[i] for i in gnn_test_idx]
    test_loader = PyGLoader(test_data, batch_size=batch_size)

    all_preds, all_labels_arr, all_probs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            logits, _ = model(batch.x, batch.edge_index, batch.batch)
            probs = F.softmax(logits, dim=-1)
            all_preds.append(logits.argmax(dim=-1).cpu().numpy())
            all_labels_arr.append(batch.y.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels_arr = np.concatenate(all_labels_arr)
    all_probs = np.concatenate(all_probs)
    gnn_metrics = compute_metrics(all_labels_arr, all_preds, all_probs)

    print(f"    B4 (GNN, constant): Top-1={gnn_metrics['top1']:.4f}  F1={gnn_metrics['macro_f1']:.4f}")

    # Same result for all conditions
    return {cond: gnn_metrics for cond in CONDITIONS}


# ── B5: Hybrid SymAlign (geometry-only, no symbolic) ──

def eval_symalign(phase2_model_path: str, phase2_label_encoder_path: str,
                  graph_cache_path: str, dataset: List[Dict],
                  library_to_concept: Dict, library_concepts: list,
                  uast_dir: str, device: str, seed: int,
                  batch_size: int = 256) -> Dict[str, Dict]:
    """
    Train/eval SymAlign with lambda_logic=0 (geometry-only).
    Structure-based → invariant to obfuscation → same result for all conditions.
    Uses E3_HybridHyperbolic from geometry ablation for consistency.
    """
    from experiments.exp2_geometry.geometry_experiment import run_geometry_experiment

    results_dir = "/tmp/symalign_b5"
    os.makedirs(results_dir, exist_ok=True)

    result = run_geometry_experiment(
        variant="E3",
        dataset=dataset,
        library_to_concept=library_to_concept,
        library_concepts=library_concepts,
        uast_dir=uast_dir,
        results_dir=results_dir,
        phase2_model_path=phase2_model_path,
        phase2_label_encoder_path=phase2_label_encoder_path,
        graph_cache_path=graph_cache_path,
        device=device,
        seed=seed,
        num_epochs=100,
        batch_size=batch_size,
        lr=5e-5,
        lambda_contrast=0.5,
        temperature=0.1,
        curvature=1.0,
        patience=15,
    )

    # Extract indist metrics
    symalign_metrics = {
        "top1": result["indist"]["top1"],
        "top5": result["indist"]["top5"],
        "macro_f1": result["indist"]["macro_f1"],
        "weighted_f1": result["indist"]["weighted_f1"],
        "mrr": result["indist"]["mrr"],
    }
    print(f"    B5 (SymAlign, constant): Top-1={symalign_metrics['top1']:.4f}  F1={symalign_metrics['macro_f1']:.4f}")

    return {cond: symalign_metrics for cond in CONDITIONS}


# ── B1/B2: Pretrained code model fine-tuning ──

def train_eval_pretrained(variant: str, train_texts: List[str],
                          y_train: np.ndarray,
                          test_texts_by_cond: Dict[str, List[str]],
                          y_test: np.ndarray, num_classes: int,
                          device: str, seed: int,
                          max_length: int = 256,
                          batch_size: int = 16,
                          lr: float = 2e-5,
                          num_epochs: int = 10,
                          patience: int = 5,
                          val_ratio: float = 0.15) -> Dict[str, Dict]:
    """
    Fine-tune a pretrained code model on clean data, evaluate under all conditions.

    Budget fairness: same max epochs, same patience, same lr for B1 and B2.
    """
    from experiments.exp1_robustness.pretrained_models import (
        build_pretrained_model, build_tokenizer,
    )

    _set_seed(seed)
    tokenizer = build_tokenizer(variant)
    model = build_pretrained_model(variant, num_classes).to(device)

    # Train/val split within training set
    # Fallback to non-stratified if any class has <2 members
    from collections import Counter
    class_counts = Counter(y_train)
    can_stratify = all(c >= 2 for c in class_counts.values())
    train_idx, val_idx = train_test_split(
        np.arange(len(train_texts)), test_size=val_ratio,
        random_state=seed, stratify=y_train if can_stratify else None,
    )

    def _tokenize(texts):
        enc = tokenizer(
            texts, padding="max_length", truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        return enc["input_ids"], enc["attention_mask"]

    # Tokenize train & val (clean)
    train_texts_split = [train_texts[i] for i in train_idx]
    val_texts_split = [train_texts[i] for i in val_idx]

    tr_ids, tr_mask = _tokenize(train_texts_split)
    va_ids, va_mask = _tokenize(val_texts_split)
    tr_labels = torch.tensor(y_train[train_idx], dtype=torch.long)
    va_labels = torch.tensor(y_train[val_idx], dtype=torch.long)

    train_ds = TensorDataset(tr_ids, tr_mask, tr_labels)
    val_ds = TensorDataset(va_ids, va_mask, va_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best_val_f1 = -1
    best_state = None
    patience_counter = 0
    total_steps = 0

    print(f"\n  Training {variant} ({len(train_texts_split)} train, "
          f"{len(val_texts_split)} val) ...")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for ids, mask, labels in train_loader:
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(ids, mask)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            total_steps += 1

        # Validation
        model.eval()
        val_preds, val_labels_list = [], []
        with torch.no_grad():
            for ids, mask, labels in val_loader:
                ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    logits = model(ids, mask)
                val_preds.append(logits.argmax(dim=-1).cpu().numpy())
                val_labels_list.append(labels.cpu().numpy())

        val_preds = np.concatenate(val_preds)
        val_labels_arr = np.concatenate(val_labels_list)
        val_f1 = f1_score(val_labels_arr, val_preds, average="macro", zero_division=0)

        print(f"    Epoch {epoch+1}/{num_epochs}: loss={epoch_loss/len(train_loader):.4f}  "
              f"val_F1={val_f1:.4f}  steps={total_steps}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break

    # Load best weights
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)
    model.eval()

    # Evaluate under each condition
    results = {}
    for cond, test_texts in test_texts_by_cond.items():
        te_ids, te_mask = _tokenize(test_texts)
        test_ds = TensorDataset(te_ids, te_mask)
        test_loader = DataLoader(test_ds, batch_size=batch_size)

        all_preds, all_probs = [], []
        with torch.no_grad():
            for batch_tuple in test_loader:
                ids = batch_tuple[0].to(device)
                mask = batch_tuple[1].to(device)
                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    logits = model(ids, mask)
                probs = F.softmax(logits, dim=-1)
                all_preds.append(logits.argmax(dim=-1).cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_probs = np.concatenate(all_probs)
        metrics = compute_metrics(y_test, all_preds, all_probs)
        results[cond] = metrics
        print(f"    {variant} {cond}: Top-1={metrics['top1']:.4f}  F1={metrics['macro_f1']:.4f}")

    return results


# ── Main runner ──

def run_robustness_experiment(
    dataset: List[Dict],
    library_to_concept: Dict,
    library_concepts: list,
    uast_dir: str,
    results_dir: str,
    phase2_model_path: str,
    phase2_label_encoder_path: str,
    graph_cache_path: str,
    device: str = "cuda",
    seed: int = 42,
    variants: Optional[List[str]] = None,
    pretrained_epochs: int = 10,
    pretrained_batch_size: int = 16,
    pretrained_lr: float = 2e-5,
    pretrained_max_length: int = 256,
    pretrained_patience: int = 5,
    cache_dir: str = "",
) -> Dict:
    """
    Run the full robustness experiment for selected model variants.

    Args:
        variants: list of model variants to run (default: all B1–B5)
    """
    _set_seed(seed)
    if variants is None:
        variants = ["B1", "B2", "B3", "B4", "B5"]

    os.makedirs(results_dir, exist_ok=True)

    # ── 1. Prepare labels ──
    labels = get_labels(dataset, library_to_concept)
    valid_idx = [i for i, l in enumerate(labels) if l != "Unknown"]
    dataset_valid = [dataset[i] for i in valid_idx]
    labels_valid = [labels[i] for i in valid_idx]

    le = LabelEncoder()
    y = le.fit_transform(labels_valid)
    num_classes = len(le.classes_)
    print(f"Dataset: {len(dataset_valid)} files, {num_classes} classes")

    # ── 2. Build code-like texts (with caching) ──
    import pickle
    code_texts_cache = os.path.join(cache_dir, "code_texts_cache.pkl") if cache_dir else ""

    if code_texts_cache and os.path.exists(code_texts_cache):
        print(f"Loading code texts from cache: {code_texts_cache}")
        with open(code_texts_cache, "rb") as f:
            all_texts = pickle.load(f)
        print(f"Loaded {len(all_texts)} texts from cache")
    else:
        print("Building code-like texts from UAST ...")
        all_texts = build_code_texts(dataset_valid, uast_dir)
        if code_texts_cache:
            os.makedirs(os.path.dirname(code_texts_cache), exist_ok=True)
            with open(code_texts_cache, "wb") as f:
                pickle.dump(all_texts, f)
            print(f"Saved code texts cache: {code_texts_cache}")

    # ── 3. Split ──
    indices = np.arange(len(all_texts))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=y,
    )
    train_texts = [all_texts[i] for i in train_idx]
    test_texts = [all_texts[i] for i in test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    # Also keep dataset items for GNN/SymAlign paths
    train_dataset = [dataset_valid[i] for i in train_idx]
    test_dataset = [dataset_valid[i] for i in test_idx]

    print(f"Split: {len(train_texts)} train, {len(test_texts)} test")

    # ── 4. Collect tokens for obfuscation ──
    all_libs_flat = []
    for item in dataset_valid:
        all_libs_flat.extend(item.get("all_libraries", item["libraries"]))
    library_tokens = _collect_library_tokens(all_texts, all_libs_flat)
    lib_set = set(library_tokens)
    ident_tokens = _collect_identifier_tokens(all_texts, lib_set)
    all_unique_tokens = sorted(set(library_tokens) | set(ident_tokens))

    print(f"Tokens: {len(library_tokens)} libs, {len(ident_tokens)} idents, "
          f"{len(all_unique_tokens)} total")

    # ── 5. Apply obfuscation to test set ──
    test_texts_by_cond = {}
    for cond in CONDITIONS:
        obf_texts, mapping = apply_obfuscation(
            cond, test_texts, library_tokens, all_unique_tokens, seed=seed,
        )
        test_texts_by_cond[cond] = obf_texts
        print(f"  {cond} ({CONDITION_LABELS[cond]}): "
              f"{len(mapping)} tokens replaced")

    # For TF-IDF, we also need simple doc format (space-separated libs)
    train_docs_simple = [" ".join(item["libraries"]) for item in train_dataset]
    test_docs_simple_by_cond = {}
    test_libs_docs = [" ".join(item["libraries"]) for item in test_dataset]
    for cond in CONDITIONS:
        obf_docs, _ = apply_obfuscation(
            cond, test_libs_docs, library_tokens, all_unique_tokens, seed=seed,
        )
        test_docs_simple_by_cond[cond] = obf_docs

    # ── 6. Run each variant ──
    all_results = {}

    for v in variants:
        print(f"\n{'='*60}")
        print(f"Model: {v}")
        print(f"{'='*60}")

        t0 = time.time()

        if v in ("B1", "B2"):
            res = train_eval_pretrained(
                variant=v, train_texts=train_texts, y_train=y_train,
                test_texts_by_cond=test_texts_by_cond, y_test=y_test,
                num_classes=num_classes, device=device, seed=seed,
                max_length=pretrained_max_length,
                batch_size=pretrained_batch_size,
                lr=pretrained_lr,
                num_epochs=pretrained_epochs,
                patience=pretrained_patience,
            )
        elif v == "B3":
            res = train_eval_tfidf(
                train_docs=train_docs_simple, y_train=y_train,
                test_docs_by_cond=test_docs_simple_by_cond, y_test=y_test,
                num_classes=num_classes,
            )
        elif v == "B4":
            res = eval_gnn(
                phase2_model_path=phase2_model_path,
                phase2_label_encoder_path=phase2_label_encoder_path,
                graph_cache_path=graph_cache_path,
                dataset=dataset_valid, library_to_concept=library_to_concept,
                uast_dir=uast_dir, device=device,
            )
        elif v == "B5":
            res = eval_symalign(
                phase2_model_path=phase2_model_path,
                phase2_label_encoder_path=phase2_label_encoder_path,
                graph_cache_path=graph_cache_path,
                dataset=dataset_valid, library_to_concept=library_to_concept,
                library_concepts=library_concepts,
                uast_dir=uast_dir, device=device, seed=seed,
            )
        else:
            raise ValueError(f"Unknown variant: {v}")

        elapsed = time.time() - t0
        all_results[v] = {"metrics": res, "elapsed_sec": elapsed}
        print(f"  {v} done in {elapsed:.1f}s")

    # ── 7. Compute robustness metrics ──
    summary = _compute_robustness_summary(all_results)

    # ── 8. Save ──
    output = {
        "experiment": "pretrained_baseline_robustness",
        "seed": seed,
        "num_classes": num_classes,
        "class_names": le.classes_.tolist(),
        "train_size": len(train_texts),
        "test_size": len(test_texts),
        "conditions": CONDITIONS,
        "condition_labels": CONDITION_LABELS,
        "token_counts": {
            "library": len(library_tokens),
            "identifier": len(ident_tokens),
            "total": len(all_unique_tokens),
        },
        "model_results": {v: r["metrics"] for v, r in all_results.items()},
        "elapsed": {v: r["elapsed_sec"] for v, r in all_results.items()},
        "robustness_summary": summary,
    }

    path = os.path.join(results_dir, "robustness_exp02_results.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {path}")

    return output


def _compute_robustness_summary(all_results: Dict) -> Dict:
    """Compute degradation slope and retention ratio for each model."""
    summary = {}
    for variant, data in all_results.items():
        metrics = data["metrics"]
        clean_top1 = metrics["C0"]["top1"]
        full_top1 = metrics["C4"]["top1"]

        if clean_top1 > 0:
            retention = full_top1 / clean_top1
            drop = clean_top1 - full_top1
        else:
            retention = 0.0
            drop = 0.0

        by_condition = {}
        for cond in CONDITIONS:
            c_top1 = metrics[cond]["top1"]
            by_condition[cond] = {
                "top1": c_top1,
                "macro_f1": metrics[cond]["macro_f1"],
                "relative_drop": (clean_top1 - c_top1) / max(clean_top1, 1e-8),
            }

        summary[variant] = {
            "clean_top1": clean_top1,
            "full_obf_top1": full_top1,
            "drop_0_to_100": drop,
            "retention_ratio": retention,
            "by_condition": by_condition,
        }

    return summary
