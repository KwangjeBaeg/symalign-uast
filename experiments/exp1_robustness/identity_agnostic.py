"""
Identity-Agnostic Robustness Experiment

Compares TF-IDF (text-based) vs GNN (structure-based) models under
progressive identifier obfuscation.

Key Design (Train-Clean / Test-Obfuscated):
    - TRAIN the TF-IDF model on clean (unobfuscated) data
    - TEST the trained model on progressively obfuscated data
    - This simulates the real-world scenario: a model trained on normal code
      encounters obfuscated/refactored code with unfamiliar identifiers
    - The GNN (structure-only, no identifiers) is inherently invariant

Obfuscation levels: 0%, 25%, 50%, 75%, 100% of unique library names
replaced with random tokens at TEST TIME.
"""

import json
import os
import pickle
import random
import string
import time
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report


def generate_random_token(length=8):
    """Generate a random alphanumeric token."""
    return "LIB_" + "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


def get_labels(dataset: list, library_to_concept: dict) -> list:
    """Assign domain labels to each file."""
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


def obfuscate_documents(
    documents: list,
    all_unique_libs: list,
    obfuscation_ratio: float,
    seed: int = 42,
) -> Tuple[list, dict]:
    """
    Obfuscate library names in test documents.

    Args:
        documents: List of space-separated library strings
        all_unique_libs: List of all unique library names (from training vocab)
        obfuscation_ratio: Fraction of unique libraries to obfuscate
        seed: Random seed

    Returns:
        obfuscated_documents: Documents with replaced library names
        mapping: Dict mapping original -> obfuscated names
    """
    rng = random.Random(seed)

    num_to_obfuscate = int(len(all_unique_libs) * obfuscation_ratio)
    libs_to_obfuscate = set(rng.sample(all_unique_libs, num_to_obfuscate))

    mapping = {}
    for lib in libs_to_obfuscate:
        mapping[lib] = generate_random_token()

    obfuscated = []
    for doc in documents:
        tokens = doc.split()
        new_tokens = [mapping.get(t, t) for t in tokens]
        obfuscated.append(" ".join(new_tokens))

    return obfuscated, mapping


def run_robustness_experiment(
    dataset: list,
    library_to_concept: dict,
    uast_dir: str,
    results_dir: str,
    phase2_model_path: str = None,
    phase2_label_encoder_path: str = None,
    device: str = "cpu",
    batch_size: int = 256,
):
    """
    Run the full Identity-Agnostic Robustness experiment.

    Protocol:
        1. Train TF-IDF + LogReg on CLEAN training data
        2. At each obfuscation level, obfuscate TEST documents only
        3. Evaluate the clean-trained model on obfuscated test data
        4. GNN (structure-only) is invariant: same result at all levels
    """
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 80)
    print("EXPERIMENT: Identity-Agnostic Robustness")
    print("  Protocol: Train on clean data → Test on obfuscated data")
    print("=" * 80)

    obfuscation_levels = [0.0, 0.25, 0.50, 0.75, 1.0]
    tfidf_results = {}
    gnn_results = {}

    # ---- Step 1: Prepare labels and documents ----
    labels = get_labels(dataset, library_to_concept)
    documents = [" ".join(item["libraries"]) for item in dataset]

    # Filter unknown
    valid_idx = [i for i, l in enumerate(labels) if l != "Unknown"]
    documents = [documents[i] for i in valid_idx]
    labels = [labels[i] for i in valid_idx]

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(labels)

    # Split (same seed for all levels)
    indices = np.arange(len(documents))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=y
    )

    train_docs = [documents[i] for i in train_idx]
    test_docs = [documents[i] for i in test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    # Collect all unique libraries (from full vocabulary)
    all_unique_libs = sorted(set(
        token for doc in documents for token in doc.split()
    ))
    print(f"  Total unique library tokens: {len(all_unique_libs)}")
    print(f"  Train samples: {len(train_docs)}, Test samples: {len(test_docs)}")
    print(f"  Number of classes: {len(le.classes_)}")

    # ---- Step 2: Train TF-IDF on CLEAN data ----
    print("\n--- Training TF-IDF on Clean Data ---")
    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8,
        token_pattern=r"\S+",
    )
    X_train = vectorizer.fit_transform(train_docs)
    clf = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    # Baseline (0% obfuscation)
    X_test_clean = vectorizer.transform(test_docs)
    y_pred_clean = clf.predict(X_test_clean)
    y_proba_clean = clf.predict_proba(X_test_clean)
    clean_top1 = accuracy_score(y_test, y_pred_clean)
    print(f"  Clean TF-IDF Top-1: {clean_top1:.2%}")

    # ---- Step 3: Test at each obfuscation level ----
    print("\n--- TF-IDF Robustness (train clean → test obfuscated) ---")
    for ratio in obfuscation_levels:
        if ratio == 0.0:
            obf_test_docs = test_docs
            mapping = {}
        else:
            obf_test_docs, mapping = obfuscate_documents(
                test_docs, all_unique_libs, ratio
            )

        X_test_obf = vectorizer.transform(obf_test_docs)
        y_pred = clf.predict(X_test_obf)
        y_proba = clf.predict_proba(X_test_obf)

        top1 = accuracy_score(y_test, y_pred)

        k = min(5, y_proba.shape[1])
        top5_preds = np.argsort(y_proba, axis=1)[:, -k:]
        top5 = np.mean([l in t for l, t in zip(y_test, top5_preds)])

        mrr_scores = []
        for i in range(len(y_test)):
            ranked = np.argsort(y_proba[i])[::-1]
            rank = np.where(ranked == y_test[i])[0][0] + 1
            mrr_scores.append(1.0 / rank)
        mrr = np.mean(mrr_scores)

        tfidf_results[f"{ratio:.2f}"] = {
            "top1": float(top1),
            "top5": float(top5),
            "mrr": float(mrr),
            "num_obfuscated": len(mapping),
        }
        print(f"  Obfuscation {ratio:.0%}: Top-1={top1:.2%} | Top-5={top5:.2%} | MRR={mrr:.4f} ({len(mapping)} libs masked)")

    # ---- Step 4: GNN evaluation (structure-only, obfuscation-invariant) ----
    print("\n--- GNN Robustness (structure-only) ---")

    if phase2_model_path and os.path.exists(phase2_model_path):
        import torch
        from src.models.gnn_encoder import UASTClassifier
        from src.data.subtree_extractor import UASTSubtreeExtractor, UAST_NODE_TYPES
        from experiments.train_gnn import prepare_labels, graphs_to_pyg
        from torch_geometric.data import DataLoader
        import torch.nn.functional as F

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
        model.load_state_dict(torch.load(phase2_model_path, weights_only=True, map_location=device))
        model = model.to(device)
        model.eval()

        # Prepare graph data
        extractor = UASTSubtreeExtractor(k_hop=3)
        phase2_cache = os.path.join(os.path.dirname(phase2_model_path), "graph_cache_k3_subTrue.pkl")
        cache_path = phase2_cache if os.path.exists(phase2_cache) else os.path.join(results_dir, "graph_cache_robustness.pkl")

        graphs = extractor.process_dataset(dataset, uast_dir, cache_path=cache_path)

        all_labels = get_labels(dataset, library_to_concept)
        file_to_label = {item["file_id"]: all_labels[i] for i, item in enumerate(dataset)}

        valid_graphs = []
        valid_labels_gnn = []
        for g in graphs:
            fid = g.get("file_id")
            if fid and fid in file_to_label and file_to_label[fid] != "Unknown":
                valid_graphs.append(g)
                valid_labels_gnn.append(file_to_label[fid])

        pyg_data = graphs_to_pyg(valid_graphs, valid_labels_gnn, label_encoder)

        gnn_indices = list(range(len(pyg_data)))
        gnn_encoded = label_encoder.transform(valid_labels_gnn)
        _, gnn_test_idx = train_test_split(
            gnn_indices, test_size=0.2, random_state=42, stratify=gnn_encoded
        )
        test_data = [pyg_data[i] for i in gnn_test_idx]
        test_loader = DataLoader(test_data, batch_size=batch_size)

        all_preds = []
        all_labels_arr = []
        all_probs = []
        use_amp = device == "cuda"

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits, _ = model(batch.x, batch.edge_index, batch.batch)
                probs = F.softmax(logits, dim=-1)
                all_preds.append(logits.argmax(dim=-1).cpu().numpy())
                all_labels_arr.append(batch.y.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels_arr = np.concatenate(all_labels_arr)
        all_probs = np.concatenate(all_probs)

        gnn_top1 = accuracy_score(all_labels_arr, all_preds)
        k = min(5, all_probs.shape[1])
        top5_preds = np.argsort(all_probs, axis=1)[:, -k:]
        gnn_top5 = np.mean([l in t for l, t in zip(all_labels_arr, top5_preds)])

        mrr_scores = []
        for i in range(len(all_labels_arr)):
            ranked = np.argsort(all_probs[i])[::-1]
            rank = np.where(ranked == all_labels_arr[i])[0][0] + 1
            mrr_scores.append(1.0 / rank)
        gnn_mrr = np.mean(mrr_scores)

        gnn_metric = {
            "top1": float(gnn_top1),
            "top5": float(gnn_top5),
            "mrr": float(gnn_mrr),
        }
        print(f"  GNN (constant): Top-1={gnn_top1:.2%} | Top-5={gnn_top5:.2%} | MRR={gnn_mrr:.4f}")

        for ratio in obfuscation_levels:
            gnn_results[f"{ratio:.2f}"] = gnn_metric
    else:
        print("  Phase 2 model not found. Skipping GNN evaluation.")

    # ---- Step 5: Analysis ----
    print("\n--- Degradation Analysis ---")
    tfidf_baseline_top1 = tfidf_results["0.00"]["top1"]
    print(f"  TF-IDF baseline (0% obf): {tfidf_baseline_top1:.2%}")

    degradation = {}
    for ratio in obfuscation_levels:
        key = f"{ratio:.2f}"
        tfidf_top1 = tfidf_results[key]["top1"]
        relative_drop = (tfidf_baseline_top1 - tfidf_top1) / max(tfidf_baseline_top1, 1e-8)
        degradation[key] = {
            "tfidf_top1": tfidf_top1,
            "tfidf_relative_drop": float(relative_drop),
        }
        if gnn_results:
            degradation[key]["gnn_top1"] = gnn_results[key]["top1"]
            degradation[key]["gnn_relative_drop"] = 0.0

        print(f"  {ratio:.0%}: TF-IDF={tfidf_top1:.2%} (drop={relative_drop:.1%})", end="")
        if gnn_results:
            print(f" | GNN={gnn_results[key]['top1']:.2%} (drop=0%)")
        else:
            print()

    # Find crossover point
    crossover = None
    if gnn_results:
        gnn_top1_val = gnn_results["0.00"]["top1"]
        for ratio in obfuscation_levels:
            key = f"{ratio:.2f}"
            if tfidf_results[key]["top1"] <= gnn_top1_val:
                crossover = ratio
                break
        if crossover is not None:
            print(f"\n  ** GNN surpasses TF-IDF at {crossover:.0%} obfuscation! **")
        else:
            print(f"\n  TF-IDF still above GNN at 100% obfuscation.")
            print(f"  However, TF-IDF dropped {tfidf_baseline_top1 - tfidf_results['1.00']['top1']:.2%} while GNN is stable.")

    # Save
    results = {
        "experiment": "identity_agnostic_robustness",
        "protocol": "train_clean_test_obfuscated",
        "obfuscation_levels": obfuscation_levels,
        "tfidf_results": tfidf_results,
        "gnn_results": gnn_results,
        "degradation": degradation,
        "crossover_point": crossover,
    }

    results_path = os.path.join(results_dir, "robustness_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results
