"""
Unified Evaluator for Geometry Ablation Experiment.

Computes three tiers of metrics:
  1. In-distribution classification: Top-1, Top-5, Macro-F1, Weighted-F1, MRR
  2. Zero-shot retrieval: ZS Top-1/5/10, ZS MRR, MAP@5, MAP@10, kNN-10
  3. Geometry diagnostics: separation ratio, silhouette, norm-depth correlation
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    silhouette_score,
)
from scipy.stats import spearmanr


# ── Tier 1: In-distribution classification ──────────────────────────────────

def compute_indist_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
) -> Dict[str, float]:
    """
    Args:
        labels: [N] int ground-truth class indices
        probs:  [N, C] softmax probabilities
    """
    preds = probs.argmax(axis=1)
    top1 = accuracy_score(labels, preds)

    k5 = min(5, probs.shape[1])
    top5_preds = np.argsort(probs, axis=1)[:, -k5:]
    top5 = np.mean([l in t for l, t in zip(labels, top5_preds)])

    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)

    # MRR
    mrr_scores = []
    for i in range(len(labels)):
        ranked = np.argsort(probs[i])[::-1]
        rank = int(np.where(ranked == labels[i])[0][0]) + 1
        mrr_scores.append(1.0 / rank)
    mrr = float(np.mean(mrr_scores))

    return {
        "top1": float(top1),
        "top5": float(top5),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "mrr": mrr,
    }


# ── Tier 2: Zero-shot retrieval ─────────────────────────────────────────────

def compute_zeroshot_metrics(
    dist_matrix: np.ndarray,
    labels: np.ndarray,
    smaller_is_better: bool = True,
) -> Dict[str, float]:
    """
    Args:
        dist_matrix: [N, D] distances/similarities to D domain prototypes
        labels:      [N] int ground-truth domain indices
        smaller_is_better: True for distances (hyperbolic), False for similarities (cosine)
    """
    N = len(labels)
    if smaller_is_better:
        # Closest = best → argsort ascending
        ranked = np.argsort(dist_matrix, axis=1)  # [N, D]
    else:
        # Highest = best → argsort descending
        ranked = np.argsort(-dist_matrix, axis=1)

    # Top-k accuracy
    top1 = float(np.mean(ranked[:, 0] == labels))
    top5 = float(np.mean([l in ranked[i, :5] for i, l in enumerate(labels)]))
    top10 = float(np.mean([l in ranked[i, :10] for i, l in enumerate(labels)]))

    # ZS MRR
    mrr_scores = []
    for i in range(N):
        rank = int(np.where(ranked[i] == labels[i])[0][0]) + 1
        mrr_scores.append(1.0 / rank)
    zs_mrr = float(np.mean(mrr_scores))

    # MAP@k
    map5 = _mean_avg_precision(ranked, labels, k=5)
    map10 = _mean_avg_precision(ranked, labels, k=10)

    # kNN-10 accuracy (leave-one-out on dist_matrix is not applicable here;
    # we compute kNN among test samples using their embeddings instead,
    # so this is handled separately via compute_knn_accuracy)

    return {
        "zs_top1": top1,
        "zs_top5": top5,
        "zs_top10": top10,
        "zs_mrr": zs_mrr,
        "map5": map5,
        "map10": map10,
    }


def _mean_avg_precision(
    ranked: np.ndarray, labels: np.ndarray, k: int
) -> float:
    """MAP@k for single-label retrieval (binary relevance: only 1 correct domain)."""
    ap_scores = []
    for i, true_label in enumerate(labels):
        hits = 0
        score = 0.0
        for j in range(min(k, ranked.shape[1])):
            if ranked[i, j] == true_label:
                hits += 1
                score += hits / (j + 1)
        ap_scores.append(score)  # either 0 or precision@rank
    return float(np.mean(ap_scores))


def compute_knn_accuracy(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k: int = 10,
    metric: str = "euclidean",
) -> float:
    """
    Leave-one-out kNN accuracy among test samples.

    Args:
        embeddings: [N, D]
        labels: [N]
        k: number of neighbors
        metric: 'euclidean' or 'cosine'
    """
    from sklearn.neighbors import KNeighborsClassifier

    if len(embeddings) < k + 1:
        return 0.0

    knn = KNeighborsClassifier(n_neighbors=k, metric=metric, algorithm="brute")
    # Leave-one-out
    correct = 0
    for i in range(len(embeddings)):
        mask = np.arange(len(embeddings)) != i
        knn.fit(embeddings[mask], labels[mask])
        pred = knn.predict(embeddings[i : i + 1])
        if pred[0] == labels[i]:
            correct += 1
    return float(correct / len(embeddings))


# ── Tier 3: Geometry diagnostics ────────────────────────────────────────────

def compute_geometry_diagnostics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    domain_names: Optional[List[str]] = None,
    behavioral_generality: Optional[Dict[str, int]] = None,
) -> Dict[str, float]:
    """
    Geometry diagnostics for embedding quality.

    Args:
        embeddings: [N, D] aligned embeddings (Euclidean or Poincaré tangent)
        labels: [N] int class indices
        domain_names: list of domain name strings (indexed by label)
        behavioral_generality: {domain_name: num_active_behaviors}
    """
    unique_labels = np.unique(labels)

    # Intra/inter-class distances
    intra_dists = []
    class_centroids = {}
    for lab in unique_labels:
        mask = labels == lab
        if mask.sum() < 2:
            continue
        pts = embeddings[mask]
        centroid = pts.mean(axis=0)
        class_centroids[lab] = centroid
        dists = np.linalg.norm(pts - centroid, axis=1)
        intra_dists.extend(dists.tolist())

    inter_dists = []
    centroid_list = list(class_centroids.values())
    for i in range(len(centroid_list)):
        for j in range(i + 1, len(centroid_list)):
            inter_dists.append(float(np.linalg.norm(centroid_list[i] - centroid_list[j])))

    mean_intra = float(np.mean(intra_dists)) if intra_dists else 0.0
    mean_inter = float(np.mean(inter_dists)) if inter_dists else 0.0
    sep_ratio = mean_inter / mean_intra if mean_intra > 1e-10 else 0.0

    # Silhouette score
    sil = -1.0
    if len(unique_labels) >= 2 and len(embeddings) > len(unique_labels):
        try:
            sil = float(silhouette_score(embeddings, labels, sample_size=min(2000, len(embeddings))))
        except Exception:
            sil = -1.0

    # Norm-depth correlation (Poincaré norms vs behavioral generality)
    norm_depth_corr = 0.0
    norm_depth_p = 1.0
    if domain_names and behavioral_generality:
        domain_norms = []
        domain_gen = []
        for lab in unique_labels:
            mask = labels == lab
            if mask.sum() == 0:
                continue
            pts = embeddings[mask]
            mean_norm = float(np.mean(np.linalg.norm(pts, axis=1)))
            dname = domain_names[lab] if lab < len(domain_names) else None
            if dname and dname in behavioral_generality:
                domain_norms.append(mean_norm)
                domain_gen.append(behavioral_generality[dname])
        if len(domain_norms) >= 4:
            # If all norms are constant (e.g. Euclidean space after L2-norm),
            # Spearman correlation is undefined — mark as NaN rather than warning.
            if np.std(domain_norms) < 1e-8:
                norm_depth_corr = float("nan")
                norm_depth_p = float("nan")
            else:
                corr, pval = spearmanr(domain_norms, domain_gen)
                norm_depth_corr = float(corr) if not np.isnan(corr) else float("nan")
                norm_depth_p = float(pval) if not np.isnan(pval) else float("nan")

    return {
        "intra_class_d": mean_intra,
        "inter_class_d": mean_inter,
        "sep_ratio": sep_ratio,
        "silhouette": sil,
        "norm_depth_corr": norm_depth_corr,
        "norm_depth_p": norm_depth_p,
    }
