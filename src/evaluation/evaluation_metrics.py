"""
Evaluation Metrics for Experiment 03: Symbolic Faithfulness.

Three layers:
  1. Performance: Top-1, Macro-F1, MRR, ZS-Top-1/5/10
  2. Symbolic validity: violation rates, mean violation magnitude, constraint satisfaction
  3. Reliability/Calibration: ECE, Brier, NLL, entropy-violation correlation
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score


# =============================================================================
# 1. Performance Metrics
# =============================================================================

def compute_performance_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    minority_classes: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Compute classification performance metrics.

    Args:
        y_true: [N] ground truth labels
        y_probs: [N, C] softmax probabilities
        minority_classes: list of class indices considered "rare"
    """
    y_pred = y_probs.argmax(axis=1)
    top1 = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # MRR
    mrr_scores = []
    for i in range(len(y_true)):
        ranked = np.argsort(y_probs[i])[::-1]
        rank = np.where(ranked == y_true[i])[0][0] + 1
        mrr_scores.append(1.0 / rank)
    mrr = float(np.mean(mrr_scores))

    result = {
        "top1": float(top1),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "mrr": mrr,
    }

    # Minority classes macro-F1
    if minority_classes:
        mask = np.isin(y_true, minority_classes)
        if mask.sum() > 0:
            minority_f1 = f1_score(
                y_true[mask], y_pred[mask],
                average="macro", zero_division=0,
                labels=minority_classes,
            )
            result["minority_macro_f1"] = float(minority_f1)

    return result


# =============================================================================
# 2. Symbolic Validity Metrics
# =============================================================================

def compute_violation_metrics(
    y_probs: np.ndarray,
    symbolic_evidence: np.ndarray,
    domain_behavior_map: np.ndarray,
    exclusion_matrix: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute rule violation rates and magnitudes.

    Uses the predicted class (argmax) for per-prediction violations.

    Returns:
        dict with structural_viol_rate, behavioral_viol_rate, exclusion_viol_rate,
        mean_struct_violation, mean_behav_violation, total_viol_rate, validity_at_1
    """
    if y_pred is None:
        y_pred = y_probs.argmax(axis=1)

    N = len(y_pred)
    num_behav = domain_behavior_map.shape[1]
    behavioral = symbolic_evidence[:, :num_behav]
    structural = symbolic_evidence[:, num_behav:]

    structural_violations = 0
    behavioral_violations = 0
    exclusion_violations = 0
    struct_magnitudes = []
    behav_magnitudes = []
    validity_count = 0  # predictions that violate NONE of the constraints

    expected_counts = domain_behavior_map.sum(axis=1).clip(min=1.0)

    for i in range(N):
        pred_class = y_pred[i]
        p_pred = y_probs[i, pred_class]

        # Structural violation: p > structural_support
        s_support = structural[i].mean()
        struct_viol = max(0, p_pred - s_support)
        if struct_viol > 0.01:  # small threshold to avoid noise
            structural_violations += 1
        struct_magnitudes.append(struct_viol)

        # Behavioral violation: p > behavior_match
        b_match = (behavioral[i] @ domain_behavior_map[pred_class]) / expected_counts[pred_class]
        b_match = min(1.0, max(0.0, b_match))
        behav_viol = max(0, p_pred - b_match)
        if behav_viol > 0.01:
            behavioral_violations += 1
        behav_magnitudes.append(behav_viol)

        # Exclusion violation: top-2 are mutually exclusive
        top2 = np.argsort(y_probs[i])[-2:]
        if exclusion_matrix[top2[0], top2[1]] > 0 or exclusion_matrix[top2[1], top2[0]] > 0:
            exclusion_violations += 1

        # Validity: no violations at all
        if struct_viol <= 0.01 and behav_viol <= 0.01:
            # Check exclusion too
            if not (exclusion_matrix[top2[0], top2[1]] > 0 or exclusion_matrix[top2[1], top2[0]] > 0):
                validity_count += 1

    return {
        "structural_viol_rate": structural_violations / N,
        "behavioral_viol_rate": behavioral_violations / N,
        "exclusion_viol_rate": exclusion_violations / N,
        "total_viol_rate": (structural_violations + behavioral_violations + exclusion_violations) / (3 * N),
        "mean_struct_violation": float(np.mean(struct_magnitudes)),
        "mean_behav_violation": float(np.mean(behav_magnitudes)),
        "validity_at_1": validity_count / N,
    }


# =============================================================================
# 3. Reliability / Calibration Metrics
# =============================================================================

def compute_ece(y_true: np.ndarray, y_probs: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error (ECE)."""
    confidences = y_probs.max(axis=1)
    predictions = y_probs.argmax(axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() > 0:
            avg_conf = confidences[mask].mean()
            avg_acc = accuracies[mask].mean()
            ece += mask.sum() * abs(avg_acc - avg_conf)

    return float(ece / len(y_true))


def compute_brier_score(y_true: np.ndarray, y_probs: np.ndarray) -> float:
    """Multi-class Brier score."""
    N, C = y_probs.shape
    one_hot = np.zeros_like(y_probs)
    one_hot[np.arange(N), y_true] = 1.0
    return float(((y_probs - one_hot) ** 2).sum(axis=1).mean())


def compute_nll(y_true: np.ndarray, y_probs: np.ndarray) -> float:
    """Negative log-likelihood."""
    eps = 1e-10
    probs_true = y_probs[np.arange(len(y_true)), y_true]
    return float(-np.log(probs_true.clip(min=eps)).mean())


def compute_entropy_violation_correlation(
    y_probs: np.ndarray,
    symbolic_evidence: np.ndarray,
    domain_behavior_map: np.ndarray,
    exclusion_matrix: np.ndarray,
) -> float:
    """
    Pearson correlation between prediction entropy and total violation magnitude.

    Key metric from the paper: negative correlation means model is uncertain
    when constraints are violated (desirable for reliability).
    """
    # Entropy
    eps = 1e-10
    entropy = -(y_probs * np.log(y_probs + eps)).sum(axis=1)

    # Total violation magnitude per sample
    N = len(y_probs)
    num_behav = domain_behavior_map.shape[1]
    behavioral = symbolic_evidence[:, :num_behav]
    structural = symbolic_evidence[:, num_behav:]
    expected_counts = domain_behavior_map.sum(axis=1).clip(min=1.0)

    violations = np.zeros(N)
    for i in range(N):
        pred_class = y_probs[i].argmax()
        p_pred = y_probs[i, pred_class]

        s_support = structural[i].mean()
        violations[i] += max(0, p_pred - s_support)

        b_match = (behavioral[i] @ domain_behavior_map[pred_class]) / expected_counts[pred_class]
        violations[i] += max(0, p_pred - min(1.0, max(0.0, b_match)))

        # Exclusion contribution
        top2 = np.argsort(y_probs[i])[-2:]
        if exclusion_matrix[top2[0], top2[1]] > 0:
            violations[i] += y_probs[i, top2[0]] * y_probs[i, top2[1]]

    if np.std(violations) < 1e-10 or np.std(entropy) < 1e-10:
        return 0.0

    corr, _ = stats.pearsonr(entropy, violations)
    return float(corr)


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    symbolic_evidence: np.ndarray,
    domain_behavior_map: np.ndarray,
    exclusion_matrix: np.ndarray,
) -> Dict[str, float]:
    """Compute all calibration/reliability metrics."""
    return {
        "ece": compute_ece(y_true, y_probs),
        "brier": compute_brier_score(y_true, y_probs),
        "nll": compute_nll(y_true, y_probs),
        "entropy_violation_corr": compute_entropy_violation_correlation(
            y_probs, symbolic_evidence, domain_behavior_map, exclusion_matrix,
        ),
    }


# =============================================================================
# Combined evaluation
# =============================================================================

def evaluate_all(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    symbolic_evidence: np.ndarray,
    domain_behavior_map: np.ndarray,
    exclusion_matrix: np.ndarray,
    minority_classes: Optional[List[int]] = None,
) -> Dict[str, float]:
    """Run all three evaluation layers and return combined dict."""
    perf = compute_performance_metrics(y_true, y_probs, minority_classes)
    viol = compute_violation_metrics(
        y_probs, symbolic_evidence, domain_behavior_map, exclusion_matrix,
    )
    cal = compute_calibration_metrics(
        y_true, y_probs, symbolic_evidence, domain_behavior_map, exclusion_matrix,
    )

    result = {}
    result.update(perf)
    result.update(viol)
    result.update(cal)
    return result
