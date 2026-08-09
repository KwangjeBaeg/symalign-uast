"""Exercise the public Experiment 1--3 implementations without the full corpus."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.exp1_robustness.robustness_runner import (
    CONDITIONS,
    _collect_identifier_tokens,
    _collect_library_tokens,
    apply_obfuscation,
    build_code_texts,
)
from experiments.exp2_geometry.geometry_experiment import build_model
from experiments.exp3_symbolic.hard_posthoc import apply_hard_posthoc
from experiments.exp3_symbolic.ontology_corruption import get_corrupted_ontology
from src.knowledge.domain_ontology import DomainOntology
from src.losses.symbolic_losses import get_symbolic_loss
from src.models.gnn_encoder import UASTGraphEncoder


def run() -> dict:
    """Run deterministic branch-level checks for the three experiment designs."""
    torch.manual_seed(42)

    records = [
        {"file_id": "sample_1", "libraries": ["numpy"], "comments": []},
        {"file_id": "sample_2", "libraries": ["flask"], "comments": []},
    ]
    texts = build_code_texts(records, str(REPOSITORY_ROOT / "data" / "missing"))
    texts = [f"{text}\ndef handler_{index}(): pass" for index, text in enumerate(texts)]
    library_tokens = _collect_library_tokens(texts, ["numpy", "flask"])
    identifier_tokens = _collect_identifier_tokens(texts, set(library_tokens))
    all_tokens = sorted(set(library_tokens) | set(identifier_tokens))
    exp1 = {}
    for condition in CONDITIONS:
        transformed, mapping = apply_obfuscation(
            condition, texts, library_tokens, all_tokens, seed=42,
        )
        if condition == "C0" and transformed != texts:
            raise AssertionError("C0 must preserve clean text.")
        if condition != "C0" and not mapping:
            raise AssertionError(f"{condition} must apply a non-empty mapping.")
        exp1[condition] = len(mapping)

    x = torch.tensor(
        [[1.0, 0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0, 0],
         [0, 0, 1.0, 0, 0, 0], [0, 0, 0, 1.0, 0, 0]],
    )
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
    batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    semantic_embeddings = torch.zeros(2, 768)
    exp2 = {}
    for variant in ("E1", "E2", "E3", "E4"):
        encoder = UASTGraphEncoder() if variant != "E4" else None
        model = build_model(variant, encoder, num_classes=2).eval()
        with torch.no_grad():
            logits, contrast_loss, aligned = model(
                x, edge_index, batch, semantic_embeddings,
            )
        if logits.shape != (2, 2) or not torch.isfinite(contrast_loss):
            raise AssertionError(f"{variant} produced invalid output.")
        if variant in ("E3", "E4") and aligned.norm(dim=-1).max() >= 1.0:
            raise AssertionError(f"{variant} left the Poincare ball.")
        exp2[variant] = {
            "logit_shape": list(logits.shape),
            "poincare_norm": float(aligned.norm(dim=-1).max()),
        }

    labels = LabelEncoder().fit(["Data Processing", "Web Development"])
    ontology = DomainOntology()
    behavior_map = ontology.get_domain_behavior_map(labels)
    exclusion_matrix = ontology.get_exclusion_matrix(labels)
    probabilities = torch.tensor([[0.7, 0.3], [0.4, 0.6]])
    evidence = torch.full((2, 14), 0.5)
    exp3 = {}
    for variant in ("S0", "S1", "S2", "S4"):
        variant_behavior, variant_exclusion = get_corrupted_ontology(
            variant, behavior_map, exclusion_matrix, seed=42,
        )
        loss, _ = get_symbolic_loss(
            variant, variant_behavior, variant_exclusion,
        )(probabilities, evidence)
        if not torch.isfinite(loss):
            raise AssertionError(f"{variant} produced a non-finite loss.")
        exp3[variant] = float(loss)
    corrected = apply_hard_posthoc(
        torch.log(probabilities), evidence, behavior_map, exclusion_matrix,
    )
    if corrected.shape != probabilities.shape:
        raise AssertionError("S3 post-hoc correction changed tensor shape.")

    return {"experiment_1": exp1, "experiment_2": exp2, "experiment_3": exp3}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON summary path.")
    arguments = parser.parse_args()
    summary = run()
    rendered = json.dumps(summary, indent=2)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n")
        print(f"Wrote experiment smoke-test summary to {arguments.output}")
    else:
        print(rendered)
    print("Experiment 1--3 smoke tests completed successfully.")


if __name__ == "__main__":
    main()