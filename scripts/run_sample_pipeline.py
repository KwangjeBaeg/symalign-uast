"""Run the complete SymAlign-UAST forward path on the bundled UAST samples."""

import argparse
import json
import sys
from pathlib import Path

import torch
from sklearn.preprocessing import LabelEncoder

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data.subtree_extractor import UASTSubtreeExtractor
from src.data.symbolic_evidence import SymbolicEvidenceExtractor
from src.knowledge.domain_ontology import DomainOntology
from src.losses.neurosymbolic_loss import NeuroSymbolicLoss
from src.models.gnn_encoder import UASTGraphEncoder
from src.models.hyperbolic_aligner import HyperbolicContrastiveAligner
from src.models.symalign_classifier import SymAlignClassifier


def run(samples_dir: Path) -> dict:
    """Execute one deterministic forward pass per bundled UAST sample."""
    torch.manual_seed(42)
    with (samples_dir / "index.json").open() as file:
        samples = json.load(file)["samples"]

    label_encoder = LabelEncoder().fit(sorted({sample["domain"] for sample in samples}))
    ontology = DomainOntology()
    logic_loss = NeuroSymbolicLoss(
        ontology.get_domain_behavior_map(label_encoder),
        ontology.get_exclusion_matrix(label_encoder),
    )
    encoder = UASTGraphEncoder(hidden_dim=32, output_dim=64, num_heads=2)
    aligner = HyperbolicContrastiveAligner(
        graph_dim=64, semantic_dim=64, projection_dim=16,
    )
    model = SymAlignClassifier(
        encoder, aligner, logic_loss, num_classes=len(label_encoder.classes_),
        projection_dim=16,
    ).eval()

    extractor = UASTSubtreeExtractor()
    evidence_extractor = SymbolicEvidenceExtractor()
    results = []
    with torch.no_grad():
        for sample in samples:
            with (samples_dir / sample["file"]).open() as file:
                uast = json.load(file)
            node_ids, edges = extractor.extract_full_graph(uast)
            graph = extractor.uast_to_graph_data(uast, node_ids, edges)
            x = torch.from_numpy(graph["x"])
            edge_index = torch.from_numpy(graph["edge_index"])
            batch = torch.zeros(graph["num_nodes"], dtype=torch.long)
            semantic_embedding = torch.zeros(1, 64)
            evidence = torch.from_numpy(evidence_extractor.extract_from_uast(uast)).unsqueeze(0)
            logits, hyperbolic_graph, contrast, logic, components = model(
                x, edge_index, batch, semantic_embedding, evidence
            )
            ball_norm = float(hyperbolic_graph.norm(dim=-1).max())
            if ball_norm >= 1.0:
                raise RuntimeError(f"{sample['file']} is outside the Poincare ball: {ball_norm}")
            results.append({
                "file": sample["file"],
                "domain": sample["domain"],
                "nodes": graph["num_nodes"],
                "edges": int(edge_index.size(1)),
                "logit_shape": list(logits.shape),
                "poincare_norm": ball_norm,
                "contrastive_loss": float(contrast),
                "logic_loss": float(logic),
                "logic_components": components,
            })

    return {"samples_processed": len(results), "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples-dir", type=Path,
        default=REPOSITORY_ROOT / "data" / "samples",
        help="Directory containing index.json and bundled UAST samples.",
    )
    parser.add_argument(
        "--output", type=Path,
        help="Optional path for the JSON execution summary.",
    )
    arguments = parser.parse_args()
    summary = run(arguments.samples_dir)
    rendered_summary = json.dumps(summary, indent=2)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered_summary + "\n")
        print(f"Wrote sample-pipeline summary to {arguments.output}")
    else:
        print(rendered_summary)
    print(f"Processed {summary['samples_processed']} bundled UAST samples successfully.")


if __name__ == "__main__":
    main()