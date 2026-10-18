"""
Unified CLI entry point for SymAlign-UAST experiments.

    python run.py <command> [options]

Commands:
    reproduce   Recompute tables from the bundled frozen experiment records.
                 Uses Python's standard library; no models or corpus required.
    sample      Run the bundled public demo on the 14 sample UAST graphs.
                 Needs no external data; this is what CI / a fresh clone can run.
    gnn         Train the structure-only GNN baseline (Table 1 "GNN-only").
    geometry    Run one E1-E4 geometry-ablation variant (Table 2).
    symbolic    Run one S0-S6 symbolic-regularization variant (Table 4).
    robustness  Run the full identifier-obfuscation robustness benchmark
                 (Table 1 / Appendix D, baselines B1-B5).
    symalign    Train the full SymAlign-UAST model end to end
                 (GAT encoder + hyperbolic alignment + symbolic regularization).

`reproduce` needs only Python. `sample` needs the demo dependencies. Training
commands operate on separately acquired corpus inputs. Point --dataset /
--uast-dir / --library-to-concept / --library-concepts at the formats described
in docs/data.md. The local default layout is data/corpus/; complete retraining
also requires the matching checkpoints, ontology, and research environment.

See docs/experiments.md for the original and follow-up configurations and
the distinction between saved-result aggregation and model reruns.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_DATASET = REPO_ROOT / "data" / "corpus" / "dataset.json"
DEFAULT_UAST_DIR = REPO_ROOT / "data" / "corpus" / "uast"
DEFAULT_LIBRARY_TO_CONCEPT = REPO_ROOT / "data" / "corpus" / "library_to_concept.json"
DEFAULT_LIBRARY_CONCEPTS = REPO_ROOT / "data" / "corpus" / "library_concepts.json"
DEFAULT_RESULTS_DIR = REPO_ROOT / "outputs"


def _load_json(path: Path):
    with Path(path).open() as file:
        return json.load(file)


def _add_corpus_args(sub: argparse.ArgumentParser, need_library_concepts: bool = True):
    sub.add_argument(
        "--dataset", type=Path, default=DEFAULT_DATASET,
        help="JSON file: list of {file_id, libraries, ...} dataset entries.",
    )
    sub.add_argument(
        "--uast-dir", type=Path, default=DEFAULT_UAST_DIR,
        help="Directory of parsed UAST JSON files, keyed by file_id.",
    )
    sub.add_argument(
        "--library-to-concept", type=Path, default=DEFAULT_LIBRARY_TO_CONCEPT,
        help="JSON file: library name -> {domain, ...} mapping.",
    )
    if need_library_concepts:
        sub.add_argument(
            "--library-concepts", type=Path, default=DEFAULT_LIBRARY_CONCEPTS,
            help="JSON file with library concept descriptions for semantic embedding.",
        )
    sub.add_argument(
        "--results-dir", type=Path, default=DEFAULT_RESULTS_DIR,
        help="Where to write checkpoints and result JSON files.",
    )
    sub.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    sub.add_argument("--seed", type=int, default=42)


def cmd_reproduce(args: argparse.Namespace) -> None:
    from src.evaluation.reproduce_results import reproduce

    report = reproduce(args.output, args.data_dir)
    if args.plots:
        from scripts.plot_results import plot_results
        plot_results(report, args.output / "figures")
    print(f"Verified {len(report['integrity'])} frozen result files.")
    print(f"Recomputed summaries; {len(report['paper_comparison']['differences'])} documented paper-cell differences.")
    print(f"Wrote tables and JSON to {args.output}")


def cmd_sample(args: argparse.Namespace) -> None:
    from scripts.run_sample_pipeline import run as run_sample

    summary = run_sample(REPO_ROOT / "data" / "samples")
    _write_summary(summary, args.output)
    print(f"Processed {summary['samples_processed']} bundled UAST samples successfully.")


def cmd_gnn(args: argparse.Namespace) -> None:
    from experiments.train_gnn import run_phase2_experiment

    dataset = _load_json(args.dataset)
    library_to_concept = _load_json(args.library_to_concept)
    result, _, _ = run_phase2_experiment(
        dataset=dataset,
        library_to_concept=library_to_concept,
        uast_dir=str(args.uast_dir),
        results_dir=str(args.results_dir),
        device=args.device,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    _write_summary(result, args.output)


def cmd_geometry(args: argparse.Namespace) -> None:
    from experiments.exp2_geometry.geometry_experiment import run_geometry_experiment

    dataset = _load_json(args.dataset)
    library_to_concept = _load_json(args.library_to_concept)
    library_concepts = _load_json(args.library_concepts)
    result = run_geometry_experiment(
        variant=args.variant,
        dataset=dataset,
        library_to_concept=library_to_concept,
        library_concepts=library_concepts,
        uast_dir=str(args.uast_dir),
        results_dir=str(args.results_dir),
        phase2_model_path=str(args.phase2_model_path),
        phase2_label_encoder_path=str(args.phase2_label_encoder_path),
        graph_cache_path=str(args.results_dir / f"graph_cache_{args.variant}.pkl"),
        device=args.device,
        seed=args.seed,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_contrast=args.lambda_contrast,
        temperature=args.temperature,
        curvature=args.curvature,
        patience=args.patience,
        val_criterion=args.val_criterion,
    )
    _write_summary(result, args.output)


def cmd_symbolic(args: argparse.Namespace) -> None:
    from experiments.exp3_symbolic.run_symbolic_faithfulness import (
        run_symbolic_faithfulness_experiment,
    )

    dataset = _load_json(args.dataset)
    library_to_concept = _load_json(args.library_to_concept)
    library_concepts = _load_json(args.library_concepts)
    result = run_symbolic_faithfulness_experiment(
        dataset=dataset,
        library_to_concept=library_to_concept,
        library_concepts=library_concepts,
        uast_dir=str(args.uast_dir),
        results_dir=str(args.results_dir),
        phase2_model_path=str(args.phase2_model_path),
        phase2_label_encoder_path=str(args.phase2_label_encoder_path),
        graph_cache_path=str(args.results_dir / "graph_cache_symbolic.pkl"),
        variant=args.variant,
        device=args.device,
        seed=args.seed,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_contrast=args.lambda_contrast,
        lambda_logic=args.lambda_logic,
        temperature=args.temperature,
        curvature=args.curvature,
        noise_level=args.noise_level,
        ontology_path=str(args.ontology_path) if args.ontology_path else None,
        cache_dir=str(args.results_dir),
    )
    _write_summary(result, args.output)


def cmd_robustness(args: argparse.Namespace) -> None:
    from experiments.exp1_robustness.robustness_runner import run_robustness_experiment

    dataset = _load_json(args.dataset)
    library_to_concept = _load_json(args.library_to_concept)
    library_concepts = _load_json(args.library_concepts)
    result = run_robustness_experiment(
        dataset=dataset,
        library_to_concept=library_to_concept,
        library_concepts=library_concepts,
        uast_dir=str(args.uast_dir),
        results_dir=str(args.results_dir),
        phase2_model_path=str(args.phase2_model_path),
        phase2_label_encoder_path=str(args.phase2_label_encoder_path),
        graph_cache_path=str(args.results_dir / "graph_cache_robustness.pkl"),
        device=args.device,
        seed=args.seed,
        variants=args.variants,
        cache_dir=str(args.results_dir),
    )
    _write_summary(result, args.output)


def cmd_symalign(args: argparse.Namespace) -> None:
    from experiments.run_symalign import run_symalign_experiment

    dataset = _load_json(args.dataset)
    library_to_concept = _load_json(args.library_to_concept)
    library_concepts = _load_json(args.library_concepts)
    result = run_symalign_experiment(
        dataset=dataset,
        library_to_concept=library_to_concept,
        library_concepts=library_concepts,
        uast_dir=str(args.uast_dir),
        results_dir=str(args.results_dir),
        phase2_model_path=str(args.phase2_model_path),
        phase2_label_encoder_path=str(args.phase2_label_encoder_path),
        ontology_path=str(args.ontology_path) if args.ontology_path else None,
        device=args.device,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_contrast=args.lambda_contrast,
        lambda_logic=args.lambda_logic,
        curvature=args.curvature,
        temperature=args.temperature,
    )
    _write_summary(result, args.output)


def _write_summary(summary: dict, output: Path) -> None:
    rendered = json.dumps(summary, indent=2, default=str)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n")
        print(f"Wrote summary to {output}")
    else:
        print(rendered)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_reproduce = sub.add_parser("reproduce", help="Recompute summaries from saved results (no ML dependencies).")
    p_reproduce.add_argument("--output", type=Path, default=REPO_ROOT / "outputs" / "reproduced")
    p_reproduce.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "results")
    p_reproduce.add_argument("--plots", action="store_true", help="Also draw summary plots; requires matplotlib.")
    p_reproduce.set_defaults(func=cmd_reproduce)

    p_sample = sub.add_parser("sample", help="Run the bundled public demo (no private data needed).")
    p_sample.add_argument("--output", type=Path, default=None)
    p_sample.set_defaults(func=cmd_sample)

    p_gnn = sub.add_parser("gnn", help="Train the structure-only GNN baseline.")
    _add_corpus_args(p_gnn, need_library_concepts=False)
    p_gnn.add_argument("--epochs", type=int, default=50)
    p_gnn.add_argument("--batch-size", type=int, default=32)
    p_gnn.add_argument("--lr", type=float, default=1e-3)
    p_gnn.add_argument("--output", type=Path, default=None)
    p_gnn.set_defaults(func=cmd_gnn)

    p_geo = sub.add_parser("geometry", help="Run one E1-E4 geometry-ablation variant.")
    _add_corpus_args(p_geo)
    p_geo.add_argument("--variant", required=True, choices=["E1", "E2", "E3", "E4"])
    p_geo.add_argument("--phase2-model-path", type=Path, required=True)
    p_geo.add_argument("--phase2-label-encoder-path", type=Path, required=True)
    p_geo.add_argument("--epochs", type=int, default=100)
    p_geo.add_argument("--batch-size", type=int, default=256)
    p_geo.add_argument("--lr", type=float, default=5e-5)
    p_geo.add_argument("--lambda-contrast", type=float, default=0.5)
    p_geo.add_argument("--temperature", type=float, default=0.1)
    p_geo.add_argument("--curvature", type=float, default=1.0)
    p_geo.add_argument("--patience", type=int, default=15)
    p_geo.add_argument("--val-criterion", default="balanced", choices=["accuracy", "balanced"])
    p_geo.add_argument("--output", type=Path, default=None)
    p_geo.set_defaults(func=cmd_geometry)

    p_sym = sub.add_parser("symbolic", help="Run one S0-S6 symbolic-regularization variant.")
    _add_corpus_args(p_sym)
    p_sym.add_argument(
        "--variant", required=True,
        choices=["S0", "S1", "S2", "S3", "S4", "S5", "S6_struct", "S6_behav", "S6_excl"],
    )
    p_sym.add_argument("--phase2-model-path", type=Path, required=True)
    p_sym.add_argument("--phase2-label-encoder-path", type=Path, required=True)
    p_sym.add_argument("--ontology-path", type=Path, default=None,
                        help="Defaults to the bundled sample ontology if omitted.")
    p_sym.add_argument("--epochs", type=int, default=100)
    p_sym.add_argument("--batch-size", type=int, default=256)
    p_sym.add_argument("--lr", type=float, default=5e-5)
    p_sym.add_argument("--lambda-contrast", type=float, default=0.5)
    p_sym.add_argument("--lambda-logic", type=float, default=0.3)
    p_sym.add_argument("--temperature", type=float, default=0.1)
    p_sym.add_argument("--curvature", type=float, default=1.0)
    p_sym.add_argument("--noise-level", type=float, default=0.2, help="Only used by S5.")
    p_sym.add_argument("--output", type=Path, default=None)
    p_sym.set_defaults(func=cmd_symbolic)

    p_rob = sub.add_parser("robustness", help="Run the full obfuscation robustness benchmark.")
    _add_corpus_args(p_rob)
    p_rob.add_argument("--phase2-model-path", type=Path, required=True)
    p_rob.add_argument("--phase2-label-encoder-path", type=Path, required=True)
    p_rob.add_argument(
        "--variants", nargs="+", default=None,
        help="Subset of B1-B5 to run (default: all).",
    )
    p_rob.add_argument("--output", type=Path, default=None)
    p_rob.set_defaults(func=cmd_robustness)

    p_full = sub.add_parser("symalign", help="Train the full SymAlign-UAST model end to end.")
    _add_corpus_args(p_full)
    p_full.add_argument("--phase2-model-path", type=Path, required=True)
    p_full.add_argument("--phase2-label-encoder-path", type=Path, required=True)
    p_full.add_argument("--ontology-path", type=Path, default=None,
                         help="Defaults to the bundled sample ontology if omitted.")
    p_full.add_argument("--epochs", type=int, default=100)
    p_full.add_argument("--batch-size", type=int, default=256)
    p_full.add_argument("--lr", type=float, default=5e-5)
    p_full.add_argument("--lambda-contrast", type=float, default=0.5)
    p_full.add_argument("--lambda-logic", type=float, default=0.3)
    p_full.add_argument("--curvature", type=float, default=1.0)
    p_full.add_argument("--temperature", type=float, default=0.1)
    p_full.add_argument("--output", type=Path, default=None)
    p_full.set_defaults(func=cmd_symalign)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command not in {"reproduce", "sample"}:
        paths = [("--dataset", args.dataset), ("--uast-dir", args.uast_dir),
                 ("--library-to-concept", args.library_to_concept)]
        for option in ("library_concepts", "phase2_model_path", "phase2_label_encoder_path", "ontology_path"):
            if getattr(args, option, None) is not None:
                paths.append(("--" + option.replace("_", "-"), getattr(args, option)))
        for option, path in paths:
            valid = path.is_dir() if option == "--uast-dir" else path.is_file()
            if not valid:
                parser.error(f"{option}: missing input {path}. See docs/data.md; the full corpus is acquired separately.")
        if args.command in {"gnn", "symalign"}:
            import random
            import numpy as np
            import torch
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
    args.func(args)


if __name__ == "__main__":
    main()
