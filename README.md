# SymAlign-UAST

Research code for **"SymAlign-UAST: Neuro-Symbolic Code Understanding via Hyperbolic Geometry and Structural Invariance."**

## Overview

SymAlign-UAST is a neuro-symbolic framework for robust code domain classification that operates on identifier-stripped Universal Abstract Syntax Trees (UASTs). It combines:

- **GAT encoder** on UAST graph topology (identifier-invariant structural encoding)
- **Hyperbolic contrastive alignment** in the Poincaré ball for hierarchy-aware domain representations
- **Differentiable symbolic regularization** via fuzzy logic constraints

## Run the bundled sample pipeline

The public package supports Python 3.10--3.12. Poetry creates and manages the
isolated environment for the bundled demonstration. If the machine has no
compatible Python interpreter, Poetry can install and select Python 3.12 first:

```bash
# Only needed when Python 3.10--3.12 is unavailable locally.
poetry python install 3.12
poetry env use 3.12
poetry install
poetry run python scripts/run_sample_pipeline.py --output results/sample_pipeline_summary.json
poetry run python scripts/run_experiment_smoke_tests.py --output results/experiment_smoke_summary.json
```

This CPU smoke test loads every bundled identifier-stripped UAST, one-hot
encodes its node types, executes the GAT encoder, projects the graph embedding
to the Poincare ball, and computes the symbolic logic loss. It writes a JSON
summary with tensor shapes, losses, and Poincare-ball norms. Deterministic
initialization (`torch.manual_seed(42)`) makes the demonstration repeatable.

`run_experiment_smoke_tests.py` also checks the public Experiment 1--3 code
paths without the full corpus: C0--C4 text transforms, E1--E4 model forwards,
and S0/S1/S2/S4 symbolic controls plus S3 post-hoc correction. It validates
implementation wiring, not the paper's reported metrics.

## Repository Structure

```text
symalign_uast/
├── src/
│   ├── models/          # GAT encoder, hyperbolic aligner, Poincaré ops
│   ├── data/            # UAST parser, obfuscation, symbolic evidence
│   ├── losses/          # Neuro-symbolic loss functions
│   ├── knowledge/       # Domain ontology, symbolic taxonomy
│   └── evaluation/      # Metrics, statistical analysis
├── experiments/
│   ├── exp1_robustness/ # Experiment 1: identity-agnostic robustness
│   ├── exp2_geometry/   # Experiment 2: geometry ablation
│   └── exp3_symbolic/   # Experiment 3: symbolic regularization
├── baselines/           # TF-IDF+LR, GNN-only, pretrained model baselines
├── data/samples/        # 14 representative UAST graphs spanning 11 domains
├── pyproject.toml        # Poetry environment for the public demo
├── requirements.txt      # Legacy research dependencies
└── scripts/              # Sample pipeline and experiment smoke tests
```

## Data and Reproducibility

This release includes only representative identifier-stripped UAST samples and
a synthetic obfuscation illustration. It does not distribute the complete
experimental corpus, train/test partitions, intermediate graph caches, trained
checkpoints, or semantic embeddings.

Consequently, `run_sample_pipeline.py` is an independently executable
implementation check, not a rerun of the paper's training protocol or a
reconstruction of reported metrics and tables. The experiment directories
preserve the implementations for the paper's evaluation settings, but their
full runs require the non-distributed corpus and experiment artifacts.

The bundled symbolic knowledge base (`src/knowledge/domain_ontology.json`) is
likewise a small illustrative sample for exercising the pipeline end-to-end,
not the full ontology used to produce the paper's reported tables.

## Experiments in the paper

**Experiment 1: Identity-Agnostic Robustness.** Five representation families
are trained on clean code and evaluated under five identifier-obfuscation
conditions, from clean input (C0) through complete identifier stripping (C4).
The primary comparison is the retention ratio C4/C0, which separates
structure-only invariance from surface-form accuracy when identifiers remain
available.

**Experiment 2: Geometric Generalization.** A controlled five-seed ablation
compares classifier-only, Euclidean-contrastive, hybrid-hyperbolic, and fully
hyperbolic variants. It measures in-distribution accuracy, zero-shot retrieval,
and geometry diagnostics such as class separation and norm-depth correlation
to isolate the effect of the retrieval space while holding the GAT encoder fixed
for E1--E3.

**Experiment 3: Symbolic Regularization Analysis.** With the hybrid
hyperbolic backbone fixed, this study compares no symbolic loss, soft
regularization variants, a shuffled-ontology control, and hard post-hoc
correction. It evaluates predictive accuracy alongside logic-violation rate,
calibration, entropy--violation alignment, and zero-shot retrieval.

For the broader research environment, install `requirements.txt`. Some legacy
preprocessing and full-experiment modules additionally expect unavailable raw
code inputs or experiment outputs; they are not public entrypoints.
