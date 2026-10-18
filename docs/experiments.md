# Running the implementations

The saved-results workflow is the fastest entry point. Full model runs require the inputs described in the [data guide](data.md); the examples below show how to invoke the retained implementations on an acquired corpus. They are not claims that a fresh clone contains everything needed for exact retraining.

## Environment

Use Python 3.10–3.12 for model code. The lightweight demo dependencies match `pyproject.toml`:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-demo.txt
python run.py sample --output outputs/sample.json
```

Alternatively, the existing Poetry lock remains available:

```sh
poetry env use 3.12
poetry install
poetry run python run.py sample --output outputs/sample.json
```

`requirements.txt` retains the broader research environment, including the historical PyTorch 2.2.0, PyG 2.5.0, Transformers 4.44.2, and Sentence Transformers 3.0.1 versions. Use a separate Python 3.11 research environment for that file, with a PyTorch build appropriate to the machine. The historical cloud runners used an NVIDIA A10G; the public CPU sample does not need CUDA, `geoopt`, API credentials, or a sentence-transformer download. Dependency ranges for the demo are not an exact lock of the original GPU environment.

## Shared inputs

All commands run from the repository root. Paths below follow `docs/data.md`:

```sh
python run.py --help
python run.py geometry --help
python run.py symbolic --help
```

Training writes to `--results-dir`. Use a separate directory for each variant and seed so graph/evidence caches and checkpoints cannot be mistaken for another run. Seed flags initialize training randomness; some historical train/test splits remain fixed at 42. No command below launches automatically on import.

### GNN pretraining

```sh
python run.py gnn \
  --dataset data/corpus/dataset.json \
  --uast-dir data/corpus/uast \
  --library-to-concept data/corpus/library_to_concept.json \
  --device cuda --seed 42 \
  --epochs 50 --batch-size 32 --lr 1e-3 \
  --results-dir outputs/gnn --output outputs/gnn/summary.json
```

This produces the `best_gnn_model.pt` and `label_encoder.pkl` consumed below. The GNN runner uses a fixed stratified split with 20% test and 15% of the remaining data for validation (68/12/20 overall). Its seed flag controls initialization; it does not change those fixed split calls.

## Experiment 1: identifier robustness

```sh
python run.py robustness \
  --dataset data/corpus/dataset.json \
  --uast-dir data/corpus/uast \
  --library-to-concept data/corpus/library_to_concept.json \
  --library-concepts data/corpus/library_concepts.json \
  --phase2-model-path outputs/gnn/best_gnn_model.pt \
  --phase2-label-encoder-path outputs/gnn/label_encoder.pkl \
  --device cuda --seed 42 --variants B1 B2 B3 B4 B5 \
  --results-dir outputs/robustness/seed42 \
  --output outputs/robustness/seed42/summary.json
```

| Variant | Representation |
| --- | --- |
| B1 | `microsoft/graphcodebert-base` |
| B2 | `microsoft/codebert-base` |
| B3 | TF-IDF with logistic regression |
| B4 | Fixed pretrained structure-only GNN |
| B5 | E3-style hybrid hyperbolic model, without symbolic loss |

C0 preserves clean text; C1 masks imports; C2 strips other identifiers; C3 corrupts 50% of the collected tokens; C4 strips all collected identifiers. The saved seeds are 42, 123, 456, 789, and 2026. B4 reuses one checkpoint rather than being retrained for each seed. The implementation uses a fixed seed-42 robustness split; seed changes affect training/obfuscation paths as implemented. The GAT observes no identifier strings.

## Experiment 2: geometry ablation

```sh
python run.py geometry --variant E3 \
  --dataset data/corpus/dataset.json \
  --uast-dir data/corpus/uast \
  --library-to-concept data/corpus/library_to_concept.json \
  --library-concepts data/corpus/library_concepts.json \
  --phase2-model-path outputs/gnn/best_gnn_model.pt \
  --phase2-label-encoder-path outputs/gnn/label_encoder.pkl \
  --device cuda --seed 42 --val-criterion balanced \
  --epochs 100 --batch-size 256 --lr 5e-5 \
  --lambda-contrast 0.5 --temperature 0.1 --curvature 1.0 --patience 15 \
  --results-dir outputs/geometry/E3/seed42 \
  --output outputs/geometry/E3/seed42/summary.json
```

| Variant | Encoder and alignment |
| --- | --- |
| E1 | Euclidean GAT, CE classifier only |
| E2 | Euclidean GAT and Euclidean contrastive alignment |
| E3 | Euclidean GAT and hyperbolic contrastive alignment |
| E4 | Exploratory fully hyperbolic encoder |

Run each requested variant with seeds 42, 123, 456, 789, and 2026. Use `--val-criterion accuracy` for the original sweep and `balanced` for the E2/E3 follow-up. The CLI defaults to `balanced`; the underlying historical function defaults to `accuracy`, so always specify the criterion in an experiment command. See [run selection](reproduction.md#run-selection-and-historical-differences) before comparing with Tables 2–3. Geometry splits use the selected seed.

## Experiment 3: symbolic regularization

```sh
python run.py symbolic --variant S1 \
  --dataset data/corpus/dataset.json \
  --uast-dir data/corpus/uast \
  --library-to-concept data/corpus/library_to_concept.json \
  --library-concepts data/corpus/library_concepts.json \
  --phase2-model-path outputs/gnn/best_gnn_model.pt \
  --phase2-label-encoder-path outputs/gnn/label_encoder.pkl \
  --ontology-path /path/to/versioned/domain_ontology.json \
  --device cuda --seed 42 \
  --epochs 100 --batch-size 256 --lr 5e-5 \
  --lambda-contrast 0.5 --lambda-logic 0.3 --temperature 0.1 --curvature 1.0 \
  --results-dir outputs/symbolic/S1/seed42 \
  --output outputs/symbolic/S1/seed42/summary.json
```

| Variant | Symbolic treatment |
| --- | --- |
| S0 | No symbolic loss |
| S1 | Current soft regularization |
| S2 | Łukasiewicz-style alternative |
| S3 | Hard post-hoc correction |
| S4 | Shuffled ontology |
| S5 | Noisy ontology (`--noise-level`, default 0.2) |
| S6_struct / S6_behav / S6_excl | Individual symbolic components |

The stored symbolic study uses three seeds: 42, 123, and 456, with fixed seed-42 train/validation/test splits. Supply the correct versioned ontology for a research comparison. Omitting `--ontology-path` uses the illustrative sample ontology, which is suitable for a wiring check but does not establish exact paper reproducibility. Label indices must match the pretrained GNN's encoder.

## Full SymAlign training

`python run.py symalign --help` exposes the original full-model runner with the same corpus, checkpoint, ontology, and loss-weight arguments. It differs from the geometry-only B5 robustness row. Use a separate result directory and retain the command, seed, resolved environment, input hashes, and ontology version with each run. This repository update does not launch training or cloud GPU jobs.
