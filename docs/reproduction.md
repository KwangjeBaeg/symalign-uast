# Reproducing the saved analyses

Run from the repository root with Python 3.10+:

```sh
python run.py reproduce --output outputs/reproduced
```

No packages, model downloads, GPU, credentials, or original corpus are required. The command verifies SHA-256 hashes in `data/results/manifest.json`, aggregates the stored per-run metrics, compares the aggregates with the historical summaries, and checks 93 registered numeric fields printed in Tables 1–4.

Outputs:

- `tables.md`: robustness, geometry performance, geometry diagnostics, symbolic controls, and printed-paper differences.
- `reproduction.json`: full recomputed summaries, seeds, integrity receipts, and comparison results. Undefined correlations are represented as JSON `null`.

This is result aggregation from recorded runs. It does not re-execute training, compute predictions, certify the original splits, or recompute metrics from prediction-level labels. The results retain the original aggregators' population standard deviation (`ddof=0`), rather than changing the uncertainty convention after publication.

## Which files support which results?

All paths below are under `data/results/`.

| Paper result | Saved source | What is recomputed |
| --- | --- | --- |
| Table 1 and full C0–C4 comparison | `pretrained_robustness_results.json` | Five model families across five condition scores, with five seed records; mean per-seed C4/C0 retention |
| Table 2, E1/E4 | `geometry_ablation_results.json` | Five seeds per variant, original accuracy-stopping runs |
| Table 2, E2/E3 | `exp1_followup_results.json` | Five seeds per variant, later balanced-stopping follow-up |
| Table 3 | The two geometry files above | Diagnostics using the same run selection as Table 2; the silhouette mismatch is reported below |
| Table 4 and S5/S6 controls | `exp03_all_results.json` | Nine symbolic variants, three seeds each |
| A1–A6 component study | `ablation_results.json` | Original individual-run records and loss histories are included for inspection; not a multi-seed aggregation |
| Earlier full-model run | `symalign_results.json` | Original individual-run configuration, metrics, and loss histories; distinct from the B5 geometry-only row |
| Supplementary diagnostics | `symalign_supplementary_local.json`, `symalign_supplementary_gpu.json` | Original author diagnostic records and embedding arrays, retained for inspection |

`paper_reference.json` transcribes selected printed cells and their displayed precision from the included author manuscript. It is a comparison target, not an experimental source. The manifest separately identifies this transcription. The eight original result files preserve the archive's bytes, including historical nonstandard `NaN` values; the reader converts undefined values to `null` in generated reports.

## Run selection and historical differences

The original geometry sweep used accuracy stopping. The follow-up changed E2/E3 to the balanced validation criterion, the harmonic mean of validation classification and prototype-retrieval Top-1. The main Table 2 E2/E3 values match the follow-up. E1/E4 records available in the archive are from the original sweep; the repository does not label them as new balanced-stopping reruns.

Table 3's E2/E3 silhouette values retained the original sweep's numbers. Using the follow-up consistently gives:

| Variant | Printed silhouette | Balanced follow-up silhouette |
| --- | --- | --- |
| E2 | −0.054 ± 0.009 | −0.078 ± 0.034 |
| E3 | −0.150 ± 0.047 | −0.137 ± 0.043 |

These four registered fields (two means and two SDs) are reported explicitly. The command does not replace the follow-up values with older cells to make the comparison appear exact. The other 89 registered numeric fields match at printed precision. Unprinted E1 zero-valued SDs and the undefined E2 norm correlation are excluded from this printed-field count. Both full sweep summaries remain in the output JSON. The accompanying paper and poster are preserved without edits.

The robustness experiment contains five seed records, but B4 evaluates the same saved GNN checkpoint on its fixed split. Its repeated invariant scores do not represent five independent retrainings. B5 in that experiment is the E3-style geometry-only model without symbolic loss; it is not the full symbolic S1 condition.

The supplementary GPU JSON labels its reference as `untrained_symalign_arch`: a trained GNN encoder with randomly initialized alignment/classifier heads. It should not be treated as the matched CE-only E1 control or combined with the S0–S6 study. The archived scatter/embedding diagnostics and the main symbolic-control table therefore have different provenance.

## Generate summary plots

```sh
python -m pip install -r requirements-reproduce.txt
python run.py reproduce --output outputs/reproduced --plots
```

This creates PNG and PDF plots under `outputs/reproduced/figures/`:

- `robustness`: all five identifier-corruption conditions.
- `geometry`: zero-shot accuracy and class separation, with population SD across five seeds.
- `symbolic`: classification and violation rates for S0–S4, with population SD across three seeds.

These are new plots of the archived summaries. The original paper figures are in [`assets/figures/`](../assets/figures/); recreating the exact Poincaré/PCA rendering additionally depends on the historical projection and plotting settings. A two-dimensional plot is a visualization, not an additional geometry evaluation.

## Checks

```sh
python -m unittest discover -s tests -v
python scripts/run_experiment_smoke_tests.py --output outputs/smoke.json
```

The unit tests require only Python. The second command requires the [demo environment](experiments.md#environment) and checks C0–C4 transformations, E1–E4 forward paths, and symbolic-control branches. Neither command reproduces full-corpus training.
