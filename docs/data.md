# Data and model inputs

## Included

- Fourteen identifier-stripped UAST examples spanning eleven domains, indexed by [`data/samples/index.json`](../data/samples/index.json).
- A synthetic identifier-corruption illustration in `data/samples/obfuscation_demo.json`.
- Eight original experiment/diagnostic JSON files plus a printed-paper reference, described in the [reproduction guide](reproduction.md).
- An illustrative domain ontology at `src/knowledge/domain_ontology.json` and the behavioral extraction rules at `src/knowledge/symbolic_taxonomy.py`.

The sample ontology retains the existing public release's values. It is not a recovered versioned ontology for every reported training run. In particular, its domain inventory and exclusion pairs must not be assumed to match every corpus class or the paper's appendix merely because dimensions coincide.

## Full experiment inputs

The paper describes a 7,615-file Python corpus from 247 source repositories. The saved robustness records report 5,892 training and 1,473 test items (7,365 retained items). This release preserves those recorded counts; a complete manifest reconciling every preprocessing/filtering stage is not available in the public package. The complete corpus, source licences, split-ID manifests, model weights, and embedding caches are not redistributed here. The author archive alone is not a licence manifest for the collected repositories.

Training entry points accept explicit paths and otherwise use this local layout:

```text
data/corpus/
  dataset.json
  library_to_concept.json
  library_concepts.json
  uast/
    UAST_<file_id>.json
outputs/gnn/
  best_gnn_model.pt
  label_encoder.pkl
```

`data/corpus/` and generated checkpoints are ignored by Git. Acquire and prepare the source data under its original terms; do not substitute the 14 examples for the paper's training corpus.

### Dataset index

The runners expect a JSON list of processed records, rather than a directory of raw extraction summaries:

```json
[
  {"file_id": "example_001", "libraries": ["numpy", "scipy"], "comments": []}
]
```

The corresponding graph is `uast/UAST_example_001.json`. Library occurrences determine the domain label through the mapping below. The graph encoder's features use node types and topology; symbolic behavioral evidence is extracted separately from package/call signatures in the original UAST. Identifier invariance of the GAT does not imply that the separate symbolic evidence extractor is identifier-free.

### Library-to-domain mapping

`library_to_concept.json` is an object keyed by library name:

```json
{
  "numpy": {"domain": "Scientific Computing"},
  "scipy": {"domain": "Scientific Computing"}
}
```

### Library descriptions

`library_concepts.json` is a list, with the original capitalized field names used by `LibrarySemanticEmbedder`:

```json
[
  {
    "Library Name": "numpy",
    "Functionality": "numerical array computation",
    "Domain": "Scientific Computing",
    "Dependency Type": "external",
    "Platform": "cross-platform",
    "Framework": null
  }
]
```

These two JSON formats serve different functions and are deliberately separate CLI arguments. Domain text is embedded with `sentence-transformers/all-mpnet-base-v2` for the research runs. B1/B2 additionally download `microsoft/graphcodebert-base` and `microsoft/codebert-base`. No weights are downloaded by result aggregation or the sample pipeline.

## Preprocessing scope

The public pipeline starts from UAST JSON. The retained `src/data/uast_parser.py` is a legacy parser requiring an external `UAST` implementation, grammar files, and language rule sets; it is not an out-of-the-box source-to-UAST command. The repository does not replace it with a different parser and claim that this recreates the historical corpus.

`scripts/generate_library_concepts.py` is an optional authoring utility that calls an external LLM API when explicitly executed. It is not part of any quick-start command or the frozen-results workflow. Existing model checkpoints and Python pickle caches should come from a source you trust; none are needed for the CPU results path.
