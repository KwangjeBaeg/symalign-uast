"""Recompute experiment summaries from frozen author results, without model jobs."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[2]
MODEL_NAMES = {
    "B1": "GraphCodeBERT-base", "B2": "CodeBERT-base", "B3": "TF-IDF + LR",
    "B4": "GNN-only (structure)", "B5": "SymAlign (geometry-only)",
}
GROUPS = ("indist", "zero_shot", "geometry")


def load_json(path: Path):
    # Historical NumPy exports contain NaN for undefined norm correlations.
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda _: None)


def stats(values):
    """Match the historical aggregators' population SD (NumPy ddof=0)."""
    finite = [v for v in values if isinstance(v, (int, float))
              and not isinstance(v, bool) and math.isfinite(v)]
    return {"mean": statistics.fmean(finite) if finite else None,
            "std": statistics.pstdev(finite) if finite else None}


def verify_files(data_dir: Path) -> list[dict]:
    manifest = load_json(data_dir / "manifest.json")
    receipts = []
    seen = set()
    for item in manifest["files"]:
        name = item["file"]
        path = (data_dir / name).resolve()
        if not path.is_relative_to(data_dir.resolve()) or name in seen:
            raise ValueError(f"Invalid or duplicate manifest path: {name}")
        seen.add(name)
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if len(content) != item["bytes"] or digest != item["sha256"]:
            raise ValueError(f"Frozen result failed integrity check: {name}")
        receipts.append({"file": name, "sha256": digest})
    if not receipts:
        raise ValueError("Frozen-result manifest is empty")
    return receipts


def _group_runs(runs: list[dict], groups=GROUPS) -> dict:
    grouped = defaultdict(list)
    for run in runs:
        grouped[run["variant"]].append(run)
    summary = {}
    for variant, records in sorted(grouped.items()):
        seeds = [r["seed"] for r in records]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Duplicate seed for {variant}")
        row = {"seeds": sorted(seeds), "n_runs": len(records)}
        for group in groups:
            # Retrieval-confusion matrices are retained in the source JSON,
            # but are not scalar metrics to be averaged here.
            keys = sorted(k for k, value in records[0][group].items()
                          if value is None or isinstance(value, (int, float)))
            row[group] = {k: stats([r[group][k] for r in records]) for k in keys}
        if groups == GROUPS:
            row["converged"] = sum(bool(r["training"]["converged"]) for r in records)
        summary[variant] = row
    return summary


def _summary_differences(computed: dict, saved: dict, label: str) -> list[dict]:
    differences = []

    def has_statistics(value):
        return isinstance(value, dict) and (
            ("mean" in value and "std" in value)
            or any(has_statistics(v) for v in value.values()))

    def walk(actual, expected, path):
        if not isinstance(expected, dict):
            return
        if not isinstance(actual, dict):
            differences.append({"path": path, "saved": expected, "recomputed": actual})
            return
        if "mean" in expected and "std" in expected:
            for field in ("mean", "std"):
                if field not in actual:
                    differences.append({"path": f"{path}.{field}",
                                        "saved": expected[field], "recomputed": "missing"})
                    continue
                a, e = actual[field], expected[field]
                equal = (a is None and e is None) or (
                    a is not None and e is not None
                    and math.isclose(a, e, rel_tol=1e-12, abs_tol=1e-12))
                if not equal:
                    differences.append({"path": f"{path}.{field}", "saved": e, "recomputed": a})
            return
        for key, value in expected.items():
            if key in actual:
                walk(actual[key], value, f"{path}.{key}")
            elif has_statistics(value):
                differences.append({"path": f"{path}.{key}",
                                    "saved": value, "recomputed": "missing"})

    walk(computed, saved, label)
    return differences


def _robustness(source: dict) -> dict:
    runs = source["per_seed"]
    if len({r["seed"] for r in runs}) != len(runs):
        raise ValueError("Duplicate robustness seed")
    summary = {}
    for variant in MODEL_NAMES:
        conditions = {}
        for condition in ("C0", "C1", "C2", "C3", "C4"):
            conditions[condition] = {
                metric: stats([r["model_results"][variant][condition][metric] for r in runs])
                for metric in ("top1", "macro_f1", "weighted_f1", "mrr")}
        clean = [r["model_results"][variant]["C0"]["top1"] for r in runs]
        stripped = [r["model_results"][variant]["C4"]["top1"] for r in runs]
        summary[variant] = {
            "name": MODEL_NAMES[variant], "seeds": sorted(r["seed"] for r in runs),
            "n_runs": len(runs), "conditions": conditions,
            "robustness": {"retention_ratio": stats([s / c if c else None
                                                       for c, s in zip(clean, stripped)]),
                           "drop_0_to_100": stats([c - s for c, s in zip(clean, stripped)])}}
    return summary


def paper_differences(report: dict, reference: dict) -> list[dict]:
    differences = []
    for cell in reference["cells"]:
        actual = report
        for key in cell["path"].split("."):
            actual = actual[key]
        rendered = None if actual is None else f"{actual * cell.get('scale', 1):.{cell['digits']}f}"
        if rendered != cell["printed"]:
            differences.append({"path": cell["path"], "printed": cell["printed"],
                                "recomputed": rendered, "note": cell.get("note"),
                                "documented": "note" in cell})
    return differences


def _value(pair, digits=3, scale=1, with_sd=True):
    if pair["mean"] is None:
        return "—"
    mean = f"{pair['mean'] * scale:.{digits}f}"
    return f"{mean} ± {pair['std'] * scale:.{digits}f}" if with_sd else mean


def _table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join(["---"] * len(headers)) + " |"]
                     + ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def render_tables(report: dict) -> str:
    rob = report["robustness"]
    text = ["# Recomputed experiment summaries", "",
            "Aggregated from frozen author result records; no training or inference was run.", "",
            "## Identifier robustness (Table 1 and full-condition appendix)", "",
            _table(["Model", "C0 (%)", "C1 (%)", "C2 (%)", "C3 (%)", "C4 (%)", "C4/C0"],
                   [[r["name"]] + [_value(r["conditions"][c]["top1"], 2, 100, False)
                                   for c in ("C0", "C1", "C2", "C3", "C4")]
                    + [_value(r["robustness"]["retention_ratio"], 3, 1, False)]
                    for r in rob.values()]), "",
            "Five seed records. B4 reuses the same pretrained GNN checkpoint; its repeated scores are not independent fits.", "",
            "## Geometry performance (Table 2)", "",
            "E2/E3: balanced-stopping follow-up. E1/E4: original accuracy-stopping runs.", ""]
    geometry = report["geometry_selected"]
    text += [_table(["Variant", "ID Top-1 (%)", "ZS Top-1 (%)", "MAP@5", "kNN-10", "Converged"],
                    [[v, _value(r["indist"]["top1"], 1, 100),
                      _value(r["zero_shot"]["zs_top1"], 1, 100),
                      _value(r["zero_shot"]["map5"]), _value(r["zero_shot"]["knn10"]),
                      f"{r['converged']}/{r['n_runs']}"] for v, r in geometry.items()]), "",
             "## Geometry diagnostics (Table 3 comparison)", "",
             _table(["Variant", "Separation ratio", "Silhouette", "Norm–depth correlation"],
                    [[v] + [_value(r["geometry"][k]) for k in
                            ("sep_ratio", "silhouette", "norm_depth_corr")]
                     for v, r in geometry.items()]), "",
             "The E2/E3 silhouette cells differ from Table 3, which retained the earlier accuracy-stopping values. Both full run sets remain in reproduction.json.", "",
             "## Symbolic regularization (Table 4 and additional controls)", "",
             _table(["Variant", "Top-1 (%)", "ECE", "Violation rate", "Entropy correlation", "ZS Top-1 (%)"],
                    [[v, _value(r["metrics"]["top1"], 2, 100, False),
                      _value(r["metrics"]["ece"], 4, 1, False),
                      _value(r["metrics"]["total_viol_rate"], 4, 1, False),
                      _value(r["metrics"]["entropy_violation_corr"], 3, 1, False),
                      _value(r["metrics"]["zs_top1"], 2, 100, False)]
                     for v, r in report["symbolic"].items()]), "",
             "Three seeds per symbolic variant. SD values are available in reproduction.json.", "",
             "## Comparison with printed paper values", ""]
    diffs = report["paper_comparison"]["differences"]
    text += [_table(["Cell", "Paper", "Recomputed", "Explanation"],
                    [[d["path"], d["printed"], d["recomputed"], d["note"] or "Unexplained"]
                     for d in diffs]) if diffs else "All registered printed cells match.", "",
             "SD uses the historical population definition (ddof=0). Undefined correlations are null in JSON and a dash here.", ""]
    return "\n".join(text)


def reproduce(output: Path, data_dir: Path | None = None) -> dict:
    data_dir = Path(data_dir) if data_dir is not None else ROOT / "data" / "results"
    receipts = verify_files(data_dir)
    robust = load_json(data_dir / "pretrained_robustness_results.json")
    original = load_json(data_dir / "geometry_ablation_results.json")
    followup = load_json(data_dir / "exp1_followup_results.json")
    symbolic = load_json(data_dir / "exp03_all_results.json")
    original_summary = _group_runs(original["per_run"])
    followup_summary = _group_runs(followup["per_run"])
    robust_summary = _robustness(robust)
    differences = (_summary_differences(robust_summary, robust["summary"], "robustness")
                   + _summary_differences(original_summary, original["summary"], "geometry_original")
                   + _summary_differences(followup_summary, followup["summary"], "geometry_balanced"))
    if differences:
        raise ValueError(f"Recomputed summaries disagree with frozen aggregates: {differences}")
    selected = {v: followup_summary[v] if v in ("E2", "E3") else original_summary[v]
                for v in ("E1", "E2", "E3", "E4")}
    report = {"scope": "Aggregation of saved result records, not a model rerun or an evaluation from raw predictions.",
              "integrity": receipts, "saved_summary_differences": differences,
              "robustness": robust_summary, "geometry_original": original_summary,
              "geometry_balanced": followup_summary, "geometry_selected": selected,
              "symbolic": _group_runs(symbolic, groups=("metrics",))}
    reference = load_json(data_dir / "paper_reference.json")
    paper_diff = paper_differences(report, reference)
    report["paper_comparison"] = {"registered_cells": len(reference["cells"]), "differences": paper_diff}
    if any(not d["documented"] for d in paper_diff):
        raise ValueError(f"Unexpected difference from printed paper: {paper_diff}")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "reproduction.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "tables.md").write_text(render_tables(report), encoding="utf-8")
    return report
