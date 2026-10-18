"""CPU-only checks for result provenance and aggregation contracts."""

import copy
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from src.evaluation.reproduce_results import (
    ROOT, _group_runs, _summary_differences, load_json, paper_differences,
    reproduce, stats, verify_files,
)


class ReproductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.report = reproduce(Path(cls.temp.name))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_matches_frozen_aggregates(self):
        self.assertEqual(self.report["saved_summary_differences"], [])

    def test_known_paper_differences_are_visible(self):
        differences = self.report["paper_comparison"]["differences"]
        self.assertEqual(len(differences), 4)
        self.assertTrue(all(d["documented"] and ".silhouette." in d["path"] for d in differences))

    def test_run_accounting(self):
        self.assertEqual(len(self.report["robustness"]), 5)
        self.assertEqual(sum(r["n_runs"] for r in self.report["geometry_original"].values()), 20)
        self.assertEqual(sum(r["n_runs"] for r in self.report["geometry_balanced"].values()), 10)
        self.assertEqual(sum(r["n_runs"] for r in self.report["symbolic"].values()), 27)

    def test_balanced_followup_is_not_mixed_with_original(self):
        self.assertEqual(self.report["geometry_selected"]["E3"], self.report["geometry_balanced"]["E3"])
        self.assertNotEqual(self.report["geometry_selected"]["E3"], self.report["geometry_original"]["E3"])

    def test_population_sd_and_missing_values(self):
        self.assertEqual(stats([1, 3, None, math.nan]), {"mean": 2.0, "std": 1.0})
        self.assertEqual(stats([None, math.nan]), {"mean": None, "std": None})

    def test_nested_robustness_summary_change_is_detected(self):
        actual = {"B1": {"conditions": {"C0": {"top1": stats([1, 3])}}}}
        saved = copy.deepcopy(actual)
        saved["B1"]["conditions"]["C0"]["top1"]["mean"] = 3
        differences = _summary_differences(actual, saved, "robustness")
        self.assertEqual(differences[0]["path"], "robustness.B1.conditions.C0.top1.mean")

    def test_duplicate_seeds_fail(self):
        run = {"variant": "S0", "seed": 42, "metrics": {"top1": 0.4}}
        with self.assertRaisesRegex(ValueError, "Duplicate seed"):
            _group_runs([run, run], groups=("metrics",))

    def test_missing_expected_metrics_fail_comparison(self):
        expected = {"B1": {"conditions": {"C0": {"top1": {"mean": 0.9, "std": 0.1}}}}}
        differences = _summary_differences({"B1": {}}, expected, "robustness")
        self.assertEqual(differences[0]["path"], "robustness.B1.conditions")
        self.assertEqual(differences[0]["recomputed"], "missing")

    def test_frozen_file_tampering_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary) / "results"
            shutil.copytree(ROOT / "data" / "results", data)
            with (data / "exp03_all_results.json").open("a") as stream:
                stream.write(" ")
            with self.assertRaisesRegex(ValueError, "integrity check"):
                verify_files(data)

    def test_undefined_correlations_are_strict_json_null(self):
        content = (Path(self.temp.name) / "reproduction.json").read_text()
        def reject(value):
            raise AssertionError(f"Nonstandard JSON constant: {value}")
        parsed = json.loads(content, parse_constant=reject)
        self.assertIsNone(parsed["geometry_selected"]["E2"]["geometry"]["norm_depth_corr"]["mean"])

    def test_unregistered_paper_drift_is_not_hidden(self):
        reference = {"cells": [{"path": "robustness.B1.conditions.C0.top1.mean",
                                 "printed": "0.00", "digits": 2}]}
        differences = paper_differences(self.report, reference)
        self.assertFalse(differences[0]["documented"])

    def test_repeated_aggregation_is_deterministic(self):
        with tempfile.TemporaryDirectory() as other:
            reproduce(Path(other))
            for name in ("tables.md", "reproduction.json"):
                self.assertEqual((Path(other) / name).read_bytes(),
                                 (Path(self.temp.name) / name).read_bytes())

    def test_cli_help_needs_no_site_packages(self):
        completed = subprocess.run([sys.executable, "-S", str(ROOT / "run.py"), "--help"],
                                   cwd=self.temp.name, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("reproduce", completed.stdout)


if __name__ == "__main__":
    unittest.main()
