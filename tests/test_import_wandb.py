"""CPU-only importer tests; no W&B installation, credentials or network needed."""

import contextlib
import copy
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from bench import import_wandb as importer


ROOT = Path(__file__).resolve().parents[1]
RESULTS = sorted((ROOT / "results").glob("*/results.json"))


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "results.json"

    def write(self, result):
        self.path.write_text(json.dumps(result))
        return self.path

    def minimal(self):
        return {"meta": {}, "variants": {"test": {"latency_s": [2.0], "success": [False]}}}

    def test_existing_results_are_preserved(self):
        self.assertEqual(len(RESULTS), 2)
        all_ids = set()
        task_count = 0
        for path in RESULTS:
            before = path.read_bytes()
            result = json.loads(before)
            plans = importer.build_plans(path)
            self.assertEqual(len(plans), 4)
            for plan in plans:
                name = plan["config"]["variant"]
                variant = result["variants"][name]
                self.assertEqual(plan["config"]["benchmark"], result["meta"])
                self.assertEqual(plan["source_bytes"], before)
                self.assertTrue(plan["config"]["import"]["historical"])
                self.assertFalse(plan["config"]["import"]["experiments_executed"])
                table = plan["tables"]["tasks"]
                task_count += len(table["data"])
                for column in importer.TASK_FIELDS:
                    index = table["columns"].index(column)
                    self.assertEqual([row[index] for row in table["data"]], variant[column])
                for metric in ("latency_median_s", "latency_mean_s", "latency_std_s", "latency_p95_s",
                               "latency_max_s", "per_cem_iter_median_ms", "candidates_per_s",
                               "peak_mem_alloc_mb", "peak_mem_reserved_mb", "success_rate"):
                    self.assertEqual(plan["summary"][f"recorded/{metric}"], variant[metric])
                self.assertIs(plan["summary"]["selfcheck/fp32_within_3x_noise_floor"], False)
                for section, expected in (
                    ("scaling", [r for r in result["scaling"] if r["variant"] == name]),
                    ("profile_top_ops", result["profile_top_ops"][name]),
                ):
                    table = plan["tables"][section]
                    actual = [dict(zip(table["columns"], row)) for row in table["data"]]
                    self.assertEqual(actual, expected)
                self.assertNotIn(plan["id"], all_ids)
                all_ids.add(plan["id"])
            self.assertEqual(path.read_bytes(), before)
        self.assertEqual(task_count, 72)

    def test_ids_ignore_location_format_and_key_order_but_track_content(self):
        result = self.minimal()
        first = importer.build_plans(self.write(result))[0]
        moved = Path(self.temp.name) / "copy.json"
        moved.write_text(json.dumps(result, sort_keys=True, indent=4))
        self.assertEqual(first["id"], importer.build_plans(moved)[0]["id"])
        result["variants"]["test"]["latency_s"] = [3.0]
        self.assertNotEqual(first["id"], importer.build_plans(self.write(result))[0]["id"])
        result["variants"]["other"] = result["variants"].pop("test")
        self.assertNotEqual(first["id"], importer.build_plans(self.write(result))[0]["id"])

    def test_missing_fields_are_not_filled_or_derived(self):
        result = self.minimal()
        result["variants"]["test"]["peak_mem_alloc_mb"] = 0.0
        result["variants"]["test"]["future_field"] = {"unknown": [1, 2]}
        plan = importer.build_plans(self.write(result))[0]
        self.assertEqual(plan["tables"]["tasks"]["columns"], ["task_index", "latency_s", "success"])
        self.assertEqual(plan["summary"], {"recorded/peak_mem_alloc_mb": 0.0})
        self.assertEqual(plan["config"]["benchmark"], {})
        self.assertEqual(json.loads(plan["source_bytes"]), result)

    def test_oom_scaling_rows_preserve_missing_values(self):
        result = self.minimal()
        result["scaling"] = [{"variant": "test", "K": 1, "median_ms": 2.0},
                             {"variant": "test", "K": 1024, "oom": True}]
        table = importer.build_plans(self.write(result))[0]["tables"]["scaling"]
        rows = [dict(zip(table["columns"], row)) for row in table["data"]]
        self.assertIsNone(rows[1]["median_ms"])
        self.assertIs(rows[1]["oom"], True)
        self.assertIsNone(rows[0]["oom"])

    def test_malformed_input_is_rejected(self):
        changes = [
            {"success": []}, {"success": [1]}, {"latency_s": "2.0"},
            {"latency_s": [-1]}, {"latency_s": [float("nan")]},
            {"latency_s": [float("inf")]}, {"variant": "wrong"},
            {"cem_loss_trace": [2.0]},
        ]
        for change in changes:
            with self.subTest(change=change):
                result = self.minimal()
                result["variants"]["test"].update(change)
                with self.assertRaises(ValueError):
                    importer.build_plans(self.write(result))
        for key, value in (("n_tasks", 2), ("n_tasks", True)):
            result = self.minimal()
            result["meta"][key] = value
            with self.assertRaises(ValueError):
                importer.build_plans(self.write(result))
        for result in ([], {}, {"meta": {}, "variants": {}},
                       {**self.minimal(), "profile_top_ops": {"test": [1]}},
                       {**self.minimal(), "scaling": [{}]}):
            with self.subTest(result=result), self.assertRaises(ValueError):
                importer.build_plans(self.write(result))

    def test_dry_run_needs_no_site_packages_and_deduplicates(self):
        result = subprocess.run(
            [sys.executable, "-S", str(ROOT / "bench/import_wandb.py"), "--dry-run",
             "--results", *map(str, RESULTS), "--results", str(RESULTS[0].parent)],
            capture_output=True, text=True, check=True,
        )
        payload = json.loads(result.stdout)
        self.assertEqual(len(payload["runs"]), 8)
        self.assertEqual(payload["project"], "hpml-world-model-planning")

    def test_validates_all_inputs_before_loading_sdk(self):
        bad = self.write({})
        with patch.dict(sys.modules, {"wandb": None}), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                importer.main(["--results", str(RESULTS[0]), str(bad), "--entity", "test"])
        self.assertEqual(error.exception.code, 2)

    def sdk(self):
        sdk = MagicMock()
        sdk.Table.MAX_ARTIFACT_ROWS = 200000
        sdk.runs_by_id = {}

        def init(**kwargs):
            run = sdk.runs_by_id.get(kwargs["id"])
            if run is None:
                run = MagicMock()
                run.config = copy.deepcopy(kwargs["config"])
                run.summary = {}
                run.resumed = False
                sdk.runs_by_id[kwargs["id"]] = run
            else:
                run.resumed = True
            run.__enter__.return_value = run
            return run

        sdk.init.side_effect = init
        return sdk

    def test_upload_contract_and_repeated_import(self):
        sdk = self.sdk()
        plan = importer.build_plans(RESULTS[0])[0]
        snapshots = []
        sdk.Artifact.return_value.add_file.side_effect = lambda path, **kw: snapshots.append(Path(path).read_bytes())
        with contextlib.redirect_stdout(io.StringIO()):
            importer.upload_plan(plan, sdk, "team", "project")
            importer.upload_plan(plan, sdk, "team", "project")
        self.assertEqual(len(sdk.runs_by_id), 1)
        run = sdk.runs_by_id[plan["id"]]
        run.log.assert_called_once()
        run.log_artifact.assert_called_once()
        self.assertEqual(snapshots, [plan["source_bytes"]])
        self.assertTrue(run.summary["import/complete"])
        for key, value in plan["summary"].items():
            self.assertEqual(run.summary[key], value)
        args = sdk.init.call_args.kwargs
        self.assertEqual((args["resume"], args["mode"], args["job_type"]),
                         ("allow", "online", "historical-import"))
        self.assertTrue(sdk.Settings.call_args.kwargs["x_disable_machine_info"])
        task_args = sdk.Table.call_args_list[0].kwargs
        trace_column = task_args["columns"].index("cem_loss_trace")
        self.assertEqual(json.loads(task_args["data"][0][trace_column]),
                         plan["tables"]["tasks"]["data"][0][trace_column])

    def test_failed_upload_retries_same_run_without_completion_marker(self):
        sdk = self.sdk()
        plan = importer.build_plans(RESULTS[0])[0]
        # Establish the run, but fail while uploading the source artifact.
        run = sdk.init(id=plan["id"], config=plan["config"])
        run.log_artifact.return_value.wait.side_effect = RuntimeError("upload failed")
        with self.assertRaisesRegex(RuntimeError, "upload failed"):
            importer.upload_plan(plan, sdk, "team", "project")
        self.assertNotIn("import/complete", run.summary)
        run.log_artifact.return_value.wait.side_effect = None
        with contextlib.redirect_stdout(io.StringIO()):
            importer.upload_plan(plan, sdk, "team", "project")
        self.assertEqual(len(sdk.runs_by_id), 1)
        self.assertTrue(run.summary["import/complete"])


if __name__ == "__main__":
    unittest.main()
