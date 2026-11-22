"""Opt-in serialization test against the pinned SDK, with no authentication/upload."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bench import import_wandb as importer


@unittest.skipUnless(os.environ.get("HPML_WANDB_SDK_TEST") == "1", "opt-in W&B SDK test")
class WandbSDKTests(unittest.TestCase):
    def test_all_existing_results_serialize_offline(self):
        with tempfile.TemporaryDirectory(prefix="hpml-wandb-sdk-") as temp, patch.dict(os.environ, {
            "WANDB_MODE": "offline",
            "WANDB_CACHE_DIR": str(Path(temp) / "cache"),
            "WANDB_CONFIG_DIR": str(Path(temp) / "config"),
            "WANDB_DATA_DIR": str(Path(temp) / "data"),
            "WANDB_ERROR_REPORTING": "false",
        }):
            import wandb

            self.assertEqual(wandb.__version__, "0.31.2")
            real_init = wandb.init
            seen = []

            def offline_init(**kwargs):
                # Override transport only: exercise the actual Settings, Config,
                # Summary, Table and Artifact serialization used by upload_plan.
                kwargs.update(mode="offline", resume=None, dir=temp)
                run = real_init(**kwargs)
                seen.append(run.id)
                return run

            root = Path(__file__).resolve().parents[1]
            try:
                # Offline artifacts have no remote upload to wait for.
                with patch.object(wandb, "init", side_effect=offline_init), \
                     patch.object(wandb.Artifact, "wait", return_value=None):
                    for source in (root / "results" / name / "results.json" for name in ("mid_seed99", "smoke")):
                        for plan in importer.build_plans(source):
                            importer.upload_plan(plan, wandb, "local-validation", "hpml-import-test")
                tables = list(Path(temp).rglob("*.table.json"))
                task_tables = [json.loads(path.read_text()) for path in tables if path.name.startswith("tasks_")]
                self.assertEqual(len(seen), 8)
                self.assertEqual(len(set(seen)), 8)
                self.assertEqual(len(tables), 24)
                self.assertEqual(sum(len(table["data"]) for table in task_tables), 72)
                for table in task_tables:
                    trace_index = table["columns"].index("cem_loss_trace")
                    self.assertIsInstance(json.loads(table["data"][0][trace_index]), list)
            finally:
                wandb.teardown()


if __name__ == "__main__":
    unittest.main()
