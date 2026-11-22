#!/usr/bin/env python3
"""Import recorded planbench JSON without loading the model or running benchmarks."""

import argparse
import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path


IMPORT_SCHEMA = 1
ROOT = Path(__file__).resolve().parents[1]
# These arrays are indexed by paired task in planbench.run_variant, not repetitions.
TASK_FIELDS = ("latency_s", "success", "state_dist", "cem_loss_trace", "actions")


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def shown_path(path):
    """Repo-relative inside the repository, else folder/file: never an absolute local path (the project is public)."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return f"{path.parent.name}/{path.name}"


def scalar_metrics(value, prefix):
    """Expose recorded scalar leaves; arrays are preserved in tables/the artifact."""
    metrics = {}
    for key, item in value.items():
        name = f"{prefix}/{key}"
        if isinstance(item, dict):
            metrics.update(scalar_metrics(item, name))
        elif isinstance(item, (int, float, bool)):
            metrics[name] = item
    return metrics


def row_table(rows):
    """Union columns so an OOM scaling row needs no fabricated latency values."""
    columns = sorted({key for row in rows for key in row})
    return {"columns": columns, "data": [[row.get(key) for key in columns] for row in rows]}


def validate_document(result):
    if not isinstance(result, dict):
        raise ValueError("expected a JSON object")
    if not isinstance(result.get("meta"), dict):
        raise ValueError("meta must be an object")
    variants = result.get("variants")
    if not isinstance(variants, dict) or not variants:
        raise ValueError("variants must be a nonempty object of recorded plan results")
    for key in ("selfcheck", "reference", "profile_top_ops"):
        if key in result and not isinstance(result[key], dict):
            raise ValueError(f"{key} must be an object")
    if "scaling" in result and (
        not isinstance(result["scaling"], list)
        or any(not isinstance(row, dict) or not isinstance(row.get("variant"), str)
               for row in result["scaling"])
    ):
        raise ValueError("scaling must be an array of objects with variant names")
    for name, variant in variants.items():
        if not name or not isinstance(variant, dict):
            raise ValueError("each variant must have a nonempty name and an object of results")
        if "variant" in variant and variant["variant"] != name:
            raise ValueError(f"{name}: variant name disagrees with its key")
        lengths = set()
        for key in TASK_FIELDS:
            if key not in variant:
                continue
            values = variant[key]
            if not isinstance(values, list):
                raise ValueError(f"{name}.{key} must be an array")
            lengths.add(len(values))
            if key == "success" and any(type(x) is not bool for x in values):
                raise ValueError(f"{name}.success must contain booleans")
            if key in ("latency_s", "state_dist") and any(
                type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in values
            ):
                raise ValueError(f"{name}.{key} must contain finite, nonnegative numbers")
            if key in ("actions", "cem_loss_trace") and any(not isinstance(x, list) for x in values):
                raise ValueError(f"{name}.{key} must contain arrays")
        if len(lengths) > 1:
            raise ValueError(f"{name}: per-task arrays have different lengths")
        n_tasks = result["meta"].get("n_tasks")
        if n_tasks is not None and (type(n_tasks) is not int or n_tasks < 0):
            raise ValueError("meta.n_tasks must be a nonnegative integer")
        if lengths and n_tasks is not None and lengths != {n_tasks}:
            raise ValueError(f"{name}: per-task arrays disagree with meta.n_tasks")
    for name, rows in result.get("profile_top_ops", {}).items():
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError(f"profile_top_ops.{name} must be an array of objects")


def build_plans(path):
    """Validate once and build the exact payload shared by dry-run and upload."""
    path = Path(path).expanduser()
    if path.is_dir():
        path = path / "results.json"
    source_bytes = path.read_bytes()
    result = json.loads(source_bytes)
    # Reject NaN/Infinity even in fields that are only retained in the artifact.
    source_hash = digest(result)
    validate_document(result)
    plans = []
    for name, variant in result["variants"].items():
        run_id = "hist-" + digest({"source_sha256": source_hash, "variant": name})[:32]
        fields = [key for key in TASK_FIELDS if key in variant]
        tables = {}
        if fields:
            n_tasks = len(variant[fields[0]])
            tables["tasks"] = {
                "columns": ["task_index", *fields],
                "data": [[i, *(variant[key][i] for key in fields)] for i in range(n_tasks)],
            }
        for section in ("scaling", "profile_top_ops"):
            rows = ([row for row in result.get(section, []) if row["variant"] == name]
                    if section == "scaling" else result.get(section, {}).get(name, []))
            if rows:
                tables[section] = row_table(rows)
        summary = scalar_metrics(variant, "recorded")
        # Keep the whole selfcheck namespace: official_rerun is a noise check,
        # not an invented equivalence check for the official variant.
        for section in ("selfcheck", "reference"):
            summary.update(scalar_metrics(result.get(section, {}), section))
        plans.append({
            "id": run_id,
            "name": f"historical-{path.parent.name}-{name}",
            "group": f"historical-{source_hash[:16]}",
            "config": {
                "variant": name,
                "benchmark": result["meta"],
                "import": {
                    "schema_version": IMPORT_SCHEMA,
                    "historical": True,
                    "experiments_executed": False,
                    "source_sha256": source_hash,
                    "source_file": shown_path(path),
                    "source_bytes_sha256": hashlib.sha256(source_bytes).hexdigest(),
                    "timestamp_semantics": "W&B timestamps describe import time; no experiment timestamp is inferred",
                },
            },
            "summary": summary,
            "tables": tables,
            # Original bytes are held separately to snapshot precisely what was validated.
            "source_bytes": source_bytes,
        })
    return plans


def upload_plan(plan, wandb, entity, project):
    """Resume a deterministic ID; finished imports never append measurements."""
    with wandb.init(
        entity=entity, project=project, id=plan["id"], resume="allow", mode="online",
        name=plan["name"], group=plan["group"], job_type="historical-import",
        tags=["historical", "imported", plan["config"]["variant"]],
        notes=("Imported historical benchmark results. No experiments were executed. "
               "Hardware and software in config.benchmark are recorded source metadata. "
               "W&B run timestamps and runtime belong to the import, not the experiment."),
        config=plan["config"], save_code=False,
        settings=wandb.Settings(
            x_disable_stats=True, x_disable_meta=True, x_disable_machine_info=True, disable_git=True,
            disable_code=True, console="off",
        ),
    ) as run:
        if run.resumed:
            expected = plan["config"]["import"]["source_sha256"]
            if run.config.get("import", {}).get("source_sha256") != expected:
                raise ValueError(f"run ID collision for {plan['id']}")
            if run.summary.get("import/complete"):
                print(f"Skipped completed import: {entity}/{project}/{plan['id']}")
                return
        # Retain the complete original document, including unrecognized fields.
        with tempfile.TemporaryDirectory(prefix="hpml-wandb-") as temp:
            source = Path(temp) / "results.json"
            source.write_bytes(plan["source_bytes"])
            artifact = wandb.Artifact(
                "historical-results-" + plan["config"]["import"]["source_sha256"],
                type="benchmark-results", metadata={"historical": True},
            )
            artifact.add_file(str(source), name="results.json")
            run.log_artifact(artifact).wait()
        tables = {}
        for name, table in plan["tables"].items():
            # Nested actions/traces use lossless JSON text in Table cells.
            data = [[canonical_json(x) if isinstance(x, (dict, list)) else x for x in row]
                    for row in table["data"]]
            if len(data) > wandb.Table.MAX_ARTIFACT_ROWS:
                raise ValueError(f"{name} exceeds W&B's Table row limit")
            tables[name] = wandb.Table(columns=table["columns"], data=data)
        if tables:
            run.log(tables)
        run.summary.update(plan["summary"])
        run.summary["import/complete"] = True
    print(f"Imported: {entity}/{project}/{plan['id']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, nargs="+", action="extend", required=True,
                        help="JSON files or directories containing results.json; may be repeated")
    parser.add_argument("--entity", help="W&B user or team (required for upload)")
    parser.add_argument("--project", default="hpml-wm-planning")
    parser.add_argument("--dry-run", action="store_true",
                        help="print validated payloads as JSON; no SDK, authentication, or network")
    args = parser.parse_args(argv)
    if not args.dry_run and not args.entity:
        parser.error("--entity is required for upload")
    if not args.project or "/" in args.project or (args.entity is not None and (not args.entity or "/" in args.entity)):
        parser.error("entity and project must be nonempty names without slashes")
    try:
        # Validate every input before making any W&B calls, and deduplicate copies.
        plans = {}
        for path in args.results:
            for plan in build_plans(path):
                plans.setdefault(plan["id"], plan)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if args.dry_run:
        public_plans = [{k: v for k, v in plan.items() if k != "source_bytes"} for plan in plans.values()]
        print(json.dumps({"entity": args.entity, "project": args.project, "runs": public_plans}, indent=2,
                         allow_nan=False))
        return 0
    try:
        import wandb
    except ImportError:
        parser.error("install W&B with: python -m pip install -r requirements-wandb.txt")
    try:
        for plan in plans.values():
            upload_plan(plan, wandb, args.entity, args.project)
    except Exception as error:
        print(f"Import failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
