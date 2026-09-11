#!/usr/bin/env python3
"""ML-score a frozen static-shape space, then replay only its top-k.

The predictor is loaded exclusively from Amoeba's pinned
``thirdparty/cgra-ii-predictor`` submodule. The input must already contain
Taskflow tasks with one pre-mapper Neura kernel per task; frontend lowering is
program-specific and remains outside this shape-only driver.

Arguments after ``--`` are forwarded to ``mlir-amoeba-opt`` after candidate
materialization. They therefore execute once per validated shortlist entry,
never once per enumerated candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Iterable, List, Mapping, Optional, Sequence


AMOEBA_ROOT = Path(__file__).resolve().parents[1]
PREDICTOR_ROOT = AMOEBA_ROOT / "thirdparty" / "cgra-ii-predictor"
ADAPTER_ROOT = PREDICTOR_ROOT / "adapters"
CANDIDATE_SCHEMA = "amoeba-analytical-task-candidates"
COST_SCHEMA = "amoeba-task-shape-cost"
SCORE_SCHEMA = "amoeba-analytical-task-scores"
SCORE_MODEL = "static-shape-compute-bottleneck"


class PipelineError(RuntimeError):
    """A checked orchestration step failed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_file(path: Path, label: str, executable: bool = False) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise PipelineError(f"{label} does not exist: {resolved}")
    if executable and not os.access(resolved, os.X_OK):
        raise PipelineError(f"{label} is not executable: {resolved}")
    return resolved


def _prepare_output(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists() and (not resolved.is_dir() or any(resolved.iterdir())):
        raise PipelineError(f"output directory is not empty: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _run(command: Sequence[str], label: str, output: Path,
         cwd: Optional[Path] = None) -> None:
    """Run one checked command and preserve both output streams."""
    completed = subprocess.run(
        list(command), cwd=cwd, capture_output=True, text=True, check=False,
    )
    (output / f"{label}.stdout.log").write_text(completed.stdout)
    (output / f"{label}.stderr.log").write_text(completed.stderr)
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()
        raise PipelineError(
            f"{label} failed with exit code {completed.returncode}: {detail}"
        )


def _task_dfg_arguments(index: Mapping[str, object]) -> List[str]:
    result: List[str] = []
    if not index:
        raise PipelineError("task DFG extraction produced an empty index")
    for task, raw_path in sorted(index.items()):
        if not isinstance(task, str) or not task or not isinstance(raw_path, str):
            raise PipelineError("task DFG index must map task names to paths")
        path = _require_file(Path(raw_path), f"DFG for task {task}")
        result.append(f"--task-dfg={task}={path}")
    return result


def _read_json_object(path: Path, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise PipelineError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise PipelineError(f"{label} is not a JSON object")
    return value


def _candidate_header_and_count(path: Path) -> tuple[Mapping[str, object], int]:
    header = None
    footer = None
    candidate_count = 0
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise PipelineError(
                f"invalid candidate JSONL at line {line_number}: {error}"
            ) from error
        if not isinstance(record, dict) or record.get("schema") != CANDIDATE_SCHEMA:
            raise PipelineError("candidate manifest schema mismatch")
        kind = record.get("record_type")
        if kind == "header":
            if header is not None or candidate_count or footer is not None:
                raise PipelineError("candidate manifest header is misplaced")
            header = record
        elif kind == "candidate":
            if header is None or footer is not None:
                raise PipelineError("candidate record is outside header/footer")
            candidate_count += 1
        elif kind == "footer":
            if header is None or footer is not None:
                raise PipelineError("candidate manifest footer is misplaced")
            footer = record
        else:
            raise PipelineError("candidate manifest has an unknown record type")
    if header is None or footer is None or footer.get("candidate_count") != candidate_count:
        raise PipelineError("candidate manifest is incomplete")
    return header, candidate_count


def _require_bound_trip_counts(path: Path) -> None:
    header, _ = _candidate_header_and_count(path)
    tasks = header.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise PipelineError("candidate manifest has no task facts")
    unbound = []
    for task in tasks:
        if not isinstance(task, dict):
            raise PipelineError("candidate manifest task fact is invalid")
        if task.get("trip_count_kind") == "symbol_dynamic":
            unbound.append(str(task.get("task", "<unnamed>")))
        elif isinstance(task.get("trip_count"), bool) or not isinstance(
            task.get("trip_count"), int
        ) or int(task["trip_count"]) <= 0:
            raise PipelineError("candidate manifest trip-count fact is invalid")
    if unbound:
        raise PipelineError(
            "program-level analytical ranking requires concrete trip counts; "
            "symbol-dynamic tasks: " + ", ".join(unbound)
        )


def _read_shortlist(path: Path, candidate_manifest: Path,
                    cost_catalog: Path) -> List[Mapping[str, object]]:
    """Validate the complete score artifact chain before replaying any ID."""
    header = None
    footer = None
    scores = {}
    saw_footer = False
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise PipelineError(
                f"invalid score JSONL at line {line_number}: {error}"
            ) from error
        if not isinstance(record, dict) or record.get("schema") != SCORE_SCHEMA:
            raise PipelineError("score manifest schema mismatch")
        kind = record.get("record_type")
        if kind == "header":
            if header is not None or scores or footer is not None:
                raise PipelineError("score manifest header is misplaced")
            header = record
        elif kind == "score":
            if header is None or saw_footer:
                raise PipelineError("score record is outside header/footer")
            candidate_id = record.get("candidate_id")
            if not isinstance(candidate_id, str) or candidate_id in scores:
                raise PipelineError("score candidate IDs are invalid or duplicated")
            scores[candidate_id] = record
        elif kind == "footer":
            if header is None or saw_footer:
                raise PipelineError("score manifest footer is misplaced")
            footer = record
            saw_footer = True
        else:
            raise PipelineError("score manifest has an unknown record type")
    if header is None or footer is None:
        raise PipelineError("score manifest is incomplete")

    candidate_header, candidate_count = _candidate_header_and_count(
        candidate_manifest
    )
    catalog = _read_json_object(cost_catalog, "cost catalogue")
    if catalog.get("schema") != COST_SCHEMA:
        raise PipelineError("cost catalogue schema mismatch")
    if (
        header.get("candidate_schema") != CANDIDATE_SCHEMA
        or header.get("score_model") != SCORE_MODEL
        or header.get("candidate_manifest_sha256")
        != _sha256_file(candidate_manifest)
        or header.get("cost_catalog_sha256") != _sha256_file(cost_catalog)
        or header.get("cost_namespace") != catalog.get("namespace")
        or header.get("function") != candidate_header.get("function")
    ):
        raise PipelineError("score manifest provenance does not match its inputs")
    valid_scores = {
        candidate_id: record for candidate_id, record in scores.items()
        if record.get("valid") is True
    }
    if (
        footer.get("candidate_count") != candidate_count
        or footer.get("scored_count") != candidate_count
        or len(scores) != candidate_count
        or footer.get("valid_count") != len(valid_scores)
    ):
        raise PipelineError("score manifest counts do not cover every candidate")

    shortlist = footer.get("shortlist")
    if not isinstance(shortlist, list) or not shortlist:
        raise PipelineError("ML scoring produced an empty shortlist")
    selected = set()
    for expected_rank, item in enumerate(shortlist):
        if not isinstance(item, dict) or item.get("rank") != expected_rank:
            raise PipelineError("score shortlist ranks are not contiguous")
        candidate_id = item.get("candidate_id")
        if (
            not isinstance(candidate_id, str)
            or re.fullmatch(r"candidate-[0-9]+", candidate_id) is None
            or candidate_id in selected
            or candidate_id not in valid_scores
            or item.get("predicted_compute_bottleneck")
            != valid_scores[candidate_id].get("predicted_compute_bottleneck")
        ):
            raise PipelineError("score shortlist is inconsistent with valid scores")
        selected.add(candidate_id)
    return shortlist


def _option(name: str, values: Iterable[tuple[str, object]]) -> str:
    fields = [
        f"{key}={value}" for key, value in values if value not in (None, "")
    ]
    return f"--{name}=" + " ".join(fields)


def run(args: argparse.Namespace) -> Mapping[str, object]:
    input_path = _require_file(args.input, "input MLIR")
    architecture = _require_file(args.architecture, "architecture YAML")
    amoeba_opt = _require_file(args.amoeba_opt, "mlir-amoeba-opt", executable=True)
    output = _prepare_output(args.output_dir)

    required_adapters = {
        "extract": ADAPTER_ROOT / "extract_amoeba_task_dfgs.py",
        "features": ADAPTER_ROOT / "generate_amoeba_query_features.py",
        "catalog": ADAPTER_ROOT / "amoeba_cost_catalog.py",
    }
    missing = [str(path) for path in required_adapters.values() if not path.is_file()]
    if missing:
        raise PipelineError(
            "predictor submodule is not initialized; run `git submodule update "
            "--init thirdparty/cgra-ii-predictor`; missing: " + ", ".join(missing)
        )

    manifest = output / "candidates.jsonl"
    bound_ir = output / "bound.mlir"
    task_dir = output / "task-dfgs"
    task_index = output / "task-dfgs.json"
    analytical_input = output / "analytical-input.json"
    analytical_timing = output / "analytical-timing.json"
    cost_catalog = output / "costs.json"
    inference_timing = output / "inference-timing.json"
    scores = output / "scores.jsonl"
    scored_ir = output / "scored.mlir"

    enumerate_option = _option(
        "enumerate-analytical-task-candidates",
        (("output", manifest), ("function", args.function)),
    )
    _run(
        [str(amoeba_opt), str(input_path), "--classify-task-and-counter",
         enumerate_option, f"--architecture-spec={architecture}",
         "-o", str(bound_ir)],
        "enumerate", output,
    )
    _require_bound_trip_counts(manifest)

    extract_command = [
        sys.executable, str(required_adapters["extract"]),
        "--input", str(bound_ir), "--output-dir", str(task_dir),
        "--index-output", str(task_index),
    ]
    if args.function:
        extract_command.extend(("--function", args.function))
    _run(extract_command, "extract-task-dfgs", output)
    index = _read_json_object(task_index, "task DFG index")
    task_arguments = _task_dfg_arguments(index)

    _run(
        [sys.executable, str(required_adapters["features"]),
         "--manifest", str(manifest), *task_arguments,
         "--neura-opt", str(amoeba_opt), "--architecture", str(architecture),
         "--output", str(analytical_input),
         "--timing-output", str(analytical_timing)],
        "analyze-task-shapes", output,
    )
    _run(
        [sys.executable, str(required_adapters["catalog"]),
         "--manifest", str(manifest),
         "--analytical-input", str(analytical_input), *task_arguments,
         "--output", str(cost_catalog),
         "--timing-output", str(inference_timing),
         "--device", args.device],
        "predict-ii", output,
    )

    score_option = _option(
        "score-analytical-task-candidates",
        (("candidates", manifest), ("cost-file", cost_catalog),
         ("output", scores), ("top-k", args.top_k),
         ("function", args.function)),
    )
    _run(
        [str(amoeba_opt), str(bound_ir), score_option,
         f"--architecture-spec={architecture}", "-o", str(scored_ir)],
        "score", output,
    )

    downstream = list(args.downstream_args)
    shortlist = _read_shortlist(scores, manifest, cost_catalog)
    shortlist_root = output / "shortlist"
    shortlist_root.mkdir()
    materialized = []
    for item in shortlist:
        rank = int(item["rank"])
        candidate_id = str(item["candidate_id"])
        candidate_root = shortlist_root / f"{rank:03d}-{candidate_id}"
        candidate_root.mkdir()
        candidate_output = candidate_root / "output.mlir"
        materialize_option = _option(
            "materialize-analytical-task-candidate",
            (("candidates", manifest), ("candidate-id", candidate_id),
             ("function", args.function)),
        )
        _run(
            [str(amoeba_opt), str(bound_ir), materialize_option,
             f"--architecture-spec={architecture}", *downstream,
             "-o", str(candidate_output)],
            f"top-{rank}-{candidate_id}", output, cwd=candidate_root,
        )
        materialized.append({
            "rank": rank,
            "candidate_id": candidate_id,
            "predicted_compute_bottleneck": item.get(
                "predicted_compute_bottleneck"
            ),
            "output": str(candidate_output),
        })

    report = {
        "input": str(input_path),
        "architecture": str(architecture),
        "predictor_submodule": str(PREDICTOR_ROOT),
        "top_k_requested": args.top_k,
        "shortlist_materialization_invocations": len(materialized),
        "downstream_arguments": downstream,
        "artifacts": {
            "candidate_manifest": str(manifest),
            "bound_ir": str(bound_ir),
            "task_dfg_index": str(task_index),
            "analytical_input": str(analytical_input),
            "analytical_timing": str(analytical_timing),
            "cost_catalog": str(cost_catalog),
            "inference_timing": str(inference_timing),
            "scores": str(scores),
            "scored_ir": str(scored_ir),
        },
        "shortlist": materialized,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    raw_arguments = list(sys.argv[1:] if argv is None else argv)
    downstream_arguments: List[str] = []
    if "--" in raw_arguments:
        separator = raw_arguments.index("--")
        downstream_arguments = raw_arguments[separator + 1:]
        raw_arguments = raw_arguments[:separator]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--architecture", type=Path, required=True)
    parser.add_argument(
        "--amoeba-opt", type=Path,
        default=AMOEBA_ROOT / "build" / "tools" / "mlir-amoeba-opt" /
        "mlir-amoeba-opt",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--function", default="")
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(raw_arguments)
    args.downstream_args = downstream_arguments
    if args.top_k <= 0:
        parser.error("--top-k must be positive for shortlist replay")
    return args


def main() -> int:
    try:
        report = run(parse_args())
    except (OSError, PipelineError, ValueError) as error:
        print(f"analytical task DSE failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
