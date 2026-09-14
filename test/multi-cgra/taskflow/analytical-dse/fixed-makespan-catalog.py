#!/usr/bin/env python3
"""Build a fixed-duration cost catalog and verify its score provenance."""

import hashlib
import json
import sys
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def create(manifest_path, catalog_path):
    header = read_jsonl(manifest_path)[0]
    architecture_sha = header["architecture"]["spec_sha256"]
    task_hashes = {
        task["task"]: task["body_sha256"] for task in header["tasks"]
    }
    catalog = {
        "schema": "amoeba-task-shape-cost",
        "function": header["function"],
        "namespace": "fixed-five-cycle-test",
        "predictor_metadata": {
            "candidate_manifest_sha256": sha256(manifest_path),
            "analytical_provenance": {
                "architecture_sha256": architecture_sha,
                "neura_opt_sha256": "0" * 64,
                "task_body_sha256": task_hashes,
                "task_dfg_sha256": {task: "1" * 64 for task in task_hashes},
            },
            "checkpoints": {
                "baseline": {
                    "sha256": "2" * 64,
                    "config_sha256": "3" * 64,
                }
            },
            "architecture_contract": {
                "contract_id": "fixed-five-cycle-test",
                "supported_architecture_sha256": [architecture_sha],
            },
            "ranking_policy": {
                "objective": "predicted_scheduler_makespan",
                "mapper_success_probability": "not_predicted",
                "uses_mapper_success_probability": False,
            },
        },
        "entries": [
            {
                "task": query["task"],
                "mapper_tile_rows": query["mapper_tile_rows"],
                "mapper_tile_cols": query["mapper_tile_cols"],
                "support_status": "supported",
                "predicted_ii": 1.0,
                "startup_cycles": 5.0,
                "analytical_lower_bound": 1.0,
            }
            for query in header["cost_queries"]
        ],
    }
    catalog_path.write_text(json.dumps(catalog, sort_keys=True) + "\n")


def verify(manifest_path, catalog_path, score_path):
    header, score, footer = read_jsonl(score_path)
    assert header["candidate_manifest_sha256"] == sha256(manifest_path)
    assert header["cost_catalog_sha256"] == sha256(catalog_path)
    assert header["score_model"] == "spatial-temporal-scheduler-makespan"
    assert score["candidate_id"] == "candidate-0"
    assert score["predicted_compute_bottleneck"] == 5
    assert score["predicted_scheduler_makespan"] == 10
    assert footer["shortlist"] == [
        {
            "candidate_id": "candidate-0",
            "predicted_scheduler_makespan": 10,
            "rank": 0,
        }
    ]


if __name__ == "__main__":
    action, manifest, catalog, *remainder = sys.argv[1:]
    if action == "create" and not remainder:
        create(Path(manifest), Path(catalog))
    elif action == "verify" and len(remainder) == 1:
        verify(Path(manifest), Path(catalog), Path(remainder[0]))
    else:
        raise SystemExit("usage: fixed-makespan-catalog.py create|verify manifest catalog [scores]")
