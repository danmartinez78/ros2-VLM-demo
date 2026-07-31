#!/usr/bin/env python3
"""Task-level evaluation harness for Cosmos reasoning runs."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SUPPORTED_MODES = {"regression", "exploratory"}


@dataclass
class RubricScore:
    name: str
    status: str
    matched: list[str]
    missed: list[str]
    forbidden_hits: list[str]



def _normalize(text: str) -> str:
    return " ".join(text.lower().split())



def _contains_any(haystack: str, alternatives: list[str]) -> tuple[bool, str | None]:
    for item in alternatives:
        if _normalize(item) in haystack:
            return True, item
    return False, None



def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    values = sorted(values)
    rank = (percentile / 100.0) * (len(values) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return values[lower]
    weight = rank - lower
    return values[lower] * (1 - weight) + values[upper] * weight



def score_rubric(response: str, rubric_name: str, rubric_definition: dict[str, Any]) -> RubricScore:
    scoring_mode = rubric_definition.get("scoring", "automatic")
    if scoring_mode == "human":
        return RubricScore(
            name=rubric_name,
            status="human_review_required",
            matched=[],
            missed=[],
            forbidden_hits=[],
        )

    normalized = _normalize(response)
    matched: list[str] = []
    missed: list[str] = []

    for rule in rubric_definition.get("required_observations", []):
        alternatives = rule if isinstance(rule, list) else [rule]
        found, hit = _contains_any(normalized, [str(item) for item in alternatives])
        if found and hit is not None:
            matched.append(hit)
        else:
            missed.append("|".join(str(item) for item in alternatives))

    forbidden_hits: list[str] = []
    for forbidden in rubric_definition.get("forbidden_observations", []):
        if _normalize(str(forbidden)) in normalized:
            forbidden_hits.append(str(forbidden))

    status = "passed" if not missed and not forbidden_hits else "failed"
    return RubricScore(
        name=rubric_name,
        status=status,
        matched=matched,
        missed=missed,
        forbidden_hits=forbidden_hits,
    )



def evaluate_run(dataset: dict[str, Any], run: dict[str, Any], mode_override: str | None = None) -> dict[str, Any]:
    dataset_id = dataset["dataset_id"]
    dataset_version = dataset["version"]

    run_mode = mode_override or run.get("mode", "regression")
    if run_mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported mode: {run_mode}")

    if run_mode == "regression":
        expected_dataset = run.get("dataset_id")
        if expected_dataset and expected_dataset != dataset_id:
            raise ValueError(
                f"Regression run dataset_id mismatch: run={expected_dataset} dataset={dataset_id}"
            )

    examples_by_id = {example["id"]: example for example in dataset["examples"]}

    per_example: list[dict[str, Any]] = []
    latencies: list[float] = []
    failures = 0
    unsupported_claim_total = 0
    auto_scored_examples = 0
    auto_scored_passed = 0
    human_review_required = 0
    rubric_totals: dict[str, dict[str, int]] = {}

    for result in run.get("results", []):
        example_id = result["example_id"]
        dataset_example = examples_by_id.get(example_id)
        if not dataset_example:
            raise ValueError(f"Run references unknown example_id: {example_id}")

        success = bool(result.get("success", False))
        response = result.get("response", "")
        latency = float(result.get("latency_seconds", 0.0))
        error = result.get("error", "")

        latencies.append(latency)
        if not success:
            failures += 1

        rubric_scores: dict[str, dict[str, Any]] = {}
        auto_rubric_failed = False
        requires_human_review = False

        for rubric_name, rubric_definition in dataset_example.get("rubrics", {}).items():
            score = score_rubric(response, rubric_name, rubric_definition)
            rubric_scores[rubric_name] = {
                "status": score.status,
                "matched": score.matched,
                "missed": score.missed,
                "forbidden_hits": score.forbidden_hits,
            }

            totals = rubric_totals.setdefault(
                rubric_name,
                {"passed": 0, "failed": 0, "human_review_required": 0},
            )
            totals[score.status] += 1

            if score.status == "failed":
                auto_rubric_failed = True
            elif score.status == "human_review_required":
                requires_human_review = True

        unsupported_terms = [str(term) for term in dataset_example.get("unsupported_claim_terms", [])]
        normalized_response = _normalize(response)
        unsupported_hits = [term for term in unsupported_terms if _normalize(term) in normalized_response]
        unsupported_claim_total += len(unsupported_hits)

        if requires_human_review:
            human_review_required += 1
            correctness = None
        else:
            auto_scored_examples += 1
            correctness = success and not auto_rubric_failed and not unsupported_hits
            if correctness:
                auto_scored_passed += 1

        per_example.append(
            {
                "example_id": example_id,
                "segment": dataset_example.get("segment", {}),
                "success": success,
                "failure": not success,
                "error": error,
                "latency_seconds": latency,
                "rubric_scores": rubric_scores,
                "unsupported_claim_hits": unsupported_hits,
                "unsupported_claim_count": len(unsupported_hits),
                "human_review_required": requires_human_review,
                "correctness": correctness,
            }
        )

    total_examples = len(per_example)
    aggregate = {
        "total_examples": total_examples,
        "auto_scored_examples": auto_scored_examples,
        "correct_examples": auto_scored_passed,
        "correctness_rate": (auto_scored_passed / auto_scored_examples) if auto_scored_examples else None,
        "unsupported_claim_total": unsupported_claim_total,
        "unsupported_claim_rate": (unsupported_claim_total / total_examples) if total_examples else None,
        "failures": failures,
        "failure_rate": (failures / total_examples) if total_examples else None,
        "human_review_required": human_review_required,
        "latency_seconds": {
            "mean": statistics.fmean(latencies) if latencies else None,
            "p50": _percentile(latencies, 50.0),
            "p95": _percentile(latencies, 95.0),
            "max": max(latencies) if latencies else None,
        },
        "rubric_summary": rubric_totals,
    }

    return {
        "dataset": {
            "dataset_id": dataset_id,
            "version": dataset_version,
            "notes": dataset.get("notes", ""),
        },
        "run": {
            "run_id": run.get("run_id"),
            "mode": run_mode,
            "dataset_id": dataset_id,
            "executed_at": run.get("executed_at")
            or datetime.now(timezone.utc).isoformat(),
            "configuration": run.get("configuration", {}),
        },
        "per_example": per_example,
        "aggregate": aggregate,
    }



def compare_reports(primary: dict[str, Any], secondary: dict[str, Any]) -> dict[str, Any]:
    def _delta(metric: str) -> float | None:
        first = primary["aggregate"].get(metric)
        second = secondary["aggregate"].get(metric)
        if first is None or second is None:
            return None
        return second - first

    return {
        "primary_run_id": primary["run"].get("run_id"),
        "secondary_run_id": secondary["run"].get("run_id"),
        "dataset_id": primary["dataset"].get("dataset_id"),
        "metric_deltas": {
            "correctness_rate": _delta("correctness_rate"),
            "unsupported_claim_rate": _delta("unsupported_claim_rate"),
            "failure_rate": _delta("failure_rate"),
            "latency_mean_seconds": (
                None
                if primary["aggregate"]["latency_seconds"]["mean"] is None
                or secondary["aggregate"]["latency_seconds"]["mean"] is None
                else secondary["aggregate"]["latency_seconds"]["mean"]
                - primary["aggregate"]["latency_seconds"]["mean"]
            ),
        },
    }



def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)



def main() -> None:
    parser = argparse.ArgumentParser(description="Task-level evaluation harness")
    parser.add_argument("--dataset", required=True, type=Path, help="Dataset JSON file")
    parser.add_argument("--run", required=True, type=Path, help="Primary run JSON file")
    parser.add_argument("--compare-run", type=Path, default=None, help="Optional second run JSON")
    parser.add_argument("--mode", choices=sorted(SUPPORTED_MODES), default=None, help="Override run mode")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON report path")
    args = parser.parse_args()

    dataset = _load_json(args.dataset)
    primary_run = _load_json(args.run)
    primary_report = evaluate_run(dataset, primary_run, args.mode)

    output: dict[str, Any] = {"primary": primary_report}
    if args.compare_run:
        secondary_run = _load_json(args.compare_run)
        secondary_report = evaluate_run(dataset, secondary_run, args.mode)
        output["secondary"] = secondary_report
        output["comparison"] = compare_reports(primary_report, secondary_report)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
