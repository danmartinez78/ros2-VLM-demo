#!/usr/bin/env python3
"""Summarize ROS VisionReasoningResult output from an observation-history run."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path


def parse_records(text: str) -> list[dict]:
    records: list[dict] = []
    for block in re.split(r"^---\s*$", text, flags=re.MULTILINE):
        success_match = re.search(r"^success:\s*(true|false)\s*$", block, re.MULTILINE)
        if not success_match:
            continue
        latency_match = re.search(
            r"^inference_seconds:\s*([0-9]+(?:\.[0-9]+)?)\s*$", block, re.MULTILINE
        )
        sequence_match = re.search(r"^frame_sequence:\s*(\d+)\s*$", block, re.MULTILINE)
        response_match = re.search(r"^response:\s*(.*)$", block, re.MULTILINE)
        error_match = re.search(r"^error:\s*(.*)$", block, re.MULTILINE)
        records.append(
            {
                "success": success_match.group(1) == "true",
                "inference_seconds": float(latency_match.group(1)) if latency_match else None,
                "frame_sequence": int(sequence_match.group(1)) if sequence_match else None,
                "response": response_match.group(1).strip() if response_match else "",
                "error": error_match.group(1).strip() if error_match else "",
            }
        )
    return records


def summarize(records: list[dict]) -> dict:
    successful = [record for record in records if record["success"]]
    latencies = [
        record["inference_seconds"]
        for record in successful
        if record["inference_seconds"] is not None
    ]
    summary = {
        "schema_version": 1,
        "result_count": len(records),
        "success_count": len(successful),
        "failure_count": len(records) - len(successful),
        "success_rate": len(successful) / len(records) if records else 0.0,
        "latency_seconds": None,
        "responses": [
            {
                "frame_sequence": record["frame_sequence"],
                "response": record["response"],
            }
            for record in successful
        ],
        "errors": [record["error"] for record in records if not record["success"]],
    }
    if latencies:
        summary["latency_seconds"] = {
            "count": len(latencies),
            "mean": statistics.fmean(latencies),
            "median": statistics.median(latencies),
            "min": min(latencies),
            "max": max(latencies),
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    records = parse_records(args.input.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summarize(records), indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
