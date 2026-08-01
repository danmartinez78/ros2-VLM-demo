#!/usr/bin/env python3
"""
Compute ROS pipeline overhead metrics from per-frame timing records.

Reads the JSON-Lines file written by cosmos_reasoner when the
`benchmark_output_file` parameter is set, then computes and reports:

  - Per-frame: image-convert time, IPC/encoding overhead, native inference
    time (from the worker's own timer), publication latency, total ROS overhead.
  - Aggregate: mean/p50/p95/max for all timing components, dropped-frame count,
    cold-start duration (first frame latency from session start), failure count.

Native engine timing (inference_seconds) is passed through unmodified from the
TensorRT worker — it is NOT recomputed here.

Usage
-----
  python3 collect_ros_metrics.py \\
      --input /tmp/cosmos_bench.jsonl \\
      --metadata metadata.json \\
      --warmup 3 \\
      --output ros_report.json \\
      [--csv ros_report.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ── schema constants ──────────────────────────────────────────────────────────

RECORD_TYPE_SESSION_START = "session_start"
RECORD_TYPE_FRAME = "frame"
RECORD_TYPE_SESSION_END = "session_end"

_NS_PER_MS = 1_000_000.0


# ── helpers ───────────────────────────────────────────────────────────────────


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    sorted_v = sorted(values)
    rank = (p / 100.0) * (len(sorted_v) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return sorted_v[lo]
    w = rank - lo
    return sorted_v[lo] * (1.0 - w) + sorted_v[hi] * w


def _stats(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": statistics.fmean(values) if values else None,
        "p50": _percentile(values, 50.0),
        "p95": _percentile(values, 95.0),
        "max": max(values) if values else None,
        "min": min(values) if values else None,
    }


# ── JSONL parsing ─────────────────────────────────────────────────────────────


def parse_jsonl(path: Path) -> tuple[dict[str, Any] | None, list[dict[str, Any]], dict[str, Any] | None]:
    """
    Parse the benchmark JSONL file.

    Returns (session_start_record, frame_records, session_end_record).
    Unknown record types are silently skipped.
    """
    session_start: dict[str, Any] | None = None
    frames: list[dict[str, Any]] = []
    session_end: dict[str, Any] | None = None

    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL parse error at line {lineno}: {exc}") from exc

            rtype = record.get("record_type", "")
            if rtype == RECORD_TYPE_SESSION_START:
                session_start = record
            elif rtype == RECORD_TYPE_FRAME:
                frames.append(record)
            elif rtype == RECORD_TYPE_SESSION_END:
                session_end = record

    return session_start, frames, session_end


# ── per-frame metric computation ──────────────────────────────────────────────


def compute_frame_metrics(record: dict[str, Any]) -> dict[str, Any]:
    """
    Compute derived timing metrics for a single frame record.

    Timing breakdown
    ────────────────
      dequeue_wall_ns  →  convert_done_ns  : image format conversion + optional resize
      convert_done_ns  →  infer_done_ns    : IPC round-trip (includes native inference)
      infer_done_ns    →  publish_done_ns  : ROS message publish

    ROS overhead = everything except the native inference_seconds timer.
    """
    dequeue = record["dequeue_wall_ns"]
    convert_done = record["convert_done_ns"]
    infer_done = record["infer_done_ns"]
    publish_done = record["publish_done_ns"]
    inference_s = float(record.get("inference_seconds", 0.0))

    image_convert_ms = (convert_done - dequeue) / _NS_PER_MS
    total_ipc_ms = (infer_done - convert_done) / _NS_PER_MS
    inference_ms = inference_s * 1000.0
    # IPC overhead = socket serialization + send + receive, excluding the
    # native engine time reported by the worker itself.
    ipc_overhead_ms = max(0.0, total_ipc_ms - inference_ms)
    publication_ms = (publish_done - infer_done) / _NS_PER_MS
    total_worker_ms = (publish_done - dequeue) / _NS_PER_MS
    ros_overhead_ms = image_convert_ms + ipc_overhead_ms + publication_ms

    return {
        "frame_seq": record["frame_seq"],
        "image_stamp_ns": record.get("image_stamp_ns", 0),
        "dequeue_wall_ns": dequeue,
        "convert_done_ns": convert_done,
        "infer_done_ns": infer_done,
        "publish_done_ns": publish_done,
        "inference_ms": inference_ms,
        "image_convert_ms": image_convert_ms,
        "ipc_overhead_ms": ipc_overhead_ms,
        "publication_ms": publication_ms,
        "ros_overhead_ms": ros_overhead_ms,
        "total_worker_ms": total_worker_ms,
        "success": record.get("success", True),
        "error": record.get("error", ""),
        "dropped_before": record.get("dropped_before", 0),
    }


# ── aggregate computation ─────────────────────────────────────────────────────


def compute_aggregate(
    frame_metrics: list[dict[str, Any]],
    session_start: dict[str, Any] | None,
    session_end: dict[str, Any] | None,
    warmup_frames: int = 0,
) -> dict[str, Any]:
    """
    Compute aggregate statistics across measured frames (excluding warmup).

    Warmup frames are included in total_frames but excluded from timing stats.
    """
    measured = frame_metrics[warmup_frames:]
    successful = [f for f in measured if f["success"]]

    inference_ms_vals = [f["inference_ms"] for f in successful]
    ros_overhead_vals = [f["ros_overhead_ms"] for f in successful]
    ipc_overhead_vals = [f["ipc_overhead_ms"] for f in successful]
    convert_vals = [f["image_convert_ms"] for f in successful]
    publication_vals = [f["publication_ms"] for f in successful]
    total_vals = [f["total_worker_ms"] for f in successful]

    total_dropped = (
        int(session_end["dropped"]) if session_end and "dropped" in session_end else
        (frame_metrics[-1]["dropped_before"] if frame_metrics else 0)
    )

    # Cold start: time from session_start to first frame dequeued
    cold_start_ms: float | None = None
    if session_start and frame_metrics:
        node_start = session_start.get("node_start_wall_ns", 0)
        first_dequeue = frame_metrics[0].get("dequeue_wall_ns", 0)
        if node_start and first_dequeue:
            cold_start_ms = (first_dequeue - node_start) / _NS_PER_MS

    return {
        "total_frames": len(frame_metrics),
        "warmup_frames": warmup_frames,
        "measured_frames": len(measured),
        "successful_frames": len(successful),
        "failed_frames": len(measured) - len(successful),
        "total_dropped": total_dropped,
        "cold_start_ms": cold_start_ms,
        "inference_ms": _stats(inference_ms_vals),
        "ros_overhead_ms": _stats(ros_overhead_vals),
        "ipc_overhead_ms": _stats(ipc_overhead_vals),
        "image_convert_ms": _stats(convert_vals),
        "publication_ms": _stats(publication_vals),
        "total_worker_ms": _stats(total_vals),
    }


# ── report assembly ───────────────────────────────────────────────────────────


def build_report(
    session_start: dict[str, Any] | None,
    frame_metrics: list[dict[str, Any]],
    session_end: dict[str, Any] | None,
    *,
    metadata: dict[str, Any] | None = None,
    warmup_frames: int = 0,
) -> dict[str, Any]:
    aggregate = compute_aggregate(
        frame_metrics, session_start, session_end, warmup_frames=warmup_frames
    )

    report_metadata: dict[str, Any] = metadata.copy() if metadata else {}
    if session_start:
        for key in (
            "task_profile", "prompt_version", "prompt_config_hash",
            "max_generate_length", "sample_period_seconds",
            "image_max_width", "jpeg_quality", "drop_old_frames",
        ):
            if key in session_start:
                report_metadata.setdefault(key, session_start[key])
    report_metadata.setdefault("recorded_at", datetime.now(timezone.utc).isoformat())
    report_metadata["warmup_frames"] = warmup_frames
    report_metadata["measured_frames"] = aggregate["measured_frames"]

    return {
        "metadata": report_metadata,
        "aggregate": aggregate,
        "frames": frame_metrics,
    }


# ── CSV export ────────────────────────────────────────────────────────────────

_CSV_FIELDS = [
    "frame_seq",
    "image_stamp_ns",
    "inference_ms",
    "image_convert_ms",
    "ipc_overhead_ms",
    "publication_ms",
    "ros_overhead_ms",
    "total_worker_ms",
    "success",
    "error",
    "dropped_before",
]


def write_csv(frame_metrics: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(frame_metrics)


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute ROS pipeline overhead metrics from JSONL benchmark output"
    )
    parser.add_argument(
        "--input", required=True, type=Path,
        help="JSONL file written by cosmos_reasoner benchmark_output_file parameter"
    )
    parser.add_argument(
        "--metadata", type=Path, default=None,
        help="Optional JSON metadata file (from benchmark_metadata.py)"
    )
    parser.add_argument(
        "--warmup", type=int, default=3,
        help="Number of leading frames to exclude from timing statistics (default: 3)"
    )
    parser.add_argument(
        "--output", required=True, type=Path,
        help="Output JSON report path"
    )
    parser.add_argument(
        "--csv", type=Path, default=None,
        help="Optional CSV output path for per-frame data"
    )
    args = parser.parse_args()

    session_start, raw_frames, session_end = parse_jsonl(args.input)
    frame_metrics = [compute_frame_metrics(r) for r in raw_frames]

    metadata: dict[str, Any] | None = None
    if args.metadata:
        with args.metadata.open("r", encoding="utf-8") as fh:
            metadata = json.load(fh)

    report = build_report(
        session_start, frame_metrics, session_end,
        metadata=metadata,
        warmup_frames=args.warmup,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print(
        f"ROS metrics report: {args.output}  "
        f"({report['aggregate']['measured_frames']} measured frames, "
        f"{report['aggregate']['failed_frames']} failures, "
        f"{report['aggregate']['total_dropped']} dropped)",
        file=sys.stderr,
    )

    if args.csv:
        write_csv(frame_metrics, args.csv)
        print(f"CSV: {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
