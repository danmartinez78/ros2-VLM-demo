#!/usr/bin/env python3
"""VLM latency benchmark parser and report generator.

The benchmark compares prompt/output-token conditions over direct cold-start and
persistent IPC runtime paths. Runtime stage timings remain null when unavailable;
this tool never fabricates missing measurements.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

SCHEMA_VERSION = "1"
RECORD_TYPE_INFERENCE = "inference"
CONDITION_ORDER = ["A", "B", "C", "D", "E"]

PROMPT_TEXTS: dict[str, str] = {
    "terse_id": "What is in this image?",
    "compact_scene_json": (
        "Describe this scene as compact JSON with keys: objects, actions, hazards, navigable. "
        "Be concise."
    ),
    "scene_description": (
        "Provide a detailed description of the scene including visible objects, their relative "
        "positions, dynamic elements, potential obstacles, and the overall environment type."
    ),
}

EXPERIMENT_MATRIX = [
    ("A", "terse_id", 16),
    ("B", "compact_scene_json", 32),
    ("C", "compact_scene_json", 64),
    ("D", "scene_description", 128),
    ("E", "scene_description", 256),
]

_PROMPT_LABELS = {
    "terse_id": "terse identification",
    "compact_scene_json": "compact structured scene JSON",
    "scene_description": "scene description (verbose)",
}


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def build_condition_spec(condition: str) -> dict[str, Any]:
    for cond, prompt_id, max_tokens in EXPERIMENT_MATRIX:
        if cond == condition:
            text = PROMPT_TEXTS[prompt_id]
            return {
                "condition": cond,
                "prompt_id": prompt_id,
                "prompt_text": text,
                "prompt_hash": prompt_hash(text),
                "max_output_tokens": max_tokens,
            }
    raise ValueError(f"Unknown condition: {condition}")


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = max(0, int(math.ceil(pct / 100.0 * len(ordered))) - 1)
    return ordered[min(idx, len(ordered) - 1)]


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "min": None, "max": None, "p50": None, "p95": None}
    return {
        "n": len(values),
        "mean": fmean(values),
        "min": min(values),
        "max": max(values),
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
    }


def parse_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"WARNING: {path}:{lineno}: skipping malformed JSON: {exc}", file=sys.stderr)
                continue
            if not isinstance(obj, dict):
                continue
            if obj.get("record_type") != RECORD_TYPE_INFERENCE:
                continue
            if str(obj.get("schema_version", "")) != SCHEMA_VERSION:
                print(f"WARNING: {path}:{lineno}: unexpected schema version; skipping", file=sys.stderr)
                continue
            records.append(obj)
    return records


def _decode_tps(rec: dict[str, Any]) -> float | None:
    decode_ms = rec.get("decode_ms")
    tokens = rec.get("actual_output_tokens")
    if decode_ms is not None and tokens is not None and decode_ms > 0:
        return float(tokens) / (float(decode_ms) / 1000.0)
    value = rec.get("decode_tokens_per_sec")
    return float(value) if value is not None else None


def aggregate_condition(records: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [r for r in records if not r.get("warmup", False) and r.get("success", False)]

    def values(key: str) -> list[float]:
        return [float(r[key]) for r in measured if r.get(key) is not None]

    failed = sum(1 for r in records if not r.get("warmup", False) and not r.get("success", False))
    finish: dict[str, int] = {}
    for rec in measured:
        key = str(rec.get("finish_reason")) if rec.get("finish_reason") is not None else "null"
        finish[key] = finish.get(key, 0) + 1

    stage_keys = [
        "prefill_ms",
        "decode_ms",
        "vision_encoder_ms",
        "average_time_per_token_ms",
        "llm_generation_total_gpu_time_ms",
    ]
    result: dict[str, Any] = {
        "n_total": len(records),
        "n_warmup": sum(1 for r in records if r.get("warmup", False)),
        "n_measured": len(measured),
        "n_failed": failed,
        "total_latency_ms": _stats(values("total_latency_ms")),
        "cold_start_total_ms": _stats(values("cold_start_total_ms")),
        "actual_output_tokens": _stats(values("actual_output_tokens")) if values("actual_output_tokens") else {"available": False},
        "finish_reason_counts": finish,
        "n_max_length": finish.get("max-length", 0) + finish.get("max_length", 0),
    }
    availability: dict[str, bool] = {}
    for key in stage_keys:
        vals = values(key)
        availability[key.removesuffix("_ms")] = bool(vals)
        result[key] = _stats(vals) if vals else {"available": False}
    tps = [v for r in measured if (v := _decode_tps(r)) is not None]
    result["decode_tokens_per_sec"] = _stats(tps) if tps else {"available": False}
    availability["decode_tokens_per_sec"] = bool(tps)
    availability["actual_output_tokens"] = bool(values("actual_output_tokens"))
    result["stage_timings_available"] = availability
    return result


def _group(records: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for rec in records:
        grouped.setdefault((str(rec.get("condition", "")), str(rec.get("path", ""))), []).append(rec)
    return grouped


def compute_token_scaling(by_condition_path: dict[tuple[str, str], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (condition, path), records in sorted(by_condition_path.items()):
        measured = [r for r in records if not r.get("warmup", False) and r.get("success", False)]
        latencies = [float(r["total_latency_ms"]) for r in measured if r.get("total_latency_ms") is not None]
        if not measured or not latencies:
            continue
        outputs = [float(r["actual_output_tokens"]) for r in measured if r.get("actual_output_tokens") is not None]
        rows.append({
            "condition": condition,
            "path": path,
            "prompt_id": measured[0].get("prompt_id", ""),
            "max_output_tokens": measured[0].get("max_output_tokens"),
            "actual_output_tokens_mean": fmean(outputs) if outputs else None,
            "total_latency_ms_mean": fmean(latencies),
            "total_latency_ms_p95": _percentile(latencies, 95),
            "n_measured": len(measured),
        })
    return rows


def compute_cold_start_scaling(by_condition_path: dict[tuple[str, str], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (condition, path), records in sorted(by_condition_path.items()):
        if path != "direct":
            continue
        measured = [r for r in records if not r.get("warmup", False) and r.get("success", False)]
        vals = [float(r["cold_start_total_ms"]) for r in measured if r.get("cold_start_total_ms") is not None]
        if not vals:
            continue
        rows.append({
            "condition": condition,
            "path": path,
            "prompt_id": measured[0].get("prompt_id", ""),
            "max_output_tokens": measured[0].get("max_output_tokens"),
            "cold_start_total_ms_mean": fmean(vals),
            "cold_start_total_ms_p95": _percentile(vals, 95),
            "n_measured": len(measured),
        })
    return rows


def compute_direct_ipc_comparison(by_condition_path: dict[tuple[str, str], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for condition in sorted({c for c, _ in by_condition_path}):
        direct = by_condition_path.get((condition, "direct"), [])
        ipc = by_condition_path.get((condition, "ipc"), [])
        if not direct or not ipc:
            continue
        d = aggregate_condition(direct)["cold_start_total_ms"].get("mean")
        i = aggregate_condition(ipc)["total_latency_ms"].get("mean")
        rows.append({
            "condition": condition,
            "direct_cold_start_total_ms_mean": d,
            "ipc_total_latency_ms_mean": i,
            "note": "direct=cold start; ipc=persistent steady state; not directly comparable as overhead",
        })
    return rows


def compute_native_profile_table(by_condition_path: dict[tuple[str, str], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for (condition, path), records in sorted(by_condition_path.items()):
        if path != "direct":
            continue
        for rec in records:
            if rec.get("warmup", False) or not rec.get("success", False):
                continue
            rows.append({
                "condition": condition,
                "image_id": rec.get("image_id"),
                "iteration": rec.get("iteration"),
                "actual_output_tokens": rec.get("actual_output_tokens"),
                "vision_encoder_ms": rec.get("vision_encoder_ms"),
                "prefill_ms": rec.get("prefill_ms"),
                "decode_tokens_per_sec": _decode_tps(rec),
                "llm_generation_total_gpu_time_ms": rec.get("llm_generation_total_gpu_time_ms"),
                "finish_reason": rec.get("finish_reason"),
            })
    return rows


def build_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped = _group(records)
    aggregates = {
        f"{condition}:{path}": aggregate_condition(group)
        for (condition, path), group in sorted(grouped.items())
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_matrix": [build_condition_spec(c) for c in CONDITION_ORDER],
        "aggregates": aggregates,
        "token_scaling": compute_token_scaling(grouped),
        "cold_start_scaling": compute_cold_start_scaling(grouped),
        "direct_ipc_comparison": compute_direct_ipc_comparison(grouped),
        "native_profile_detail": compute_native_profile_table(grouped),
        "raw_records": records,
    }


def _fmt(value: Any, digits: int = 1) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def format_text_report(report: dict[str, Any]) -> str:
    lines = [
        "VLM Latency Characterization Benchmark Report",
        "=" * 47,
        "",
        "Inference Latency vs Output-Token Cap",
        "-------------------------------------",
    ]
    for row in report.get("token_scaling", []):
        lines.append(
            f"{row['condition']} {row['path']}: cap={row['max_output_tokens']} "
            f"mean={_fmt(row['total_latency_ms_mean'])} ms p95={_fmt(row['total_latency_ms_p95'])} ms"
        )

    lines += ["", "Cold-Start Wall Time vs Output-Token Cap", "-----------------------------------------"]
    lines.append("Direct cold-start includes process/engine/tokenizer initialisation.")
    for row in report.get("cold_start_scaling", []):
        lines.append(
            f"{row['condition']} direct: cap={row['max_output_tokens']} "
            f"mean={_fmt(row['cold_start_total_ms_mean'])} ms"
        )

    comparisons = report.get("direct_ipc_comparison", [])
    if comparisons:
        lines += [
            "",
            "Direct (cold-start, per-process) vs IPC (persistent server, steady-state)",
            "-----------------------------------------------------------------------",
        ]
        for row in comparisons:
            lines.append(
                f"{row['condition']}: direct={_fmt(row['direct_cold_start_total_ms_mean'])} ms "
                f"ipc={_fmt(row['ipc_total_latency_ms_mean'])} ms"
            )

    profiles = report.get("native_profile_detail", [])
    if profiles:
        lines += ["", "Native NVIDIA Profile Detail", "----------------------------"]
        for row in profiles:
            lines.append(
                f"{row['condition']} image={row.get('image_id')} vision={_fmt(row.get('vision_encoder_ms'))} "
                f"prefill={_fmt(row.get('prefill_ms'))} finish={row.get('finish_reason') or 'n/a'}"
            )

    for key, agg in report.get("aggregates", {}).items():
        unavailable = []
        for field, label in (("prefill_ms", "prefill time"), ("decode_ms", "decode time"), ("vision_encoder_ms", "vision encoder time")):
            if agg.get(field, {}).get("available") is False:
                unavailable.append(label)
        if unavailable:
            path = key.split(":", 1)[1] if ":" in key else key
            for label in unavailable:
                lines.append(f"{path}: {label}: NOT available from runtime (null, not inferred)")

    finish_values = [
        rec.get("finish_reason")
        for rec in report.get("raw_records", [])
        if rec.get("finish_reason") is not None
    ]
    if any(v in ("max-length", "max_length") for v in finish_values):
        lines += ["", "NOTE: requests with finish_reason=max-length are capped, not naturally task-complete."]

    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--text", type=Path)
    args = ap.parse_args()

    report = build_report(parse_jsonl(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    text = format_text_report(report)
    if args.text:
        args.text.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
