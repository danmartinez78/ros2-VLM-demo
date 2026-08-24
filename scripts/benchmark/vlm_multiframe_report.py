#!/usr/bin/env python3
"""
VLM multi-frame latency characterization benchmark — parser and report generator.

Reads a JSONL file of per-inference records written by
run_vlm_multiframe_benchmark.sh and produces:

  - Machine-readable JSON summary (per-frame-count aggregates, raw records)
  - Human-readable text frame-scaling report

Frame-count conditions: F1 (1 frame), F2 (2 frames), F4 (4 frames), F8 (8 frames).

All frame-count conditions hold constant: model/engines/precision, prompt text,
max output tokens = 32, image resolution/preprocessing, runtime power/clocks.

TTFT and stage timings are preserved as null when the runtime does not
expose them; they are never inferred or fabricated.

NOTE on lifecycle semantics
---------------------------
  direct : cold-start, per-process — includes engine/tokenizer init.
  ipc    : persistent server steady-state — connects to a running server.
  These two quantities are NOT comparable as "overhead".  The report
  presents them side-by-side for independent analysis only.
  Cold-start process wall time (``cold_start_total_ms``) is kept separate
  from model/runtime timings (``total_latency_ms``).

Usage
-----
  python3 vlm_multiframe_report.py \\
      --input vlm_multiframe_YYYYMMDD_HHMMSS.jsonl \\
      --output vlm_multiframe_report.json \\
      [--text vlm_multiframe_report.txt]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any


# ── constants ─────────────────────────────────────────────────────────────────

SCHEMA_VERSION = "1"
RECORD_TYPE_INFERENCE = "inference"

# Ordered frame-count condition labels
FRAME_COUNT_ORDER = ["F1", "F2", "F4", "F8"]

# Compact temporal prompt text (semantically equivalent to B/C compact_odd_json
# but requests one structured result for the whole frame set).
MULTIFRAME_PROMPT_TEXT = (
    "You are an autonomous robot perception system. "
    "Analyze this temporal sequence of images and describe the scene as compact JSON "
    "with keys: objects, actions, hazards, navigable. "
    "Be concise. Provide one result for the full sequence."
)

MAX_OUTPUT_TOKENS = 32
ENGINE_PROVENANCE_KEYS = (
    "model_name",
    "engine_profile_id",
    "llm_engine_dir",
    "multimodal_engine_dir",
    "engine_manifest_path",
    "engine_manifest_sha256",
    "engine_identity",
    "engine_manifest_status",
)


# ── statistics helpers ────────────────────────────────────────────────────────


def _percentile(values: list[float], pct: float) -> float | None:
    """Return the pct-th percentile (0–100) of a sorted list."""
    if not values:
        return None
    ordered = sorted(values)
    idx = max(0, int(math.ceil(pct / 100.0 * len(ordered))) - 1)
    idx = min(idx, len(ordered) - 1)
    return ordered[idx]


def _stats(values: list[float]) -> dict[str, Any]:
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


# ── JSONL parsing ─────────────────────────────────────────────────────────────


def parse_jsonl(path: Path) -> list[dict[str, Any]]:
    """
    Parse a JSONL file of per-inference records.

    Returns only records with record_type == "inference" and schema_version == "1".
    Malformed lines are skipped with a warning to stderr.
    """
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"WARNING: {path}:{lineno}: skipping malformed JSON: {exc}",
                    file=sys.stderr,
                )
                continue
            if not isinstance(obj, dict):
                continue
            if obj.get("record_type") != RECORD_TYPE_INFERENCE:
                continue
            if str(obj.get("schema_version", "")) != SCHEMA_VERSION:
                print(
                    f"WARNING: {path}:{lineno}: unexpected schema_version "
                    f"{obj.get('schema_version')!r}, skipping",
                    file=sys.stderr,
                )
                continue
            records.append(obj)
    return records


# ── frame selection ───────────────────────────────────────────────────────────


def select_frames(
    all_frames: list[str],
    frame_count: int,
) -> list[str]:
    """
    Deterministically select ``frame_count`` frames from ``all_frames``.

    Frames are assumed to be in temporal order.  Selection uses evenly-spaced
    indices so that the first and last frames are always included.

    Raises ValueError if ``len(all_frames) < frame_count``.
    """
    n = len(all_frames)
    if n < frame_count:
        raise ValueError(
            f"Insufficient frames: need {frame_count}, have {n}. "
            f"Provide a sequence directory with at least {frame_count} frames."
        )
    if frame_count == 1:
        return [all_frames[0]]
    if frame_count == n:
        return list(all_frames)
    # Evenly-spaced indices, always including index 0 and index n-1.
    step = (n - 1) / (frame_count - 1)
    indices = [round(i * step) for i in range(frame_count)]
    return [all_frames[i] for i in indices]


def frame_condition_label(frame_count: int) -> str:
    """Return the condition label for a given frame count (e.g., 1 → 'F1')."""
    return f"F{frame_count}"


# ── content hashing ───────────────────────────────────────────────────────────


def file_sha256(path: str | Path) -> str:
    """Return the full SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def file_sha256_prefix(path: str | Path) -> str:
    """Return the first 12 hex characters of the SHA-256 digest of a file."""
    return file_sha256(path)[:12]


def prompt_hash(text: str) -> str:
    """Return the first 12 hex characters of the SHA-256 digest of a prompt string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


# ── NVIDIA request shape construction ────────────────────────────────────────


def build_multiframe_request(
    frame_paths: list[str],
    prompt_text: str,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
) -> dict[str, Any]:
    """
    Build the Thor-validated TensorRT Edge-LLM VLM request JSON for multiple image frames.

    Uses the pinned Thor request shape:
      requests -> messages -> content[]
    with role: user.

    Multiple image content items (``{"type":"image","image":path}``) are placed
    in temporal order, followed by one text prompt.  This is the exact shape
    used by ``tests/test_cases/vlm_basic.json`` in the TensorRT Edge-LLM checkout
    and validated by the single-frame benchmark.

    ``max_output_tokens`` is NOT embedded in the request payload; it is passed
    to the runtime via ``--maxGenerateLength``.

    SHA-256 content hashes are stored in benchmark metadata (JSONL
    ``frame_paths`` field), not inside the model message payload.
    """
    content: list[dict[str, Any]] = [
        {"type": "image", "image": img_path}
        for img_path in frame_paths
    ]
    content.append({"type": "text", "text": prompt_text})

    return {
        "requests": [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": content,
                    }
                ]
            }
        ],
    }


def build_multiframe_request_metadata(
    frame_paths: list[str],
    prompt_text: str,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
) -> dict[str, Any]:
    """
    Return metadata dict describing a multi-frame request without reading file bytes.

    Suitable for unit tests and dry-run modes where actual image files may not exist.
    """
    images_meta = [
        {"path": p, "index": i} for i, p in enumerate(frame_paths)
    ]
    return {
        "frame_count": len(frame_paths),
        "frame_paths": frame_paths,
        "images": images_meta,
        "prompt_text": prompt_text,
        "prompt_hash": prompt_hash(prompt_text),
        "max_output_tokens": max_output_tokens,
    }


# ── per-condition aggregation ─────────────────────────────────────────────────


def _decode_tps(rec: dict[str, Any]) -> float | None:
    """Compute decode tokens/sec if decode_ms and actual_output_tokens are available."""
    decode_ms = rec.get("decode_ms")
    tokens = rec.get("actual_output_tokens")
    if decode_ms is not None and tokens is not None and decode_ms > 0:
        return tokens / (decode_ms / 1000.0)
    return rec.get("decode_tokens_per_sec")


def aggregate_frame_condition(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Aggregate inference records for a single (frame_condition, path) group.

    Only non-warmup, successful records are included in timing aggregates.
    """
    measured = [r for r in records if not r.get("warmup", False) and r.get("success", False)]
    total_latency = [r["total_latency_ms"] for r in measured if r.get("total_latency_ms") is not None]
    cold_start_total = [r["cold_start_total_ms"] for r in measured if r.get("cold_start_total_ms") is not None]
    prefill = [r["prefill_ms"] for r in measured if r.get("prefill_ms") is not None]
    decode = [r["decode_ms"] for r in measured if r.get("decode_ms") is not None]
    vision_encoder = [r["vision_encoder_ms"] for r in measured if r.get("vision_encoder_ms") is not None]
    llm_gen_gpu = [r["llm_generation_total_gpu_time_ms"] for r in measured if r.get("llm_generation_total_gpu_time_ms") is not None]
    output_tokens = [r["actual_output_tokens"] for r in measured if r.get("actual_output_tokens") is not None]
    visual_tokens = [r["total_image_tokens"] for r in measured if r.get("total_image_tokens") is not None]
    decode_tps_vals = [v for r in measured if (v := _decode_tps(r)) is not None]

    failed = sum(1 for r in records if not r.get("warmup", False) and not r.get("success", False))
    warmup_count = sum(1 for r in records if r.get("warmup", False))

    # Determine whether stage timings were available at all
    prefill_available = any(r.get("prefill_ms") is not None for r in measured)
    decode_available = any(r.get("decode_ms") is not None for r in measured)
    vision_encoder_available = any(r.get("vision_encoder_ms") is not None for r in measured)
    llm_gen_gpu_available = any(r.get("llm_generation_total_gpu_time_ms") is not None for r in measured)
    visual_tokens_available = any(r.get("total_image_tokens") is not None for r in measured)

    # finish_reason breakdown
    finish_reason_counts: dict[str, int] = {}
    for r in measured:
        fr = r.get("finish_reason")
        key = str(fr) if fr is not None else "null"
        finish_reason_counts[key] = finish_reason_counts.get(key, 0) + 1
    n_max_length = (
        finish_reason_counts.get("max-length", 0)
        + finish_reason_counts.get("max_length", 0)
    )

    return {
        "n_total": len(records),
        "n_warmup": warmup_count,
        "n_measured": len(measured),
        "n_failed": failed,
        "total_latency_ms": _stats(total_latency),
        "cold_start_total_ms": _stats(cold_start_total),
        "prefill_ms": _stats(prefill) if prefill_available else {"available": False},
        "decode_ms": _stats(decode) if decode_available else {"available": False},
        "vision_encoder_ms": _stats(vision_encoder) if vision_encoder_available else {"available": False},
        "llm_generation_total_gpu_time_ms": _stats(llm_gen_gpu) if llm_gen_gpu_available else {"available": False},
        "actual_output_tokens": _stats(output_tokens) if output_tokens else {"available": False},
        "total_image_tokens": _stats(visual_tokens) if visual_tokens_available else {"available": False},
        "decode_tokens_per_sec": _stats(decode_tps_vals) if decode_tps_vals else {"available": False},
        "stage_timings_available": {
            "prefill": prefill_available,
            "decode": decode_available,
            "vision_encoder": vision_encoder_available,
            "llm_generation_total_gpu_time": llm_gen_gpu_available,
            "total_image_tokens": visual_tokens_available,
            "actual_output_tokens": bool(output_tokens),
            "decode_tokens_per_sec": bool(decode_tps_vals),
        },
        "finish_reason_counts": finish_reason_counts,
        "n_max_length": n_max_length,
    }


# ── frame-scaling table ───────────────────────────────────────────────────────


def compute_frame_scaling_table(
    by_condition_path: dict[tuple[str, str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """
    Build a frame-scaling table showing how metrics scale with frame count.

    Columns per the issue specification:
    Frames | Images/request | Visual tokens | Vision ms | Prefill ms |
    Generated tokens | Gen tok/s | Generation GPU ms |
    IPC mean ms | IPC p95 ms | Finish reason

    One row per (frame_condition, path) combination.
    """
    rows: list[dict[str, Any]] = []
    for (frame_condition, path), records in sorted(by_condition_path.items()):
        measured = [
            r for r in records
            if not r.get("warmup", False) and r.get("success", False)
        ]
        if not measured:
            continue

        frame_count = measured[0].get("frame_count")
        visual_tokens = [r["total_image_tokens"] for r in measured if r.get("total_image_tokens") is not None]
        vision_ms = [r["vision_encoder_ms"] for r in measured if r.get("vision_encoder_ms") is not None]
        prefill_ms = [r["prefill_ms"] for r in measured if r.get("prefill_ms") is not None]
        out_tokens = [r["actual_output_tokens"] for r in measured if r.get("actual_output_tokens") is not None]
        decode_tps = [v for r in measured if (v := _decode_tps(r)) is not None]
        gpu_ms = [r["llm_generation_total_gpu_time_ms"] for r in measured if r.get("llm_generation_total_gpu_time_ms") is not None]
        ipc_latency = [
            r["total_latency_ms"] for r in measured
            if path == "ipc" and r.get("total_latency_ms") is not None
        ]
        finish_reasons = list({r.get("finish_reason") for r in measured if r.get("finish_reason") is not None})

        row: dict[str, Any] = {
            "frame_condition": frame_condition,
            "path": path,
            "frames": frame_count,
            "images_per_request": frame_count,
            "visual_tokens_mean": fmean(visual_tokens) if visual_tokens else None,
            "vision_encoder_ms_mean": fmean(vision_ms) if vision_ms else None,
            "prefill_ms_mean": fmean(prefill_ms) if prefill_ms else None,
            "generated_tokens_mean": fmean(out_tokens) if out_tokens else None,
            "gen_tokens_per_sec_mean": fmean(decode_tps) if decode_tps else None,
            "generation_gpu_ms_mean": fmean(gpu_ms) if gpu_ms else None,
            "ipc_total_latency_ms_mean": fmean(ipc_latency) if ipc_latency else None,
            "ipc_total_latency_ms_p95": _percentile(ipc_latency, 95) if ipc_latency else None,
            "finish_reason": finish_reasons[0] if len(finish_reasons) == 1 else (finish_reasons or None),
            "n_measured": len(measured),
        }
        rows.append(row)
    return rows


# ── IPC artifact table ────────────────────────────────────────────────────────


def compute_ipc_artifact_table(
    by_condition_path: dict[tuple[str, str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """
    Build a table of IPC result artifact paths for each measured inference.

    Preserves: ipc_result_path, outer client total latency, backend-reported
    inference_seconds, output text, output words.

    Does NOT infer output token count or finish reason from IPC backend.
    """
    rows: list[dict[str, Any]] = []
    for (frame_condition, path), records in sorted(by_condition_path.items()):
        if path != "ipc":
            continue
        for rec in records:
            if rec.get("warmup", False) or not rec.get("success", False):
                continue
            rows.append(
                {
                    "frame_condition": frame_condition,
                    "iteration": rec.get("iteration"),
                    "ipc_result_path": rec.get("ipc_result_path"),
                    "total_latency_ms": rec.get("total_latency_ms"),
                    "inference_seconds": rec.get("inference_seconds"),
                    "output_text": rec.get("output_text"),
                    "output_words": rec.get("output_words"),
                }
            )
    return rows


def normalize_engine_provenance(provenance: Any) -> dict[str, Any] | None:
    """Normalize a raw provenance object from benchmark records."""
    if not isinstance(provenance, dict):
        return None
    normalized = {key: provenance.get(key) for key in ENGINE_PROVENANCE_KEYS}
    warnings = provenance.get("provenance_warnings")
    normalized["provenance_warnings"] = list(warnings) if isinstance(warnings, list) else []
    if not any(value for key, value in normalized.items() if key != "provenance_warnings"):
        return None
    return normalized


def collect_unique_engine_provenance(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collect distinct normalized provenance objects from raw records."""
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        normalized = normalize_engine_provenance(record.get("engine_provenance"))
        if normalized is None:
            continue
        key = json.dumps(normalized, sort_keys=True)
        if key in seen:
            continue
        unique.append(normalized)
        seen.add(key)
    return unique


def collect_unique_max_output_tokens(records: list[dict[str, Any]]) -> list[int | str]:
    """Collect distinct max_output_tokens values from raw records."""
    unique: list[int | str] = []
    seen: set[str] = set()
    for record in records:
        value = record.get("max_output_tokens")
        if value is None:
            continue
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        key = json.dumps(value, sort_keys=True)
        if key in seen:
            continue
        unique.append(value)
        seen.add(key)
    if all(isinstance(value, int) and not isinstance(value, bool) for value in unique):
        unique.sort()
    return unique


def collect_temporal_config_variants(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collect distinct temporal configuration/effective-runtime combinations."""
    variants: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        item = {
            "sequence_type": record.get("sequence_type"),
            "fps": record.get("fps"),
            "frame_timestamps_sec": record.get("frame_timestamps_sec"),
            "frame_timestamp_policy": record.get("frame_timestamp_policy"),
            "rendered_timestamps": record.get("rendered_timestamps"),
            "requested_sequence_type": record.get("requested_sequence_type"),
            "runtime_temporal_encoding": record.get("runtime_temporal_encoding"),
            "temporal_fallback_used": record.get("temporal_fallback_used"),
        }
        key = json.dumps(item, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        variants.append(item)
    return variants


# ── full report generation ────────────────────────────────────────────────────


def build_report(
    records: list[dict[str, Any]],
    *,
    source_path: Path | None = None,
) -> dict[str, Any]:
    """
    Build a machine-readable JSON report from raw multi-frame inference records.

    Raw per-inference records are preserved in the report so later analysis
    is not limited to averages.
    """
    by_condition_path: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for rec in records:
        key = (rec.get("frame_condition", "?"), rec.get("path", "?"))
        by_condition_path.setdefault(key, []).append(rec)

    conditions_summary: dict[str, Any] = {}
    for (frame_condition, path), group in sorted(by_condition_path.items()):
        conditions_summary.setdefault(frame_condition, {})[path] = aggregate_frame_condition(group)

    run_ids = list({r.get("run_id", "") for r in records if r.get("run_id")})
    model_names = list({r.get("model_name", "") for r in records if r.get("model_name")})
    provenance_variants = collect_unique_engine_provenance(records)
    mixed_engine_provenance = len(provenance_variants) > 1
    temporal_variants = collect_temporal_config_variants(records)

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_file": str(source_path) if source_path else None,
        "run_ids": run_ids,
        "model_names": model_names,
        "n_total_records": len(records),
        "n_measured_records": sum(
            1 for r in records if not r.get("warmup", False) and r.get("success", False)
        ),
        "engine_provenance": provenance_variants[0] if len(provenance_variants) == 1 else None,
        "engine_provenance_variants": provenance_variants,
        "mixed_engine_provenance": mixed_engine_provenance,
        "temporal_config_variants": temporal_variants,
        "frame_conditions": conditions_summary,
        "frame_scaling_table": compute_frame_scaling_table(by_condition_path),
        "ipc_artifact_table": compute_ipc_artifact_table(by_condition_path),
        "raw_records": records,
    }


# ── text report formatting ────────────────────────────────────────────────────


def _fmt_ms(val: float | None) -> str:
    if val is None:
        return "    n/a"
    return f"{val:7.1f}"


def _fmt_f(val: float | None, width: int = 8) -> str:
    if val is None:
        return (" " * (width - 3)) + "n/a"
    return f"{val:{width}.1f}"


def format_text_report(report: dict[str, Any]) -> str:
    lines: list[str] = [
        "=" * 80,
        "  VLM Multi-Frame Latency Characterization Benchmark Report",
        f"  Generated: {report.get('generated_at', 'unknown')}",
        "=" * 80,
    ]

    model_names = report.get("model_names") or []
    run_ids = report.get("run_ids") or []
    max_output_tokens_values = collect_unique_max_output_tokens(report.get("raw_records") or [])
    mixed_model_or_engine = report.get("mixed_engine_provenance") or len(model_names) > 1
    if model_names:
        lines += ["", f"  Model(s): {', '.join(model_names)}"]
    if run_ids:
        lines += [f"  Run ID(s): {', '.join(run_ids)}"]
    provenance = report.get("engine_provenance")
    provenance_variants = report.get("engine_provenance_variants") or []
    if provenance:
        lines += [
            "",
            "  Engine provenance:",
            f"    Identity: {provenance.get('engine_identity') or 'unknown'}",
            f"    Model/profile: {provenance.get('model_name') or 'unknown'} / {provenance.get('engine_profile_id') or 'unknown'}",
            f"    LLM engine dir: {provenance.get('llm_engine_dir') or 'n/a'}",
            f"    Multimodal dir: {provenance.get('multimodal_engine_dir') or 'n/a'}",
        ]
        if provenance.get("engine_manifest_path"):
            lines.append(f"    Engine manifest: {provenance.get('engine_manifest_path')}")
        if provenance.get("engine_manifest_sha256"):
            lines.append(f"    Manifest SHA256: {provenance.get('engine_manifest_sha256')}")
        if provenance.get("engine_manifest_status"):
            lines.append(f"    Manifest status: {provenance.get('engine_manifest_status')}")
        for warning in provenance.get("provenance_warnings") or []:
            lines.append(f"    WARNING: {warning}")
    elif report.get("mixed_engine_provenance"):
        lines += [
            "",
            "  Engine provenance: MIXED",
            "    Multiple engine identities were recorded in this JSONL file.",
        ]
        for item in provenance_variants:
            lines.append(
                f"    - {item.get('engine_identity') or 'unknown'}"
                f" ({item.get('llm_engine_dir') or 'n/a'} | {item.get('multimodal_engine_dir') or 'n/a'})"
            )
    lines += [
        f"  Total records: {report.get('n_total_records', 'n/a')}",
        f"  Measured (non-warmup, success): {report.get('n_measured_records', 'n/a')}",
        "",
        "  Frame-count conditions: F1=1 frame, F2=2 frames, F4=4 frames, F8=8 frames",
    ]
    if mixed_model_or_engine:
        lines.append(
            "  Mixed/non-comparable: model/engine configuration varies across records"
        )
        fixed_fields = ["precision", "prompt text"]
    else:
        fixed_fields = ["model", "engines", "precision", "prompt text"]
    if len(max_output_tokens_values) == 1:
        fixed_fields.append(f"max_output_tokens={max_output_tokens_values[0]}")
    elif not max_output_tokens_values:
        fixed_fields.append("max_output_tokens=unknown")
    lines.append(f"  Fixed: {', '.join(fixed_fields)}")
    if len(max_output_tokens_values) > 1:
        mixed_values = ", ".join(str(value) for value in max_output_tokens_values)
        lines.append(
            f"  Mixed request config: max_output_tokens varies across records ({mixed_values})"
        )
    lines.append("  Prompt policy: compact temporal JSON (one structured result for full sequence)")
    temporal_variants = report.get("temporal_config_variants") or []
    if temporal_variants:
        lines += ["", "  Temporal representation variants:"]
        for variant in temporal_variants:
            lines.append(
                "    - "
                f"sequence_type={variant.get('sequence_type')}, "
                f"fps={variant.get('fps')}, "
                f"timestamp_policy={variant.get('frame_timestamp_policy')}, "
                f"rendered_timestamps={variant.get('rendered_timestamps')}, "
                f"runtime_temporal_encoding={variant.get('runtime_temporal_encoding')}, "
                f"fallback={variant.get('temporal_fallback_used')}"
            )

    # ── Frame-scaling table ───────────────────────────────────────────────
    scaling = report.get("frame_scaling_table") or []
    if scaling:
        lines += [
            "",
            "Frame Scaling Table",
            "  Primary comparison: how additional frames affect visual-token count,",
            "  vision-encoder time, prefill time, generation latency, and behavior.",
            "-" * 80,
            f"  {'Cond':<5} {'Path':<8} {'F':>2} {'Img/req':>7} {'VisTok':>7} "
            f"{'VisionMs':>9} {'PrefMs':>7} {'GenTok':>6} "
            f"{'TokS':>6} {'GpuMs':>7} {'IPCmn':>7} {'IPCp95':>7} {'FinRsn':<10}",
            "  " + "-" * 76,
        ]
        for row in scaling:
            fin = row.get("finish_reason")
            if isinstance(fin, list):
                fin_str = "/".join(str(x) for x in fin) if fin else "n/a"
            else:
                fin_str = str(fin) if fin is not None else "n/a"
            lines.append(
                f"  {row.get('frame_condition', '?'):<5} "
                f"{row.get('path', '?'):<8} "
                f"{str(row.get('frames') or '?'):>2} "
                f"{str(row.get('images_per_request') or '?'):>7} "
                f"{_fmt_f(row.get('visual_tokens_mean'), 7):>7} "
                f"{_fmt_f(row.get('vision_encoder_ms_mean'), 9):>9} "
                f"{_fmt_f(row.get('prefill_ms_mean'), 7):>7} "
                f"{_fmt_f(row.get('generated_tokens_mean'), 6):>6} "
                f"{_fmt_f(row.get('gen_tokens_per_sec_mean'), 6):>6} "
                f"{_fmt_f(row.get('generation_gpu_ms_mean'), 7):>7} "
                f"{_fmt_ms(row.get('ipc_total_latency_ms_mean')):>7} "
                f"{_fmt_ms(row.get('ipc_total_latency_ms_p95')):>7} "
                f"{fin_str:<10}"
            )
        lines += [
            "",
            "  NOTE: VisionMs/PrefMs/GpuMs are NVIDIA-authoritative from --dumpProfile.",
            "  NOTE: null fields mean the runtime did not expose them; they are never inferred.",
            "  NOTE: IPC latency is outer client total (persistent server steady-state).",
            "  NOTE: direct cold-start wall time is kept in the Cold-Start section below.",
        ]

    # ── Cold-start section ────────────────────────────────────────────────
    conditions_summary = report.get("frame_conditions") or {}
    cold_start_rows = []
    for fc in FRAME_COUNT_ORDER:
        if fc not in conditions_summary:
            continue
        direct_data = conditions_summary[fc].get("direct")
        if not direct_data:
            continue
        cs = direct_data.get("cold_start_total_ms") or {}
        cs_mean = cs.get("mean")
        cs_p95 = cs.get("p95")
        if cs_mean is not None:
            cold_start_rows.append((fc, cs_mean, cs_p95, direct_data.get("n_measured", 0)))

    if cold_start_rows:
        lines += [
            "",
            "Direct Cold-Start Wall Time  (per-process, includes engine/tokenizer init)",
            "  NOTE: kept separate from IPC steady-state; not comparable as overhead.",
            "-" * 80,
            f"  {'Cond':<5} {'Cold-start mean ms':>20} {'p95 ms':>8} {'N':>4}",
            "  " + "-" * 40,
        ]
        for fc, mean_ms, p95_ms, n in cold_start_rows:
            lines.append(
                f"  {fc:<5} {_fmt_ms(mean_ms):>20} {_fmt_ms(p95_ms):>8} {n:>4}"
            )

    # ── IPC artifact table ────────────────────────────────────────────────
    ipc_artifacts = report.get("ipc_artifact_table") or []
    if ipc_artifacts:
        lines += [
            "",
            "IPC Result Artifacts  (ipc path, measured inferences only)",
            "-" * 80,
            f"  {'Cond':<5} {'Iter':>4} {'IPCms':>7} {'InfSec':>8}  {'Words':>5}  Path",
            "  " + "-" * 72,
        ]
        for row in ipc_artifacts:
            inf_s = row.get("inference_seconds")
            inf_str = f"{inf_s:8.3f}" if inf_s is not None else "     n/a"
            words = row.get("output_words")
            words_str = f"{words:5d}" if words is not None else "  n/a"
            lines.append(
                f"  {row.get('frame_condition', '?'):<5} "
                f"{row.get('iteration', '?'):>4} "
                f"{_fmt_ms(row.get('total_latency_ms')):>7} "
                f"{inf_str:>8}  "
                f"{words_str:>5}  "
                f"{row.get('ipc_result_path') or 'n/a'}"
            )

    # ── Stage timing availability ─────────────────────────────────────────
    stage_avail_by_path: dict[str, dict[str, bool]] = {}
    for fc_data in conditions_summary.values():
        for path_label, path_data in fc_data.items():
            avail = path_data.get("stage_timings_available") or {}
            entry = stage_avail_by_path.setdefault(path_label, {})
            for stage_key, stage_label in (
                ("prefill", "prefill time"),
                ("decode", "decode time"),
                ("vision_encoder", "vision encoder time"),
                ("total_image_tokens", "visual token count"),
            ):
                present = avail.get(stage_key, False)
                if present:
                    entry[stage_label] = True
                elif stage_label not in entry:
                    entry[stage_label] = False

    avail_lines: list[str] = []
    for path_label in ("direct", "ipc"):
        if path_label not in stage_avail_by_path:
            continue
        for stage_label, is_available in sorted(stage_avail_by_path[path_label].items()):
            if not is_available:
                avail_lines.append(
                    f"  {path_label}: {stage_label}: NOT available from runtime"
                    " — reported as null, not inferred"
                )
    if avail_lines:
        lines += [
            "",
            "Stage Timing Availability  (per path)",
            "-" * 80,
        ]
        lines += avail_lines

    lines += ["", "=" * 80, ""]
    return "\n".join(lines)


# ── record serialisers (called by run_vlm_multiframe_benchmark.sh) ────────────
#
# The shell script passes all dynamic values as environment variables with the
# ``_BM_`` prefix so that no shell variable is ever interpolated directly into
# Python source code.  This prevents NameError on JSON ``null``/``true``/``false``
# and injection of arbitrary JSON strings.
#
# Scalar typed values (null, bool, number, JSON string, JSON array/object) are
# passed pre-JSON-encoded; they are decoded with json.loads() below.
# Plain strings (timestamps, path strings, model name) are passed raw.


def _jl(env: dict[str, str], key: str, default: Any = None) -> Any:
    """JSON-decode an environment variable; return ``default`` when absent/invalid."""
    v = env.get(key)
    if v is None:
        return default
    try:
        return json.loads(v)
    except (json.JSONDecodeError, ValueError):
        return default


def _js(env: dict[str, str], key: str, default: str = "") -> str:
    """Return the raw string value of an environment variable."""
    return env.get(key, default)


def build_direct_record(env: dict[str, str] | None = None) -> str:
    """Build a direct-path JSONL inference record from ``_BM_*`` environment variables.

    Reads typed values with :func:`_jl` (JSON-decoded) and plain strings with
    :func:`_js`.  Returns a compact JSON string suitable for appending to a
    JSONL file.

    Environment variables consumed
    ──────────────────────────────
    _BM_RUN_ID                run timestamp string
    _BM_RECORDED_AT           ISO-8601 timestamp string
    _BM_FRAME_CONDITION       label e.g. "F1"
    _BM_FRAME_COUNT           JSON integer
    _BM_FRAME_HASHES          JSON array  [{path, sha256}, ...]
    _BM_PROMPT_HASH           12-hex string
    _BM_SEQUENCE_TYPE         JSON string
    _BM_FPS                   JSON number or null
    _BM_FRAME_TIMESTAMPS_SEC  JSON array or null
    _BM_FRAME_TIMESTAMP_POLICY JSON string
    _BM_RENDERED_TIMESTAMPS   JSON bool
    _BM_RUNTIME_TEMPORAL_ENCODING JSON string or null
    _BM_TEMPORAL_FALLBACK_USED JSON bool or null
    _BM_MAX_OUTPUT_TOKENS     JSON integer
    _BM_ACTUAL_OUTPUT_TOKENS  JSON integer or null
    _BM_TOTAL_IMAGE_TOKENS    JSON integer or null
    _BM_FINISH_REASON         JSON string or null
    _BM_SUCCESS               JSON bool  (true / false)
    _BM_ERROR                 JSON string or null
    _BM_COLD_START_MS         JSON number or null
    _BM_VISION_ENCODER_MS     JSON number or null
    _BM_PREFILL_MS            JSON number or null
    _BM_DECODE_MS             JSON number or null
    _BM_DECODE_TOKENS_PER_SEC JSON number or null
    _BM_LLM_GEN_GPU_MS        JSON number or null
    _BM_RESPONSE_PATH         plain string path
    _BM_PROFILE_PATH          plain string path
    _BM_MODEL_NAME            plain string
    _BM_ENGINE_PROVENANCE     JSON object
    _BM_ITERATION             JSON integer
    _BM_IS_WARMUP             JSON bool
    """
    if env is None:
        env = dict(os.environ)
    return json.dumps({
        "schema_version": "1",
        "record_type": "inference",
        "run_id": _js(env, "_BM_RUN_ID"),
        "recorded_at": _js(env, "_BM_RECORDED_AT"),
        "frame_condition": _js(env, "_BM_FRAME_CONDITION"),
        "frame_count": _jl(env, "_BM_FRAME_COUNT"),
        "path": "direct",
        "frame_paths": _jl(env, "_BM_FRAME_HASHES"),
        "prompt_hash": _js(env, "_BM_PROMPT_HASH"),
        "sequence_type": _jl(env, "_BM_SEQUENCE_TYPE"),
        "fps": _jl(env, "_BM_FPS"),
        "frame_timestamps_sec": _jl(env, "_BM_FRAME_TIMESTAMPS_SEC"),
        "frame_timestamp_policy": _jl(env, "_BM_FRAME_TIMESTAMP_POLICY"),
        "rendered_timestamps": _jl(env, "_BM_RENDERED_TIMESTAMPS"),
        "requested_sequence_type": _jl(env, "_BM_SEQUENCE_TYPE"),
        "runtime_temporal_encoding": _jl(env, "_BM_RUNTIME_TEMPORAL_ENCODING"),
        "temporal_fallback_used": _jl(env, "_BM_TEMPORAL_FALLBACK_USED"),
        "max_output_tokens": _jl(env, "_BM_MAX_OUTPUT_TOKENS"),
        "actual_output_tokens": _jl(env, "_BM_ACTUAL_OUTPUT_TOKENS"),
        "total_image_tokens": _jl(env, "_BM_TOTAL_IMAGE_TOKENS"),
        "finish_reason": _jl(env, "_BM_FINISH_REASON"),
        "success": _jl(env, "_BM_SUCCESS"),
        "error": _jl(env, "_BM_ERROR"),
        "cold_start_total_ms": _jl(env, "_BM_COLD_START_MS"),
        "total_latency_ms": None,
        "ttft_ms": None,
        "vision_encoder_ms": _jl(env, "_BM_VISION_ENCODER_MS"),
        "prefill_ms": _jl(env, "_BM_PREFILL_MS"),
        "decode_ms": _jl(env, "_BM_DECODE_MS"),
        "decode_tokens_per_sec": _jl(env, "_BM_DECODE_TOKENS_PER_SEC"),
        "llm_generation_total_gpu_time_ms": _jl(env, "_BM_LLM_GEN_GPU_MS"),
        "inference_seconds": None,
        "output_text": None,
        "output_words": None,
        "native_response_path": _js(env, "_BM_RESPONSE_PATH"),
        "native_profile_path": _js(env, "_BM_PROFILE_PATH"),
        "ipc_result_path": None,
        "model_name": _js(env, "_BM_MODEL_NAME"),
        "engine_provenance": _jl(env, "_BM_ENGINE_PROVENANCE"),
        "iteration": _jl(env, "_BM_ITERATION"),
        "warmup": _jl(env, "_BM_IS_WARMUP"),
    })


def build_ipc_record(env: dict[str, str] | None = None) -> str:
    """Build an IPC-path JSONL inference record from ``_BM_*`` environment variables.

    Environment variables consumed
    ──────────────────────────────
    _BM_RUN_ID             run timestamp string
    _BM_RECORDED_AT        ISO-8601 timestamp string
    _BM_FRAME_CONDITION    label e.g. "F1"
    _BM_FRAME_COUNT        JSON integer
    _BM_FRAME_HASHES       JSON array  [{path, sha256}, ...]
    _BM_PROMPT_HASH        12-hex string
    _BM_SEQUENCE_TYPE      JSON string
    _BM_FPS                JSON number or null
    _BM_FRAME_TIMESTAMPS_SEC JSON array or null
    _BM_FRAME_TIMESTAMP_POLICY JSON string
    _BM_RENDERED_TIMESTAMPS JSON bool
    _BM_REQUESTED_SEQUENCE_TYPE JSON string or null
    _BM_RUNTIME_TEMPORAL_ENCODING JSON string or null
    _BM_TEMPORAL_FALLBACK_USED JSON bool or null
    _BM_MAX_OUTPUT_TOKENS  JSON integer
    _BM_SUCCESS            JSON bool
    _BM_ERROR              JSON string or null
    _BM_TOTAL_LATENCY      JSON number or null  (outer client round-trip ms)
    _BM_INFERENCE_SECONDS  JSON number or null  (backend inference_seconds)
    _BM_OUTPUT_TEXT        JSON string or null
    _BM_OUTPUT_WORDS       JSON integer or null
    _BM_IPC_RESULT_PATH    JSON string or null  (path to result artifact)
    _BM_MODEL_NAME         plain string
    _BM_ENGINE_PROVENANCE  JSON object
    _BM_ITERATION          JSON integer
    _BM_IS_WARMUP          JSON bool
    """
    if env is None:
        env = dict(os.environ)
    return json.dumps({
        "schema_version": "1",
        "record_type": "inference",
        "run_id": _js(env, "_BM_RUN_ID"),
        "recorded_at": _js(env, "_BM_RECORDED_AT"),
        "frame_condition": _js(env, "_BM_FRAME_CONDITION"),
        "frame_count": _jl(env, "_BM_FRAME_COUNT"),
        "path": "ipc",
        "frame_paths": _jl(env, "_BM_FRAME_HASHES"),
        "prompt_hash": _js(env, "_BM_PROMPT_HASH"),
        "sequence_type": _jl(env, "_BM_SEQUENCE_TYPE"),
        "fps": _jl(env, "_BM_FPS"),
        "frame_timestamps_sec": _jl(env, "_BM_FRAME_TIMESTAMPS_SEC"),
        "frame_timestamp_policy": _jl(env, "_BM_FRAME_TIMESTAMP_POLICY"),
        "rendered_timestamps": _jl(env, "_BM_RENDERED_TIMESTAMPS"),
        "requested_sequence_type": _jl(env, "_BM_REQUESTED_SEQUENCE_TYPE"),
        "runtime_temporal_encoding": _jl(env, "_BM_RUNTIME_TEMPORAL_ENCODING"),
        "temporal_fallback_used": _jl(env, "_BM_TEMPORAL_FALLBACK_USED"),
        "max_output_tokens": _jl(env, "_BM_MAX_OUTPUT_TOKENS"),
        "actual_output_tokens": None,
        "total_image_tokens": None,
        "finish_reason": None,
        "success": _jl(env, "_BM_SUCCESS"),
        "error": _jl(env, "_BM_ERROR"),
        "cold_start_total_ms": None,
        "total_latency_ms": _jl(env, "_BM_TOTAL_LATENCY"),
        "ttft_ms": None,
        "vision_encoder_ms": None,
        "prefill_ms": None,
        "decode_ms": None,
        "decode_tokens_per_sec": None,
        "llm_generation_total_gpu_time_ms": None,
        "inference_seconds": _jl(env, "_BM_INFERENCE_SECONDS"),
        "output_text": _jl(env, "_BM_OUTPUT_TEXT"),
        "output_words": _jl(env, "_BM_OUTPUT_WORDS"),
        "native_response_path": None,
        "native_profile_path": None,
        "ipc_result_path": _jl(env, "_BM_IPC_RESULT_PATH"),
        "model_name": _js(env, "_BM_MODEL_NAME"),
        "engine_provenance": _jl(env, "_BM_ENGINE_PROVENANCE"),
        "iteration": _jl(env, "_BM_ITERATION"),
        "warmup": _jl(env, "_BM_IS_WARMUP"),
    })


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Parse multi-frame VLM inference JSONL records and generate "
            "frame-scaling report"
        )
    )
    parser.add_argument(
        "--input", required=True, type=Path,
        help="JSONL file of per-inference records from run_vlm_multiframe_benchmark.sh"
    )
    parser.add_argument(
        "--output", required=True, type=Path,
        help="Output JSON report path"
    )
    parser.add_argument(
        "--text", type=Path, default=None,
        help="Optional plain-text report output path (stdout when omitted)"
    )
    args = parser.parse_args()

    records = parse_jsonl(args.input)
    if not records:
        print(
            f"WARNING: no valid inference records found in {args.input}",
            file=sys.stderr,
        )

    report = build_report(records, source_path=args.input)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"JSON report: {args.output}", file=sys.stderr)

    text = format_text_report(report)
    if args.text:
        args.text.parent.mkdir(parents=True, exist_ok=True)
        args.text.write_text(text, encoding="utf-8")
        print(f"Text report: {args.text}", file=sys.stderr)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
