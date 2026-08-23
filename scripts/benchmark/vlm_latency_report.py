#!/usr/bin/env python3
"""
VLM latency characterization benchmark — parser and report generator.

Reads a JSONL file of per-inference records written by
run_vlm_latency_benchmark.sh and produces:

  - Machine-readable JSON summary (per-condition aggregates, raw records)
  - Human-readable text comparison report (latency vs output tokens,
    direct (cold-start, per-process) vs IPC (persistent server) side-by-side)

TTFT and stage timings are preserved as null when the runtime does not
expose them; they are never inferred or fabricated.

NOTE on lifecycle semantics
---------------------------
  direct : cold-start, per-process — includes engine/tokenizer init.
  ipc    : persistent server steady-state — connects to a running server.
  These two quantities are NOT comparable as "overhead".  The report
  presents them side-by-side for independent analysis only.

Usage
-----
  python3 vlm_latency_report.py \\
      --input vlm_latency_YYYYMMDD_HHMMSS.jsonl \\
      --output vlm_latency_report.json \\
      [--text vlm_latency_report.txt]

  # Print text report to stdout (omit --text):
  python3 vlm_latency_report.py --input results.jsonl --output report.json
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


# ── constants ─────────────────────────────────────────────────────────────────

SCHEMA_VERSION = "1"
RECORD_TYPE_INFERENCE = "inference"

# Experiment matrix conditions in display order
CONDITION_ORDER = ["A", "B", "C", "D", "E"]

# Prompt labels for display
_PROMPT_LABELS: dict[str, str] = {
    "terse_id": "terse identification",
    "compact_odd_json": "compact structured ODD JSON",
    "scene_description": "scene description (verbose)",
}


# ── statistics helpers ────────────────────────────────────────────────────────


def _percentile(values: list[float], pct: float) -> float | None:
    """Return the pct-th percentile (0–100) of a sorted list."""
    if not values:
        return None
    ordered = sorted(values)
    idx = max(0, int(math.ceil(pct / 100.0 * len(ordered))) - 1)
    idx = min(idx, len(ordered) - 1)
    return ordered[idx]


def _stats(values: list[float]) -> dict[str, float | None]:
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


# ── per-condition aggregation ─────────────────────────────────────────────────


def _decode_tps(rec: dict[str, Any]) -> float | None:
    """Compute decode tokens/sec if decode_ms and actual_output_tokens are available."""
    decode_ms = rec.get("decode_ms")
    tokens = rec.get("actual_output_tokens")
    if decode_ms is not None and tokens is not None and decode_ms > 0:
        return tokens / (decode_ms / 1000.0)
    return rec.get("decode_tokens_per_sec")


def aggregate_condition(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Aggregate inference records for a single (condition, path) group.

    Only non-warmup, successful records are included in timing aggregates.
    """
    measured = [r for r in records if not r.get("warmup", False) and r.get("success", False)]
    total_latency = [r["total_latency_ms"] for r in measured if r.get("total_latency_ms") is not None]
    cold_start_total = [r["cold_start_total_ms"] for r in measured if r.get("cold_start_total_ms") is not None]
    prefill = [r["prefill_ms"] for r in measured if r.get("prefill_ms") is not None]
    decode = [r["decode_ms"] for r in measured if r.get("decode_ms") is not None]
    vision_encoder = [r["vision_encoder_ms"] for r in measured if r.get("vision_encoder_ms") is not None]
    avg_token_ms = [r["average_time_per_token_ms"] for r in measured if r.get("average_time_per_token_ms") is not None]
    llm_gen_gpu = [r["llm_generation_total_gpu_time_ms"] for r in measured if r.get("llm_generation_total_gpu_time_ms") is not None]
    output_tokens = [r["actual_output_tokens"] for r in measured if r.get("actual_output_tokens") is not None]
    decode_tps_vals = [v for r in measured if (v := _decode_tps(r)) is not None]

    failed = sum(1 for r in records if not r.get("warmup", False) and not r.get("success", False))
    warmup_count = sum(1 for r in records if r.get("warmup", False))

    # Determine whether stage timings were available at all
    prefill_available = any(r.get("prefill_ms") is not None for r in measured)
    decode_available = any(r.get("decode_ms") is not None for r in measured)
    vision_encoder_available = any(r.get("vision_encoder_ms") is not None for r in measured)
    avg_token_available = any(r.get("average_time_per_token_ms") is not None for r in measured)
    llm_gen_gpu_available = any(r.get("llm_generation_total_gpu_time_ms") is not None for r in measured)

    # finish_reason breakdown — count by value; null when not reported.
    finish_reason_counts: dict[str, int] = {}
    for r in measured:
        fr = r.get("finish_reason")
        key = str(fr) if fr is not None else "null"
        finish_reason_counts[key] = finish_reason_counts.get(key, 0) + 1
    n_max_length = finish_reason_counts.get("max-length", 0) + finish_reason_counts.get("max_length", 0)

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
        "average_time_per_token_ms": _stats(avg_token_ms) if avg_token_available else {"available": False},
        "llm_generation_total_gpu_time_ms": _stats(llm_gen_gpu) if llm_gen_gpu_available else {"available": False},
        "actual_output_tokens": _stats(output_tokens) if output_tokens else {"available": False},
        "decode_tokens_per_sec": _stats(decode_tps_vals) if decode_tps_vals else {"available": False},
        "stage_timings_available": {
            "prefill": prefill_available,
            "decode": decode_available,
            "vision_encoder": vision_encoder_available,
            "average_time_per_token": avg_token_available,
            "llm_generation_total_gpu_time": llm_gen_gpu_available,
            "actual_output_tokens": bool(output_tokens),
            "decode_tokens_per_sec": bool(decode_tps_vals),
        },
        "finish_reason_counts": finish_reason_counts,
        "n_max_length": n_max_length,
    }


# ── direct-vs-IPC comparison ──────────────────────────────────────────────────


def compute_direct_ipc_comparison(
    by_condition_path: dict[tuple[str, str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """
    For each condition where both 'direct' and 'ipc' records exist, tabulate
    the mean total latency of each path side by side.

    NOTE: 'direct' measures cold-start latency (fresh process per inference,
    including engine/tokenizer initialisation).  'ipc' measures steady-state
    latency against an already-running, warmed edge_vlm_server.  These two
    quantities are *not* comparable as overhead; the table is provided for
    independent per-path analysis only.
    """
    rows: list[dict[str, Any]] = []
    conditions = sorted({c for c, _ in by_condition_path})
    for condition in conditions:
        direct_records = by_condition_path.get((condition, "direct"), [])
        ipc_records = by_condition_path.get((condition, "ipc"), [])
        if not direct_records or not ipc_records:
            continue
        direct_agg = aggregate_condition(direct_records)
        ipc_agg = aggregate_condition(ipc_records)
        # direct path: total_latency_ms is null (cold_start includes process init);
        # use cold_start_total_ms as the direct lifecycle metric.
        direct_mean = direct_agg["cold_start_total_ms"].get("mean")
        ipc_mean = ipc_agg["total_latency_ms"].get("mean")
        rows.append(
            {
                "condition": condition,
                "direct_cold_start_total_ms_mean": direct_mean,
                "ipc_total_latency_ms_mean": ipc_mean,
                "note": (
                    "direct=cold_start (per-process init included); "
                    "ipc=persistent server steady-state; not directly comparable"
                ),
            }
        )
    return rows


# ── token-scaling table ───────────────────────────────────────────────────────


def compute_token_scaling(
    by_condition_path: dict[tuple[str, str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """
    Build a table of mean steady-state inference latency vs max_output_tokens
    for IPC-path rows (which populate ``total_latency_ms``).

    Direct-path rows have ``total_latency_ms=null`` because their wall-clock
    measure includes process/engine initialisation; those rows are excluded
    here and reported separately in the cold-start scaling table.  A direct
    row is included only when the pinned runtime profile exposes an
    authoritative inference total (i.e. ``total_latency_ms`` is not null).
    """
    rows: list[dict[str, Any]] = []
    for (condition, path), records in sorted(by_condition_path.items()):
        measured = [
            r
            for r in records
            if not r.get("warmup", False) and r.get("success", False)
        ]
        if not measured:
            continue
        max_tokens = measured[0].get("max_output_tokens")
        prompt_id = measured[0].get("prompt_id", "")
        latencies = [r["total_latency_ms"] for r in measured if r.get("total_latency_ms") is not None]
        if not latencies:
            # No runtime inference total available for this path (expected for
            # direct/cold-start rows) — omit from this table.
            continue
        out_tokens = [r["actual_output_tokens"] for r in measured if r.get("actual_output_tokens") is not None]
        rows.append(
            {
                "condition": condition,
                "path": path,
                "prompt_id": prompt_id,
                "max_output_tokens": max_tokens,
                "actual_output_tokens_mean": fmean(out_tokens) if out_tokens else None,
                "total_latency_ms_mean": fmean(latencies) if latencies else None,
                "total_latency_ms_p95": _percentile(latencies, 95),
                "n_measured": len(measured),
            }
        )
    return rows


def compute_cold_start_scaling(
    by_condition_path: dict[tuple[str, str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """
    Build a table of mean cold-start wall time (``cold_start_total_ms``) vs
    max_output_tokens for direct-path rows only.

    ``cold_start_total_ms`` includes per-process engine/tokenizer initialisation
    and is the correct lifecycle metric for the direct path.  It is kept
    separate from ``total_latency_ms`` (steady-state model/runtime inference)
    so the two are never confused.
    """
    rows: list[dict[str, Any]] = []
    for (condition, path), records in sorted(by_condition_path.items()):
        if path != "direct":
            continue
        measured = [
            r
            for r in records
            if not r.get("warmup", False) and r.get("success", False)
        ]
        if not measured:
            continue
        max_tokens = measured[0].get("max_output_tokens")
        prompt_id = measured[0].get("prompt_id", "")
        cold_start_vals = [r["cold_start_total_ms"] for r in measured if r.get("cold_start_total_ms") is not None]
        if not cold_start_vals:
            continue
        out_tokens = [r["actual_output_tokens"] for r in measured if r.get("actual_output_tokens") is not None]
        rows.append(
            {
                "condition": condition,
                "path": path,
                "prompt_id": prompt_id,
                "max_output_tokens": max_tokens,
                "actual_output_tokens_mean": fmean(out_tokens) if out_tokens else None,
                "cold_start_total_ms_mean": fmean(cold_start_vals) if cold_start_vals else None,
                "cold_start_total_ms_p95": _percentile(cold_start_vals, 95),
                "n_measured": len(measured),
            }
        )
    return rows


# ── native profile detail table ───────────────────────────────────────────────


def compute_native_profile_table(
    by_condition_path: dict[tuple[str, str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """
    Build a per-inference row for every measured, successful direct-path record.

    These rows expose the NVIDIA-authoritative profile metrics:
    actual_output_tokens, vision_encoder_ms, prefill_ms, decode_tokens_per_sec,
    llm_generation_total_gpu_time_ms, and finish_reason.

    Null values mean the field was not exposed by the runtime — they are never
    inferred or fabricated.  Per-image EOS vs max-length breakdown and
    generated-token counts are essential for interpreting the latency sweep.
    """
    rows: list[dict[str, Any]] = []
    for (condition, path), records in sorted(by_condition_path.items()):
        if path != "direct":
            continue
        for rec in records:
            if rec.get("warmup", False) or not rec.get("success", False):
                continue
            rows.append(
                {
                    "condition": condition,
                    "image_id": rec.get("image_id"),
                    "iteration": rec.get("iteration"),
                    "actual_output_tokens": rec.get("actual_output_tokens"),
                    "vision_encoder_ms": rec.get("vision_encoder_ms"),
                    "prefill_ms": rec.get("prefill_ms"),
                    "decode_tokens_per_sec": rec.get("decode_tokens_per_sec"),
                    "llm_generation_total_gpu_time_ms": rec.get("llm_generation_total_gpu_time_ms"),
                    "finish_reason": rec.get("finish_reason"),
                }
            )
    return rows


# ── full report generation ────────────────────────────────────────────────────


def build_report(
    records: list[dict[str, Any]],
    *,
    source_path: Path | None = None,
) -> dict[str, Any]:
    """
    Build a machine-readable JSON report from raw inference records.

    Raw per-inference records are preserved in the report so later analysis
    is not limited to averages.
    """
    by_condition_path: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for rec in records:
        key = (rec.get("condition", "?"), rec.get("path", "?"))
        by_condition_path.setdefault(key, []).append(rec)

    conditions_summary: dict[str, Any] = {}
    for (condition, path), group in sorted(by_condition_path.items()):
        conditions_summary.setdefault(condition, {})[path] = aggregate_condition(group)

    run_ids = list({r.get("run_id", "") for r in records if r.get("run_id")})
    model_names = list({r.get("model_name", "") for r in records if r.get("model_name")})

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
        "conditions": conditions_summary,
        "token_scaling_table": compute_token_scaling(by_condition_path),
        "cold_start_scaling_table": compute_cold_start_scaling(by_condition_path),
        "native_profile_table": compute_native_profile_table(by_condition_path),
        "direct_ipc_comparison": compute_direct_ipc_comparison(by_condition_path),
        "raw_records": records,
    }


# ── text report formatting ────────────────────────────────────────────────────


def _fmt_ms(val: float | None) -> str:
    if val is None:
        return "    n/a"
    return f"{val:7.1f}"


def _fmt_pct(val: float | None) -> str:
    if val is None:
        return "   n/a"
    return f"{val:+6.1f}%"


def format_text_report(report: dict[str, Any]) -> str:
    lines: list[str] = [
        "=" * 72,
        "  VLM Latency Characterization Benchmark Report",
        f"  Generated: {report.get('generated_at', 'unknown')}",
        "=" * 72,
    ]

    model_names = report.get("model_names") or []
    run_ids = report.get("run_ids") or []
    if model_names:
        lines += ["", f"  Model(s): {', '.join(model_names)}"]
    if run_ids:
        lines += [f"  Run ID(s): {', '.join(run_ids)}"]
    lines += [
        f"  Total records: {report.get('n_total_records', 'n/a')}",
        f"  Measured (non-warmup, success): {report.get('n_measured_records', 'n/a')}",
    ]

    # ── Native NVIDIA profile detail (direct path, per-inference) ────────
    native_profile = report.get("native_profile_table") or []
    if native_profile:
        lines += [
            "",
            "Native NVIDIA Profile Detail  (direct path, measured inferences only)",
            "  NOTE: null = field not exposed by the runtime; never inferred.",
            "-" * 72,
            f"  {'Cond':<5} {'Image':<12} {'Iter':>4} {'GenTok':>6} "
            f"{'Vision ms':>9} {'Prefill ms':>10} {'Gen tok/s':>9} {'GPU ms':>8} {'FinishReason':<14}",
            "  " + "-" * 68,
        ]
        def _fmt_f(v: float | None, w: int = 8) -> str:
            return f"{v:{w}.1f}" if v is not None else " " * (w - 3) + "n/a"
        for row in native_profile:
            lines.append(
                f"  {row.get('condition', '?'):<5} "
                f"{str(row.get('image_id', '?')):<12} "
                f"{row.get('iteration', '?'):>4} "
                f"{str(row.get('actual_output_tokens') or 'n/a'):>6} "
                f"{_fmt_f(row.get('vision_encoder_ms'), 9):>9} "
                f"{_fmt_f(row.get('prefill_ms'), 10):>10} "
                f"{_fmt_f(row.get('decode_tokens_per_sec'), 9):>9} "
                f"{_fmt_f(row.get('llm_generation_total_gpu_time_ms'), 8):>8} "
                f"{str(row.get('finish_reason') or 'n/a'):<14}"
            )

    # ── Token scaling table (IPC steady-state inference latency) ─────────
    scaling = report.get("token_scaling_table") or []
    if scaling:
        lines += [
            "",
            "Inference Latency vs Output-Token Cap  (IPC path, persistent server)",
            "  NOTE: direct-path rows are excluded here; see Cold-Start table below.",
            "-" * 72,
            f"  {'Cond':<5} {'Path':<8} {'Prompt style':<28} {'MaxTok':>6} "
            f"{'AvgActual':>9} {'Mean ms':>8} {'P95 ms':>7} {'N':>4}",
            "  " + "-" * 68,
        ]
        for row in scaling:
            prompt_label = _PROMPT_LABELS.get(row.get("prompt_id", ""), row.get("prompt_id", ""))
            lines.append(
                f"  {row.get('condition', '?'):<5} {row.get('path', '?'):<8} "
                f"{prompt_label:<28} {row.get('max_output_tokens') or '?':>6} "
                f"{row.get('actual_output_tokens_mean') or 'n/a':>9} "
                f"{_fmt_ms(row.get('total_latency_ms_mean')):>8} "
                f"{_fmt_ms(row.get('total_latency_ms_p95')):>7} "
                f"{row.get('n_measured', '?'):>4}"
            )

    # ── Cold-start scaling table (direct path, per-process wall time) ─────
    cold_start_scaling = report.get("cold_start_scaling_table") or []
    if cold_start_scaling:
        lines += [
            "",
            "Cold-Start Wall Time vs Output-Token Cap  (direct path, per-process)",
            "  NOTE: includes process/engine/tokenizer initialisation — not model inference latency.",
            "-" * 72,
            f"  {'Cond':<5} {'Path':<8} {'Prompt style':<28} {'MaxTok':>6} "
            f"{'AvgActual':>9} {'Mean ms':>8} {'P95 ms':>7} {'N':>4}",
            "  " + "-" * 68,
        ]
        for row in cold_start_scaling:
            prompt_label = _PROMPT_LABELS.get(row.get("prompt_id", ""), row.get("prompt_id", ""))
            lines.append(
                f"  {row.get('condition', '?'):<5} {row.get('path', '?'):<8} "
                f"{prompt_label:<28} {row.get('max_output_tokens') or '?':>6} "
                f"{row.get('actual_output_tokens_mean') or 'n/a':>9} "
                f"{_fmt_ms(row.get('cold_start_total_ms_mean')):>8} "
                f"{_fmt_ms(row.get('cold_start_total_ms_p95')):>7} "
                f"{row.get('n_measured', '?'):>4}"
            )

    # ── Direct vs IPC comparison ──────────────────────────────────────────
    comparisons = report.get("direct_ipc_comparison") or []
    if comparisons:
        lines += [
            "",
            "Direct (cold-start, per-process) vs IPC (persistent server, steady-state)",
            "-" * 72,
            "  NOTE: direct includes process/engine init; ipc connects to a running server.",
            "  These values reflect different lifecycle phases and are not comparable as overhead.",
            "-" * 72,
            f"  {'Cond':<5} {'Direct cold-start ms':>20} {'IPC steady-state ms':>19}",
            "  " + "-" * 44,
        ]
        for row in comparisons:
            lines.append(
                f"  {row.get('condition', '?'):<5} "
                f"{_fmt_ms(row.get('direct_cold_start_total_ms_mean')):>20} "
                f"{_fmt_ms(row.get('ipc_total_latency_ms_mean')):>19}"
            )

    # ── finish_reason / truncation summary ───────────────────────────────────
    conditions_summary = report.get("conditions") or {}
    has_finish_reason = False
    for cond_data in conditions_summary.values():
        for path_data in cond_data.values():
            if path_data.get("finish_reason_counts"):
                has_finish_reason = True
                break

    if has_finish_reason:
        lines += [
            "",
            "Response Finish-Reason Summary  (measured, non-warmup requests only)",
            "-" * 72,
            f"  {'Cond':<5} {'Path':<8} {'Total':>5} {'max-length':>10} {'frac':>6}  reasons",
            "  " + "-" * 68,
        ]
        for cond in CONDITION_ORDER:
            if cond not in conditions_summary:
                continue
            for path in ("direct", "ipc"):
                path_data = conditions_summary[cond].get(path)
                if not path_data:
                    continue
                n_measured = path_data.get("n_measured", 0)
                n_max_len = path_data.get("n_max_length", 0)
                fr_counts = path_data.get("finish_reason_counts") or {}
                frac = f"{n_max_len / n_measured:.2f}" if n_measured > 0 else "n/a"
                reasons = ", ".join(f"{k}:{v}" for k, v in sorted(fr_counts.items()))
                lines.append(
                    f"  {cond:<5} {path:<8} {n_measured:>5} {n_max_len:>10} {frac:>6}  {reasons}"
                )
        if any(
            path_data.get("n_max_length", 0) > 0
            for cond_data in conditions_summary.values()
            for path_data in cond_data.values()
        ):
            lines += [
                "  NOTE: requests with finish_reason='max-length' were capped at",
                "        max_output_tokens; they do not represent natural task completion.",
            ]

    # ── Stage-timing availability note (scoped by path) ──────────────────
    # NVIDIA-emitted profile stages (prefill, decode, vision_encoder) are
    # expected only on the direct/native path.  IPC correctly has these null;
    # mixing paths would produce a misleading global "unavailable" conclusion.
    stage_avail_by_path: dict[str, dict[str, bool]] = {}
    for cond_data in conditions_summary.values():
        for path_label, path_data in cond_data.items():
            avail = path_data.get("stage_timings_available") or {}
            entry = stage_avail_by_path.setdefault(path_label, {})
            for stage_key, stage_label in (
                ("prefill", "prefill time"),
                ("decode", "decode time"),
                ("vision_encoder", "vision encoder time"),
            ):
                present = avail.get(stage_key, False)
                # Once any condition has it available, mark path as available.
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
            "-" * 72,
        ]
        lines += avail_lines

    lines += ["", "=" * 72, ""]
    return "\n".join(lines)


# ── prompt utilities (used by benchmark runner construction helpers) ──────────


# Experiment matrix: (condition, prompt_id, max_output_tokens)
EXPERIMENT_MATRIX: list[tuple[str, str, int]] = [
    ("A", "terse_id", 16),
    ("B", "compact_odd_json", 32),
    ("C", "compact_odd_json", 64),
    ("D", "scene_description", 128),
    ("E", "scene_description", 256),
]

PROMPT_TEXTS: dict[str, str] = {
    "terse_id": "What is in this image?",
    "compact_odd_json": (
        "Describe this scene as compact JSON with keys: "
        "objects, actions, hazards, navigable. "
        "Be concise."
    ),
    "scene_description": (
        "You are an autonomous robot perception system. "
        "Provide a detailed description of the scene including all visible objects, "
        "their positions relative to the robot, any dynamic elements, "
        "potential obstacles, and the overall environment type. "
        "Be thorough and precise."
    ),
}


def prompt_hash(prompt_text: str) -> str:
    """Return a short SHA-256 hex prefix of the prompt text."""
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:12]


def build_condition_spec(condition: str) -> dict[str, Any] | None:
    """Return the condition specification for a given condition label, or None."""
    for cond, prompt_id, max_tokens in EXPERIMENT_MATRIX:
        if cond == condition:
            text = PROMPT_TEXTS.get(prompt_id, "")
            return {
                "condition": cond,
                "prompt_id": prompt_id,
                "max_output_tokens": max_tokens,
                "prompt_text": text,
                "prompt_hash": prompt_hash(text),
            }
    return None


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse VLM latency JSONL records and generate comparison report"
    )
    parser.add_argument(
        "--input", required=True, type=Path,
        help="JSONL file of per-inference records from run_vlm_latency_benchmark.sh"
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
        print(f"WARNING: no valid inference records found in {args.input}", file=sys.stderr)

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
