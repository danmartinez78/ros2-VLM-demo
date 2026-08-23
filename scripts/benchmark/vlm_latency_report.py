#!/usr/bin/env python3
"""
VLM latency characterization benchmark — parser and report generator.

Reads a JSONL file of per-inference records written by
run_vlm_latency_benchmark.sh and produces:

  - Machine-readable JSON summary (per-condition aggregates, raw records)
  - Human-readable text comparison report (latency vs output tokens,
    direct-vs-ROS overhead for equivalent conditions)

TTFT and stage timings are preserved as null when the runtime does not
expose them; they are never inferred or fabricated.

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
    ttft = [r["ttft_ms"] for r in measured if r.get("ttft_ms") is not None]
    decode = [r["decode_ms"] for r in measured if r.get("decode_ms") is not None]
    visual = [r["visual_preprocess_ms"] for r in measured if r.get("visual_preprocess_ms") is not None]
    output_tokens = [r["actual_output_tokens"] for r in measured if r.get("actual_output_tokens") is not None]
    decode_tps_vals = [v for r in measured if (v := _decode_tps(r)) is not None]

    failed = sum(1 for r in records if not r.get("warmup", False) and not r.get("success", False))
    warmup_count = sum(1 for r in records if r.get("warmup", False))

    # Determine whether stage timings were available at all
    ttft_available = any(r.get("ttft_ms") is not None for r in measured)
    decode_available = any(r.get("decode_ms") is not None for r in measured)
    visual_available = any(r.get("visual_preprocess_ms") is not None for r in measured)

    return {
        "n_total": len(records),
        "n_warmup": warmup_count,
        "n_measured": len(measured),
        "n_failed": failed,
        "total_latency_ms": _stats(total_latency),
        "ttft_ms": _stats(ttft) if ttft_available else {"available": False},
        "decode_ms": _stats(decode) if decode_available else {"available": False},
        "visual_preprocess_ms": _stats(visual) if visual_available else {"available": False},
        "actual_output_tokens": _stats(output_tokens) if output_tokens else {"available": False},
        "decode_tokens_per_sec": _stats(decode_tps_vals) if decode_tps_vals else {"available": False},
        "stage_timings_available": {
            "ttft": ttft_available,
            "decode": decode_available,
            "visual_preprocess": visual_available,
            "actual_output_tokens": bool(output_tokens),
            "decode_tokens_per_sec": bool(decode_tps_vals),
        },
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
        direct_mean = direct_agg["total_latency_ms"].get("mean")
        ipc_mean = ipc_agg["total_latency_ms"].get("mean")
        rows.append(
            {
                "condition": condition,
                "direct_total_latency_ms_mean": direct_mean,
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
    Build a table of mean total latency vs max_output_tokens, per path, for all
    conditions.  Makes scaling obvious.
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

    # ── Token scaling table ───────────────────────────────────────────────
    scaling = report.get("token_scaling_table") or []
    if scaling:
        lines += [
            "",
            "Latency vs Output-Token Cap  (total end-to-end wall-clock time)",
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
            f"  {'Cond':<5} {'Direct mean ms':>14} {'IPC mean ms':>11}",
            "  " + "-" * 34,
        ]
        for row in comparisons:
            lines.append(
                f"  {row.get('condition', '?'):<5} "
                f"{_fmt_ms(row.get('direct_total_latency_ms_mean')):>14} "
                f"{_fmt_ms(row.get('ipc_total_latency_ms_mean')):>11}"
            )

    # ── Stage-timing availability note ───────────────────────────────────
    conditions_summary = report.get("conditions") or {}
    unavailable_stages: list[str] = []
    for cond_data in conditions_summary.values():
        for path_data in cond_data.values():
            avail = path_data.get("stage_timings_available") or {}
            if avail.get("ttft") is False:
                unavailable_stages.append("TTFT / prefill")
            if avail.get("decode") is False:
                unavailable_stages.append("decode time")
            if avail.get("visual_preprocess") is False:
                unavailable_stages.append("visual preprocessing time")
    if unavailable_stages:
        lines += [
            "",
            "Stage Timing Availability",
            "-" * 72,
        ]
        for stage in sorted(set(unavailable_stages)):
            lines.append(f"  {stage}: NOT available from runtime — reported as null, not inferred")

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
