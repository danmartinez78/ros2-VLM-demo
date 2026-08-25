#!/usr/bin/env python3
"""Generic multi-frame VLM benchmark parser and report generator."""
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
FRAME_COUNT_ORDER = ["F1", "F2", "F4", "F8"]
MAX_OUTPUT_TOKENS = 32
MULTIFRAME_PROMPT_TEXT = (
    "Analyze this ordered temporal image sequence and describe the scene as compact JSON "
    "with keys: objects, actions, hazards, navigable. Be concise and provide one result "
    "for the full sequence."
)
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


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = max(0, int(math.ceil(pct / 100.0 * len(ordered))) - 1)
    return ordered[min(idx, len(ordered) - 1)]


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
                continue
            records.append(obj)
    return records


def select_frames(all_frames: list[str], frame_count: int) -> list[str]:
    n = len(all_frames)
    if n < frame_count:
        raise ValueError(f"Insufficient frames: need {frame_count}, have {n}")
    if frame_count == 1:
        return [all_frames[0]]
    if frame_count == n:
        return list(all_frames)
    step = (n - 1) / (frame_count - 1)
    return [all_frames[round(i * step)] for i in range(frame_count)]


def frame_condition_label(frame_count: int) -> str:
    return f"F{frame_count}"


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def file_sha256_prefix(path: str | Path) -> str:
    return file_sha256(path)[:12]


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def build_multiframe_request(
    frame_paths: list[str],
    prompt_text: str,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
) -> dict[str, Any]:
    del max_output_tokens  # runtime receives this through its generation argument
    content = [{"type": "image", "image": path} for path in frame_paths]
    content.append({"type": "text", "text": prompt_text})
    return {"requests": [{"messages": [{"role": "user", "content": content}]}]}


def build_multiframe_request_metadata(
    frame_paths: list[str],
    prompt_text: str,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
) -> dict[str, Any]:
    return {
        "frame_count": len(frame_paths),
        "frame_paths": list(frame_paths),
        "images": [{"path": path, "index": i} for i, path in enumerate(frame_paths)],
        "prompt_text": prompt_text,
        "prompt_hash": prompt_hash(prompt_text),
        "max_output_tokens": max_output_tokens,
    }


def _decode_tps(rec: dict[str, Any]) -> float | None:
    if rec.get("decode_ms") and rec.get("actual_output_tokens") is not None:
        return float(rec["actual_output_tokens"]) / (float(rec["decode_ms"]) / 1000.0)
    value = rec.get("decode_tokens_per_sec")
    return float(value) if value is not None else None


def aggregate_frame_condition(records: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [r for r in records if not r.get("warmup", False) and r.get("success", False)]

    def vals(key: str) -> list[float]:
        return [float(r[key]) for r in measured if r.get(key) is not None]

    finish: dict[str, int] = {}
    for rec in measured:
        key = str(rec.get("finish_reason")) if rec.get("finish_reason") is not None else "null"
        finish[key] = finish.get(key, 0) + 1

    result = {
        "n_total": len(records),
        "n_warmup": sum(1 for r in records if r.get("warmup", False)),
        "n_measured": len(measured),
        "n_failed": sum(1 for r in records if not r.get("warmup", False) and not r.get("success", False)),
        "total_latency_ms": _stats(vals("total_latency_ms")),
        "cold_start_total_ms": _stats(vals("cold_start_total_ms")),
        "finish_reason_counts": finish,
        "n_max_length": finish.get("max-length", 0) + finish.get("max_length", 0),
    }
    for key in (
        "prefill_ms",
        "decode_ms",
        "vision_encoder_ms",
        "llm_generation_total_gpu_time_ms",
        "actual_output_tokens",
        "total_image_tokens",
    ):
        values = vals(key)
        result[key] = _stats(values) if values else {"available": False}
    tps = [v for r in measured if (v := _decode_tps(r)) is not None]
    result["decode_tokens_per_sec"] = _stats(tps) if tps else {"available": False}
    return result


def _provenance(record: dict[str, Any]) -> dict[str, Any]:
    source = record.get("engine_provenance") or {}
    result = {key: source.get(key, record.get(key)) for key in ENGINE_PROVENANCE_KEYS}
    if source.get("provenance_warnings"):
        result["provenance_warnings"] = list(source["provenance_warnings"])
    return result


def build_direct_record(
    *,
    run_id: str,
    frame_condition: str,
    frame_paths: list[dict[str, Any]],
    prompt_hash_value: str,
    max_output_tokens: int,
    iteration: int,
    warmup: bool,
    success: bool,
    error: str | None = None,
    cold_start_total_ms: float | None = None,
    total_latency_ms: float | None = None,
    actual_output_tokens: int | None = None,
    total_image_tokens: int | None = None,
    finish_reason: str | None = None,
    output_text: str | None = None,
    native_response_path: str | None = None,
    native_profile_path: str | None = None,
    model_name: str | None = None,
    engine_provenance: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    record = {
        "schema_version": SCHEMA_VERSION,
        "record_type": RECORD_TYPE_INFERENCE,
        "run_id": run_id,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "frame_condition": frame_condition,
        "frame_count": len(frame_paths),
        "path": "direct",
        "frame_paths": frame_paths,
        "prompt_hash": prompt_hash_value,
        "max_output_tokens": max_output_tokens,
        "actual_output_tokens": actual_output_tokens,
        "total_image_tokens": total_image_tokens,
        "finish_reason": finish_reason,
        "output_text": output_text,
        "success": success,
        "error": error,
        "cold_start_total_ms": cold_start_total_ms,
        "total_latency_ms": total_latency_ms,
        "native_response_path": native_response_path,
        "native_profile_path": native_profile_path,
        "ipc_result_path": None,
        "model_name": model_name,
        "engine_provenance": engine_provenance or {},
        "iteration": iteration,
        "warmup": warmup,
    }
    record.update(extra)
    return record


def build_ipc_record(
    *,
    run_id: str,
    frame_condition: str,
    frame_paths: list[dict[str, Any]],
    prompt_hash_value: str,
    max_output_tokens: int,
    iteration: int,
    warmup: bool,
    success: bool,
    error: str | None = None,
    total_latency_ms: float | None = None,
    inference_seconds: float | None = None,
    output_text: str | None = None,
    output_words: int | None = None,
    ipc_result_path: str | None = None,
    model_name: str | None = None,
    engine_provenance: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    record = build_direct_record(
        run_id=run_id,
        frame_condition=frame_condition,
        frame_paths=frame_paths,
        prompt_hash_value=prompt_hash_value,
        max_output_tokens=max_output_tokens,
        iteration=iteration,
        warmup=warmup,
        success=success,
        error=error,
        total_latency_ms=total_latency_ms,
        output_text=output_text,
        model_name=model_name,
        engine_provenance=engine_provenance,
        **extra,
    )
    record.update({
        "path": "ipc",
        "cold_start_total_ms": None,
        "inference_seconds": inference_seconds,
        "output_words": output_words,
        "native_response_path": None,
        "native_profile_path": None,
        "ipc_result_path": ipc_result_path,
    })
    return record


def _group(records: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for rec in records:
        grouped.setdefault((str(rec.get("frame_condition", "")), str(rec.get("path", ""))), []).append(rec)
    return grouped


def compute_frame_scaling_table(by_condition_path: dict[tuple[str, str], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for (condition, path), records in sorted(by_condition_path.items()):
        measured = [r for r in records if not r.get("warmup", False) and r.get("success", False)]
        if not measured:
            continue
        def vals(key: str) -> list[float]:
            return [float(r[key]) for r in measured if r.get(key) is not None]
        ipc = vals("total_latency_ms") if path == "ipc" else []
        rows.append({
            "frame_condition": condition,
            "path": path,
            "frame_count": measured[0].get("frame_count"),
            "images_per_request": measured[0].get("frame_count"),
            "visual_tokens_mean": fmean(vals("total_image_tokens")) if vals("total_image_tokens") else None,
            "vision_encoder_ms_mean": fmean(vals("vision_encoder_ms")) if vals("vision_encoder_ms") else None,
            "prefill_ms_mean": fmean(vals("prefill_ms")) if vals("prefill_ms") else None,
            "actual_output_tokens_mean": fmean(vals("actual_output_tokens")) if vals("actual_output_tokens") else None,
            "decode_tokens_per_sec_mean": fmean([v for r in measured if (v := _decode_tps(r)) is not None]) if any(_decode_tps(r) is not None for r in measured) else None,
            "llm_generation_total_gpu_time_ms_mean": fmean(vals("llm_generation_total_gpu_time_ms")) if vals("llm_generation_total_gpu_time_ms") else None,
            "ipc_latency_ms_mean": fmean(ipc) if ipc else None,
            "ipc_latency_ms_p95": _percentile(ipc, 95) if ipc else None,
            "finish_reason_counts": aggregate_frame_condition(records)["finish_reason_counts"],
        })
    return rows


def compute_ipc_artifact_table(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "frame_condition": rec.get("frame_condition"),
            "iteration": rec.get("iteration"),
            "ipc_result_path": rec.get("ipc_result_path"),
            "runtime_temporal_encoding": rec.get("runtime_temporal_encoding"),
            "requested_sequence_type": rec.get("requested_sequence_type"),
            "temporal_fallback_used": rec.get("temporal_fallback_used"),
        }
        for rec in records
        if rec.get("path") == "ipc" and rec.get("ipc_result_path")
    ]


def build_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped = _group(records)
    provenances = [_provenance(r) for r in records if r.get("engine_provenance") or r.get("model_name")]
    unique_identities = sorted({p.get("engine_identity") for p in provenances if p.get("engine_identity")})
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "prompt": {"text": MULTIFRAME_PROMPT_TEXT, "hash": prompt_hash(MULTIFRAME_PROMPT_TEXT)},
        "aggregates": {f"{c}:{p}": aggregate_frame_condition(rs) for (c, p), rs in sorted(grouped.items())},
        "frame_scaling": compute_frame_scaling_table(grouped),
        "ipc_artifacts": compute_ipc_artifact_table(records),
        "engine_identities": unique_identities,
        "mixed_engine_provenance": len(unique_identities) > 1,
        "raw_records": records,
    }


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.1f}"


def format_text_report(report: dict[str, Any]) -> str:
    lines = [
        "VLM Multi-Frame Latency Characterization Report",
        "=" * 48,
        "",
        "Frame Scaling",
        "-------------",
    ]
    for row in report.get("frame_scaling", []):
        lines.append(
            f"{row['frame_condition']} {row['path']}: frames={row['frame_count']} "
            f"vision={_fmt(row['vision_encoder_ms_mean'])} ms prefill={_fmt(row['prefill_ms_mean'])} ms "
            f"ipc_mean={_fmt(row['ipc_latency_ms_mean'])} ms"
        )
    if report.get("mixed_engine_provenance"):
        lines += ["", "WARNING: mixed engine provenance detected; cross-path comparisons are not controlled."]
    if report.get("ipc_artifacts"):
        lines += ["", "IPC Artifact Provenance", "-----------------------"]
        for row in report["ipc_artifacts"]:
            lines.append(f"{row['frame_condition']} iter={row['iteration']}: {row['ipc_result_path']}")
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
