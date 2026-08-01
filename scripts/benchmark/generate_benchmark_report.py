#!/usr/bin/env python3
"""
Generate a comparison report separating native engine time from ROS overhead.

Takes:
  - A ROS metrics JSON (from collect_ros_metrics.py)
  - Optionally: a native benchmark artifacts directory (from run_native_benchmarks.sh)

Outputs a human-readable text + machine-readable JSON report that clearly shows
what fraction of end-to-end latency is from the TensorRT engine vs ROS overhead.

Usage
-----
  python3 generate_benchmark_report.py \\
      --ros-report ros_report.json \\
      [--native-dir /path/to/native_artifacts_YYYYMMDD_HHMMSS] \\
      --output comparison_report.json \\
      [--text comparison_report.txt]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ── native artifact parsing ───────────────────────────────────────────────────


def _load_native_manifest(native_dir: Path) -> dict[str, Any]:
    manifest_path = native_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    with manifest_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _extract_llm_bench_summary(bench_file: Path) -> dict[str, Any] | None:
    """
    Extract key metrics from an llm_bench JSON output file.

    llm_bench writes structured JSON when --outputFormat json is used.
    If the file is not JSON (plain text output), returns None.
    """
    if not bench_file.exists():
        return None
    try:
        with bench_file.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data
    except (json.JSONDecodeError, OSError):
        # Plain text output — preserve as-is
        try:
            return {"raw_text": bench_file.read_text(encoding="utf-8", errors="replace")}
        except OSError:
            return None


def _extract_profile_summary(profile_file: Path) -> dict[str, Any] | None:
    """
    Extract top-level timings from an llm_inference --dumpProfile output file.

    The profile JSON structure is not reimplemented here — the full artifact
    is referenced and a minimal summary (total_ms) is extracted if available.
    """
    if not profile_file.exists():
        return None
    try:
        with profile_file.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return {"profile_keys": list(data.keys()), "artifact_path": str(profile_file)}
    except (json.JSONDecodeError, OSError):
        return None


def load_native_results(native_dir: Path) -> dict[str, Any]:
    """Load all native benchmark artifacts from a directory."""
    manifest = _load_native_manifest(native_dir)
    results: dict[str, Any] = {
        "native_dir": str(native_dir),
        "manifest": manifest,
    }

    # llm_bench results
    for mode in ("prefill", "decode", "visual"):
        key = f"llm_bench_{mode}"
        artifact_rel = manifest.get(key)
        if artifact_rel:
            artifact_path = native_dir / artifact_rel
            results[key] = _extract_llm_bench_summary(artifact_path)
        else:
            results[key] = None

    # llm_inference profile
    profile_rel = manifest.get("llm_inference_profile")
    if profile_rel:
        results["llm_inference_profile"] = _extract_profile_summary(
            native_dir / profile_rel
        )
    else:
        results["llm_inference_profile"] = None

    return results


# ── comparison report generation ─────────────────────────────────────────────


def _fmt(val: float | None, *, unit: str = "ms", places: int = 1) -> str:
    if val is None:
        return "n/a"
    return f"{val:.{places}f} {unit}"


def generate_comparison(
    ros_report: dict[str, Any],
    native_results: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Produce a machine-readable comparison dict.

    The comparison table has three sections:
      1. native_engine  — timing from NVIDIA llm_bench / llm_inference
      2. ros_overhead   — measured by this repository's instrumentation
      3. pipeline_total — sum and fractions
    """
    agg = ros_report.get("aggregate", {})
    meta = ros_report.get("metadata", {})

    inference_ms = agg.get("inference_ms", {})
    ros_overhead = agg.get("ros_overhead_ms", {})
    ipc_overhead = agg.get("ipc_overhead_ms", {})
    convert = agg.get("image_convert_ms", {})
    publication = agg.get("publication_ms", {})
    total = agg.get("total_worker_ms", {})

    inf_mean = inference_ms.get("mean")
    ros_mean = ros_overhead.get("mean")
    total_mean = total.get("mean")

    ros_fraction: float | None = None
    if inf_mean is not None and total_mean and total_mean > 0:
        ros_fraction = ros_mean / total_mean if ros_mean is not None else None

    comparison: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metadata": {
            "model_name": meta.get("model_name", ""),
            "quantization": meta.get("quantization", ""),
            "edge_llm_commit": meta.get("edge_llm_commit"),
            "max_generate_length": meta.get("max_generate_length"),
            "image_max_width": meta.get("image_max_width"),
            "jpeg_quality": meta.get("jpeg_quality"),
            "sample_period_seconds": meta.get("sample_period_seconds"),
            "warmup_frames": meta.get("warmup_frames"),
            "measured_frames": meta.get("measured_frames"),
            "platform": {
                "arch": meta.get("arch"),
                "jetpack_version": meta.get("jetpack_version"),
                "cuda_version": meta.get("cuda_version"),
                "tensorrt_version": meta.get("tensorrt_version"),
                "gpu_name": meta.get("gpu_name"),
                "gpu_compute_capability": meta.get("gpu_compute_capability"),
                "nvpmodel_mode": meta.get("nvpmodel_mode"),
            },
        },
        "native_engine": {
            "source": "TensorRT Edge-LLM worker timer (inference_seconds field)",
            "inference_ms_mean": inf_mean,
            "inference_ms_p50": inference_ms.get("p50"),
            "inference_ms_p95": inference_ms.get("p95"),
            "llm_bench_prefill": native_results.get("llm_bench_prefill") if native_results else None,
            "llm_bench_decode": native_results.get("llm_bench_decode") if native_results else None,
            "llm_bench_visual": native_results.get("llm_bench_visual") if native_results else None,
            "llm_inference_profile": native_results.get("llm_inference_profile") if native_results else None,
        },
        "ros_overhead": {
            "source": "cosmos_reasoner benchmark_output_file instrumentation",
            "image_convert_ms_mean": convert.get("mean"),
            "image_convert_ms_p50": convert.get("p50"),
            "ipc_overhead_ms_mean": ipc_overhead.get("mean"),
            "ipc_overhead_ms_p50": ipc_overhead.get("p50"),
            "publication_ms_mean": publication.get("mean"),
            "publication_ms_p50": publication.get("p50"),
            "ros_overhead_ms_mean": ros_mean,
            "ros_overhead_ms_p50": ros_overhead.get("p50"),
            "ros_overhead_ms_p95": ros_overhead.get("p95"),
            "total_dropped": agg.get("total_dropped"),
            "failed_frames": agg.get("failed_frames"),
            "ready_to_first_frame_ms": agg.get("ready_to_first_frame_ms"),
        },
        "pipeline_total": {
            "total_worker_ms_mean": total_mean,
            "total_worker_ms_p50": total.get("p50"),
            "total_worker_ms_p95": total.get("p95"),
            "ros_fraction_of_total": ros_fraction,
            "engine_fraction_of_total": (
                (inf_mean / total_mean) if inf_mean is not None and total_mean else None
            ),
        },
    }
    return comparison


def format_text_report(comparison: dict[str, Any]) -> str:
    """Format a human-readable text comparison report."""
    meta = comparison.get("metadata", {})
    native = comparison.get("native_engine", {})
    ros = comparison.get("ros_overhead", {})
    total = comparison.get("pipeline_total", {})
    platform = meta.get("platform", {})

    lines: list[str] = [
        "=" * 72,
        "  Cosmos ROS2 VLM Benchmark Report",
        f"  Generated: {comparison.get('generated_at', 'unknown')}",
        "=" * 72,
        "",
        "Platform",
        "--------",
        f"  GPU:          {platform.get('gpu_name', 'n/a')}",
        f"  Compute cap:  {platform.get('gpu_compute_capability', 'n/a')}",
        f"  JetPack:      {platform.get('jetpack_version', 'n/a')}",
        f"  CUDA:         {platform.get('cuda_version', 'n/a')}",
        f"  TensorRT:     {platform.get('tensorrt_version', 'n/a')}",
        f"  Power mode:   {platform.get('nvpmodel_mode', 'n/a')}",
        "",
        "Model / Engine",
        "--------------",
        f"  Model:        {meta.get('model_name', 'n/a')}",
        f"  Quantization: {meta.get('quantization', 'n/a')}",
        f"  EdgeLLM:      {meta.get('edge_llm_commit', 'n/a')}",
        f"  Max tokens:   {meta.get('max_generate_length', 'n/a')}",
        f"  Image width:  {meta.get('image_max_width', 'n/a')}",
        f"  JPEG quality: {meta.get('jpeg_quality', 'n/a')}",
        "",
        "Run Configuration",
        "-----------------",
        f"  Warmup frames:   {meta.get('warmup_frames', 'n/a')}",
        f"  Measured frames: {meta.get('measured_frames', 'n/a')}",
        f"  Sample period:   {meta.get('sample_period_seconds', 'n/a')} s",
        "",
        "Native Engine Timing  (authoritative — NVIDIA worker timer)",
        "------------------------------------------------------------",
        f"  Inference mean:   {_fmt(native.get('inference_ms_mean'))}",
        f"  Inference p50:    {_fmt(native.get('inference_ms_p50'))}",
        f"  Inference p95:    {_fmt(native.get('inference_ms_p95'))}",
        "",
        "ROS Pipeline Overhead  (repository instrumentation)",
        "----------------------------------------------------",
        f"  Image convert mean:  {_fmt(ros.get('image_convert_ms_mean'))}",
        f"  IPC overhead mean:   {_fmt(ros.get('ipc_overhead_ms_mean'))}",
        f"  Publication mean:    {_fmt(ros.get('publication_ms_mean'))}",
        f"  Total ROS mean:      {_fmt(ros.get('ros_overhead_ms_mean'))}",
        f"  Total ROS p95:       {_fmt(ros.get('ros_overhead_ms_p95'))}",
        f"  Ready to first frame: {_fmt(ros.get('ready_to_first_frame_ms'))}",
        f"  Dropped frames:      {ros.get('total_dropped', 'n/a')}",
        f"  Failed frames:       {ros.get('failed_frames', 'n/a')}",
        "",
        "End-to-End Pipeline  (native engine + ROS overhead)",
        "-----------------------------------------------------",
        f"  Total mean:       {_fmt(total.get('total_worker_ms_mean'))}",
        f"  Total p50:        {_fmt(total.get('total_worker_ms_p50'))}",
        f"  Total p95:        {_fmt(total.get('total_worker_ms_p95'))}",
    ]

    ros_frac = total.get("ros_fraction_of_total")
    eng_frac = total.get("engine_fraction_of_total")
    if ros_frac is not None and eng_frac is not None:
        lines += [
            "",
            "Time breakdown (mean)",
            "  Engine: {:5.1f}%  |  ROS overhead: {:5.1f}%".format(
                eng_frac * 100.0, ros_frac * 100.0
            ),
        ]

    if native.get("llm_bench_prefill") or native.get("llm_bench_decode") or native.get("llm_bench_visual"):
        lines += [
            "",
            "Native llm_bench Artifacts",
            "---------------------------",
            "  (See native_engine section of JSON report for full data)",
        ]

    lines += ["", "=" * 72, ""]
    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate comparison report (native engine vs ROS overhead)"
    )
    parser.add_argument(
        "--ros-report", required=True, type=Path,
        help="ROS metrics JSON from collect_ros_metrics.py"
    )
    parser.add_argument(
        "--native-dir", type=Path, default=None,
        help="Native benchmark artifacts directory from run_native_benchmarks.sh"
    )
    parser.add_argument(
        "--output", required=True, type=Path,
        help="Output JSON comparison report path"
    )
    parser.add_argument(
        "--text", type=Path, default=None,
        help="Optional plain-text report output path"
    )
    args = parser.parse_args()

    with args.ros_report.open("r", encoding="utf-8") as fh:
        ros_report = json.load(fh)

    native_results: dict[str, Any] | None = None
    if args.native_dir:
        native_results = load_native_results(args.native_dir)

    comparison = generate_comparison(ros_report, native_results)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(comparison, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"Comparison report: {args.output}", file=sys.stderr)

    text_report = format_text_report(comparison)
    if args.text:
        args.text.parent.mkdir(parents=True, exist_ok=True)
        args.text.write_text(text_report, encoding="utf-8")
        print(f"Text report: {args.text}", file=sys.stderr)
    else:
        sys.stdout.write(text_report)


if __name__ == "__main__":
    main()
