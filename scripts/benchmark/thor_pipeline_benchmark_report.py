#!/usr/bin/env python3
"""Summarize Thor full-pipeline benchmark runs and compare benchmark modes."""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

_RE_RAM = re.compile(r"RAM\s+(\d+)/(\d+)MB")
_RE_EMC = re.compile(r"EMC_FREQ\s+(\d+)%@([0-9]+)")
_RE_GR3D = re.compile(r"GR3D_FREQ\s+(?:(\d+)%@)?(?:@?\[([0-9,\s]+)\]|([0-9]+))")
_RE_POWER_RAIL = re.compile(r"\b((?:VDD|VIN)[A-Z0-9_]*)\s+([0-9]+)mW(?:/[0-9]+mW/[0-9]+mW)?")
_RE_TEMP = re.compile(r"(?:GPU|gpu)@([0-9]+(?:\.[0-9]+)?)C")
_RE_TJ = re.compile(r"(?:Tj|tj|Tboard|AO)@([0-9]+(?:\.[0-9]+)?)C")
_RE_CPU = re.compile(r"CPU\s*\[([^\]]+)\]")
_RE_CPU_PCT = re.compile(r"(\d+)%@")
_RE_TOPIC_HZ = re.compile(r"average rate:\s*([0-9]+(?:\.[0-9]+)?)")


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "max": None, "p95": None}
    ordered = sorted(values)
    idx = max(0, int(math.ceil(0.95 * len(ordered))) - 1)
    idx = max(0, min(idx, len(ordered) - 1))
    return {"mean": fmean(values), "max": max(values), "p95": ordered[idx]}


def parse_tegrastats_log(path: Path) -> dict[str, Any]:
    emc_pct: list[float] = []
    emc_mhz: list[float] = []
    gr3d_pct: list[float] = []
    gr3d_mhz: list[float] = []
    ram_used_mb: list[float] = []
    ram_total_mb: list[float] = []
    vdd_in_w: list[float] = []
    rail_power_w: dict[str, list[float]] = {}
    gpu_temp_c: list[float] = []
    tj_temp_c: list[float] = []
    cpu_core_samples: list[list[float]] = []

    if not path.exists():
        return {"samples": 0}

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if match := _RE_RAM.search(line):
            ram_used_mb.append(float(match.group(1)))
            ram_total_mb.append(float(match.group(2)))
        if match := _RE_EMC.search(line):
            emc_pct.append(float(match.group(1)))
            emc_mhz.append(float(match.group(2)))
        if match := _RE_GR3D.search(line):
            if match.group(1):
                gr3d_pct.append(float(match.group(1)))
            clocks_raw = match.group(2) or match.group(3)
            if clocks_raw:
                clocks = [float(value.strip()) for value in clocks_raw.split(",") if value.strip()]
                if clocks:
                    gr3d_mhz.append(fmean(clocks))
        line_rails: dict[str, float] = {}
        for rail_match in _RE_POWER_RAIL.finditer(line):
            rail = rail_match.group(1)
            line_rails[rail] = float(rail_match.group(2)) / 1000.0
        if "VDD_IN" in line_rails:
            vdd_in_w.append(line_rails["VDD_IN"])
        elif "VIN" in line_rails:
            vdd_in_w.append(line_rails["VIN"])
        for rail, value_w in line_rails.items():
            if rail in {"VIN", "VDD_IN"}:
                continue
            rail_power_w.setdefault(rail, []).append(value_w)
        if match := _RE_TEMP.search(line):
            gpu_temp_c.append(float(match.group(1)))
        if match := _RE_TJ.search(line):
            tj_temp_c.append(float(match.group(1)))
        if match := _RE_CPU.search(line):
            cpu_core_samples.append([float(v) for v in _RE_CPU_PCT.findall(match.group(1))])

    per_core: list[list[float]] = []
    max_cores = max((len(s) for s in cpu_core_samples), default=0)
    for core_idx in range(max_cores):
        per_core.append([sample[core_idx] for sample in cpu_core_samples if core_idx < len(sample)])

    hottest_core_p95 = None
    if per_core:
        hottest_core_p95 = max(_stats(core)["p95"] or 0.0 for core in per_core)

    ram_pct = []
    for used, total in zip(ram_used_mb, ram_total_mb):
        if total > 0:
            ram_pct.append((used / total) * 100.0)

    return {
        "samples": max(len(cpu_core_samples), len(emc_pct), len(gr3d_pct), len(gr3d_mhz), len(vdd_in_w), len(ram_used_mb)),
        "emc_pct": _stats(emc_pct),
        "emc_mhz": _stats(emc_mhz),
        "gr3d_pct": _stats(gr3d_pct),
        "gr3d_mhz": _stats(gr3d_mhz),
        "ram_used_mb": _stats(ram_used_mb),
        "ram_pct": _stats(ram_pct),
        "module_power_w": _stats(vdd_in_w),
        "power_rails_w": {rail: _stats(values) for rail, values in sorted(rail_power_w.items())},
        "gpu_temp_c": _stats(gpu_temp_c),
        "junction_temp_c": _stats(tj_temp_c),
        "cpu_hottest_core_p95_pct": hottest_core_p95,
    }


def parse_topic_hz_log(path: Path) -> float | None:
    if not path.exists():
        return None
    # ros2 topic hz prints a cumulative rolling average; use the final value as
    # the run-level average emitted at the end of capture.
    # Precondition: this log should be complete (not truncated).
    rates = [float(match.group(1)) for match in _RE_TOPIC_HZ.finditer(path.read_text(encoding="utf-8", errors="replace"))]
    return rates[-1] if rates else None


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_run(run_dir: Path) -> dict[str, Any]:
    config = _load_json(run_dir / "run_config.json") or {}
    ros_metrics = _load_json(run_dir / "ros_metrics.json") or {}
    aggregate = ros_metrics.get("aggregate", {})

    summary = {
        "run_id": config.get("run_id", run_dir.name),
        "mode": config.get("mode", run_dir.name),
        "description": config.get("description", ""),
        "settings": config,
        "tegrastats": parse_tegrastats_log(run_dir / "tegrastats.log"),
        "rates_hz": {
            "detections": parse_topic_hz_log(run_dir / "detections_hz.log"),
            "tracked_observation": parse_topic_hz_log(run_dir / "tracked_observation_hz.log"),
            "vlm_result": parse_topic_hz_log(run_dir / "vlm_result_hz.log"),
        },
        "vlm": {
            "inference_ms_mean": ((aggregate.get("inference_ms") or {}).get("mean")),
            "inference_ms_p95": ((aggregate.get("inference_ms") or {}).get("p95")),
            "result_frames": aggregate.get("successful_frames"),
            "failed_frames": aggregate.get("failed_frames"),
            "dropped_frames": aggregate.get("total_dropped"),
        },
    }
    return summary


def _score_contention(run: dict[str, Any], *, use_gr3d_mhz: bool) -> float:
    tegra = run.get("tegrastats", {})
    emc = ((tegra.get("emc_pct") or {}).get("mean"))
    if use_gr3d_mhz:
        gr3d = ((tegra.get("gr3d_mhz") or {}).get("mean"))
    else:
        gr3d = ((tegra.get("gr3d_pct") or {}).get("mean"))
    cpu = tegra.get("cpu_hottest_core_p95_pct")
    if emc is None or gr3d is None or cpu is None:
        return float("inf")
    return emc + gr3d + 0.5 * cpu


def compare_runs(run_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode = {run.get("mode"): run for run in run_summaries}

    contention_baseline = by_mode.get("C")
    cadence_modes = [mode for mode in ("D", "E", "F") if mode in by_mode]
    recommendation = None
    if cadence_modes:
        has_gr3d_pct = any(((by_mode[mode].get("tegrastats", {}).get("gr3d_pct") or {}).get("mean")) is not None for mode in cadence_modes)
        has_gr3d_mhz_only = any(
            ((by_mode[mode].get("tegrastats", {}).get("gr3d_pct") or {}).get("mean")) is None
            and ((by_mode[mode].get("tegrastats", {}).get("gr3d_mhz") or {}).get("mean")) is not None
            for mode in cadence_modes
        )
        if has_gr3d_pct and has_gr3d_mhz_only:
            recommendation = {
                "recommended_mode": None,
                "reason": "Cadence modes reported mixed GR3D units (percent and MHz); recommendation withheld.",
                "ranked_modes": [],
                "unavailable_modes": cadence_modes,
            }
        else:
            use_gr3d_mhz = not has_gr3d_pct
            scored_modes = {
                mode: _score_contention(by_mode[mode], use_gr3d_mhz=use_gr3d_mhz)
                for mode in cadence_modes
            }
            ranked = sorted(cadence_modes, key=lambda mode: scored_modes[mode])
            ranked_available = [mode for mode in ranked if math.isfinite(scored_modes[mode])]
            unavailable = [mode for mode in ranked if mode not in ranked_available]
            if ranked_available:
                basis = "GR3D MHz" if use_gr3d_mhz else "GR3D percent"
                recommendation = {
                    "recommended_mode": ranked_available[0],
                    "reason": f"Lowest combined EMC/{basis}/CPU contention score among tested cadence modes.",
                    "ranked_modes": ranked_available,
                    "unavailable_modes": unavailable,
                }
            else:
                recommendation = {
                    "recommended_mode": None,
                    "reason": "No cadence mode had complete EMC/GR3D/CPU metrics for contention scoring.",
                    "ranked_modes": [],
                    "unavailable_modes": unavailable,
                }

    findings = {
        "rtdetr_only_mode": by_mode.get("A", {}).get("tegrastats"),
        "vlm_only_mode": by_mode.get("B", {}).get("tegrastats"),
        "contention_mode": contention_baseline.get("tegrastats") if contention_baseline else None,
        "cpu_single_core_hotspot_present": any(
            (run.get("tegrastats", {}).get("cpu_hottest_core_p95_pct") or 0.0) >= 90.0
            for run in run_summaries
        ),
    }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_count": len(run_summaries),
        "runs": run_summaries,
        "findings": findings,
        "recommendation": recommendation,
    }


def format_text(report: dict[str, Any]) -> str:
    def _fmt(value: Any) -> str:
        if value is None:
            return "unavailable"
        return str(value)

    lines = [
        "================================================================",
        " Thor Full-Pipeline Benchmark Summary",
        f" Generated: {report.get('generated_at', 'n/a')}",
        "================================================================",
        "",
    ]
    for run in report.get("runs", []):
        tegra = run.get("tegrastats", {})
        lines += [
            f"[{run.get('mode')}] {run.get('description', '')}",
            f"  EMC mean: {_fmt(((tegra.get('emc_pct') or {}).get('mean')))}",
            f"  GR3D mean (%): {_fmt(((tegra.get('gr3d_pct') or {}).get('mean')))}",
            f"  GR3D mean (MHz): {_fmt(((tegra.get('gr3d_mhz') or {}).get('mean')))}",
            f"  Module power mean (W): {_fmt(((tegra.get('module_power_w') or {}).get('mean')))}",
            f"  CPU hottest-core p95: {_fmt(tegra.get('cpu_hottest_core_p95_pct'))}",
            f"  Detections Hz: {_fmt((run.get('rates_hz') or {}).get('detections'))}",
            f"  Tracked-observation Hz: {_fmt((run.get('rates_hz') or {}).get('tracked_observation'))}",
            f"  VLM result Hz: {_fmt((run.get('rates_hz') or {}).get('vlm_result'))}",
            f"  VLM inference mean ms: {_fmt((run.get('vlm') or {}).get('inference_ms_mean'))}",
            "",
        ]

    recommendation = report.get("recommendation") or {}
    if recommendation:
        lines += [
            "Recommendation",
            "--------------",
            f"Mode {recommendation.get('recommended_mode')}: {recommendation.get('reason')}",
            f"Ranking: {', '.join(recommendation.get('ranked_modes', [])) or 'unavailable'}",
            f"Unavailable for scoring: {', '.join(recommendation.get('unavailable_modes', [])) or 'none'}",
            "",
        ]

    findings = report.get("findings") or {}
    lines += [
        "Key Findings",
        "------------",
        f"CPU single-core hotspot observed: {findings.get('cpu_single_core_hotspot_present')}",
        "================================================================",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Thor full-pipeline benchmark comparison report")
    parser.add_argument("--run-root", required=True, type=Path, help="Directory containing run subdirectories")
    parser.add_argument("--modes", default="A,B,C,D,E,F", help="Comma-separated mode list to include")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON report path")
    parser.add_argument("--text", type=Path, default=None, help="Optional plain-text summary output path")
    args = parser.parse_args()

    modes = [value.strip() for value in args.modes.split(",") if value.strip()]
    runs: list[dict[str, Any]] = []
    for mode in modes:
        run_dir = args.run_root / f"run_{mode}"
        if run_dir.exists():
            runs.append(summarize_run(run_dir))

    report = compare_runs(runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    text = format_text(report)
    if args.text:
        args.text.parent.mkdir(parents=True, exist_ok=True)
        args.text.write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
