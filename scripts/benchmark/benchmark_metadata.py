#!/usr/bin/env python3
"""
Collect system, model, and ROS metadata for benchmark records.

All fields are best-effort; unavailable fields are None.  This module never
raises on a missing tool — it logs a warning and continues so that
benchmarking is not blocked on a partially-configured system.
"""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ── helpers ──────────────────────────────────────────────────────────────────


def _run(cmd: list[str], *, timeout: int = 5) -> str:
    """Run a command and return its stdout; return '' on any error."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def _read_file(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""


# ── platform ─────────────────────────────────────────────────────────────────


def collect_platform_metadata() -> dict[str, Any]:
    """Return hardware/OS/JetPack/CUDA/TensorRT version fields."""
    meta: dict[str, Any] = {
        "hostname": platform.node(),
        "arch": platform.machine(),
        "os": platform.system(),
        "os_release": _read_file("/etc/os-release")[:256] or None,
        "kernel": platform.release(),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }

    # JetPack / L4T
    nv_tegra = _read_file("/etc/nv_tegra_release")
    meta["l4t_release"] = nv_tegra or None

    jetpack_raw = _run(["dpkg-query", "-W", "-f=${Version}", "nvidia-jetpack"])
    meta["jetpack_version"] = jetpack_raw or None

    # CUDA
    nvcc_out = _run(["nvcc", "--version"])
    m = re.search(r"release\s+([\d.]+)", nvcc_out)
    meta["cuda_version"] = m.group(1) if m else None

    cuda_lib = _run(["dpkg-query", "-W", "-f=${Version}", "libcuda1"])
    meta["cuda_driver_version"] = cuda_lib or None

    # TensorRT
    trt_raw = _run(["dpkg-query", "-W", "-f=${Version}", "libnvinfer-dev"])
    meta["tensorrt_version"] = trt_raw or None

    # GPU power mode and clocks (Jetson-specific)
    meta["nvpmodel_mode"] = _run(["nvpmodel", "-q"]) or None
    meta["jetson_clocks_active"] = (
        _run(["jetson_clocks", "--show"]).startswith("CPU") if Path("/usr/bin/jetson_clocks").exists() else None
    )

    # GPU compute capability (from nvidia-smi)
    smi_cap = _run(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"]
    )
    meta["gpu_compute_capability"] = smi_cap.split("\n")[0].strip() or None
    smi_name = _run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
    )
    meta["gpu_name"] = smi_name.split("\n")[0].strip() or None

    return meta


# ── model / engine ────────────────────────────────────────────────────────────


def collect_model_metadata(
    llm_engine_dir: str = "",
    multimodal_engine_dir: str = "",
    *,
    model_name: str = "",
    quantization: str = "",
    edge_llm_root: str = "",
) -> dict[str, Any]:
    """Return model/engine/token-budget configuration fields."""
    meta: dict[str, Any] = {
        "model_name": model_name or os.environ.get("EDGE_VLM_MODEL_NAME", ""),
        "quantization": quantization or "",
        "llm_engine_dir": llm_engine_dir,
        "multimodal_engine_dir": multimodal_engine_dir,
    }

    # TensorRT Edge-LLM git commit / version
    edge_llm_root = edge_llm_root or os.environ.get("TENSORRT_EDGE_LLM_ROOT", "")
    if edge_llm_root:
        commit = _run(
            ["git", "-C", edge_llm_root, "rev-parse", "--short", "HEAD"]
        )
        meta["edge_llm_commit"] = commit or None
        tag = _run(
            ["git", "-C", edge_llm_root, "describe", "--tags", "--abbrev=0"]
        )
        meta["edge_llm_version_tag"] = tag or None
    else:
        meta["edge_llm_commit"] = None
        meta["edge_llm_version_tag"] = None

    # Engine config.json if present
    config_path = Path(llm_engine_dir) / "config.json" if llm_engine_dir else None
    if config_path and config_path.exists():
        try:
            engine_cfg = json.loads(config_path.read_text(encoding="utf-8"))
            meta["engine_config"] = engine_cfg
        except Exception:
            meta["engine_config"] = None
    else:
        meta["engine_config"] = None

    return meta


# ── ROS ───────────────────────────────────────────────────────────────────────


def collect_ros_metadata(
    *,
    image_topic: str = "",
    sample_period_seconds: float = 0.0,
    max_generate_length: int = 0,
    jpeg_quality: int = 0,
    image_max_width: int = 0,
    drop_old_frames: bool = True,
    task_profile: str = "",
    prompt_version: str = "",
) -> dict[str, Any]:
    """Return ROS pipeline configuration fields."""
    return {
        "ros_distro": os.environ.get("ROS_DISTRO", None),
        "image_topic": image_topic,
        "sample_period_seconds": sample_period_seconds,
        "max_generate_length": max_generate_length,
        "jpeg_quality": jpeg_quality,
        "image_max_width": image_max_width,
        "drop_old_frames": drop_old_frames,
        "task_profile": task_profile,
        "prompt_version": prompt_version,
    }


# ── combined ─────────────────────────────────────────────────────────────────


def collect_all_metadata(**kwargs: Any) -> dict[str, Any]:
    """Return the union of platform, model, and ROS metadata."""
    meta: dict[str, Any] = {}
    meta.update(collect_platform_metadata())
    meta.update(
        collect_model_metadata(
            llm_engine_dir=kwargs.get("llm_engine_dir", ""),
            multimodal_engine_dir=kwargs.get("multimodal_engine_dir", ""),
            model_name=kwargs.get("model_name", ""),
            quantization=kwargs.get("quantization", ""),
            edge_llm_root=kwargs.get("edge_llm_root", ""),
        )
    )
    meta.update(
        collect_ros_metadata(
            image_topic=kwargs.get("image_topic", ""),
            sample_period_seconds=kwargs.get("sample_period_seconds", 0.0),
            max_generate_length=kwargs.get("max_generate_length", 0),
            jpeg_quality=kwargs.get("jpeg_quality", 0),
            image_max_width=kwargs.get("image_max_width", 0),
            drop_old_frames=kwargs.get("drop_old_frames", True),
            task_profile=kwargs.get("task_profile", ""),
            prompt_version=kwargs.get("prompt_version", ""),
        )
    )
    return meta


# ── CLI entry point ───────────────────────────────────────────────────────────


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Print system/model/ROS benchmark metadata as JSON"
    )
    parser.add_argument("--llm-engine-dir", default="")
    parser.add_argument("--multimodal-engine-dir", default="")
    parser.add_argument("--model-name", default="")
    parser.add_argument("--quantization", default="")
    parser.add_argument("--edge-llm-root", default="")
    parser.add_argument("--output", default="-", help="Output JSON path (- for stdout)")
    args = parser.parse_args()

    meta = collect_all_metadata(
        llm_engine_dir=args.llm_engine_dir,
        multimodal_engine_dir=args.multimodal_engine_dir,
        model_name=args.model_name,
        quantization=args.quantization,
        edge_llm_root=args.edge_llm_root,
    )

    text = json.dumps(meta, indent=2, sort_keys=True) + "\n"
    if args.output == "-":
        sys.stdout.write(text)
    else:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
