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
import hashlib
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ── helpers ──────────────────────────────────────────────────────────────────

#
# edge_vlm_server is launched as:
#   edge_vlm_server <llm_engine_dir> <multimodal_engine_dir> <plugin_path> <socket_path> ...
#
_EDGE_VLM_SERVER_ARGV_LLM_DIR = 1
_EDGE_VLM_SERVER_ARGV_MULTIMODAL_DIR = 2
_EDGE_VLM_SERVER_ARGV_SOCKET_PATH = 4


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


def _canonical_path(path: str) -> str:
    if not path:
        return ""
    return str(Path(path).expanduser().resolve(strict=False))


def _sha256_file(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _json_digest(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _infer_profile_id(llm_engine_dir: str, multimodal_engine_dir: str) -> str | None:
    llm_path = Path(llm_engine_dir) if llm_engine_dir else None
    multimodal_path = Path(multimodal_engine_dir) if multimodal_engine_dir else None
    if not llm_path or not multimodal_path:
        return None
    if multimodal_path.name == "engine" and llm_path == multimodal_path / "llm":
        return "legacy"
    if multimodal_path.parent.name == "engines" and llm_path == multimodal_path / "llm":
        return multimodal_path.name
    return None


def _infer_model_name(
    llm_engine_dir: str,
    multimodal_engine_dir: str,
    *,
    fallback: str = "",
) -> str:
    if fallback:
        return fallback
    multimodal_path = Path(multimodal_engine_dir) if multimodal_engine_dir else None
    if not multimodal_path:
        return ""
    if multimodal_path.name == "engine":
        return multimodal_path.parent.name
    if multimodal_path.parent.name == "engines":
        return multimodal_path.parent.parent.name
    return multimodal_path.parent.name


def _candidate_manifest_paths(llm_engine_dir: str, multimodal_engine_dir: str) -> list[Path]:
    candidates: list[Path] = []
    llm_path = Path(llm_engine_dir) if llm_engine_dir else None
    multimodal_path = Path(multimodal_engine_dir) if multimodal_engine_dir else None
    if multimodal_path:
        candidates.append(multimodal_path / "engine-manifest.json")
    if llm_path and llm_path.name == "llm":
        candidates.append(llm_path.parent / "engine-manifest.json")
    deduped: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate not in seen:
            deduped.append(candidate)
            seen.add(candidate)
    return deduped


def _socket_listener_pid(socket_path: str) -> int | None:
    canonical_socket = _canonical_path(socket_path)
    if not canonical_socket:
        return None
    output = _run(["ss", "-lxnp"])
    if not output:
        return None
    for line in output.splitlines():
        if canonical_socket not in line:
            continue
        match = re.search(r"pid=(\d+)", line)
        if match:
            return int(match.group(1))
    return None


def _proc_argv(pid: int, index: int) -> str:
    try:
        parts = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except OSError:
        return ""
    if index < 0 or index >= len(parts):
        return ""
    return parts[index].decode("utf-8", errors="replace").strip()


def collect_engine_provenance(
    llm_engine_dir: str = "",
    multimodal_engine_dir: str = "",
    *,
    model_name: str = "",
    engine_profile_id: str = "",
) -> dict[str, Any]:
    """Return canonical engine provenance for the runtime paths in use."""
    canonical_llm = _canonical_path(llm_engine_dir or os.environ.get("EDGE_VLM_LLM_ENGINE_DIR", ""))
    canonical_multimodal = _canonical_path(
        multimodal_engine_dir or os.environ.get("EDGE_VLM_MULTIMODAL_ENGINE_DIR", "")
    )
    requested_model_name = model_name or os.environ.get("EDGE_VLM_MODEL_NAME", "")
    requested_profile_id = engine_profile_id or os.environ.get("EDGE_VLM_ENGINE_PROFILE_ID", "")

    provenance: dict[str, Any] = {
        "model_name": requested_model_name,
        "engine_profile_id": requested_profile_id,
        "llm_engine_dir": canonical_llm,
        "multimodal_engine_dir": canonical_multimodal,
        "engine_manifest_path": None,
        "engine_manifest_sha256": None,
        "engine_identity": None,
        "engine_manifest_status": "missing",
        "provenance_warnings": [],
    }

    manifest_path: Path | None = None
    for candidate in _candidate_manifest_paths(canonical_llm, canonical_multimodal):
        if candidate.is_file():
            manifest_path = candidate.resolve(strict=False)
            break

    manifest_payload: dict[str, Any] | None = None
    if manifest_path is not None:
        provenance["engine_manifest_path"] = str(manifest_path)
        provenance["engine_manifest_sha256"] = _sha256_file(manifest_path)
        try:
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            provenance["engine_manifest_status"] = "matched"
        except (OSError, json.JSONDecodeError) as exc:
            provenance["engine_manifest_status"] = "invalid"
            provenance["provenance_warnings"].append(
                f"engine-manifest.json could not be parsed: {exc}"
            )

    if isinstance(manifest_payload, dict):
        manifest_model_name = str(manifest_payload.get("model_name") or "")
        manifest_profile_id = str(manifest_payload.get("engine_profile_id") or "")
        if manifest_model_name:
            provenance["model_name"] = manifest_model_name
        elif not provenance["model_name"]:
            provenance["model_name"] = _infer_model_name(
                canonical_llm,
                canonical_multimodal,
                fallback=requested_model_name,
            )
        if manifest_profile_id:
            provenance["engine_profile_id"] = manifest_profile_id
        elif not provenance["engine_profile_id"]:
            inferred_profile = _infer_profile_id(canonical_llm, canonical_multimodal)
            provenance["engine_profile_id"] = inferred_profile or requested_profile_id

        engine_paths = manifest_payload.get("engine_paths")
        manifest_llm = ""
        manifest_multimodal = ""
        if isinstance(engine_paths, dict):
            manifest_llm = _canonical_path(str(engine_paths.get("llm_dir") or ""))
            manifest_multimodal = _canonical_path(str(engine_paths.get("multimodal_dir") or ""))
        if manifest_llm and manifest_llm != canonical_llm:
            provenance["engine_manifest_status"] = "mismatch"
            provenance["provenance_warnings"].append(
                f"engine-manifest llm_dir {manifest_llm} does not match runtime path {canonical_llm}"
            )
        if manifest_multimodal and manifest_multimodal != canonical_multimodal:
            provenance["engine_manifest_status"] = "mismatch"
            provenance["provenance_warnings"].append(
                "engine-manifest multimodal_dir "
                f"{manifest_multimodal} does not match runtime path {canonical_multimodal}"
            )
        if requested_model_name and manifest_model_name and requested_model_name != manifest_model_name:
            provenance["engine_manifest_status"] = "mismatch"
            provenance["provenance_warnings"].append(
                f"requested model_name {requested_model_name!r} does not match manifest model_name {manifest_model_name!r}"
            )
        if requested_profile_id and manifest_profile_id and requested_profile_id != manifest_profile_id:
            provenance["engine_manifest_status"] = "mismatch"
            provenance["provenance_warnings"].append(
                "requested engine_profile_id "
                f"{requested_profile_id!r} does not match manifest engine_profile_id {manifest_profile_id!r}"
            )
    else:
        inferred_profile = _infer_profile_id(canonical_llm, canonical_multimodal)
        provenance["engine_profile_id"] = inferred_profile or requested_profile_id or "legacy"
        provenance["model_name"] = _infer_model_name(
            canonical_llm,
            canonical_multimodal,
            fallback=requested_model_name,
        )
        if inferred_profile and inferred_profile != "legacy":
            provenance["provenance_warnings"].append(
                "managed-style engine layout has no engine-manifest.json"
            )

    identity_model = provenance["model_name"] or "unknown-model"
    identity_profile = provenance["engine_profile_id"] or "unknown-profile"
    manifest_sha = provenance.get("engine_manifest_sha256")
    if manifest_sha:
        provenance["engine_identity"] = f"{identity_model}/{identity_profile}@{manifest_sha[:12]}"
    else:
        fallback_sha = _json_digest(
            {
                "model_name": identity_model,
                "engine_profile_id": identity_profile,
                "llm_engine_dir": canonical_llm,
                "multimodal_engine_dir": canonical_multimodal,
            }
        )
        provenance["engine_identity"] = f"{identity_model}/{identity_profile}@{fallback_sha[:12]}"

    return provenance


def collect_server_engine_provenance(
    socket_path: str = "",
    *,
    model_name: str = "",
    engine_profile_id: str = "",
) -> dict[str, Any]:
    """Return canonical engine provenance for the live edge_vlm_server listener."""
    requested_socket = socket_path or os.environ.get("EDGE_VLM_WORKER_SOCKET", "/tmp/edge_vlm.sock")
    server_pid = _socket_listener_pid(requested_socket)
    if server_pid is None:
        raise RuntimeError(f"no live edge_vlm_server listener found for socket {requested_socket!r}")

    llm_engine_dir = _proc_argv(server_pid, _EDGE_VLM_SERVER_ARGV_LLM_DIR)
    multimodal_engine_dir = _proc_argv(server_pid, _EDGE_VLM_SERVER_ARGV_MULTIMODAL_DIR)
    server_socket_path = _proc_argv(server_pid, _EDGE_VLM_SERVER_ARGV_SOCKET_PATH) or requested_socket
    if not llm_engine_dir or not multimodal_engine_dir:
        raise RuntimeError(
            f"could not determine edge_vlm_server engine directories from /proc/{server_pid}/cmdline"
        )

    provenance = collect_engine_provenance(
        llm_engine_dir=llm_engine_dir,
        multimodal_engine_dir=multimodal_engine_dir,
        model_name=model_name,
        engine_profile_id=engine_profile_id,
    )
    provenance["server_pid"] = server_pid
    provenance["server_socket_path"] = _canonical_path(server_socket_path)
    provenance["provenance_source"] = "server_process"
    return provenance


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
    engine_profile_id: str = "",
    quantization: str = "",
    edge_llm_root: str = "",
) -> dict[str, Any]:
    """Return model/engine/token-budget configuration fields."""
    provenance = collect_engine_provenance(
        llm_engine_dir=llm_engine_dir,
        multimodal_engine_dir=multimodal_engine_dir,
        model_name=model_name,
        engine_profile_id=engine_profile_id,
    )
    meta: dict[str, Any] = {
        "model_name": provenance["model_name"],
        "engine_profile_id": provenance["engine_profile_id"],
        "quantization": quantization or "",
        "llm_engine_dir": provenance["llm_engine_dir"],
        "multimodal_engine_dir": provenance["multimodal_engine_dir"],
        "engine_manifest_path": provenance["engine_manifest_path"],
        "engine_manifest_sha256": provenance["engine_manifest_sha256"],
        "engine_identity": provenance["engine_identity"],
        "engine_manifest_status": provenance["engine_manifest_status"],
        "engine_provenance": provenance,
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
    llm_dir = provenance["llm_engine_dir"]
    config_path = Path(llm_dir) / "config.json" if llm_dir else None
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
            engine_profile_id=kwargs.get("engine_profile_id", ""),
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
    parser.add_argument("--engine-profile-id", default="")
    parser.add_argument("--quantization", default="")
    parser.add_argument("--edge-llm-root", default="")
    parser.add_argument("--server-socket-path", default="")
    parser.add_argument(
        "--output-provenance-lines",
        action="store_true",
        help="Print canonical provenance as 5 newline-delimited fields instead of JSON",
    )
    parser.add_argument("--output", default="-", help="Output JSON path (- for stdout)")
    args = parser.parse_args()

    if args.output_provenance_lines:
        if args.server_socket_path:
            provenance = collect_server_engine_provenance(
                socket_path=args.server_socket_path,
                model_name=args.model_name,
                engine_profile_id=args.engine_profile_id,
            )
        else:
            provenance = collect_engine_provenance(
                llm_engine_dir=args.llm_engine_dir,
                multimodal_engine_dir=args.multimodal_engine_dir,
                model_name=args.model_name,
                engine_profile_id=args.engine_profile_id,
            )
        sys.stdout.write((provenance.get("model_name") or "") + "\n")
        sys.stdout.write((provenance.get("engine_profile_id") or "") + "\n")
        sys.stdout.write((provenance.get("llm_engine_dir") or "") + "\n")
        sys.stdout.write((provenance.get("multimodal_engine_dir") or "") + "\n")
        sys.stdout.write(json.dumps(provenance, sort_keys=True) + "\n")
        return

    meta = collect_all_metadata(
        llm_engine_dir=args.llm_engine_dir,
        multimodal_engine_dir=args.multimodal_engine_dir,
        model_name=args.model_name,
        engine_profile_id=args.engine_profile_id,
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
