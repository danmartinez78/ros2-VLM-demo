# Copyright 2025 edge_vlm_ros contributors
"""
ROS-independent experiment engine for the web workbench.

Runs ordered image sequences through the edge_vlm inference service using
configurable strategies and produces structured JSONL artifacts.

Strategies
----------
single_frame
    Each image is sent to the service independently with the task prompt.
    No context accumulates between frames.

single_frame_observation_history
    Each image is sent with the task prompt plus a rolling window of the last
    N model responses ("semantic continuity context").  This is a baseline for
    studying the effect of observation history depth — it is NOT a true
    multi-frame visual window.

Usage (no ROS required)
-----------------------
    from web_console.experiment_engine import (
        ExperimentDefinition, run_experiment
    )
    from web_console.inference_client import run_inference

    defn = ExperimentDefinition(
        strategy="single_frame",
        image_paths=["/data/frame_001.jpg", "/data/frame_002.jpg"],
        task_prompt="Describe the scene.",
    )
    results = run_experiment(defn, inference_fn=run_inference, ...)

Thread safety
-------------
Each call to run_experiment is independent.  No shared mutable state is
modified.  Concurrent experiments may write to different artifact directories
without interference.
"""
from __future__ import annotations

import datetime
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .inference_client import (
    ALLOWED_IMAGE_EXTENSIONS,
    InferenceResult,
    run_inference,
)

# ── constants ─────────────────────────────────────────────────────────────────

_VALID_STRATEGIES = frozenset({"single_frame", "single_frame_observation_history"})
_MAX_HISTORY_ENTRIES: int = 256
_MAX_HISTORY_CHARS: int = 1_000_000
_MAX_IMAGES: int = 10_000
# Repetition/contradiction heuristics: flag when last N responses are ≥ 80 %
# similar by normalised length-3 token overlap (a weak but fast signal).
_REPETITION_WINDOW: int = 3
_REPETITION_THRESHOLD: float = 0.8
_SCHEMA_VERSION: int = 1


# ── type alias ────────────────────────────────────────────────────────────────

InferenceFn = Callable[..., InferenceResult]


# ── dataclasses ───────────────────────────────────────────────────────────────


@dataclass
class ExperimentDefinition:
    """Fully reproducible description of one experiment run.

    All fields that affect the output are stored here so that a JSONL artifact
    can be replayed or compared at a later date without consulting external
    state.
    """

    # ── required ──────────────────────────────────────────────────────────────
    strategy: str
    """One of the values in _VALID_STRATEGIES."""

    image_paths: List[str]
    """Ordered sequence of absolute or relative image paths."""

    task_prompt: str
    """User-visible task / question prompt."""

    # ── optional ──────────────────────────────────────────────────────────────
    system_instruction: str = (
        "You are a vision observer. Base claims on the current image."
    )
    """System-level instruction prepended to the prompt (when non-empty)."""

    observation_history_max_entries: int = 0
    """
    Maximum number of prior model responses to include in the prompt context.
    Only used by the ``single_frame_observation_history`` strategy.
    0 means no history (equivalent to ``single_frame``).
    """

    observation_history_max_chars: int = 4000
    """Hard character budget for the concatenated history block."""

    max_generate_length: int = 96
    temperature: float = 0.2
    top_p: float = 0.9
    top_k: int = 20
    timeout_seconds: int = 120

    # ── metadata ──────────────────────────────────────────────────────────────
    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: _now_iso())
    run_id: Optional[str] = None
    """Populated by run_experiment() when executed via the web console."""

    notes: str = ""
    """Free-form notes for reproducibility."""

    # ── source provenance (set by server when running from a frame dataset) ──
    source_dataset_id: Optional[str] = None
    """UUID of the FrameDataset this experiment draws from, or None."""

    source_frame_records: Optional[List[Dict[str, Any]]] = None
    """Ordered list of {frame_index, timestamp_ns} dicts, parallel to image_paths."""

    # ── profile provenance (set by server when a task profile is selected) ──
    profile_name: Optional[str] = None
    """Name of the task profile used (e.g. 'warehouse_awareness'), or None."""

    profile_version: Optional[str] = None
    profile_hash: Optional[str] = None


@dataclass
class FrameResult:
    """Structured result for a single frame."""

    frame_index: int
    image_path: str
    prompt_used: str
    success: bool
    text: str = ""
    error: str = ""
    latency_ms: float = 0.0
    history_entries_used: int = 0
    repetition_flag: bool = False
    """Set when the response appears repetitive relative to recent history."""

    # ── source provenance ─────────────────────────────────────────────────────
    source_dataset_id: Optional[str] = None
    """UUID of the frame dataset this image came from."""

    source_frame_index: Optional[int] = None
    """Frame index within the source dataset (may differ from iteration index)."""

    source_timestamp_ns: Optional[int] = None
    """Timestamp of the source frame in nanoseconds."""


@dataclass
class ExperimentSummary:
    """Aggregate metrics computed after all frames have been processed."""

    total_frames: int
    successful_frames: int
    failed_frames: int
    mean_latency_ms: Optional[float]
    min_latency_ms: Optional[float]
    max_latency_ms: Optional[float]
    repetition_flags: int
    strategy: str
    history_depth: int


# ── helpers ───────────────────────────────────────────────────────────────────


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _safe_filename(run_id: str) -> bool:
    """Return True only for UUID-shaped run IDs."""
    return bool(
        re.match(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            run_id,
        )
    )


def _build_prompt(
    task_prompt: str,
    system_instruction: str,
    history: List[str],
    max_history_chars: int,
) -> str:
    """Build the final prompt string with optional observation history."""
    parts: List[str] = []
    if system_instruction:
        parts.append(system_instruction)
    if history:
        history_block = _truncate_history(history, max_history_chars)
        if history_block:
            parts.append("Prior observations:")
            parts.append(history_block)
    parts.append(task_prompt)
    return "\n\n".join(parts)


def _truncate_history(history: List[str], max_chars: int) -> str:
    """Return a history block that fits within max_chars characters.

    Entries are added from most-recent to least-recent; the resulting block
    preserves chronological order (oldest first).
    """
    if not history or max_chars <= 0:
        return ""
    selected: List[str] = []
    used = 0
    for entry in reversed(history):
        need = len(entry) + (2 if selected else 0)  # "; " separator
        if used + need > max_chars:
            break
        selected.insert(0, entry)
        used += need
    return "; ".join(selected)


def _compute_repetition(new_text: str, recent_texts: List[str]) -> bool:
    """Weak heuristic: flag repetition when trigram overlap with recent entries
    exceeds the threshold.  Returns False when there is insufficient history or
    the texts are empty."""
    if not new_text or not recent_texts:
        return False

    def _trigrams(text: str):
        words = re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()
        return set(
            " ".join(words[i : i + 3]) for i in range(len(words) - 2)
        )

    new_set = _trigrams(new_text)
    if not new_set:
        return False

    for prev_text in recent_texts[-_REPETITION_WINDOW :]:
        prev_set = _trigrams(prev_text)
        if not prev_set:
            continue
        overlap = len(new_set & prev_set) / min(len(new_set), len(prev_set))
        if overlap >= _REPETITION_THRESHOLD:
            return True
    return False


def _write_artifact_jsonl(
    artifact_dir: Path, filename: str, records: List[Dict[str, Any]]
) -> None:
    """Write records as newline-delimited JSON.  The directory must already exist."""
    path = artifact_dir / filename
    tmp = artifact_dir / f"{filename}.tmp.{os.getpid()}"
    with tmp.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, separators=(",", ":")) + "\n")
    tmp.rename(path)


# ── validation ────────────────────────────────────────────────────────────────


def validate_definition(defn: ExperimentDefinition) -> Optional[str]:
    """Return an error string if the definition is invalid, else None."""
    if defn.strategy not in _VALID_STRATEGIES:
        return (
            f"Unknown strategy: {defn.strategy!r}. "
            f"Valid strategies: {sorted(_VALID_STRATEGIES)}"
        )
    if not defn.image_paths:
        return "image_paths must not be empty"
    if len(defn.image_paths) > _MAX_IMAGES:
        return f"Too many images: {len(defn.image_paths)} (max {_MAX_IMAGES})"
    for p in defn.image_paths:
        ext = Path(p).suffix.lower()
        if ext not in ALLOWED_IMAGE_EXTENSIONS:
            return (
                f"Unsupported image extension: {ext!r} in {p!r}. "
                f"Allowed: {sorted(ALLOWED_IMAGE_EXTENSIONS)}"
            )
    if not defn.task_prompt:
        return "task_prompt must not be empty"
    if not (0 < defn.max_generate_length <= 4096):
        return f"max_generate_length must be in [1, 4096]; got {defn.max_generate_length}"
    if not (0.0 <= defn.temperature <= 2.0):
        return f"temperature must be in [0.0, 2.0]; got {defn.temperature}"
    if not (0.0 < defn.top_p <= 1.0):
        return f"top_p must be in (0.0, 1.0]; got {defn.top_p}"
    if defn.top_k < 1:
        return f"top_k must be >= 1; got {defn.top_k}"
    if not (0 <= defn.observation_history_max_entries <= _MAX_HISTORY_ENTRIES):
        return (
            f"observation_history_max_entries must be in [0, {_MAX_HISTORY_ENTRIES}]; "
            f"got {defn.observation_history_max_entries}"
        )
    if not (0 <= defn.observation_history_max_chars <= _MAX_HISTORY_CHARS):
        return (
            f"observation_history_max_chars must be in [0, {_MAX_HISTORY_CHARS}]; "
            f"got {defn.observation_history_max_chars}"
        )
    return None


# ── experiment execution ──────────────────────────────────────────────────────


def run_experiment(
    defn: ExperimentDefinition,
    *,
    cli_path: str = "edge_vlm_cli",
    socket_path: str = "/tmp/edge_vlm.sock",
    artifact_dir: Optional[Path] = None,
    inference_fn: Optional[InferenceFn] = None,
    cancel_fn: Optional[Callable[[], bool]] = None,
    on_frame: Optional[Callable[[int, "FrameResult"], None]] = None,
) -> List[FrameResult]:
    """Execute the experiment described by *defn* and return per-frame results.

    Parameters
    ----------
    defn:
        Fully-specified experiment definition.
    cli_path:
        Path to the edge_vlm_cli binary (forwarded to inference_fn).
    socket_path:
        IPC socket path (forwarded to inference_fn).
    artifact_dir:
        When provided, JSONL artifacts are written here.
    inference_fn:
        Callable with the same signature as
        ``inference_client.run_inference``.  Defaults to
        ``run_inference`` from ``inference_client``.

    Returns
    -------
    List[FrameResult]
        One entry per image, in input order.

    Raises
    ------
    ValueError
        If the definition is invalid.
    """
    if inference_fn is None:
        inference_fn = run_inference

    error = validate_definition(defn)
    if error:
        raise ValueError(error)

    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)

    if defn.strategy == "single_frame":
        return _run_single_frame(
            defn,
            cli_path=cli_path,
            socket_path=socket_path,
            artifact_dir=artifact_dir,
            inference_fn=inference_fn,
            cancel_fn=cancel_fn,
            on_frame=on_frame,
        )
    elif defn.strategy == "single_frame_observation_history":
        return _run_observation_history(
            defn,
            cli_path=cli_path,
            socket_path=socket_path,
            artifact_dir=artifact_dir,
            inference_fn=inference_fn,
            cancel_fn=cancel_fn,
            on_frame=on_frame,
        )
    else:
        # Unreachable after validate_definition, but defensive.
        raise ValueError(f"Unimplemented strategy: {defn.strategy!r}")


def _run_single_frame(
    defn: ExperimentDefinition,
    *,
    cli_path: str,
    socket_path: str,
    artifact_dir: Optional[Path],
    inference_fn: InferenceFn,
    cancel_fn: Optional[Callable[[], bool]] = None,
    on_frame: Optional[Callable[[int, "FrameResult"], None]] = None,
) -> List[FrameResult]:
    """Strategy: single_frame — no accumulated context between frames."""
    prompt = _build_prompt(defn.task_prompt, defn.system_instruction, [], 0)
    results: List[FrameResult] = []
    jsonl_records: List[Dict[str, Any]] = []

    for idx, image_path in enumerate(defn.image_paths):
        if cancel_fn is not None and cancel_fn():
            break

        t0 = time.monotonic()
        infer_result = inference_fn(
            cli_path=cli_path,
            socket_path=socket_path,
            image_path=image_path,
            prompt=prompt,
            max_generate_length=defn.max_generate_length,
            temperature=defn.temperature,
            top_p=defn.top_p,
            top_k=defn.top_k,
            timeout_seconds=defn.timeout_seconds,
        )
        elapsed_ms = (time.monotonic() - t0) * 1000.0

        src_rec = (
            defn.source_frame_records[idx]
            if defn.source_frame_records and idx < len(defn.source_frame_records)
            else None
        )
        fr = FrameResult(
            frame_index=idx,
            image_path=image_path,
            prompt_used=prompt,
            success=infer_result.success,
            text=infer_result.text,
            error=infer_result.error,
            latency_ms=round(elapsed_ms, 2),
            history_entries_used=0,
            source_dataset_id=defn.source_dataset_id,
            source_frame_index=src_rec.get("frame_index") if src_rec else None,
            source_timestamp_ns=src_rec.get("timestamp_ns") if src_rec else None,
        )
        results.append(fr)
        if artifact_dir is not None:
            jsonl_records.append(
                _frame_result_to_record(fr, defn, record_type="frame")
            )
        if on_frame is not None:
            on_frame(idx, fr)

    if artifact_dir is not None:
        _write_artifact_jsonl(artifact_dir, "experiment.jsonl", jsonl_records)
        _write_manifest(artifact_dir, defn, results)

    return results


def _run_observation_history(
    defn: ExperimentDefinition,
    *,
    cli_path: str,
    socket_path: str,
    artifact_dir: Optional[Path],
    inference_fn: InferenceFn,
    cancel_fn: Optional[Callable[[], bool]] = None,
    on_frame: Optional[Callable[[int, "FrameResult"], None]] = None,
) -> List[FrameResult]:
    """Strategy: single_frame_observation_history.

    Each frame is processed with an accumulated rolling window of previous
    model responses.  The history window is bounded by both entry count and
    total character budget.
    """
    max_entries = defn.observation_history_max_entries
    max_chars = defn.observation_history_max_chars
    history: List[str] = []
    results: List[FrameResult] = []
    jsonl_records: List[Dict[str, Any]] = []

    for idx, image_path in enumerate(defn.image_paths):
        if cancel_fn is not None and cancel_fn():
            break

        prompt = _build_prompt(
            defn.task_prompt,
            defn.system_instruction,
            history if max_entries > 0 else [],
            max_chars,
        )

        t0 = time.monotonic()
        infer_result = inference_fn(
            cli_path=cli_path,
            socket_path=socket_path,
            image_path=image_path,
            prompt=prompt,
            max_generate_length=defn.max_generate_length,
            temperature=defn.temperature,
            top_p=defn.top_p,
            top_k=defn.top_k,
            timeout_seconds=defn.timeout_seconds,
        )
        elapsed_ms = (time.monotonic() - t0) * 1000.0

        rep_flag = False
        if infer_result.success and infer_result.text and history:
            rep_flag = _compute_repetition(infer_result.text, history)

        src_rec = (
            defn.source_frame_records[idx]
            if defn.source_frame_records and idx < len(defn.source_frame_records)
            else None
        )
        fr = FrameResult(
            frame_index=idx,
            image_path=image_path,
            prompt_used=prompt,
            success=infer_result.success,
            text=infer_result.text,
            error=infer_result.error,
            latency_ms=round(elapsed_ms, 2),
            history_entries_used=len(history),
            repetition_flag=rep_flag,
            source_dataset_id=defn.source_dataset_id,
            source_frame_index=src_rec.get("frame_index") if src_rec else None,
            source_timestamp_ns=src_rec.get("timestamp_ns") if src_rec else None,
        )
        results.append(fr)
        if artifact_dir is not None:
            jsonl_records.append(
                _frame_result_to_record(fr, defn, record_type="frame")
            )
        if on_frame is not None:
            on_frame(idx, fr)

        # Update history only on success.
        if infer_result.success and infer_result.text:
            history.append(infer_result.text)
            if max_entries > 0:
                history = history[-max_entries:]

    if artifact_dir is not None:
        _write_artifact_jsonl(artifact_dir, "experiment.jsonl", jsonl_records)
        _write_manifest(artifact_dir, defn, results)

    return results


def _frame_result_to_record(
    fr: FrameResult,
    defn: ExperimentDefinition,
    record_type: str = "frame",
) -> Dict[str, Any]:
    rec: Dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "record_type": record_type,
        "experiment_id": defn.experiment_id,
        "frame_index": fr.frame_index,
        "image_path": fr.image_path,
        "strategy": defn.strategy,
        "history_entries_used": fr.history_entries_used,
        "success": fr.success,
        "text": fr.text,
        "error": fr.error,
        "latency_ms": fr.latency_ms,
        "repetition_flag": fr.repetition_flag,
        "timestamp": _now_iso(),
    }
    if fr.source_dataset_id is not None:
        rec["source_dataset_id"] = fr.source_dataset_id
    if fr.source_frame_index is not None:
        rec["source_frame_index"] = fr.source_frame_index
    if fr.source_timestamp_ns is not None:
        rec["source_timestamp_ns"] = fr.source_timestamp_ns
    return rec


def _write_manifest(
    artifact_dir: Path,
    defn: ExperimentDefinition,
    results: List[FrameResult],
) -> None:
    """Write a JSON manifest summarising the experiment run."""
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    latencies = [r.latency_ms for r in successful]

    summary: Dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "record_type": "experiment_manifest",
        "experiment_id": defn.experiment_id,
        "run_id": defn.run_id,
        "created_at": defn.created_at,
        "completed_at": _now_iso(),
        "strategy": defn.strategy,
        "image_count": len(defn.image_paths),
        "successful_frames": len(successful),
        "failed_frames": len(failed),
        "repetition_flags": sum(1 for r in results if r.repetition_flag),
        "mean_latency_ms": (
            round(sum(latencies) / len(latencies), 2) if latencies else None
        ),
        "min_latency_ms": round(min(latencies), 2) if latencies else None,
        "max_latency_ms": round(max(latencies), 2) if latencies else None,
        "definition": {
            "task_prompt": defn.task_prompt,
            "system_instruction": defn.system_instruction,
            "observation_history_max_entries": defn.observation_history_max_entries,
            "observation_history_max_chars": defn.observation_history_max_chars,
            "max_generate_length": defn.max_generate_length,
            "temperature": defn.temperature,
            "top_p": defn.top_p,
            "top_k": defn.top_k,
            "timeout_seconds": defn.timeout_seconds,
        },
        "notes": defn.notes,
    }
    if defn.source_dataset_id is not None:
        summary["source_dataset_id"] = defn.source_dataset_id
    if defn.profile_name is not None:
        summary["profile_name"] = defn.profile_name
    if defn.profile_version is not None:
        summary["profile_version"] = defn.profile_version
    if defn.profile_hash is not None:
        summary["profile_hash"] = defn.profile_hash

    tmp = artifact_dir / "_manifest.tmp"
    manifest_path = artifact_dir / "manifest.json"
    tmp.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    tmp.rename(manifest_path)


def compute_history_matrix(
    runs: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Compare experiment runs across different history depths.

    Accepts a sequence of experiment manifests (loaded from manifest.json) and
    returns a list of comparison rows sorted by observation_history_max_entries.

    This is the observation-history baseline matrix described in the issue.
    """
    rows = []
    for manifest in runs:
        defn = manifest.get("definition", {})
        row: Dict[str, Any] = {
            "experiment_id": manifest.get("experiment_id"),
            "strategy": manifest.get("strategy"),
            "history_depth": defn.get("observation_history_max_entries", 0),
            "image_count": manifest.get("image_count"),
            "successful_frames": manifest.get("successful_frames"),
            "failed_frames": manifest.get("failed_frames"),
            "repetition_flags": manifest.get("repetition_flags"),
            "mean_latency_ms": manifest.get("mean_latency_ms"),
            "min_latency_ms": manifest.get("min_latency_ms"),
            "max_latency_ms": manifest.get("max_latency_ms"),
            "task_prompt": defn.get("task_prompt"),
            "max_generate_length": defn.get("max_generate_length"),
            "temperature": defn.get("temperature"),
        }
        rows.append(row)
    return sorted(rows, key=lambda r: (r.get("history_depth") or 0))
