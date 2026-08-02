# Copyright 2025 edge_vlm_ros contributors
"""
Rosbag frame extraction for the warehouse workbench.

Creates bounded, sampled-frame datasets from catalog-allowlisted rosbags.
The actual image conversion is delegated to ``scripts/extract_bag_frames.py``
via subprocess (argument list only, never ``shell=True``).

Catalog allowlisting
--------------------
Bag paths are never accepted directly from the browser.  Only bags whose
``local_path`` appears in the installed catalog are accepted.

Subprocess contract
-------------------
The extraction subprocess receives all parameters via command-line arguments
and writes:
  - ``<output_dir>/frame_dataset.json``  — machine-readable manifest
  - ``<output_dir>/frame_NNNN.jpg``      — JPEG images, one per extracted frame

The extraction run is tracked via the existing ``ProcessManager`` so it can
be cancelled and cleaned up normally.

Manifest schema
---------------
``frame_dataset.json``::

    {
      "schema_version": 1,
      "dataset_id": "<uuid>",
      "bag_key": "image-proc",
      "bag_path": "/path/to/bag",
      "topic": "/hawk_0_left_rgb_image",
      "start_offset_sec": 0.0,
      "end_offset_sec": null,
      "sample_interval_sec": 0.5,
      "max_frames": 100,
      "frames": [
        {
          "index": 0,
          "filename": "frame_0000.jpg",
          "timestamp_ns": 1234567890000000000,
          "timestamp_sec": 1234567890.0
        }
      ],
      "extracted_at": "2025-01-01T00:00:00+00:00",
      "bag_duration_sec": 10.0,
      "frame_count": 20
    }
"""
from __future__ import annotations

import datetime
import json
import os
import re
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .dataset_catalog import discover_datasets

# ── constants ─────────────────────────────────────────────────────────────────

_SCHEMA_VERSION = 1
_DEFAULT_MAX_FRAMES = 100
_HARD_MAX_FRAMES = 500
_MANIFEST_FILENAME = "frame_dataset.json"
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)

# Allowed image file extension for frame images served by the web console.
_FRAME_IMAGE_EXT = ".jpg"


# ── dataclasses ───────────────────────────────────────────────────────────────


@dataclass
class ExtractionParams:
    """User-supplied extraction parameters (validated before subprocess launch)."""

    bag_key: str
    """Catalog key for the rosbag (e.g. ``"image-proc"``)."""

    bag_path: str
    """Resolved, catalog-allowlisted absolute path to the rosbag."""

    image_topic: str
    """Image topic name (e.g. ``"/hawk_0_left_rgb_image"``)."""

    dataset_id: str
    """UUID for the output dataset directory."""

    output_dir: str
    """Absolute path to the output directory for this dataset."""

    start_offset: float = 0.0
    """Start time offset from the beginning of the bag in seconds."""

    end_offset: Optional[float] = None
    """End time offset; None means play to the end of the bag."""

    duration: Optional[float] = None
    """Play duration in seconds (alternative to end_offset)."""

    sample_interval: Optional[float] = None
    """Minimum time between sampled frames in seconds."""

    target_sample_count: Optional[int] = None
    """When set, derive sample_interval to target approximately this many frames."""

    max_frames: int = _DEFAULT_MAX_FRAMES
    """Hard upper bound on the number of extracted frames."""

    # Backwards-compatible aliases
    @property
    def topic(self) -> str:
        return self.image_topic

    @property
    def start_offset_sec(self) -> float:
        return self.start_offset

    @property
    def end_offset_sec(self) -> Optional[float]:
        return self.end_offset

    @property
    def sample_interval_sec(self) -> float:
        return self.sample_interval if self.sample_interval is not None else 0.5


@dataclass
class FrameDatasetManifest:
    """Loaded frame-dataset manifest."""

    schema_version: int
    dataset_id: str
    bag_key: str
    bag_path: str
    topic: str
    start_offset_sec: float
    end_offset_sec: Optional[float]
    sample_interval_sec: float
    max_frames: int
    frames: List[Dict[str, Any]]
    extracted_at: str
    frame_count: int
    output_dir: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "bag_key": self.bag_key,
            "bag_path": self.bag_path,
            "topic": self.topic,
            "start_offset_sec": self.start_offset_sec,
            "end_offset_sec": self.end_offset_sec,
            "sample_interval_sec": self.sample_interval_sec,
            "max_frames": self.max_frames,
            "frames": self.frames,
            "extracted_at": self.extracted_at,
            "frame_count": self.frame_count,
            "output_dir": self.output_dir,
        }


# ── validation ────────────────────────────────────────────────────────────────


def validate_extraction_params(body: Dict[str, Any]) -> Optional[str]:
    """Validate a raw extraction request body dict.  Returns an error string or None."""
    bag_key = body.get("bag_key", "")
    if not bag_key or not re.match(r"^[a-zA-Z0-9_-]{1,64}$", str(bag_key)):
        return "bag_key is required and must match [a-zA-Z0-9_-]{1,64}"
    image_topic = body.get("image_topic", "")
    if not image_topic or not str(image_topic).startswith("/"):
        return "image_topic is required and must start with '/'"
    start_offset = body.get("start_offset", 0.0)
    try:
        start_offset = float(start_offset)
    except (TypeError, ValueError):
        return "start_offset must be a number"
    if start_offset < 0.0:
        return "start_offset must be >= 0"
    max_frames = body.get("max_frames", _DEFAULT_MAX_FRAMES)
    try:
        max_frames = int(max_frames)
    except (TypeError, ValueError):
        return "max_frames must be an integer"
    if not (1 <= max_frames <= _HARD_MAX_FRAMES):
        return f"max_frames must be in [1, {_HARD_MAX_FRAMES}]"
    return None


def _validate_params_object(params: ExtractionParams) -> Optional[str]:
    """Validate an ExtractionParams dataclass instance.  Returns an error string or None."""
    if not re.match(r"^[a-zA-Z0-9_-]{1,64}$", params.bag_key):
        return "Invalid bag_key"
    if not params.image_topic or not params.image_topic.startswith("/"):
        return "image_topic must be a non-empty ROS topic path starting with '/'"
    if params.start_offset < 0.0:
        return "start_offset must be >= 0"
    if params.end_offset is not None:
        if params.end_offset <= params.start_offset:
            return "end_offset must be > start_offset"
    if not (1 <= params.max_frames <= _HARD_MAX_FRAMES):
        return f"max_frames must be in [1, {_HARD_MAX_FRAMES}]"
    return None


def allowlist_bag_path(bag_key: str, catalog: Dict[str, Any]) -> str:
    """Return the catalog-resolved bag path for *bag_key*.

    Only paths returned by ``discover_datasets`` are allowed.  Arbitrary
    filesystem paths are never accepted.

    Parameters
    ----------
    bag_key:
        Catalog key (e.g. ``"image-proc"``).
    catalog:
        Pre-loaded catalog dict from ``discover_datasets()``.

    Raises
    ------
    ValueError
        When the key is not in the catalog or the bag is not installed.
    """
    for bag in catalog.get("rosbags", []):
        if bag.get("key") == bag_key and bag.get("installed") and bag.get("local_path"):
            return bag["local_path"]
    raise ValueError(f"Bag {bag_key!r} is not an installed catalog entry")


# ── subprocess construction ───────────────────────────────────────────────────


def build_extraction_args(
    script_path: str,
    params: ExtractionParams,
) -> List[str]:
    """Return the argument array for the extraction subprocess.

    Never uses shell interpolation; the caller must pass ``shell=False``.
    """
    args = [
        "python3",
        script_path,
        "--bag-path", params.bag_path,
        "--output-dir", params.output_dir,
        "--topic", params.image_topic,
        "--start-offset", str(params.start_offset),
        "--max-frames", str(params.max_frames),
        "--dataset-id", params.dataset_id,
        "--bag-key", params.bag_key,
    ]
    if params.end_offset is not None:
        args += ["--end-offset", str(params.end_offset)]
    if params.duration is not None:
        args += ["--duration", str(params.duration)]
    if params.sample_interval is not None:
        args += ["--sample-interval", str(params.sample_interval)]
    if params.target_sample_count is not None:
        args += ["--target-count", str(params.target_sample_count)]
    return args


# ── manifest I/O ──────────────────────────────────────────────────────────────


def _is_safe_dataset_id(dataset_id: str) -> bool:
    """Return True only for UUID-shaped dataset IDs."""
    return bool(_UUID_RE.match(dataset_id))


def load_frame_manifest(manifest_path: Path) -> Optional[FrameDatasetManifest]:
    """Load a frame dataset manifest from disk.  Returns None on any error."""
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(data, dict):
        return None

    dataset_id = str(data.get("dataset_id", ""))
    if not _is_safe_dataset_id(dataset_id):
        return None

    # Validate that all frame filenames are safe (basename only, .jpg extension).
    frames = data.get("frames", [])
    if not isinstance(frames, list):
        return None

    safe_frames: List[Dict[str, Any]] = []
    for fr in frames:
        if not isinstance(fr, dict):
            continue
        fn = str(fr.get("filename", ""))
        # Only allow plain filenames (no path components, correct extension).
        if "/" in fn or "\\" in fn or not fn.endswith(_FRAME_IMAGE_EXT):
            continue
        if not re.match(r"^frame_\d{4,6}\.jpg$", fn):
            continue
        safe_frames.append({
            "index": int(fr.get("index", 0)),
            "filename": fn,
            "timestamp_ns": int(fr.get("timestamp_ns", 0)),
            "timestamp_sec": float(fr.get("timestamp_sec", 0.0)),
        })

    # Ensure frames are ordered by index.
    safe_frames.sort(key=lambda f: f["index"])

    return FrameDatasetManifest(
        schema_version=int(data.get("schema_version", _SCHEMA_VERSION)),
        dataset_id=dataset_id,
        bag_key=str(data.get("bag_key", "")),
        bag_path=str(data.get("bag_path", "")),
        topic=str(data.get("topic", "")),
        start_offset_sec=float(data.get("start_offset_sec", 0.0)),
        end_offset_sec=data.get("end_offset_sec"),
        sample_interval_sec=float(data.get("sample_interval_sec", 0.5)),
        max_frames=int(data.get("max_frames", _DEFAULT_MAX_FRAMES)),
        frames=safe_frames,
        extracted_at=str(data.get("extracted_at", "")),
        frame_count=int(data.get("frame_count", len(safe_frames))),
        output_dir=str(data.get("output_dir", str(manifest_path.parent))),
    )


def write_frame_manifest(output_dir: Path, manifest: Dict[str, Any]) -> Path:
    """Write a frame dataset manifest atomically.  Returns the manifest path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / _MANIFEST_FILENAME
    tmp = output_dir / f"{_MANIFEST_FILENAME}.tmp"
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(str(tmp), str(path))
    return path


# ── frame dataset store ───────────────────────────────────────────────────────


class FrameDatasetStore:
    """Manages frame dataset directories under a base directory.

    Datasets are stored as::

        <base_dir>/<dataset_id>/frame_dataset.json
        <base_dir>/<dataset_id>/frame_0000.jpg
        ...
    """

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def dataset_dir(self, dataset_id: str) -> Optional[Path]:
        """Return the directory for *dataset_id*, or None if it does not exist."""
        if not _is_safe_dataset_id(dataset_id):
            return None
        d = self.base_dir / dataset_id
        return d if d.is_dir() else None

    def get_manifest(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Return the manifest dict for *dataset_id*, or None."""
        d = self.dataset_dir(dataset_id)
        if d is None:
            return None
        manifest = load_frame_manifest(d / _MANIFEST_FILENAME)
        return manifest.to_dict() if manifest is not None else None

    def get_frame_path(self, dataset_id: str, frame_index: int) -> Optional[Path]:
        """Return the absolute path to a frame image, or None if not found or unsafe.

        Path is always within ``<base_dir>/<dataset_id>/``.
        """
        if frame_index < 0:
            return None
        d = self.dataset_dir(dataset_id)
        if d is None:
            return None
        manifest_data = self.get_manifest(dataset_id)
        if manifest_data is None:
            return None
        # Find the frame record for the given index.
        frames = manifest_data.get("frames", [])
        frame_rec = next(
            (f for f in frames if f.get("index") == frame_index), None
        )
        if frame_rec is None:
            return None
        filename = str(frame_rec.get("filename", ""))
        # Double-check safety: filename must still be a plain basename.
        if "/" in filename or "\\" in filename or not filename.endswith(_FRAME_IMAGE_EXT):
            return None
        candidate = d / filename
        # Final containment check: resolved path must be inside dataset dir.
        try:
            candidate.resolve().relative_to(d.resolve())
        except ValueError:
            return None
        return candidate if candidate.is_file() else None

    def list_datasets(self) -> List[Dict[str, Any]]:
        """Return a list of all dataset manifest summaries (newest first)."""
        datasets: List[Dict[str, Any]] = []
        try:
            dirs = sorted(
                self.base_dir.iterdir(),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        except OSError:
            return datasets
        for d in dirs:
            if not d.is_dir() or not _is_safe_dataset_id(d.name):
                continue
            manifest = self.get_manifest(d.name)
            if manifest is not None:
                datasets.append(manifest)  # already a dict
        return datasets

    def prepare_output_dir(self, dataset_id: str) -> Path:
        """Create and return the output directory for a new extraction."""
        if not _is_safe_dataset_id(dataset_id):
            raise ValueError(f"Invalid dataset_id: {dataset_id!r}")
        d = self.base_dir / dataset_id
        d.mkdir(parents=True, exist_ok=True)
        return d
