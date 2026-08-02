# Copyright 2025 edge_vlm_ros contributors
"""
Dataset and rosbag catalog for the web workbench.

Discovers locally-installed datasets (rosbags, image directories, video files)
and the known set of downloadable NVIDIA Isaac ROS assets.  All information is
read-only; download actions are delegated to the existing
``scripts/test_data/download_rosbags.sh`` script via the web-console API.

No second downloader is implemented.  Arbitrary download URLs are not accepted.
"""
from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── constants ─────────────────────────────────────────────────────────────────

# Known downloadable rosbag assets registered in download_rosbags.sh.
# This list mirrors the case statement in that script.
_DOWNLOADABLE_BAGS: List[Dict[str, str]] = [
    {
        "key": "image-proc",
        "name": "Isaac ROS image_proc quickstart",
        "description": (
            "Raw RGB + camera_info topics; directly usable by the Cosmos/VLM node."
        ),
        "source": "NGC (nvidia/isaac/isaac_ros_image_proc_assets)",
        "content_types": "sensor_msgs/Image, sensor_msgs/CameraInfo",
    },
    {
        "key": "h264",
        "name": "Isaac ROS H.264 decoder quickstart",
        "description": (
            "Dual H.264 CompressedImage streams; requires the Isaac ROS decoder."
        ),
        "source": "NGC (nvidia/isaac/isaac_ros_h264_decoder_assets)",
        "content_types": "sensor_msgs/CompressedImage",
    },
]

# Rosbag directory extensions that ros2 bag produces.
_ROSBAG_METADATA_FILENAME = "metadata.yaml"
# Image file extensions to scan in image directories.
_IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
)
# Video file extensions.
_VIDEO_EXTENSIONS = frozenset(
    {".mp4", ".avi", ".mkv", ".mov", ".webm", ".ts", ".m2v"}
)

_SCHEMA_VERSION = 1


# ── dataclasses ───────────────────────────────────────────────────────────────


@dataclass
class RosbagEntry:
    """Description of a rosbag (installed or downloadable)."""

    key: str
    """Stable download key used as the argument to download_rosbags.sh."""

    name: str
    source: str
    description: str

    # Installed state
    installed: bool = False
    local_path: str = ""
    size_bytes: int = 0

    # Bag content metadata (populated when metadata.yaml is parseable)
    duration_seconds: Optional[float] = None
    topics: List[str] = field(default_factory=list)
    topic_types: Dict[str, str] = field(default_factory=dict)
    message_counts: Dict[str, int] = field(default_factory=dict)
    image_topics: List[str] = field(default_factory=list)
    content_types: str = ""

    # Downloadable metadata
    downloadable: bool = True
    download_source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "kind": "rosbag",
            "key": self.key,
            "name": self.name,
            "source": self.source,
            "description": self.description,
            "installed": self.installed,
            "local_path": self.local_path,
            "size_bytes": self.size_bytes,
            "duration_seconds": self.duration_seconds,
            "topics": self.topics,
            "topic_types": self.topic_types,
            "message_counts": self.message_counts,
            "image_topics": self.image_topics,
            "content_types": self.content_types,
            "downloadable": self.downloadable,
            "download_source": self.download_source,
        }


@dataclass
class ImageDatasetEntry:
    """Description of a local image directory dataset."""

    name: str
    local_path: str
    image_count: int
    extensions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "kind": "images",
            "name": self.name,
            "local_path": self.local_path,
            "image_count": self.image_count,
            "extensions": self.extensions,
        }


@dataclass
class VideoDatasetEntry:
    """Description of a local video file dataset."""

    name: str
    local_path: str
    size_bytes: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "kind": "video",
            "name": self.name,
            "local_path": self.local_path,
            "size_bytes": self.size_bytes,
        }


# ── helpers ───────────────────────────────────────────────────────────────────


def _dir_size_bytes(path: Path) -> int:
    """Return the total byte size of all files under *path* (non-recursive slow path)."""
    total = 0
    try:
        for entry in path.rglob("*"):
            if entry.is_file():
                try:
                    total += entry.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total


def _parse_bag_metadata(metadata_path: Path) -> Dict[str, Any]:
    """Parse a rosbag2 metadata.yaml for the fields we display.

    Uses plain text parsing to avoid a PyYAML dependency; extracts only the
    fields needed by the UI.  Returns an empty dict on any parse error.
    """
    result: Dict[str, Any] = {}
    try:
        text = metadata_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return result

    # Extract duration (nanoseconds)
    dur_match = re.search(r"duration:\s*\{nanoseconds:\s*(\d+)", text)
    if dur_match:
        result["duration_seconds"] = int(dur_match.group(1)) / 1e9

    # Extract topic names and types
    topics: List[str] = []
    topic_types: Dict[str, str] = {}
    message_counts: Dict[str, int] = {}
    # Pattern: - topic_metadata: {name: /topic, type: sensor_msgs/msg/Image, ...}
    for m in re.finditer(
        r"topic_metadata:\s*\{name:\s*([^,}]+),\s*type:\s*([^,}]+)", text
    ):
        topic = m.group(1).strip()
        ttype = m.group(2).strip()
        if topic not in topics:
            topics.append(topic)
            topic_types[topic] = ttype

    # message_count per topic
    for m in re.finditer(r"topic:\s*([^\s]+).*?message_count:\s*(\d+)", text, re.DOTALL):
        pass  # complex — skip; not critical for MVP

    result["topics"] = topics
    result["topic_types"] = topic_types
    result["message_counts"] = message_counts
    result["image_topics"] = [
        t for t, typ in topic_types.items() if "Image" in typ
    ]
    return result


def _find_rosbag_dirs(root: Path) -> List[Path]:
    """Recursively find directories that contain a rosbag2 metadata.yaml.

    Isaac ROS archives commonly nest the actual bag several levels below the
    catalog key, such as h264/isaac_ros_h264_decoder/quickstart.
    """
    if not root.is_dir():
        return []
    try:
        return sorted(
            metadata.parent
            for metadata in root.rglob(_ROSBAG_METADATA_FILENAME)
            if metadata.is_file()
        )
    except OSError:
        return []


def _scan_rosbags(rosbag_root: Path) -> Dict[str, RosbagEntry]:
    """Build installed entries keyed by the dataset top-level catalog key.

    The playable bag path remains the directory containing metadata.yaml. The
    first path component relative to rosbag_root is used as the key so nested
    NVIDIA assets merge with their matching downloadable definitions.
    """
    installed: Dict[str, RosbagEntry] = {}
    for bag_dir in _find_rosbag_dirs(rosbag_root):
        try:
            relative_parts = bag_dir.relative_to(rosbag_root).parts
        except ValueError:
            continue
        if not relative_parts:
            continue
        key = relative_parts[0]
        meta = _parse_bag_metadata(bag_dir / _ROSBAG_METADATA_FILENAME)
        dataset_root = rosbag_root / key
        entry = RosbagEntry(
            key=key,
            name=key,
            source="local",
            description="Locally installed rosbag",
            installed=True,
            local_path=str(bag_dir),
            size_bytes=_dir_size_bytes(dataset_root),
            duration_seconds=meta.get("duration_seconds"),
            topics=meta.get("topics", []),
            topic_types=meta.get("topic_types", {}),
            message_counts=meta.get("message_counts", {}),
            image_topics=meta.get("image_topics", []),
            downloadable=False,
        )
        # A catalog key currently represents one playable bag. Keep the first
        # deterministic match if an archive contains multiple metadata files.
        installed.setdefault(key, entry)
    return installed


def _scan_image_dirs(image_root: Path) -> List[ImageDatasetEntry]:
    """Discover image-directory datasets under image_root."""
    entries: List[ImageDatasetEntry] = []
    if not image_root.is_dir():
        return entries
    try:
        for d in sorted(image_root.iterdir()):
            if not d.is_dir():
                continue
            images = [
                f for f in d.iterdir()
                if f.is_file() and f.suffix.lower() in _IMAGE_EXTENSIONS
            ]
            if not images:
                continue
            exts = sorted({f.suffix.lower() for f in images})
            entries.append(
                ImageDatasetEntry(
                    name=d.name,
                    local_path=str(d),
                    image_count=len(images),
                    extensions=exts,
                )
            )
    except OSError:
        pass
    return entries


def _scan_videos(video_root: Path) -> List[VideoDatasetEntry]:
    """Discover video files under video_root."""
    entries: List[VideoDatasetEntry] = []
    if not video_root.is_dir():
        return entries
    try:
        for f in sorted(video_root.rglob("*")):
            if f.is_file() and f.suffix.lower() in _VIDEO_EXTENSIONS:
                try:
                    size = f.stat().st_size
                except OSError:
                    size = 0
                entries.append(
                    VideoDatasetEntry(
                        name=f.name,
                        local_path=str(f),
                        size_bytes=size,
                    )
                )
    except OSError:
        pass
    return entries


# ── download construction ─────────────────────────────────────────────────────


def build_download_command(
    download_script: str,
    bag_key: str,
) -> Optional[List[str]]:
    """Construct the argument array for downloading a rosbag via download_rosbags.sh.

    Only keys registered in _DOWNLOADABLE_BAGS are accepted; unknown keys return
    None (no arbitrary-URL or path-traversal risk).

    Returns a list suitable for ``subprocess.run(..., shell=False)`` or None
    when bag_key is not downloadable.
    """
    known_keys = {d["key"] for d in _DOWNLOADABLE_BAGS}
    if bag_key not in known_keys:
        return None
    # Argument array only — never shell=True.
    return ["bash", download_script, "download", bag_key]


# ── public API ────────────────────────────────────────────────────────────────


def discover_datasets(
    rosbag_root: Optional[str] = None,
    image_root: Optional[str] = None,
    video_root: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a catalogue of all discoverable datasets.

    Parameters
    ----------
    rosbag_root:
        Root directory to scan for rosbag2 bag directories.
        Defaults to the ``ROSBAG_DIR`` environment variable or the
        ``test_data/rosbags/`` directory in the repository root.
    image_root:
        Directory to scan for image-directory datasets.
        Defaults to ``IMAGE_DATASET_DIR`` env var or None (skipped).
    video_root:
        Directory to scan for video files.
        Defaults to ``VIDEO_DATASET_DIR`` env var or None (skipped).

    Returns a dict with keys:
      ``rosbags``       — list of RosbagEntry dicts (installed + downloadable)
      ``image_datasets`` — list of ImageDatasetEntry dicts
      ``video_datasets`` — list of VideoDatasetEntry dicts
    """
    # ── resolve rosbag root ───────────────────────────────────────────────────
    if rosbag_root is None:
        rosbag_root = os.environ.get("ROSBAG_DIR", "")
    if not rosbag_root:
        # Try the conventional test_data/rosbags path relative to repo root.
        here = Path(__file__).parent.parent
        candidate = here / "test_data" / "rosbags"
        if candidate.is_dir():
            rosbag_root = str(candidate)

    # ── merge installed + downloadable bags ──────────────────────────────────
    installed_map: Dict[str, RosbagEntry] = {}
    if rosbag_root:
        installed_map = _scan_rosbags(Path(rosbag_root))

    rosbag_entries: List[Dict[str, Any]] = []

    # Start with downloadable definitions and overlay installed state.
    seen_keys: set = set()
    for dinfo in _DOWNLOADABLE_BAGS:
        key = dinfo["key"]
        if key in installed_map:
            entry = installed_map[key]
            entry.downloadable = True
            entry.download_source = dinfo["source"]
            entry.content_types = dinfo.get("content_types", "")
        else:
            entry = RosbagEntry(
                key=key,
                name=dinfo["name"],
                source=dinfo["source"],
                description=dinfo["description"],
                installed=False,
                downloadable=True,
                download_source=dinfo["source"],
                content_types=dinfo.get("content_types", ""),
            )
        rosbag_entries.append(entry.to_dict())
        seen_keys.add(key)

    # Append any extra locally-installed bags not in the downloadable list.
    for key, entry in sorted(installed_map.items()):
        if key not in seen_keys:
            rosbag_entries.append(entry.to_dict())

    # ── image datasets ────────────────────────────────────────────────────────
    if image_root is None:
        image_root = os.environ.get("IMAGE_DATASET_DIR", "")
    image_entries = _scan_image_dirs(Path(image_root)) if image_root else []

    # ── video datasets ────────────────────────────────────────────────────────
    if video_root is None:
        video_root = os.environ.get("VIDEO_DATASET_DIR", "")
    video_entries = _scan_videos(Path(video_root)) if video_root else []

    return {
        "rosbags": rosbag_entries,
        "image_datasets": [e.to_dict() for e in image_entries],
        "video_datasets": [e.to_dict() for e in video_entries],
    }
