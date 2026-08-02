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
_DOWNLOADABLE_BAGS: List[Dict[str, Any]] = [
    {
        "key": "image-proc",
        "name": "Isaac ROS image_proc quickstart",
        "description": (
            "Raw RGB + camera_info topics; directly usable by the Cosmos/VLM node."
        ),
        "source": "NGC (nvidia/isaac/isaac_ros_image_proc_assets)",
        "content_types": "sensor_msgs/Image, sensor_msgs/CameraInfo",
        "raw_image_compatible": True,
        "compatibility_note": "",
    },
    {
        "key": "h264",
        "name": "Isaac ROS H.264 decoder quickstart",
        "description": (
            "Dual H.264 CompressedImage streams; requires the Isaac ROS decoder."
        ),
        "source": "NGC (nvidia/isaac/isaac_ros_h264_decoder_assets)",
        "content_types": "sensor_msgs/CompressedImage",
        "raw_image_compatible": False,
        "compatibility_note": "Requires Isaac ROS H.264 decoder",
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
    """Stable bag key used by browser APIs (and download key for placeholders)."""

    name: str
    source: str
    description: str

    asset_key: str = ""
    """Top-level downloadable asset key (first path component under rosbag root)."""

    bag_key: str = ""
    """Stable per-playable-bag identity."""

    display_name: str = ""
    """User-facing bag name including nested bag identity when needed."""

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
    topic_details: List[Dict[str, Any]] = field(default_factory=list)
    storage_identifier: str = ""
    content_types: str = ""

    # Compatibility metadata
    raw_image_compatible: bool = False
    """True when the bag has at least one sensor_msgs/Image (raw) topic and can
    be fed directly to the VLM image reasoner without a decoder."""
    compatibility_note: str = ""
    """Human-readable note explaining why the bag is not directly usable, e.g.
    ``"Requires Isaac ROS H.264 decoder"``."""

    # Downloadable metadata
    downloadable: bool = True
    download_source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        bag_key = self.bag_key or self.key
        asset_key = self.asset_key or self.key
        display_name = self.display_name or self.name
        return {
            "schema_version": _SCHEMA_VERSION,
            "kind": "rosbag",
            "key": bag_key,
            "asset_key": asset_key,
            "bag_key": bag_key,
            "name": display_name,
            "display_name": display_name,
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
            "topic_details": self.topic_details,
            "storage_identifier": self.storage_identifier,
            "content_types": self.content_types,
            "raw_image_compatible": self.raw_image_compatible,
            "compatibility_note": self.compatibility_note,
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

    # Extract duration (nanoseconds): flow style or block style.
    dur_match = re.search(r"duration:\s*\{[^}]*nanoseconds:\s*(\d+)", text)
    if not dur_match:
        dur_match = re.search(r"duration:\s*\n\s*nanoseconds:\s*(\d+)", text)
    if dur_match:
        result["duration_seconds"] = int(dur_match.group(1)) / 1e9

    storage_match = re.search(r"storage_identifier:\s*([^\s#]+)", text)
    if storage_match:
        result["storage_identifier"] = storage_match.group(1).strip().strip("'\"")

    topics: List[str] = []
    topic_types: Dict[str, str] = {}
    message_counts: Dict[str, int] = {}
    topic_details: List[Dict[str, Any]] = []

    topic_entries: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    current_indent = -1
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()

        inline = re.search(r"topic_metadata:\s*\{([^}]*)\}", stripped)
        if inline:
            mapping = inline.group(1)
            name_match = re.search(r"name:\s*([^,}]+)", mapping)
            type_match = re.search(r"type:\s*([^,}]+)", mapping)
            current = {
                "name": name_match.group(1).strip().strip("'\"") if name_match else "",
                "type": type_match.group(1).strip().strip("'\"") if type_match else "",
                "message_count": None,
            }
            current_indent = indent
            topic_entries.append(current)
            continue

        if stripped in ("topic_metadata:", "- topic_metadata:") or stripped.startswith("- topic_metadata:"):
            current = {"name": "", "type": "", "message_count": None}
            current_indent = indent
            topic_entries.append(current)
            continue

        if current is not None and indent > current_indent:
            if stripped.startswith("name:"):
                current["name"] = stripped.split(":", 1)[1].strip().strip("'\"")
                continue
            if stripped.startswith("type:"):
                current["type"] = stripped.split(":", 1)[1].strip().strip("'\"")
                continue

        if stripped.startswith("message_count:") and current is not None:
            val = stripped.split(":", 1)[1].strip()
            if val.isdigit():
                current["message_count"] = int(val)

    for entry in topic_entries:
        topic = str(entry.get("name", "")).strip()
        ttype = str(entry.get("type", "")).strip()
        if not topic or not ttype:
            continue
        if topic not in topics:
            topics.append(topic)
            topic_types[topic] = ttype
        msg_count = entry.get("message_count")
        if isinstance(msg_count, int):
            message_counts[topic] = msg_count

    for topic in topics:
        ttype = topic_types.get(topic, "")
        lowered = topic.lower()
        modality = "non_image"
        selectable = False
        if ttype == "sensor_msgs/msg/Image":
            selectable = True
            if "depth" in lowered:
                modality = "depth"
            elif "/ir" in lowered or "infra" in lowered:
                modality = "infrared"
            else:
                modality = "color"
        elif ttype == "sensor_msgs/msg/CompressedImage":
            modality = "compressed"
            selectable = False
        topic_details.append({
            "name": topic,
            "type": ttype,
            "message_count": message_counts.get(topic),
            "selectable": selectable,
            "modality": modality,
        })

    result["topics"] = topics
    result["topic_types"] = topic_types
    result["message_counts"] = message_counts
    result["topic_details"] = topic_details
    result["image_topics"] = [
        t["name"] for t in topic_details if t.get("selectable")
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


def _slug_token(token: str) -> str:
    """Return a URL/JSON-safe token segment."""
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", token).strip("-").lower() or "bag"


def _build_bag_key(asset_key: str, relative_parts: tuple, asset_bag_count: int) -> str:
    """Build a stable bag key from the root-relative path."""
    if asset_bag_count == 1:
        return asset_key
    nested = [_slug_token(p) for p in relative_parts[1:]]
    return ":".join([_slug_token(asset_key)] + nested)


def _scan_rosbags(rosbag_root: Path) -> List[RosbagEntry]:
    """Build installed entries with one playable entry per discovered metadata.yaml."""
    installed: List[RosbagEntry] = []
    discovered: List[Dict[str, Any]] = []
    asset_counts: Dict[str, int] = {}

    for bag_dir in _find_rosbag_dirs(rosbag_root):
        try:
            relative_parts = bag_dir.relative_to(rosbag_root).parts
        except ValueError:
            continue
        if not relative_parts:
            continue
        asset_key = relative_parts[0]
        meta = _parse_bag_metadata(bag_dir / _ROSBAG_METADATA_FILENAME)
        discovered.append({
            "asset_key": asset_key,
            "relative_parts": relative_parts,
            "bag_dir": bag_dir,
            "meta": meta,
        })
        asset_counts[asset_key] = asset_counts.get(asset_key, 0) + 1

    downloadable_name_map = {d["key"]: d["name"] for d in _DOWNLOADABLE_BAGS}
    for rec in discovered:
        asset_key = rec["asset_key"]
        relative_parts = rec["relative_parts"]
        bag_dir = rec["bag_dir"]
        meta = rec["meta"]
        bag_key = _build_bag_key(asset_key, relative_parts, asset_counts.get(asset_key, 1))
        topic_types: Dict[str, str] = meta.get("topic_types", {})
        topic_details: List[Dict[str, Any]] = meta.get("topic_details", [])
        raw_compatible = any(t.get("selectable") for t in topic_details)
        has_compressed = any(
            t.get("type") == "sensor_msgs/msg/CompressedImage" for t in topic_details
        )
        if raw_compatible:
            note = ""
        elif has_compressed:
            note = "CompressedImage requires decoding/materialization"
        else:
            note = "No raw sensor_msgs/msg/Image topic found"
        dataset_root = rosbag_root / asset_key
        display_asset = downloadable_name_map.get(asset_key, asset_key)
        display_suffix = relative_parts[-1]
        display_name = display_asset if asset_counts.get(asset_key, 1) == 1 else f"{display_asset} — {display_suffix}"
        entry = RosbagEntry(
            key=bag_key,
            asset_key=asset_key,
            bag_key=bag_key,
            name=display_name,
            display_name=display_name,
            source="local",
            description="Locally installed rosbag",
            installed=True,
            local_path=str(bag_dir),
            size_bytes=_dir_size_bytes(dataset_root),
            duration_seconds=meta.get("duration_seconds"),
            topics=meta.get("topics", []),
            topic_types=topic_types,
            message_counts=meta.get("message_counts", {}),
            image_topics=meta.get("image_topics", []),
            topic_details=topic_details,
            storage_identifier=meta.get("storage_identifier", ""),
            raw_image_compatible=raw_compatible,
            compatibility_note=note,
            downloadable=False,
            content_types=", ".join(sorted({t for t in topic_types.values() if t})),
        )
        installed.append(entry)
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
    installed_entries: List[RosbagEntry] = []
    if rosbag_root:
        installed_entries = _scan_rosbags(Path(rosbag_root))

    rosbag_entries: List[Dict[str, Any]] = []
    installed_by_asset: Dict[str, List[RosbagEntry]] = {}
    for e in installed_entries:
        installed_by_asset.setdefault(e.asset_key or e.key, []).append(e)

    # Start with downloadable definitions and overlay single-bag installs.
    for dinfo in _DOWNLOADABLE_BAGS:
        key = dinfo["key"]
        asset_entries = installed_by_asset.get(key, [])
        if len(asset_entries) == 1 and asset_entries[0].key == key:
            entry = asset_entries[0]
            entry.downloadable = True
            entry.download_source = dinfo["source"]
            entry.content_types = dinfo.get("content_types", "")
            # Prefer the explicit downloadable-catalog compatibility metadata
            # over the parsed-from-metadata.yaml value when available, so that
            # known assets like h264 always carry the correct note even if the
            # bag metadata was not parseable (e.g. first-time download).
            if "raw_image_compatible" in dinfo:
                entry.raw_image_compatible = bool(dinfo["raw_image_compatible"])
                entry.compatibility_note = dinfo.get("compatibility_note", "")
            rosbag_entries.append(entry.to_dict())
        else:
            if not asset_entries:
                entry = RosbagEntry(
                    key=key,
                    asset_key=key,
                    bag_key=key,
                    name=dinfo["name"],
                    display_name=dinfo["name"],
                    source=dinfo["source"],
                    description=dinfo["description"],
                    installed=False,
                    downloadable=True,
                    download_source=dinfo["source"],
                    content_types=dinfo.get("content_types", ""),
                    raw_image_compatible=bool(dinfo.get("raw_image_compatible", False)),
                    compatibility_note=dinfo.get("compatibility_note", ""),
                )
                rosbag_entries.append(entry.to_dict())

    # Append installed playable bags (multi-bag assets and non-downloadable assets).
    for entry in sorted(installed_entries, key=lambda e: e.key):
        if not any(e.get("key") == entry.key for e in rosbag_entries):
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
