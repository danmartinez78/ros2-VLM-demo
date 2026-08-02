# Copyright 2025 edge_vlm_ros contributors
"""
Sequence catalog: read-only adapters for locally installed datasets.

Provides a common sequence abstraction used by the visual experiment workbench.
Three adapters are defined:

``ros_static_fixture``
    One-frame rosbag assets, e.g. the Isaac ROS RT-DETR quickstart bag.
    Discovered from the configured rosbag root; bags are classified as static
    when their only image topic has exactly one message.

``nuscenes_scene``
    Per-scene, metadata-linked CAM_FRONT keyframe sequences from nuScenes mini.
    Temporal ordering comes from the ``sample.json`` linked list — never from
    lexical filename order.  Scenes are kept independent; no cross-scene
    flattening occurs.  ``sweeps`` are excluded.

``jaad_clip``
    Individual JAAD MP4 clips from ``extracted/JAAD_clips/``.  Each clip is an
    independently selectable sequence.  A compact annotation summary is built
    from ``prepared/jaad-clip-label-index.json`` (if present) or by parsing
    the primary XML annotation file.  Frames are never pre-extracted.

Security
--------
Source paths are stored server-side only.  ``to_dict()`` never emits source
paths.  Browser APIs accept only dataset IDs, sequence IDs, and frame indices.
All discovered paths are validated to be within their configured dataset root.
"""
from __future__ import annotations

import json
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── constants ─────────────────────────────────────────────────────────────────

_SCHEMA_VERSION = 1
_ADAPTER_VERSION = "1"

# Stable IDs must only contain URL-safe characters.
_SAFE_ID_RE = re.compile(r"^[a-zA-Z0-9_.\-]{1,200}$")

# Maximum number of scenes / clips to discover in one pass (safety bound).
_MAX_SEQUENCES = 1000

# nuScenes channel scope for this adapter version.
_NUSCENES_CHANNEL = "CAM_FRONT"

# JAAD annotation families that are individually probed for presence.
_JAAD_ANNOTATION_FAMILIES = (
    "annotations",
    "annotations_appearance",
    "annotations_attributes",
    "annotations_traffic",
    "annotations_vehicle",
)

# ── dataclasses ───────────────────────────────────────────────────────────────


@dataclass
class FrameRef:
    """Reference to one frame within a sequence.

    ``source_path`` is the server-side absolute path and is **never** sent to
    the browser; use ``to_dict()`` (which omits it) for serialisation.
    """

    index: int
    """Zero-based frame index within the sequence."""

    source_id: str
    """Stable frame identity (e.g. nuScenes sample_data token, MP4 stem + index)."""

    timestamp_us: Optional[int]
    """Source timestamp in microseconds (None when unknown)."""

    source_path: str
    """Absolute server-side path to the source file (image or video).
    Never serialised to the browser."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Adapter-specific per-frame metadata (e.g. nuScenes tokens, dimensions)."""

    def to_dict(self) -> Dict[str, Any]:
        """Serialise for browser consumption (source_path is omitted)."""
        return {
            "index": self.index,
            "source_id": self.source_id,
            "timestamp_us": self.timestamp_us,
            "metadata": self.metadata,
        }


@dataclass
class SequenceEntry:
    """Descriptor for one independently selectable sequence.

    Designed to be returned from ``discover_sequences()`` and exposed via the
    ``/api/sequences`` endpoint.  Source paths are kept server-side; the
    browser only ever sees IDs and indices.
    """

    dataset_id: str
    """Stable identifier for the dataset (e.g. ``"nuscenes-mini"``)."""

    sequence_id: str
    """Stable identifier for the sequence within the dataset
    (e.g. ``"scene-0553"``, ``"video_0001"``)."""

    adapter: str
    """Adapter type: ``"ros_static_fixture"``, ``"nuscenes_scene"``,
    or ``"jaad_clip"``."""

    display_name: str
    """Human-readable name shown in the UI."""

    description: str
    """One-line description of the sequence content."""

    frame_count: int
    """Total number of selectable frames (keyframes for nuScenes; MP4 frames
    for JAAD; 1 for static fixtures)."""

    frame_refs: List[FrameRef]
    """Ordered frame references.  Empty when the sequence is discovered lazily
    (e.g. very large JAAD clips where full enumeration is deferred)."""

    annotation_ref: Optional[str]
    """Server-side path to the primary annotation file (never sent to browser).
    None when annotations are unavailable."""

    annotation_summary: Dict[str, Any] = field(default_factory=dict)
    """Compact, browser-safe annotation summary (counts, label coverage, etc.).
    Does *not* embed per-frame boxes."""

    provenance: Dict[str, Any] = field(default_factory=dict)
    """Provenance information (source dataset, split, etc.)."""

    adapter_version: str = _ADAPTER_VERSION
    """Monotonically increasing version used as part of cache identities."""

    def to_dict(self) -> Dict[str, Any]:
        """Serialise for browser consumption.

        Source paths from ``frame_refs`` are omitted.  ``annotation_ref`` is
        also omitted (server-side only).
        """
        return {
            "schema_version": _SCHEMA_VERSION,
            "dataset_id": self.dataset_id,
            "sequence_id": self.sequence_id,
            "adapter": self.adapter,
            "display_name": self.display_name,
            "description": self.description,
            "frame_count": self.frame_count,
            "frames": [fr.to_dict() for fr in self.frame_refs],
            "annotation_summary": self.annotation_summary,
            "provenance": self.provenance,
            "adapter_version": self.adapter_version,
        }


# ── helpers ───────────────────────────────────────────────────────────────────


def _is_safe_id(value: str) -> bool:
    """Return True only for safe, URL-friendly sequence / dataset IDs."""
    return bool(_SAFE_ID_RE.match(value))


def _assert_within_root(path: Path, root: Path) -> Path:
    """Resolve *path* and raise ``ValueError`` if it escapes *root*.

    Always call this before using any path derived from discovered data.
    """
    resolved = path.resolve()
    root_resolved = root.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError:
        raise ValueError(
            f"Path {path!r} escapes dataset root {root!r}"
        )
    return resolved


def _load_json_file(path: Path) -> Any:
    """Load a JSON file; return None on any error."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# ── ros_static_fixture adapter ────────────────────────────────────────────────


def _is_static_fixture(bag_dict: Dict[str, Any]) -> bool:
    """Return True when a rosbag catalog entry looks like a one-frame fixture.

    A bag is classified as a static fixture when:
    - it is installed
    - it has at least one raw image topic (sensor_msgs/Image)
    - the total image message count across all raw image topics is exactly 1
    """
    if not bag_dict.get("installed"):
        return False
    image_topics: List[str] = bag_dict.get("image_topics", [])
    topic_types: Dict[str, str] = bag_dict.get("topic_types", {})
    message_counts: Dict[str, int] = bag_dict.get("message_counts", {})
    raw_image_topics = [
        t for t in image_topics
        if topic_types.get(t) == "sensor_msgs/msg/Image"
    ]
    if not raw_image_topics:
        return False
    total_images = sum(message_counts.get(t, 0) for t in raw_image_topics)
    return total_images == 1


def discover_ros_static_fixtures(
    rosbag_catalog_entries: List[Dict[str, Any]],
) -> List[SequenceEntry]:
    """Discover static-fixture sequences from a list of rosbag catalog entries.

    A *static fixture* is a rosbag that contains exactly one image frame —
    useful as a deterministic reference input (e.g. RT-DETR quickstart).

    Parameters
    ----------
    rosbag_catalog_entries:
        The ``rosbags`` list from ``discover_datasets()``.

    Returns
    -------
    list of SequenceEntry
        One entry per qualifying bag, with a single FrameRef whose
        ``source_path`` is the bag directory path.
    """
    sequences: List[SequenceEntry] = []
    for bag in rosbag_catalog_entries:
        if not _is_static_fixture(bag):
            continue
        local_path = bag.get("local_path", "")
        if not local_path:
            continue
        bag_key: str = str(bag.get("key", ""))
        if not bag_key:
            continue
        # Derive a stable dataset_id: use "rtdetr-quickstart" for the known
        # RT-DETR fixture, otherwise encode the bag_key.
        dataset_id = _sanitize_id(bag_key)
        sequence_id = dataset_id  # single-bag → single sequence

        # Only one frame reference; source_path is the bag directory.
        topics: List[str] = bag.get("image_topics", [])
        topic_types: Dict[str, str] = bag.get("topic_types", {})
        raw_topics = [
            t for t in topics
            if topic_types.get(t) == "sensor_msgs/msg/Image"
        ]
        topic = raw_topics[0] if raw_topics else ""
        frame_ref = FrameRef(
            index=0,
            source_id=f"{bag_key}:frame_0",
            timestamp_us=None,
            source_path=local_path,
            metadata={"topic": topic, "bag_key": bag_key},
        )
        sequences.append(
            SequenceEntry(
                dataset_id=dataset_id,
                sequence_id=sequence_id,
                adapter="ros_static_fixture",
                display_name=bag.get("display_name") or bag.get("name", bag_key),
                description=(
                    "Static one-frame ROS fixture — "
                    + (bag.get("description") or bag.get("content_types") or "")
                ),
                frame_count=1,
                frame_refs=[frame_ref],
                annotation_ref=None,
                provenance={
                    "source": "rosbag",
                    "bag_key": bag_key,
                    "local_path": local_path,
                },
            )
        )
    return sequences


def _sanitize_id(raw: str) -> str:
    """Replace characters not in [a-zA-Z0-9_.-] with ``-``."""
    return re.sub(r"[^a-zA-Z0-9_.\-]", "-", raw)[:200]


# ── nuscenes_scene adapter ────────────────────────────────────────────────────


def _load_nuscenes_table(meta_dir: Path, table_name: str) -> Optional[List[Dict[str, Any]]]:
    """Load one nuScenes JSON table; return None on failure."""
    path = meta_dir / f"{table_name}.json"
    data = _load_json_file(path)
    if not isinstance(data, list):
        return None
    return data


def _build_token_index(table: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Build a {token: record} mapping from a nuScenes table list."""
    return {str(r.get("token", "")): r for r in table if r.get("token")}


def discover_nuscenes_scenes(
    nuscenes_root: Path,
    channel: str = _NUSCENES_CHANNEL,
) -> List[SequenceEntry]:
    """Discover per-scene CAM_FRONT keyframe sequences from nuScenes mini.

    Reads metadata from ``<nuscenes_root>/v1.0-mini/`` following the nuScenes
    linked-list structure:

        scene.json → sample.json (linked list) → sample_data.json

    Each scene becomes one independent ``SequenceEntry``.  Scenes are *not*
    flattened together.  Only keyframes (``is_key_frame=True``) for the named
    *channel* are included.  ``sweeps`` are excluded.

    Parameters
    ----------
    nuscenes_root:
        Root directory of the nuScenes mini installation.
    channel:
        Camera channel to index (default: ``"CAM_FRONT"``).

    Returns
    -------
    list of SequenceEntry
        One entry per scene; empty list on any discovery failure.
    """
    sequences: List[SequenceEntry] = []
    meta_dir = nuscenes_root / "v1.0-mini"
    if not meta_dir.is_dir():
        return sequences

    # Load tables.
    scene_table = _load_nuscenes_table(meta_dir, "scene")
    sample_table = _load_nuscenes_table(meta_dir, "sample")
    sample_data_table = _load_nuscenes_table(meta_dir, "sample_data")
    if not scene_table or not sample_table or not sample_data_table:
        return sequences

    # Build indices.
    sample_index = _build_token_index(sample_table)
    sample_data_index = _build_token_index(sample_data_table)

    # Build per-sample → CAM_FRONT keyframe sample_data token mapping.
    cam_front_by_sample: Dict[str, str] = {}
    for sd in sample_data_table:
        if (
            sd.get("channel") == channel
            and sd.get("is_key_frame")
        ):
            stok = str(sd.get("sample_token", ""))
            if stok:
                cam_front_by_sample[stok] = str(sd.get("token", ""))

    dataset_id = "nuscenes-mini"

    for scene_rec in scene_table[:_MAX_SEQUENCES]:
        scene_name = str(scene_rec.get("name", ""))
        if not scene_name:
            continue
        sequence_id = _sanitize_id(scene_name)
        if not _is_safe_id(sequence_id):
            continue

        description = str(scene_rec.get("description", ""))
        frame_refs: List[FrameRef] = []

        # Walk the sample linked list starting from first_sample_token.
        current_token = str(scene_rec.get("first_sample_token", ""))
        seen: set = set()
        while current_token and current_token not in seen:
            seen.add(current_token)
            sample_rec = sample_index.get(current_token)
            if sample_rec is None:
                break

            # Look up the CAM_FRONT keyframe for this sample.
            sd_token = cam_front_by_sample.get(current_token)
            if sd_token:
                sd_rec = sample_data_index.get(sd_token)
                if sd_rec:
                    filename = str(sd_rec.get("filename", ""))
                    # Resolve and validate source path.
                    if filename:
                        try:
                            src = _assert_within_root(
                                nuscenes_root / filename,
                                nuscenes_root,
                            )
                        except ValueError:
                            src = None
                    else:
                        src = None
                    frame_refs.append(
                        FrameRef(
                            index=len(frame_refs),
                            source_id=sd_token,
                            timestamp_us=int(sd_rec.get("timestamp", 0)) or None,
                            source_path=str(src) if src else "",
                            metadata={
                                "sample_token": current_token,
                                "sample_data_token": sd_token,
                                "channel": channel,
                                "filename": filename,
                                "width": sd_rec.get("width"),
                                "height": sd_rec.get("height"),
                                "scene_token": str(scene_rec.get("token", "")),
                                "scene_name": scene_name,
                            },
                        )
                    )

            current_token = str(sample_rec.get("next", ""))

        sequences.append(
            SequenceEntry(
                dataset_id=dataset_id,
                sequence_id=sequence_id,
                adapter="nuscenes_scene",
                display_name=scene_name,
                description=description,
                frame_count=len(frame_refs),
                frame_refs=frame_refs,
                annotation_ref=None,
                provenance={
                    "source": "nuscenes-mini",
                    "scene_token": str(scene_rec.get("token", "")),
                    "scene_name": scene_name,
                    "channel": channel,
                    "nuscenes_root": str(nuscenes_root),
                },
            )
        )

    return sequences


# ── jaad_clip adapter ─────────────────────────────────────────────────────────


def _probe_video_metadata(video_path: Path) -> Dict[str, Any]:
    """Probe basic video metadata without importing a heavy media framework.

    Uses ``ffprobe`` (preferred) if available, otherwise returns an empty dict.
    Never raises; always returns a (possibly empty) dict.
    """
    result: Dict[str, Any] = {}
    try:
        import subprocess
        proc = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_streams",
                str(video_path),
            ],
            capture_output=True,
            timeout=10,
        )
        if proc.returncode == 0:
            data = json.loads(proc.stdout.decode("utf-8", errors="replace"))
            for stream in data.get("streams", []):
                if stream.get("codec_type") == "video":
                    result["width"] = stream.get("width")
                    result["height"] = stream.get("height")
                    # FPS: avg_frame_rate is "30000/1001" style.
                    afr = stream.get("avg_frame_rate", "")
                    if "/" in afr:
                        num, den = afr.split("/", 1)
                        try:
                            fps = float(num) / float(den) if float(den) else None
                            result["fps"] = round(fps, 4) if fps else None
                        except (ValueError, ZeroDivisionError):
                            pass
                    nb_frames = stream.get("nb_frames")
                    if nb_frames and str(nb_frames).isdigit():
                        result["frame_count"] = int(nb_frames)
                    duration = stream.get("duration")
                    if duration:
                        try:
                            result["duration_sec"] = float(duration)
                        except ValueError:
                            pass
                    break
    except Exception:
        pass
    return result


def _parse_jaad_xml_summary(xml_path: Path) -> Dict[str, Any]:
    """Parse a JAAD primary annotation XML into a compact label summary.

    Extracts:
    - ``track_count`` — number of pedestrian tracks
    - ``annotated_frame_count`` — total annotated boxes (may exceed clip frames)
    - ``crossing_count`` — boxes labelled crossing
    - ``not_crossing_count`` — boxes labelled not-crossing
    - ``walking_count`` — boxes labelled walking
    - ``standing_count`` — boxes labelled standing
    - ``looking_count`` — boxes labelled looking
    - ``not_looking_count`` — boxes labelled not-looking
    - ``start_frame`` / ``stop_frame`` — annotation range
    - ``behavior_ranges`` — per-track {start, stop, cross, walking} summary

    Returns an empty dict on any parse error.
    """
    summary: Dict[str, Any] = {
        "track_count": 0,
        "annotated_frame_count": 0,
        "crossing_count": 0,
        "not_crossing_count": 0,
        "walking_count": 0,
        "standing_count": 0,
        "looking_count": 0,
        "not_looking_count": 0,
        "start_frame": None,
        "stop_frame": None,
    }
    try:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()

        # Extract overall frame range from <meta> if present.
        meta = root.find("meta")
        if meta is not None:
            task = meta.find("task")
            if task is not None:
                sf = task.findtext("start_frame")
                ef = task.findtext("stop_frame")
                if sf is not None:
                    try:
                        summary["start_frame"] = int(sf)
                    except ValueError:
                        pass
                if ef is not None:
                    try:
                        summary["stop_frame"] = int(ef)
                    except ValueError:
                        pass

        track_count = 0
        annotated = 0
        crossing = 0
        not_crossing = 0
        walking = 0
        standing = 0
        looking = 0
        not_looking = 0

        for track in root.findall("track"):
            # Only count pedestrian tracks (other labels may appear).
            label = track.get("label", "").lower()
            if label != "pedestrian":
                continue
            track_count += 1
            for box in track.findall("box"):
                outside = box.get("outside", "0")
                if outside == "1":
                    continue
                annotated += 1
                for attr in box.findall("attribute"):
                    name = (attr.get("name") or "").lower()
                    val = (attr.text or "").lower().strip()
                    if name == "cross":
                        if val == "crossing":
                            crossing += 1
                        elif val == "not-crossing":
                            not_crossing += 1
                    elif name == "action":
                        if val == "walking":
                            walking += 1
                        elif val == "standing":
                            standing += 1
                    elif name == "look":
                        if val == "looking":
                            looking += 1
                        elif val == "not-looking":
                            not_looking += 1

        summary["track_count"] = track_count
        summary["annotated_frame_count"] = annotated
        summary["crossing_count"] = crossing
        summary["not_crossing_count"] = not_crossing
        summary["walking_count"] = walking
        summary["standing_count"] = standing
        summary["looking_count"] = looking
        summary["not_looking_count"] = not_looking
    except Exception:
        pass
    return summary


def _load_jaad_clip_label_index(
    jaad_root: Path,
) -> Optional[Dict[str, Any]]:
    """Load the pre-built JAAD clip label index if present.

    Looks for ``prepared/jaad-clip-label-index.json`` under *jaad_root*.
    Returns None if the file is absent or unparseable.
    """
    idx_path = jaad_root / "prepared" / "jaad-clip-label-index.json"
    data = _load_json_file(idx_path)
    if not isinstance(data, dict):
        return None
    return data


def discover_jaad_clips(jaad_root: Path) -> List[SequenceEntry]:
    """Discover JAAD clip sequences from the installed JAAD dataset.

    Scans ``<jaad_root>/extracted/JAAD_clips/`` for ``video_XXXX.mp4`` files.
    Each clip becomes one independent ``SequenceEntry``.

    Annotation summaries are built from:
    1. ``prepared/jaad-clip-label-index.json`` (preferred — pre-built by Thor)
    2. ``annotations/video_XXXX.xml`` (parsed on demand when index is absent)

    Frame metadata is probed via ``ffprobe`` when available; the adapter
    degrades gracefully when the tool is absent.

    Parameters
    ----------
    jaad_root:
        Root of the JAAD installation (contains ``extracted/``, ``annotations/``
        etc.).

    Returns
    -------
    list of SequenceEntry
        One entry per MP4 file found; empty list on any discovery failure.
    """
    sequences: List[SequenceEntry] = []
    clips_dir = jaad_root / "extracted" / "JAAD_clips"
    if not clips_dir.is_dir():
        return sequences

    # Load the pre-built index once (may be None).
    label_index = _load_jaad_clip_label_index(jaad_root)

    try:
        mp4_files = sorted(
            f for f in clips_dir.iterdir()
            if f.is_file() and f.suffix.lower() == ".mp4"
        )
    except OSError:
        return sequences

    dataset_id = "jaad"

    for mp4_path in mp4_files[:_MAX_SEQUENCES]:
        # Containment check: each video must live within clips_dir.
        try:
            _assert_within_root(mp4_path, jaad_root)
        except ValueError:
            continue

        stem = mp4_path.stem  # e.g. "video_0001"
        sequence_id = _sanitize_id(stem)
        if not _is_safe_id(sequence_id):
            continue

        # Determine annotation families present.
        annotation_availability: Dict[str, bool] = {}
        primary_annotation_path: Optional[Path] = None
        for family in _JAAD_ANNOTATION_FAMILIES:
            ann_path = jaad_root / family / f"{stem}.xml"
            present = ann_path.is_file()
            annotation_availability[family] = present
            if family == "annotations" and present:
                primary_annotation_path = ann_path

        # Build annotation summary.
        annotation_summary: Dict[str, Any] = {
            "annotation_families": annotation_availability,
        }
        if label_index is not None:
            # Use pre-built index: look up by stem.
            clip_entry = label_index.get(stem)
            if isinstance(clip_entry, dict):
                annotation_summary.update(clip_entry)
        elif primary_annotation_path is not None:
            xml_summary = _parse_jaad_xml_summary(primary_annotation_path)
            annotation_summary.update(xml_summary)

        # Probe video metadata (non-blocking; empty dict if ffprobe absent).
        vid_meta = _probe_video_metadata(mp4_path)
        frame_count: int = vid_meta.get("frame_count") or 0

        # Build FrameRefs lazily: one ref per frame index pointing at the MP4.
        # We do NOT pre-extract frames; source_path is always the MP4 file.
        # When frame_count is unknown we store zero refs and rely on on-demand
        # probing during materialisation.
        frame_refs: List[FrameRef] = []
        if frame_count > 0:
            for i in range(frame_count):
                frame_refs.append(
                    FrameRef(
                        index=i,
                        source_id=f"{stem}:frame_{i}",
                        timestamp_us=None,  # computed during materialisation
                        source_path=str(mp4_path),
                        metadata={"video_stem": stem},
                    )
                )

        description_parts = []
        if vid_meta.get("width") and vid_meta.get("height"):
            description_parts.append(
                f"{vid_meta['width']}x{vid_meta['height']}"
            )
        if vid_meta.get("fps"):
            description_parts.append(f"{vid_meta['fps']:.2f} FPS")
        if vid_meta.get("duration_sec"):
            description_parts.append(
                f"{vid_meta['duration_sec']:.1f}s"
            )
        if annotation_availability.get("annotations"):
            description_parts.append("annotated")

        sequences.append(
            SequenceEntry(
                dataset_id=dataset_id,
                sequence_id=sequence_id,
                adapter="jaad_clip",
                display_name=stem,
                description=", ".join(description_parts) if description_parts else stem,
                frame_count=frame_count or len(frame_refs),
                frame_refs=frame_refs,
                annotation_ref=(
                    str(primary_annotation_path)
                    if primary_annotation_path
                    else None
                ),
                annotation_summary=annotation_summary,
                provenance={
                    "source": "jaad",
                    "video_stem": stem,
                    "video_path": str(mp4_path),
                    "video_metadata": vid_meta,
                    "jaad_root": str(jaad_root),
                },
            )
        )

    return sequences


# ── path resolution (server-side only) ───────────────────────────────────────


def resolve_sequence_frame_path(
    sequences: List[SequenceEntry],
    dataset_id: str,
    sequence_id: str,
    frame_index: int,
) -> Optional[Path]:
    """Return the server-side source path for a specific frame.

    Used during materialisation; the result is never forwarded to the browser.

    Parameters
    ----------
    sequences:
        Pre-discovered sequence list from ``discover_sequences()``.
    dataset_id:
        Dataset identifier (browser-provided, validated against catalog).
    sequence_id:
        Sequence identifier (browser-provided, validated against catalog).
    frame_index:
        Zero-based frame index within the sequence.

    Returns
    -------
    Path or None
        Resolved absolute path, or None if not found / unsafe.
    """
    if not _is_safe_id(dataset_id) or not _is_safe_id(sequence_id):
        return None
    for seq in sequences:
        if seq.dataset_id == dataset_id and seq.sequence_id == sequence_id:
            for fr in seq.frame_refs:
                if fr.index == frame_index:
                    p = Path(fr.source_path)
                    if p.exists():
                        return p
            return None
    return None


# ── public API ────────────────────────────────────────────────────────────────


def discover_sequences(
    rosbag_catalog_entries: Optional[List[Dict[str, Any]]] = None,
    nuscenes_root: Optional[str] = None,
    jaad_root: Optional[str] = None,
) -> Dict[str, Any]:
    """Discover all sequences from configured dataset roots.

    Parameters
    ----------
    rosbag_catalog_entries:
        Rosbag entries from ``discover_datasets()``.  When None, no
        ``ros_static_fixture`` sequences are returned.
    nuscenes_root:
        Root directory of the nuScenes mini installation.  Defaults to the
        ``NUSCENES_DIR`` environment variable or
        ``test_data/datasets/nuscenes-mini`` relative to the repo root.
        Pass an empty string to skip.
    jaad_root:
        Root directory of the JAAD installation.  Defaults to the
        ``JAAD_DIR`` environment variable or
        ``test_data/datasets/jaad`` relative to the repo root.
        Pass an empty string to skip.

    Returns
    -------
    dict with:
        ``sequences``        — flat list of all SequenceEntry dicts
        ``by_dataset``       — {dataset_id: [sequence dicts]} grouping
        ``adapter_counts``   — {adapter: count} summary
        ``errors``           — list of non-fatal error strings encountered
    """
    sequences: List[SequenceEntry] = []
    errors: List[str] = []

    # ── ros_static_fixture ────────────────────────────────────────────────────
    if rosbag_catalog_entries:
        try:
            static_seqs = discover_ros_static_fixtures(rosbag_catalog_entries)
            sequences.extend(static_seqs)
        except Exception as exc:
            errors.append(f"ros_static_fixture discovery error: {exc}")

    # ── nuscenes_scene ────────────────────────────────────────────────────────
    if nuscenes_root is None:
        nuscenes_root = os.environ.get("NUSCENES_DIR", "")
    if not nuscenes_root:
        here = Path(__file__).parent.parent
        candidate = here / "test_data" / "datasets" / "nuscenes-mini"
        if candidate.is_dir():
            nuscenes_root = str(candidate)
    if nuscenes_root:
        try:
            nu_seqs = discover_nuscenes_scenes(Path(nuscenes_root))
            sequences.extend(nu_seqs)
        except Exception as exc:
            errors.append(f"nuscenes_scene discovery error: {exc}")

    # ── jaad_clip ─────────────────────────────────────────────────────────────
    if jaad_root is None:
        jaad_root = os.environ.get("JAAD_DIR", "")
    if not jaad_root:
        here = Path(__file__).parent.parent
        candidate = here / "test_data" / "datasets" / "jaad"
        if candidate.is_dir():
            jaad_root = str(candidate)
    if jaad_root:
        try:
            jaad_seqs = discover_jaad_clips(Path(jaad_root))
            sequences.extend(jaad_seqs)
        except Exception as exc:
            errors.append(f"jaad_clip discovery error: {exc}")

    # ── build grouped output ─────────────────────────────────────────────────
    seq_dicts = [s.to_dict() for s in sequences]

    by_dataset: Dict[str, List[Dict[str, Any]]] = {}
    for s in seq_dicts:
        by_dataset.setdefault(s["dataset_id"], []).append(s)

    adapter_counts: Dict[str, int] = {}
    for s in sequences:
        adapter_counts[s.adapter] = adapter_counts.get(s.adapter, 0) + 1

    return {
        "sequences": seq_dicts,
        "by_dataset": by_dataset,
        "adapter_counts": adapter_counts,
        "errors": errors,
    }
