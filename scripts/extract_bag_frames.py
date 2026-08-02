#!/usr/bin/env python3
# Copyright 2025 edge_vlm_ros contributors
"""
Rosbag frame extraction script for the warehouse workbench.

Reads a rosbag2 bag, extracts sensor_msgs/Image frames from a specified topic,
converts them to JPEG, and writes a JSON manifest.

This script is designed to run as a subprocess launched by the web console.
All parameters are passed as command-line arguments (no environment variables
required for the actual extraction logic).

Requirements (hardware-dependent, not needed for CI tests):
  - rosbag2_py (installed with ROS 2)
  - cv_bridge or manual image conversion
  - numpy, OpenCV (or Pillow for JPEG encoding)

When rosbag2_py is unavailable (CI environment), the script exits with code 2
and writes an appropriate message to stderr.

Output:
  <output_dir>/frame_NNNN.jpg   — extracted JPEG frames
  <output_dir>/frame_dataset.json — manifest with frame metadata
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Extract frames from a rosbag2 bag")
    p.add_argument("--bag-path", required=True, help="Path to rosbag2 directory")
    p.add_argument("--output-dir", required=True, help="Output directory for frames and manifest")
    p.add_argument("--topic", required=True, help="ROS topic name (sensor_msgs/Image)")
    p.add_argument("--dataset-id", required=True, help="Dataset UUID")
    p.add_argument("--bag-key", default="", help="Catalog key for the rosbag (e.g. 'image-proc')")
    p.add_argument("--start-offset", type=float, default=0.0, help="Start time offset (seconds)")
    p.add_argument("--end-offset", type=float, default=None, help="End time offset (seconds)")
    p.add_argument("--duration", type=float, default=None, help="Play duration in seconds (alternative to end-offset)")
    p.add_argument("--sample-interval", type=float, default=0.5, help="Minimum interval between frames (seconds)")
    p.add_argument("--max-frames", type=int, default=100, help="Maximum frames to extract")
    p.add_argument("--target-count", type=int, default=None, help="Target number of frames (auto-compute interval)")
    return p.parse_args(argv)


def _try_encode_image(msg_data: bytes, encoding: str, width: int, height: int) -> bytes:
    """Convert raw sensor_msgs/Image data to JPEG bytes.

    Supports bgr8, rgb8, and mono8 encodings.  Returns empty bytes on failure.
    """
    try:
        import numpy as np
        import cv2

        if encoding in ("bgr8", "rgb8", "bgra8", "rgba8"):
            channels = 4 if encoding.endswith("a8") else 3
            arr = np.frombuffer(msg_data, dtype=np.uint8).reshape((height, width, channels))
            if encoding in ("rgb8", "rgba8"):
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            _, buf = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, 90])
            return bytes(buf)
        elif encoding == "mono8":
            arr = np.frombuffer(msg_data, dtype=np.uint8).reshape((height, width))
            _, buf = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, 90])
            return bytes(buf)
        else:
            # Attempt grey-scale fallback for 16-bit formats
            if "16" in encoding:
                arr = np.frombuffer(msg_data, dtype=np.uint16).reshape((height, width))
                arr8 = (arr >> 8).astype(np.uint8)
                _, buf = cv2.imencode(".jpg", arr8, [cv2.IMWRITE_JPEG_QUALITY, 90])
                return bytes(buf)
    except Exception as exc:
        print(f"[extract_bag_frames] Image encode error: {exc}", file=sys.stderr)
    return b""


def main(argv=None) -> int:
    args = _parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Try importing rosbag2_py; if absent, exit cleanly with code 2 so the
    # web console can display a meaningful error rather than a crash.
    try:
        import rosbag2_py  # noqa: F401
    except ImportError:
        print(
            "[extract_bag_frames] rosbag2_py not available; "
            "this script requires a ROS 2 environment.",
            file=sys.stderr,
        )
        return 2

    try:
        from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError as exc:
        print(f"[extract_bag_frames] Import error: {exc}", file=sys.stderr)
        return 2

    storage_options = StorageOptions(uri=args.bag_path, storage_id="sqlite3")
    converter_options = ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )

    reader = SequentialReader()
    try:
        reader.open(storage_options, converter_options)
    except Exception as exc:
        print(f"[extract_bag_frames] Failed to open bag: {exc}", file=sys.stderr)
        return 1

    # Read bag metadata for duration
    bag_metadata = reader.get_metadata()
    bag_duration_ns = bag_metadata.duration.nanoseconds
    bag_duration_sec = bag_duration_ns / 1e9

    # Validate topic exists in bag
    topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if args.topic not in topic_types:
        available = list(topic_types.keys())
        print(
            f"[extract_bag_frames] Topic {args.topic!r} not found in bag. "
            f"Available: {available}",
            file=sys.stderr,
        )
        return 1

    # Set up topic filter
    filter_obj = rosbag2_py.StorageFilter(topics=[args.topic])
    reader.set_filter(filter_obj)

    # Get message type
    msg_type_str = topic_types[args.topic]
    try:
        MsgClass = get_message(msg_type_str)
    except Exception as exc:
        print(f"[extract_bag_frames] Cannot load message type {msg_type_str!r}: {exc}", file=sys.stderr)
        return 1

    start_ns = int(args.start_offset * 1e9)
    end_ns = int(args.end_offset * 1e9) if args.end_offset is not None else None
    sample_interval_ns = int(args.sample_interval * 1e9)

    # Auto-compute interval from target count if requested
    if args.target_count is not None and args.target_count > 0:
        available_sec = (
            (args.end_offset - args.start_offset)
            if args.end_offset is not None
            else bag_duration_sec - args.start_offset
        )
        computed_interval = max(0.05, available_sec / args.target_count)
        sample_interval_ns = int(computed_interval * 1e9)

    frames: list = []
    last_sample_ns: int = -1
    frame_count = 0
    bag_start_ns: int = -1

    while reader.has_next() and frame_count < args.max_frames:
        topic, data, timestamp_ns = reader.read_next()

        if bag_start_ns < 0:
            bag_start_ns = timestamp_ns

        relative_ns = timestamp_ns - bag_start_ns

        if relative_ns < start_ns:
            continue
        if end_ns is not None and relative_ns > end_ns:
            break
        if last_sample_ns >= 0 and (relative_ns - last_sample_ns) < sample_interval_ns:
            continue

        # Deserialize and encode
        try:
            msg = deserialize_message(data, MsgClass)
            img_bytes = _try_encode_image(
                bytes(msg.data),
                msg.encoding,
                msg.width,
                msg.height,
            )
        except Exception as exc:
            print(f"[extract_bag_frames] Frame {frame_count} decode error: {exc}", file=sys.stderr)
            continue

        if not img_bytes:
            continue

        filename = f"frame_{frame_count:04d}.jpg"
        frame_path = output_dir / filename
        frame_path.write_bytes(img_bytes)

        frames.append({
            "index": frame_count,
            "filename": filename,
            "timestamp_ns": timestamp_ns,
            "timestamp_sec": timestamp_ns / 1e9,
        })
        last_sample_ns = relative_ns
        frame_count += 1

    # Write manifest
    manifest = {
        "schema_version": 1,
        "dataset_id": args.dataset_id,
        "bag_key": args.bag_key,
        "bag_path": args.bag_path,
        "topic": args.topic,
        "start_offset_sec": args.start_offset,
        "end_offset_sec": args.end_offset,
        "sample_interval_sec": args.sample_interval,
        "max_frames": args.max_frames,
        "frames": frames,
        "extracted_at": _now_iso(),
        "bag_duration_sec": bag_duration_sec,
        "frame_count": frame_count,
        "output_dir": str(output_dir),
    }
    manifest_path = output_dir / "frame_dataset.json"
    tmp_path = output_dir / "frame_dataset.json.tmp"
    tmp_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(str(tmp_path), str(manifest_path))

    print(
        f"[extract_bag_frames] Extracted {frame_count} frames to {output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
