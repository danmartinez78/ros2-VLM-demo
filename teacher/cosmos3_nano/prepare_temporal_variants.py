#!/usr/bin/env python3
"""Convert a saved temporal capture into reproducible teacher-model media variants."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


def shuffled_frame_order(n: int) -> list[int]:
    order: list[int] = []
    left = 0
    right = n - 1
    while left <= right:
        order.append(left)
        left += 1
        if left <= right:
            order.append(right)
            right -= 1
    return order


def require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"required tool not found on PATH: {name}")


def write_video(
    capture_dir: Path,
    frame_files: list[str],
    order: list[int],
    fps: float,
    output_path: Path,
) -> dict:
    with tempfile.TemporaryDirectory(prefix="cosmos3_nano_frames_") as temp_root:
        temp_dir = Path(temp_root)
        for dst_index, src_index in enumerate(order):
            source = capture_dir / frame_files[src_index]
            destination = temp_dir / f"frame_{dst_index:04d}.png"
            shutil.copy2(source, destination)

        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-framerate",
                f"{fps:.12g}",
                "-i",
                str(temp_dir / "frame_%04d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(output_path),
            ],
            check=True,
        )

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames,r_frame_rate,duration",
            "-of",
            "json",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(probe.stdout)["streams"][0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build forward/reverse/shuffled/static media from a temporal capture"
    )
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path.home() / "cosmos3_teacher",
        help="Host directory mounted into the vLLM container as /data",
    )
    parser.add_argument(
        "--output-name",
        help="Output directory name under DATA_ROOT/generated; defaults to capture name",
    )
    args = parser.parse_args()

    require_tool("ffmpeg")
    require_tool("ffprobe")

    capture_dir = args.capture_dir.resolve()
    manifest_path = capture_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"capture manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frame_files = [str(value) for value in manifest["frame_files"]]
    timestamps = [float(value) for value in manifest["ros_timestamps_sec"]]
    if len(frame_files) != len(timestamps):
        raise ValueError("frame_files and ros_timestamps_sec have different lengths")
    if len(frame_files) < 2:
        raise ValueError("at least two frames are required")

    for filename in frame_files:
        path = capture_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)

    span = timestamps[-1] - timestamps[0]
    if span <= 0.0:
        raise ValueError("capture timestamp span must be positive")
    fps = (len(frame_files) - 1) / span

    data_root = args.data_root.expanduser().resolve()
    output_name = args.output_name or capture_dir.name
    output_dir = data_root / "generated" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    container_dir = Path("/data/generated") / output_name

    n = len(frame_files)
    variants = {
        "forward": list(range(n)),
        "reverse": list(reversed(range(n))),
        "shuffled": shuffled_frame_order(n),
        "static_terminal": [n - 1] * n,
    }

    output_variants: dict[str, dict] = {}
    for name, order in variants.items():
        output_path = output_dir / f"{name}.mp4"
        probe = write_video(capture_dir, frame_files, order, fps, output_path)
        output_variants[name] = {
            "host_path": str(output_path),
            "container_path": str(container_dir / f"{name}.mp4"),
            "frame_order": order,
            "frame_count": n,
            "probe": probe,
        }
        print(
            f"{name}: {output_path} "
            f"frames={probe.get('nb_read_frames')} duration={probe.get('duration')}"
        )

    terminal_source = capture_dir / frame_files[-1]
    terminal_output = output_dir / "terminal_only.png"
    shutil.copy2(terminal_source, terminal_output)

    dataset_manifest = {
        "schema_version": 1,
        "source_capture": str(capture_dir),
        "source_manifest": str(manifest_path),
        "source_motion_score": manifest.get("motion_score"),
        "source_relative_timestamps_sec": manifest.get("relative_timestamps_sec"),
        "source_timestamp_span_sec": span,
        "effective_constant_fps": fps,
        "data_root": str(data_root),
        "container_data_root": "/data",
        "variants": output_variants,
        "terminal_only": {
            "host_path": str(terminal_output),
            "container_path": str(container_dir / "terminal_only.png"),
            "frame_order": [n - 1],
        },
    }
    output_manifest = output_dir / "teacher_media_manifest.json"
    output_manifest.write_text(
        json.dumps(dataset_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"manifest: {output_manifest}")


if __name__ == "__main__":
    main()
