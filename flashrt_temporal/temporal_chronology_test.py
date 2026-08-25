#!/usr/bin/env python3
"""Capture and replay controlled Cosmos3 temporal chronology tests.

The harness captures one contiguous sampled ROS image window, saves the exact
frames and ROS timestamps, then sends three controlled variants to the existing
FlashRT IPC worker:

  forward       F1 F2 ... F8
  reverse       F8 F7 ... F1, using the same monotonic timestamp schedule
  terminal_only F8 only

Keeping the timestamp schedule fixed for forward/reverse isolates frame order as
the experimental variable. A saved capture can be replayed later without ROS bag
playback by passing --load-capture.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rclpy
from PIL import Image as PilImage
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from temporal_ros_node import IpcClient, ros_image_to_bgr, stamp_to_sec


DEFAULT_PROMPT = (
    "Analyze these frames as one chronological video sequence.\n"
    "Report only changes directly supported across the frames.\n"
    "For each dynamic object or scene element, identify what changed and the "
    "direction or trend. State whether it appeared, disappeared, approached, "
    "receded, or moved laterally when supported. Do not describe static objects "
    "unless needed to explain a change. Do not infer motion from a single frame. "
    "Do not speculate when temporal evidence is insufficient.\n"
    "Return exactly this compact format:\n"
    "CHANGES:\n"
    "- <object>: <temporal change>\n"
    "CAMERA_MOTION: <stationary|forward|backward|pan_left|pan_right|uncertain>\n"
    "SUMMARY: <one sentence describing the most important temporal event>\n"
    "If there are no supported changes, write '- none' under CHANGES."
)


@dataclass
class CapturedFrame:
    image: np.ndarray
    stamp_sec: float


class WindowCaptureNode(Node):
    def __init__(
        self,
        image_topic: str,
        sample_period: float,
        window_frames: int,
        max_gap: float,
    ) -> None:
        super().__init__("flashrt_temporal_capture")
        self.image_topic = image_topic
        self.sample_period = sample_period
        self.window_frames = window_frames
        self.max_gap = max_gap
        self.buffer: deque[CapturedFrame] = deque(maxlen=window_frames)
        self.last_sampled: float | None = None
        self.captured: list[CapturedFrame] | None = None
        self.subscription = self.create_subscription(
            Image,
            image_topic,
            self._image_callback,
            qos_profile_sensor_data,
        )
        self.get_logger().info(
            f"capturing topic={image_topic} window={window_frames} "
            f"sample_period={sample_period:.3f}s max_gap={max_gap:.3f}s"
        )

    def _reset(self, reason: str) -> None:
        self.buffer.clear()
        self.last_sampled = None
        self.get_logger().warning(f"capture window reset: {reason}")

    def _image_callback(self, msg: Image) -> None:
        if self.captured is not None:
            return

        stamp = stamp_to_sec(msg.header.stamp)
        if self.last_sampled is not None:
            elapsed = stamp - self.last_sampled
            if elapsed < 0.0:
                self._reset(f"timestamp moved backward by {-elapsed:.3f}s")
            elif elapsed > self.max_gap:
                self._reset(f"gap={elapsed:.3f}s > {self.max_gap:.3f}s")
            elif elapsed < self.sample_period:
                return

        try:
            frame = ros_image_to_bgr(msg)
        except Exception as exc:
            self.get_logger().error(f"image conversion failed: {exc}")
            return

        self.last_sampled = stamp
        self.buffer.append(CapturedFrame(frame, stamp))
        if len(self.buffer) == self.window_frames:
            self.captured = list(self.buffer)
            span = self.captured[-1].stamp_sec - self.captured[0].stamp_sec
            self.get_logger().info(
                f"captured {self.window_frames} contiguous frames spanning {span:.3f}s"
            )


def capture_window(args) -> list[CapturedFrame]:
    rclpy.init()
    node = WindowCaptureNode(
        args.image_topic,
        args.sample_period_seconds,
        args.window_frames,
        args.max_gap_seconds,
    )
    deadline = time.monotonic() + args.capture_timeout_seconds
    try:
        while node.captured is None and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if node.captured is None:
            raise TimeoutError(
                f"no contiguous {args.window_frames}-frame window captured within "
                f"{args.capture_timeout_seconds:.1f}s"
            )
        return node.captured
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def save_capture(
    frames: list[CapturedFrame],
    args,
    prompt: str,
) -> Path:
    root = Path(args.output_root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    capture_dir = root / f"capture_{stamp}"
    suffix = 1
    while capture_dir.exists():
        capture_dir = root / f"capture_{stamp}_{suffix:02d}"
        suffix += 1
    capture_dir.mkdir(parents=True, exist_ok=False)

    frame_files: list[str] = []
    for i, item in enumerate(frames):
        name = f"frame_{i:02d}.png"
        rgb = item.image[:, :, ::-1]
        PilImage.fromarray(rgb, mode="RGB").save(capture_dir / name)
        frame_files.append(name)

    timestamps = [item.stamp_sec for item in frames]
    t0 = timestamps[0]
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "image_topic": args.image_topic,
        "sample_period_seconds": args.sample_period_seconds,
        "max_gap_seconds": args.max_gap_seconds,
        "frame_count": len(frames),
        "frame_files": frame_files,
        "ros_timestamps_sec": timestamps,
        "relative_timestamps_sec": [ts - t0 for ts in timestamps],
        "timestamp_span_sec": timestamps[-1] - timestamps[0],
        "prompt": prompt,
    }
    (capture_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return capture_dir


def load_capture(capture_dir: Path) -> tuple[list[CapturedFrame], dict]:
    manifest_path = capture_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest["frame_files"]
    timestamps = [float(v) for v in manifest["ros_timestamps_sec"]]
    if len(files) != len(timestamps):
        raise ValueError("capture manifest frame/timestamp count mismatch")

    frames: list[CapturedFrame] = []
    for name, stamp in zip(files, timestamps):
        rgb = np.asarray(PilImage.open(capture_dir / name).convert("RGB"), dtype=np.uint8)
        frames.append(CapturedFrame(rgb[:, :, ::-1].copy(), stamp))
    return frames, manifest


def run_variant(
    client: IpcClient,
    name: str,
    source_frames: list[CapturedFrame],
    frame_order: list[int],
    timestamp_schedule: list[float],
    prompt: str,
    max_tokens: int,
) -> dict:
    images = [source_frames[i].image for i in frame_order]
    success, text, error, infer_s, encoding = client.infer(
        images,
        timestamp_schedule,
        prompt,
        max_tokens,
    )
    base = timestamp_schedule[0]
    result = {
        "variant": name,
        "success": success,
        "frame_order": frame_order,
        "relative_timestamps_sec": [ts - base for ts in timestamp_schedule],
        "inference_seconds": infer_s,
        "temporal_encoding": encoding,
        "response": text,
        "error": error,
    }
    print("\n" + "=" * 78)
    print(f"VARIANT: {name}")
    print(f"frame_order={frame_order}")
    print(
        "timestamps="
        + str([round(ts - base, 3) for ts in timestamp_schedule])
    )
    print(f"success={success} inference_seconds={infer_s:.3f} encoding={encoding}")
    if text:
        print(text)
    if error:
        print(f"ERROR: {error}")
    return result


def run_experiment(
    frames: list[CapturedFrame],
    capture_dir: Path,
    args,
    prompt: str,
) -> list[dict]:
    if len(frames) < 2:
        raise ValueError("chronology experiment requires at least two captured frames")

    timestamps = [item.stamp_sec for item in frames]
    n = len(frames)
    client = IpcClient(args.worker_socket_path, args.worker_timeout_seconds)
    try:
        results = [
            run_variant(
                client,
                "forward",
                frames,
                list(range(n)),
                timestamps,
                prompt,
                args.max_generate_length,
            ),
            run_variant(
                client,
                "reverse",
                frames,
                list(reversed(range(n))),
                timestamps,
                prompt,
                args.max_generate_length,
            ),
            run_variant(
                client,
                "terminal_only",
                frames,
                [n - 1],
                [timestamps[-1]],
                prompt,
                args.max_generate_length,
            ),
        ]
    finally:
        client.close()

    output = {
        "capture_dir": str(capture_dir),
        "worker_socket_path": args.worker_socket_path,
        "max_generate_length": args.max_generate_length,
        "prompt": prompt,
        "experimental_control": (
            "forward and reverse use identical monotonic timestamp schedules; "
            "only frame order changes"
        ),
        "results": results,
    }
    (capture_dir / "chronology_results.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Capture and replay forward/reverse/terminal-only Cosmos3 video tests"
    )
    ap.add_argument("--image-topic", default="/camera0/color/image_raw")
    ap.add_argument("--sample-period-seconds", type=float, default=0.25)
    ap.add_argument("--window-frames", type=int, default=8)
    ap.add_argument("--max-gap-seconds", type=float, default=1.25)
    ap.add_argument("--capture-timeout-seconds", type=float, default=60.0)
    ap.add_argument("--worker-socket-path", default="/tmp/edge_vlm_flashrt.sock")
    ap.add_argument("--worker-timeout-seconds", type=float, default=90.0)
    ap.add_argument("--max-generate-length", type=int, default=256)
    ap.add_argument("--output-root", default="temporal_captures")
    ap.add_argument(
        "--load-capture",
        type=Path,
        help="Replay an existing saved capture instead of subscribing to ROS",
    )
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.sample_period_seconds < 0.0:
        raise ValueError("--sample-period-seconds must be >= 0")
    if args.window_frames < 2:
        raise ValueError("--window-frames must be >= 2")
    if args.max_gap_seconds <= 0.0:
        raise ValueError("--max-gap-seconds must be > 0")
    if args.capture_timeout_seconds <= 0.0:
        raise ValueError("--capture-timeout-seconds must be > 0")
    if args.max_generate_length <= 0:
        raise ValueError("--max-generate-length must be > 0")

    if args.load_capture is not None:
        capture_dir = args.load_capture.resolve()
        frames, manifest = load_capture(capture_dir)
        prompt = args.prompt if args.prompt != DEFAULT_PROMPT else manifest.get("prompt", DEFAULT_PROMPT)
        print(
            f"loaded capture {capture_dir} with {len(frames)} frames; "
            f"span={frames[-1].stamp_sec - frames[0].stamp_sec:.3f}s"
        )
    else:
        frames = capture_window(args)
        prompt = args.prompt
        capture_dir = save_capture(frames, args, prompt).resolve()
        print(f"saved capture to {capture_dir}")

    run_experiment(frames, capture_dir, args, prompt)
    print(f"\nresults saved to {capture_dir / 'chronology_results.json'}")


if __name__ == "__main__":
    main()
