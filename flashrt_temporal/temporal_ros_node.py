#!/usr/bin/env python3
"""Experimental ROS 2 temporal window node for the FlashRT Cosmos3 worker.

This node deliberately keeps the existing downstream VlmResult contract while
using a rolling ordered frame window upstream. It is an additive research path;
the existing C++ single-frame node remains unchanged for baselines.
"""
from __future__ import annotations

import ctypes
import hashlib
import socket
import threading
from collections import deque
from dataclasses import dataclass

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from edge_vlm_ros.msg import VlmResult
from ipc_protocol import (
    ENCODING_BGR8,
    MAGIC,
    SCHEMA_FLAG_HAS_FRAME_TIMESTAMPS,
    SCHEMA_FLAG_MULTI_IMAGE,
    SEQUENCE_IMAGES,
    SEQUENCE_VIDEO,
    VERSION,
    PerImageHeader,
    RequestHeader,
    ResponseHeader,
    recv_exact,
    recv_struct,
    send_struct,
)


@dataclass
class FrameRecord:
    image: np.ndarray
    stamp_sec: float
    header: object
    sequence: int


class IpcClient:
    def __init__(self, socket_path: str, timeout_seconds: float):
        self.socket_path = socket_path
        self.timeout_seconds = timeout_seconds
        self.sock: socket.socket | None = None
        self.next_request_id = 1

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None

    def _connect(self) -> socket.socket:
        if self.sock is not None:
            return self.sock
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout_seconds)
        sock.connect(self.socket_path)
        self.sock = sock
        return sock

    def infer(
        self,
        frames: list[np.ndarray],
        timestamps: list[float],
        prompt: str,
        max_tokens: int,
        sequence_type: int = SEQUENCE_VIDEO,
    ):
        if not frames:
            raise ValueError("temporal request requires at least one frame")
        if len(frames) != len(timestamps):
            raise ValueError("timestamps must match frame count")
        if sequence_type not in (SEQUENCE_IMAGES, SEQUENCE_VIDEO):
            raise ValueError("unsupported IPC sequence type")
        sock = self._connect()
        request_id = self.next_request_id
        self.next_request_id += 1

        encoded_prompt = prompt.encode("utf-8")
        primary = np.ascontiguousarray(frames[0], dtype=np.uint8)
        extras = [np.ascontiguousarray(frame, dtype=np.uint8) for frame in frames[1:]]
        all_frames = [primary] + extras
        for frame in all_frames:
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError("IPC frames must be HxWx3 BGR8")

        header = RequestHeader()
        header.magic = MAGIC
        header.version = VERSION
        header.request_id = request_id
        header.width = primary.shape[1]
        header.height = primary.shape[0]
        header.step = primary.shape[1] * 3
        header.encoding = ENCODING_BGR8
        header.image_bytes = primary.nbytes
        header.prompt_bytes = len(encoded_prompt)
        header.max_generate_length = max_tokens
        header.temperature = 0.0
        header.top_p = 1.0
        header.top_k = 1
        header.schema_flags = SCHEMA_FLAG_HAS_FRAME_TIMESTAMPS
        if extras:
            header.schema_flags |= SCHEMA_FLAG_MULTI_IMAGE
            header.image_count = len(all_frames)
        else:
            header.image_count = 0
        header.sequence_type = sequence_type
        header.fps = 0.0
        header.timestamp_count = len(timestamps)

        try:
            send_struct(sock, header)
            for frame in extras:
                pih = PerImageHeader()
                pih.width = frame.shape[1]
                pih.height = frame.shape[0]
                pih.step = frame.shape[1] * 3
                pih.image_bytes = frame.nbytes
                send_struct(sock, pih)
            sock.sendall(primary.tobytes(order="C"))
            for frame in extras:
                sock.sendall(frame.tobytes(order="C"))
            ts_array = (ctypes.c_double * len(timestamps))(*timestamps)
            sock.sendall(bytes(ts_array))
            sock.sendall(encoded_prompt)

            response = recv_struct(sock, ResponseHeader)
            if response.magic != MAGIC or response.version != VERSION or response.request_id != request_id:
                raise RuntimeError("invalid worker response header")
            text = recv_exact(sock, response.text_bytes).decode("utf-8")
            error = recv_exact(sock, response.error_bytes).decode("utf-8")
            encoding = recv_exact(sock, response.temporal_encoding_bytes).decode("utf-8")
            return bool(response.success), text, error, float(response.inference_seconds), encoding
        except Exception:
            self.close()
            raise


def ros_image_to_bgr(msg: Image) -> np.ndarray:
    if msg.encoding not in ("bgr8", "rgb8", "mono8"):
        raise ValueError(f"unsupported image encoding: {msg.encoding}")
    channels = 1 if msg.encoding == "mono8" else 3
    min_step = int(msg.width) * channels
    if int(msg.step) < min_step:
        raise ValueError("image step is smaller than packed row width")
    required = int(msg.step) * int(msg.height)
    if len(msg.data) < required:
        raise ValueError("image payload is smaller than step * height")
    raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(int(msg.height), int(msg.step))
    packed = raw[:, :min_step]
    if channels == 1:
        mono = packed.reshape(int(msg.height), int(msg.width))
        return np.repeat(mono[:, :, None], 3, axis=2).copy()
    arr = packed.reshape(int(msg.height), int(msg.width), 3)
    if msg.encoding == "rgb8":
        return arr[:, :, ::-1].copy()
    return arr.copy()


def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class FlashRtTemporalNode(Node):
    def __init__(self):
        super().__init__("flashrt_temporal_vlm_node")
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("result_topic", "/vlm/result")
        self.declare_parameter("worker_socket_path", "/tmp/edge_vlm_flashrt.sock")
        self.declare_parameter("worker_timeout_seconds", 90.0)
        self.declare_parameter("sample_period_seconds", 0.5)
        self.declare_parameter("temporal_window_frames", 8)
        self.declare_parameter("temporal_require_full_window", True)
        self.declare_parameter("temporal_max_gap_seconds", 1.25)
        self.declare_parameter("max_generate_length", 256)
        self.declare_parameter("task_profile", "temporal_change")
        self.declare_parameter("prompt_version", "flashrt-temporal-change-v2")
        self.declare_parameter(
            "prompt",
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
            "If there are no supported changes, write '- none' under CHANGES.",
        )

        self.image_topic = self.get_parameter("image_topic").value
        self.result_topic = self.get_parameter("result_topic").value
        self.sample_period = float(self.get_parameter("sample_period_seconds").value)
        self.window_frames = int(self.get_parameter("temporal_window_frames").value)
        self.require_full = bool(self.get_parameter("temporal_require_full_window").value)
        self.max_gap = float(self.get_parameter("temporal_max_gap_seconds").value)
        self.max_tokens = int(self.get_parameter("max_generate_length").value)
        self.prompt = str(self.get_parameter("prompt").value)
        self.task_profile = str(self.get_parameter("task_profile").value)
        self.prompt_version = str(self.get_parameter("prompt_version").value)
        self.prompt_hash = hashlib.sha256(self.prompt.encode("utf-8")).hexdigest()[:16]
        socket_path = str(self.get_parameter("worker_socket_path").value)
        timeout = float(self.get_parameter("worker_timeout_seconds").value)

        if self.sample_period < 0.0:
            raise ValueError("sample_period_seconds must be >= 0")
        if self.window_frames < 1 or self.window_frames > 32:
            raise ValueError("temporal_window_frames must be in [1, 32]")
        if self.max_gap <= 0.0:
            raise ValueError("temporal_max_gap_seconds must be > 0")
        if self.sample_period > 0.0 and self.max_gap <= self.sample_period:
            raise ValueError("temporal_max_gap_seconds must be greater than sample_period_seconds")
        if self.max_tokens <= 0:
            raise ValueError("max_generate_length must be > 0")

        self.client = IpcClient(socket_path, timeout)
        self.buffer: deque[FrameRecord] = deque(maxlen=self.window_frames)
        self.last_sampled: float | None = None
        self.sequence = 0
        self.pending: list[FrameRecord] | None = None
        self.cv = threading.Condition()
        self.running = True
        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.publisher = self.create_publisher(VlmResult, self.result_topic, 10)
        self.subscription = self.create_subscription(
            Image,
            self.image_topic,
            self._image_callback,
            qos_profile_sensor_data,
        )
        self.worker.start()
        self.get_logger().info(
            f"FlashRT temporal node: topic={self.image_topic} window={self.window_frames} "
            f"sample_period={self.sample_period:.3f}s max_gap={self.max_gap:.3f}s "
            f"socket={socket_path}"
        )

    def destroy_node(self):
        with self.cv:
            self.running = False
            self.cv.notify_all()
        if self.worker.is_alive():
            self.worker.join(timeout=5.0)
        self.client.close()
        super().destroy_node()

    def _reset_temporal_window(self, reason: str) -> None:
        self.buffer.clear()
        self.last_sampled = None
        with self.cv:
            self.pending = None
        self.get_logger().warning(reason)

    def _image_callback(self, msg: Image) -> None:
        stamp = stamp_to_sec(msg.header.stamp)
        if self.last_sampled is not None:
            elapsed = stamp - self.last_sampled
            if elapsed < 0.0:
                self._reset_temporal_window(
                    f"temporal discontinuity: timestamp moved backward by {-elapsed:.3f}s; "
                    "resetting window"
                )
            elif elapsed > self.max_gap:
                self._reset_temporal_window(
                    f"temporal discontinuity: gap={elapsed:.3f}s > {self.max_gap:.3f}s; "
                    "resetting window"
                )
            elif elapsed < self.sample_period:
                return
        try:
            frame = ros_image_to_bgr(msg)
        except Exception as exc:
            self.get_logger().error(f"image conversion failed: {exc}")
            return
        self.last_sampled = stamp
        self.sequence += 1
        record = FrameRecord(frame, stamp, msg.header, self.sequence)
        self.buffer.append(record)
        if self.require_full and len(self.buffer) < self.window_frames:
            return
        with self.cv:
            self.pending = list(self.buffer)
            self.cv.notify()

    def _worker_loop(self) -> None:
        while True:
            with self.cv:
                self.cv.wait_for(lambda: self.pending is not None or not self.running)
                if not self.running and self.pending is None:
                    return
                window = self.pending
                self.pending = None
            if not window:
                continue
            frames = [item.image for item in window]
            timestamps = [item.stamp_sec for item in window]
            latest = window[-1]
            try:
                success, text, error, infer_s, encoding = self.client.infer(
                    frames, timestamps, self.prompt, self.max_tokens
                )
            except Exception as exc:
                success, text, error, infer_s, encoding = False, "", str(exc), 0.0, ""

            if not self.running or not rclpy.ok():
                return

            result = VlmResult()
            result.header = latest.header
            result.source_topic = self.image_topic
            result.detector_id = ""
            result.tracker_id = ""
            result.task_profile = self.task_profile
            result.prompt_version = self.prompt_version
            result.prompt_config_hash = self.prompt_hash
            result.prompt = self.prompt
            result.response = text
            result.inference_seconds = infer_s
            result.frame_sequence = latest.sequence
            result.observation_age_seconds = 0.0
            result.tracker_context = f"temporal_encoding={encoding}; frames={len(window)}"
            result.tracked_object_count = 0
            result.source_sequence = 0
            result.success = success
            result.error = error
            try:
                self.publisher.publish(result)
            except Exception:
                if not self.running or not rclpy.ok():
                    return
                raise
            if success:
                self.get_logger().info(
                    f"[window ending frame {latest.sequence} | {len(window)} frames | "
                    f"{infer_s:.3f}s | {encoding}] {text}"
                )
            else:
                self.get_logger().error(
                    f"[window ending frame {latest.sequence}] inference failed: {error}"
                )


def main() -> None:
    rclpy.init()
    node = FlashRtTemporalNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
