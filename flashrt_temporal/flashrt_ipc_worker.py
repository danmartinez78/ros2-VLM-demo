#!/usr/bin/env python3
"""FlashRT Cosmos3 IPC worker for native temporal/video inference on Thor.

The worker speaks edge_vlm_ros IPC protocol v3. It accepts the same ordered BGR
frames, sequence_type, fps, timestamps, prompt, system message, and history used
by the existing C++ inference worker. For sequence_type=video and
sequence_type=temporal_images, the ordered frame list is presented to Cosmos3 as
one native video media item.
"""
from __future__ import annotations

import argparse
import ctypes
import importlib.util
import math
import os
import signal
import socket
import statistics
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from ipc_protocol import (
    ENCODING_BGR8,
    MAGIC,
    MAX_EXTRA_IMAGES,
    MAX_HISTORY_ENTRIES,
    MAX_IMAGE_BYTES,
    MAX_TEXT_BYTES,
    SCHEMA_FLAG_HAS_FPS,
    SCHEMA_FLAG_HAS_FRAME_TIMESTAMPS,
    SCHEMA_FLAG_MULTI_IMAGE,
    SCHEMA_FLAG_STRUCTURED,
    SEQUENCE_IMAGES,
    SEQUENCE_TEMPORAL_IMAGES,
    SEQUENCE_VIDEO,
    VERSION,
    HistoryEntryHeader,
    PerImageHeader,
    RequestHeader,
    ResponseHeader,
    recv_exact,
    recv_struct,
    send_struct,
)


@dataclass
class ParsedRequest:
    header: RequestHeader
    frames: list[Image.Image]
    timestamps: list[float]
    system_message: str
    prompt: str
    history: list[tuple[str, str]]


class DeadlineGuard:
    def __init__(self, seconds: int, request_id: int):
        self._done = threading.Event()
        self._seconds = seconds
        self._request_id = request_id
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def _run(self) -> None:
        if not self._done.wait(self._seconds):
            print(
                f"FlashRT worker inference deadline exceeded for request "
                f"{self._request_id}; terminating worker",
                flush=True,
            )
            os._exit(1)

    def __exit__(self, exc_type, exc, tb):
        self._done.set()
        return False


def build_processor(checkpoint: str, cosmos_framework_root: str):
    processor_path = Path(cosmos_framework_root) / (
        "cosmos_framework/data/generator/processors/cosmos3_edge_processing.py"
    )
    if not processor_path.is_file():
        raise FileNotFoundError(f"Cosmos3 processor module not found: {processor_path}")
    spec = importlib.util.spec_from_file_location(
        "cosmos3_edge_processing_standalone", processor_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load processor module: {processor_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_cosmos3_edge_processor(checkpoint)


def bgr_payload_to_pil(payload: bytes, width: int, height: int, step: int) -> Image.Image:
    if step != width * 3:
        raise ValueError("only packed BGR8 frames are supported")
    arr = np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 3)
    rgb = arr[:, :, ::-1].copy()
    return Image.fromarray(rgb, mode="RGB")


def _validate_header(header: RequestHeader) -> None:
    if header.magic != MAGIC or header.version != VERSION:
        raise ValueError("invalid IPC magic/version")
    if header.encoding != ENCODING_BGR8:
        raise ValueError("FlashRT worker requires BGR8 IPC frames")
    if not header.width or not header.height:
        raise ValueError("invalid zero-sized frame")
    if header.step != header.width * 3:
        raise ValueError("invalid packed BGR step")
    if header.image_bytes != header.step * header.height:
        raise ValueError("invalid primary image payload size")
    if header.image_bytes > MAX_IMAGE_BYTES:
        raise ValueError("primary image exceeds protocol limit")
    if header.prompt_bytes > MAX_TEXT_BYTES or header.system_bytes > MAX_TEXT_BYTES:
        raise ValueError("text payload exceeds protocol limit")
    if header.history_count > MAX_HISTORY_ENTRIES:
        raise ValueError("history count exceeds protocol limit")
    if header.sequence_type not in (SEQUENCE_IMAGES, SEQUENCE_TEMPORAL_IMAGES, SEQUENCE_VIDEO):
        raise ValueError("invalid sequence_type")
    has_fps = bool(header.schema_flags & SCHEMA_FLAG_HAS_FPS)
    if has_fps and (not math.isfinite(header.fps) or header.fps <= 0.0):
        raise ValueError("invalid fps")
    if not has_fps and header.fps != 0.0:
        raise ValueError("fps provided without schema flag")


def read_request(sock: socket.socket) -> ParsedRequest:
    header = recv_struct(sock, RequestHeader)
    _validate_header(header)
    is_multi = bool(header.schema_flags & SCHEMA_FLAG_MULTI_IMAGE)
    has_timestamps = bool(header.schema_flags & SCHEMA_FLAG_HAS_FRAME_TIMESTAMPS)
    is_structured = bool(header.schema_flags & SCHEMA_FLAG_STRUCTURED)

    image_count = header.image_count if is_multi else 1
    if is_multi and (image_count < 2 or image_count > MAX_EXTRA_IMAGES + 1):
        raise ValueError("invalid multi-image count")
    if has_timestamps and header.timestamp_count != image_count:
        raise ValueError("timestamp count must equal frame count")
    if not has_timestamps and header.timestamp_count != 0:
        raise ValueError("timestamp count provided without schema flag")

    extra_headers: list[PerImageHeader] = []
    for _ in range(image_count - 1):
        pih = recv_struct(sock, PerImageHeader)
        if (
            not pih.width
            or not pih.height
            or pih.step != pih.width * 3
            or pih.image_bytes != pih.step * pih.height
            or pih.image_bytes > MAX_IMAGE_BYTES
        ):
            raise ValueError("invalid extra image header")
        extra_headers.append(pih)

    frames = [
        bgr_payload_to_pil(
            recv_exact(sock, header.image_bytes),
            int(header.width),
            int(header.height),
            int(header.step),
        )
    ]
    for pih in extra_headers:
        frames.append(
            bgr_payload_to_pil(
                recv_exact(sock, pih.image_bytes),
                int(pih.width),
                int(pih.height),
                int(pih.step),
            )
        )

    timestamps: list[float] = []
    if has_timestamps:
        raw = recv_exact(sock, header.timestamp_count * ctypes.sizeof(ctypes.c_double))
        values = (ctypes.c_double * header.timestamp_count).from_buffer_copy(raw)
        timestamps = [float(v) for v in values]
        if any(not math.isfinite(v) for v in timestamps):
            raise ValueError("timestamps must be finite")
        if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
            raise ValueError("timestamps must be strictly increasing")

    system_message = ""
    if is_structured and header.system_bytes:
        system_message = recv_exact(sock, header.system_bytes).decode("utf-8")
    prompt = recv_exact(sock, header.prompt_bytes).decode("utf-8")

    history: list[tuple[str, str]] = []
    if is_structured:
        for _ in range(header.history_count):
            eh = recv_struct(sock, HistoryEntryHeader)
            if eh.user_bytes > MAX_TEXT_BYTES or eh.asst_bytes > MAX_TEXT_BYTES:
                raise ValueError("history entry exceeds protocol limit")
            user_text = recv_exact(sock, eh.user_bytes).decode("utf-8")
            asst_text = recv_exact(sock, eh.asst_bytes).decode("utf-8")
            history.append((user_text, asst_text))

    return ParsedRequest(header, frames, timestamps, system_message, prompt, history)


def effective_fps(req: ParsedRequest) -> float:
    if req.header.schema_flags & SCHEMA_FLAG_HAS_FPS:
        return float(req.header.fps)
    if len(req.timestamps) >= 2:
        dts = [b - a for a, b in zip(req.timestamps, req.timestamps[1:])]
        dt = statistics.median(dts)
        if dt > 0.0:
            return 1.0 / dt
    return 1.0


def build_messages(req: ParsedRequest):
    messages: list[dict] = []
    if req.system_message:
        messages.append({"role": "system", "content": [{"type": "text", "text": req.system_message}]})
    for user_text, asst_text in req.history:
        messages.append({"role": "user", "content": [{"type": "text", "text": user_text}]})
        messages.append({"role": "assistant", "content": [{"type": "text", "text": asst_text}]})

    if req.header.sequence_type in (SEQUENCE_TEMPORAL_IMAGES, SEQUENCE_VIDEO):
        media = {
            "type": "video",
            "video": req.frames,
            "fps": effective_fps(req),
        }
        content = [media, {"type": "text", "text": req.prompt}]
    else:
        content = [{"type": "image", "image": frame} for frame in req.frames]
        content.append({"type": "text", "text": req.prompt})
    messages.append({"role": "user", "content": content})
    return messages


def infer(engine, processor, req: ParsedRequest, engine_max_new_tokens: int):
    if req.header.max_generate_length <= 0:
        raise ValueError("max_generate_length must be > 0")
    if req.header.max_generate_length > engine_max_new_tokens:
        raise ValueError(
            f"request max_generate_length={req.header.max_generate_length} exceeds "
            f"worker engine_max_new_tokens={engine_max_new_tokens}"
        )

    start = time.perf_counter()
    messages = build_messages(req)
    pin = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    input_ids = pin["input_ids"].reshape(-1)

    kwargs = {}
    if req.header.sequence_type in (SEQUENCE_TEMPORAL_IMAGES, SEQUENCE_VIDEO):
        kwargs = {
            "pixel_values": pin["pixel_values_videos"],
            "grid_thw": pin["video_grid_thw"],
            "is_video": True,
        }
        temporal_encoding = "flashrt_cosmos3_native_video"
    elif len(req.frames) > 1:
        kwargs = {
            "pixel_values": pin["pixel_values"],
            "grid_thw": pin["image_grid_thw"],
            "is_video": False,
        }
        temporal_encoding = "flashrt_ordered_multi_image"
    else:
        kwargs = {
            "pixel_values": pin["pixel_values"],
            "grid_thw": pin["image_grid_thw"],
            "is_video": False,
        }
        temporal_encoding = "flashrt_single_image"

    out, _stats = engine.generate(
        input_ids,
        max_new_tokens=int(req.header.max_generate_length),
        ignore_eos=False,
        **kwargs,
    )
    tok = getattr(processor, "tokenizer", processor)
    eos = getattr(engine, "eos_token_id", None)
    tokens = [int(t) for t in out if eos is None or int(t) != int(eos)]
    text = tok.decode(tokens, skip_special_tokens=True)
    elapsed = time.perf_counter() - start
    return text, elapsed, temporal_encoding


def send_response(
    sock: socket.socket,
    request_id: int,
    *,
    success: bool,
    text: str = "",
    error: str = "",
    inference_seconds: float = 0.0,
    temporal_encoding: str = "",
    temporal_fallback_used: bool = False,
) -> None:
    text_b = text.encode("utf-8")
    error_b = error.encode("utf-8")
    enc_b = temporal_encoding.encode("utf-8")
    if max(len(text_b), len(error_b), len(enc_b)) > MAX_TEXT_BYTES:
        raise ValueError("response exceeds protocol text limit")
    header = ResponseHeader()
    header.magic = MAGIC
    header.version = VERSION
    header.request_id = request_id
    header.success = 1 if success else 0
    header.text_bytes = len(text_b)
    header.error_bytes = len(error_b)
    header.temporal_encoding_bytes = len(enc_b)
    header.temporal_fallback_used = 1 if temporal_fallback_used else 0
    header.inference_seconds = inference_seconds
    send_struct(sock, header)
    sock.sendall(text_b)
    sock.sendall(error_b)
    sock.sendall(enc_b)


def serve(args) -> None:
    from flash_rt.models.cosmos3_reasoner.pipeline_thor import CosmosReasonerThor

    processor = build_processor(args.checkpoint, args.cosmos_framework_root)
    engine = CosmosReasonerThor(
        args.checkpoint,
        max_new_tokens=args.engine_max_new_tokens,
        quant=args.quant,
        use_graph=not args.no_graph,
    )

    socket_path = Path(args.socket_path)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        socket_path.unlink()
    except FileNotFoundError:
        pass

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(socket_path))
    os.chmod(socket_path, 0o666)
    server.listen(1)
    server.settimeout(1.0)
    print(
        f"flashrt_cosmos3_worker ready on {socket_path} "
        f"quant={args.quant} checkpoint={args.checkpoint}",
        flush=True,
    )

    stopping = False

    def stop_handler(_signum, _frame):
        nonlocal stopping
        stopping = True
        try:
            server.close()
        except OSError:
            pass

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)

    try:
        while not stopping:
            try:
                client, _ = server.accept()
            except socket.timeout:
                continue
            except OSError:
                if stopping:
                    break
                raise
            with client:
                while not stopping:
                    try:
                        req = read_request(client)
                    except EOFError:
                        break
                    except Exception as exc:
                        print(f"IPC request parse failed: {exc}", flush=True)
                        break
                    try:
                        with DeadlineGuard(args.inference_deadline_seconds, req.header.request_id):
                            text, elapsed, encoding = infer(
                                engine, processor, req, args.engine_max_new_tokens
                            )
                        send_response(
                            client,
                            req.header.request_id,
                            success=True,
                            text=text,
                            inference_seconds=elapsed,
                            temporal_encoding=encoding,
                            temporal_fallback_used=False,
                        )
                    except Exception as exc:
                        send_response(
                            client,
                            req.header.request_id,
                            success=False,
                            error=str(exc),
                            temporal_encoding="flashrt_cosmos3_error",
                        )
    finally:
        try:
            server.close()
        except OSError:
            pass
        try:
            socket_path.unlink()
        except FileNotFoundError:
            pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--socket-path", default="/tmp/edge_vlm_flashrt.sock")
    ap.add_argument(
        "--cosmos-framework-root",
        default="/opt/cosmos-framework",
        help="Root containing cosmos_framework/ in the derived FlashRT image",
    )
    ap.add_argument("--quant", choices=("bf16", "fp4"), default="bf16")
    ap.add_argument("--engine-max-new-tokens", type=int, default=512)
    ap.add_argument("--inference-deadline-seconds", type=int, default=60)
    ap.add_argument("--no-graph", action="store_true")
    args = ap.parse_args()
    if args.engine_max_new_tokens <= 0:
        ap.error("--engine-max-new-tokens must be > 0")
    if args.inference_deadline_seconds <= 0:
        ap.error("--inference-deadline-seconds must be > 0")
    serve(args)


if __name__ == "__main__":
    main()
