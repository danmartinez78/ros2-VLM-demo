#!/usr/bin/env python3
"""Python mirror of edge_vlm_ros IPC protocol v3.

This module intentionally mirrors include/edge_vlm_ros/ipc_protocol.hpp so the
FlashRT worker and experimental Python ROS bridge can interoperate with the
existing C++ IpcInferenceBackend without changing the wire contract.
"""
from __future__ import annotations

import ctypes
import socket

MAGIC = 0x45564C4D
VERSION = 3
ENCODING_BGR8 = 1
MAX_TEXT_BYTES = 1024 * 1024
MAX_IMAGE_BYTES = 256 * 1024 * 1024
MAX_HISTORY_ENTRIES = 256
MAX_EXTRA_IMAGES = 31

SCHEMA_FLAG_STRUCTURED = 1 << 0
SCHEMA_FLAG_SYS_CACHE = 1 << 1
SCHEMA_FLAG_MULTI_IMAGE = 1 << 2
SCHEMA_FLAG_HAS_FPS = 1 << 3
SCHEMA_FLAG_HAS_FRAME_TIMESTAMPS = 1 << 4

SEQUENCE_IMAGES = 0
SEQUENCE_TEMPORAL_IMAGES = 1
SEQUENCE_VIDEO = 2


class RequestHeader(ctypes.Structure):
    _fields_ = [
        ("magic", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("request_id", ctypes.c_uint64),
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("step", ctypes.c_uint32),
        ("encoding", ctypes.c_uint32),
        ("image_bytes", ctypes.c_uint32),
        ("prompt_bytes", ctypes.c_uint32),
        ("max_generate_length", ctypes.c_int32),
        ("temperature", ctypes.c_float),
        ("top_p", ctypes.c_float),
        ("top_k", ctypes.c_int32),
        ("schema_flags", ctypes.c_uint32),
        ("system_bytes", ctypes.c_uint32),
        ("history_count", ctypes.c_uint32),
        ("image_count", ctypes.c_uint32),
        ("sequence_type", ctypes.c_uint32),
        ("fps", ctypes.c_double),
        ("timestamp_count", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


class PerImageHeader(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("step", ctypes.c_uint32),
        ("image_bytes", ctypes.c_uint32),
    ]


class HistoryEntryHeader(ctypes.Structure):
    _fields_ = [
        ("user_bytes", ctypes.c_uint32),
        ("asst_bytes", ctypes.c_uint32),
    ]


class ResponseHeader(ctypes.Structure):
    _fields_ = [
        ("magic", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("request_id", ctypes.c_uint64),
        ("success", ctypes.c_uint32),
        ("text_bytes", ctypes.c_uint32),
        ("error_bytes", ctypes.c_uint32),
        ("temporal_encoding_bytes", ctypes.c_uint32),
        ("temporal_fallback_used", ctypes.c_uint32),
        ("inference_seconds", ctypes.c_double),
    ]


# ABI guardrails. These sizes match the C++ structs on aarch64/x86_64 Linux.
assert ctypes.sizeof(RequestHeader) == 96
assert ctypes.sizeof(PerImageHeader) == 16
assert ctypes.sizeof(HistoryEntryHeader) == 8
assert ctypes.sizeof(ResponseHeader) == 48


def recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError("IPC peer closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_struct(sock: socket.socket, cls):
    return cls.from_buffer_copy(recv_exact(sock, ctypes.sizeof(cls)))


def send_struct(sock: socket.socket, value: ctypes.Structure) -> None:
    sock.sendall(bytes(value))
