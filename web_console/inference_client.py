# Copyright 2025 edge_vlm_ros contributors
"""
Standalone image inference via the existing edge_vlm_cli binary.

Calls the existing IPC client/service boundary; never reimplements the
IPC protocol in Python and never uses shell=True.
"""
from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from typing import Optional

_DEFAULT_TIMEOUT_SECONDS: int = 120

# Allowed image MIME types / extensions accepted by edge_vlm_cli → OpenCV.
ALLOWED_IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
)
# Maximum upload size enforced before writing to disk.
MAX_IMAGE_BYTES: int = 64 * 1024 * 1024  # 64 MiB


@dataclass
class InferenceResult:
    success: bool
    text: str = ""
    error: str = ""
    inference_seconds: float = 0.0


def run_inference(
    cli_path: str,
    socket_path: str,
    image_path: str,
    prompt: str,
    max_generate_length: int = 64,
    temperature: float = 0.2,
    top_p: float = 0.9,
    top_k: int = 20,
    timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
) -> InferenceResult:
    """Run edge_vlm_cli as a subprocess and return a structured result.

    Never uses shell=True; arguments are constructed as a list.
    """
    if max_generate_length < 1 or max_generate_length > 4096:
        return InferenceResult(
            success=False,
            error=f"max_generate_length must be in [1, 4096]; got {max_generate_length}",
        )
    if not (0.0 <= temperature <= 2.0):
        return InferenceResult(
            success=False,
            error=f"temperature must be in [0.0, 2.0]; got {temperature}",
        )
    if not (0.0 < top_p <= 1.0):
        return InferenceResult(
            success=False,
            error=f"top_p must be in (0.0, 1.0]; got {top_p}",
        )
    if top_k < 1:
        return InferenceResult(
            success=False,
            error=f"top_k must be >= 1; got {top_k}",
        )

    args = [
        cli_path,
        "--socket", socket_path,
        "--image", image_path,
        "--prompt", prompt,
        "--max-generate-length", str(max_generate_length),
        "--temperature", str(temperature),
        "--top-p", str(top_p),
        "--top-k", str(top_k),
    ]

    t0 = time.monotonic()
    try:
        proc_result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        elapsed = time.monotonic() - t0
        if proc_result.returncode == 0:
            return InferenceResult(
                success=True,
                text=proc_result.stdout.strip(),
                inference_seconds=elapsed,
            )
        return InferenceResult(
            success=False,
            error=(proc_result.stderr.strip() or proc_result.stdout.strip()),
            inference_seconds=elapsed,
        )
    except subprocess.TimeoutExpired:
        return InferenceResult(
            success=False,
            error=f"Inference timed out after {timeout_seconds} s",
            inference_seconds=float(timeout_seconds),
        )
    except FileNotFoundError:
        return InferenceResult(
            success=False,
            error=f"edge_vlm_cli not found: {cli_path!r}",
        )
