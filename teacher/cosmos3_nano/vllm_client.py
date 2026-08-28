#!/usr/bin/env python3
"""Minimal OpenAI-compatible client for the local Cosmos3-Nano vLLM teacher."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "cosmos3-nano-teacher"
DEFAULT_PROMPT = (
    "Analyze the complete video sequence. Report only temporal facts supported by "
    "changes across multiple frames. Distinguish instantaneous pose or apparent "
    "action from actual displacement. Do not infer displacement from pose alone. "
    "Do not report demographics, clothing details, facial characteristics, or other "
    "static appearance unless needed to identify an object. For every displacement "
    "claim, briefly state the visual evidence across frames."
)


def request_json(url: str, payload: dict[str, Any] | None, timeout: float) -> tuple[int, dict, float]:
    data = None
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            status = response.status
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
    elapsed = time.perf_counter() - start
    return status, json.loads(body), elapsed


def list_models(base_url: str = DEFAULT_BASE_URL, timeout: float = 10.0) -> dict:
    _, payload, _ = request_json(f"{base_url.rstrip('/')}/models", None, timeout)
    return payload


def load_schema(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_temporal_video(
    video_container_path: str,
    *,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    schema: dict,
    prompt: str = DEFAULT_PROMPT,
    max_tokens: int = 512,
    timeout: float = 180.0,
) -> dict:
    if not video_container_path.startswith("/data/"):
        raise ValueError(
            "video_container_path must be inside the vLLM /data mount, for example "
            "/data/generated/capture_x/forward.mp4"
        )

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {"url": f"file://{video_container_path}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "temporal_observation",
                "schema": schema,
            },
        },
    }

    status, response, elapsed = request_json(
        f"{base_url.rstrip('/')}/chat/completions", payload, timeout
    )
    content = response["choices"][0]["message"]["content"]
    try:
        parsed_content = json.loads(content)
    except json.JSONDecodeError:
        parsed_content = None

    return {
        "http_status": status,
        "elapsed_seconds": elapsed,
        "request": {
            "model": model,
            "video_container_path": video_container_path,
            "max_tokens": max_tokens,
            "prompt": prompt,
        },
        "parsed_content": parsed_content,
        "response": response,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Call the local Cosmos3-Nano teacher")
    parser.add_argument("video_container_path")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path(__file__).with_name("temporal_observation_v1.schema.json"),
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    schema = load_schema(args.schema)
    result = run_temporal_video(
        args.video_container_path,
        base_url=args.base_url,
        model=args.model,
        schema=schema,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(result["parsed_content"], indent=2))
    print(f"elapsed_seconds={result['elapsed_seconds']:.3f}")


if __name__ == "__main__":
    main()
