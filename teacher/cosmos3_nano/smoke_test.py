#!/usr/bin/env python3
"""Verify that the local Cosmos3-Nano vLLM server is healthy and generating."""

from __future__ import annotations

import argparse

from vllm_client import DEFAULT_BASE_URL, DEFAULT_MODEL, list_models, request_json


def run_generation(base_url: str, model: str, timeout: float) -> tuple[int, float]:
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": "Reply with exactly: COSMOS3 NANO READY"}
        ],
        "temperature": 0,
        "max_tokens": 32,
    }
    status, response, elapsed = request_json(
        f"{base_url.rstrip('/')}/chat/completions", payload, timeout
    )
    content = response["choices"][0]["message"]["content"]
    if content.strip() != "COSMOS3 NANO READY":
        raise RuntimeError(f"unexpected smoke response: {content!r}")
    return status, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the Cosmos3-Nano teacher server")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    models = list_models(args.base_url, timeout=args.timeout)
    model_ids = [item.get("id") for item in models.get("data", [])]
    if args.model not in model_ids:
        raise RuntimeError(f"model {args.model!r} not served; available={model_ids}")
    print(f"model registration: OK ({args.model})")

    status, first_elapsed = run_generation(args.base_url, args.model, args.timeout)
    print(f"generation 1 (warmup candidate): OK HTTP={status} elapsed={first_elapsed:.3f}s")

    status, second_elapsed = run_generation(args.base_url, args.model, args.timeout)
    print(f"generation 2 (steady-state candidate): OK HTTP={status} elapsed={second_elapsed:.3f}s")

    if first_elapsed > second_elapsed * 2.0:
        print(
            f"warmup effect: first request was {first_elapsed / second_elapsed:.1f}x slower "
            "than the second request"
        )


if __name__ == "__main__":
    main()
