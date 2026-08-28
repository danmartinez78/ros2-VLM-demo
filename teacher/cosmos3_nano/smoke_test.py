#!/usr/bin/env python3
"""Verify that the local Cosmos3-Nano vLLM server is healthy and generating."""

from __future__ import annotations

import argparse

from vllm_client import DEFAULT_BASE_URL, DEFAULT_MODEL, list_models, request_json


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

    payload = {
        "model": args.model,
        "messages": [
            {"role": "user", "content": "Reply with exactly: COSMOS3 NANO READY"}
        ],
        "temperature": 0,
        "max_tokens": 32,
    }
    status, response, elapsed = request_json(
        f"{args.base_url.rstrip('/')}/chat/completions", payload, args.timeout
    )
    content = response["choices"][0]["message"]["content"]
    if content.strip() != "COSMOS3 NANO READY":
        raise RuntimeError(f"unexpected smoke response: {content!r}")
    print(f"generation: OK HTTP={status} elapsed={elapsed:.3f}s")


if __name__ == "__main__":
    main()
