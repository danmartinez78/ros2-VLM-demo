#!/usr/bin/env python3
"""Run saved temporal variants through the local Cosmos3-Nano teacher server."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from vllm_client import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    list_models,
    load_schema,
    run_temporal_video,
)

DEFAULT_VARIANTS = ("forward", "reverse", "shuffled", "static_terminal")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a temporal teacher evaluation suite")
    parser.add_argument("media_manifest", type=Path)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path(__file__).with_name("temporal_observation_v1.schema.json"),
    )
    parser.add_argument(
        "--variants",
        default=",".join(DEFAULT_VARIANTS),
        help="Comma-separated subset of variants to run",
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    media_manifest_path = args.media_manifest.resolve()
    media_manifest = json.loads(media_manifest_path.read_text(encoding="utf-8"))
    schema = load_schema(args.schema)

    models = list_models(args.base_url, timeout=min(args.timeout, 30.0))
    model_ids = [item.get("id") for item in models.get("data", [])]
    if args.model not in model_ids:
        raise RuntimeError(f"model {args.model!r} not served; available={model_ids}")

    selected = [value.strip() for value in args.variants.split(",") if value.strip()]
    unknown = [name for name in selected if name not in media_manifest["variants"]]
    if unknown:
        raise ValueError(f"variants not present in manifest: {unknown}")

    results: list[dict] = []
    for name in selected:
        variant = media_manifest["variants"][name]
        print(f"running {name} -> {variant['container_path']}")
        result = run_temporal_video(
            variant["container_path"],
            base_url=args.base_url,
            model=args.model,
            schema=schema,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )
        results.append(
            {
                "variant": name,
                "frame_order": variant["frame_order"],
                "source_probe": variant.get("probe"),
                **result,
            }
        )
        print(
            f"  elapsed={result['elapsed_seconds']:.3f}s "
            f"output={json.dumps(result['parsed_content'], sort_keys=True)}"
        )

    output = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "base_url": args.base_url,
        "media_manifest": str(media_manifest_path),
        "teacher_schema": str(args.schema.resolve()),
        "results": results,
    }

    output_path = args.output
    if output_path is None:
        output_path = media_manifest_path.with_name("cosmos3_nano_results.json")
    output_path = output_path.resolve()
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"results: {output_path}")


if __name__ == "__main__":
    main()
