#!/usr/bin/env python3
"""Dry-run Thor command generator for nvidia/Cosmos3-Edge Phase 2 bring-up.

This module emits the exact sequence of commands required to acquire,
export, build, and smoke-test the Cosmos3-Edge checkpoint on Jetson AGX
Thor.  Commands are printed to stdout and validated for structural
correctness on CPU-only CI; nothing is executed.

No large downloads, TensorRT engine builds, or hardware operations are
performed.  Workspace paths are read from the runtime context (environment
or jp72_manifest.json defaults) and printed but never created here.

Usage::

    python3 scripts/models/cosmos3_edge_commands.py
    python3 scripts/models/cosmos3_edge_commands.py --validate
    python3 scripts/models/cosmos3_edge_commands.py --json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shlex
import sys
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
SCRIPTS_DIR = THIS_FILE.parent.parent
MODELCTL_PATH = THIS_FILE.parent / "modelctl.py"

_spec = importlib.util.spec_from_file_location("edge_vlm_modelctl", MODELCTL_PATH)
_modelctl = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _modelctl
assert _spec.loader is not None
_spec.loader.exec_module(_modelctl)

MODEL_ID = "cosmos3-edge"
PROFILE_ID = "cosmos3-edge-thor-f8"

# Smoke-test frame count for Phase 2.  Phase 2 starts with the documented
# single-image (F1) path per the pinned TRT Edge-LLM 0.10.0 guide.
# Multi-frame / native-video smoke tests are deferred until the Cosmos3
# reasoner input contract is confirmed on hardware.
SMOKE_FRAME_COUNTS = [1]

# HuggingFace model identifier (no secrets — public model card).
HF_MODEL_ID = "nvidia/Cosmos3-Edge"

# Committed smoke-input fixture relative to this file's directory.
# The Phase-2 operator must replace <absolute_path_to_image> with a real
# image path before executing on hardware.
SMOKE_INPUT_FIXTURE = THIS_FILE.parent / "cosmos3_edge_smoke_input_f1.json"

# Sub-directory written by tensorrt-edgellm-export under <model_root>/onnx/.
# Per the pinned TRT Edge-LLM 0.10.0 guide the export command is:
#   tensorrt-edgellm-export <checkpoint> <onnx_dir/reasoning> --task reasoning
# producing two sub-trees: reasoning/llm and reasoning/visual.
EXPORT_SUBDIR = "reasoning"


def _ctx() -> "_modelctl.RuntimeContext":  # type: ignore[name-defined]
    return _modelctl._runtime_context()


def _model() -> "_modelctl.ModelRecord":  # type: ignore[name-defined]
    models, _, _ = _modelctl.load_registries()
    return _modelctl.resolve_model(MODEL_ID, models)


def _profile() -> "_modelctl.ProfileRecord":  # type: ignore[name-defined]
    _, profiles, _ = _modelctl.load_registries()
    return _modelctl.resolve_profile(PROFILE_ID, profiles)


# ── individual command builders ───────────────────────────────────────────────

def acquire_command(ctx: "_modelctl.RuntimeContext", model: "_modelctl.ModelRecord") -> list[str]:
    """Step 1: download checkpoint from HuggingFace Hub."""
    model_root = _modelctl.model_root(model, ctx)
    return [
        "huggingface-cli",
        "download",
        HF_MODEL_ID,
        "--local-dir",
        str(model_root / "hf_checkpoint"),
        "--local-dir-use-symlinks",
        "False",
    ]


def export_command(
    ctx: "_modelctl.RuntimeContext",
    model: "_modelctl.ModelRecord",
) -> list[str]:
    """Step 2: export reasoning ONNX via tensorrt-edgellm-export.

    Uses the documented Cosmos3-Edge reasoner export workflow from the pinned
    TRT Edge-LLM 0.10.0 guide::

        tensorrt-edgellm-export <checkpoint> <onnx_dir/reasoning> --task reasoning
    """
    model_root = _modelctl.model_root(model, ctx)
    return [
        "tensorrt-edgellm-export",
        str(model_root / "hf_checkpoint"),
        str(model_root / "onnx" / "reasoning"),
        "--task",
        "reasoning",
    ]


def build_llm_command(
    ctx: "_modelctl.RuntimeContext",
    model: "_modelctl.ModelRecord",
    profile: "_modelctl.ProfileRecord",
) -> list[str]:
    """Step 3a: build LLM TensorRT engine.

    Reads the exported ONNX from the Cosmos3-Edge reasoner export path
    ``<model_root>/onnx/reasoning/llm``, which is the layout written by
    ``tensorrt-edgellm-export … --task reasoning``.
    """
    paths = _modelctl.engine_paths(model, profile, ctx)
    model_root = _modelctl.model_root(model, ctx)
    onnx_llm_dir = model_root / "onnx" / EXPORT_SUBDIR / "llm"
    return [
        str(Path(ctx.edge_build) / "examples" / "llm" / "llm_build"),
        "--onnxDir",
        str(onnx_llm_dir),
        "--engineDir",
        str(paths.llm_dir),
        "--maxBatchSize",
        str(profile.llm["maxBatchSize"]),
        "--maxInputLen",
        str(profile.llm["maxInputLen"]),
        "--maxKVCacheCapacity",
        str(profile.llm["maxKVCacheCapacity"]),
    ]


def build_visual_command(
    ctx: "_modelctl.RuntimeContext",
    model: "_modelctl.ModelRecord",
    profile: "_modelctl.ProfileRecord",
) -> list[str]:
    """Step 3b: build visual TensorRT engine.

    Reads the exported ONNX from the Cosmos3-Edge reasoner export path
    ``<model_root>/onnx/reasoning/visual``, which is the layout written by
    ``tensorrt-edgellm-export … --task reasoning``.
    """
    paths = _modelctl.engine_paths(model, profile, ctx)
    model_root = _modelctl.model_root(model, ctx)
    onnx_visual_dir = model_root / "onnx" / EXPORT_SUBDIR / "visual"
    command = [
        str(Path(ctx.edge_build) / "examples" / "multimodal" / "visual_build"),
        "--onnxDir",
        str(onnx_visual_dir),
        "--engineDir",
        str(paths.multimodal_dir),
    ]
    if profile.visual.get("maxImageTokens") is not None:
        command.extend(["--maxImageTokens", str(profile.visual["maxImageTokens"])])
    if profile.visual.get("maxImageTokensPerImage") is not None:
        command.extend(["--maxImageTokensPerImage", str(profile.visual["maxImageTokensPerImage"])])
    return command


def smoke_inference_command(
    ctx: "_modelctl.RuntimeContext",
    model: "_modelctl.ModelRecord",
    profile: "_modelctl.ProfileRecord",
    frame_count: int,
) -> list[str]:
    """Step 4: run smoke inference for a given number of frames.

    Uses the documented Phase-2 CLI contract from the pinned TRT Edge-LLM
    0.10.0 guide::

        build/examples/llm/llm_inference \\
            --engineDir <llm_dir> \\
            --multimodalEngineDir <visual_dir> \\
            --inputFile <smoke_input.json> \\
            --outputFile <smoke_output.json>

    The input JSON uses the documented image message format.  Phase 2 starts
    with the single-image (F1) path; multi-frame / native-video smoke tests
    are deferred until the Cosmos3 reasoner parser/runner contract is confirmed
    on hardware.
    """
    paths = _modelctl.engine_paths(model, profile, ctx)
    label = f"f{frame_count}"
    return [
        str(Path(ctx.edge_build) / "examples" / "llm" / "llm_inference"),
        "--engineDir",
        str(paths.llm_dir),
        "--multimodalEngineDir",
        str(paths.multimodal_dir),
        "--inputFile",
        str(SMOKE_INPUT_FIXTURE),
        "--outputFile",
        str(_modelctl.model_root(model, ctx) / f"smoke_output_{label}.json"),
    ]


def provenance_capture_command(
    ctx: "_modelctl.RuntimeContext",
    model: "_modelctl.ModelRecord",
    profile: "_modelctl.ProfileRecord",
) -> list[str]:
    """Step 5: capture provenance manifest via modelctl."""
    return [
        "python3",
        str(MODELCTL_PATH),
        "validate",
        model.model_id,
        profile.profile_id,
    ]


# ── full procedure ─────────────────────────────────────────────────────────────

def build_procedure(
    ctx: "_modelctl.RuntimeContext",
    model: "_modelctl.ModelRecord",
    profile: "_modelctl.ProfileRecord",
) -> list[tuple[str, list[str]]]:
    """Return the ordered list of (label, command) pairs for Phase 2 Thor bring-up."""
    steps: list[tuple[str, list[str]]] = [
        ("1. Checkpoint acquisition", acquire_command(ctx, model)),
        ("2. ONNX export (tensorrt-edgellm-export)", export_command(ctx, model)),
        ("3a. LLM engine build", build_llm_command(ctx, model, profile)),
        ("3b. Visual engine build", build_visual_command(ctx, model, profile)),
    ]
    for n_frames in SMOKE_FRAME_COUNTS:
        steps.append(
            (f"4. Smoke inference (F{n_frames})", smoke_inference_command(ctx, model, profile, n_frames)),
        )
    steps.append(
        ("5. Provenance / manifest capture", provenance_capture_command(ctx, model, profile)),
    )
    return steps


# ── validation ────────────────────────────────────────────────────────────────

def _required_flags() -> dict[str, list[str]]:
    """Build the required-step/token map dynamically from SMOKE_FRAME_COUNTS."""
    flags: dict[str, list[str]] = {
        "1. Checkpoint acquisition": ["huggingface-cli", "--local-dir"],
        "2. ONNX export (tensorrt-edgellm-export)": [
            "tensorrt-edgellm-export",
            "--task",
            "reasoning",
        ],
        "3a. LLM engine build": ["llm_build", "--onnxDir", "--engineDir", "--maxBatchSize"],
        "3b. Visual engine build": ["visual_build", "--onnxDir", "--engineDir"],
        "5. Provenance / manifest capture": ["modelctl.py", "validate"],
    }
    for n in SMOKE_FRAME_COUNTS:
        flags[f"4. Smoke inference (F{n})"] = [
            "llm_inference",
            "--engineDir",
            "--multimodalEngineDir",
            "--inputFile",
            "--outputFile",
        ]
    return flags


def validate_procedure(steps: list[tuple[str, list[str]]]) -> list[str]:
    """Return a list of validation errors; empty list means all checks passed."""
    errors: list[str] = []

    for required_label, required_tokens in _required_flags().items():
        matched_steps = [(label, cmd) for label, cmd in steps if label == required_label]
        if not matched_steps:
            errors.append(f"Missing required step: '{required_label}'")
            continue
        for label, cmd in matched_steps:
            cmd_str = " ".join(cmd)
            for token in required_tokens:
                if token not in cmd_str:
                    errors.append(f"Step '{label}': missing required token '{token}'")

    # Ensure acquire does not reference a policy component.
    for label, cmd in steps:
        if "acquire" in label.lower() or "checkpoint" in label.lower():
            cmd_str = " ".join(cmd)
            for forbidden in ("Policy-DROID", "und_prefill", "gen_engine", "vae_encoder"):
                if forbidden in cmd_str:
                    errors.append(
                        f"Step '{label}': policy-only token '{forbidden}' must not appear "
                        "in Cosmos3-Edge base checkpoint acquisition."
                    )

    # Ensure no engine paths are shared with Cosmos-Reason2.
    for label, cmd in steps:
        cmd_str = " ".join(cmd)
        if "Cosmos-Reason2" in cmd_str and "Cosmos3-Edge" not in cmd_str:
            errors.append(
                f"Step '{label}': command references a Cosmos-Reason2 path. "
                "Cosmos3-Edge must use an isolated workspace."
            )

    # Ensure the export step does not use a policy-style or invented tool.
    for label, cmd in steps:
        if "export" in label.lower():
            cmd_str = " ".join(cmd)
            for forbidden in ("modelopt.onnx.quantization", "cosmos3_policy_inference"):
                if forbidden in cmd_str:
                    errors.append(
                        f"Step '{label}': '{forbidden}' is not the documented Cosmos3-Edge "
                        "reasoner export workflow. Use tensorrt-edgellm-export --task reasoning."
                    )
    return errors


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Emit or validate the Phase 2 Thor bring-up commands for nvidia/Cosmos3-Edge. "
        "Commands are printed to stdout only; nothing is executed."
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate structural correctness of the generated commands and exit.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit commands as a JSON array of {label, command} objects.",
    )
    args = parser.parse_args(argv)

    try:
        ctx = _ctx()
        model = _model()
        profile = _profile()
    except Exception as exc:
        print(f"ERROR: failed to load registry: {exc}", file=sys.stderr)
        return 1

    steps = build_procedure(ctx, model, profile)

    if args.validate:
        errors = validate_procedure(steps)
        if errors:
            for error in errors:
                print(f"VALIDATION ERROR: {error}", file=sys.stderr)
            return 1
        print("Validation OK: all required steps and tokens present.")
        return 0

    if args.json:
        payload = [{"label": label, "command": cmd} for label, cmd in steps]
        print(json.dumps(payload, indent=2))
        return 0

    for label, cmd in steps:
        print(f"\n# {label}")
        print(shlex.join(cmd))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
