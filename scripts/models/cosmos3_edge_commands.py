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

# Smoke-test frame counts for Phase 2.
SMOKE_FRAME_COUNTS = [1, 4, 8]

# HuggingFace model identifier (no secrets — public model card).
HF_MODEL_ID = "nvidia/Cosmos3-Edge"

# ModelOpt quantization container (from jp72_manifest.json).
PYTORCH_CONTAINER = "nvcr.io/nvidia/pytorch:26.05-py3"
MODELOPT_VERSION = "0.45.0"


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


def export_quantize_command(
    ctx: "_modelctl.RuntimeContext",
    model: "_modelctl.ModelRecord",
) -> list[str]:
    """Step 2: quantize and export to ONNX via ModelOpt in PyTorch container."""
    model_root = _modelctl.model_root(model, ctx)
    # The container-internal paths mirror the workspace layout.
    return [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "-v",
        f"{model_root}:{model_root}",
        PYTORCH_CONTAINER,
        "python3",
        "-m",
        "modelopt.onnx.quantization",
        "--model_path",
        str(model_root / "hf_checkpoint"),
        "--quant_type",
        str(model.manifest_model.get("quantization", "nvfp4")),
        "--output_path",
        str(model_root / "onnx"),
    ]


def build_llm_command(
    ctx: "_modelctl.RuntimeContext",
    model: "_modelctl.ModelRecord",
    profile: "_modelctl.ProfileRecord",
) -> list[str]:
    """Step 3a: build LLM TensorRT engine."""
    return _modelctl.build_llm_command(model, profile, ctx)


def build_visual_command(
    ctx: "_modelctl.RuntimeContext",
    model: "_modelctl.ModelRecord",
    profile: "_modelctl.ProfileRecord",
) -> list[str]:
    """Step 3b: build visual TensorRT engine."""
    return _modelctl.build_visual_command(model, profile, ctx)


def smoke_inference_command(
    ctx: "_modelctl.RuntimeContext",
    model: "_modelctl.ModelRecord",
    profile: "_modelctl.ProfileRecord",
    frame_count: int,
) -> list[str]:
    """Step 4: run smoke inference for a given number of video frames."""
    paths = _modelctl.engine_paths(model, profile, ctx)
    return [
        str(Path(ctx.edge_build) / "examples" / "multimodal" / "llm_inference"),
        "--llmEngineDir",
        str(paths.llm_dir),
        "--multimodalEngineDir",
        str(paths.multimodal_dir),
        "--pluginPath",
        ctx.plugin_path,
        "--maxGenerateLength",
        "256",
        "--frameCount",
        str(frame_count),
        "--inputPrompt",
        "Describe the scene.",
    ]


def provenance_capture_command(
    ctx: "_modelctl.RuntimeContext",
    model: "_modelctl.ModelRecord",
    profile: "_modelctl.ProfileRecord",
) -> list[str]:
    """Step 5: capture provenance manifest via modelctl."""
    paths = _modelctl.engine_paths(model, profile, ctx)
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
        ("2. ONNX export / quantization", export_quantize_command(ctx, model)),
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
        "3a. LLM engine build": ["llm_build", "--onnxDir", "--engineDir", "--maxBatchSize"],
        "3b. Visual engine build": ["visual_build", "--onnxDir", "--engineDir"],
        "5. Provenance / manifest capture": ["modelctl.py", "validate"],
    }
    for n in SMOKE_FRAME_COUNTS:
        flags[f"4. Smoke inference (F{n})"] = ["llm_inference", "--frameCount"]
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
            # For smoke steps, verify the frame count argument value matches the label.
            if "Smoke inference" in label and "(F" in label:
                frame_str = label.split("(F")[1].rstrip(")")
                if "--frameCount" in cmd:
                    idx = cmd.index("--frameCount")
                    if idx + 1 < len(cmd) and cmd[idx + 1] != frame_str:
                        errors.append(
                            f"Step '{label}': --frameCount value {cmd[idx + 1]!r} does not match label frame count {frame_str!r}"
                        )

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
