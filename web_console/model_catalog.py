# Copyright 2025 edge_vlm_ros contributors
"""
Model profile discovery for the web workbench.

Discovers configured engine bundles from environment variables and an optional
workspace directory.  Returns model-neutral profiles; no Cosmos-specific
assumptions are hard-coded.  When environment variables are absent the catalog
degrades gracefully to an empty list.

No model-switching logic is implemented here: switching requires an explicit
restart of the owned edge_vlm_server, which is the responsibility of the
experiment_stack.sh lifecycle script.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── dataclass ─────────────────────────────────────────────────────────────────


@dataclass
class ModelProfile:
    """Stable description of one discoverable model configuration.

    All paths are stored as strings so the profile can be serialised to JSON
    without any special encoder.
    """

    model_id: str
    """Opaque stable identifier derived from the model name and LLM engine path."""

    model_name: str
    """Human-readable name (e.g. ``Cosmos-Reason2-8B``)."""

    llm_engine_dir: str
    """Path to the TRT LLM engine directory (may be absent on disk)."""

    multimodal_engine_dir: str
    """Path to the multimodal engine directory (may be absent on disk)."""

    plugin_path: str
    """Path to the EdgeLLM TensorRT plugin shared library."""

    workspace_dir: str
    """Root workspace directory if discovered from EDGE_VLM_WORKSPACE_DIR."""

    llm_engine_exists: bool
    """True if the LLM engine directory exists on disk at discovery time."""

    multimodal_engine_exists: bool
    """True if the multimodal engine directory exists on disk at discovery time."""

    plugin_exists: bool
    """True if the plugin file exists on disk at discovery time."""

    is_active: bool
    """True if this is the currently active model (all env vars point to it)."""

    modalities: List[str] = field(default_factory=lambda: ["vision", "language"])
    """Declared modalities supported by this model."""

    notes: str = ""
    """Free-form notes or build metadata (e.g. quantization)."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "llm_engine_dir": self.llm_engine_dir,
            "multimodal_engine_dir": self.multimodal_engine_dir,
            "plugin_path": self.plugin_path,
            "workspace_dir": self.workspace_dir,
            "llm_engine_exists": self.llm_engine_exists,
            "multimodal_engine_exists": self.multimodal_engine_exists,
            "plugin_exists": self.plugin_exists,
            "is_active": self.is_active,
            "modalities": self.modalities,
            "notes": self.notes,
        }


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_model_id(model_name: str, llm_engine_dir: str) -> str:
    """Derive a stable opaque ID from model_name and llm_engine_dir."""
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", model_name or "unknown")
    if llm_engine_dir:
        # Hash the path so the ID is short but path-specific.
        path_suffix = abs(hash(llm_engine_dir)) % (10**8)
        return f"{safe_name}_{path_suffix}"
    return safe_name


def _profile_from_env() -> Optional[ModelProfile]:
    """Build a ModelProfile from the current environment variables.

    Returns None if neither LLM engine dir nor model name is set.

    Uses::

        EDGE_VLM_MODEL_NAME           — human-readable model name
        EDGE_VLM_LLM_ENGINE_DIR       — path to TRT LLM engine
        EDGE_VLM_MULTIMODAL_ENGINE_DIR — path to TRT multimodal engine
        EDGELLM_PLUGIN_PATH           — path to EdgeLLM plugin .so
        EDGE_VLM_WORKSPACE_DIR        — optional workspace root
    """
    model_name = os.environ.get("EDGE_VLM_MODEL_NAME", "")
    llm_dir = os.environ.get("EDGE_VLM_LLM_ENGINE_DIR", "")
    mm_dir = os.environ.get("EDGE_VLM_MULTIMODAL_ENGINE_DIR", "")
    plugin = os.environ.get("EDGELLM_PLUGIN_PATH", "")
    workspace = os.environ.get("EDGE_VLM_WORKSPACE_DIR", "")

    if not llm_dir and not model_name:
        return None

    # Derive model name from engine directory if missing.
    if not model_name and llm_dir:
        # e.g. /workspace/Cosmos-Reason2-8B/engine/llm → Cosmos-Reason2-8B
        parts = Path(llm_dir).parts
        if len(parts) >= 3:
            model_name = parts[-3]
        else:
            model_name = Path(llm_dir).name

    model_id = _make_model_id(model_name, llm_dir)

    return ModelProfile(
        model_id=model_id,
        model_name=model_name,
        llm_engine_dir=llm_dir,
        multimodal_engine_dir=mm_dir,
        plugin_path=plugin,
        workspace_dir=workspace,
        llm_engine_exists=bool(llm_dir and Path(llm_dir).is_dir()),
        multimodal_engine_exists=bool(mm_dir and Path(mm_dir).is_dir()),
        plugin_exists=bool(plugin and Path(plugin).is_file()),
        is_active=True,
    )


def _scan_workspace(workspace_dir: str) -> List[ModelProfile]:
    """Discover additional model profiles by scanning the workspace directory.

    Looks for subdirectories whose structure matches::

        <workspace>/<ModelName>/engine/llm/
        <workspace>/<ModelName>/engine/

    Returns an empty list when the directory does not exist or is unreadable.
    """
    if not workspace_dir:
        return []
    ws = Path(workspace_dir)
    if not ws.is_dir():
        return []

    active_llm_dir = os.environ.get("EDGE_VLM_LLM_ENGINE_DIR", "")
    active_mm_dir = os.environ.get("EDGE_VLM_MULTIMODAL_ENGINE_DIR", "")
    plugin = os.environ.get("EDGELLM_PLUGIN_PATH", "")

    profiles: List[ModelProfile] = []
    try:
        candidates = sorted(ws.iterdir())
    except OSError:
        return []

    for entry in candidates:
        if not entry.is_dir():
            continue
        llm_dir = entry / "engine" / "llm"
        mm_dir = entry / "engine"
        model_name = entry.name
        # Exclude non-model directories (hidden, common non-model names).
        if model_name.startswith(".") or model_name in ("lost+found", "tmp"):
            continue
        # Only include if at least the engine subdir structure is present.
        if not mm_dir.is_dir():
            continue

        llm_path = str(llm_dir)
        mm_path = str(mm_dir)
        is_active = (
            (active_llm_dir and Path(active_llm_dir).resolve() == llm_dir.resolve())
            or (active_mm_dir and Path(active_mm_dir).resolve() == mm_dir.resolve())
        )
        model_id = _make_model_id(model_name, llm_path)
        profiles.append(
            ModelProfile(
                model_id=model_id,
                model_name=model_name,
                llm_engine_dir=llm_path,
                multimodal_engine_dir=mm_path,
                plugin_path=plugin,
                workspace_dir=str(ws),
                llm_engine_exists=llm_dir.is_dir(),
                multimodal_engine_exists=mm_dir.is_dir(),
                plugin_exists=bool(plugin and Path(plugin).is_file()),
                is_active=is_active,
            )
        )
    return profiles


# ── public API ────────────────────────────────────────────────────────────────


def discover_models(
    workspace_dir: Optional[str] = None,
) -> List[ModelProfile]:
    """Return all discoverable model profiles.

    Discovery order:
    1. Active profile from environment variables (EDGE_VLM_MODEL_NAME etc.)
    2. Additional profiles found by scanning *workspace_dir*
       (defaults to EDGE_VLM_WORKSPACE_DIR env var).

    Profiles are deduplicated by model_id; the environment-variable profile
    (is_active=True) always appears first.
    """
    if workspace_dir is None:
        workspace_dir = os.environ.get("EDGE_VLM_WORKSPACE_DIR", "")

    seen: set = set()
    profiles: List[ModelProfile] = []

    active = _profile_from_env()
    if active is not None:
        profiles.append(active)
        seen.add(active.model_id)

    for p in _scan_workspace(workspace_dir):
        if p.model_id not in seen:
            profiles.append(p)
            seen.add(p.model_id)

    return profiles
