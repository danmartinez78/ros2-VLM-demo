# Copyright 2025 edge_vlm_ros contributors
"""
Task profile discovery, loading, and structured-output parsing.

Profiles are versioned JSON files stored in a config directory.  The default
location is ``config/task_profiles/`` relative to the repository root, which
can be overridden via the ``TASK_PROFILES_DIR`` environment variable or the
``task_profiles_dir`` web-console config key.

A profile contains a system instruction, task prompt, schema version, and
optional output schema.  Each loaded profile is identified by a short
SHA-256 hash of its prompt text so that every experiment artifact records
exactly which prompt drove inference.

Structured-output parsing
--------------------------
When the model returns a valid JSON object that matches the profile's expected
keys, the parsed dict is returned alongside the raw text.  When parsing fails
(malformed JSON, non-object root, or missing keys), ``parsed_response`` is
``None`` and ``parse_success`` is ``False``.  The raw model text is always
preserved.  ``json.loads`` is the only parser used — model output is never
evaluated.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ── constants ─────────────────────────────────────────────────────────────────

_SCHEMA_VERSION = 1
_PROFILE_EXTENSIONS = frozenset({".json"})
# Maximum profile file size to prevent memory exhaustion.
_MAX_PROFILE_BYTES = 64 * 1024  # 64 KiB


# ── dataclasses ───────────────────────────────────────────────────────────────


@dataclass
class TaskProfile:
    """A fully resolved and hashed task profile.

    Every field that affects inference output is stored here so that
    experiment artifacts can be reproduced without consulting external state.
    """

    name: str
    """Stable machine name, e.g. ``warehouse_awareness``."""

    version: str
    """Human-readable version string, e.g. ``"1.0"``."""

    description: str

    system_instruction: str
    """System-level instruction prepended to the prompt."""

    task_prompt: str
    """The task-specific prompt sent to the model."""

    prompt_hash: str
    """Truncated SHA-256 of the full prompt (system_instruction + task_prompt).
    Recorded verbatim in every run artifact for reproducibility.
    """

    profile_path: str
    """Absolute path to the source JSON file on disk."""

    output_schema: Optional[Dict[str, Any]] = None
    """Optional JSON-schema-like dict describing the expected structured output."""

    schema_version: int = _SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "system_instruction": self.system_instruction,
            "task_prompt": self.task_prompt,
            "prompt_hash": self.prompt_hash,
            "profile_path": self.profile_path,
            "output_schema": self.output_schema,
        }

    def profile_id(self) -> str:
        """Return a stable profile identifier for use in run manifests."""
        safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", self.name)
        safe_ver = re.sub(r"[^a-zA-Z0-9_.-]", "_", self.version)
        return f"{safe_name}_v{safe_ver}"

    @property
    def schema_example(self) -> Optional[Dict[str, Any]]:
        """Alias for output_schema for API compatibility."""
        return self.output_schema


@dataclass
class ParsedOutput:
    """Result of attempting to parse structured model output.

    ``raw_text`` is always preserved.  ``parsed`` is non-None only when the
    model returned a valid JSON object that contained all required keys.
    ``parse_success`` is False on any parse failure.  ``parse_error`` records
    the human-readable failure reason.
    """

    raw_text: str
    parsed: Optional[Dict[str, Any]] = None
    parse_success: bool = False
    parse_error: str = ""
    malformed_flag: bool = False
    """True when the response appears to be structurally invalid (truncated,
    non-JSON preamble, or missing required keys)."""

    @property
    def raw(self) -> str:
        """Alias for raw_text for API compatibility."""
        return self.raw_text

    @property
    def parsed_ok(self) -> bool:
        """Alias for parse_success for API compatibility."""
        return self.parse_success


# ── helpers ───────────────────────────────────────────────────────────────────


def _prompt_hash(system_instruction: str, task_prompt: str) -> str:
    """Return a hex digest of system_instruction + "\\n\\n" + task_prompt."""
    combined = f"{system_instruction}\n\n{task_prompt}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def _load_profile_file(path: Path) -> Optional[TaskProfile]:
    """Parse a single profile JSON file and return a TaskProfile, or None on error.

    Errors (invalid JSON, oversized, missing required keys) are silently
    swallowed so a single bad profile does not prevent others from loading.
    """
    try:
        size = path.stat().st_size
        if size > _MAX_PROFILE_BYTES:
            return None
        raw = path.read_text(encoding="utf-8", errors="replace")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(data, dict):
        return None

    name = str(data.get("name", "") or path.stem)
    version = str(data.get("version", "1.0"))
    description = str(data.get("description", ""))
    system_instruction = str(data.get("system_instruction", ""))
    task_prompt = str(data.get("task_prompt", ""))

    if not task_prompt:
        return None

    ph = _prompt_hash(system_instruction, task_prompt)

    return TaskProfile(
        name=name,
        version=version,
        description=description,
        system_instruction=system_instruction,
        task_prompt=task_prompt,
        prompt_hash=ph,
        profile_path=str(path.resolve()),
        output_schema=data.get("output_schema"),
        schema_version=int(data.get("schema_version", _SCHEMA_VERSION)),
    )


# ── public API ────────────────────────────────────────────────────────────────


def default_profiles_dir() -> Path:
    """Return the default task-profiles directory (repo root / config / task_profiles)."""
    return Path(__file__).parent.parent / "config" / "task_profiles"


def discover_profiles(
    profiles_dir: Optional[str] = None,
) -> List[TaskProfile]:
    """Return all valid task profiles found under *profiles_dir*.

    Parameters
    ----------
    profiles_dir:
        Directory to scan for ``*.json`` profile files.  Defaults to the
        ``TASK_PROFILES_DIR`` environment variable, then the repository-root
        ``config/task_profiles/`` directory.

    Returns an empty list when the directory does not exist or is empty.
    """
    if profiles_dir is None:
        profiles_dir = os.environ.get("TASK_PROFILES_DIR", "")
    if not profiles_dir:
        profiles_dir = str(default_profiles_dir())

    root = Path(profiles_dir)
    if not root.is_dir():
        return []

    profiles: List[TaskProfile] = []
    try:
        entries = sorted(root.iterdir())
    except OSError:
        return []

    for entry in entries:
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in _PROFILE_EXTENSIONS:
            continue
        p = _load_profile_file(entry)
        if p is not None:
            profiles.append(p)

    return profiles


def get_profile_by_name(
    profiles: List[TaskProfile],
    name: str,
) -> Optional[TaskProfile]:
    """Return the first profile whose ``name`` field matches *name*, or None.

    Parameters
    ----------
    profiles:
        List of already-loaded TaskProfile objects (e.g. from discover_profiles).
    name:
        The profile name to search for.
    """
    for p in profiles:
        if p.name == name:
            return p
    return None


def parse_structured_output(
    raw_text: str,
    profile: Optional[TaskProfile] = None,
) -> ParsedOutput:
    """Attempt to parse structured JSON output from *raw_text*.

    Extracts the first JSON object found in the text (to handle models that
    emit preamble prose before the JSON).  Never uses ``eval``; only
    ``json.loads`` is used.  Required keys are sourced from
    ``profile.output_schema`` when available.

    Parameters
    ----------
    raw_text:
        The raw model response string.
    profile:
        Optional profile whose ``output_schema`` defines required keys.

    Returns
    -------
    ParsedOutput
        Always contains the original ``raw_text``.  ``parsed`` is non-None
        only on a fully successful parse.
    """
    if not raw_text or not raw_text.strip():
        return ParsedOutput(
            raw_text=raw_text,
            parse_success=False,
            parse_error="empty response",
            malformed_flag=True,
        )

    # Locate the first '{' to skip model preamble prose.
    brace_start = raw_text.find("{")
    if brace_start < 0:
        return ParsedOutput(
            raw_text=raw_text,
            parse_success=False,
            parse_error="no JSON object found in response",
            malformed_flag=True,
        )

    candidate = raw_text[brace_start:]

    # Find the matching closing brace (naive bracket counting — sufficient
    # for well-formed JSON; falls back to the full tail for truncated output).
    depth = 0
    end_idx = len(candidate)
    for i, ch in enumerate(candidate):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end_idx = i + 1
                break

    json_str = candidate[:end_idx]

    try:
        obj = json.loads(json_str)
    except json.JSONDecodeError as exc:
        return ParsedOutput(
            raw_text=raw_text,
            parse_success=False,
            parse_error=f"JSON parse error: {exc}",
            malformed_flag=True,
        )

    if not isinstance(obj, dict):
        return ParsedOutput(
            raw_text=raw_text,
            parse_success=False,
            parse_error="parsed value is not a JSON object",
            malformed_flag=True,
        )

    # Validate required keys when an output schema is provided.
    if profile is not None and profile.output_schema is not None:
        required_keys = profile.output_schema.get("required", [])
        missing = [k for k in required_keys if k not in obj]
        if missing:
            return ParsedOutput(
                raw_text=raw_text,
                parsed=obj,  # preserve partial parse
                parse_success=False,
                parse_error=f"missing required keys: {missing}",
                malformed_flag=True,
            )

    return ParsedOutput(
        raw_text=raw_text,
        parsed=obj,
        parse_success=True,
        parse_error="",
        malformed_flag=False,
    )
