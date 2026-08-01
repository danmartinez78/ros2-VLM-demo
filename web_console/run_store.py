# Copyright 2025 edge_vlm_ros contributors
"""
On-disk run record storage for the web console.

Each run is stored as a directory under base_dir/<run_id>/manifest.json.
Path traversal is prevented by validating run IDs against a UUID pattern.
Oldest runs are evicted when the count exceeds _MAX_RUNS.
"""
from __future__ import annotations

import json
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

_MAX_RUNS: int = 100
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)


def _is_safe_run_id(run_id: str) -> bool:
    """Return True only if run_id is a standard UUID string (path-traversal guard)."""
    return bool(_UUID_RE.match(run_id))


class RunStore:
    """Manages run directories and manifests under a base directory."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ── factory ───────────────────────────────────────────────────────────────

    @staticmethod
    def new_run_id() -> str:
        return str(uuid.uuid4())

    # ── write ─────────────────────────────────────────────────────────────────

    def save_run(self, run_id: str, record: Dict[str, Any]) -> Path:
        """Write manifest.json for a run and evict oldest runs if necessary."""
        if not _is_safe_run_id(run_id):
            raise ValueError(f"Invalid run_id: {run_id!r}")
        run_dir = self.base_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = run_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2)
            fh.write("\n")
        self._evict_oldest()
        return manifest_path

    def write_artifact(self, run_id: str, filename: str, data: bytes) -> Path:
        """Write a binary artifact file to a run directory."""
        if not _is_safe_run_id(run_id):
            raise ValueError(f"Invalid run_id: {run_id!r}")
        if "/" in filename or "\\" in filename or filename.startswith("."):
            raise ValueError(f"Invalid artifact filename: {filename!r}")
        run_dir = self.base_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = run_dir / filename
        artifact_path.write_bytes(data)
        return artifact_path

    # ── read ──────────────────────────────────────────────────────────────────

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Return the manifest dict for a run, or None if not found."""
        if not _is_safe_run_id(run_id):
            return None
        manifest_path = self.base_dir / run_id / "manifest.json"
        if not manifest_path.is_file():
            return None
        try:
            with open(manifest_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            return None

    def list_runs(self) -> List[Dict[str, Any]]:
        """Return manifests for all runs, newest first."""
        entries = []
        try:
            dirs = sorted(
                self.base_dir.iterdir(),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        except OSError:
            return []
        for d in dirs:
            if not d.is_dir() or not _is_safe_run_id(d.name):
                continue
            manifest = self.get_run(d.name)
            if manifest is not None:
                entries.append(manifest)
        return entries

    def run_dir(self, run_id: str) -> Optional[Path]:
        """Return the run directory path, or None if it does not exist."""
        if not _is_safe_run_id(run_id):
            return None
        d = self.base_dir / run_id
        return d if d.is_dir() else None

    def artifact_path(self, run_id: str, filename: str) -> Optional[Path]:
        """Return the path of an artifact file, or None if unsafe or missing."""
        if not _is_safe_run_id(run_id):
            return None
        if "/" in filename or "\\" in filename or filename.startswith("."):
            return None
        p = self.base_dir / run_id / filename
        return p if p.is_file() else None

    # ── maintenance ───────────────────────────────────────────────────────────

    def _evict_oldest(self) -> None:
        try:
            dirs = sorted(
                [d for d in self.base_dir.iterdir() if d.is_dir() and _is_safe_run_id(d.name)],
                key=lambda p: p.stat().st_mtime,
            )
        except OSError:
            return
        while len(dirs) > _MAX_RUNS:
            shutil.rmtree(dirs.pop(0), ignore_errors=True)
