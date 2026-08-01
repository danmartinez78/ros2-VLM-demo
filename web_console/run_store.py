# Copyright 2025 edge_vlm_ros contributors
"""
On-disk run record storage for the web console.

Each run is stored as a directory under base_dir/<run_id>/manifest.json.
Path traversal is prevented by validating run IDs against a UUID pattern.
Oldest runs are evicted when the count exceeds _MAX_RUNS.

Thread safety: all manifest reads and writes are protected by a per-instance
RLock and use an atomic temp-file + os.replace pattern so concurrent callers
never observe partial JSON.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

_MAX_RUNS: int = 100
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
_TERMINAL_STATUSES = frozenset({"completed", "failed", "stopped"})


def _is_safe_run_id(run_id: str) -> bool:
    """Return True only if run_id is a standard UUID string (path-traversal guard)."""
    return bool(_UUID_RE.match(run_id))


class RunStore:
    """Manages run directories and manifests under a base directory.

    All manifest reads and writes are serialised by a per-instance RLock.
    Manifests are written via a same-directory temporary file followed by
    os.replace so readers never observe a partial write even under concurrent
    access.
    """

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    # ── factory ───────────────────────────────────────────────────────────────

    @staticmethod
    def new_run_id() -> str:
        return str(uuid.uuid4())

    # ── write ─────────────────────────────────────────────────────────────────

    def save_run(self, run_id: str, record: Dict[str, Any]) -> Path:
        """Write manifest.json for a run atomically and evict oldest if necessary."""
        if not _is_safe_run_id(run_id):
            raise ValueError(f"Invalid run_id: {run_id!r}")
        run_dir = self.base_dir / run_id
        data = json.dumps(record, indent=2) + "\n"
        with self._lock:
            run_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = run_dir / "manifest.json"
            tmp_path = run_dir / "_manifest.tmp"
            tmp_path.write_text(data, encoding="utf-8")
            os.replace(str(tmp_path), str(manifest_path))
            self._evict_oldest()
        return manifest_path

    def update_run_if_status(
        self, run_id: str, expected_status: str, updates: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Atomically update a run record only if its current status equals expected_status.

        Returns the (possibly updated) record, or None if the run is not found.
        If the current status does not match, the unchanged record is returned.
        """
        if not _is_safe_run_id(run_id):
            return None
        with self._lock:
            record = self._read_nolock(run_id)
            if record is None:
                return None
            if record.get("status") != expected_status:
                return record
            record.update(updates)
            self._write_nolock(run_id, record)
            return record

    def finalize_run(self, run_id: str, updates: Dict[str, Any]) -> bool:
        """Atomically transition a run from any non-terminal state to a terminal state.

        A no-op (returns False) if the run is already terminal or not found.
        Returns True when the update is applied.
        """
        if not _is_safe_run_id(run_id):
            return False
        with self._lock:
            record = self._read_nolock(run_id)
            if record is None:
                return False
            if record.get("status") in _TERMINAL_STATUSES:
                return False
            record.update(updates)
            self._write_nolock(run_id, record)
            return True

    def write_artifact(self, run_id: str, filename: str, data: bytes) -> Path:
        """Write a binary artifact file to a run directory."""
        if not _is_safe_run_id(run_id):
            raise ValueError(f"Invalid run_id: {run_id!r}")
        if "/" in filename or "\\" in filename or filename.startswith("."):
            raise ValueError(f"Invalid artifact filename: {filename!r}")
        run_dir = self.base_dir / run_id
        with self._lock:
            run_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = run_dir / filename
            artifact_path.write_bytes(data)
        return artifact_path

    # ── read ──────────────────────────────────────────────────────────────────

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Return the manifest dict for a run, or None if not found."""
        if not _is_safe_run_id(run_id):
            return None
        with self._lock:
            return self._read_nolock(run_id)

    def list_runs(self) -> List[Dict[str, Any]]:
        """Return manifests for all runs, newest first."""
        with self._lock:
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
                manifest = self._read_nolock(d.name)
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

    # ── internal helpers (must be called with _lock held) ─────────────────────

    def _read_nolock(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Read the manifest for run_id without acquiring _lock."""
        manifest_path = self.base_dir / run_id / "manifest.json"
        if not manifest_path.is_file():
            return None
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def _write_nolock(self, run_id: str, record: Dict[str, Any]) -> None:
        """Write the manifest for run_id atomically without acquiring _lock."""
        run_dir = self.base_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = run_dir / "manifest.json"
        tmp_path = run_dir / "_manifest.tmp"
        data = json.dumps(record, indent=2) + "\n"
        tmp_path.write_text(data, encoding="utf-8")
        os.replace(str(tmp_path), str(manifest_path))

    # ── maintenance ───────────────────────────────────────────────────────────

    def _evict_oldest(self) -> None:
        """Evict oldest runs beyond _MAX_RUNS. RLock must be held by caller."""
        try:
            dirs = sorted(
                [d for d in self.base_dir.iterdir() if d.is_dir() and _is_safe_run_id(d.name)],
                key=lambda p: p.stat().st_mtime,
            )
        except OSError:
            return
        while len(dirs) > _MAX_RUNS:
            shutil.rmtree(dirs.pop(0), ignore_errors=True)
