# Copyright 2025 edge_vlm_ros contributors
"""
Lightweight human-review annotation store for the warehouse workbench.

Reviews are stored separately from immutable inference artifacts so that
changing or adding annotations never modifies the original run manifest.
Each review applies to one result (identified by run_id + frame_index) and
carries a label and an optional free-text note.

Storage layout::

    <runs_dir>/<run_id>/reviews.json

``reviews.json`` is a JSON array of annotation objects.  Writes are atomic
(temp-file + os.replace) under a per-run lock so concurrent browser clients
do not corrupt each other.

Allowed labels
--------------
``acceptable``
    The model response is correct and operationally useful.
``unsupported_hallucinated``
    The response contains claims not supported by the image.
``missed_important_detail``
    A visible, operationally significant detail was not reported.
``ambiguous``
    The image quality or scene make a definitive assessment impossible.
"""
from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── constants ─────────────────────────────────────────────────────────────────

_SCHEMA_VERSION = 1
_REVIEWS_FILENAME = "reviews.json"
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)

ALLOWED_REVIEW_LABELS = frozenset(
    {
        "acceptable",
        "unsupported_hallucinated",
        "missed_important_detail",
        "ambiguous",
    }
)

_MAX_NOTE_CHARS = 1000
_MAX_REVIEWS_PER_RUN = 10_000


# ── dataclass ─────────────────────────────────────────────────────────────────


@dataclass
class ReviewAnnotation:
    """A single human-review annotation for one frame result."""

    run_id: str
    frame_index: int
    label: str
    note: str = ""
    created_at: str = ""
    updated_at: str = ""
    schema_version: int = _SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "frame_index": self.frame_index,
            "label": self.label,
            "note": self.note,
            "created_at": self.created_at,
            "updated_at": self.updated_at or self.created_at,
        }


# ── validation ────────────────────────────────────────────────────────────────


def _is_safe_run_id(run_id: str) -> bool:
    return bool(_UUID_RE.match(run_id))


def validate_review(body: Dict[str, Any]) -> Optional[str]:
    """Validate a review request body dict.  Returns an error string or None."""
    frame_index = body.get("frame_index")
    if frame_index is None:
        return "frame_index is required"
    try:
        fi = int(frame_index)
    except (TypeError, ValueError):
        return "frame_index must be an integer"
    if fi < 0:
        return "frame_index must be >= 0"
    label = body.get("label", "")
    if not label:
        return "label is required"
    if label not in ALLOWED_REVIEW_LABELS:
        return (
            f"Unknown label {label!r}. Allowed: {sorted(ALLOWED_REVIEW_LABELS)}"
        )
    note = body.get("note", "")
    if note and len(str(note)) > _MAX_NOTE_CHARS:
        return f"note exceeds maximum length ({_MAX_NOTE_CHARS} chars)"
    return None


# ── store ─────────────────────────────────────────────────────────────────────


class ReviewStore:
    """Manages per-run review annotation files under the run store base dir.

    Thread-safe: a per-run lock serialises reads and writes so concurrent
    browser submissions do not interleave.
    """

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)
        # Per-run locks to prevent concurrent write corruption.
        self._locks: Dict[str, threading.Lock] = {}
        self._meta_lock = threading.Lock()

    def _run_dir(self, run_id: str) -> Optional[Path]:
        """Return the run directory. Creates it if it does not exist."""
        if not _is_safe_run_id(run_id):
            return None
        d = self.base_dir / run_id
        try:
            d.mkdir(parents=True, exist_ok=True)
        except OSError:
            return None
        return d

    def _get_lock(self, run_id: str) -> threading.Lock:
        with self._meta_lock:
            if run_id not in self._locks:
                self._locks[run_id] = threading.Lock()
            return self._locks[run_id]

    def _read_reviews(self, run_id: str) -> List[Dict[str, Any]]:
        """Read reviews from disk (caller must hold the run lock)."""
        d = self._run_dir(run_id)
        if d is None:
            return []
        path = d / _REVIEWS_FILENAME
        if not path.is_file():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [r for r in data if isinstance(r, dict)]
        except (json.JSONDecodeError, OSError):
            pass
        return []

    def _write_reviews(
        self, run_id: str, reviews: List[Dict[str, Any]]
    ) -> None:
        """Write reviews to disk atomically (caller must hold the run lock)."""
        d = self._run_dir(run_id)
        if d is None:
            return
        path = d / _REVIEWS_FILENAME
        tmp = d / f"{_REVIEWS_FILENAME}.tmp"
        tmp.write_text(json.dumps(reviews, indent=2), encoding="utf-8")
        os.replace(str(tmp), str(path))

    def upsert_review(self, annotation: ReviewAnnotation) -> bool:
        """Insert or replace the review for (run_id, frame_index).

        Returns False when the run directory does not exist or an error occurs.
        """
        run_id = annotation.run_id
        if not _is_safe_run_id(run_id):
            return False
        lock = self._get_lock(run_id)
        with lock:
            reviews = self._read_reviews(run_id)
            if len(reviews) >= _MAX_REVIEWS_PER_RUN and not any(
                r.get("frame_index") == annotation.frame_index for r in reviews
            ):
                return False  # cap reached, won't insert new entry
            # Replace existing or append.
            updated = False
            for i, r in enumerate(reviews):
                if r.get("frame_index") == annotation.frame_index:
                    # Preserve the original creation timestamp on update.
                    d = annotation.to_dict()
                    original_created_at = r.get("created_at")
                    if original_created_at:
                        d["created_at"] = original_created_at
                    reviews[i] = d
                    updated = True
                    break
            if not updated:
                reviews.append(annotation.to_dict())
            try:
                self._write_reviews(run_id, reviews)
                return True
            except OSError:
                return False

    def get_reviews(self, run_id: str) -> List[ReviewAnnotation]:
        """Return all review annotations for *run_id*, sorted by frame_index."""
        if not _is_safe_run_id(run_id):
            return []
        lock = self._get_lock(run_id)
        with lock:
            raw = self._read_reviews(run_id)
        result: List[ReviewAnnotation] = []
        for r in sorted(raw, key=lambda x: x.get("frame_index", 0)):
            try:
                ann = ReviewAnnotation(
                    run_id=str(r.get("run_id", run_id)),
                    frame_index=int(r.get("frame_index", 0)),
                    label=str(r.get("label", "")),
                    note=str(r.get("note", "")),
                    created_at=str(r.get("created_at", "")),
                    updated_at=str(r.get("updated_at", r.get("created_at", ""))),
                    schema_version=int(r.get("schema_version", _SCHEMA_VERSION)),
                )
                result.append(ann)
            except (TypeError, ValueError):
                continue
        return result

    def get_review_for_frame(
        self, run_id: str, frame_index: int
    ) -> Optional[ReviewAnnotation]:
        """Return the review for a specific frame, or None."""
        for r in self.get_reviews(run_id):
            if r.frame_index == frame_index:
                return r
        return None
