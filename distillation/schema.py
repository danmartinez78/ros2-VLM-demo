"""
Canonical temporal sample / dataset schema for the VLM distillation pipeline.

Every sample that flows through the pipeline (teacher generation, validation,
export, evaluation) is represented as a ``TemporalSample``. The representation
is framework-neutral; training-framework-specific export lives in
``distillation.dataset``.

Schema version
--------------
``SCHEMA_VERSION`` is bumped whenever the required field set or semantics
change. Stored samples carry the version so downstream code can detect
mismatches.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

SCHEMA_VERSION: str = "temporal_sample_v2"


@dataclass
class FrameRef:
    """One frame in an ordered temporal sequence."""

    path: str
    """Relative or absolute path to the image file."""

    t_seconds: float
    """Capture time in seconds relative to the sequence start."""

    def to_dict(self) -> dict:
        return {"path": self.path, "t_seconds": self.t_seconds}

    @classmethod
    def from_dict(cls, d: dict) -> "FrameRef":
        return cls(path=d["path"], t_seconds=float(d["t_seconds"]))


@dataclass
class TemporalTarget:
    """Compact structured temporal observation used as a training target."""

    change_detected: bool
    change: str
    """Direction/kind of change, e.g. 'approaching', 'receding', 'none'."""

    state_start: str
    """Scene state at the beginning of the sequence."""

    state_end: str
    """Scene state at the end of the sequence."""

    evidence_start_s: float
    """Time (seconds) at which the change first becomes evident."""

    evidence_end_s: float
    """Time (seconds) through which the change is observed."""

    confidence: float
    """Model confidence in [0.0, 1.0]."""

    scene_observation: str
    """One-line scene observation for downstream consumers."""

    def to_dict(self) -> dict:
        return {
            "change_detected": self.change_detected,
            "change": self.change,
            "state_start": self.state_start,
            "state_end": self.state_end,
            "evidence_start_s": self.evidence_start_s,
            "evidence_end_s": self.evidence_end_s,
            "confidence": self.confidence,
            "scene_observation": self.scene_observation,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TemporalTarget":
        cd = d["change_detected"]
        if not isinstance(cd, bool):
            raise TypeError(
                f"change_detected must be a JSON boolean, got {type(cd).__name__}: {cd!r}"
            )
        conf = d["confidence"]
        if not isinstance(conf, (int, float)):
            raise TypeError(
                f"confidence must be a JSON number, got {type(conf).__name__}: {conf!r}"
            )
        for float_field in ("evidence_start_s", "evidence_end_s"):
            v = d[float_field]
            if not isinstance(v, (int, float)):
                raise TypeError(
                    f"{float_field} must be a JSON number, got {type(v).__name__}: {v!r}"
                )
        for str_field in ("change", "state_start", "state_end", "scene_observation"):
            if not isinstance(d[str_field], str):
                raise TypeError(
                    f"{str_field} must be a JSON string, got {type(d[str_field]).__name__}: {d[str_field]!r}"
                )
        return cls(
            change_detected=cd,
            change=d["change"],
            state_start=d["state_start"],
            state_end=d["state_end"],
            evidence_start_s=float(d["evidence_start_s"]),
            evidence_end_s=float(d["evidence_end_s"]),
            confidence=float(d["confidence"]),
            scene_observation=d["scene_observation"],
        )


@dataclass
class Provenance:
    """Full provenance record for a generated sample."""

    repo_commit: str = ""
    teacher_model: str = ""
    teacher_engine_identity: str = ""
    teacher_prompt_version: str = ""
    validation_version: str = ""
    generated_at_utc: str = ""

    input_representation: str = ""
    # Examples: ordered_images, rendered_timestamps, temporal_images_meta,
    # native_video, or empty for legacy/unspecified data.

    sequence_type: str = ""
    # Examples: image_sequence, temporal_images, video, or empty.

    timestamp_policy: str = ""
    # Examples: capture_time_s, frame_index, rendered, or empty.

    effective_fps: float = 0.0
    rendered_timestamp_control: bool = False
    runtime_temporal_encoding: str = ""

    def to_dict(self) -> dict:
        return {
            "repo_commit": self.repo_commit,
            "teacher_model": self.teacher_model,
            "teacher_engine_identity": self.teacher_engine_identity,
            "teacher_prompt_version": self.teacher_prompt_version,
            "validation_version": self.validation_version,
            "generated_at_utc": self.generated_at_utc,
            "input_representation": self.input_representation,
            "sequence_type": self.sequence_type,
            "timestamp_policy": self.timestamp_policy,
            "effective_fps": self.effective_fps,
            "rendered_timestamp_control": self.rendered_timestamp_control,
            "runtime_temporal_encoding": self.runtime_temporal_encoding,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Provenance":
        import dataclasses

        kwargs: dict = {}
        for f in dataclasses.fields(cls):
            if f.name not in d:
                kwargs[f.name] = f.default
            elif f.type == "float" or f.default == 0.0:
                kwargs[f.name] = float(d[f.name])
            elif f.type == "bool" or isinstance(f.default, bool):
                kwargs[f.name] = bool(d[f.name])
            else:
                kwargs[f.name] = d[f.name]
        return cls(**kwargs)


@dataclass
class TemporalSample:
    """One fully described temporal training/evaluation sample."""

    sample_id: str
    schema_version: str = SCHEMA_VERSION
    frames: List[FrameRef] = field(default_factory=list)
    prompt_profile: str = "temporal_observation_v1"
    target: Optional[TemporalTarget] = None
    raw_teacher_response: str = ""
    provenance: Provenance = field(default_factory=Provenance)
    validation_status: str = "pending"
    rejection_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "schema_version": self.schema_version,
            "frames": [f.to_dict() for f in self.frames],
            "prompt_profile": self.prompt_profile,
            "target": self.target.to_dict() if self.target is not None else None,
            "raw_teacher_response": self.raw_teacher_response,
            "provenance": self.provenance.to_dict(),
            "validation_status": self.validation_status,
            "rejection_reasons": self.rejection_reasons,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TemporalSample":
        target_d = d.get("target")
        return cls(
            sample_id=d["sample_id"],
            schema_version=d.get("schema_version", SCHEMA_VERSION),
            frames=[FrameRef.from_dict(f) for f in d.get("frames", [])],
            prompt_profile=d.get("prompt_profile", "temporal_observation_v1"),
            target=TemporalTarget.from_dict(target_d) if target_d else None,
            raw_teacher_response=d.get("raw_teacher_response", ""),
            provenance=Provenance.from_dict(d.get("provenance", {})),
            validation_status=d.get("validation_status", "pending"),
            rejection_reasons=d.get("rejection_reasons", []),
        )

    def content_hash(self) -> str:
        raw = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(raw).hexdigest()[:16]

    def frame_count(self) -> int:
        return len(self.frames)

    def max_frame_index(self) -> int:
        """1-based maximum frame index (== ``frame_count()``)."""
        return len(self.frames)


def load_manifest(manifest_path: Path) -> List[TemporalSample]:
    """Load a JSON-lines or JSON-array manifest into TemporalSamples."""
    text = manifest_path.read_text()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [TemporalSample.from_dict(d) for d in data]
    except json.JSONDecodeError:
        pass

    samples = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            samples.append(TemporalSample.from_dict(json.loads(line)))
    return samples


def save_manifest(samples: List[TemporalSample], manifest_path: Path) -> None:
    """Write samples to a JSON-array manifest file."""
    manifest_path.write_text(json.dumps([s.to_dict() for s in samples], indent=2))
