"""
Teacher-label generation stage.

The ``TeacherRuntime`` abstract base class decouples the pipeline from the
actual model invocation layer.  CPU tests use ``FakeTeacherRuntime``; real
hardware uses a concrete implementation backed by the existing VLM IPC layer.

Usage (fake / CPU):

    runtime = FakeTeacherRuntime(responses={...})
    gen = TeacherLabelGenerator(runtime=runtime, output_dir=Path("..."))
    gen.run(manifest)

Every generated sample is written to ``output_dir/<sample_id>.json`` immediately
after generation so that a run can be resumed without regenerating completed
samples.
"""

from __future__ import annotations

import datetime
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

from distillation.schema import (
    FrameRef,
    Provenance,
    TemporalSample,
    TemporalTarget,
    load_manifest,
    save_manifest,
)

TEACHER_PROMPT_VERSION: str = "temporal_teacher_v1"

TEACHER_PROMPT_TEMPLATE: str = (
    "You are a temporal observation model. You are given {n} ordered frames "
    "from a sensor sequence captured at the following times: {timestamps}. "
    "Analyze the sequence and return a JSON object with these fields: "
    "change_detected (bool), change (str), state_start (str), state_end (str), "
    "evidence_start_s (float), evidence_end_s (float), confidence (float 0-1), "
    "odd_observation (str). "
    "Return ONLY the JSON object, no additional text."
)


class TeacherRuntime(ABC):
    """Abstract interface for the model invocation layer."""

    @property
    @abstractmethod
    def model_identity(self) -> str:
        """Human-readable model identity string."""

    @property
    @abstractmethod
    def engine_identity(self) -> str:
        """Hardware/engine identity string."""

    @abstractmethod
    def generate(self, frames: List[FrameRef], prompt: str) -> str:
        """
        Run the model and return the raw string response.

        Parameters
        ----------
        frames : ordered list of FrameRefs (paths + timestamps)
        prompt : formatted prompt string

        Returns
        -------
        Raw model response string (should be a JSON object but may be malformed).
        """


class FakeTeacherRuntime(TeacherRuntime):
    """
    Deterministic fake runtime for CPU / CI tests.

    ``responses`` maps ``sample_id`` to a raw response string.  If a sample_id
    is not in the map the default_response is returned.
    """

    def __init__(
        self,
        responses: Optional[Dict[str, str]] = None,
        default_response: Optional[str] = None,
        model_id: str = "FakeTeacher-0B",
        engine_id: str = "fake-engine-cpu",
    ) -> None:
        self._responses = responses or {}
        self._default = default_response or json.dumps(
            {
                "change_detected": False,
                "change": "none",
                "state_start": "scene stable",
                "state_end": "scene stable",
                "evidence_start_s": 0.0,
                "evidence_end_s": 1.0,
                "confidence": 0.95,
                "odd_observation": "no significant change detected",
            }
        )
        self._model_id = model_id
        self._engine_id = engine_id
        # Track calls for auditability in tests
        self.call_log: List[str] = []

    @property
    def model_identity(self) -> str:
        return self._model_id

    @property
    def engine_identity(self) -> str:
        return self._engine_id

    def generate(self, frames: List[FrameRef], prompt: str) -> str:  # noqa: ARG002
        # The sample_id is not directly available here; callers inject it via
        # the generate_for_sample wrapper below.
        return self._default

    def generate_for_sample(self, sample_id: str, frames: List[FrameRef], prompt: str) -> str:
        self.call_log.append(sample_id)
        return self._responses.get(sample_id, self.generate(frames, prompt))


class TeacherLabelGenerator:
    """
    Runs teacher inference over a list of TemporalSamples and writes labelled
    outputs to ``output_dir``.

    Resume behaviour
    ----------------
    If ``<output_dir>/<sample_id>.json`` already exists the sample is skipped,
    allowing interrupted runs to be resumed without re-running the model.
    """

    def __init__(
        self,
        runtime: TeacherRuntime,
        output_dir: Path,
        repo_commit: str = "",
        prompt_version: str = TEACHER_PROMPT_VERSION,
    ) -> None:
        self.runtime = runtime
        self.output_dir = output_dir
        self.repo_commit = repo_commit
        self.prompt_version = prompt_version
        output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, samples: List[TemporalSample]) -> List[TemporalSample]:
        """
        Generate teacher labels for all samples.

        Returns the updated list (including skipped/resumed samples).
        """
        results: List[TemporalSample] = []
        for sample in samples:
            out_path = self.output_dir / f"{sample.sample_id}.json"
            if out_path.exists():
                # Resume: load the already-generated label.
                loaded = TemporalSample.from_dict(json.loads(out_path.read_text()))
                results.append(loaded)
                continue
            labelled = self._generate_one(sample)
            out_path.write_text(json.dumps(labelled.to_dict(), indent=2))
            results.append(labelled)
        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_prompt(self, sample: TemporalSample) -> str:
        timestamps = ", ".join(
            f"frame {i + 1} @ {f.t_seconds:.3f}s"
            for i, f in enumerate(sample.frames)
        )
        return TEACHER_PROMPT_TEMPLATE.format(
            n=len(sample.frames), timestamps=timestamps
        )

    def _generate_one(self, sample: TemporalSample) -> TemporalSample:
        prompt = self._build_prompt(sample)

        # Support FakeTeacherRuntime.generate_for_sample for deterministic tests.
        if isinstance(self.runtime, FakeTeacherRuntime):
            raw = self.runtime.generate_for_sample(sample.sample_id, sample.frames, prompt)
        else:
            raw = self.runtime.generate(sample.frames, prompt)

        provenance = Provenance(
            repo_commit=self.repo_commit,
            teacher_model=self.runtime.model_identity,
            teacher_engine_identity=self.runtime.engine_identity,
            teacher_prompt_version=self.prompt_version,
            validation_version="",
            generated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )

        # Attempt to parse target; leave as None if malformed.
        target: Optional[TemporalTarget] = None
        try:
            parsed = json.loads(raw)
            target = TemporalTarget.from_dict(parsed)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass

        return TemporalSample(
            sample_id=sample.sample_id,
            frames=sample.frames,
            prompt_profile=sample.prompt_profile,
            target=target,
            raw_teacher_response=raw,
            provenance=provenance,
            validation_status="pending",
            rejection_reasons=[],
        )
