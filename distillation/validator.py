"""
Validation and filtering stage for teacher-generated labels.

All validation rules are collected in ``VALIDATORS``. A sample passes only
when every validator accepts it. Rejected samples are kept with
machine-readable ``rejection_reasons``.

Any sample whose ``raw_teacher_response`` or ``scene_observation`` references a
frame/image index that exceeds the actual frame count is rejected. For example,
a reference to ``Image 9`` or ``Image 10`` in an eight-frame sample must be
caught here.
"""

from __future__ import annotations

import re
from typing import Callable, List, Optional, Tuple

from distillation.schema import TemporalSample

VALIDATION_VERSION: str = "validator_v2"
ValidatorFn = Callable[[TemporalSample], Tuple[bool, str]]


def _validate_has_target(sample: TemporalSample) -> Tuple[bool, str]:
    if sample.target is None:
        return False, "missing_target: teacher did not produce a parseable target"
    return True, ""


def _validate_required_fields(sample: TemporalSample) -> Tuple[bool, str]:
    t = sample.target
    if t is None:
        return True, ""
    missing = []
    for attr in (
        "change_detected",
        "change",
        "state_start",
        "state_end",
        "evidence_start_s",
        "evidence_end_s",
        "confidence",
        "scene_observation",
    ):
        val = getattr(t, attr, None)
        if val is None or (isinstance(val, str) and not val.strip()):
            missing.append(attr)
    if missing:
        return False, f"missing_fields: {', '.join(missing)}"
    return True, ""


def _validate_confidence_range(sample: TemporalSample) -> Tuple[bool, str]:
    t = sample.target
    if t is None:
        return True, ""
    if not (0.0 <= t.confidence <= 1.0):
        return False, f"confidence_out_of_range: {t.confidence}"
    return True, ""


def _validate_has_frames(sample: TemporalSample) -> Tuple[bool, str]:
    if not sample.frames:
        return False, "no_frames: sample has no frame references"
    return True, ""


def _validate_timestamp_ordering(sample: TemporalSample) -> Tuple[bool, str]:
    for i in range(1, len(sample.frames)):
        if sample.frames[i].t_seconds < sample.frames[i - 1].t_seconds:
            return (
                False,
                f"frame_timestamp_not_monotonic: frame {i} t={sample.frames[i].t_seconds} "
                f"< frame {i-1} t={sample.frames[i-1].t_seconds}",
            )
    return True, ""


def _validate_evidence_ordering(sample: TemporalSample) -> Tuple[bool, str]:
    t = sample.target
    if t is None:
        return True, ""
    if t.evidence_end_s < t.evidence_start_s:
        return False, f"evidence_time_inverted: start={t.evidence_start_s} end={t.evidence_end_s}"
    return True, ""


def _validate_evidence_within_sequence(sample: TemporalSample) -> Tuple[bool, str]:
    t = sample.target
    if t is None or not sample.frames:
        return True, ""
    t_min = sample.frames[0].t_seconds
    t_max = sample.frames[-1].t_seconds
    if t.evidence_start_s < t_min - 1e-6:
        return False, f"evidence_start_before_sequence: {t.evidence_start_s} < {t_min}"
    if t.evidence_end_s > t_max + 1e-6:
        return False, f"evidence_end_after_sequence: {t.evidence_end_s} > {t_max}"
    return True, ""


def _validate_no_hallucinated_frame_refs(sample: TemporalSample) -> Tuple[bool, str]:
    """Reject output references to frame indices beyond the supplied sequence."""
    n_frames = sample.frame_count()
    text_to_check = sample.raw_teacher_response
    if sample.target:
        text_to_check += " " + sample.target.scene_observation
        text_to_check += " " + sample.target.state_start
        text_to_check += " " + sample.target.state_end
        text_to_check += " " + sample.target.change

    pattern = re.compile(r"\b(?:image|frame)\s+(\d+)\b", re.IGNORECASE)
    hallucinated = []
    for match in pattern.finditer(text_to_check):
        idx = int(match.group(1))
        if idx > n_frames:
            hallucinated.append(match.group(0))
    if hallucinated:
        refs = ", ".join(sorted(set(hallucinated)))
        return False, f"hallucinated_frame_reference: references {refs} but sample has {n_frames} frames"
    return True, ""


def _validate_static_consistency(sample: TemporalSample) -> Tuple[bool, str]:
    if sample.target is None:
        return True, ""
    if sample.prompt_profile.endswith("_static") and sample.target.change_detected:
        return False, "static_sequence_false_positive: teacher claimed change in a known-static sequence"
    return True, ""


VALIDATORS: List[ValidatorFn] = [
    _validate_has_frames,
    _validate_has_target,
    _validate_required_fields,
    _validate_confidence_range,
    _validate_timestamp_ordering,
    _validate_evidence_ordering,
    _validate_evidence_within_sequence,
    _validate_no_hallucinated_frame_refs,
    _validate_static_consistency,
]


def validate_sample(
    sample: TemporalSample,
    validators: Optional[List[ValidatorFn]] = None,
) -> TemporalSample:
    """Run validators and return a copy with status/rejection reasons updated."""
    if validators is None:
        validators = VALIDATORS

    reasons: List[str] = []
    for fn in validators:
        passed, reason = fn(sample)
        if not passed:
            reasons.append(reason)

    import copy

    updated = copy.copy(sample)
    updated.rejection_reasons = reasons
    updated.validation_status = "accepted" if not reasons else "rejected"
    updated.provenance = copy.copy(sample.provenance)
    updated.provenance.validation_version = VALIDATION_VERSION
    return updated


def validate_batch(
    samples: List[TemporalSample],
    validators: Optional[List[ValidatorFn]] = None,
) -> Tuple[List[TemporalSample], List[TemporalSample]]:
    """Validate a list of samples and return ``(accepted, rejected)``."""
    accepted: List[TemporalSample] = []
    rejected: List[TemporalSample] = []
    for sample in samples:
        result = validate_sample(sample, validators)
        if result.validation_status == "accepted":
            accepted.append(result)
        else:
            rejected.append(result)
    return accepted, rejected
