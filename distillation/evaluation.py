"""
Temporal-quality evaluation hooks for the distillation pipeline.

Computes metrics against labelled ground-truth samples or against the teacher
labels after student inference.

Supported metrics
-----------------
- change_detection_accuracy
- direction_accuracy       (change direction when change_detected=True)
- event_time_bucket_accuracy
- static_false_positive_rate
- schema_adherence
- hallucinated_frame_reference_rate
- mean_confidence          (average reported confidence across predictions)

Note: a ``confidence_mae`` metric against the binary change-detected label was
removed because it is semantically incorrect — a high-confidence correct
no-change response would appear maximally miscalibrated.  Calibration can be
re-added once the schema carries an explicit ``change_probability`` float field.

Controlled sequence transforms supported in ``ControlledSequenceEvaluator``:
  chronological, reversed, shuffled, duplicated_frame, single_terminal_frame.

For ``reversed`` and ``shuffled`` transforms the evaluator derives
control-specific expected targets rather than scoring against the original
label unchanged.  See ``_derive_control_target``.

Linkage to hardware latency benchmarks is provided through the
``attach_latency_record`` helper which grafts F1/F2/F4/F8 timing into the
evaluation report.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from distillation.schema import TemporalSample, TemporalTarget


# ---------------------------------------------------------------------------
# Event-time bucket helpers
# ---------------------------------------------------------------------------

# Buckets: early (< 33%), mid (33–66%), late (>66%).
_BUCKET_EARLY = "early"
_BUCKET_MID = "mid"
_BUCKET_LATE = "late"


def _evidence_bucket(evidence_start_s: float, t_min: float, t_max: float) -> str:
    duration = t_max - t_min
    if duration <= 0:
        return _BUCKET_EARLY
    frac = (evidence_start_s - t_min) / duration
    if frac < 1 / 3:
        return _BUCKET_EARLY
    if frac < 2 / 3:
        return _BUCKET_MID
    return _BUCKET_LATE


# ---------------------------------------------------------------------------
# Hallucination detection (reuses same regex as validator)
# ---------------------------------------------------------------------------

_HALLUC_PATTERN = re.compile(r"\b(?:image|frame)\s+(\d+)\b", re.IGNORECASE)


def _has_hallucinated_frame_ref(text: str, n_frames: int) -> bool:
    for m in _HALLUC_PATTERN.finditer(text):
        if int(m.group(1)) > n_frames:
            return True
    return False


# ---------------------------------------------------------------------------
# Single-sample comparison
# ---------------------------------------------------------------------------

@dataclass
class SampleMetrics:
    sample_id: str
    change_detected_correct: Optional[bool] = None
    direction_correct: Optional[bool] = None
    event_bucket_correct: Optional[bool] = None
    is_false_positive: Optional[bool] = None
    schema_valid: bool = True
    hallucinated_ref: bool = False
    confidence: Optional[float] = None
    gt_change_detected: Optional[bool] = None


def compare_sample(
    predicted: TemporalSample,
    ground_truth: TemporalSample,
) -> SampleMetrics:
    """
    Compare a student-predicted TemporalSample against a ground-truth sample.
    """
    m = SampleMetrics(sample_id=predicted.sample_id)

    pred_t = predicted.target
    gt_t = ground_truth.target

    if pred_t is None:
        m.schema_valid = False
        return m

    m.confidence = pred_t.confidence

    if gt_t is not None:
        m.gt_change_detected = gt_t.change_detected
        m.change_detected_correct = pred_t.change_detected == gt_t.change_detected

        if gt_t.change_detected:
            m.direction_correct = pred_t.change == gt_t.change
            gt_t_min = ground_truth.frames[0].t_seconds if ground_truth.frames else 0.0
            gt_t_max = ground_truth.frames[-1].t_seconds if ground_truth.frames else 1.0
            gt_bucket = _evidence_bucket(gt_t.evidence_start_s, gt_t_min, gt_t_max)
            if predicted.frames:
                pred_t_min = predicted.frames[0].t_seconds
                pred_t_max = predicted.frames[-1].t_seconds
                pred_bucket = _evidence_bucket(pred_t.evidence_start_s, pred_t_min, pred_t_max)
            else:
                pred_bucket = _evidence_bucket(pred_t.evidence_start_s, gt_t_min, gt_t_max)
            m.event_bucket_correct = pred_bucket == gt_bucket

        # Static false positive
        if not gt_t.change_detected:
            m.is_false_positive = pred_t.change_detected

    # Hallucination check
    text = pred_t.odd_observation + " " + pred_t.state_start + " " + pred_t.state_end
    m.hallucinated_ref = _has_hallucinated_frame_ref(text, predicted.frame_count())

    return m


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

@dataclass
class EvaluationReport:
    """Aggregated evaluation report over a batch of predictions."""

    n_samples: int = 0

    change_detection_accuracy: Optional[float] = None
    direction_accuracy: Optional[float] = None
    event_time_bucket_accuracy: Optional[float] = None
    static_false_positive_rate: Optional[float] = None
    schema_adherence: Optional[float] = None
    hallucinated_frame_reference_rate: Optional[float] = None
    mean_confidence: Optional[float] = None

    # Optional latency entries from hardware benchmark linkage.
    latency_records: Dict[str, dict] = field(default_factory=dict)

    per_sample: List[SampleMetrics] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_samples": self.n_samples,
            "change_detection_accuracy": self.change_detection_accuracy,
            "direction_accuracy": self.direction_accuracy,
            "event_time_bucket_accuracy": self.event_time_bucket_accuracy,
            "static_false_positive_rate": self.static_false_positive_rate,
            "schema_adherence": self.schema_adherence,
            "hallucinated_frame_reference_rate": self.hallucinated_frame_reference_rate,
            "mean_confidence": self.mean_confidence,
            "latency_records": self.latency_records,
        }


def evaluate_batch(
    predictions: List[TemporalSample],
    ground_truths: List[TemporalSample],
) -> EvaluationReport:
    """
    Evaluate a list of student predictions against ground-truth samples.

    ``ground_truths`` is matched by ``sample_id``.  Predictions without a
    matching ground truth are evaluated for schema/hallucination only.
    """
    gt_map = {s.sample_id: s for s in ground_truths}
    report = EvaluationReport(n_samples=len(predictions))

    per_sample: List[SampleMetrics] = []
    for pred in predictions:
        gt = gt_map.get(pred.sample_id)
        if gt is not None:
            m = compare_sample(pred, gt)
        else:
            # Schema + hallucination only.
            m = SampleMetrics(sample_id=pred.sample_id)
            if pred.target is None:
                m.schema_valid = False
            else:
                text = (
                    pred.target.odd_observation
                    + " "
                    + pred.target.state_start
                    + " "
                    + pred.target.state_end
                )
                m.hallucinated_ref = _has_hallucinated_frame_ref(
                    text, pred.frame_count()
                )
                m.confidence = pred.target.confidence
        per_sample.append(m)

    report.per_sample = per_sample

    def _rate(values: List[bool]) -> Optional[float]:
        if not values:
            return None
        return sum(values) / len(values)

    cd_vals = [m.change_detected_correct for m in per_sample if m.change_detected_correct is not None]
    dir_vals = [m.direction_correct for m in per_sample if m.direction_correct is not None]
    bkt_vals = [m.event_bucket_correct for m in per_sample if m.event_bucket_correct is not None]
    fp_vals = [m.is_false_positive for m in per_sample if m.is_false_positive is not None]
    schema_vals = [m.schema_valid for m in per_sample]
    halluc_vals = [m.hallucinated_ref for m in per_sample]
    conf_vals = [m.confidence for m in per_sample if m.confidence is not None]

    report.change_detection_accuracy = _rate(cd_vals)  # type: ignore[arg-type]
    report.direction_accuracy = _rate(dir_vals)  # type: ignore[arg-type]
    report.event_time_bucket_accuracy = _rate(bkt_vals)  # type: ignore[arg-type]
    report.static_false_positive_rate = _rate(fp_vals)  # type: ignore[arg-type]
    report.schema_adherence = _rate(schema_vals)  # type: ignore[arg-type]
    report.hallucinated_frame_reference_rate = _rate(halluc_vals)  # type: ignore[arg-type]
    if conf_vals:
        report.mean_confidence = statistics.mean(conf_vals)

    return report


# ---------------------------------------------------------------------------
# Controlled-sequence evaluation helpers
# ---------------------------------------------------------------------------

class ControlledSequenceEvaluator:
    """
    Runs evaluation over multiple controlled sequence transforms.

    Supported transforms: chronological, reversed, shuffled,
    duplicated_frame, single_terminal_frame.

    For ``reversed`` and ``shuffled`` transforms, the expected ground-truth
    target is derived from the original via ``_derive_control_target`` rather
    than being used unchanged.  This avoids the semantic error of scoring a
    reversed-approaching sequence against an "approaching" target when the
    correct expectation is "receding".
    """

    TRANSFORMS = [
        "chronological",
        "reversed",
        "shuffled",
        "duplicated_frame",
        "single_terminal_frame",
    ]

    # Directions whose semantic meaning inverts under time-reversal.
    _REVERSE_DIRECTION_MAP: dict = {
        "approaching": "receding",
        "receding": "approaching",
        "accelerating": "decelerating",
        "decelerating": "accelerating",
    }

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def apply_transform(
        self, sample: TemporalSample, transform: str
    ) -> TemporalSample:
        """Return a new sample with the frames transformed."""
        import copy
        import random

        s = copy.deepcopy(sample)
        if transform == "chronological":
            s.frames = sorted(s.frames, key=lambda f: f.t_seconds)
        elif transform == "reversed":
            s.frames = list(reversed(s.frames))
        elif transform == "shuffled":
            rng = random.Random(self.seed)
            rng.shuffle(s.frames)
        elif transform == "duplicated_frame":
            if s.frames:
                s.frames = s.frames + [s.frames[-1]]
        elif transform == "single_terminal_frame":
            if s.frames:
                s.frames = [s.frames[-1]]
        else:
            raise ValueError(f"Unknown transform: {transform}")
        return s

    def _derive_control_target(
        self, original: TemporalSample, transform: str
    ) -> TemporalSample:
        """
        Derive the expected ground-truth sample for control-scoring purposes.

        For ``reversed``: invert the temporal direction of the change label
        (e.g. approaching → receding) and swap state_start/state_end.
        For ``shuffled``: change_detected semantics are ambiguous for arbitrary
        orderings, so the sample is marked as control-only (no expected change
        direction), and only schema/hallucination metrics are meaningful.
        All other transforms: return the original unchanged.
        """
        import copy

        if transform == "reversed" and original.target is not None:
            s = copy.deepcopy(original)
            t = s.target
            inverted_dir = self._REVERSE_DIRECTION_MAP.get(t.change, t.change)
            s.target = TemporalTarget(
                change_detected=t.change_detected,
                change=inverted_dir,
                state_start=t.state_end,
                state_end=t.state_start,
                evidence_start_s=t.evidence_start_s,
                evidence_end_s=t.evidence_end_s,
                confidence=t.confidence,
                odd_observation=t.odd_observation,
            )
            return s

        if transform == "shuffled":
            # For shuffled sequences the original temporal ordering is lost;
            # direction/time metrics are not meaningful.  Return a copy with
            # the target cleared so that only schema/hallucination are scored.
            s = copy.deepcopy(original)
            s.target = None
            return s

        return original

    def evaluate_transforms(
        self,
        predictions_by_transform: Dict[str, List[TemporalSample]],
        ground_truths: List[TemporalSample],
    ) -> Dict[str, "EvaluationReport"]:
        """
        Evaluate predictions for each transform using control-appropriate
        expected targets.

        Parameters
        ----------
        predictions_by_transform : dict mapping transform name -> predictions
        ground_truths            : reference ground-truth samples

        Returns
        -------
        dict mapping transform name -> EvaluationReport
        """
        gt_map = {s.sample_id: s for s in ground_truths}
        results: Dict[str, EvaluationReport] = {}
        for transform, preds in predictions_by_transform.items():
            control_gts = [
                self._derive_control_target(gt_map[p.sample_id], transform)
                if p.sample_id in gt_map
                else p
                for p in preds
            ]
            results[transform] = evaluate_batch(preds, control_gts)
        return results


# ---------------------------------------------------------------------------
# Latency benchmark linkage
# ---------------------------------------------------------------------------

def attach_latency_record(report: EvaluationReport, frame_count: int, latency_ms: float) -> None:
    """
    Attach a hardware latency observation to an EvaluationReport.

    Parameters
    ----------
    report      : EvaluationReport to update in-place
    frame_count : F-count (1, 2, 4, or 8)
    latency_ms  : measured IPC mean latency in milliseconds
    """
    report.latency_records[f"F{frame_count}"] = {"latency_ms": latency_ms}
