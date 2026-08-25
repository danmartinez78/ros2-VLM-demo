"""
Temporal-quality evaluation hooks for the distillation pipeline.

Computes metrics against labelled ground-truth samples or against teacher
labels after student inference.

Supported metrics
-----------------
- change_detection_accuracy
- direction_accuracy
- event_time_bucket_accuracy
- static_false_positive_rate
- schema_adherence
- hallucinated_frame_reference_rate
- mean_confidence

Controlled sequence transforms supported in ``ControlledSequenceEvaluator``:
chronological, reversed, shuffled, duplicated_frame, single_terminal_frame.

For ``reversed`` and ``shuffled`` transforms the evaluator derives
control-specific expected targets rather than scoring against the original
label unchanged.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from distillation.schema import TemporalSample, TemporalTarget

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


_HALLUC_PATTERN = re.compile(r"\b(?:image|frame)\s+(\d+)\b", re.IGNORECASE)


def _has_hallucinated_frame_ref(text: str, n_frames: int) -> bool:
    return any(int(m.group(1)) > n_frames for m in _HALLUC_PATTERN.finditer(text))


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


def compare_sample(predicted: TemporalSample, ground_truth: TemporalSample) -> SampleMetrics:
    """Compare a student-predicted sample against a ground-truth sample."""
    metrics = SampleMetrics(sample_id=predicted.sample_id)
    pred_t = predicted.target
    gt_t = ground_truth.target

    if pred_t is None:
        metrics.schema_valid = False
        return metrics

    metrics.confidence = pred_t.confidence

    if gt_t is not None:
        metrics.gt_change_detected = gt_t.change_detected
        metrics.change_detected_correct = pred_t.change_detected == gt_t.change_detected

        if gt_t.change_detected:
            metrics.direction_correct = pred_t.change == gt_t.change
            gt_t_min = ground_truth.frames[0].t_seconds if ground_truth.frames else 0.0
            gt_t_max = ground_truth.frames[-1].t_seconds if ground_truth.frames else 1.0
            gt_bucket = _evidence_bucket(gt_t.evidence_start_s, gt_t_min, gt_t_max)
            if predicted.frames:
                pred_bucket = _evidence_bucket(
                    pred_t.evidence_start_s,
                    predicted.frames[0].t_seconds,
                    predicted.frames[-1].t_seconds,
                )
            else:
                pred_bucket = _evidence_bucket(pred_t.evidence_start_s, gt_t_min, gt_t_max)
            metrics.event_bucket_correct = pred_bucket == gt_bucket

        if not gt_t.change_detected:
            metrics.is_false_positive = pred_t.change_detected

    text = pred_t.scene_observation + " " + pred_t.state_start + " " + pred_t.state_end
    metrics.hallucinated_ref = _has_hallucinated_frame_ref(text, predicted.frame_count())
    return metrics


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


def evaluate_batch(predictions: List[TemporalSample], ground_truths: List[TemporalSample]) -> EvaluationReport:
    """Evaluate predictions against ground truths matched by ``sample_id``."""
    gt_map = {s.sample_id: s for s in ground_truths}
    report = EvaluationReport(n_samples=len(predictions))
    per_sample: List[SampleMetrics] = []

    for pred in predictions:
        gt = gt_map.get(pred.sample_id)
        if gt is not None:
            metric = compare_sample(pred, gt)
        else:
            metric = SampleMetrics(sample_id=pred.sample_id)
            if pred.target is None:
                metric.schema_valid = False
            else:
                text = (
                    pred.target.scene_observation
                    + " "
                    + pred.target.state_start
                    + " "
                    + pred.target.state_end
                )
                metric.hallucinated_ref = _has_hallucinated_frame_ref(text, pred.frame_count())
                metric.confidence = pred.target.confidence
        per_sample.append(metric)

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


class ControlledSequenceEvaluator:
    """Run evaluation over controlled temporal sequence transforms."""

    TRANSFORMS = [
        "chronological",
        "reversed",
        "shuffled",
        "duplicated_frame",
        "single_terminal_frame",
    ]

    _REVERSE_DIRECTION_MAP = {
        "approaching": "receding",
        "receding": "approaching",
        "accelerating": "decelerating",
        "decelerating": "accelerating",
    }

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def apply_transform(self, sample: TemporalSample, transform: str) -> TemporalSample:
        import copy
        import random

        result = copy.deepcopy(sample)
        if transform == "chronological":
            result.frames = sorted(result.frames, key=lambda frame: frame.t_seconds)
        elif transform == "reversed":
            result.frames = list(reversed(result.frames))
        elif transform == "shuffled":
            rng = random.Random(self.seed)
            rng.shuffle(result.frames)
        elif transform == "duplicated_frame":
            if result.frames:
                result.frames = result.frames + [result.frames[-1]]
        elif transform == "single_terminal_frame":
            if result.frames:
                result.frames = [result.frames[-1]]
        else:
            raise ValueError(f"Unknown transform: {transform}")
        return result

    def _derive_control_target(self, original: TemporalSample, transform: str) -> TemporalSample:
        import copy

        if transform == "reversed" and original.target is not None:
            result = copy.deepcopy(original)
            target = result.target
            inverted_dir = self._REVERSE_DIRECTION_MAP.get(target.change, target.change)

            if result.frames:
                times = sorted(frame.t_seconds for frame in result.frames)
                t_min, t_max = times[0], times[-1]
                duration = t_max - t_min
                if duration > 0:
                    new_start = min(
                        t_max + t_min - target.evidence_start_s,
                        t_max + t_min - target.evidence_end_s,
                    )
                    new_end = max(
                        t_max + t_min - target.evidence_start_s,
                        t_max + t_min - target.evidence_end_s,
                    )
                else:
                    new_start = target.evidence_start_s
                    new_end = target.evidence_end_s
                result.frames = sorted(result.frames, key=lambda frame: frame.t_seconds)
            else:
                new_start = target.evidence_start_s
                new_end = target.evidence_end_s

            result.target = TemporalTarget(
                change_detected=target.change_detected,
                change=inverted_dir,
                state_start=target.state_end,
                state_end=target.state_start,
                evidence_start_s=new_start,
                evidence_end_s=new_end,
                confidence=target.confidence,
                scene_observation=target.scene_observation,
            )
            return result

        if transform == "shuffled":
            result = copy.deepcopy(original)
            result.target = None
            return result

        return original

    def evaluate_transforms(
        self,
        predictions_by_transform: Dict[str, List[TemporalSample]],
        ground_truths: List[TemporalSample],
    ) -> Dict[str, EvaluationReport]:
        gt_map = {sample.sample_id: sample for sample in ground_truths}
        results: Dict[str, EvaluationReport] = {}
        for transform, predictions in predictions_by_transform.items():
            control_gts = [
                self._derive_control_target(gt_map[pred.sample_id], transform)
                if pred.sample_id in gt_map
                else pred
                for pred in predictions
            ]
            results[transform] = evaluate_batch(predictions, control_gts)
        return results


def attach_latency_record(report: EvaluationReport, frame_count: int, latency_ms: float) -> None:
    """Attach a hardware latency observation to an evaluation report."""
    report.latency_records[f"F{frame_count}"] = {"latency_ms": latency_ms}
