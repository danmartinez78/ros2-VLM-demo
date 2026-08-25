#!/usr/bin/env python3
"""CPU-only regression tests for the generic temporal distillation pipeline."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from distillation.evaluation import ControlledSequenceEvaluator, compare_sample  # noqa: E402
from distillation.schema import FrameRef, TemporalSample, TemporalTarget  # noqa: E402
from distillation.teacher import FakeTeacherRuntime, TeacherLabelGenerator  # noqa: E402
from distillation.validator import validate_sample  # noqa: E402


def frames(n: int = 4) -> list[FrameRef]:
    return [FrameRef(path=f"frame_{i}.png", t_seconds=float(i)) for i in range(n)]


def target(
    *,
    change_detected: bool = True,
    change: str = "approaching",
    scene_observation: str = "person approaches camera",
) -> TemporalTarget:
    return TemporalTarget(
        change_detected=change_detected,
        change=change,
        state_start="person far",
        state_end="person near",
        evidence_start_s=0.0,
        evidence_end_s=3.0,
        confidence=0.9,
        scene_observation=scene_observation,
    )


def sample(sample_id: str = "s1", *, tgt: TemporalTarget | None = None) -> TemporalSample:
    return TemporalSample(
        sample_id=sample_id,
        frames=frames(),
        prompt_profile="temporal_observation_v1",
        target=tgt,
    )


class SchemaTests(unittest.TestCase):
    def test_round_trip_preserves_scene_observation(self) -> None:
        original = sample(tgt=target())
        restored = TemporalSample.from_dict(original.to_dict())
        self.assertEqual(restored.prompt_profile, "temporal_observation_v1")
        self.assertEqual(restored.target.scene_observation, "person approaches camera")
        self.assertEqual(restored.frame_count(), 4)

    def test_strict_boolean_validation(self) -> None:
        data = target().to_dict()
        data["change_detected"] = "false"
        with self.assertRaises(TypeError):
            TemporalTarget.from_dict(data)


class TeacherTests(unittest.TestCase):
    def test_fake_teacher_generates_generic_target(self) -> None:
        response = json.dumps(target(change="receding").to_dict())
        runtime = FakeTeacherRuntime(responses={"s1": response})
        with tempfile.TemporaryDirectory() as tmp:
            generator = TeacherLabelGenerator(runtime, Path(tmp))
            result = generator.run([sample()])[0]
        self.assertIsNotNone(result.target)
        self.assertEqual(result.target.change, "receding")
        self.assertEqual(result.target.scene_observation, "person approaches camera")

    def test_cache_fingerprint_changes_with_representation(self) -> None:
        runtime = FakeTeacherRuntime()
        with tempfile.TemporaryDirectory() as tmp:
            first = TeacherLabelGenerator(runtime, Path(tmp), input_representation="ordered_images")
            first.run([sample()])
            calls_after_first = len(runtime.call_log)
            second = TeacherLabelGenerator(runtime, Path(tmp), input_representation="native_video")
            second.run([sample()])
        self.assertGreater(len(runtime.call_log), calls_after_first)


class ValidationTests(unittest.TestCase):
    def test_rejects_hallucinated_frame_reference(self) -> None:
        tgt = target(scene_observation="change becomes visible in Image 9")
        result = validate_sample(sample(tgt=tgt))
        self.assertEqual(result.validation_status, "rejected")
        self.assertTrue(any("hallucinated_frame_reference" in r for r in result.rejection_reasons))

    def test_accepts_valid_target(self) -> None:
        result = validate_sample(sample(tgt=target()))
        self.assertEqual(result.validation_status, "accepted")
        self.assertEqual(result.rejection_reasons, [])

    def test_static_profile_rejects_false_change(self) -> None:
        s = sample(tgt=target(change_detected=True))
        s.prompt_profile = "temporal_observation_v1_static"
        result = validate_sample(s)
        self.assertEqual(result.validation_status, "rejected")
        self.assertTrue(any("static_sequence_false_positive" in r for r in result.rejection_reasons))


class EvaluationTests(unittest.TestCase):
    def test_reverse_control_inverts_known_direction(self) -> None:
        original = sample(tgt=target(change="approaching"))
        evaluator = ControlledSequenceEvaluator()
        reversed_gt = evaluator._derive_control_target(original, "reversed")
        self.assertEqual(reversed_gt.target.change, "receding")
        self.assertEqual(reversed_gt.target.state_start, original.target.state_end)
        self.assertEqual(reversed_gt.target.state_end, original.target.state_start)

    def test_compare_sample_scores_direction(self) -> None:
        gt = sample(tgt=target(change="approaching"))
        pred = sample(tgt=target(change="approaching"))
        metrics = compare_sample(pred, gt)
        self.assertTrue(metrics.change_detected_correct)
        self.assertTrue(metrics.direction_correct)
        self.assertFalse(metrics.hallucinated_ref)


if __name__ == "__main__":
    unittest.main()
