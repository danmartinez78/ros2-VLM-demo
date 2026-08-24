"""
CPU-only unit and integration tests for the temporal VLM distillation pipeline.

Covers all hardware-independent acceptance criteria:
1. Synthetic ordered-frame fixtures -> canonical manifest
2. Mock teacher generation, resume, parse, audit
3. Valid labels accepted, malformed labels rejected
4. Eight-frame sample with Image 9/10 reference -> rejected
5. Export to versioned SFT dataset
6. Deterministic train/val/test split with provenance
7. Training launcher: dry-run plan without model load
8. Evaluation over synthetic student outputs (temporal + schema metrics)
9. Existing test infrastructure unaffected
"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure the repo root is on the path for all imports.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from distillation.schema import (
    FrameRef,
    Provenance,
    TemporalSample,
    TemporalTarget,
    load_manifest,
    save_manifest,
    SCHEMA_VERSION,
)
from distillation.teacher import (
    FakeTeacherRuntime,
    TeacherLabelGenerator,
    TEACHER_PROMPT_VERSION,
)
from distillation.validator import (
    validate_sample,
    validate_batch,
    VALIDATORS,
)
from distillation.dataset import split_dataset, export_sft_dataset
from distillation.training import TrainingConfig, run_training
from distillation.evaluation import (
    EvaluationReport,
    evaluate_batch,
    attach_latency_record,
    ControlledSequenceEvaluator,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_frame(path: str, t: float) -> FrameRef:
    return FrameRef(path=path, t_seconds=t)


def _make_target(**kwargs) -> TemporalTarget:
    defaults = {
        "change_detected": True,
        "change": "approaching",
        "state_start": "vehicle distant",
        "state_end": "vehicle closer",
        "evidence_start_s": 0.25,
        "evidence_end_s": 1.75,
        "confidence": 0.92,
        "odd_observation": "vehicle closing distance",
    }
    defaults.update(kwargs)
    return TemporalTarget(**defaults)


def _make_sample(
    sample_id: str = "seq-000-0",
    n_frames: int = 4,
    with_target: bool = True,
) -> TemporalSample:
    frames = [_make_frame(f"frame_{i:03d}.jpg", i * 0.25) for i in range(n_frames)]
    target = _make_target(
        evidence_start_s=0.25,
        evidence_end_s=frames[-1].t_seconds,
    ) if with_target else None
    return TemporalSample(
        sample_id=sample_id,
        frames=frames,
        target=target,
        validation_status="pending",
    )


def _make_fake_teacher_response(**kwargs) -> str:
    defaults = {
        "change_detected": True,
        "change": "approaching",
        "state_start": "vehicle distant",
        "state_end": "vehicle closer",
        "evidence_start_s": 0.25,
        "evidence_end_s": 0.75,
        "confidence": 0.90,
        "odd_observation": "vehicle closing distance",
    }
    defaults.update(kwargs)
    return json.dumps(defaults)


# ---------------------------------------------------------------------------
# 1. Schema / manifest round-trip
# ---------------------------------------------------------------------------

class TestSchemaRoundTrip(unittest.TestCase):
    def test_frame_ref_round_trip(self):
        f = FrameRef(path="img/frame_001.jpg", t_seconds=0.25)
        self.assertEqual(FrameRef.from_dict(f.to_dict()), f)

    def test_temporal_target_round_trip(self):
        t = _make_target()
        self.assertEqual(TemporalTarget.from_dict(t.to_dict()), t)

    def test_sample_round_trip(self):
        s = _make_sample()
        restored = TemporalSample.from_dict(s.to_dict())
        self.assertEqual(restored.sample_id, s.sample_id)
        self.assertEqual(len(restored.frames), len(s.frames))
        self.assertIsNotNone(restored.target)

    def test_content_hash_stable(self):
        s = _make_sample()
        self.assertEqual(s.content_hash(), s.content_hash())

    def test_content_hash_changes(self):
        s1 = _make_sample("seq-001-0")
        s2 = _make_sample("seq-002-0")
        self.assertNotEqual(s1.content_hash(), s2.content_hash())

    def test_manifest_save_load(self):
        samples = [_make_sample(f"seq-{i:03d}-0") for i in range(5)]
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "manifest.json"
            save_manifest(samples, p)
            loaded = load_manifest(p)
        self.assertEqual(len(loaded), 5)
        self.assertEqual([s.sample_id for s in loaded], [s.sample_id for s in samples])

    def test_schema_version_present(self):
        s = _make_sample()
        self.assertEqual(s.schema_version, SCHEMA_VERSION)

    def test_sample_no_target(self):
        s = _make_sample(with_target=False)
        d = s.to_dict()
        self.assertIsNone(d["target"])
        restored = TemporalSample.from_dict(d)
        self.assertIsNone(restored.target)


# ---------------------------------------------------------------------------
# 2. Teacher generation (fake runtime)
# ---------------------------------------------------------------------------

class TestTeacherGeneration(unittest.TestCase):
    def _gen_dir(self, tmpdir: str) -> Path:
        return Path(tmpdir) / "teacher_output"

    def test_basic_generation(self):
        samples = [_make_sample(f"seq-{i:03d}-0", n_frames=4, with_target=False) for i in range(3)]
        runtime = FakeTeacherRuntime(
            responses={s.sample_id: _make_fake_teacher_response() for s in samples}
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = TeacherLabelGenerator(runtime=runtime, output_dir=self._gen_dir(tmpdir))
            results = gen.run(samples)
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertIsNotNone(r.target)
            self.assertEqual(r.provenance.teacher_model, "FakeTeacher-0B")
            self.assertEqual(r.provenance.teacher_prompt_version, TEACHER_PROMPT_VERSION)

    def test_resume_skips_completed(self):
        samples = [_make_sample(f"seq-{i:03d}-0", with_target=False) for i in range(3)]
        runtime = FakeTeacherRuntime(
            responses={s.sample_id: _make_fake_teacher_response() for s in samples}
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = TeacherLabelGenerator(runtime=runtime, output_dir=self._gen_dir(tmpdir))
            gen.run(samples)
            # Second run should not call generate again.
            runtime.call_log.clear()
            results = gen.run(samples)
        self.assertEqual(runtime.call_log, [])
        self.assertEqual(len(results), 3)

    def test_raw_response_preserved(self):
        raw = _make_fake_teacher_response(odd_observation="test audit trail")
        sample = _make_sample("seq-000-0", with_target=False)
        runtime = FakeTeacherRuntime(responses={"seq-000-0": raw})
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = TeacherLabelGenerator(runtime=runtime, output_dir=self._gen_dir(tmpdir))
            results = gen.run([sample])
        self.assertIn("test audit trail", results[0].raw_teacher_response)

    def test_malformed_response_gives_no_target(self):
        sample = _make_sample("seq-000-0", with_target=False)
        runtime = FakeTeacherRuntime(responses={"seq-000-0": "not valid json {"})
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = TeacherLabelGenerator(runtime=runtime, output_dir=self._gen_dir(tmpdir))
            results = gen.run([sample])
        self.assertIsNone(results[0].target)

    def test_deterministic_ordering(self):
        """Frames must be passed to the teacher in chronological order."""
        frames = [FrameRef(path=f"f{i}.jpg", t_seconds=float(i)) for i in range(4)]
        sample = TemporalSample(sample_id="seq-ord-0", frames=frames)
        runtime = FakeTeacherRuntime()
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = TeacherLabelGenerator(runtime=runtime, output_dir=Path(tmpdir) / "out")
            gen.run([sample])
            # The output file should preserve the frame order.
            out = json.loads((Path(tmpdir) / "out" / "seq-ord-0.json").read_text())
        ts = [f["t_seconds"] for f in out["frames"]]
        self.assertEqual(ts, sorted(ts))


# ---------------------------------------------------------------------------
# 3. Validation / filtering
# ---------------------------------------------------------------------------

class TestValidator(unittest.TestCase):
    def _valid_sample(self) -> TemporalSample:
        s = _make_sample("seq-val-0", n_frames=4)
        s.target = _make_target(evidence_start_s=0.0, evidence_end_s=0.75)
        return s

    def test_valid_sample_accepted(self):
        s = validate_sample(self._valid_sample())
        self.assertEqual(s.validation_status, "accepted")
        self.assertEqual(s.rejection_reasons, [])

    def test_missing_target_rejected(self):
        s = _make_sample(with_target=False)
        result = validate_sample(s)
        self.assertEqual(result.validation_status, "rejected")
        self.assertTrue(any("missing_target" in r for r in result.rejection_reasons))

    def test_confidence_out_of_range_rejected(self):
        s = self._valid_sample()
        s.target = _make_target(confidence=1.5, evidence_start_s=0.0, evidence_end_s=0.75)
        result = validate_sample(s)
        self.assertEqual(result.validation_status, "rejected")
        self.assertTrue(any("confidence_out_of_range" in r for r in result.rejection_reasons))

    def test_inverted_evidence_rejected(self):
        s = self._valid_sample()
        s.target = _make_target(evidence_start_s=1.5, evidence_end_s=0.5)
        result = validate_sample(s)
        self.assertEqual(result.validation_status, "rejected")
        self.assertTrue(any("evidence_time_inverted" in r for r in result.rejection_reasons))

    def test_non_monotonic_timestamps_rejected(self):
        frames = [FrameRef("f0.jpg", 0.0), FrameRef("f1.jpg", 0.5), FrameRef("f2.jpg", 0.25)]
        s = TemporalSample(
            sample_id="seq-nm-0",
            frames=frames,
            target=_make_target(evidence_start_s=0.0, evidence_end_s=0.25),
        )
        result = validate_sample(s)
        self.assertEqual(result.validation_status, "rejected")
        self.assertTrue(
            any("frame_timestamp_not_monotonic" in r for r in result.rejection_reasons)
        )

    def test_static_sequence_false_positive_rejected(self):
        s = _make_sample("seq-static-0_static")
        s.prompt_profile = "temporal_odd_v1_static"
        s.target = _make_target(change_detected=True, evidence_start_s=0.0, evidence_end_s=0.75)
        result = validate_sample(s)
        self.assertEqual(result.validation_status, "rejected")
        self.assertTrue(
            any("static_sequence_false_positive" in r for r in result.rejection_reasons)
        )

    # Regression case: Image 9/10 in an eight-frame sample
    def test_hallucinated_image9_eight_frame_rejected(self):
        """Image 9 reference in an 8-frame sample must be rejected."""
        frames = [FrameRef(f"f{i}.jpg", i * 0.25) for i in range(8)]
        # Target is clean - hallucination is only in the raw teacher response.
        target = _make_target(
            evidence_start_s=0.0,
            evidence_end_s=1.75,
            odd_observation="vehicle approaching",
        )
        # Raw teacher response includes the out-of-bounds image reference.
        raw_response = (
            '{"change_detected": true, "change": "approaching", '
            '"state_start": "vehicle distant", "state_end": "vehicle closer", '
            '"evidence_start_s": 0.0, "evidence_end_s": 1.75, '
            '"confidence": 0.9, "odd_observation": "vehicle approaching (see Image 9)"}'
        )
        s = TemporalSample(
            sample_id="seq-8frame-0",
            frames=frames,
            target=target,
            raw_teacher_response=raw_response,
        )
        result = validate_sample(s)
        self.assertEqual(result.validation_status, "rejected")
        reasons_text = " ".join(result.rejection_reasons)
        self.assertIn("hallucinated_frame_reference", reasons_text)
        self.assertIn("Image 9", reasons_text)

    def test_hallucinated_image10_eight_frame_rejected(self):
        """Image 10 reference in an 8-frame sample must be rejected."""
        frames = [FrameRef(f"f{i}.jpg", i * 0.25) for i in range(8)]
        raw = "vehicle approaching; refer to Image 10 for detail"
        s = TemporalSample(
            sample_id="seq-8frame-halluc10",
            frames=frames,
            target=_make_target(evidence_start_s=0.0, evidence_end_s=1.75),
            raw_teacher_response=raw,
        )
        result = validate_sample(s)
        self.assertEqual(result.validation_status, "rejected")
        self.assertTrue(
            any("hallucinated_frame_reference" in r for r in result.rejection_reasons)
        )

    def test_valid_image8_reference_accepted(self):
        """A reference to Image 8 in an 8-frame sample is fine."""
        frames = [FrameRef(f"f{i}.jpg", i * 0.25) for i in range(8)]
        raw = "change visible by Image 8"
        s = TemporalSample(
            sample_id="seq-8frame-valid8",
            frames=frames,
            target=_make_target(evidence_start_s=0.0, evidence_end_s=1.75),
            raw_teacher_response=raw,
        )
        result = validate_sample(s)
        self.assertEqual(result.validation_status, "accepted")

    def test_validate_batch_separates_accepted_rejected(self):
        valid_s = self._valid_sample()
        invalid_s = _make_sample("seq-bad-0", with_target=False)
        accepted, rejected = validate_batch([valid_s, invalid_s])
        self.assertEqual(len(accepted), 1)
        self.assertEqual(len(rejected), 1)
        self.assertEqual(accepted[0].sample_id, valid_s.sample_id)

    def test_validation_version_stamped(self):
        s = validate_sample(self._valid_sample())
        from distillation.validator import VALIDATION_VERSION
        self.assertEqual(s.provenance.validation_version, VALIDATION_VERSION)

    def test_no_frames_rejected(self):
        s = TemporalSample(sample_id="seq-noframe-0", frames=[], target=_make_target())
        result = validate_sample(s)
        self.assertEqual(result.validation_status, "rejected")


# ---------------------------------------------------------------------------
# 4. Dataset split and export
# ---------------------------------------------------------------------------

class TestDataset(unittest.TestCase):
    def _accepted_samples(self, n: int = 20) -> list:
        samples = []
        for i in range(n):
            s = _make_sample(f"group-{i // 3:02d}-{i % 3}", n_frames=4)
            s.target = _make_target(evidence_start_s=0.0, evidence_end_s=0.75)
            s.validation_status = "accepted"
            samples.append(s)
        return samples

    def test_split_deterministic(self):
        samples = self._accepted_samples(20)
        train1, val1, test1 = split_dataset(samples, seed=42)
        train2, val2, test2 = split_dataset(samples, seed=42)
        self.assertEqual([s.sample_id for s in train1], [s.sample_id for s in train2])
        self.assertEqual([s.sample_id for s in val1], [s.sample_id for s in val2])

    def test_split_different_seeds(self):
        samples = self._accepted_samples(20)
        train1, _, _ = split_dataset(samples, seed=42)
        train2, _, _ = split_dataset(samples, seed=99)
        # Different seeds should (usually) give different splits for large enough sets.
        # Not guaranteed for very small sets, but 20 samples is sufficient.
        self.assertNotEqual(
            [s.sample_id for s in train1],
            [s.sample_id for s in train2],
        )

    def test_split_covers_all_samples(self):
        samples = self._accepted_samples(20)
        train, val, test = split_dataset(samples)
        total = len(train) + len(val) + len(test)
        self.assertEqual(total, len(samples))

    def test_split_no_duplicates(self):
        samples = self._accepted_samples(20)
        train, val, test = split_dataset(samples)
        all_ids = [s.sample_id for s in train + val + test]
        self.assertEqual(len(all_ids), len(set(all_ids)))

    def test_export_creates_files(self):
        samples = self._accepted_samples(9)
        train, val, test = split_dataset(samples, seed=42)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "sft_dataset"
            manifest = export_sft_dataset(train, val, test, out)
            self.assertTrue((out / "train.jsonl").exists())
            self.assertTrue((out / "val.jsonl").exists())
            self.assertTrue((out / "test.jsonl").exists())
            self.assertTrue((out / "dataset_manifest.json").exists())

    def test_export_manifest_structure(self):
        samples = self._accepted_samples(9)
        train, val, test = split_dataset(samples, seed=42)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "sft_dataset"
            manifest = export_sft_dataset(train, val, test, out, repo_commit="abc1234")
        from distillation.dataset import EXPORTER_VERSION
        self.assertEqual(manifest["exporter_version"], EXPORTER_VERSION)
        self.assertEqual(manifest["repo_commit"], "abc1234")
        for split_name in ("train", "val", "test"):
            self.assertIn(split_name, manifest["splits"])
            self.assertIn("count", manifest["splits"][split_name])
            self.assertIn("hash", manifest["splits"][split_name])

    def test_export_jsonl_parseable(self):
        samples = self._accepted_samples(6)
        train, val, test = split_dataset(samples, seed=42)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "sft"
            export_sft_dataset(train, val, test, out)
            train_lines = (out / "train.jsonl").read_text().splitlines()
        for line in train_lines:
            if line.strip():
                obj = json.loads(line)
                self.assertIn("messages", obj)
                self.assertIn("sample_id", obj)


# ---------------------------------------------------------------------------
# 5. Training config and dry-run launcher
# ---------------------------------------------------------------------------

class TestTrainingConfig(unittest.TestCase):
    def _default_config(self) -> TrainingConfig:
        return TrainingConfig(
            experiment_id="test-exp-001",
            base_model="nvidia/Cosmos-Reason2-2B",
            dataset_dir="distillation/datasets/sft_v1",
            output_dir="distillation/runs/{experiment_id}",
        )

    def test_valid_config_no_errors(self):
        cfg = self._default_config()
        self.assertEqual(cfg.validate(), [])

    def test_empty_experiment_id_invalid(self):
        cfg = self._default_config()
        cfg.experiment_id = ""
        errors = cfg.validate()
        self.assertTrue(any("experiment_id" in e for e in errors))

    def test_bad_learning_rate_invalid(self):
        cfg = self._default_config()
        cfg.learning_rate = 5.0
        errors = cfg.validate()
        self.assertTrue(any("learning_rate" in e for e in errors))

    def test_fp16_and_bf16_both_true_invalid(self):
        cfg = self._default_config()
        cfg.fp16 = True
        cfg.bf16 = True
        errors = cfg.validate()
        self.assertTrue(any("fp16" in e and "bf16" in e for e in errors))

    def test_absolute_path_flagged(self):
        cfg = self._default_config()
        cfg.dataset_dir = "/absolute/path/to/dataset"
        errors = cfg.validate()
        self.assertTrue(any("absolute path" in e for e in errors))

    def test_dry_run_no_model_load(self):
        cfg = self._default_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_training(
                cfg, dry_run=True, output_dir_override=Path(tmpdir) / "run"
            )
        self.assertTrue(result["dry_run"])
        self.assertTrue(result["valid"])
        self.assertIn("plan", result)
        self.assertEqual(result["plan"]["experiment_id"], "test-exp-001")

    def test_dry_run_writes_effective_config(self):
        cfg = self._default_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "run"
            run_training(cfg, dry_run=True, output_dir_override=out)
            self.assertTrue((out / "effective_config.json").exists())
            saved = json.loads((out / "effective_config.json").read_text())
        self.assertEqual(saved["experiment_id"], "test-exp-001")
        from distillation.training import TRAINING_CONFIG_VERSION
        self.assertEqual(saved["config_version"], TRAINING_CONFIG_VERSION)

    def test_config_round_trip_json(self):
        cfg = self._default_config()
        d = cfg.to_dict()
        restored = TrainingConfig.from_dict(d)
        self.assertEqual(restored.experiment_id, cfg.experiment_id)
        self.assertEqual(restored.lora.r, cfg.lora.r)

    def test_dry_run_invalid_config_returns_errors(self):
        cfg = self._default_config()
        cfg.experiment_id = ""
        result = run_training(cfg, dry_run=True)
        self.assertFalse(result["valid"])
        self.assertTrue(len(result["errors"]) > 0)

    def test_config_hash_stable(self):
        cfg = self._default_config()
        self.assertEqual(cfg.config_hash(), cfg.config_hash())

    def test_real_training_raises_not_implemented(self):
        cfg = self._default_config()
        with self.assertRaises(NotImplementedError):
            run_training(cfg, dry_run=False)


# ---------------------------------------------------------------------------
# 6. Evaluation
# ---------------------------------------------------------------------------

class TestEvaluation(unittest.TestCase):
    def _pred_sample(
        self,
        sample_id: str,
        change_detected: bool = True,
        change: str = "approaching",
        confidence: float = 0.9,
    ) -> TemporalSample:
        frames = [FrameRef(f"f{i}.jpg", i * 0.25) for i in range(4)]
        target = TemporalTarget(
            change_detected=change_detected,
            change=change,
            state_start="vehicle distant",
            state_end="vehicle closer",
            evidence_start_s=0.25,
            evidence_end_s=0.75,
            confidence=confidence,
            odd_observation="vehicle closing distance",
        )
        return TemporalSample(sample_id=sample_id, frames=frames, target=target)

    def _gt_sample(self, sample_id: str, change_detected: bool = True) -> TemporalSample:
        return self._pred_sample(sample_id, change_detected=change_detected)

    def test_perfect_predictions(self):
        preds = [self._pred_sample(f"s{i}") for i in range(4)]
        gts = [self._gt_sample(f"s{i}") for i in range(4)]
        report = evaluate_batch(preds, gts)
        self.assertEqual(report.change_detection_accuracy, 1.0)
        self.assertEqual(report.direction_accuracy, 1.0)

    def test_all_wrong_predictions(self):
        preds = [self._pred_sample(f"s{i}", change_detected=False) for i in range(4)]
        gts = [self._gt_sample(f"s{i}", change_detected=True) for i in range(4)]
        report = evaluate_batch(preds, gts)
        self.assertEqual(report.change_detection_accuracy, 0.0)

    def test_schema_adherence_no_target(self):
        pred = TemporalSample(sample_id="s0", frames=[FrameRef("f0.jpg", 0.0)])
        gt = self._gt_sample("s0")
        report = evaluate_batch([pred], [gt])
        self.assertEqual(report.schema_adherence, 0.0)

    def test_static_false_positive_rate(self):
        preds = [self._pred_sample(f"s{i}", change_detected=True) for i in range(3)]
        gts = [self._gt_sample(f"s{i}", change_detected=False) for i in range(3)]
        report = evaluate_batch(preds, gts)
        self.assertEqual(report.static_false_positive_rate, 1.0)

    def test_hallucination_detection(self):
        frames = [FrameRef(f"f{i}.jpg", i * 0.25) for i in range(4)]
        target = TemporalTarget(
            change_detected=True, change="approaching",
            state_start="start", state_end="end",
            evidence_start_s=0.0, evidence_end_s=0.75,
            confidence=0.8, odd_observation="see Image 9",
        )
        pred = TemporalSample(sample_id="s0", frames=frames, target=target)
        report = evaluate_batch([pred], [])
        self.assertEqual(report.hallucinated_frame_reference_rate, 1.0)

    def test_latency_attachment(self):
        report = EvaluationReport(n_samples=1)
        attach_latency_record(report, 8, 415.3)
        self.assertIn("F8", report.latency_records)
        self.assertEqual(report.latency_records["F8"]["latency_ms"], 415.3)

    def test_evaluation_report_to_dict(self):
        preds = [self._pred_sample("s0")]
        gts = [self._gt_sample("s0")]
        report = evaluate_batch(preds, gts)
        d = report.to_dict()
        self.assertIn("change_detection_accuracy", d)
        self.assertIn("schema_adherence", d)

    def test_controlled_sequence_evaluator(self):
        evaluator = ControlledSequenceEvaluator(seed=42)
        sample = self._pred_sample("s0")
        reversed_s = evaluator.apply_transform(sample, "reversed")
        self.assertEqual(
            [f.t_seconds for f in reversed_s.frames],
            [f.t_seconds for f in reversed(sample.frames)],
        )

    def test_controlled_sequence_single_terminal(self):
        evaluator = ControlledSequenceEvaluator()
        sample = self._pred_sample("s0")
        result = evaluator.apply_transform(sample, "single_terminal_frame")
        self.assertEqual(len(result.frames), 1)
        self.assertEqual(result.frames[0].t_seconds, sample.frames[-1].t_seconds)

    def test_controlled_sequence_duplicated_frame(self):
        evaluator = ControlledSequenceEvaluator()
        sample = self._pred_sample("s0")
        result = evaluator.apply_transform(sample, "duplicated_frame")
        self.assertEqual(len(result.frames), len(sample.frames) + 1)

    def test_evaluate_transforms(self):
        evaluator = ControlledSequenceEvaluator()
        sample = self._pred_sample("s0")
        preds_by_transform = {
            "chronological": [sample],
            "reversed": [evaluator.apply_transform(sample, "reversed")],
        }
        gts = [self._gt_sample("s0")]
        reports = evaluator.evaluate_transforms(preds_by_transform, gts)
        self.assertIn("chronological", reports)
        self.assertIn("reversed", reports)


# ---------------------------------------------------------------------------
# End-to-end integration test
# ---------------------------------------------------------------------------

class TestEndToEndPipeline(unittest.TestCase):
    """
    Full pipeline: synthetic manifest -> teacher -> validate -> split -> export
    -> dry-run training launch -> evaluation.
    """

    def test_full_pipeline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # 1. Build a synthetic manifest of 12 samples.
            samples = []
            for i in range(12):
                frames = [FrameRef(f"f{j}.jpg", j * 0.25) for j in range(4)]
                samples.append(
                    TemporalSample(sample_id=f"group-{i // 4:02d}-{i % 4}", frames=frames)
                )
            manifest_path = tmp / "manifest.json"
            save_manifest(samples, manifest_path)
            loaded = load_manifest(manifest_path)
            self.assertEqual(len(loaded), 12)

            # 2. Teacher generation with fake runtime.
            fake_responses = {s.sample_id: _make_fake_teacher_response() for s in loaded}
            runtime = FakeTeacherRuntime(responses=fake_responses)
            gen = TeacherLabelGenerator(runtime=runtime, output_dir=tmp / "teacher_out")
            labelled = gen.run(loaded)
            self.assertEqual(len(labelled), 12)

            # 3. Validate.
            accepted, rejected = validate_batch(labelled)
            self.assertGreater(len(accepted), 0)
            self.assertEqual(len(rejected), 0)

            # 4. Split.
            train, val, test = split_dataset(accepted, seed=42)
            self.assertGreater(len(train), 0)

            # 5. Export.
            manifest_exported = export_sft_dataset(
                train, val, test, tmp / "sft", repo_commit="test-commit"
            )
            self.assertIn("train", manifest_exported["splits"])

            # 6. Dry-run training launch.
            cfg = TrainingConfig(
                experiment_id="e2e-test",
                base_model="nvidia/Cosmos-Reason2-2B",
                dataset_dir="distillation/datasets/sft_v1",
                output_dir="distillation/runs/{experiment_id}",
            )
            train_result = run_training(cfg, dry_run=True, output_dir_override=tmp / "run")
            self.assertTrue(train_result["valid"])

            # 7. Evaluation over synthetic outputs.
            preds = accepted[:3]
            gts = accepted[:3]
            report = evaluate_batch(preds, gts)
            self.assertIsNotNone(report.schema_adherence)
            self.assertEqual(report.schema_adherence, 1.0)


if __name__ == "__main__":
    unittest.main()
