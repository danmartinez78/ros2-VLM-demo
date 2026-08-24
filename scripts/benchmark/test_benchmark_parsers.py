"""
CI-safe tests for benchmark parsers and schemas.

These tests do NOT require TensorRT, CUDA, ROS, or any hardware.
They validate:
  - JSONL parsing logic in collect_ros_metrics.py
  - Per-frame metric computation (timing breakdowns)
  - Aggregate statistics (mean/percentile/dropped/cold-start)
  - CSV export
  - generate_benchmark_report.py comparison generation and text formatting
  - JSON schema structure (syntactic validity)
  - benchmark_metadata.py data collection utilities (mocked system calls)

Hardware benchmarks (llm_bench, llm_inference) are NOT run here.
"""

from __future__ import annotations

import io
import json
import sys
import textwrap
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

# Add scripts/benchmark to the path so imports work without installation.
_BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_BENCH_DIR))

from collect_ros_metrics import (
    build_report,
    compute_aggregate,
    compute_frame_metrics,
    parse_jsonl,
    write_csv,
)
from generate_benchmark_report import (
    format_text_report,
    generate_comparison,
    load_native_results,
)


# ── fixtures ─────────────────────────────────────────────────────────────────

_NS = 1_000_000  # 1 ms in nanoseconds


def _make_frame_record(
    *,
    frame_seq: int = 1,
    inference_seconds: float = 1.5,
    queue_delay_ms: float = 2.0,
    convert_ms: float = 5.0,
    ipc_overhead_ms: float = 3.0,
    publication_ms: float = 1.0,
    success: bool = True,
    error: str = "",
    dropped_before: int = 0,
) -> dict[str, Any]:
    """Build a synthetic JSONL frame record with consistent nanosecond timestamps."""
    subscribe = 1_000_000_000
    dequeue = subscribe + int(queue_delay_ms * _NS)
    convert_done = dequeue + int(convert_ms * _NS)
    infer_done = convert_done + int(inference_seconds * 1_000 * _NS) + int(ipc_overhead_ms * _NS)
    publish_done = infer_done + int(publication_ms * _NS)
    return {
        "record_type": "frame",
        "frame_seq": frame_seq,
        "image_stamp_ns": subscribe - 50 * _NS,
        "subscribe_wall_ns": subscribe,
        "dequeue_wall_ns": dequeue,
        "convert_done_ns": convert_done,
        "infer_done_ns": infer_done,
        "publish_done_ns": publish_done,
        "inference_seconds": inference_seconds,
        "dropped_before": dropped_before,
        "success": success,
        "error": error,
    }


def _make_session_start(
    node_init_ns: int = 400_000_000,
    worker_ready_ns: int = 500_000_000,
) -> dict[str, Any]:
    return {
        "record_type": "session_start",
        "node_init_wall_ns": node_init_ns,
        "worker_ready_wall_ns": worker_ready_ns,
        "task_profile": "scene_description",
        "prompt_version": "v1",
        "prompt_config_hash": "deadbeef0000",
        "max_generate_length": 64,
        "sample_period_seconds": 2.0,
        "image_max_width": 1280,
        "jpeg_quality": 90,
        "drop_old_frames": True,
    }


def _make_session_end(*, dropped: int = 2) -> dict[str, Any]:
    return {
        "record_type": "session_end",
        "received": 100,
        "sampled": 50,
        "dropped": dropped,
        "success": 48,
        "failure": 2,
    }


def _build_jsonl(records: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(r) + "\n" for r in records)


# ── JSONL parsing tests ───────────────────────────────────────────────────────


class TestParseJsonl(unittest.TestCase):
    def test_parse_valid_session_with_frames(self):
        records = [
            _make_session_start(),
            _make_frame_record(frame_seq=1),
            _make_frame_record(frame_seq=2),
            _make_session_end(dropped=1),
        ]
        path = Path("/tmp/test_bench.jsonl")
        path.write_text(_build_jsonl(records), encoding="utf-8")
        try:
            session_start, frames, session_end = parse_jsonl(path)
            self.assertIsNotNone(session_start)
            self.assertEqual(len(frames), 2)
            self.assertIsNotNone(session_end)
            self.assertEqual(session_start["task_profile"], "scene_description")
            self.assertEqual(frames[0]["frame_seq"], 1)
            self.assertEqual(session_end["dropped"], 1)
        finally:
            path.unlink(missing_ok=True)

    def test_parse_frames_only(self):
        records = [
            _make_frame_record(frame_seq=1),
            _make_frame_record(frame_seq=2),
        ]
        path = Path("/tmp/test_bench_frames_only.jsonl")
        path.write_text(_build_jsonl(records), encoding="utf-8")
        try:
            session_start, frames, session_end = parse_jsonl(path)
            self.assertIsNone(session_start)
            self.assertIsNone(session_end)
            self.assertEqual(len(frames), 2)
        finally:
            path.unlink(missing_ok=True)

    def test_parse_skips_blank_lines(self):
        content = "\n" + json.dumps(_make_frame_record()) + "\n\n"
        path = Path("/tmp/test_bench_blanks.jsonl")
        path.write_text(content, encoding="utf-8")
        try:
            _, frames, _ = parse_jsonl(path)
            self.assertEqual(len(frames), 1)
        finally:
            path.unlink(missing_ok=True)

    def test_parse_raises_on_invalid_json(self):
        path = Path("/tmp/test_bench_bad.jsonl")
        path.write_text('{"record_type":"frame"}\n{bad json}\n', encoding="utf-8")
        try:
            with self.assertRaises(ValueError):
                parse_jsonl(path)
        finally:
            path.unlink(missing_ok=True)


# ── per-frame metric tests ────────────────────────────────────────────────────


class TestComputeFrameMetrics(unittest.TestCase):
    def test_basic_timing_breakdown(self):
        record = _make_frame_record(
            inference_seconds=1.5,
            queue_delay_ms=2.0,
            convert_ms=5.0,
            ipc_overhead_ms=3.0,
            publication_ms=1.0,
        )
        metrics = compute_frame_metrics(record)

        self.assertAlmostEqual(metrics["inference_ms"], 1500.0, places=1)
        self.assertAlmostEqual(metrics["queue_delay_ms"], 2.0, delta=0.5)
        self.assertAlmostEqual(metrics["image_convert_ms"], 5.0, places=1)
        self.assertAlmostEqual(metrics["ipc_overhead_ms"], 3.0, delta=0.5)
        self.assertAlmostEqual(metrics["publication_ms"], 1.0, delta=0.5)
        self.assertGreater(metrics["ros_overhead_ms"], 0)
        self.assertGreater(metrics["total_worker_ms"], 1500.0)
        self.assertTrue(metrics["success"])
        self.assertEqual(metrics["frame_seq"], 1)

    def test_ros_overhead_excludes_inference(self):
        record = _make_frame_record(
            inference_seconds=2.0,
            queue_delay_ms=2.0,
            convert_ms=4.0,
            ipc_overhead_ms=2.0,
            publication_ms=1.0,
        )
        metrics = compute_frame_metrics(record)
        # ros_overhead = queue_delay + convert + ipc_overhead + publication
        expected_ros = (
            metrics["queue_delay_ms"] + metrics["image_convert_ms"]
            + metrics["ipc_overhead_ms"] + metrics["publication_ms"]
        )
        self.assertAlmostEqual(metrics["ros_overhead_ms"], expected_ros, places=3)
        # inference_ms should NOT be included in ros_overhead
        self.assertAlmostEqual(metrics["inference_ms"], 2000.0, places=1)
        self.assertNotAlmostEqual(metrics["ros_overhead_ms"], metrics["total_worker_ms"], places=0)

    def test_failed_frame_preserved(self):
        record = _make_frame_record(success=False, error="backend exception: timeout")
        metrics = compute_frame_metrics(record)
        self.assertFalse(metrics["success"])
        self.assertIn("timeout", metrics["error"])

    def test_ipc_overhead_non_negative(self):
        # Even if timestamps are slightly off, ipc_overhead_ms must be >= 0
        record = _make_frame_record(inference_seconds=2.0, ipc_overhead_ms=0.0)
        metrics = compute_frame_metrics(record)
        self.assertGreaterEqual(metrics["ipc_overhead_ms"], 0.0)


# ── aggregate statistic tests ─────────────────────────────────────────────────


class TestComputeAggregate(unittest.TestCase):
    def _make_metrics(self, n: int, *, warmup: int = 0) -> list[dict[str, Any]]:
        records = [
            _make_frame_record(
                frame_seq=i + 1,
                inference_seconds=1.0 + i * 0.1,
                convert_ms=5.0,
            )
            for i in range(n)
        ]
        return [compute_frame_metrics(r) for r in records]

    def test_warmup_excluded_from_stats(self):
        metrics = self._make_metrics(10, warmup=3)
        agg = compute_aggregate(metrics, None, None, warmup_frames=3)
        self.assertEqual(agg["total_frames"], 10)
        self.assertEqual(agg["warmup_frames"], 3)
        self.assertEqual(agg["measured_frames"], 7)
        self.assertEqual(agg["successful_frames"], 7)

    def test_aggregate_all_successful(self):
        metrics = self._make_metrics(5)
        agg = compute_aggregate(metrics, None, None)
        self.assertEqual(agg["failed_frames"], 0)
        self.assertIsNotNone(agg["inference_ms"]["mean"])
        self.assertIsNotNone(agg["ros_overhead_ms"]["mean"])

    def test_cold_start_computed_from_worker_ready(self):
        # worker_ready_ns=500_000_000; first frame subscribe at 1_000_000_000
        # (dequeue is 2 ms after subscribe = 1_002_000_000)
        # ready_to_first_frame = (dequeue - worker_ready) / 1e6 = (1_002_000_000 - 500_000_000) / 1e6 = 502 ms
        session_start = _make_session_start(node_init_ns=400_000_000, worker_ready_ns=500_000_000)
        metrics = [compute_frame_metrics(_make_frame_record(queue_delay_ms=2.0))]
        agg = compute_aggregate(metrics, session_start, None)
        # dequeue = subscribe + 2ms = 1_002_000_000
        self.assertAlmostEqual(agg["ready_to_first_frame_ms"], 502.0, places=0)

    def test_backend_init_ms_computed(self):
        session_start = _make_session_start(node_init_ns=400_000_000, worker_ready_ns=500_000_000)
        metrics = [compute_frame_metrics(_make_frame_record())]
        agg = compute_aggregate(metrics, session_start, None)
        # backend_init = (500_000_000 - 400_000_000) / 1e6 = 100 ms
        self.assertAlmostEqual(agg["backend_init_ms"], 100.0, places=0)

    def test_cold_start_none_without_session_start(self):
        metrics = [compute_frame_metrics(_make_frame_record())]
        agg = compute_aggregate(metrics, None, None)
        self.assertIsNone(agg["ready_to_first_frame_ms"])

    def test_dropped_count_from_session_end(self):
        metrics = [compute_frame_metrics(_make_frame_record(dropped_before=5))]
        session_end = _make_session_end(dropped=7)
        agg = compute_aggregate(metrics, None, session_end)
        self.assertEqual(agg["total_dropped"], 7)

    def test_empty_frame_list(self):
        agg = compute_aggregate([], None, None)
        self.assertEqual(agg["total_frames"], 0)
        self.assertIsNone(agg["inference_ms"]["mean"])

    def test_failure_not_counted_in_timing_stats(self):
        records = [
            _make_frame_record(frame_seq=1, success=True, inference_seconds=1.0),
            _make_frame_record(frame_seq=2, success=False, inference_seconds=0.0),
        ]
        metrics = [compute_frame_metrics(r) for r in records]
        agg = compute_aggregate(metrics, None, None)
        self.assertEqual(agg["successful_frames"], 1)
        self.assertEqual(agg["failed_frames"], 1)
        # Mean should be computed from the 1 successful frame only
        self.assertAlmostEqual(agg["inference_ms"]["mean"], 1000.0, places=1)


# ── build_report tests ────────────────────────────────────────────────────────


class TestBuildReport(unittest.TestCase):
    def test_report_schema_keys(self):
        session_start = _make_session_start()
        raw_frames = [_make_frame_record(frame_seq=i + 1) for i in range(5)]
        frame_metrics = [compute_frame_metrics(r) for r in raw_frames]
        report = build_report(session_start, frame_metrics, _make_session_end(), warmup_frames=2)

        self.assertIn("metadata", report)
        self.assertIn("aggregate", report)
        self.assertIn("frames", report)
        self.assertEqual(len(report["frames"]), 5)
        # session_start fields should be copied into metadata
        self.assertEqual(report["metadata"]["task_profile"], "scene_description")
        self.assertEqual(report["metadata"]["warmup_frames"], 2)

    def test_report_with_external_metadata_merge(self):
        ext_meta = {"model_name": "Cosmos-Reason2-8B", "quantization": "nvfp4"}
        frame_metrics = [compute_frame_metrics(_make_frame_record())]
        report = build_report(None, frame_metrics, None, metadata=ext_meta)
        self.assertEqual(report["metadata"]["model_name"], "Cosmos-Reason2-8B")


# ── CSV export tests ──────────────────────────────────────────────────────────


class TestWriteCsv(unittest.TestCase):
    def test_csv_round_trip(self):
        import csv as _csv

        frame_metrics = [
            compute_frame_metrics(_make_frame_record(frame_seq=i + 1)) for i in range(3)
        ]
        path = Path("/tmp/test_bench_metrics.csv")
        try:
            write_csv(frame_metrics, path)
            with path.open("r", encoding="utf-8") as fh:
                reader = _csv.DictReader(fh)
                rows = list(reader)
            self.assertEqual(len(rows), 3)
            self.assertIn("frame_seq", rows[0])
            self.assertIn("inference_ms", rows[0])
            self.assertIn("ros_overhead_ms", rows[0])
        finally:
            path.unlink(missing_ok=True)


# ── comparison report tests ───────────────────────────────────────────────────


class TestGenerateComparison(unittest.TestCase):
    def _make_ros_report(self, *, inf_ms: float = 1500.0, ros_ms: float = 50.0) -> dict[str, Any]:
        records = [_make_frame_record(inference_seconds=inf_ms / 1000.0, ipc_overhead_ms=ros_ms)]
        frame_metrics = [compute_frame_metrics(r) for r in records]
        return build_report(
            _make_session_start(),
            frame_metrics,
            _make_session_end(),
            metadata={"model_name": "Cosmos-Reason2-8B", "quantization": "nvfp4"},
        )

    def test_comparison_keys_present(self):
        ros_report = self._make_ros_report()
        comparison = generate_comparison(ros_report, None)

        self.assertIn("native_engine", comparison)
        self.assertIn("ros_overhead", comparison)
        self.assertIn("pipeline_total", comparison)
        self.assertIn("metadata", comparison)
        self.assertIn("generated_at", comparison)

    def test_fractions_sum_to_approx_one(self):
        ros_report = self._make_ros_report(inf_ms=1000.0, ros_ms=50.0)
        comparison = generate_comparison(ros_report, None)
        total = comparison["pipeline_total"]
        ros_frac = total.get("ros_fraction_of_total")
        eng_frac = total.get("engine_fraction_of_total")
        if ros_frac is not None and eng_frac is not None:
            self.assertAlmostEqual(ros_frac + eng_frac, 1.0, delta=0.1)

    def test_text_report_contains_section_headers(self):
        ros_report = self._make_ros_report()
        comparison = generate_comparison(ros_report, None)
        text = format_text_report(comparison)

        self.assertIn("Native Engine Timing", text)
        self.assertIn("ROS Pipeline Overhead", text)
        self.assertIn("End-to-End Pipeline", text)

    def test_no_native_dir_is_handled_gracefully(self):
        ros_report = self._make_ros_report()
        # native_results is None — no native dir provided
        comparison = generate_comparison(ros_report, None)
        self.assertIsNone(comparison["native_engine"]["llm_bench_prefill"])
        self.assertIsNone(comparison["native_engine"]["llm_bench_decode"])
        self.assertIsNone(comparison["native_engine"]["llm_bench_visual"])


# ── schema structure tests ────────────────────────────────────────────────────


class TestSchemaStructure(unittest.TestCase):
    """Validate that schema JSON files are parseable and have required keys."""

    _SCHEMA_DIR = _BENCH_DIR / "schemas"

    def _load_schema(self, name: str) -> dict[str, Any]:
        path = self._SCHEMA_DIR / name
        self.assertTrue(path.exists(), f"Schema file not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def test_ros_benchmark_schema_valid_json(self):
        schema = self._load_schema("ros_benchmark_result.schema.json")
        self.assertEqual(schema["type"], "object")
        self.assertIn("metadata", schema["properties"])
        self.assertIn("aggregate", schema["properties"])
        self.assertIn("frames", schema["properties"])
        self.assertIn("required", schema)
        self.assertIn("definitions", schema)

    def test_native_benchmark_schema_valid_json(self):
        schema = self._load_schema("native_benchmark_result.schema.json")
        self.assertEqual(schema["type"], "object")
        self.assertIn("metadata", schema["properties"])
        self.assertIn("run_id", schema["properties"])
        self.assertIn("recorded_at", schema["properties"])

    def test_ros_schema_frame_record_has_required_inference_ms(self):
        schema = self._load_schema("ros_benchmark_result.schema.json")
        frame_def = schema["definitions"]["frame_record"]
        self.assertIn("inference_ms", frame_def["required"])
        self.assertIn("success", frame_def["required"])

    def test_ros_schema_timing_stats_has_percentiles(self):
        schema = self._load_schema("ros_benchmark_result.schema.json")
        timing_stats = schema["definitions"]["timing_stats"]
        self.assertIn("mean", timing_stats["properties"])
        self.assertIn("p50", timing_stats["properties"])
        self.assertIn("p95", timing_stats["properties"])


# ── benchmark_metadata mock tests ─────────────────────────────────────────────


class TestBenchmarkMetadata(unittest.TestCase):
    """Test metadata collection with mocked subprocess calls."""

    def test_collect_platform_metadata_returns_dict(self):
        from benchmark_metadata import collect_platform_metadata

        with patch("subprocess.run") as mock_run:
            mock_run.return_value.stdout = ""
            mock_run.return_value.returncode = 1
            meta = collect_platform_metadata()

        self.assertIn("arch", meta)
        self.assertIn("recorded_at", meta)
        # Fields that couldn't be obtained should be None, not raise
        for key in ("jetpack_version", "cuda_version", "tensorrt_version", "gpu_name"):
            self.assertIn(key, meta)

    def test_collect_model_metadata_no_edge_llm_root(self):
        from benchmark_metadata import collect_model_metadata

        meta = collect_model_metadata(model_name="test-model")
        self.assertEqual(meta["model_name"], "test-model")
        self.assertIsNone(meta["edge_llm_commit"])

    def test_collect_all_metadata_merges_sections(self):
        from benchmark_metadata import collect_all_metadata

        with patch("subprocess.run") as mock_run:
            mock_run.return_value.stdout = ""
            mock_run.return_value.returncode = 1
            meta = collect_all_metadata(model_name="demo", task_profile="scene_description")

        self.assertIn("arch", meta)
        self.assertIn("model_name", meta)
        self.assertIn("task_profile", meta)
        self.assertEqual(meta["task_profile"], "scene_description")

    def test_collect_engine_provenance_managed_manifest(self):
        import tempfile
        from benchmark_metadata import collect_engine_provenance

        with tempfile.TemporaryDirectory() as tmpdir:
            profile_dir = Path(tmpdir) / "Cosmos-Reason2-8B" / "engines" / "thor-f8"
            llm_dir = profile_dir / "llm"
            llm_dir.mkdir(parents=True)
            manifest_path = profile_dir / "engine-manifest.json"
            manifest_path.write_text(json.dumps({
                "model_name": "Cosmos-Reason2-8B",
                "engine_profile_id": "thor-f8",
                "engine_paths": {
                    "llm_dir": str(llm_dir),
                    "multimodal_dir": str(profile_dir),
                },
            }), encoding="utf-8")

            provenance = collect_engine_provenance(
                llm_engine_dir=str(llm_dir / ".." / "llm"),
                multimodal_engine_dir=str(profile_dir / "."),
                model_name="Cosmos-Reason2-8B",
                engine_profile_id="thor-f8",
            )

        self.assertEqual(provenance["engine_manifest_status"], "matched")
        self.assertEqual(provenance["engine_profile_id"], "thor-f8")
        self.assertEqual(provenance["llm_engine_dir"], str(llm_dir.resolve()))
        self.assertEqual(provenance["multimodal_engine_dir"], str(profile_dir.resolve()))
        self.assertEqual(provenance["engine_manifest_path"], str(manifest_path.resolve()))
        self.assertTrue(provenance["engine_identity"].startswith("Cosmos-Reason2-8B/thor-f8@"))

    def test_collect_engine_provenance_legacy_fallback(self):
        import tempfile
        from benchmark_metadata import collect_engine_provenance

        with tempfile.TemporaryDirectory() as tmpdir:
            legacy_root = Path(tmpdir) / "Cosmos-Reason2-8B" / "engine"
            llm_dir = legacy_root / "llm"
            llm_dir.mkdir(parents=True)
            provenance = collect_engine_provenance(
                llm_engine_dir=str(llm_dir),
                multimodal_engine_dir=str(legacy_root),
                model_name="Cosmos-Reason2-8B",
            )

        self.assertEqual(provenance["engine_profile_id"], "legacy")
        self.assertIsNone(provenance["engine_manifest_path"])
        self.assertIsNone(provenance["engine_manifest_sha256"])
        self.assertTrue(provenance["engine_identity"].startswith("Cosmos-Reason2-8B/legacy@"))

    def test_collect_engine_provenance_missing_manifest_managed_layout(self):
        import tempfile
        from benchmark_metadata import collect_engine_provenance

        with tempfile.TemporaryDirectory() as tmpdir:
            profile_dir = Path(tmpdir) / "Cosmos-Reason2-8B" / "engines" / "thor-f8"
            llm_dir = profile_dir / "llm"
            llm_dir.mkdir(parents=True)
            provenance = collect_engine_provenance(
                llm_engine_dir=str(llm_dir),
                multimodal_engine_dir=str(profile_dir),
            )

        self.assertEqual(provenance["engine_profile_id"], "thor-f8")
        self.assertEqual(provenance["engine_manifest_status"], "missing")
        self.assertTrue(any("no engine-manifest" in w for w in provenance["provenance_warnings"]))

    def test_collect_engine_provenance_manifest_mismatch(self):
        import tempfile
        from benchmark_metadata import collect_engine_provenance

        with tempfile.TemporaryDirectory() as tmpdir:
            profile_dir = Path(tmpdir) / "Cosmos-Reason2-8B" / "engines" / "thor-f8"
            llm_dir = profile_dir / "llm"
            llm_dir.mkdir(parents=True)
            other_profile_dir = Path(tmpdir) / "other"
            other_profile_dir.mkdir()
            manifest_path = profile_dir / "engine-manifest.json"
            manifest_path.write_text(json.dumps({
                "model_name": "Cosmos-Reason2-8B",
                "engine_profile_id": "thor-current",
                "engine_paths": {
                    "llm_dir": str(other_profile_dir / "llm"),
                    "multimodal_dir": str(other_profile_dir),
                },
            }), encoding="utf-8")

            provenance = collect_engine_provenance(
                llm_engine_dir=str(llm_dir),
                multimodal_engine_dir=str(profile_dir),
                model_name="Cosmos-Reason2-8B",
                engine_profile_id="thor-f8",
            )

        self.assertEqual(provenance["engine_manifest_status"], "mismatch")
        warnings = "\n".join(provenance["provenance_warnings"])
        self.assertIn("thor-f8", warnings)
        self.assertIn("thor-current", warnings)

    def test_collect_server_engine_provenance_uses_live_server_paths(self):
        import tempfile
        from benchmark_metadata import collect_server_engine_provenance

        with tempfile.TemporaryDirectory() as tmpdir:
            profile_dir = Path(tmpdir) / "Cosmos-Reason2-8B" / "engines" / "thor-f8"
            llm_dir = profile_dir / "llm"
            llm_dir.mkdir(parents=True)
            manifest_path = profile_dir / "engine-manifest.json"
            manifest_path.write_text(json.dumps({
                "model_name": "Cosmos-Reason2-8B",
                "engine_profile_id": "thor-f8",
                "engine_paths": {
                    "llm_dir": str(llm_dir),
                    "multimodal_dir": str(profile_dir),
                },
            }), encoding="utf-8")

            with patch("benchmark_metadata._socket_listener_pid", return_value=4242), \
                 patch("benchmark_metadata._proc_argv") as mock_proc_argv:
                def _mock_proc_argv(pid, index):
                    self.assertEqual(pid, 4242)
                    return {
                        1: str(llm_dir),
                        2: str(profile_dir),
                        4: "/tmp/edge_vlm.sock",
                    }.get(index, "")

                mock_proc_argv.side_effect = _mock_proc_argv
                provenance = collect_server_engine_provenance(
                    socket_path="/tmp/edge_vlm.sock",
                    model_name="Cosmos-Reason2-8B",
                    engine_profile_id="thor-f8",
                )

        self.assertEqual(provenance["engine_manifest_status"], "matched")
        self.assertEqual(provenance["llm_engine_dir"], str(llm_dir.resolve()))
        self.assertEqual(provenance["multimodal_engine_dir"], str(profile_dir.resolve()))
        self.assertEqual(provenance["server_pid"], 4242)
        self.assertEqual(provenance["server_socket_path"], str(Path("/tmp/edge_vlm.sock").resolve()))
        self.assertEqual(provenance["provenance_source"], "server_process")

    def test_collect_server_engine_provenance_raises_without_listener(self):
        from benchmark_metadata import collect_server_engine_provenance

        with patch("benchmark_metadata._socket_listener_pid", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "no live edge_vlm_server listener"):
                collect_server_engine_provenance(socket_path="/tmp/missing.sock")

    def test_collect_server_engine_provenance_missing_manifest(self):
        import tempfile
        from benchmark_metadata import collect_server_engine_provenance

        with tempfile.TemporaryDirectory() as tmpdir:
            profile_dir = Path(tmpdir) / "Cosmos-Reason2-8B" / "engines" / "thor-f8"
            llm_dir = profile_dir / "llm"
            llm_dir.mkdir(parents=True)

            with patch("benchmark_metadata._socket_listener_pid", return_value=4242), \
                 patch("benchmark_metadata._proc_argv") as mock_proc_argv:
                def _mock_proc_argv(pid, index):
                    self.assertEqual(pid, 4242)
                    return {
                        1: str(llm_dir),
                        2: str(profile_dir),
                        4: "/tmp/edge_vlm.sock",
                    }.get(index, "")

                mock_proc_argv.side_effect = _mock_proc_argv
                provenance = collect_server_engine_provenance(socket_path="/tmp/edge_vlm.sock")

        self.assertEqual(provenance["engine_manifest_status"], "missing")
        self.assertEqual(provenance["engine_profile_id"], "thor-f8")
        self.assertTrue(any("no engine-manifest" in w for w in provenance["provenance_warnings"]))


# ── end-to-end round-trip test ────────────────────────────────────────────────


class TestEndToEndRoundTrip(unittest.TestCase):
    """
    Parse synthetic JSONL → compute metrics → build report → generate comparison.
    Validates the full data pipeline without any hardware.
    """

    def test_full_pipeline_cpu_only(self):
        # Build synthetic JSONL content
        records = [_make_session_start()]
        for i in range(10):
            records.append(
                _make_frame_record(
                    frame_seq=i + 1,
                    inference_seconds=1.5 + i * 0.05,
                    convert_ms=4.0 + i * 0.1,
                    ipc_overhead_ms=2.0,
                    publication_ms=0.8,
                    success=(i != 7),  # one failure
                )
            )
        records.append(_make_session_end(dropped=3))

        path = Path("/tmp/test_e2e_bench.jsonl")
        path.write_text(_build_jsonl(records), encoding="utf-8")

        try:
            session_start, raw_frames, session_end = parse_jsonl(path)
            self.assertEqual(len(raw_frames), 10)

            frame_metrics = [compute_frame_metrics(r) for r in raw_frames]
            report = build_report(
                session_start, frame_metrics, session_end, warmup_frames=2
            )

            # Validate report structure
            self.assertEqual(report["aggregate"]["total_frames"], 10)
            self.assertEqual(report["aggregate"]["warmup_frames"], 2)
            self.assertEqual(report["aggregate"]["measured_frames"], 8)
            self.assertEqual(report["aggregate"]["failed_frames"], 1)
            self.assertEqual(report["aggregate"]["total_dropped"], 3)
            self.assertIsNotNone(report["aggregate"]["inference_ms"]["mean"])
            self.assertIsNotNone(report["aggregate"]["ros_overhead_ms"]["p95"])

            # ROS overhead should be much less than inference time
            ros_mean = report["aggregate"]["ros_overhead_ms"]["mean"]
            inf_mean = report["aggregate"]["inference_ms"]["mean"]
            self.assertLess(ros_mean, inf_mean)

            # Generate comparison
            comparison = generate_comparison(report, None)
            self.assertIsNotNone(comparison["pipeline_total"]["total_worker_ms_mean"])

            text = format_text_report(comparison)
            self.assertIn("Native Engine Timing", text)
            self.assertIn("ROS Pipeline Overhead", text)
        finally:
            path.unlink(missing_ok=True)


# ── native benchmark script tests ─────────────────────────────────────────────


class TestNativeBenchmarkDryRun(unittest.TestCase):
    """
    Verify that dry-run output contains NVIDIA's documented CLI flags without
    executing any binaries.  The environment variables required by the script
    are satisfied by stubs; all modes default to included.
    """

    _SCRIPT = Path(__file__).resolve().parent / "run_native_benchmarks.sh"

    def _run_dry(self, extra_args: list[str] | None = None, env_updates: dict[str, str] | None = None) -> str:
        import subprocess
        env = {
            "PATH": "/usr/bin:/bin",
            "TENSORRT_EDGE_LLM_ROOT": "/fake/edgellm",
            "EDGE_VLM_LLM_ENGINE_DIR": "/fake/llm_engine",
            "EDGE_VLM_MULTIMODAL_ENGINE_DIR": "/fake/mm_engine",
            "EDGELLM_PLUGIN_PATH": "/fake/plugin.so",
        }
        if env_updates:
            env.update(env_updates)
        cmd = ["bash", str(self._SCRIPT), "--dry-run"]
        if extra_args:
            cmd.extend(extra_args)
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True
        )
        return result.stdout + result.stderr

    def test_prefill_contains_batchsize_and_inputlen(self):
        out = self._run_dry(["--batch-size", "4", "--input-len", "256"])
        self.assertIn("--batchSize", out)
        self.assertIn("--inputLen", out)
        self.assertIn("--mode prefill", out)

    def test_decode_contains_batchsize_and_pastkv(self):
        out = self._run_dry(["--batch-size", "4", "--past-kv-len", "512"])
        self.assertIn("--pastKVLen", out)
        self.assertIn("--mode decode", out)

    def test_visual_contains_imagesize(self):
        out = self._run_dry(["--image-size", "448x448"])
        self.assertIn("--imageSize", out)
        self.assertIn("448x448", out)
        self.assertIn("--mode visual", out)
        self.assertIn("--engineDir /fake/mm_engine/visual", out)
        self.assertNotIn("--multimodalEngineDir", out)

    def test_visual_override_same_canonical_path_is_allowed(self):
        out = self._run_dry(
            ["--skip-prefill", "--skip-decode", "--skip-profile"],
            env_updates={"EDGE_VLM_VISUAL_ENGINE_DIR": "/fake/mm_engine/subdir/../visual"},
        )
        self.assertIn("--engineDir /fake/mm_engine/visual", out)

    def test_visual_override_mismatch_is_rejected(self):
        import subprocess

        env = {
            "PATH": "/usr/bin:/bin",
            "TENSORRT_EDGE_LLM_ROOT": "/fake/edgellm",
            "EDGE_VLM_LLM_ENGINE_DIR": "/fake/llm_engine",
            "EDGE_VLM_MULTIMODAL_ENGINE_DIR": "/fake/mm_engine",
            "EDGE_VLM_VISUAL_ENGINE_DIR": "/different/visual",
            "EDGELLM_PLUGIN_PATH": "/fake/plugin.so",
        }
        result = subprocess.run(
            ["bash", str(self._SCRIPT), "--dry-run", "--skip-prefill", "--skip-decode", "--skip-profile"],
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("EDGE_VLM_VISUAL_ENGINE_DIR must resolve to /fake/mm_engine/visual", result.stderr)

    def test_llm_bench_uses_warmup_and_iterations_flags(self):
        out = self._run_dry(["--warmup", "5", "--iterations", "20"])
        # All three llm_bench modes must use --warmup and --iterations
        self.assertEqual(out.count("--warmup 5"), 3)
        self.assertEqual(out.count("--iterations 20"), 3)

    def test_all_modes_include_profile_flag(self):
        out = self._run_dry()
        # Each of the three llm_bench dry-run lines must include --profile
        lines_with_mode = [l for l in out.splitlines() if "--mode" in l and "llm_bench" in l.lower()]
        for line in lines_with_mode:
            self.assertIn("--profile", line, f"--profile missing from: {line}")

    def test_inference_uses_warmup_and_dumpprofile(self):
        out = self._run_dry(
            ["--input-vlm-json", "/fake/input.json", "--inference-warmup", "7"]
        )
        self.assertIn("--dumpProfile", out)
        self.assertIn("--profileOutputFile", out)
        self.assertIn("--warmup 7", out)

    def test_skip_flags_omit_modes(self):
        out = self._run_dry(["--skip-prefill", "--skip-visual"])
        self.assertNotIn("--mode prefill", out)
        self.assertNotIn("--mode visual", out)
        self.assertIn("--mode decode", out)

    def test_visual_does_not_contain_inputimage(self):
        # --inputImage is not a documented llm_bench flag; only --imageSize is used.
        out = self._run_dry()
        self.assertNotIn("--inputImage", out)

    def test_quick_mode_uses_smaller_parameters(self):
        out_default = self._run_dry()
        out_quick = self._run_dry(["--quick"])
        # Default must contain 2048; quick must not
        self.assertIn("2048", out_default)
        self.assertNotIn("2048", out_quick)
        self.assertIn("--imageSize 320x320", out_quick)


class TestNativeBenchmarkDefaultFlags(unittest.TestCase):
    """
    Verify that the default invocation (no overrides) uses NVIDIA's published
    benchmark workload parameters:
      --batch-size 1  --input-len 2048  --past-kv-len 2048  --image-size 1024x2048
      --warmup 3  --iterations 10  --inference-warmup 10
    """

    _SCRIPT = Path(__file__).resolve().parent / "run_native_benchmarks.sh"

    def _run_dry(self, extra_args: list[str] | None = None) -> str:
        import subprocess
        env = {
            "PATH": "/usr/bin:/bin",
            "TENSORRT_EDGE_LLM_ROOT": "/fake/edgellm",
            "EDGE_VLM_LLM_ENGINE_DIR": "/fake/llm_engine",
            "EDGE_VLM_MULTIMODAL_ENGINE_DIR": "/fake/mm_engine",
            "EDGELLM_PLUGIN_PATH": "/fake/plugin.so",
        }
        cmd = ["bash", str(self._SCRIPT), "--dry-run"]
        if extra_args:
            cmd.extend(extra_args)
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        return result.stdout + result.stderr

    def test_prefill_default_flags(self):
        out = self._run_dry(["--skip-decode", "--skip-visual", "--skip-profile"])
        self.assertIn("--mode prefill", out)
        self.assertIn("--batchSize 1", out)
        self.assertIn("--inputLen 2048", out)
        self.assertIn("--warmup 3", out)
        self.assertIn("--iterations 10", out)
        self.assertIn("--profile", out)

    def test_decode_default_flags(self):
        out = self._run_dry(["--skip-prefill", "--skip-visual", "--skip-profile"])
        self.assertIn("--mode decode", out)
        self.assertIn("--batchSize 1", out)
        self.assertIn("--pastKVLen 2048", out)
        self.assertIn("--warmup 3", out)
        self.assertIn("--iterations 10", out)
        self.assertIn("--profile", out)

    def test_visual_default_flags(self):
        out = self._run_dry(["--skip-prefill", "--skip-decode", "--skip-profile"])
        self.assertIn("--mode visual", out)
        self.assertIn("--imageSize 1024x2048", out)
        self.assertIn("--warmup 3", out)
        self.assertIn("--iterations 10", out)
        self.assertIn("--profile", out)
        self.assertIn("--engineDir /fake/mm_engine/visual", out)
        self.assertNotIn("--multimodalEngineDir", out)
        # --inputImage must not appear; only --imageSize is used for the synthetic benchmark
        self.assertNotIn("--inputImage", out)

    def test_llm_inference_default_flags(self):
        out = self._run_dry([
            "--skip-prefill", "--skip-decode", "--skip-visual",
            "--input-vlm-json", "/fake/input.json",
        ])
        self.assertIn("--warmup 10", out)
        self.assertIn("--dumpProfile", out)
        self.assertIn("--profileOutputFile", out)

    def test_failure_exits_nonzero_and_manifest_errors_array(self):
        """A failing benchmark must exit 1 and the manifest errors field must be a list."""
        import subprocess
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_build = Path(tmpdir) / "build" / "examples" / "llm"
            fake_build.mkdir(parents=True)
            # llm_bench exits 1 (simulates a benchmark failure)
            llm_bench = fake_build / "llm_bench"
            llm_bench.write_text("#!/bin/sh\nexit 1\n")
            llm_bench.chmod(0o755)
            llm_inference = fake_build / "llm_inference"
            llm_inference.write_text("#!/bin/sh\nexit 0\n")
            llm_inference.chmod(0o755)

            out_dir = Path(tmpdir) / "bench_out"
            env = {
                "PATH": "/usr/bin:/bin",
                "TENSORRT_EDGE_LLM_ROOT": tmpdir,
                "EDGE_VLM_LLM_ENGINE_DIR": "/fake/llm_engine",
                "EDGE_VLM_MULTIMODAL_ENGINE_DIR": "/fake/mm_engine",
                "EDGELLM_PLUGIN_PATH": "/fake/plugin.so",
            }
            result = subprocess.run(
                [
                    "bash", str(self._SCRIPT),
                    "--skip-decode", "--skip-visual", "--skip-profile",
                    "--output-dir", str(out_dir),
                ],
                env=env, capture_output=True, text=True,
            )

            self.assertNotEqual(result.returncode, 0,
                                "Script must exit nonzero when a requested benchmark fails")

            manifest_path = out_dir / "manifest.json"
            if not manifest_path.exists():
                self.skipTest("manifest not written")
            with manifest_path.open() as fh:
                manifest = json.load(fh)
            self.assertIsInstance(manifest["errors"], list,
                                  "errors must be a JSON array, not a string")
            self.assertTrue(len(manifest["errors"]) > 0,
                            "errors array must be non-empty after a failure")


class TestNativeBenchmarkFailureExitsNonzero(unittest.TestCase):
    """
    Verify that a failing benchmark mode causes the script to exit nonzero
    rather than swallowing the error with `|| true`.
    """

    _SCRIPT = Path(__file__).resolve().parent / "run_native_benchmarks.sh"

    def test_failing_prefill_exits_nonzero(self):
        import subprocess
        import tempfile

        # Create a fake llm_bench that exits 1
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_build = Path(tmpdir) / "build" / "examples" / "llm"
            fake_build.mkdir(parents=True)
            llm_bench = fake_build / "llm_bench"
            llm_bench.write_text("#!/bin/sh\nexit 1\n")
            llm_bench.chmod(0o755)
            llm_inference = fake_build / "llm_inference"
            llm_inference.write_text("#!/bin/sh\nexit 1\n")
            llm_inference.chmod(0o755)

            env = {
                "PATH": "/usr/bin:/bin",
                "TENSORRT_EDGE_LLM_ROOT": tmpdir,
                "EDGE_VLM_LLM_ENGINE_DIR": "/fake/llm_engine",
                "EDGE_VLM_MULTIMODAL_ENGINE_DIR": "/fake/mm_engine",
                "EDGELLM_PLUGIN_PATH": "/fake/plugin.so",
            }
            result = subprocess.run(
                [
                    "bash", str(self._SCRIPT),
                    "--skip-decode", "--skip-visual", "--skip-profile",
                    "--output-dir", str(Path(tmpdir) / "out"),
                ],
                env=env, capture_output=True, text=True,
            )
        self.assertNotEqual(result.returncode, 0,
                            "Script must exit nonzero when a benchmark fails")

    def test_all_skipped_exits_zero(self):
        import subprocess

        env = {
            "PATH": "/usr/bin:/bin",
            "TENSORRT_EDGE_LLM_ROOT": "/fake/edgellm",
            "EDGE_VLM_LLM_ENGINE_DIR": "/fake/llm_engine",
            "EDGE_VLM_MULTIMODAL_ENGINE_DIR": "/fake/mm_engine",
            "EDGELLM_PLUGIN_PATH": "/fake/plugin.so",
        }
        result = subprocess.run(
            [
                "bash", str(self._SCRIPT),
                "--dry-run",
                "--skip-prefill", "--skip-decode", "--skip-visual", "--skip-profile",
            ],
            env=env, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0,
                         f"Script should exit 0 when all modes skipped (stderr: {result.stderr})")


class TestManifestArrayTypes(unittest.TestCase):
    """
    Verify that the manifest JSON written by run_native_benchmarks.sh uses
    proper JSON arrays for skipped_modes and errors, not bash-quoted strings.
    """

    _SCRIPT = Path(__file__).resolve().parent / "run_native_benchmarks.sh"

    def test_skipped_modes_is_json_array(self):
        import subprocess
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_build = Path(tmpdir) / "build" / "examples" / "llm"
            fake_build.mkdir(parents=True)
            # Fake binaries that succeed
            for name in ("llm_bench", "llm_inference"):
                b = fake_build / name
                b.write_text("#!/bin/sh\necho 'ok'\n")
                b.chmod(0o755)

            out_dir = Path(tmpdir) / "bench_out"
            env = {
                "PATH": "/usr/bin:/bin",
                "TENSORRT_EDGE_LLM_ROOT": tmpdir,
                "EDGE_VLM_LLM_ENGINE_DIR": "/fake/llm_engine",
                "EDGE_VLM_MULTIMODAL_ENGINE_DIR": "/fake/mm_engine",
                "EDGELLM_PLUGIN_PATH": "/fake/plugin.so",
            }
            subprocess.run(
                [
                    "bash", str(self._SCRIPT),
                    "--skip-decode", "--skip-visual", "--skip-profile",
                    "--output-dir", str(out_dir),
                ],
                env=env, capture_output=True, text=True,
            )

            manifest_path = out_dir / "manifest.json"
            if not manifest_path.exists():
                self.skipTest("manifest not written (likely missing Python or shell)")
            with manifest_path.open() as fh:
                manifest = json.load(fh)

        self.assertIsInstance(manifest["skipped_modes"], list,
                              "skipped_modes must be a JSON array")
        self.assertIsInstance(manifest["errors"], list,
                              "errors must be a JSON array")
        # decode + visual + profile were skipped
        self.assertIn("decode", manifest["skipped_modes"])
        self.assertIn("visual", manifest["skipped_modes"])
        self.assertIn("profile", manifest["skipped_modes"])

    def test_manifest_includes_engine_provenance(self):
        import subprocess
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_build = Path(tmpdir) / "build" / "examples" / "llm"
            fake_build.mkdir(parents=True)
            for name in ("llm_bench", "llm_inference"):
                binary = fake_build / name
                binary.write_text("#!/bin/sh\necho 'ok'\n", encoding="utf-8")
                binary.chmod(0o755)

            profile_dir = Path(tmpdir) / "workspace" / "Cosmos-Reason2-8B" / "engines" / "thor-f8"
            llm_dir = profile_dir / "llm"
            llm_dir.mkdir(parents=True)
            manifest_path = profile_dir / "engine-manifest.json"
            manifest_path.write_text(json.dumps({
                "model_name": "Cosmos-Reason2-8B",
                "engine_profile_id": "thor-f8",
                "engine_paths": {
                    "llm_dir": str(llm_dir),
                    "multimodal_dir": str(profile_dir),
                },
            }), encoding="utf-8")

            out_dir = Path(tmpdir) / "bench_out"
            env = {
                "PATH": "/usr/bin:/bin",
                "TENSORRT_EDGE_LLM_ROOT": tmpdir,
                "EDGE_VLM_MODEL_NAME": "Cosmos-Reason2-8B",
                "EDGE_VLM_ENGINE_PROFILE_ID": "thor-f8",
                "EDGE_VLM_LLM_ENGINE_DIR": str(llm_dir),
                "EDGE_VLM_MULTIMODAL_ENGINE_DIR": str(profile_dir),
                "EDGELLM_PLUGIN_PATH": "/fake/plugin.so",
            }
            subprocess.run(
                [
                    "bash", str(self._SCRIPT),
                    "--skip-decode", "--skip-visual", "--skip-profile",
                    "--output-dir", str(out_dir),
                ],
                env=env, capture_output=True, text=True, check=False,
            )

            manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(manifest["engine_provenance"]["engine_profile_id"], "thor-f8")
        self.assertEqual(
            manifest["engine_provenance"]["llm_engine_dir"],
            str(llm_dir.resolve()),
        )


class TestComparisonMetadata(unittest.TestCase):
    def test_generate_comparison_preserves_engine_provenance_fields(self):
        ros_report = {
            "metadata": {
                "model_name": "Cosmos-Reason2-8B",
                "engine_profile_id": "thor-f8",
                "llm_engine_dir": "/workspace/engines/thor-f8/llm",
                "multimodal_engine_dir": "/workspace/engines/thor-f8",
                "engine_manifest_path": "/workspace/engines/thor-f8/engine-manifest.json",
                "engine_manifest_sha256": "a" * 64,
                "engine_identity": "Cosmos-Reason2-8B/thor-f8@aaaaaaaaaaaa",
                "engine_provenance": {"engine_identity": "Cosmos-Reason2-8B/thor-f8@aaaaaaaaaaaa"},
            },
            "aggregate": {
                "inference_ms": {"mean": 100.0, "p50": 100.0, "p95": 100.0},
                "ros_overhead_ms": {"mean": 20.0, "p50": 20.0, "p95": 20.0},
                "ipc_overhead_ms": {"mean": 10.0, "p50": 10.0},
                "image_convert_ms": {"mean": 5.0, "p50": 5.0},
                "publication_ms": {"mean": 5.0, "p50": 5.0},
                "total_worker_ms": {"mean": 120.0, "p50": 120.0, "p95": 120.0},
                "total_dropped": 0,
                "failed_frames": 0,
                "ready_to_first_frame_ms": 0.0,
            },
        }

        comparison = generate_comparison(ros_report, None)
        meta = comparison["metadata"]
        self.assertEqual(meta["engine_profile_id"], "thor-f8")
        self.assertEqual(meta["engine_identity"], "Cosmos-Reason2-8B/thor-f8@aaaaaaaaaaaa")
        self.assertEqual(meta["llm_engine_dir"], "/workspace/engines/thor-f8/llm")

    def test_format_text_report_includes_engine_identity(self):
        comparison = {
            "generated_at": "2025-01-01T00:00:00Z",
            "metadata": {
                "model_name": "Cosmos-Reason2-8B",
                "engine_profile_id": "thor-f8",
                "quantization": "fp8",
                "engine_identity": "Cosmos-Reason2-8B/thor-f8@aaaaaaaaaaaa",
                "llm_engine_dir": "/workspace/engines/thor-f8/llm",
                "multimodal_engine_dir": "/workspace/engines/thor-f8",
                "edge_llm_commit": "abc123",
                "max_generate_length": 64,
                "image_max_width": 1280,
                "jpeg_quality": 90,
                "warmup_frames": 0,
                "measured_frames": 1,
                "sample_period_seconds": 1.0,
                "platform": {},
            },
            "native_engine": {"source": "", "inference_ms_mean": 1, "inference_ms_p50": 1, "inference_ms_p95": 1},
            "ros_overhead": {"source": "", "image_convert_ms_mean": 1, "image_convert_ms_p50": 1, "ipc_overhead_ms_mean": 1, "ipc_overhead_ms_p50": 1, "publication_ms_mean": 1, "publication_ms_p50": 1, "ros_overhead_ms_mean": 1, "ros_overhead_ms_p50": 1, "ros_overhead_ms_p95": 1, "total_dropped": 0, "failed_frames": 0, "ready_to_first_frame_ms": 0},
            "pipeline_total": {"total_worker_ms_mean": 1, "total_worker_ms_p50": 1, "total_worker_ms_p95": 1, "ros_fraction_of_total": 0.5, "engine_fraction_of_total": 0.5},
        }
        text = format_text_report(comparison)
        self.assertIn("Engine identity", text)
        self.assertIn("thor-f8", text)


class TestQueueDelayMetric(unittest.TestCase):
    """Verify that queue_delay_ms is computed from subscribe_wall_ns."""

    def test_queue_delay_positive(self):
        record = _make_frame_record(queue_delay_ms=5.0)
        metrics = compute_frame_metrics(record)
        self.assertAlmostEqual(metrics["queue_delay_ms"], 5.0, delta=0.5)

    def test_queue_delay_zero_when_no_subscribe_wall(self):
        record = _make_frame_record()
        del record["subscribe_wall_ns"]
        metrics = compute_frame_metrics(record)
        self.assertEqual(metrics["queue_delay_ms"], 0.0)

    def test_queue_delay_included_in_ros_overhead(self):
        record = _make_frame_record(queue_delay_ms=10.0, convert_ms=2.0, ipc_overhead_ms=1.0, publication_ms=0.5)
        metrics = compute_frame_metrics(record)
        self.assertGreaterEqual(metrics["ros_overhead_ms"], metrics["queue_delay_ms"])
        self.assertAlmostEqual(
            metrics["ros_overhead_ms"],
            metrics["queue_delay_ms"] + metrics["image_convert_ms"]
            + metrics["ipc_overhead_ms"] + metrics["publication_ms"],
            places=3,
        )


class TestColdStartVsBackendInit(unittest.TestCase):
    """Verify ready_to_first_frame_ms and backend_init_ms use correct reference points."""

    def test_ready_to_first_frame_uses_worker_ready_not_node_init(self):
        # node_init=400ms, worker_ready=600ms, first_dequeue=1002ms (subscribe=1000ms + 2ms queue)
        session_start = _make_session_start(node_init_ns=400_000_000, worker_ready_ns=600_000_000)
        metrics = [compute_frame_metrics(_make_frame_record(queue_delay_ms=2.0))]
        agg = compute_aggregate(metrics, session_start, None)
        # ready_to_first_frame = (1_002_000_000 - 600_000_000) / 1e6 = 402 ms
        self.assertAlmostEqual(agg["ready_to_first_frame_ms"], 402.0, places=0)
        # Should NOT be 602 ms (which would use node_init_ns)
        self.assertFalse(abs(agg["ready_to_first_frame_ms"] - 602.0) < 2.0)

    def test_backend_init_ms_is_worker_ready_minus_node_init(self):
        session_start = _make_session_start(node_init_ns=400_000_000, worker_ready_ns=600_000_000)
        metrics = [compute_frame_metrics(_make_frame_record())]
        agg = compute_aggregate(metrics, session_start, None)
        self.assertAlmostEqual(agg["backend_init_ms"], 200.0, places=0)


if __name__ == "__main__":
    unittest.main()
