"""
CPU-only CI tests for the VLM latency characterization benchmark.

These tests do NOT require TensorRT, CUDA, ROS, or any hardware.
They validate:
  - JSONL parsing (vlm_latency_report.parse_jsonl)
  - Per-condition aggregation including null stage timing handling
  - Token scaling table construction
  - Direct-vs-ROS comparison computation
  - Full report building (preserves raw records, correct structure)
  - Text report formatting (expected sections present)
  - JSON schema syntactic validity
  - Experiment matrix construction (build_condition_spec, prompt_hash)
  - Shell script syntax (bash -n dry-run check)
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

_BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_BENCH_DIR))

from vlm_latency_report import (  # noqa: E402
    EXPERIMENT_MATRIX,
    PROMPT_TEXTS,
    aggregate_condition,
    build_condition_spec,
    build_report,
    compute_direct_ros_comparison,
    compute_token_scaling,
    format_text_report,
    parse_jsonl,
    prompt_hash,
)


# ── fixtures ──────────────────────────────────────────────────────────────────

_RUN_ID = "20250101_120000"


def _make_record(
    *,
    condition: str = "A",
    path: str = "direct",
    image_id: str = "red_panda",
    prompt_id: str = "terse_id",
    max_output_tokens: int = 16,
    actual_output_tokens: int | None = 12,
    success: bool = True,
    error: str | None = None,
    total_latency_ms: float | None = 500.0,
    visual_preprocess_ms: float | None = None,
    ttft_ms: float | None = None,
    decode_ms: float | None = None,
    decode_tokens_per_sec: float | None = None,
    iteration: int = 0,
    warmup: bool = False,
    model_name: str | None = "TestModel-8B",
) -> dict[str, Any]:
    return {
        "schema_version": "1",
        "record_type": "inference",
        "run_id": _RUN_ID,
        "recorded_at": "2025-01-01T12:00:00Z",
        "condition": condition,
        "path": path,
        "image_id": image_id,
        "image_path": f"/tmp/{image_id}.jpg",
        "image_width_px": None,
        "image_height_px": None,
        "prompt_id": prompt_id,
        "prompt_hash": "abc123def456",
        "max_output_tokens": max_output_tokens,
        "actual_output_tokens": actual_output_tokens,
        "success": success,
        "error": error,
        "total_latency_ms": total_latency_ms,
        "visual_preprocess_ms": visual_preprocess_ms,
        "ttft_ms": ttft_ms,
        "decode_ms": decode_ms,
        "decode_tokens_per_sec": decode_tokens_per_sec,
        "model_name": model_name,
        "iteration": iteration,
        "warmup": warmup,
        "tegrastats": None,
    }


def _jsonl_from(records: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(r) + "\n" for r in records)


# ── JSONL parsing tests ───────────────────────────────────────────────────────


class TestParseJsonl(unittest.TestCase):
    def test_parses_valid_records(self):
        records = [_make_record(condition="A"), _make_record(condition="B")]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            f.write(_jsonl_from(records))
            path = Path(f.name)
        try:
            parsed = parse_jsonl(path)
            self.assertEqual(len(parsed), 2)
            self.assertEqual(parsed[0]["condition"], "A")
            self.assertEqual(parsed[1]["condition"], "B")
        finally:
            path.unlink(missing_ok=True)

    def test_skips_wrong_schema_version(self):
        bad = _make_record()
        bad["schema_version"] = "99"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            f.write(json.dumps(bad) + "\n")
            path = Path(f.name)
        try:
            parsed = parse_jsonl(path)
            self.assertEqual(len(parsed), 0)
        finally:
            path.unlink(missing_ok=True)

    def test_skips_wrong_record_type(self):
        wrong = _make_record()
        wrong["record_type"] = "session_start"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            f.write(json.dumps(wrong) + "\n")
            path = Path(f.name)
        try:
            parsed = parse_jsonl(path)
            self.assertEqual(len(parsed), 0)
        finally:
            path.unlink(missing_ok=True)

    def test_skips_malformed_json_lines(self):
        good = _make_record(condition="C")
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            f.write("NOT_JSON\n")
            f.write(json.dumps(good) + "\n")
            f.write("{broken\n")
            path = Path(f.name)
        try:
            parsed = parse_jsonl(path)
            self.assertEqual(len(parsed), 1)
            self.assertEqual(parsed[0]["condition"], "C")
        finally:
            path.unlink(missing_ok=True)

    def test_skips_empty_lines(self):
        good = _make_record()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            f.write("\n\n")
            f.write(json.dumps(good) + "\n")
            f.write("   \n")
            path = Path(f.name)
        try:
            parsed = parse_jsonl(path)
            self.assertEqual(len(parsed), 1)
        finally:
            path.unlink(missing_ok=True)


# ── aggregation tests ─────────────────────────────────────────────────────────


class TestAggregateCondition(unittest.TestCase):
    def test_basic_stats(self):
        records = [
            _make_record(total_latency_ms=400.0, warmup=False),
            _make_record(total_latency_ms=500.0, warmup=False),
            _make_record(total_latency_ms=600.0, warmup=False),
        ]
        agg = aggregate_condition(records)
        self.assertEqual(agg["n_measured"], 3)
        self.assertAlmostEqual(agg["total_latency_ms"]["mean"], 500.0, places=1)
        self.assertAlmostEqual(agg["total_latency_ms"]["min"], 400.0, places=1)
        self.assertAlmostEqual(agg["total_latency_ms"]["max"], 600.0, places=1)

    def test_warmup_records_excluded_from_aggregates(self):
        records = [
            _make_record(total_latency_ms=9999.0, warmup=True),
            _make_record(total_latency_ms=500.0, warmup=False),
        ]
        agg = aggregate_condition(records)
        self.assertEqual(agg["n_warmup"], 1)
        self.assertEqual(agg["n_measured"], 1)
        self.assertAlmostEqual(agg["total_latency_ms"]["mean"], 500.0, places=1)

    def test_failed_records_excluded_from_timing(self):
        records = [
            _make_record(success=False, total_latency_ms=None, error="timeout"),
            _make_record(total_latency_ms=500.0),
        ]
        agg = aggregate_condition(records)
        self.assertEqual(agg["n_failed"], 1)
        self.assertEqual(agg["n_measured"], 1)
        self.assertAlmostEqual(agg["total_latency_ms"]["mean"], 500.0, places=1)

    def test_null_stage_timings_reported_as_unavailable(self):
        records = [
            _make_record(ttft_ms=None, decode_ms=None, visual_preprocess_ms=None),
        ]
        agg = aggregate_condition(records)
        self.assertFalse(agg["stage_timings_available"]["ttft"])
        self.assertFalse(agg["stage_timings_available"]["decode"])
        self.assertFalse(agg["stage_timings_available"]["visual_preprocess"])
        # The stats dict should carry the availability flag, not fabricated values.
        self.assertIn("available", agg["ttft_ms"])
        self.assertFalse(agg["ttft_ms"]["available"])

    def test_stage_timings_aggregated_when_present(self):
        records = [
            _make_record(ttft_ms=50.0, decode_ms=400.0, visual_preprocess_ms=30.0),
            _make_record(ttft_ms=60.0, decode_ms=420.0, visual_preprocess_ms=25.0),
        ]
        agg = aggregate_condition(records)
        self.assertTrue(agg["stage_timings_available"]["ttft"])
        self.assertAlmostEqual(agg["ttft_ms"]["mean"], 55.0, places=1)
        self.assertAlmostEqual(agg["decode_ms"]["mean"], 410.0, places=1)
        self.assertAlmostEqual(agg["visual_preprocess_ms"]["mean"], 27.5, places=1)

    def test_empty_records(self):
        agg = aggregate_condition([])
        self.assertEqual(agg["n_total"], 0)
        self.assertEqual(agg["n_measured"], 0)
        self.assertIsNone(agg["total_latency_ms"]["mean"])

    def test_decode_tps_computed_from_tokens_and_time(self):
        records = [
            _make_record(
                actual_output_tokens=20,
                decode_ms=2000.0,
                decode_tokens_per_sec=None,
            )
        ]
        agg = aggregate_condition(records)
        self.assertTrue(agg["stage_timings_available"]["decode_tokens_per_sec"])
        self.assertAlmostEqual(agg["decode_tokens_per_sec"]["mean"], 10.0, places=1)


# ── token scaling table tests ─────────────────────────────────────────────────


class TestComputeTokenScaling(unittest.TestCase):
    def test_one_row_per_condition_path(self):
        by_cp = {
            ("A", "direct"): [_make_record(condition="A", path="direct", total_latency_ms=500.0)],
            ("B", "direct"): [_make_record(condition="B", path="direct", max_output_tokens=32, total_latency_ms=700.0)],
        }
        rows = compute_token_scaling(by_cp)
        self.assertEqual(len(rows), 2)
        conditions = {r["condition"] for r in rows}
        self.assertEqual(conditions, {"A", "B"})

    def test_warmup_excluded_from_scaling(self):
        by_cp = {
            ("A", "direct"): [
                _make_record(total_latency_ms=9999.0, warmup=True),
                _make_record(total_latency_ms=500.0, warmup=False),
            ],
        }
        rows = compute_token_scaling(by_cp)
        self.assertEqual(rows[0]["n_measured"], 1)
        self.assertAlmostEqual(rows[0]["total_latency_ms_mean"], 500.0, places=1)


# ── direct-vs-ROS comparison tests ───────────────────────────────────────────


class TestComputeDirectRosComparison(unittest.TestCase):
    def test_overhead_computed_correctly(self):
        by_cp = {
            ("D", "direct"): [_make_record(condition="D", path="direct", total_latency_ms=800.0)],
            ("D", "ros"): [_make_record(condition="D", path="ros", total_latency_ms=1000.0)],
        }
        rows = compute_direct_ros_comparison(by_cp)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["condition"], "D")
        self.assertAlmostEqual(row["ros_overhead_ms_mean"], 200.0, places=1)
        self.assertAlmostEqual(row["ros_overhead_pct"], 25.0, places=1)

    def test_condition_with_only_direct_excluded(self):
        by_cp = {
            ("A", "direct"): [_make_record(condition="A", path="direct")],
        }
        rows = compute_direct_ros_comparison(by_cp)
        self.assertEqual(rows, [])

    def test_condition_with_only_ros_excluded(self):
        by_cp = {
            ("B", "ros"): [_make_record(condition="B", path="ros")],
        }
        rows = compute_direct_ros_comparison(by_cp)
        self.assertEqual(rows, [])

    def test_multiple_conditions(self):
        by_cp = {
            ("A", "direct"): [_make_record(condition="A", path="direct", total_latency_ms=400.0)],
            ("A", "ros"): [_make_record(condition="A", path="ros", total_latency_ms=500.0)],
            ("E", "direct"): [_make_record(condition="E", path="direct", total_latency_ms=7000.0)],
            ("E", "ros"): [_make_record(condition="E", path="ros", total_latency_ms=8500.0)],
        }
        rows = compute_direct_ros_comparison(by_cp)
        self.assertEqual(len(rows), 2)
        conds = {r["condition"] for r in rows}
        self.assertEqual(conds, {"A", "E"})


# ── full report building tests ────────────────────────────────────────────────


class TestBuildReport(unittest.TestCase):
    def _make_full_records(self) -> list[dict[str, Any]]:
        recs: list[dict[str, Any]] = []
        for cond, pid, mtok in [("A", "terse_id", 16), ("D", "scene_description", 128)]:
            for path in ("direct", "ros"):
                recs.append(
                    _make_record(
                        condition=cond,
                        path=path,
                        prompt_id=pid,
                        max_output_tokens=mtok,
                        total_latency_ms=500.0 if path == "direct" else 650.0,
                    )
                )
        return recs

    def test_report_structure(self):
        records = self._make_full_records()
        report = build_report(records)
        self.assertIn("schema_version", report)
        self.assertIn("generated_at", report)
        self.assertIn("conditions", report)
        self.assertIn("token_scaling_table", report)
        self.assertIn("direct_ros_comparison", report)
        self.assertIn("raw_records", report)

    def test_raw_records_preserved(self):
        records = self._make_full_records()
        report = build_report(records)
        self.assertEqual(len(report["raw_records"]), len(records))

    def test_correct_measured_count(self):
        records = self._make_full_records() + [
            _make_record(warmup=True, total_latency_ms=9999.0)
        ]
        report = build_report(records)
        self.assertEqual(report["n_measured_records"], len(self._make_full_records()))

    def test_empty_input(self):
        report = build_report([])
        self.assertEqual(report["n_total_records"], 0)
        self.assertEqual(report["raw_records"], [])

    def test_run_id_and_model_collected(self):
        records = [_make_record(model_name="TestVLM-8B")]
        report = build_report(records)
        self.assertIn(_RUN_ID, report["run_ids"])
        self.assertIn("TestVLM-8B", report["model_names"])


# ── text report formatting tests ──────────────────────────────────────────────


class TestFormatTextReport(unittest.TestCase):
    def _make_report(self) -> dict[str, Any]:
        records = [
            _make_record(condition="A", path="direct", total_latency_ms=480.0),
            _make_record(condition="A", path="ros", total_latency_ms=600.0),
            _make_record(condition="E", path="direct", max_output_tokens=256, total_latency_ms=8000.0),
            _make_record(condition="E", path="ros", max_output_tokens=256, total_latency_ms=9500.0),
        ]
        return build_report(records)

    def test_report_contains_key_sections(self):
        text = format_text_report(self._make_report())
        self.assertIn("VLM Latency Characterization Benchmark Report", text)
        self.assertIn("Latency vs Output-Token Cap", text)
        self.assertIn("Direct (native) vs ROS Path Overhead", text)

    def test_report_mentions_unavailable_stage_timings(self):
        records = [_make_record(ttft_ms=None, decode_ms=None)]
        report = build_report(records)
        text = format_text_report(report)
        self.assertIn("null, not inferred", text)

    def test_report_does_not_contain_fabricated_values(self):
        # Null stage timings must appear as n/a, not as numeric values.
        records = [_make_record(ttft_ms=None, decode_ms=None, visual_preprocess_ms=None)]
        report = build_report(records)
        text = format_text_report(report)
        # The text should not contain a raw float for an unavailable TTFT.
        self.assertNotIn("ttft_ms: 0", text)

    def test_report_with_stage_timings_present(self):
        records = [_make_record(ttft_ms=55.0, decode_ms=400.0, visual_preprocess_ms=30.0)]
        report = build_report(records)
        text = format_text_report(report)
        self.assertIn("VLM Latency Characterization Benchmark Report", text)

    def test_report_with_no_ros_path(self):
        records = [_make_record(path="direct")]
        report = build_report(records)
        text = format_text_report(report)
        # No direct-vs-ROS section when ROS data absent
        self.assertNotIn("Direct (native) vs ROS Path Overhead", text)


# ── experiment matrix tests ───────────────────────────────────────────────────


class TestExperimentMatrix(unittest.TestCase):
    def test_all_five_conditions_defined(self):
        conditions = [c for c, _, _ in EXPERIMENT_MATRIX]
        self.assertEqual(sorted(conditions), ["A", "B", "C", "D", "E"])

    def test_max_tokens_match_spec(self):
        expected = {"A": 16, "B": 32, "C": 64, "D": 128, "E": 256}
        for cond, _, max_tokens in EXPERIMENT_MATRIX:
            self.assertEqual(max_tokens, expected[cond], f"Condition {cond}")

    def test_prompt_texts_non_empty(self):
        for prompt_id in ("terse_id", "compact_odd_json", "scene_description"):
            self.assertIn(prompt_id, PROMPT_TEXTS)
            self.assertTrue(PROMPT_TEXTS[prompt_id].strip())

    def test_build_condition_spec_returns_correct_data(self):
        spec = build_condition_spec("A")
        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertEqual(spec["condition"], "A")
        self.assertEqual(spec["max_output_tokens"], 16)
        self.assertEqual(spec["prompt_id"], "terse_id")
        self.assertIn("prompt_text", spec)
        self.assertIn("prompt_hash", spec)

    def test_build_condition_spec_unknown_returns_none(self):
        self.assertIsNone(build_condition_spec("Z"))

    def test_prompt_hash_deterministic(self):
        h1 = prompt_hash("What is in this image?")
        h2 = prompt_hash("What is in this image?")
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 12)

    def test_prompt_hash_differs_for_different_texts(self):
        h1 = prompt_hash("What is in this image?")
        h2 = prompt_hash("Describe the scene.")
        self.assertNotEqual(h1, h2)


# ── JSON schema validity test ─────────────────────────────────────────────────


class TestSchemaValidity(unittest.TestCase):
    def test_vlm_latency_record_schema_is_valid_json(self):
        schema_path = (
            _BENCH_DIR / "schemas" / "vlm_latency_record.schema.json"
        )
        self.assertTrue(schema_path.exists(), f"Schema file missing: {schema_path}")
        with schema_path.open("r", encoding="utf-8") as fh:
            schema = json.load(fh)
        self.assertEqual(schema.get("$id"), "vlm_latency_record.schema.json")
        self.assertEqual(schema.get("type"), "object")
        required = schema.get("required", [])
        for field in (
            "schema_version",
            "record_type",
            "run_id",
            "condition",
            "path",
            "image_id",
            "max_output_tokens",
            "success",
            "total_latency_ms",
        ):
            self.assertIn(field, required, f"Required field {field!r} missing from schema")

    def test_schema_requires_null_not_absent_for_optional_timings(self):
        schema_path = _BENCH_DIR / "schemas" / "vlm_latency_record.schema.json"
        with schema_path.open("r", encoding="utf-8") as fh:
            schema = json.load(fh)
        props = schema.get("properties", {})
        for field in ("ttft_ms", "decode_ms", "visual_preprocess_ms"):
            self.assertIn(field, props, f"Property {field!r} missing from schema")
            type_val = props[field].get("type", [])
            self.assertIn("null", type_val, f"{field!r} must allow null")


# ── shell script syntax test ──────────────────────────────────────────────────


class TestShellScriptSyntax(unittest.TestCase):
    def test_run_vlm_latency_benchmark_syntax(self):
        script = _BENCH_DIR / "run_vlm_latency_benchmark.sh"
        result = subprocess.run(
            ["bash", "-n", str(script)],
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"bash -n failed:\n{result.stderr}",
        )


# ── round-trip test (build records → JSONL → parse → report) ─────────────────


class TestRoundTrip(unittest.TestCase):
    def test_write_read_build_report(self):
        records = [
            _make_record(condition=c, path=p, total_latency_ms=lat)
            for c, p, lat in [
                ("A", "direct", 500.0),
                ("A", "ros", 650.0),
                ("E", "direct", 7800.0),
                ("E", "ros", 9200.0),
            ]
        ]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            f.write(_jsonl_from(records))
            path = Path(f.name)
        try:
            parsed = parse_jsonl(path)
            self.assertEqual(len(parsed), len(records))
            report = build_report(parsed, source_path=path)
            self.assertEqual(report["n_total_records"], len(records))
            self.assertGreater(len(report["token_scaling_table"]), 0)
            # Both direct-vs-ROS conditions with paired runs should appear.
            comparison_conditions = {r["condition"] for r in report["direct_ros_comparison"]}
            self.assertIn("A", comparison_conditions)
            self.assertIn("E", comparison_conditions)
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
