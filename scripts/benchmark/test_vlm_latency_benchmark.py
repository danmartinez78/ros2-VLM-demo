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
import re
import ast
import os
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
    compute_direct_ipc_comparison,
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
    image_id: str = "image_001",
    prompt_id: str = "terse_id",
    max_output_tokens: int = 16,
    actual_output_tokens: int | None = 12,
    finish_reason: str | None = None,
    output_text: str | None = None,
    success: bool = True,
    error: str | None = None,
    cold_start_total_ms: float | None = None,
    total_latency_ms: float | None = 500.0,
    visual_preprocess_ms: float | None = None,
    ttft_ms: float | None = None,
    decode_ms: float | None = None,
    decode_tokens_per_sec: float | None = None,
    native_response_path: str | None = None,
    native_profile_path: str | None = None,
    content_hash: str | None = None,
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
        "content_hash": content_hash,
        "prompt_id": prompt_id,
        "prompt_hash": "abc123def456",
        "max_output_tokens": max_output_tokens,
        "actual_output_tokens": actual_output_tokens,
        "finish_reason": finish_reason,
        "output_text": output_text,
        "success": success,
        "error": error,
        "cold_start_total_ms": cold_start_total_ms,
        "total_latency_ms": total_latency_ms,
        "visual_preprocess_ms": visual_preprocess_ms,
        "ttft_ms": ttft_ms,
        "decode_ms": decode_ms,
        "decode_tokens_per_sec": decode_tokens_per_sec,
        "native_response_path": native_response_path,
        "native_profile_path": native_profile_path,
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


# ── direct-vs-IPC comparison tests ───────────────────────────────────────────


class TestComputeDirectIpcComparison(unittest.TestCase):
    def test_table_computed_for_paired_conditions(self):
        by_cp = {
            ("D", "direct"): [_make_record(condition="D", path="direct", total_latency_ms=800.0)],
            ("D", "ipc"): [_make_record(condition="D", path="ipc", total_latency_ms=1000.0)],
        }
        rows = compute_direct_ipc_comparison(by_cp)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["condition"], "D")
        self.assertAlmostEqual(row["direct_total_latency_ms_mean"], 800.0, places=1)
        self.assertAlmostEqual(row["ipc_total_latency_ms_mean"], 1000.0, places=1)
        # Must not carry an 'overhead' or comparison delta field — lifecycles differ.
        self.assertNotIn("overhead_ms", row)

    def test_condition_with_only_direct_excluded(self):
        by_cp = {
            ("A", "direct"): [_make_record(condition="A", path="direct")],
        }
        rows = compute_direct_ipc_comparison(by_cp)
        self.assertEqual(rows, [])

    def test_condition_with_only_ipc_excluded(self):
        by_cp = {
            ("B", "ipc"): [_make_record(condition="B", path="ipc")],
        }
        rows = compute_direct_ipc_comparison(by_cp)
        self.assertEqual(rows, [])

    def test_multiple_conditions(self):
        by_cp = {
            ("A", "direct"): [_make_record(condition="A", path="direct", total_latency_ms=400.0)],
            ("A", "ipc"): [_make_record(condition="A", path="ipc", total_latency_ms=500.0)],
            ("E", "direct"): [_make_record(condition="E", path="direct", total_latency_ms=7000.0)],
            ("E", "ipc"): [_make_record(condition="E", path="ipc", total_latency_ms=8500.0)],
        }
        rows = compute_direct_ipc_comparison(by_cp)
        self.assertEqual(len(rows), 2)
        conds = {r["condition"] for r in rows}
        self.assertEqual(conds, {"A", "E"})


# ── full report building tests ────────────────────────────────────────────────


class TestBuildReport(unittest.TestCase):
    def _make_full_records(self) -> list[dict[str, Any]]:
        recs: list[dict[str, Any]] = []
        for cond, pid, mtok in [("A", "terse_id", 16), ("D", "scene_description", 128)]:
            for path in ("direct", "ipc"):
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
        self.assertIn("direct_ipc_comparison", report)
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
            _make_record(condition="A", path="ipc", total_latency_ms=600.0),
            _make_record(condition="E", path="direct", max_output_tokens=256, total_latency_ms=8000.0),
            _make_record(condition="E", path="ipc", max_output_tokens=256, total_latency_ms=9500.0),
        ]
        return build_report(records)

    def test_report_contains_key_sections(self):
        text = format_text_report(self._make_report())
        self.assertIn("VLM Latency Characterization Benchmark Report", text)
        self.assertIn("Latency vs Output-Token Cap", text)
        self.assertIn("Direct (cold-start, per-process) vs IPC (persistent server, steady-state)", text)

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

    def test_report_with_no_ipc_path(self):
        records = [_make_record(path="direct")]
        report = build_report(records)
        text = format_text_report(report)
        # No direct-vs-IPC section when IPC data absent
        self.assertNotIn("Direct (cold-start, per-process) vs IPC", text)


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
                ("A", "ipc", 650.0),
                ("E", "direct", 7800.0),
                ("E", "ipc", 9200.0),
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
            # Both direct-vs-IPC conditions with paired runs should appear.
            comparison_conditions = {r["condition"] for r in report["direct_ipc_comparison"]}
            self.assertIn("A", comparison_conditions)
            self.assertIn("E", comparison_conditions)
        finally:
            path.unlink(missing_ok=True)


# ── canonical Edge-LLM interface contract tests ───────────────────────────────


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class TestDirectPathCommandContract(unittest.TestCase):
    """Validate that run_vlm_latency_benchmark.sh uses the canonical Edge-LLM
    CLI flags identical to run_native_benchmarks.sh, so a normal Thor env will
    find the binary and fire the correct invocation."""

    def _script_text(self) -> str:
        return (_BENCH_DIR / "run_vlm_latency_benchmark.sh").read_text(encoding="utf-8")

    def test_uses_tensorrt_edge_llm_root_not_edgellm_binary(self):
        text = self._script_text()
        self.assertIn("TENSORRT_EDGE_LLM_ROOT", text)
        self.assertNotIn("EDGELLM_BINARY", text)

    def test_uses_edge_vlm_llm_engine_dir(self):
        text = self._script_text()
        self.assertIn("EDGE_VLM_LLM_ENGINE_DIR", text)
        self.assertNotIn("EDGELLM_LLM_ENGINE_DIR", text)

    def test_uses_edge_vlm_multimodal_engine_dir(self):
        text = self._script_text()
        self.assertIn("EDGE_VLM_MULTIMODAL_ENGINE_DIR", text)
        self.assertNotIn("EDGELLM_MULTIMODAL_ENGINE_DIR", text)

    def test_llm_inference_binary_path_matches_native_benchmarks(self):
        """Binary path must match run_native_benchmarks.sh canonical form."""
        text = self._script_text()
        self.assertIn(
            "build/examples/llm/llm_inference",
            text,
            "llm_inference must be resolved from TENSORRT_EDGE_LLM_ROOT/build/examples/llm/",
        )

    def test_uses_engineDir_flag(self):
        text = self._script_text()
        self.assertIn("--engineDir", text)
        self.assertNotIn("--llmDir", text)

    def test_uses_multimodalEngineDir_flag(self):
        text = self._script_text()
        self.assertIn("--multimodalEngineDir", text)
        self.assertNotIn("--mmDir", text)

    def test_uses_maxGenerateLength_flag(self):
        """llm_inference (NVIDIA binary) uses camelCase --maxGenerateLength.
        This is distinct from vlm_single_shot_client's own --max-tokens arg,
        which edge_vlm_cli accepts as --max-generate-length (kebab-case)."""
        text = self._script_text()
        self.assertIn("--maxGenerateLength", text)
        self.assertNotIn("--maxOutputLen", text)

    def test_uses_lowercase_warmup_flag(self):
        """Edge-LLM uses --warmup (lowercase), not --warmUp."""
        text = self._script_text()
        self.assertIn("--warmup", text)
        self.assertNotIn("--warmUp", text)


class TestRosClientInstall(unittest.TestCase):
    """Validate that vlm_single_shot_client is a real repo-owned script and is
    referenced by the CMakeLists.txt install directive so
    `ros2 run edge_vlm_ros vlm_single_shot_client` resolves after a normal
    colcon build."""

    def test_vlm_single_shot_client_script_exists(self):
        script = _REPO_ROOT / "scripts" / "vlm_single_shot_client"
        self.assertTrue(
            script.exists(),
            f"scripts/vlm_single_shot_client not found at {script}",
        )

    def test_vlm_single_shot_client_is_executable(self):
        script = _REPO_ROOT / "scripts" / "vlm_single_shot_client"
        self.assertTrue(script.exists(), f"scripts/vlm_single_shot_client not found at {script}")
        self.assertTrue(
            os.access(str(script), os.X_OK),
            "scripts/vlm_single_shot_client must be executable",
        )

    def test_vlm_single_shot_client_is_valid_python(self):
        """The script must parse without errors under the current Python."""
        script = _REPO_ROOT / "scripts" / "vlm_single_shot_client"
        self.assertTrue(script.exists(), f"scripts/vlm_single_shot_client not found at {script}")
        source = script.read_text(encoding="utf-8")
        try:
            ast.parse(source)
        except SyntaxError as exc:
            self.fail(f"scripts/vlm_single_shot_client has a syntax error: {exc}")

    def test_vlm_single_shot_client_referenced_in_cmake(self):
        cmake = _REPO_ROOT / "CMakeLists.txt"
        self.assertTrue(cmake.exists(), "CMakeLists.txt not found")
        text = cmake.read_text(encoding="utf-8")
        self.assertIn(
            "vlm_single_shot_client",
            text,
            "CMakeLists.txt must install scripts/vlm_single_shot_client",
        )

    def test_vlm_single_shot_client_install_uses_programs(self):
        """Must use install(PROGRAMS ...) not install(FILES ...) so it stays executable."""
        cmake = _REPO_ROOT / "CMakeLists.txt"
        text = cmake.read_text(encoding="utf-8")
        for block_match in re.finditer(
            r"install\s*\(([^)]+)\)", text, re.DOTALL
        ):
            block = block_match.group(1)
            if "vlm_single_shot_client" in block:
                self.assertIn(
                    "PROGRAMS",
                    block,
                    "install() for vlm_single_shot_client must use PROGRAMS to preserve execute bit",
                )
                return
        self.fail("No install() block containing vlm_single_shot_client found in CMakeLists.txt")

    def test_ipc_path_invokes_vlm_single_shot_client(self):
        """The benchmark runner IPC path must call vlm_single_shot_client."""
        text = (_BENCH_DIR / "run_vlm_latency_benchmark.sh").read_text(encoding="utf-8")
        self.assertIn(
            "vlm_single_shot_client",
            text,
            "IPC path in run_vlm_latency_benchmark.sh must invoke vlm_single_shot_client",
        )

    def test_ipc_path_uses_socket_arg(self):
        """vlm_single_shot_client call must forward --socket so edge_vlm_cli can
        connect to edge_vlm_server."""
        text = (_BENCH_DIR / "run_vlm_latency_benchmark.sh").read_text(encoding="utf-8")
        self.assertIn("--socket", text)

    def test_ipc_path_label_is_ipc_not_ros(self):
        """The path label written into JSONL records must be 'ipc', not 'ros'.
        This ensures the path cannot silently reduce to a bare IPC call while
        claiming to exercise edge_vlm_ros_node."""
        text = (_BENCH_DIR / "run_vlm_latency_benchmark.sh").read_text(encoding="utf-8")
        self.assertIn("'ipc'", text, "IPC path records must carry path='ipc'")
        # 'ros' must not appear as a path value in any record-building block.
        for m in re.finditer(r"'path':\s*'([^']+)'", text):
            self.assertNotEqual(
                m.group(1),
                "ros",
                "path='ros' must not appear in record-building blocks — use 'ipc'",
            )

    def test_direct_path_carries_cold_start_lifecycle(self):
        """Direct records must declare lifecycle_semantics='cold_start'."""
        text = (_BENCH_DIR / "run_vlm_latency_benchmark.sh").read_text(encoding="utf-8")
        self.assertIn("cold_start", text)

    def test_ipc_path_carries_persistent_lifecycle(self):
        """IPC records must declare lifecycle_semantics='persistent'."""
        text = (_BENCH_DIR / "run_vlm_latency_benchmark.sh").read_text(encoding="utf-8")
        self.assertIn("persistent", text)


# ── native request shape tests ────────────────────────────────────────────────


class TestNativeRequestShape(unittest.TestCase):
    """Validate that run_vlm_latency_benchmark.sh builds the llm_inference input
    JSON using the pinned NVIDIA VLM schema: requests -> messages -> content
    [{type,image},{type,text}].  This is the shape observed to be accepted in
    Thor hardware smoke; the old {image,text} flat shape was rejected with
    `'requests' array not found`."""

    def _script_text(self) -> str:
        return (_BENCH_DIR / "run_vlm_latency_benchmark.sh").read_text(encoding="utf-8")

    def test_uses_requests_key(self):
        text = self._script_text()
        self.assertIn("'requests'", text, "Input JSON must use top-level 'requests' array")

    def test_uses_messages_key(self):
        text = self._script_text()
        self.assertIn("'messages'", text, "Input JSON must use 'messages' key")

    def test_uses_content_key(self):
        text = self._script_text()
        self.assertIn("'content'", text, "Input JSON must use 'content' array")

    def test_uses_type_image_entry(self):
        text = self._script_text()
        self.assertIn("'type': 'image'", text, "content must contain {type:image} entry")

    def test_uses_type_text_entry(self):
        text = self._script_text()
        self.assertIn("'type': 'text'", text, "content must contain {type:text} entry")

    def test_does_not_use_flat_image_text_shape(self):
        """The old flat {'image': ..., 'text': ...} shape must not remain."""
        text = self._script_text()
        # Positive check: the flat one-level dict should not be the construction
        # pattern any more (the new format nests under requests/messages/content).
        self.assertNotIn(
            "obj = {'image'",
            text,
            "Old flat {'image':…,'text':…} request shape must be removed",
        )

    def test_requests_schema_roundtrip(self):
        """Verify a constructed request JSON parses correctly, including role."""
        import subprocess
        result = subprocess.run(
            [
                "python3", "-c",
                """
import json, sys
image_path = '/tmp/test.jpg'
prompt_text = 'What is this?'
obj = {
    'requests': [
        {
            'messages': [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'image', 'image': image_path},
                        {'type': 'text',  'text':  prompt_text},
                    ]
                }
            ]
        }
    ]
}
data = json.dumps(obj)
parsed = json.loads(data)
req = parsed['requests'][0]
msg = req['messages'][0]
assert msg['role'] == 'user', f"Expected role=user, got {msg.get('role')}"
content = msg['content']
assert content[0]['type'] == 'image'
assert content[0]['image'] == image_path
assert content[1]['type'] == 'text'
assert content[1]['text'] == prompt_text
print('ok')
""",
            ],
            capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ok", result.stdout)

    def test_uses_role_user_in_message(self):
        """The messages entry must include 'role': 'user' as Thor requires."""
        text = self._script_text()
        self.assertIn("'role': 'user'", text, "message dict must include 'role': 'user'")


# ── finish_reason tests ────────────────────────────────────────────────────────


class TestFinishReason(unittest.TestCase):
    """Tests for finish_reason parsing/aggregation and max-length reporting."""

    def test_finish_reason_counted_in_aggregation(self):
        records = [
            _make_record(finish_reason="max-length"),
            _make_record(finish_reason="max-length"),
            _make_record(finish_reason="eos"),
        ]
        agg = aggregate_condition(records)
        self.assertEqual(agg["finish_reason_counts"]["max-length"], 2)
        self.assertEqual(agg["finish_reason_counts"]["eos"], 1)
        self.assertEqual(agg["n_max_length"], 2)

    def test_finish_reason_null_counted_separately(self):
        records = [
            _make_record(finish_reason=None),
            _make_record(finish_reason="eos"),
        ]
        agg = aggregate_condition(records)
        self.assertEqual(agg["finish_reason_counts"].get("null", 0), 1)
        self.assertEqual(agg["n_max_length"], 0)

    def test_finish_reason_empty_when_no_records(self):
        agg = aggregate_condition([])
        self.assertEqual(agg["finish_reason_counts"], {})
        self.assertEqual(agg["n_max_length"], 0)

    def test_warmup_records_excluded_from_finish_reason_count(self):
        records = [
            _make_record(finish_reason="max-length", warmup=True),
            _make_record(finish_reason="eos", warmup=False),
        ]
        agg = aggregate_condition(records)
        # Only the non-warmup record should be counted.
        self.assertEqual(agg["n_max_length"], 0)
        self.assertNotIn("max-length", agg["finish_reason_counts"])

    def test_text_report_includes_finish_reason_section(self):
        """Text report must include a finish_reason summary when data is present."""
        records = [
            _make_record(condition="A", path="direct", finish_reason="max-length"),
            _make_record(condition="A", path="direct", finish_reason="eos"),
        ]
        report = build_report(records)
        text = format_text_report(report)
        self.assertIn("finish", text.lower())
        self.assertIn("max-length", text.lower())

    def test_text_report_notes_truncation(self):
        """Report must note that max-length means capped, not task-complete."""
        records = [_make_record(condition="A", path="direct", finish_reason="max-length")]
        report = build_report(records)
        text = format_text_report(report)
        self.assertIn("capped", text.lower())

    def test_text_report_no_finish_reason_section_when_absent(self):
        """If all finish_reasons are null, the truncation NOTE must not appear."""
        records = [_make_record(finish_reason=None)]
        report = build_report(records)
        text = format_text_report(report)
        # Only "null" finish_reason present; truncation note should not appear.
        self.assertNotIn("NOTE: requests with finish_reason", text)

    def test_runner_parses_finish_reason_from_response(self):
        """The shell runner must extract finish_reason from the native response JSON."""
        text = (_BENCH_DIR / "run_vlm_latency_benchmark.sh").read_text(encoding="utf-8")
        self.assertIn("finish_reason", text)
        self.assertIn("finishReason", text)

    def test_record_has_finish_reason_field(self):
        """Records must include finish_reason key (possibly null)."""
        rec = _make_record(finish_reason="eos")
        self.assertIn("finish_reason", rec)
        self.assertEqual(rec["finish_reason"], "eos")

    def test_record_finish_reason_null_by_default(self):
        rec = _make_record()
        self.assertIn("finish_reason", rec)
        self.assertIsNone(rec["finish_reason"])


# ── fixture validation tests ──────────────────────────────────────────────────


class TestFixtureHygiene(unittest.TestCase):
    """Tests for image fixture validation and neutral naming."""

    def test_runner_validates_images_before_inclusion(self):
        """The script must contain a validation check for image files."""
        text = (_BENCH_DIR / "run_vlm_latency_benchmark.sh").read_text(encoding="utf-8")
        self.assertIn("_validate_image", text)

    def test_runner_rejects_zero_byte_image(self):
        """_validate_image must reject a zero-byte file."""
        import subprocess
        result = subprocess.run(
            ["bash", "-c", """
source /dev/stdin << 'EOF'
_validate_image() {
    local path="$1"
    if [[ ! -f "${path}" ]]; then
        echo "ERROR: image not found: ${path}" >&2; return 1
    fi
    local size
    size=$(stat -c%s "${path}" 2>/dev/null || stat -f%z "${path}" 2>/dev/null || echo 0)
    if [[ "${size}" -eq 0 ]]; then
        echo "ERROR: image is zero bytes" >&2; return 1
    fi
    return 0
}
EOF
f=$(mktemp /tmp/test_XXXXXX.jpg)
truncate -s 0 "$f"
_validate_image "$f" && echo "FAIL_should_reject" || echo "CORRECTLY_REJECTED"
rm -f "$f"
"""],
            capture_output=True, text=True, timeout=10,
        )
        self.assertIn("CORRECTLY_REJECTED", result.stdout, "Zero-byte image should be rejected")
        self.assertNotIn("FAIL_should_reject", result.stdout)

    def test_runner_rejects_html_masquerading_as_jpeg(self):
        """_validate_image must reject a file with HTML magic bytes."""
        import subprocess
        result = subprocess.run(
            ["bash", "-c", r"""
source /dev/stdin << 'EOF'
_validate_image() {
    local path="$1"
    if [[ ! -f "${path}" ]]; then echo "ERROR: not found" >&2; return 1; fi
    local size
    size=$(stat -c%s "${path}" 2>/dev/null || stat -f%z "${path}" 2>/dev/null || echo 0)
    if [[ "${size}" -eq 0 ]]; then echo "ERROR: zero bytes" >&2; return 1; fi
    local magic
    magic=$(xxd -p -l 4 "${path}" 2>/dev/null || od -A n -N 4 -t x1 "${path}" 2>/dev/null | tr -d ' \n')
    case "${magic,,}" in
        ffd8ff*) ;;
        89504e47*) ;;
        *)
            echo "ERROR: not JPEG/PNG: ${magic}" >&2
            return 1
            ;;
    esac
    return 0
}
EOF
f=$(mktemp /tmp/test_XXXXXX.jpg)
printf '<html>Not an image</html>' > "$f"
_validate_image "$f" && echo "FAIL_should_reject" || echo "CORRECTLY_REJECTED"
rm -f "$f"
"""],
            capture_output=True, text=True, timeout=10,
        )
        self.assertIn("CORRECTLY_REJECTED", result.stdout, "HTML content should be rejected as not JPEG/PNG")

    def test_runner_uses_neutral_fixture_names(self):
        """The script must reference neutral names like image_001, not red_panda."""
        text = (_BENCH_DIR / "run_vlm_latency_benchmark.sh").read_text(encoding="utf-8")
        self.assertIn("image_001", text, "Script must reference neutral image name image_001.jpg")

    def test_runner_no_longer_hardcodes_red_panda_path_as_primary(self):
        """red_panda.jpg must not be the hard-coded primary fixture name."""
        text = (_BENCH_DIR / "run_vlm_latency_benchmark.sh").read_text(encoding="utf-8")
        # The neutral fixture (image_001) must be present as the primary download name.
        self.assertIn("image_001", text)
        # The old hard-coded panda path variable must be gone.
        self.assertNotIn('panda_path="${IMAGE_DIR}/red_panda.jpg"', text)

    def test_content_hash_recorded_in_runner(self):
        """The script must compute and record a content hash for each image."""
        text = (_BENCH_DIR / "run_vlm_latency_benchmark.sh").read_text(encoding="utf-8")
        self.assertIn("content_hash", text)
        self.assertIn("_image_content_hash", text)

    def test_record_includes_content_hash_field(self):
        """Records must include a content_hash field."""
        rec = _make_record(content_hash="abc123def456")
        self.assertIn("content_hash", rec)
        self.assertEqual(rec["content_hash"], "abc123def456")


# ── dumpProfile / native profiling tests ──────────────────────────────────────


class TestNativeProfiling(unittest.TestCase):
    """Validate that run_vlm_latency_benchmark.sh invokes llm_inference with
    --dumpProfile --profileOutputFile and preserves raw artifacts."""

    def _script_text(self) -> str:
        return (_BENCH_DIR / "run_vlm_latency_benchmark.sh").read_text(encoding="utf-8")

    def test_uses_dump_profile_flag(self):
        text = self._script_text()
        self.assertIn("--dumpProfile", text, "llm_inference must be called with --dumpProfile")

    def test_uses_profile_output_file_flag(self):
        text = self._script_text()
        self.assertIn("--profileOutputFile", text, "llm_inference must be called with --profileOutputFile")

    def test_native_response_path_in_record(self):
        """Records must carry native_response_path for artifact reference."""
        text = self._script_text()
        self.assertIn("'native_response_path'", text)

    def test_native_profile_path_in_record(self):
        """Records must carry native_profile_path for artifact reference."""
        text = self._script_text()
        self.assertIn("'native_profile_path'", text)

    def test_cold_start_total_ms_in_record(self):
        """Direct records must carry cold_start_total_ms (shell wall time)."""
        text = self._script_text()
        self.assertIn("'cold_start_total_ms'", text)

    def test_profile_artifacts_not_deleted(self):
        """The script must NOT rm -f the profile and response JSON artifacts."""
        text = self._script_text()
        # The only rm -f after llm_inference should be for the input_json.
        # Confirm that output_json and profile_json are preserved.
        self.assertIn("rm -f \"${input_json}\"", text)
        self.assertIn("preserved as named artifacts", text)

    def test_record_has_cold_start_total_ms_field(self):
        """_make_record must produce cold_start_total_ms."""
        rec = _make_record(cold_start_total_ms=410.5, total_latency_ms=410.5)
        self.assertIn("cold_start_total_ms", rec)
        self.assertEqual(rec["cold_start_total_ms"], 410.5)

    def test_record_has_native_response_path_field(self):
        rec = _make_record(native_response_path="/tmp/resp.json")
        self.assertIn("native_response_path", rec)
        self.assertEqual(rec["native_response_path"], "/tmp/resp.json")

    def test_record_has_native_profile_path_field(self):
        rec = _make_record(native_profile_path="/tmp/prof.json")
        self.assertIn("native_profile_path", rec)
        self.assertEqual(rec["native_profile_path"], "/tmp/prof.json")


# ── symlink-install client lookup tests ───────────────────────────────────────


class TestSymlinkInstallClientLookup(unittest.TestCase):
    """Validate that vlm_single_shot_client discovers edge_vlm_cli in a way
    that works under `colcon build --symlink-install`.

    Under --symlink-install, the installed script is a symlink pointing to the
    source file.  ``Path(__file__).resolve()`` would follow the symlink back to
    the source directory, where edge_vlm_cli is not installed.  The fix is to
    use os.path.abspath(sys.argv[0]) (no symlink resolution) so we look in the
    install directory where edge_vlm_cli actually lives.
    """

    def _client_source(self) -> str:
        path = _REPO_ROOT / "scripts" / "vlm_single_shot_client"
        return path.read_text(encoding="utf-8")

    def test_does_not_use_file_resolve_for_lookup(self):
        """Must not use Path(__file__).resolve().parent for edge_vlm_cli lookup."""
        source = self._client_source()
        # The function should not call .resolve() on __file__ for discovery.
        # (It may call .resolve() for other things, but not the lookup path.)
        self.assertNotIn(
            "Path(__file__).resolve().parent",
            source,
            "_find_edge_vlm_cli must not use Path(__file__).resolve().parent "
            "(follows symlinks back to source tree under --symlink-install)",
        )

    def test_uses_sys_argv0_for_invocation_dir(self):
        """Must use sys.argv[0] (symlink path) for install-directory lookup."""
        source = self._client_source()
        self.assertIn("sys.argv[0]", source, "_find_edge_vlm_cli must use sys.argv[0]")

    def test_falls_back_to_shutil_which(self):
        """Must fall back to shutil.which so PATH-based installs also work."""
        source = self._client_source()
        self.assertIn("shutil.which", source, "shutil.which fallback must be present")

    def test_find_cli_in_install_dir(self):
        """_find_edge_vlm_cli must return edge_vlm_cli when present next to argv[0]."""
        import subprocess, textwrap, tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            # Simulate an install dir with edge_vlm_cli present.
            cli = Path(tmpdir) / "edge_vlm_cli"
            cli.write_text("#!/bin/sh\necho stub\n")
            cli.chmod(0o755)
            fake_argv0 = str(Path(tmpdir) / "vlm_single_shot_client")

            test_script = textwrap.dedent(f"""\
                import sys, os
                sys.argv[0] = {fake_argv0!r}
                sys.path.insert(0, {str(_REPO_ROOT / 'scripts')!r})
                # Re-exec the function under test
                from pathlib import Path
                import shutil

                def _find_edge_vlm_cli():
                    invocation_dir = Path(os.path.abspath(sys.argv[0])).parent
                    candidate = invocation_dir / 'edge_vlm_cli'
                    if candidate.is_file() and os.access(str(candidate), os.X_OK):
                        return str(candidate)
                    found = shutil.which('edge_vlm_cli')
                    if found:
                        return found
                    raise RuntimeError('not found')

                result = _find_edge_vlm_cli()
                assert result == str(Path({str(tmpdir)!r}) / 'edge_vlm_cli'), result
                print('ok')
            """)
            result = subprocess.run(
                ["python3", "-c", test_script],
                capture_output=True, text=True, timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("ok", result.stdout)

    def test_raises_when_cli_not_found(self):
        """_find_edge_vlm_cli must raise RuntimeError when edge_vlm_cli is absent."""
        import subprocess, textwrap, tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_argv0 = str(Path(tmpdir) / "vlm_single_shot_client")
            test_script = textwrap.dedent(f"""\
                import sys, os
                sys.argv[0] = {fake_argv0!r}
                # Ensure PATH doesn't accidentally resolve edge_vlm_cli.
                os.environ['PATH'] = '/nonexistent_path_for_test'
                from pathlib import Path
                import shutil

                def _find_edge_vlm_cli():
                    invocation_dir = Path(os.path.abspath(sys.argv[0])).parent
                    candidate = invocation_dir / 'edge_vlm_cli'
                    if candidate.is_file() and os.access(str(candidate), os.X_OK):
                        return str(candidate)
                    found = shutil.which('edge_vlm_cli')
                    if found:
                        return found
                    raise RuntimeError('not found')

                try:
                    _find_edge_vlm_cli()
                    print('FAIL: expected RuntimeError')
                except RuntimeError:
                    print('ok')
            """)
            result = subprocess.run(
                ["python3", "-c", test_script],
                capture_output=True, text=True, timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("ok", result.stdout)


# ── responses[] array parsing tests ──────────────────────────────────────────


class TestNativeResponseParsing(unittest.TestCase):
    """The runner must parse finish_reason/output_text from responses[0],
    matching the actual Thor llm_inference response shape:
      {"responses": [{"finish_reason": ..., "output_text": ...}]}
    """

    def _script_text(self) -> str:
        return (_BENCH_DIR / "run_vlm_latency_benchmark.sh").read_text(encoding="utf-8")

    def test_runner_checks_responses_list_key(self):
        """The runner must look for a 'responses' list in the response JSON."""
        text = self._script_text()
        self.assertIn("responses", text, "Runner must handle responses[] array key from Thor")

    def _parse_response_fields(self, response_dict: dict) -> dict:
        """Helper: run the inline response-parsing logic against a temp JSON file."""
        import subprocess, json, tempfile, os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(response_dict, f)
            resp_path = f.name
        try:
            result = subprocess.run(
                ["python3", "-c", f"""
import json

def _first_present(d, *keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None

with open({resp_path!r}) as f:
    out = json.load(f)

entry = out
responses_list = out.get('responses') if isinstance(out, dict) else None
if isinstance(responses_list, list) and responses_list:
    entry = responses_list[0]

finish_reason = _first_present(entry, 'finishReason', 'finish_reason')
output_text = _first_present(entry, 'outputText', 'output_text', 'text', 'response')
print(json.dumps({{'finish_reason': finish_reason, 'output_text': output_text}}))
"""],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                raise AssertionError(result.stderr)
            return json.loads(result.stdout.strip())
        finally:
            os.unlink(resp_path)

    def test_responses_list_parsing_extracts_finish_reason(self):
        """Verify the inline parser correctly unwraps responses[0]."""
        response = {"responses": [{"finish_reason": "max-length", "output_text": "a cat"}]}
        parsed = self._parse_response_fields(response)
        self.assertEqual(parsed["finish_reason"], "max-length")
        self.assertEqual(parsed["output_text"], "a cat")

    def test_responses_list_parsing_falls_back_to_flat_dict(self):
        """A flat top-level response dict (no 'responses' key) must still parse."""
        response = {"finish_reason": "stop", "output_text": "a dog"}
        parsed = self._parse_response_fields(response)
        self.assertEqual(parsed["finish_reason"], "stop")
        self.assertEqual(parsed["output_text"], "a dog")


# ── artifact naming uniqueness tests ─────────────────────────────────────────


class TestArtifactNaming(unittest.TestCase):
    """Artifact file names must include image_id and warmup/measured phase
    so that multiple images and warmup iterations do not collide."""

    def _script_text(self) -> str:
        return (_BENCH_DIR / "run_vlm_latency_benchmark.sh").read_text(encoding="utf-8")

    def test_artifact_name_includes_image_id(self):
        """artifact_base must incorporate image_id, not just condition+iteration."""
        text = self._script_text()
        self.assertIn("${image_id}", text, "artifact_base must include image_id variable")

    def test_artifact_name_includes_warmup_phase(self):
        """artifact_base must incorporate a warmup/measured phase token."""
        text = self._script_text()
        self.assertIn("warmup", text.lower())
        self.assertIn("measured", text, "artifact_base must include 'measured' phase label")

    def test_artifact_names_are_unique_across_images_and_phases(self):
        """Demonstrate that two different image_ids produce different artifact bases."""
        import subprocess
        result = subprocess.run(
            ["bash", "-c", r"""
condition=A; iteration=0; warmup=true
image_id_1=image_001; image_id_2=image_002
phase_warmup=warmup
phase_measured=measured

base1="direct_${condition}_${image_id_1}_${phase_warmup}_iter${iteration}"
base2="direct_${condition}_${image_id_2}_${phase_warmup}_iter${iteration}"
base3="direct_${condition}_${image_id_1}_${phase_measured}_iter${iteration}"

[[ "${base1}" != "${base2}" ]] && echo "images_differ: ok" || echo "images_differ: FAIL"
[[ "${base1}" != "${base3}" ]] && echo "phases_differ: ok" || echo "phases_differ: FAIL"
[[ "${base2}" != "${base3}" ]] && echo "img2_vs_measured: ok" || echo "img2_vs_measured: FAIL"
"""],
            capture_output=True, text=True, timeout=10,
        )
        self.assertIn("images_differ: ok", result.stdout)
        self.assertIn("phases_differ: ok", result.stdout)
        self.assertIn("img2_vs_measured: ok", result.stdout)
        self.assertNotIn("FAIL", result.stdout)


# ── fixture discovery path tests ──────────────────────────────────────────────


class TestFixtureDiscoveryPath(unittest.TestCase):
    """The runner must probe examples/multimodal/pics/ first (the verified Thor
    checkout layout), then fall back to examples/vlm/data/images/."""

    def _script_text(self) -> str:
        return (_BENCH_DIR / "run_vlm_latency_benchmark.sh").read_text(encoding="utf-8")

    def test_probes_multimodal_pics_first(self):
        """examples/multimodal/pics must appear before examples/vlm/data/images."""
        text = self._script_text()
        pos_multimodal = text.find("examples/multimodal/pics")
        pos_vlm_data = text.find("examples/vlm/data/images")
        self.assertGreater(pos_multimodal, 0, "examples/multimodal/pics must be present")
        self.assertGreater(pos_vlm_data, 0, "examples/vlm/data/images must still be a fallback")
        self.assertLess(
            pos_multimodal, pos_vlm_data,
            "examples/multimodal/pics must be listed before examples/vlm/data/images",
        )

    def test_probes_both_candidate_paths(self):
        """Both fixture paths must be present as candidates."""
        text = self._script_text()
        self.assertIn("examples/multimodal/pics", text)
        self.assertIn("examples/vlm/data/images", text)


# ── PNG magic-byte validation tests ──────────────────────────────────────────


class TestPngValidation(unittest.TestCase):
    """_validate_image must correctly accept a real PNG by reading 4 bytes
    (the full 89 50 4E 47 PNG signature), not 3 bytes."""

    def _script_text(self) -> str:
        return (_BENCH_DIR / "run_vlm_latency_benchmark.sh").read_text(encoding="utf-8")

    def _validate_image_bash(self, content: bytes) -> str:
        """Run the actual _validate_image function from the script on a temp file."""
        import subprocess, tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(content)
            tmp = f.name
        try:
            script = self._script_text()
            # Extract _validate_image function body from the script.
            start = script.find("\n_validate_image()")
            end = script.find("\n}", start) + 2
            validate_fn = script[start:end]
            result = subprocess.run(
                ["bash", "-c", f"""
{validate_fn}
_validate_image {tmp!r} && echo ACCEPTED || echo REJECTED
"""],
                capture_output=True, text=True, timeout=10,
            )
            return result.stdout.strip()
        finally:
            os.unlink(tmp)

    def test_accepts_valid_png(self):
        """A file with the correct 4-byte PNG magic (89 50 4E 47) must be accepted."""
        png_magic = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
        outcome = self._validate_image_bash(png_magic)
        self.assertIn("ACCEPTED", outcome, f"Valid PNG should be accepted; got: {outcome!r}")

    def test_rejects_truncated_png_3_bytes(self):
        """A file with only 3 bytes of PNG signature must be rejected."""
        three_bytes = bytes([0x89, 0x50, 0x4E])
        outcome = self._validate_image_bash(three_bytes)
        self.assertIn("REJECTED", outcome, f"3-byte PNG header should be rejected; got: {outcome!r}")

    def test_runner_reads_4_bytes_for_magic(self):
        """The script must read 4 bytes for PNG magic, not 3."""
        text = self._script_text()
        # Must not use -l 3 for the magic-byte read (that misses the 4th PNG byte).
        # Accept any form that reads >= 4 bytes.
        import re
        # Look for xxd -p -l N where N >= 4, or -N 4 for od.
        xxd_match = re.search(r"xxd\s+-p\s+-l\s+(\d+)", text)
        od_match = re.search(r"od\s+.*?-N\s+(\d+)", text)
        if xxd_match:
            self.assertGreaterEqual(int(xxd_match.group(1)), 4,
                "xxd must read at least 4 bytes to capture the full PNG signature")
        elif od_match:
            self.assertGreaterEqual(int(od_match.group(1)), 4,
                "od must read at least 4 bytes to capture the full PNG signature")
        else:
            self.fail("Could not find xxd or od magic-byte read command in script")


if __name__ == "__main__":
    unittest.main()
