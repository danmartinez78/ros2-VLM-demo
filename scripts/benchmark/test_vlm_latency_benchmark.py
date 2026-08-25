"""CPU-only tests for the generic VLM latency benchmark helpers."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_BENCH_DIR))

from vlm_latency_report import (  # noqa: E402
    EXPERIMENT_MATRIX,
    PROMPT_TEXTS,
    aggregate_condition,
    build_condition_spec,
    build_report,
    compute_cold_start_scaling,
    compute_direct_ipc_comparison,
    compute_native_profile_table,
    compute_token_scaling,
    format_text_report,
    parse_jsonl,
    prompt_hash,
)


def record(**overrides):
    base = {
        "schema_version": "1",
        "record_type": "inference",
        "run_id": "test",
        "recorded_at": "2026-08-25T00:00:00Z",
        "condition": "A",
        "path": "ipc",
        "image_id": "image_001",
        "prompt_id": "terse_id",
        "max_output_tokens": 16,
        "actual_output_tokens": 12,
        "finish_reason": None,
        "success": True,
        "error": None,
        "cold_start_total_ms": None,
        "total_latency_ms": 500.0,
        "vision_encoder_ms": None,
        "prefill_ms": None,
        "decode_ms": None,
        "decode_tokens_per_sec": None,
        "average_time_per_token_ms": None,
        "llm_generation_total_gpu_time_ms": None,
        "iteration": 0,
        "warmup": False,
    }
    base.update(overrides)
    return base


class TestMatrix(unittest.TestCase):
    def test_conditions_and_generic_prompt_names(self):
        self.assertEqual([c for c, _, _ in EXPERIMENT_MATRIX], ["A", "B", "C", "D", "E"])
        self.assertIn("compact_scene_json", PROMPT_TEXTS)
        self.assertNotIn("compact_domain_json", PROMPT_TEXTS)
        self.assertEqual(build_condition_spec("C")["max_output_tokens"], 64)
        self.assertEqual(len(prompt_hash("abc")), 12)


class TestParsingAndAggregation(unittest.TestCase):
    def test_parse_jsonl(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as fh:
            fh.write(json.dumps(record()) + "\n")
            fh.write("not-json\n")
            path = Path(fh.name)
        try:
            parsed = parse_jsonl(path)
            self.assertEqual(len(parsed), 1)
        finally:
            path.unlink(missing_ok=True)

    def test_warmup_and_failure_excluded(self):
        agg = aggregate_condition([
            record(total_latency_ms=400.0),
            record(total_latency_ms=9999.0, warmup=True),
            record(total_latency_ms=None, success=False),
        ])
        self.assertEqual(agg["n_measured"], 1)
        self.assertEqual(agg["n_warmup"], 1)
        self.assertEqual(agg["n_failed"], 1)
        self.assertEqual(agg["total_latency_ms"]["mean"], 400.0)

    def test_profile_fields_preserve_unavailable(self):
        agg = aggregate_condition([record()])
        self.assertFalse(agg["prefill_ms"]["available"])
        self.assertFalse(agg["decode_ms"]["available"])

    def test_decode_throughput_can_be_derived(self):
        agg = aggregate_condition([record(actual_output_tokens=20, decode_ms=2000.0)])
        self.assertAlmostEqual(agg["decode_tokens_per_sec"]["mean"], 10.0)


class TestTables(unittest.TestCase):
    def test_token_scaling_uses_runtime_latency(self):
        grouped = {("A", "ipc"): [record(total_latency_ms=500.0)]}
        rows = compute_token_scaling(grouped)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["total_latency_ms_mean"], 500.0)

    def test_cold_start_scaling_is_direct_only(self):
        grouped = {
            ("A", "direct"): [record(path="direct", total_latency_ms=None, cold_start_total_ms=2500.0)],
            ("A", "ipc"): [record()],
        }
        rows = compute_cold_start_scaling(grouped)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["path"], "direct")

    def test_direct_ipc_comparison(self):
        grouped = {
            ("A", "direct"): [record(path="direct", total_latency_ms=None, cold_start_total_ms=2500.0)],
            ("A", "ipc"): [record(total_latency_ms=500.0)],
        }
        rows = compute_direct_ipc_comparison(grouped)
        self.assertEqual(rows[0]["direct_cold_start_total_ms_mean"], 2500.0)
        self.assertEqual(rows[0]["ipc_total_latency_ms_mean"], 500.0)

    def test_native_profile_table(self):
        grouped = {("A", "direct"): [record(path="direct", prefill_ms=30.0, vision_encoder_ms=20.0)]}
        rows = compute_native_profile_table(grouped)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["prefill_ms"], 30.0)


class TestReport(unittest.TestCase):
    def test_report_preserves_raw_records(self):
        records = [record()]
        report = build_report(records)
        self.assertEqual(report["raw_records"], records)
        self.assertIn("experiment_matrix", report)

    def test_text_sections_and_unavailable_semantics(self):
        records = [
            record(path="direct", total_latency_ms=None, cold_start_total_ms=2500.0, finish_reason="max-length"),
            record(path="ipc", total_latency_ms=500.0),
        ]
        text = format_text_report(build_report(records))
        self.assertIn("VLM Latency Characterization Benchmark Report", text)
        self.assertIn("Inference Latency vs Output-Token Cap", text)
        self.assertIn("Cold-Start Wall Time vs Output-Token Cap", text)
        self.assertIn("process/engine/tokenizer initialisation", text)
        self.assertIn("Direct (cold-start, per-process) vs IPC", text)
        self.assertIn("null, not inferred", text)
        self.assertIn("capped", text.lower())


if __name__ == "__main__":
    unittest.main()
