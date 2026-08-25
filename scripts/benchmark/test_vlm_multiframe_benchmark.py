"""CPU-only tests for the generic multi-frame VLM benchmark helpers."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

_BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_BENCH_DIR))

from vlm_multiframe_report import (  # noqa: E402
    FRAME_COUNT_ORDER,
    MAX_OUTPUT_TOKENS,
    MULTIFRAME_PROMPT_TEXT,
    aggregate_frame_condition,
    build_direct_record,
    build_ipc_record,
    build_multiframe_request,
    build_multiframe_request_metadata,
    build_report,
    compute_frame_scaling_table,
    compute_ipc_artifact_table,
    file_sha256_prefix,
    format_text_report,
    frame_condition_label,
    parse_jsonl,
    prompt_hash,
    select_frames,
)


def provenance() -> dict:
    return {
        "model_name": "TestModel-8B",
        "engine_profile_id": "thor-f8",
        "llm_engine_dir": "/tmp/llm",
        "multimodal_engine_dir": "/tmp/mm",
        "engine_manifest_path": "/tmp/engine-manifest.json",
        "engine_manifest_sha256": "1" * 64,
        "engine_identity": "TestModel-8B/thor-f8@111111111111",
        "engine_manifest_status": "matched",
    }


def frame_meta(n: int) -> list[dict]:
    return [{"path": f"/tmp/frame_{i:03d}.jpg", "sha256": "a" * 64} for i in range(n)]


def direct_record(**extra):
    kwargs = dict(
        run_id="test",
        frame_condition="F4",
        frame_paths=frame_meta(4),
        prompt_hash_value=prompt_hash(MULTIFRAME_PROMPT_TEXT),
        max_output_tokens=MAX_OUTPUT_TOKENS,
        iteration=0,
        warmup=False,
        success=True,
        cold_start_total_ms=900.0,
        total_latency_ms=800.0,
        actual_output_tokens=20,
        total_image_tokens=768,
        vision_encoder_ms=50.0,
        prefill_ms=40.0,
        model_name="TestModel-8B",
        engine_provenance=provenance(),
    )
    kwargs.update(extra)
    return build_direct_record(**kwargs)


class TestRequestShape(unittest.TestCase):
    def test_request_preserves_image_order(self):
        paths = [f"/tmp/frame_{i}.jpg" for i in range(4)]
        req = build_multiframe_request(paths, MULTIFRAME_PROMPT_TEXT)
        content = req["requests"][0]["messages"][0]["content"]
        self.assertEqual([item["image"] for item in content[:-1]], paths)
        self.assertEqual(content[-1]["type"], "text")

    def test_metadata(self):
        paths = ["a.jpg", "b.jpg"]
        meta = build_multiframe_request_metadata(paths, MULTIFRAME_PROMPT_TEXT)
        self.assertEqual(meta["frame_count"], 2)
        self.assertEqual(meta["images"][1]["index"], 1)
        self.assertEqual(len(meta["prompt_hash"]), 12)


class TestFrameSelection(unittest.TestCase):
    def test_conditions(self):
        self.assertEqual(FRAME_COUNT_ORDER, ["F1", "F2", "F4", "F8"])
        self.assertEqual(frame_condition_label(8), "F8")

    def test_even_selection_includes_endpoints(self):
        seq = [f"frame_{i}" for i in range(8)]
        selected = select_frames(seq, 4)
        self.assertEqual(selected[0], "frame_0")
        self.assertEqual(selected[-1], "frame_7")
        self.assertEqual(len(selected), 4)

    def test_insufficient_frames(self):
        with self.assertRaises(ValueError):
            select_frames(["a", "b"], 4)


class TestHashing(unittest.TestCase):
    def test_file_hash(self):
        with tempfile.NamedTemporaryFile(delete=False) as fh:
            fh.write(b"temporal benchmark")
            path = fh.name
        try:
            self.assertEqual(len(file_sha256_prefix(path)), 12)
        finally:
            os.unlink(path)


class TestRecordsAndAggregation(unittest.TestCase):
    def test_direct_record(self):
        rec = direct_record()
        self.assertEqual(rec["path"], "direct")
        self.assertEqual(rec["frame_count"], 4)
        self.assertEqual(rec["engine_provenance"]["engine_identity"], provenance()["engine_identity"])

    def test_ipc_record(self):
        rec = build_ipc_record(
            run_id="test",
            frame_condition="F8",
            frame_paths=frame_meta(8),
            prompt_hash_value=prompt_hash(MULTIFRAME_PROMPT_TEXT),
            max_output_tokens=32,
            iteration=1,
            warmup=False,
            success=True,
            total_latency_ms=420.0,
            inference_seconds=0.39,
            ipc_result_path="/tmp/result.json",
            runtime_temporal_encoding="native_video",
            requested_sequence_type="video",
            temporal_fallback_used=False,
            engine_provenance=provenance(),
        )
        self.assertEqual(rec["path"], "ipc")
        self.assertEqual(rec["runtime_temporal_encoding"], "native_video")
        self.assertIsNone(rec["cold_start_total_ms"])

    def test_aggregate(self):
        agg = aggregate_frame_condition([direct_record(), direct_record(warmup=True)])
        self.assertEqual(agg["n_measured"], 1)
        self.assertEqual(agg["n_warmup"], 1)
        self.assertEqual(agg["vision_encoder_ms"]["mean"], 50.0)

    def test_parse_jsonl(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as fh:
            fh.write(json.dumps(direct_record()) + "\n")
            path = Path(fh.name)
        try:
            self.assertEqual(len(parse_jsonl(path)), 1)
        finally:
            path.unlink(missing_ok=True)


class TestReports(unittest.TestCase):
    def test_scaling_and_artifact_tables(self):
        ipc = build_ipc_record(
            run_id="test",
            frame_condition="F4",
            frame_paths=frame_meta(4),
            prompt_hash_value=prompt_hash(MULTIFRAME_PROMPT_TEXT),
            max_output_tokens=32,
            iteration=0,
            warmup=False,
            success=True,
            total_latency_ms=500.0,
            ipc_result_path="/tmp/result.json",
            engine_provenance=provenance(),
        )
        grouped = {("F4", "ipc"): [ipc]}
        scaling = compute_frame_scaling_table(grouped)
        self.assertEqual(scaling[0]["ipc_latency_ms_mean"], 500.0)
        self.assertEqual(compute_ipc_artifact_table([ipc])[0]["ipc_result_path"], "/tmp/result.json")

    def test_report_and_text(self):
        report = build_report([direct_record()])
        self.assertEqual(report["raw_records"][0]["frame_condition"], "F4")
        text = format_text_report(report)
        self.assertIn("VLM Multi-Frame Latency Characterization Report", text)
        self.assertIn("Frame Scaling", text)


if __name__ == "__main__":
    unittest.main()
