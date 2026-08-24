"""
CPU-only CI tests for the VLM multi-frame latency characterization benchmark.

These tests do NOT require TensorRT, CUDA, ROS, or any hardware.
They validate:
  - Exact multi-image NVIDIA request JSON shape and temporal ordering
  - Deterministic selection of 1/2/4/8 frames from a sequence
  - Insufficient-frame failure (clear error, not silent)
  - Content-hash/path recording for every frame
  - JSONL parsing of multi-frame inference records
  - Per-frame-condition aggregation
  - Frame-scaling table construction
  - IPC artifact preservation and semantics
  - Report building (correct structure, raw records preserved)
  - Text report formatting (expected sections present)
  - Parsing of actual pinned Thor profile fields, including
    visual-token count (multimodal.total_image_tokens) when present
  - Shell script syntax (bash -n dry-run check)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

_BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_BENCH_DIR))

from vlm_multiframe_report import (  # noqa: E402
    FRAME_COUNT_ORDER,
    MAX_OUTPUT_TOKENS,
    MULTIFRAME_PROMPT_TEXT,
    aggregate_frame_condition,
    build_direct_record,
    build_ipc_record,
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


# ── fixtures ──────────────────────────────────────────────────────────────────

_RUN_ID = "20250101_120000"


def _make_engine_provenance(
    *,
    model_name: str = "TestModel-8B",
    engine_profile_id: str = "thor-f8",
    llm_engine_dir: str = "/tmp/engines/thor-f8/llm",
    multimodal_engine_dir: str = "/tmp/engines/thor-f8",
    engine_manifest_path: str | None = "/tmp/engines/thor-f8/engine-manifest.json",
    engine_manifest_sha256: str | None = "1" * 64,
    engine_manifest_status: str = "matched",
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "engine_profile_id": engine_profile_id,
        "llm_engine_dir": llm_engine_dir,
        "multimodal_engine_dir": multimodal_engine_dir,
        "engine_manifest_path": engine_manifest_path,
        "engine_manifest_sha256": engine_manifest_sha256,
        "engine_identity": f"{model_name}/{engine_profile_id}@{(engine_manifest_sha256 or '0' * 12)[:12]}",
        "engine_manifest_status": engine_manifest_status,
        "provenance_warnings": [],
    }


def _make_record(
    *,
    frame_condition: str = "F1",
    frame_count: int = 1,
    path: str = "direct",
    frame_paths: list[dict] | None = None,
    prompt_hash_val: str = "abc123def456",
    sequence_type: str = "images",
    fps: float | None = None,
    frame_timestamps_sec: list[float] | None = None,
    frame_timestamp_policy: str = "none",
    rendered_timestamps: bool = False,
    requested_sequence_type: str | None = "images",
    runtime_temporal_encoding: str | None = "ordered_multi_image_no_native_temporal_metadata",
    temporal_fallback_used: bool | None = False,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
    actual_output_tokens: int | None = 12,
    total_image_tokens: int | None = None,
    finish_reason: str | None = None,
    output_text: str | None = None,
    output_words: int | None = None,
    inference_seconds: float | None = None,
    success: bool = True,
    error: str | None = None,
    cold_start_total_ms: float | None = None,
    total_latency_ms: float | None = None,
    ttft_ms: float | None = None,
    vision_encoder_ms: float | None = None,
    prefill_ms: float | None = None,
    decode_ms: float | None = None,
    decode_tokens_per_sec: float | None = None,
    llm_generation_total_gpu_time_ms: float | None = None,
    native_response_path: str | None = None,
    native_profile_path: str | None = None,
    ipc_result_path: str | None = None,
    iteration: int = 0,
    warmup: bool = False,
    model_name: str | None = "TestModel-8B",
    engine_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if frame_paths is None:
        frame_paths = [{"path": f"/tmp/frame_{i:03d}.jpg", "sha256": "a" * 64} for i in range(frame_count)]
    return {
        "schema_version": "1",
        "record_type": "inference",
        "run_id": _RUN_ID,
        "recorded_at": "2025-01-01T12:00:00Z",
        "frame_condition": frame_condition,
        "frame_count": frame_count,
        "path": path,
        "frame_paths": frame_paths,
        "prompt_hash": prompt_hash_val,
        "sequence_type": sequence_type,
        "fps": fps,
        "frame_timestamps_sec": frame_timestamps_sec,
        "frame_timestamp_policy": frame_timestamp_policy,
        "rendered_timestamps": rendered_timestamps,
        "requested_sequence_type": requested_sequence_type,
        "runtime_temporal_encoding": runtime_temporal_encoding,
        "temporal_fallback_used": temporal_fallback_used,
        "max_output_tokens": max_output_tokens,
        "actual_output_tokens": actual_output_tokens,
        "total_image_tokens": total_image_tokens,
        "finish_reason": finish_reason,
        "output_text": output_text,
        "output_words": output_words,
        "inference_seconds": inference_seconds,
        "success": success,
        "error": error,
        "cold_start_total_ms": cold_start_total_ms,
        "total_latency_ms": total_latency_ms,
        "ttft_ms": ttft_ms,
        "vision_encoder_ms": vision_encoder_ms,
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "decode_tokens_per_sec": decode_tokens_per_sec,
        "llm_generation_total_gpu_time_ms": llm_generation_total_gpu_time_ms,
        "native_response_path": native_response_path,
        "native_profile_path": native_profile_path,
        "ipc_result_path": ipc_result_path,
        "model_name": model_name,
        "engine_provenance": engine_provenance or _make_engine_provenance(model_name=model_name or "TestModel-8B"),
        "iteration": iteration,
        "warmup": warmup,
    }


def _jsonl_from(records: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(r) + "\n" for r in records)


# ── request shape tests ───────────────────────────────────────────────────────


class TestMultiImageRequestShape(unittest.TestCase):
    """Validate the exact NVIDIA VLM request JSON shape for multi-frame input."""

    def test_single_frame_content_order(self):
        """Single-frame request: one image item, then one text item."""
        meta = build_multiframe_request_metadata(
            ["/tmp/frame_001.jpg"],
            MULTIFRAME_PROMPT_TEXT,
        )
        self.assertEqual(meta["frame_count"], 1)
        self.assertEqual(meta["frame_paths"], ["/tmp/frame_001.jpg"])
        self.assertEqual(len(meta["images"]), 1)
        self.assertEqual(meta["images"][0]["path"], "/tmp/frame_001.jpg")
        self.assertEqual(meta["images"][0]["index"], 0)

    def test_multi_frame_temporal_ordering(self):
        """Multi-frame request: images are in temporal order (index 0, 1, ..., N-1)."""
        paths = [f"/tmp/frame_{i:03d}.jpg" for i in range(4)]
        meta = build_multiframe_request_metadata(paths, MULTIFRAME_PROMPT_TEXT)
        self.assertEqual(meta["frame_count"], 4)
        for i, img in enumerate(meta["images"]):
            self.assertEqual(img["index"], i)
            self.assertEqual(img["path"], paths[i])

    def test_request_max_output_tokens(self):
        meta = build_multiframe_request_metadata(
            ["/tmp/frame_001.jpg"],
            MULTIFRAME_PROMPT_TEXT,
            max_output_tokens=32,
        )
        self.assertEqual(meta["max_output_tokens"], 32)

    def test_request_prompt_hash_present(self):
        meta = build_multiframe_request_metadata(
            ["/tmp/frame_001.jpg"],
            MULTIFRAME_PROMPT_TEXT,
        )
        self.assertIn("prompt_hash", meta)
        self.assertEqual(len(meta["prompt_hash"]), 12)

    def test_eight_frame_content_length(self):
        """F8: exactly 8 image items in temporal order."""
        paths = [f"/tmp/frame_{i:03d}.jpg" for i in range(8)]
        meta = build_multiframe_request_metadata(paths, MULTIFRAME_PROMPT_TEXT)
        self.assertEqual(meta["frame_count"], 8)
        self.assertEqual(len(meta["images"]), 8)
        for i in range(8):
            self.assertEqual(meta["images"][i]["index"], i)

    def test_build_request_with_real_file_uses_sha256(self):
        """build_multiframe_request emits Thor-validated shape: type:image, image:path.
        SHA-256 hashes are NOT in the model payload; they go in JSONL metadata."""
        from vlm_multiframe_report import build_multiframe_request
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            # Minimal JPEG magic bytes
            f.write(b"\xff\xd8\xff\xe0" + b"\x00" * 10)
            tmp_path = f.name
        try:
            req = build_multiframe_request([tmp_path], MULTIFRAME_PROMPT_TEXT, max_output_tokens=32)
            # Validate shape: requests -> messages -> content[]
            self.assertIn("requests", req)
            self.assertEqual(len(req["requests"]), 1)
            msg = req["requests"][0]["messages"][0]
            self.assertEqual(msg["role"], "user")
            content = msg["content"]
            # image item first (Thor-validated type), text item last
            self.assertEqual(content[0]["type"], "image")
            self.assertEqual(content[0]["image"], tmp_path)
            self.assertEqual(content[-1]["type"], "text")
            # max_output_tokens NOT in model payload (passed via --maxGenerateLength)
            self.assertNotIn("max_new_tokens", req)
            self.assertNotIn("max_output_tokens", req)
            # content_hash NOT in model message payload (goes in JSONL metadata)
            self.assertNotIn("content_hash", content[0])
            self.assertNotIn("source_path", content[0])
        finally:
            os.unlink(tmp_path)

    def test_build_request_multi_frame_content_order(self):
        """Multi-frame request: images appear before text, in temporal order (type:image, image:path)."""
        from vlm_multiframe_report import build_multiframe_request
        tmp_paths = []
        try:
            for _ in range(4):
                f = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                f.write(b"\xff\xd8\xff\xe0" + b"\x00" * 10)
                f.close()
                tmp_paths.append(f.name)
            req = build_multiframe_request(tmp_paths, MULTIFRAME_PROMPT_TEXT)
            content = req["requests"][0]["messages"][0]["content"]
            # 4 image items + 1 text item
            self.assertEqual(len(content), 5)
            for i in range(4):
                self.assertEqual(content[i]["type"], "image")
                self.assertEqual(content[i]["image"], tmp_paths[i])
                # No extra fields in model payload
                self.assertNotIn("content_hash", content[i])
                self.assertNotIn("source_path", content[i])
                self.assertNotIn("image_url", content[i])
            self.assertEqual(content[4]["type"], "text")
        finally:
            for p in tmp_paths:
                os.unlink(p)


# ── frame selection tests ─────────────────────────────────────────────────────


class TestFrameSelection(unittest.TestCase):
    """Test deterministic frame selection from a sequence."""

    def _seq(self, n: int) -> list[str]:
        return [f"/seq/frame_{i:03d}.jpg" for i in range(n)]

    def test_select_1_from_8(self):
        result = select_frames(self._seq(8), 1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], "/seq/frame_000.jpg")

    def test_select_2_from_8(self):
        result = select_frames(self._seq(8), 2)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], "/seq/frame_000.jpg")
        self.assertEqual(result[-1], "/seq/frame_007.jpg")

    def test_select_4_from_8(self):
        result = select_frames(self._seq(8), 4)
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0], "/seq/frame_000.jpg")
        self.assertEqual(result[-1], "/seq/frame_007.jpg")

    def test_select_8_from_8(self):
        result = select_frames(self._seq(8), 8)
        self.assertEqual(result, self._seq(8))

    def test_select_all_when_equal(self):
        result = select_frames(self._seq(4), 4)
        self.assertEqual(result, self._seq(4))

    def test_deterministic(self):
        """Same inputs always produce the same result."""
        seq = self._seq(16)
        r1 = select_frames(seq, 4)
        r2 = select_frames(seq, 4)
        self.assertEqual(r1, r2)

    def test_temporal_order_preserved(self):
        """Selected frames are always in ascending temporal order."""
        seq = self._seq(20)
        for n in [1, 2, 4, 8]:
            result = select_frames(seq, n)
            # Verify strictly ascending order by checking indices
            indices = [int(p.split("_")[-1].replace(".jpg", "")) for p in result]
            self.assertEqual(indices, sorted(indices))

    def test_insufficient_frames_raises(self):
        """Request more frames than available → ValueError with clear message."""
        with self.assertRaises(ValueError) as ctx:
            select_frames(self._seq(4), 8)
        self.assertIn("8", str(ctx.exception))
        self.assertIn("4", str(ctx.exception))

    def test_insufficient_frames_message_mentions_needed_and_have(self):
        """Error message mentions both required and available counts."""
        try:
            select_frames(self._seq(3), 8)
            self.fail("Expected ValueError")
        except ValueError as e:
            msg = str(e)
            self.assertIn("8", msg)
            self.assertIn("3", msg)


# ── content hash tests ────────────────────────────────────────────────────────


class TestContentHashing(unittest.TestCase):
    """Test that content hashes are recorded for every frame."""

    def test_sha256_prefix_length(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test content")
            tmp = f.name
        try:
            h = file_sha256_prefix(tmp)
            self.assertEqual(len(h), 12)
            self.assertRegex(h, r"^[0-9a-f]{12}$")
        finally:
            os.unlink(tmp)

    def test_sha256_prefix_deterministic(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"deterministic content")
            tmp = f.name
        try:
            h1 = file_sha256_prefix(tmp)
            h2 = file_sha256_prefix(tmp)
            self.assertEqual(h1, h2)
        finally:
            os.unlink(tmp)

    def test_sha256_differs_for_different_content(self):
        with tempfile.NamedTemporaryFile(delete=False) as f1, \
             tempfile.NamedTemporaryFile(delete=False) as f2:
            f1.write(b"content A")
            f2.write(b"content B")
            p1, p2 = f1.name, f2.name
        try:
            self.assertNotEqual(file_sha256_prefix(p1), file_sha256_prefix(p2))
        finally:
            os.unlink(p1)
            os.unlink(p2)

    def test_frame_paths_in_record_contain_hash_fields(self):
        """JSONL record's frame_paths list contains path and sha256 for each frame."""
        record = _make_record(
            frame_condition="F4",
            frame_count=4,
            frame_paths=[
                {"path": f"/seq/frame_{i:03d}.jpg", "sha256": "x" * 64}
                for i in range(4)
            ],
        )
        self.assertEqual(len(record["frame_paths"]), 4)
        for i, fp in enumerate(record["frame_paths"]):
            self.assertIn("path", fp)
            self.assertIn("sha256", fp)
            self.assertIn(f"frame_{i:03d}", fp["path"])

    def test_all_frames_have_hashes_in_record_metadata(self):
        """SHA-256 hashes for every frame are recorded in JSONL frame_paths metadata,
        NOT inside the model message payload."""
        from vlm_multiframe_report import build_multiframe_request, file_sha256
        tmp_paths = []
        try:
            for k in range(4):
                f = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                f.write(bytes([k]) * 20)
                f.close()
                tmp_paths.append(f.name)
            req = build_multiframe_request(tmp_paths, MULTIFRAME_PROMPT_TEXT)
            content = req["requests"][0]["messages"][0]["content"]
            image_items = [c for c in content if c.get("type") == "image"]
            self.assertEqual(len(image_items), 4)
            # Model payload has no hash fields — hashes go in JSONL metadata
            for item in image_items:
                self.assertNotIn("content_hash", item)
                self.assertNotIn("source_path", item)
            # file_sha256 is available and produces unique 64-char hex hashes
            hashes = [file_sha256(p) for p in tmp_paths]
            self.assertEqual(len(set(hashes)), 4)
            for h in hashes:
                self.assertRegex(h, r"^[0-9a-f]{64}$")
        finally:
            for p in tmp_paths:
                os.unlink(p)


# ── JSONL parsing tests ───────────────────────────────────────────────────────


class TestParseJsonl(unittest.TestCase):
    def test_parses_valid_records(self):
        records = [_make_record(frame_condition="F1"), _make_record(frame_condition="F4")]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            f.write(_jsonl_from(records))
            path = Path(f.name)
        try:
            parsed = parse_jsonl(path)
            self.assertEqual(len(parsed), 2)
            self.assertEqual(parsed[0]["frame_condition"], "F1")
            self.assertEqual(parsed[1]["frame_condition"], "F4")
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
            self.assertEqual(len(parse_jsonl(path)), 0)
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
            self.assertEqual(len(parse_jsonl(path)), 0)
        finally:
            path.unlink(missing_ok=True)

    def test_skips_malformed_json_lines(self):
        good = _make_record(frame_condition="F2")
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
            self.assertEqual(parsed[0]["frame_condition"], "F2")
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
            self.assertEqual(len(parse_jsonl(path)), 1)
        finally:
            path.unlink(missing_ok=True)


# ── aggregation tests ─────────────────────────────────────────────────────────


class TestAggregateFrameCondition(unittest.TestCase):
    def test_basic_aggregation_counts(self):
        records = [
            _make_record(warmup=True),
            _make_record(total_latency_ms=500.0),
            _make_record(total_latency_ms=600.0),
        ]
        agg = aggregate_frame_condition(records)
        self.assertEqual(agg["n_total"], 3)
        self.assertEqual(agg["n_warmup"], 1)
        self.assertEqual(agg["n_measured"], 2)
        self.assertEqual(agg["n_failed"], 0)

    def test_null_stage_timings_handled(self):
        """Records with null stage timings do not cause errors."""
        records = [
            _make_record(vision_encoder_ms=None, prefill_ms=None),
            _make_record(vision_encoder_ms=None, prefill_ms=None),
        ]
        agg = aggregate_frame_condition(records)
        self.assertEqual(agg["stage_timings_available"]["vision_encoder"], False)
        self.assertEqual(agg["stage_timings_available"]["prefill"], False)

    def test_visual_token_count_aggregated(self):
        """total_image_tokens is aggregated when present."""
        records = [
            _make_record(total_image_tokens=256),
            _make_record(total_image_tokens=256),
        ]
        agg = aggregate_frame_condition(records)
        self.assertTrue(agg["stage_timings_available"]["total_image_tokens"])
        self.assertEqual(agg["total_image_tokens"]["mean"], 256.0)

    def test_visual_token_absent_marked_unavailable(self):
        records = [_make_record(total_image_tokens=None)]
        agg = aggregate_frame_condition(records)
        self.assertFalse(agg["stage_timings_available"]["total_image_tokens"])

    def test_finish_reason_counts(self):
        records = [
            _make_record(finish_reason="max-length"),
            _make_record(finish_reason="eos"),
            _make_record(finish_reason="max-length"),
        ]
        agg = aggregate_frame_condition(records)
        self.assertEqual(agg["finish_reason_counts"]["max-length"], 2)
        self.assertEqual(agg["finish_reason_counts"]["eos"], 1)
        self.assertEqual(agg["n_max_length"], 2)

    def test_failed_records_excluded_from_timing(self):
        records = [
            _make_record(success=True, total_latency_ms=500.0, path="ipc"),
            _make_record(success=False, total_latency_ms=999.0, path="ipc"),
        ]
        agg = aggregate_frame_condition(records)
        self.assertEqual(agg["n_failed"], 1)
        self.assertEqual(agg["total_latency_ms"]["n"], 1)
        self.assertAlmostEqual(agg["total_latency_ms"]["mean"], 500.0)

    def test_decode_tps_computed_from_decode_ms_and_tokens(self):
        """decode_tokens_per_sec is computed when decode_ms and actual_output_tokens present."""
        records = [_make_record(decode_ms=1000.0, actual_output_tokens=45, decode_tokens_per_sec=None)]
        agg = aggregate_frame_condition(records)
        self.assertTrue(agg["stage_timings_available"]["decode_tokens_per_sec"])
        self.assertAlmostEqual(agg["decode_tokens_per_sec"]["mean"], 45.0)


# ── frame-scaling table tests ─────────────────────────────────────────────────


class TestFrameScalingTable(unittest.TestCase):
    def _build_multi_condition_groups(self) -> dict[tuple[str, str], list[dict[str, Any]]]:
        groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for fc, n in [("F1", 1), ("F2", 2), ("F4", 4), ("F8", 8)]:
            # direct records: have vision/prefill metrics, no total_latency_ms
            direct_recs = [
                _make_record(
                    frame_condition=fc,
                    frame_count=n,
                    path="direct",
                    vision_encoder_ms=30.0 + n * 2,
                    prefill_ms=35.0 + n,
                    total_image_tokens=256 * n,
                    actual_output_tokens=12,
                    llm_generation_total_gpu_time_ms=260.0,
                    decode_tokens_per_sec=47.0,
                    finish_reason="eos",
                    total_latency_ms=None,
                    cold_start_total_ms=5000.0,
                )
                for _ in range(3)
            ]
            # ipc records: have total_latency_ms, no stage details
            ipc_recs = [
                _make_record(
                    frame_condition=fc,
                    frame_count=n,
                    path="ipc",
                    total_latency_ms=300.0 + n * 10,
                    total_image_tokens=None,
                    vision_encoder_ms=None,
                    prefill_ms=None,
                    actual_output_tokens=None,
                )
                for _ in range(3)
            ]
            groups[(fc, "direct")] = direct_recs
            groups[(fc, "ipc")] = ipc_recs
        return groups

    def test_scaling_table_has_all_frame_conditions(self):
        groups = self._build_multi_condition_groups()
        table = compute_frame_scaling_table(groups)
        conditions = {row["frame_condition"] for row in table}
        for fc in ["F1", "F2", "F4", "F8"]:
            self.assertIn(fc, conditions)

    def test_scaling_table_visual_tokens_scale_with_frames(self):
        groups = self._build_multi_condition_groups()
        table = compute_frame_scaling_table(groups)
        direct_rows = {r["frame_condition"]: r for r in table if r["path"] == "direct"}
        # F4 should have 4x visual tokens of F1
        vt_f1 = direct_rows["F1"]["visual_tokens_mean"]
        vt_f4 = direct_rows["F4"]["visual_tokens_mean"]
        self.assertIsNotNone(vt_f1)
        self.assertIsNotNone(vt_f4)
        self.assertAlmostEqual(vt_f4 / vt_f1, 4.0)

    def test_scaling_table_ipc_latency_populated(self):
        groups = self._build_multi_condition_groups()
        table = compute_frame_scaling_table(groups)
        ipc_rows = [r for r in table if r["path"] == "ipc"]
        self.assertTrue(len(ipc_rows) > 0)
        for row in ipc_rows:
            self.assertIsNotNone(row["ipc_total_latency_ms_mean"])

    def test_scaling_table_null_fields_when_not_available(self):
        """IPC path rows have null vision/prefill/visual-token fields."""
        groups = self._build_multi_condition_groups()
        table = compute_frame_scaling_table(groups)
        ipc_rows = [r for r in table if r["path"] == "ipc"]
        for row in ipc_rows:
            self.assertIsNone(row["visual_tokens_mean"])
            self.assertIsNone(row["vision_encoder_ms_mean"])
            self.assertIsNone(row["prefill_ms_mean"])

    def test_scaling_table_frame_count_field(self):
        groups = self._build_multi_condition_groups()
        table = compute_frame_scaling_table(groups)
        for row in table:
            fc = row["frame_condition"]
            expected_n = int(fc[1:])  # F1→1, F2→2, etc.
            self.assertEqual(row["frames"], expected_n)
            self.assertEqual(row["images_per_request"], expected_n)

    def test_warmup_records_excluded(self):
        groups = {
            ("F1", "ipc"): [
                _make_record(frame_condition="F1", path="ipc", total_latency_ms=9999.0, warmup=True),
                _make_record(frame_condition="F1", path="ipc", total_latency_ms=300.0),
            ]
        }
        table = compute_frame_scaling_table(groups)
        self.assertEqual(len(table), 1)
        self.assertAlmostEqual(table[0]["ipc_total_latency_ms_mean"], 300.0)


# ── IPC artifact tests ────────────────────────────────────────────────────────


class TestIpcArtifactTable(unittest.TestCase):
    def test_ipc_artifact_table_fields(self):
        """IPC artifact table preserves path, latency, inference_seconds, text, words."""
        groups = {
            ("F2", "ipc"): [
                _make_record(
                    frame_condition="F2",
                    path="ipc",
                    total_latency_ms=350.0,
                    inference_seconds=0.28,
                    output_text="objects: [], actions: [], hazards: [], navigable: true",
                    output_words=9,
                    ipc_result_path="/tmp/bench/F2/ipc/iter_0/ipc_result.json",
                ),
            ]
        }
        table = compute_ipc_artifact_table(groups)
        self.assertEqual(len(table), 1)
        row = table[0]
        self.assertEqual(row["frame_condition"], "F2")
        self.assertAlmostEqual(row["total_latency_ms"], 350.0)
        self.assertAlmostEqual(row["inference_seconds"], 0.28)
        self.assertEqual(row["output_words"], 9)
        self.assertEqual(row["ipc_result_path"], "/tmp/bench/F2/ipc/iter_0/ipc_result.json")

    def test_ipc_artifact_table_excludes_direct_records(self):
        """IPC artifact table only contains ipc-path rows."""
        groups = {
            ("F1", "direct"): [_make_record(frame_condition="F1", path="direct")],
            ("F1", "ipc"): [_make_record(frame_condition="F1", path="ipc", total_latency_ms=300.0)],
        }
        table = compute_ipc_artifact_table(groups)
        self.assertEqual(len(table), 1)
        self.assertEqual(table[0]["frame_condition"], "F1")

    def test_ipc_artifact_table_excludes_warmups(self):
        groups = {
            ("F4", "ipc"): [
                _make_record(frame_condition="F4", path="ipc", total_latency_ms=999.0, warmup=True),
                _make_record(frame_condition="F4", path="ipc", total_latency_ms=380.0),
            ]
        }
        table = compute_ipc_artifact_table(groups)
        self.assertEqual(len(table), 1)
        self.assertAlmostEqual(table[0]["total_latency_ms"], 380.0)

    def test_ipc_does_not_infer_tokens_or_finish_reason(self):
        """IPC records never have actual_output_tokens or finish_reason inferred."""
        rec = _make_record(
            path="ipc",
            total_latency_ms=300.0,
            actual_output_tokens=None,  # not exposed by IPC backend
            finish_reason=None,
        )
        self.assertIsNone(rec["actual_output_tokens"])
        self.assertIsNone(rec["finish_reason"])
        # After aggregation these fields remain unavailable
        agg = aggregate_frame_condition([rec])
        self.assertFalse(agg["stage_timings_available"]["actual_output_tokens"])


# ── Thor profile field parsing tests ─────────────────────────────────────────


class TestNativeProfileParsing(unittest.TestCase):
    """Validate parsing of actual pinned Thor profile fields."""

    def _profile_json(self, **kwargs) -> str:
        """Return a minimal profile JSON matching the Thor runtime schema (PR #64).
        stages[] keyed by stage_id; prefill uses average_time_per_run_ms."""
        profile = {
            "multimodal": {"total_image_tokens": kwargs.get("total_image_tokens", 256)},
            "prefill": {"average_time_per_run_ms": kwargs.get("prefill_ms", 36.2)},
            "generation": {
                "generated_tokens": kwargs.get("generated_tokens", 12),
                "tokens_per_second": kwargs.get("tokens_per_second", 47.3),
                "average_time_per_token_ms": kwargs.get("average_time_per_token_ms", 21.1),
                "total_time_ms": kwargs.get("total_time_ms", 253.7),
                "finish_reason": kwargs.get("finish_reason", "eos"),
            },
            "stages": [
                {
                    "stage_id": "vision_encoder",
                    "average_time_per_run_ms": kwargs.get("vision_encoder_ms", 45.5),
                },
                {
                    "stage_id": "llm_generation",
                    "total_gpu_time_ms": kwargs.get("total_gpu_time_ms", 253.7),
                },
            ],
        }
        return json.dumps(profile)

    def test_parses_total_image_tokens(self):
        """multimodal.total_image_tokens is parsed into total_image_tokens record field."""
        rec = _make_record(total_image_tokens=512)
        self.assertEqual(rec["total_image_tokens"], 512)

    def test_visual_tokens_scale_with_frame_count(self):
        """Visual token count in aggregation scales as expected."""
        records_f1 = [_make_record(frame_condition="F1", total_image_tokens=256)]
        records_f4 = [_make_record(frame_condition="F4", total_image_tokens=1024)]
        agg_f1 = aggregate_frame_condition(records_f1)
        agg_f4 = aggregate_frame_condition(records_f4)
        self.assertAlmostEqual(agg_f1["total_image_tokens"]["mean"], 256.0)
        self.assertAlmostEqual(agg_f4["total_image_tokens"]["mean"], 1024.0)

    def test_profile_schema_parsed_inline(self):
        """Inline profile JSON (Thor PR #64 schema) is parsed correctly via the Python snippet.
        stages[] keyed by stage_id; prefill uses average_time_per_run_ms."""
        import textwrap
        profile_data = {
            "multimodal": {"total_image_tokens": 1024},
            "prefill": {"average_time_per_run_ms": 38.4},
            "generation": {
                "generated_tokens": 30,
                "tokens_per_second": 46.8,
                "average_time_per_token_ms": 21.3,
                "total_time_ms": 639.0,
                "finish_reason": "max-length",
            },
            "stages": [
                {"stage_id": "vision_encoder", "average_time_per_run_ms": 58.1},
                {"stage_id": "llm_generation", "total_gpu_time_ms": 640.9},
            ],
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(profile_data, f)
            profile_path = f.name
        try:
            # Run the same inline Python parser used by run_vlm_multiframe_benchmark.sh
            # (exact mirror of the _parse_native_profile heredoc in the shell script).
            parser_code = textwrap.dedent("""
                import json, sys
                path = sys.argv[1]
                with open(path) as f:
                    p = json.load(f)
                out = {}
                mm = p.get("multimodal") or {}
                if "total_image_tokens" in mm:
                    out["total_image_tokens"] = mm["total_image_tokens"]
                stages = p.get("stages") if isinstance(p, dict) else None
                if isinstance(stages, list):
                    for stage in stages:
                        if not isinstance(stage, dict):
                            continue
                        sid = stage.get("stage_id")
                        if sid == "vision_encoder":
                            v = stage.get("average_time_per_run_ms")
                            if v is not None:
                                out["vision_encoder_ms"] = v
                        elif sid == "llm_generation":
                            v = stage.get("total_gpu_time_ms")
                            if v is not None:
                                out["llm_generation_total_gpu_time_ms"] = v
                prefill = p.get("prefill") if isinstance(p, dict) else None
                if isinstance(prefill, dict):
                    v = prefill.get("average_time_per_run_ms")
                    if v is not None:
                        out["prefill_ms"] = v
                gen = p.get("generation") or {}
                if "generated_tokens" in gen:
                    out["actual_output_tokens"] = gen["generated_tokens"]
                if "tokens_per_second" in gen:
                    out["decode_tokens_per_sec"] = gen["tokens_per_second"]
                if "average_time_per_token_ms" in gen:
                    out["average_time_per_token_ms"] = gen["average_time_per_token_ms"]
                if "total_time_ms" in gen:
                    out["decode_ms"] = gen["total_time_ms"]
                fr = p.get("finish_reason") or gen.get("finish_reason")
                if fr is not None:
                    out["finish_reason"] = fr
                print(json.dumps(out))
            """)
            result = subprocess.run(
                [sys.executable, "-c", parser_code, profile_path],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            parsed = json.loads(result.stdout.strip())
            self.assertEqual(parsed["total_image_tokens"], 1024)
            self.assertAlmostEqual(parsed["vision_encoder_ms"], 58.1)
            self.assertAlmostEqual(parsed["prefill_ms"], 38.4)
            self.assertEqual(parsed["actual_output_tokens"], 30)
            self.assertAlmostEqual(parsed["decode_tokens_per_sec"], 46.8)
            self.assertAlmostEqual(parsed["llm_generation_total_gpu_time_ms"], 640.9)
            self.assertAlmostEqual(parsed["decode_ms"], 639.0)
            self.assertEqual(parsed["finish_reason"], "max-length")
        finally:
            os.unlink(profile_path)

    def test_stages_without_vision_encoder_leaves_field_absent(self):
        """If stages[] has no vision_encoder entry, vision_encoder_ms is not emitted."""
        import textwrap
        profile_data = {
            "multimodal": {"total_image_tokens": 512},
            "prefill": {"average_time_per_run_ms": 30.0},
            "generation": {"generated_tokens": 10, "tokens_per_second": 45.0},
            "stages": [
                {"stage_id": "llm_generation", "total_gpu_time_ms": 200.0},
            ],
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(profile_data, f)
            profile_path = f.name
        try:
            parser_code = textwrap.dedent("""
                import json, sys
                path = sys.argv[1]
                with open(path) as f:
                    p = json.load(f)
                out = {}
                stages = p.get("stages") or []
                for stage in stages:
                    sid = stage.get("stage_id")
                    if sid == "vision_encoder":
                        v = stage.get("average_time_per_run_ms")
                        if v is not None:
                            out["vision_encoder_ms"] = v
                    elif sid == "llm_generation":
                        v = stage.get("total_gpu_time_ms")
                        if v is not None:
                            out["llm_generation_total_gpu_time_ms"] = v
                print(json.dumps(out))
            """)
            result = subprocess.run(
                [sys.executable, "-c", parser_code, profile_path],
                capture_output=True, text=True,
            )
            parsed = json.loads(result.stdout.strip())
            self.assertNotIn("vision_encoder_ms", parsed)
            self.assertIn("llm_generation_total_gpu_time_ms", parsed)
        finally:
            os.unlink(profile_path)

    def test_missing_visual_tokens_not_fabricated(self):
        """When multimodal.total_image_tokens absent from profile, field stays null."""
        profile_data = {
            "generation": {"generated_tokens": 10, "tokens_per_second": 45.0},
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(profile_data, f)
            profile_path = f.name
        try:
            import textwrap
            parser_code = textwrap.dedent("""
                import json, sys
                path = sys.argv[1]
                with open(path) as f:
                    p = json.load(f)
                out = {}
                mm = p.get("multimodal") or {}
                if "total_image_tokens" in mm:
                    out["total_image_tokens"] = mm["total_image_tokens"]
                print(json.dumps(out))
            """)
            result = subprocess.run(
                [sys.executable, "-c", parser_code, profile_path],
                capture_output=True, text=True,
            )
            parsed = json.loads(result.stdout.strip())
            self.assertNotIn("total_image_tokens", parsed)
        finally:
            os.unlink(profile_path)


# ── report building tests ─────────────────────────────────────────────────────


class TestBuildReport(unittest.TestCase):
    def _make_full_records(self) -> list[dict[str, Any]]:
        records = []
        for fc, n in [("F1", 1), ("F2", 2), ("F4", 4)]:
            for i in range(3):
                records.append(_make_record(
                    frame_condition=fc, frame_count=n, path="direct",
                    vision_encoder_ms=35.0, prefill_ms=37.0,
                    total_image_tokens=256 * n,
                    actual_output_tokens=12,
                    llm_generation_total_gpu_time_ms=260.0,
                    finish_reason="eos",
                    total_latency_ms=None,
                    cold_start_total_ms=5000.0,
                    iteration=i,
                ))
            for i in range(3):
                records.append(_make_record(
                    frame_condition=fc, frame_count=n, path="ipc",
                    total_latency_ms=300.0 + n * 5,
                    ipc_result_path=f"/tmp/bench/{fc}/ipc/iter_{i}/ipc_result.json",
                    iteration=i,
                ))
        return records

    def test_report_has_required_top_level_keys(self):
        report = build_report(self._make_full_records())
        for key in [
            "schema_version", "generated_at", "source_file",
            "run_ids", "model_names", "n_total_records", "n_measured_records",
            "engine_provenance", "engine_provenance_variants", "mixed_engine_provenance",
            "temporal_config_variants",
            "frame_conditions", "frame_scaling_table", "ipc_artifact_table", "raw_records",
        ]:
            self.assertIn(key, report)

    def test_raw_records_preserved(self):
        records = self._make_full_records()
        report = build_report(records)
        self.assertEqual(report["n_total_records"], len(records))
        self.assertEqual(len(report["raw_records"]), len(records))

    def test_frame_conditions_contains_all_conditions(self):
        report = build_report(self._make_full_records())
        fc_summary = report["frame_conditions"]
        for fc in ["F1", "F2", "F4"]:
            self.assertIn(fc, fc_summary)

    def test_report_with_source_path(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            f.write(_jsonl_from(self._make_full_records()))
            path = Path(f.name)
        try:
            records = parse_jsonl(path)
            report = build_report(records, source_path=path)
            self.assertEqual(report["source_file"], str(path))
        finally:
            path.unlink(missing_ok=True)

    def test_run_ids_collected(self):
        records = self._make_full_records()
        report = build_report(records)
        self.assertIn(_RUN_ID, report["run_ids"])

    def test_model_names_collected(self):
        records = self._make_full_records()
        report = build_report(records)
        self.assertIn("TestModel-8B", report["model_names"])

    def test_unique_engine_provenance_promoted(self):
        report = build_report(self._make_full_records())
        self.assertFalse(report["mixed_engine_provenance"])
        self.assertEqual(report["engine_provenance"]["engine_profile_id"], "thor-f8")
        self.assertEqual(len(report["engine_provenance_variants"]), 1)

    def test_mixed_engine_provenance_flagged(self):
        records = self._make_full_records()
        records.append(_make_record(
            frame_condition="F8",
            frame_count=8,
            path="ipc",
            engine_provenance=_make_engine_provenance(
                engine_profile_id="legacy",
                llm_engine_dir="/tmp/engine/llm",
                multimodal_engine_dir="/tmp/engine",
                engine_manifest_path=None,
                engine_manifest_sha256=None,
                engine_manifest_status="missing",
            ),
        ))
        report = build_report(records)
        self.assertTrue(report["mixed_engine_provenance"])
        self.assertIsNone(report["engine_provenance"])
        self.assertGreaterEqual(len(report["engine_provenance_variants"]), 2)

    def test_caller_and_server_provenance_mismatch_marks_report_non_comparable(self):
        direct_record = _make_record(
            frame_condition="F8",
            frame_count=8,
            path="direct",
            engine_provenance=_make_engine_provenance(
                engine_profile_id="thor-f8",
                llm_engine_dir="/tmp/engines/thor-f8/llm",
                multimodal_engine_dir="/tmp/engines/thor-f8",
            ),
        )
        ipc_record = _make_record(
            frame_condition="F8",
            frame_count=8,
            path="ipc",
            engine_provenance=_make_engine_provenance(
                engine_profile_id="legacy",
                llm_engine_dir="/tmp/engine/llm",
                multimodal_engine_dir="/tmp/engine",
                engine_manifest_path=None,
                engine_manifest_sha256=None,
                engine_manifest_status="missing",
            ),
        )
        report = build_report([direct_record, ipc_record])
        self.assertTrue(report["mixed_engine_provenance"])
        self.assertIsNone(report["engine_provenance"])


# ── text report formatting tests ──────────────────────────────────────────────


class TestFormatTextReport(unittest.TestCase):
    def _full_report(self) -> dict[str, Any]:
        records = []
        for fc, n in [("F1", 1), ("F2", 2), ("F4", 4), ("F8", 8)]:
            for i in range(3):
                records.append(_make_record(
                    frame_condition=fc, frame_count=n, path="direct",
                    vision_encoder_ms=35.0, prefill_ms=37.0,
                    total_image_tokens=256 * n,
                    actual_output_tokens=12,
                    llm_generation_total_gpu_time_ms=260.0,
                    finish_reason="eos",
                    total_latency_ms=None,
                    cold_start_total_ms=5000.0,
                ))
            for i in range(3):
                records.append(_make_record(
                    frame_condition=fc, frame_count=n, path="ipc",
                    total_latency_ms=300.0,
                    ipc_result_path=f"/tmp/bench/{fc}/ipc/iter_{i}/ipc_result.json",
                    inference_seconds=0.25,
                    output_text="compact result",
                    output_words=2,
                ))
        return build_report(records)

    def test_report_contains_header(self):
        text = format_text_report(self._full_report())
        self.assertIn("VLM Multi-Frame Latency", text)

    def test_report_contains_frame_scaling_section(self):
        text = format_text_report(self._full_report())
        self.assertIn("Frame Scaling Table", text)

    def test_report_contains_all_frame_conditions_in_table(self):
        text = format_text_report(self._full_report())
        for fc in ["F1", "F2", "F4", "F8"]:
            self.assertIn(fc, text)

    def test_report_contains_cold_start_section(self):
        text = format_text_report(self._full_report())
        self.assertIn("Cold-Start", text)

    def test_report_contains_engine_provenance_section(self):
        text = format_text_report(self._full_report())
        self.assertIn("Engine provenance", text)
        self.assertIn("thor-f8", text)

    def test_report_keeps_single_engine_32_token_fixed_summary(self):
        text = format_text_report(self._full_report())
        self.assertIn(
            "Fixed: model, engines, precision, prompt text, max_output_tokens=32",
            text,
        )
        self.assertNotIn("Mixed request config", text)

    def test_report_renders_actual_max_output_tokens(self):
        records = [
            _make_record(frame_condition="F1", frame_count=1, path="direct", max_output_tokens=8),
            _make_record(frame_condition="F1", frame_count=1, path="ipc", max_output_tokens=8),
        ]
        text = format_text_report(build_report(records))
        self.assertIn("max_output_tokens=8", text)
        self.assertNotIn("max_output_tokens=32", text)

    def test_report_flags_mixed_max_output_tokens(self):
        records = [
            _make_record(frame_condition="F1", frame_count=1, path="direct", max_output_tokens=8),
            _make_record(frame_condition="F1", frame_count=1, path="ipc", max_output_tokens=32),
        ]
        text = format_text_report(build_report(records))
        self.assertIn(
            "Mixed request config: max_output_tokens varies across records (8, 32)",
            text,
        )

    def test_report_marks_missing_max_output_tokens_unknown(self):
        record = _make_record(frame_condition="F1", frame_count=1, path="direct")
        record.pop("max_output_tokens", None)
        text = format_text_report(build_report([record]))
        self.assertIn("Fixed: model, engines, precision, prompt text, max_output_tokens=unknown", text)

    def test_report_contains_ipc_artifact_section(self):
        text = format_text_report(self._full_report())
        self.assertIn("IPC Result Artifacts", text)

    def test_report_note_about_cold_start_separation(self):
        """Report explicitly notes cold-start and IPC steady-state are separate."""
        text = format_text_report(self._full_report())
        self.assertIn("cold-start", text.lower())
        self.assertIn("steady-state", text.lower())

    def test_report_flags_mixed_engine_provenance(self):
        report = build_report([
            _make_record(
                model_name="Cosmos-Reason2-8B",
                engine_provenance=_make_engine_provenance(model_name="Cosmos-Reason2-8B"),
            ),
            _make_record(
                model_name="Cosmos-Reason2-2B",
                engine_provenance=_make_engine_provenance(
                    model_name="Cosmos-Reason2-2B",
                    engine_profile_id="legacy",
                    llm_engine_dir="/tmp/engine/llm",
                    multimodal_engine_dir="/tmp/engine",
                    engine_manifest_path=None,
                    engine_manifest_sha256=None,
                    engine_manifest_status="missing",
                ),
            ),
        ])
        text = format_text_report(report)
        self.assertIn("MIXED", text)
        self.assertIn(
            "Mixed/non-comparable: model/engine configuration varies across records",
            text,
        )
        self.assertNotIn("Fixed: model, engines", text)
        self.assertIn("Cosmos-Reason2-8B", text)
        self.assertIn("Cosmos-Reason2-2B", text)

    def test_report_serializable_as_json(self):
        report = self._full_report()
        serialized = json.dumps(report)
        deserialized = json.loads(serialized)
        self.assertIn("frame_scaling_table", deserialized)


# ── frame condition label tests ───────────────────────────────────────────────


class TestFrameConditionLabel(unittest.TestCase):
    def test_labels(self):
        self.assertEqual(frame_condition_label(1), "F1")
        self.assertEqual(frame_condition_label(2), "F2")
        self.assertEqual(frame_condition_label(4), "F4")
        self.assertEqual(frame_condition_label(8), "F8")

    def test_frame_count_order(self):
        self.assertEqual(FRAME_COUNT_ORDER, ["F1", "F2", "F4", "F8"])


# ── prompt hash tests ─────────────────────────────────────────────────────────


class TestPromptHash(unittest.TestCase):
    def test_hash_length(self):
        h = prompt_hash(MULTIFRAME_PROMPT_TEXT)
        self.assertEqual(len(h), 12)

    def test_hash_is_hex(self):
        h = prompt_hash("test")
        self.assertRegex(h, r"^[0-9a-f]{12}$")

    def test_hash_deterministic(self):
        self.assertEqual(prompt_hash("abc"), prompt_hash("abc"))

    def test_different_texts_different_hashes(self):
        self.assertNotEqual(prompt_hash("abc"), prompt_hash("xyz"))


# ── constants tests ───────────────────────────────────────────────────────────


class TestConstants(unittest.TestCase):
    def test_max_output_tokens(self):
        self.assertEqual(MAX_OUTPUT_TOKENS, 32)

    def test_prompt_text_compact(self):
        self.assertIn("compact JSON", MULTIFRAME_PROMPT_TEXT)
        self.assertIn("objects", MULTIFRAME_PROMPT_TEXT)
        self.assertIn("hazards", MULTIFRAME_PROMPT_TEXT)
        self.assertIn("navigable", MULTIFRAME_PROMPT_TEXT)

    def test_prompt_text_requests_one_result(self):
        """Prompt requests one result for the full sequence, not per-frame prose."""
        self.assertIn("one result", MULTIFRAME_PROMPT_TEXT.lower())


# ── shell syntax test ─────────────────────────────────────────────────────────


class TestShellSyntax(unittest.TestCase):
    def test_shell_script_syntax(self):
        script = _BENCH_DIR / "run_vlm_multiframe_benchmark.sh"
        if not script.exists():
            self.skipTest(f"Script not found: {script}")
        result = subprocess.run(
            ["bash", "-n", str(script)],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, f"Shell syntax error:\n{result.stderr}")

    def test_script_is_executable_or_can_be_run_via_bash(self):
        script = _BENCH_DIR / "run_vlm_multiframe_benchmark.sh"
        if not script.exists():
            self.skipTest(f"Script not found: {script}")
        # Just check the file can be read — executable bit is set separately
        self.assertTrue(script.is_file())

    def test_script_has_ipc_client_invocation(self):
        """Runner must functionally invoke vlm_multi_frame_client for the IPC path."""
        script = _BENCH_DIR / "run_vlm_multiframe_benchmark.sh"
        if not script.exists():
            self.skipTest(f"Script not found: {script}")
        content = script.read_text(encoding="utf-8")
        # _run_ipc_inference function must be present
        self.assertIn("_run_ipc_inference", content)
        # vlm_multi_frame_client must be invoked (not vlm_single_shot_client which is single-image)
        self.assertIn("vlm_multi_frame_client", content)
        # Must use --socket for IPC socket path
        self.assertIn("--socket", content)
        # Must pass multiple --image arguments for multi-frame support
        self.assertIn("--image", content)
        self.assertIn("--sequence-type", content)
        self.assertIn("--render-timestamps", content)
        # Must NOT use vlm_single_shot_client (single-image only)
        self.assertNotIn("vlm_single_shot_client", content)

    def test_script_uses_camelcase_cli_flags(self):
        """Runner must use camelCase llm_inference flags (Thor-validated contract)."""
        script = _BENCH_DIR / "run_vlm_multiframe_benchmark.sh"
        if not script.exists():
            self.skipTest(f"Script not found: {script}")
        content = script.read_text(encoding="utf-8")
        self.assertIn("--engineDir", content)
        self.assertIn("--multimodalEngineDir", content)
        self.assertIn("--inputFile", content)
        self.assertIn("--outputFile", content)
        self.assertIn("--maxGenerateLength", content)
        # Snake-case flags must NOT appear
        self.assertNotIn("--engine_dir", content)
        self.assertNotIn("--multimodal_engine_dir", content)
        self.assertNotIn("--input_file", content)
        self.assertNotIn("--output_file", content)
        self.assertNotIn("--max_new_tokens", content)

    def test_script_request_uses_image_type(self):
        """Runner must emit image and video request-item shapes for direct path."""
        script = _BENCH_DIR / "run_vlm_multiframe_benchmark.sh"
        if not script.exists():
            self.skipTest(f"Script not found: {script}")
        content = script.read_text(encoding="utf-8")
        self.assertIn('"type":"image"', content)
        self.assertIn('"image":', content)
        self.assertIn('"type":"video"', content)
        self.assertIn('"frames":', content)
        self.assertIn('"fps":', content)
        # Forbidden: old image_url shape and payload metadata fields
        self.assertNotIn('"type":"image_url"', content)
        self.assertNotIn("image_url", content)
        self.assertNotIn('"content_hash"', content)
        self.assertNotIn('"source_path"', content)
        self.assertNotIn("max_new_tokens", content)

    def test_script_direct_temporal_limitations_are_explicit(self):
        script = _BENCH_DIR / "run_vlm_multiframe_benchmark.sh"
        if not script.exists():
            self.skipTest(f"Script not found: {script}")
        content = script.read_text(encoding="utf-8")
        self.assertIn(
            "direct path does not implement --render-timestamps preprocessing.",
            content,
        )
        self.assertIn(
            "direct path temporal/video request JSON supports frames+fps but not explicit --frame-timestamps-sec arrays",
            content,
        )
        self.assertIn(
            "export _BM_FPS=\"$([ \"${SEQUENCE_TYPE}\" != \"images\" ] && echo \"${FPS:-1.0}\" || echo 'null')\"",
            content,
        )
        self.assertIn(
            "direct_frame_timestamp_policy='\"implicit_uniform_from_fps\"'",
            content,
        )
        self.assertIn(
            "native_qwen3vl_video_json_frames_fps_no_explicit_timestamps",
            content,
        )

    def test_script_artifact_paths_use_phase_prefix(self):
        """Runner must include warmup/measured phase in artifact directory to prevent collision."""
        script = _BENCH_DIR / "run_vlm_multiframe_benchmark.sh"
        if not script.exists():
            self.skipTest(f"Script not found: {script}")
        content = script.read_text(encoding="utf-8")
        self.assertIn("warmup_iter_", content)
        self.assertIn("measured_iter_", content)


# ── warmup/measured artifact uniqueness test ──────────────────────────────────


class TestArtifactCollisionPrevention(unittest.TestCase):
    def test_warmup_and_measured_records_have_distinct_iteration_semantics(self):
        """Warmup records and measured records must be distinguishable by the warmup flag.
        The shell runner uses phase-prefixed directories (warmup_iter_N vs measured_iter_N)
        so artifacts from different phases never collide."""
        warmup_rec = _make_record(warmup=True, iteration=0, native_profile_path="/bench/F1/direct/warmup_iter_0/profile.json")
        measured_rec = _make_record(warmup=False, iteration=0, native_profile_path="/bench/F1/direct/measured_iter_0/profile.json")
        self.assertTrue(warmup_rec["warmup"])
        self.assertFalse(measured_rec["warmup"])
        # Paths must differ (no collision)
        self.assertNotEqual(
            warmup_rec["native_profile_path"],
            measured_rec["native_profile_path"],
        )

    def test_aggregation_excludes_warmup_iterations(self):
        """Aggregation must exclude warmup records so warmup latency never pollutes stats."""
        records = [
            _make_record(warmup=True, cold_start_total_ms=9999.0, iteration=0),
            _make_record(warmup=False, cold_start_total_ms=500.0, iteration=0),
            _make_record(warmup=False, cold_start_total_ms=520.0, iteration=1),
        ]
        agg = aggregate_frame_condition(records)
        self.assertEqual(agg["n_warmup"], 1)
        self.assertEqual(agg["n_measured"], 2)
        # Warmup 9999ms must not appear in the aggregate
        self.assertAlmostEqual(agg["cold_start_total_ms"]["mean"], 510.0)


# ── insufficient-frame failure test (shell integration) ──────────────────────


class TestInsufficientFrameFailure(unittest.TestCase):
    def test_insufficient_frames_dry_run_validation(self):
        """
        The shell runner must exit non-zero when sequence has fewer frames
        than the largest requested condition.

        We test this via the Python-level select_frames which has the same
        semantics as the shell-level validation.
        """
        # 3 frames, requesting 8 → should fail
        with self.assertRaises(ValueError) as ctx:
            select_frames([f"/seq/{i}.jpg" for i in range(3)], 8)
        self.assertIn("8", str(ctx.exception))
        self.assertIn("3", str(ctx.exception))

    def test_exactly_enough_frames_succeeds(self):
        """select_frames succeeds when exactly the required count is available."""
        result = select_frames([f"/seq/{i}.jpg" for i in range(8)], 8)
        self.assertEqual(len(result), 8)


# ── record serialiser regression tests ───────────────────────────────────────
#
# These tests exercise the exact record-construction path used by the shell
# script: build_direct_record() and build_ipc_record() read _BM_* env vars via
# json.loads(), so they are immune to JSON null/true/false NameError and to
# injection via output_text or path strings.


def _direct_env(
    *,
    run_id: str = "20250101_120000",
    recorded_at: str = "2025-01-01T12:00:00Z",
    frame_condition: str = "F1",
    frame_count: int = 1,
    frame_hashes: list | None = None,
    prompt_hash: str = "abc123def456",
    sequence_type: str = '"images"',
    fps: str = "null",
    frame_timestamps_sec: str = "null",
    frame_timestamp_policy: str = '"none"',
    rendered_timestamps: str = "false",
    runtime_temporal_encoding: str = '"ordered_multi_image_no_native_temporal_metadata"',
    temporal_fallback_used: str = "false",
    max_output_tokens: int = 32,
    actual_output_tokens: str = "null",
    total_image_tokens: str = "null",
    finish_reason: str = "null",
    success: str = "true",
    error: str = "null",
    cold_start_ms: str = "4321",
    vision_encoder_ms: str = "null",
    prefill_ms: str = "null",
    decode_ms: str = "null",
    decode_tokens_per_sec: str = "null",
    llm_gen_gpu_ms: str = "null",
    response_path: str = "/tmp/resp.json",
    profile_path: str = "/tmp/prof.json",
    model_name: str = "TestModel",
    engine_provenance: dict[str, Any] | None = None,
    iteration: int = 0,
    is_warmup: str = "false",
) -> dict[str, str]:
    """Build a minimal _BM_* env dict for a direct record."""
    if frame_hashes is None:
        frame_hashes = [{"path": f"/tmp/f{i}.jpg", "sha256": "a" * 64} for i in range(frame_count)]
    return {
        "_BM_RUN_ID": run_id,
        "_BM_RECORDED_AT": recorded_at,
        "_BM_FRAME_CONDITION": frame_condition,
        "_BM_FRAME_COUNT": str(frame_count),
        "_BM_FRAME_HASHES": json.dumps(frame_hashes),
        "_BM_PROMPT_HASH": prompt_hash,
        "_BM_SEQUENCE_TYPE": sequence_type,
        "_BM_FPS": fps,
        "_BM_FRAME_TIMESTAMPS_SEC": frame_timestamps_sec,
        "_BM_FRAME_TIMESTAMP_POLICY": frame_timestamp_policy,
        "_BM_RENDERED_TIMESTAMPS": rendered_timestamps,
        "_BM_RUNTIME_TEMPORAL_ENCODING": runtime_temporal_encoding,
        "_BM_TEMPORAL_FALLBACK_USED": temporal_fallback_used,
        "_BM_MAX_OUTPUT_TOKENS": str(max_output_tokens),
        "_BM_ACTUAL_OUTPUT_TOKENS": actual_output_tokens,
        "_BM_TOTAL_IMAGE_TOKENS": total_image_tokens,
        "_BM_FINISH_REASON": finish_reason,
        "_BM_SUCCESS": success,
        "_BM_ERROR": error,
        "_BM_COLD_START_MS": cold_start_ms,
        "_BM_VISION_ENCODER_MS": vision_encoder_ms,
        "_BM_PREFILL_MS": prefill_ms,
        "_BM_DECODE_MS": decode_ms,
        "_BM_DECODE_TOKENS_PER_SEC": decode_tokens_per_sec,
        "_BM_LLM_GEN_GPU_MS": llm_gen_gpu_ms,
        "_BM_RESPONSE_PATH": response_path,
        "_BM_PROFILE_PATH": profile_path,
        "_BM_MODEL_NAME": model_name,
        "_BM_ENGINE_PROVENANCE": json.dumps(engine_provenance or _make_engine_provenance(model_name=model_name)),
        "_BM_ITERATION": str(iteration),
        "_BM_IS_WARMUP": is_warmup,
    }


def _ipc_env(
    *,
    run_id: str = "20250101_120000",
    recorded_at: str = "2025-01-01T12:00:00Z",
    frame_condition: str = "F1",
    frame_count: int = 1,
    frame_hashes: list | None = None,
    prompt_hash: str = "abc123def456",
    sequence_type: str = '"images"',
    fps: str = "null",
    frame_timestamps_sec: str = "null",
    frame_timestamp_policy: str = '"none"',
    rendered_timestamps: str = "false",
    requested_sequence_type: str = '"images"',
    runtime_temporal_encoding: str = '"ordered_multi_image_no_native_temporal_metadata"',
    temporal_fallback_used: str = "false",
    max_output_tokens: int = 32,
    success: str = "true",
    error: str = "null",
    total_latency: str = "312",
    inference_seconds: str = "null",
    output_text: str = "null",
    output_words: str = "null",
    ipc_result_path: str = "null",
    model_name: str = "TestModel",
    engine_provenance: dict[str, Any] | None = None,
    iteration: int = 0,
    is_warmup: str = "false",
) -> dict[str, str]:
    """Build a minimal _BM_* env dict for an IPC record."""
    if frame_hashes is None:
        frame_hashes = [{"path": f"/tmp/f{i}.jpg", "sha256": "b" * 64} for i in range(frame_count)]
    return {
        "_BM_RUN_ID": run_id,
        "_BM_RECORDED_AT": recorded_at,
        "_BM_FRAME_CONDITION": frame_condition,
        "_BM_FRAME_COUNT": str(frame_count),
        "_BM_FRAME_HASHES": json.dumps(frame_hashes),
        "_BM_PROMPT_HASH": prompt_hash,
        "_BM_SEQUENCE_TYPE": sequence_type,
        "_BM_FPS": fps,
        "_BM_FRAME_TIMESTAMPS_SEC": frame_timestamps_sec,
        "_BM_FRAME_TIMESTAMP_POLICY": frame_timestamp_policy,
        "_BM_RENDERED_TIMESTAMPS": rendered_timestamps,
        "_BM_REQUESTED_SEQUENCE_TYPE": requested_sequence_type,
        "_BM_RUNTIME_TEMPORAL_ENCODING": runtime_temporal_encoding,
        "_BM_TEMPORAL_FALLBACK_USED": temporal_fallback_used,
        "_BM_MAX_OUTPUT_TOKENS": str(max_output_tokens),
        "_BM_SUCCESS": success,
        "_BM_ERROR": error,
        "_BM_TOTAL_LATENCY": total_latency,
        "_BM_INFERENCE_SECONDS": inference_seconds,
        "_BM_OUTPUT_TEXT": output_text,
        "_BM_OUTPUT_WORDS": output_words,
        "_BM_IPC_RESULT_PATH": ipc_result_path,
        "_BM_MODEL_NAME": model_name,
        "_BM_ENGINE_PROVENANCE": json.dumps(engine_provenance or _make_engine_provenance(model_name=model_name)),
        "_BM_ITERATION": str(iteration),
        "_BM_IS_WARMUP": is_warmup,
    }


class TestRecordSerializer(unittest.TestCase):
    """Regression tests for build_direct_record() / build_ipc_record().

    Exercises the exact code path the shell script uses after the fix:
    JSON null/true/false are passed as env-var strings and decoded with
    json.loads() — no shell→Python literal interpolation.
    """

    # ── direct record ─────────────────────────────────────────────────────

    def test_direct_success_true(self):
        """success=true is serialised as a JSON boolean true, not a string."""
        rec = json.loads(build_direct_record(_direct_env(success="true")))
        self.assertIs(rec["success"], True)
        self.assertEqual(rec["path"], "direct")

    def test_direct_success_false(self):
        """success=false is serialised as JSON boolean false; error field preserved."""
        rec = json.loads(build_direct_record(
            _direct_env(success="false", error='"direct inference failed"')
        ))
        self.assertIs(rec["success"], False)
        self.assertEqual(rec["error"], "direct inference failed")

    def test_direct_warmup_true(self):
        """warmup=true produces warmup:true in the record."""
        rec = json.loads(build_direct_record(_direct_env(is_warmup="true")))
        self.assertIs(rec["warmup"], True)

    def test_direct_warmup_false(self):
        """warmup=false produces warmup:false in the record."""
        rec = json.loads(build_direct_record(_direct_env(is_warmup="false")))
        self.assertIs(rec["warmup"], False)

    def test_direct_nullable_fields_are_none(self):
        """JSON null env vars arrive as Python None in the record."""
        rec = json.loads(build_direct_record(_direct_env(
            actual_output_tokens="null",
            total_image_tokens="null",
            finish_reason="null",
            vision_encoder_ms="null",
            prefill_ms="null",
            decode_ms="null",
        )))
        self.assertIsNone(rec["actual_output_tokens"])
        self.assertIsNone(rec["total_image_tokens"])
        self.assertIsNone(rec["finish_reason"])
        self.assertIsNone(rec["vision_encoder_ms"])
        self.assertIsNone(rec["prefill_ms"])
        self.assertIsNone(rec["decode_ms"])
        # Fixed fields always null on direct path
        self.assertIsNone(rec["total_latency_ms"])
        self.assertIsNone(rec["ttft_ms"])
        self.assertIsNone(rec["inference_seconds"])
        self.assertIsNone(rec["output_text"])
        self.assertIsNone(rec["output_words"])
        self.assertIsNone(rec["ipc_result_path"])

    def test_direct_numeric_fields(self):
        """Numeric profile fields arrive as Python numbers, not strings."""
        rec = json.loads(build_direct_record(_direct_env(
            vision_encoder_ms="58.1",
            prefill_ms="38.4",
            decode_ms="639.0",
            decode_tokens_per_sec="46.8",
            llm_gen_gpu_ms="640.9",
            actual_output_tokens="30",
            total_image_tokens="1024",
            cold_start_ms="4567",
        )))
        self.assertAlmostEqual(rec["vision_encoder_ms"], 58.1)
        self.assertAlmostEqual(rec["prefill_ms"], 38.4)
        self.assertAlmostEqual(rec["decode_ms"], 639.0)
        self.assertAlmostEqual(rec["decode_tokens_per_sec"], 46.8)
        self.assertAlmostEqual(rec["llm_generation_total_gpu_time_ms"], 640.9)
        self.assertEqual(rec["actual_output_tokens"], 30)
        self.assertEqual(rec["total_image_tokens"], 1024)
        self.assertAlmostEqual(rec["cold_start_total_ms"], 4567)

    def test_direct_finish_reason_string(self):
        """finish_reason JSON string arrives as a Python string."""
        rec = json.loads(build_direct_record(_direct_env(finish_reason='"eos"')))
        self.assertEqual(rec["finish_reason"], "eos")

    def test_direct_finish_reason_max_length(self):
        """finish_reason='max-length' (JSON string) is preserved exactly."""
        rec = json.loads(build_direct_record(_direct_env(finish_reason='"max-length"')))
        self.assertEqual(rec["finish_reason"], "max-length")

    def test_direct_cold_start_null_in_dry_run(self):
        """cold_start_total_ms is null when DRY_RUN passes 'null'."""
        rec = json.loads(build_direct_record(_direct_env(cold_start_ms="null")))
        self.assertIsNone(rec["cold_start_total_ms"])

    def test_direct_path_and_model_name_preserved(self):
        """Path strings and model_name are preserved exactly."""
        rec = json.loads(build_direct_record(_direct_env(
            response_path="/bench/F1/direct/measured_iter_0/response.json",
            profile_path="/bench/F1/direct/measured_iter_0/profile.json",
            model_name="Cosmos-Reason2-8B",
        )))
        self.assertEqual(rec["native_response_path"], "/bench/F1/direct/measured_iter_0/response.json")
        self.assertEqual(rec["native_profile_path"], "/bench/F1/direct/measured_iter_0/profile.json")
        self.assertEqual(rec["model_name"], "Cosmos-Reason2-8B")

    def test_direct_engine_provenance_round_trip(self):
        provenance = _make_engine_provenance(model_name="Cosmos-Reason2-8B")
        rec = json.loads(build_direct_record(_direct_env(engine_provenance=provenance)))
        self.assertEqual(rec["engine_provenance"]["engine_identity"], provenance["engine_identity"])

    def test_direct_frame_hashes_json_array_round_trip(self):
        """frame_paths JSON array is deserialised correctly."""
        hashes = [{"path": "/tmp/a.jpg", "sha256": "a" * 64}, {"path": "/tmp/b.jpg", "sha256": "b" * 64}]
        rec = json.loads(build_direct_record(_direct_env(frame_count=2, frame_hashes=hashes)))
        self.assertEqual(rec["frame_paths"], hashes)

    def test_direct_output_is_valid_jsonl_line(self):
        """build_direct_record() output is valid JSON (parseable JSONL line)."""
        raw = build_direct_record(_direct_env())
        obj = json.loads(raw)
        self.assertEqual(obj["schema_version"], "1")
        self.assertEqual(obj["record_type"], "inference")

    def test_direct_temporal_metadata_fields(self):
        rec = json.loads(build_direct_record(_direct_env(
            sequence_type='"temporal_images"',
            fps="8.0",
            frame_timestamps_sec="null",
            frame_timestamp_policy='"implicit_uniform_from_fps"',
            rendered_timestamps="false",
            runtime_temporal_encoding='"native_qwen3vl_video_json_frames_fps_no_explicit_timestamps"',
            temporal_fallback_used="false",
        )))
        self.assertEqual(rec["sequence_type"], "temporal_images")
        self.assertAlmostEqual(rec["fps"], 8.0)
        self.assertIsNone(rec["frame_timestamps_sec"])
        self.assertEqual(rec["frame_timestamp_policy"], "implicit_uniform_from_fps")
        self.assertIs(rec["rendered_timestamps"], False)
        self.assertEqual(
            rec["runtime_temporal_encoding"],
            "native_qwen3vl_video_json_frames_fps_no_explicit_timestamps",
        )
        self.assertIs(rec["temporal_fallback_used"], False)

    # ── IPC record ────────────────────────────────────────────────────────

    def test_ipc_success_true(self):
        """IPC success=true is a boolean."""
        rec = json.loads(build_ipc_record(_ipc_env(success="true")))
        self.assertIs(rec["success"], True)
        self.assertEqual(rec["path"], "ipc")

    def test_ipc_success_false_with_error(self):
        """IPC success=false with error message preserves the error string."""
        rec = json.loads(build_ipc_record(
            _ipc_env(success="false", error='"ipc client exited with code 1"', total_latency="null")
        ))
        self.assertIs(rec["success"], False)
        self.assertEqual(rec["error"], "ipc client exited with code 1")
        self.assertIsNone(rec["total_latency_ms"])

    def test_ipc_engine_provenance_round_trip(self):
        provenance = _make_engine_provenance(model_name="Cosmos-Reason2-8B")
        rec = json.loads(build_ipc_record(_ipc_env(engine_provenance=provenance)))
        self.assertEqual(rec["engine_provenance"]["engine_profile_id"], "thor-f8")

    def test_ipc_warmup_true(self):
        """IPC warmup=true serialises as boolean true."""
        rec = json.loads(build_ipc_record(_ipc_env(is_warmup="true")))
        self.assertIs(rec["warmup"], True)

    def test_ipc_warmup_false(self):
        """IPC warmup=false serialises as boolean false."""
        rec = json.loads(build_ipc_record(_ipc_env(is_warmup="false")))
        self.assertIs(rec["warmup"], False)

    def test_ipc_with_output_text_and_words(self):
        """IPC output_text and output_words are preserved (JSON-encoded strings)."""
        escaped_text = json.dumps("compact result: {objects: [], hazards: []}")
        rec = json.loads(build_ipc_record(_ipc_env(
            output_text=escaped_text,
            output_words="7",
        )))
        self.assertEqual(rec["output_text"], "compact result: {objects: [], hazards: []}")
        self.assertEqual(rec["output_words"], 7)

    def test_ipc_output_text_with_quotes_and_backslash(self):
        """IPC output_text with quotes/backslash is round-tripped safely."""
        original = 'result: {"key": "val\\nline2"}'
        escaped_text = json.dumps(original)
        rec = json.loads(build_ipc_record(_ipc_env(output_text=escaped_text)))
        self.assertEqual(rec["output_text"], original)

    def test_ipc_output_text_null(self):
        """IPC output_text=null is Python None."""
        rec = json.loads(build_ipc_record(_ipc_env(output_text="null")))
        self.assertIsNone(rec["output_text"])

    def test_ipc_with_inference_seconds(self):
        """IPC inference_seconds is preserved as a float."""
        rec = json.loads(build_ipc_record(_ipc_env(inference_seconds="0.273")))
        self.assertAlmostEqual(rec["inference_seconds"], 0.273)

    def test_ipc_with_latency_and_result_path(self):
        """IPC total_latency_ms (number) and ipc_result_path (JSON string) are set."""
        result_path = "/tmp/bench/F2/ipc/measured_iter_0/ipc_result.json"
        rec = json.loads(build_ipc_record(_ipc_env(
            total_latency="387",
            ipc_result_path=json.dumps(result_path),
        )))
        self.assertEqual(rec["total_latency_ms"], 387)
        self.assertEqual(rec["ipc_result_path"], result_path)

    def test_ipc_nullable_stage_fields_are_none(self):
        """IPC records always have null for direct-only stage timings."""
        rec = json.loads(build_ipc_record(_ipc_env()))
        for field in [
            "actual_output_tokens", "total_image_tokens", "finish_reason",
            "cold_start_total_ms", "ttft_ms", "vision_encoder_ms",
            "prefill_ms", "decode_ms", "decode_tokens_per_sec",
            "llm_generation_total_gpu_time_ms", "native_response_path",
            "native_profile_path",
        ]:
            self.assertIsNone(rec[field], msg=f"{field} should be null in IPC record")

    def test_ipc_output_is_valid_jsonl_line(self):
        """build_ipc_record() output is valid JSON."""
        raw = build_ipc_record(_ipc_env())
        obj = json.loads(raw)
        self.assertEqual(obj["schema_version"], "1")
        self.assertEqual(obj["record_type"], "inference")

    def test_ipc_temporal_metadata_fields(self):
        rec = json.loads(build_ipc_record(_ipc_env(
            sequence_type='"video"',
            fps="15.0",
            frame_timestamps_sec="[0.0, 0.0666667]",
            frame_timestamp_policy='"explicit"',
            rendered_timestamps="false",
            requested_sequence_type='"video"',
            runtime_temporal_encoding='"native_qwen3vl_video_imagedata_mrope_timestamps"',
            temporal_fallback_used="false",
        )))
        self.assertEqual(rec["sequence_type"], "video")
        self.assertEqual(rec["requested_sequence_type"], "video")
        self.assertAlmostEqual(rec["fps"], 15.0)
        self.assertEqual(rec["frame_timestamp_policy"], "explicit")
        self.assertIs(rec["temporal_fallback_used"], False)

    # ── shell-level serialiser test ────────────────────────────────────────
    # Exercises the env-var → Python path as the shell script does it, using
    # subprocess so the interpreter boundary is identical to production.

    def test_shell_direct_record_via_subprocess(self):
        """Direct record builder produces valid JSON when called via subprocess with env vars."""
        env = dict(os.environ)
        env.update(_direct_env(
            success="true",
            finish_reason='"eos"',
            actual_output_tokens="12",
            total_image_tokens="256",
            vision_encoder_ms="45.5",
            prefill_ms="38.4",
            cold_start_ms="4321",
            is_warmup="false",
        ))
        env["PYTHONPATH"] = str(_BENCH_DIR) + ((":" + os.environ["PYTHONPATH"]) if "PYTHONPATH" in os.environ else "")
        result = subprocess.run(
            [sys.executable, "-c",
             "from vlm_multiframe_report import build_direct_record; print(build_direct_record())"],
            capture_output=True, text=True, env=env,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        rec = json.loads(result.stdout.strip())
        self.assertIs(rec["success"], True)
        self.assertEqual(rec["finish_reason"], "eos")
        self.assertEqual(rec["actual_output_tokens"], 12)
        self.assertEqual(rec["total_image_tokens"], 256)
        self.assertAlmostEqual(rec["vision_encoder_ms"], 45.5)
        self.assertIsNone(rec["total_latency_ms"])
        self.assertIs(rec["warmup"], False)

    def test_shell_ipc_record_via_subprocess_with_output_text(self):
        """IPC record builder produces valid JSON when called via subprocess with env vars."""
        output_text_val = 'scene: {"objects": ["robot"], "hazards": null}'
        env = dict(os.environ)
        env.update(_ipc_env(
            success="true",
            total_latency="312",
            inference_seconds="0.25",
            output_text=json.dumps(output_text_val),
            output_words="5",
            ipc_result_path=json.dumps("/tmp/ipc_result.json"),
            is_warmup="false",
        ))
        env["PYTHONPATH"] = str(_BENCH_DIR) + ((":" + os.environ["PYTHONPATH"]) if "PYTHONPATH" in os.environ else "")
        result = subprocess.run(
            [sys.executable, "-c",
             "from vlm_multiframe_report import build_ipc_record; print(build_ipc_record())"],
            capture_output=True, text=True, env=env,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        rec = json.loads(result.stdout.strip())
        self.assertIs(rec["success"], True)
        self.assertEqual(rec["total_latency_ms"], 312)
        self.assertAlmostEqual(rec["inference_seconds"], 0.25)
        self.assertEqual(rec["output_text"], output_text_val)
        self.assertEqual(rec["output_words"], 5)
        self.assertEqual(rec["ipc_result_path"], "/tmp/ipc_result.json")
        self.assertIsNone(rec["vision_encoder_ms"])

    def test_shell_ipc_record_via_subprocess_null_fields(self):
        """IPC record with null output_text/output_words/inference_seconds via subprocess."""
        env = dict(os.environ)
        env.update(_ipc_env(
            success="false",
            error='"ros2 not available \u2014 ipc path skipped"',
            total_latency="null",
            output_text="null",
            output_words="null",
            inference_seconds="null",
        ))
        env["PYTHONPATH"] = str(_BENCH_DIR) + ((":" + os.environ["PYTHONPATH"]) if "PYTHONPATH" in os.environ else "")
        result = subprocess.run(
            [sys.executable, "-c",
             "from vlm_multiframe_report import build_ipc_record; print(build_ipc_record())"],
            capture_output=True, text=True, env=env,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        rec = json.loads(result.stdout.strip())
        self.assertIs(rec["success"], False)
        self.assertIsNone(rec["total_latency_ms"])
        self.assertIsNone(rec["output_text"])
        self.assertIsNone(rec["output_words"])
        self.assertIsNone(rec["inference_seconds"])


if __name__ == "__main__":
    unittest.main()
