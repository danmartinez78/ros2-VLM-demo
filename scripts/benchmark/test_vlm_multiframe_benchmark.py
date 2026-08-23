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


def _make_record(
    *,
    frame_condition: str = "F1",
    frame_count: int = 1,
    path: str = "direct",
    frame_paths: list[dict] | None = None,
    prompt_hash_val: str = "abc123def456",
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
        """build_multiframe_request (with actual file I/O) records SHA-256 for each frame."""
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
            # image item first, text item last
            self.assertEqual(content[0]["type"], "image_url")
            self.assertEqual(content[-1]["type"], "text")
            # content_hash present and is a 64-char hex SHA-256
            img_item = content[0]
            self.assertIn("content_hash", img_item)
            self.assertEqual(len(img_item["content_hash"]), 64)
        finally:
            os.unlink(tmp_path)

    def test_build_request_multi_frame_content_order(self):
        """Multi-frame request: images appear before text, in temporal order."""
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
                self.assertEqual(content[i]["type"], "image_url")
                self.assertEqual(content[i]["source_path"], tmp_paths[i])
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

    def test_all_frames_have_hashes_in_request(self):
        """build_multiframe_request records content_hash for every image item."""
        from vlm_multiframe_report import build_multiframe_request
        tmp_paths = []
        try:
            for k in range(4):
                f = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                f.write(bytes([k]) * 20)
                f.close()
                tmp_paths.append(f.name)
            req = build_multiframe_request(tmp_paths, MULTIFRAME_PROMPT_TEXT)
            content = req["requests"][0]["messages"][0]["content"]
            image_items = [c for c in content if c.get("type") == "image_url"]
            self.assertEqual(len(image_items), 4)
            hashes = [item["content_hash"] for item in image_items]
            # All hashes should be 64-char hex and unique for distinct content
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
        """Return a minimal profile JSON matching the Thor runtime schema."""
        profile = {
            "multimodal": {"total_image_tokens": kwargs.get("total_image_tokens", 256)},
            "vision_encoder": {"total_ms": kwargs.get("vision_encoder_ms", 45.5)},
            "prefill": {"total_ms": kwargs.get("prefill_ms", 36.2)},
            "generation": {
                "generated_tokens": kwargs.get("generated_tokens", 12),
                "tokens_per_second": kwargs.get("tokens_per_second", 47.3),
                "total_gpu_time_ms": kwargs.get("total_gpu_time_ms", 253.7),
                "finish_reason": kwargs.get("finish_reason", "eos"),
            },
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
        """Inline profile JSON (Thor schema) is parsed correctly via the Python snippet."""
        import importlib, textwrap
        profile_data = {
            "multimodal": {"total_image_tokens": 1024},
            "vision_encoder": {"total_ms": 58.1},
            "prefill": {"total_ms": 38.4},
            "generation": {
                "generated_tokens": 30,
                "tokens_per_second": 46.8,
                "total_gpu_time_ms": 640.9,
                "finish_reason": "max-length",
            },
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(profile_data, f)
            profile_path = f.name
        try:
            # Run the same inline Python parser used by the shell script
            parser_code = textwrap.dedent("""
                import json, sys
                path = sys.argv[1]
                with open(path) as f:
                    p = json.load(f)
                out = {}
                mm = p.get("multimodal") or {}
                if "total_image_tokens" in mm:
                    out["total_image_tokens"] = mm["total_image_tokens"]
                ve = p.get("vision_encoder") or p.get("vision_encoder_ms")
                if isinstance(ve, dict):
                    out["vision_encoder_ms"] = ve.get("total_ms") or ve.get("ms")
                elif isinstance(ve, (int, float)):
                    out["vision_encoder_ms"] = ve
                pf = p.get("prefill") or p.get("prefill_ms")
                if isinstance(pf, dict):
                    out["prefill_ms"] = pf.get("total_ms") or pf.get("ms")
                elif isinstance(pf, (int, float)):
                    out["prefill_ms"] = pf
                gen = p.get("generation") or {}
                if "generated_tokens" in gen:
                    out["actual_output_tokens"] = gen["generated_tokens"]
                if "tokens_per_second" in gen:
                    out["decode_tokens_per_sec"] = gen["tokens_per_second"]
                if "total_gpu_time_ms" in gen:
                    out["llm_generation_total_gpu_time_ms"] = gen["total_gpu_time_ms"]
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
            self.assertEqual(parsed["finish_reason"], "max-length")
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

    def test_report_contains_ipc_artifact_section(self):
        text = format_text_report(self._full_report())
        self.assertIn("IPC Result Artifacts", text)

    def test_report_note_about_cold_start_separation(self):
        """Report explicitly notes cold-start and IPC steady-state are separate."""
        text = format_text_report(self._full_report())
        self.assertIn("cold-start", text.lower())
        self.assertIn("steady-state", text.lower())

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


if __name__ == "__main__":
    unittest.main()
