#!/usr/bin/env python3
"""CPU-only tests for observation-history result summarization."""

import importlib.util
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("summarize_observation_history.py")
SPEC = importlib.util.spec_from_file_location("summarize_observation_history", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ObservationHistorySummaryTests(unittest.TestCase):
    def test_parses_success_failure_and_latency(self):
        records = MODULE.parse_records(
            """header:
  frame_id: camera
response: first observation
inference_seconds: 1.5
frame_sequence: 1
success: true
error: ''
---
response: ''
inference_seconds: 0.0
frame_sequence: 2
success: false
error: worker unavailable
---
response: third observation
inference_seconds: 2.5
frame_sequence: 3
success: true
error: ''
---
"""
        )
        summary = MODULE.summarize(records)
        self.assertEqual(summary["result_count"], 3)
        self.assertEqual(summary["success_count"], 2)
        self.assertEqual(summary["failure_count"], 1)
        self.assertAlmostEqual(summary["success_rate"], 2 / 3)
        self.assertEqual(summary["latency_seconds"]["mean"], 2.0)
        self.assertEqual(summary["latency_seconds"]["median"], 2.0)
        self.assertEqual(summary["errors"], ["worker unavailable"])

    def test_empty_input_is_valid(self):
        summary = MODULE.summarize([])
        self.assertEqual(summary["result_count"], 0)
        self.assertEqual(summary["success_rate"], 0.0)
        self.assertIsNone(summary["latency_seconds"])


if __name__ == "__main__":
    unittest.main()
