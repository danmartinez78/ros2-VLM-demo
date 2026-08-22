"""CPU-only tests for Thor full-pipeline benchmark report utilities."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_BENCH_DIR))

from thor_pipeline_benchmark_report import (  # noqa: E402
    compare_runs,
    parse_tegrastats_log,
    parse_topic_hz_log,
    summarize_run,
)


class TestParseTegraStats(unittest.TestCase):
    def test_extracts_expected_metrics_desktop_style(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tegrastats.log"
            path.write_text(
                """
RAM 24300/125700MB CPU [98%@2016,25%@2016,17%@2016] EMC_FREQ 63%@4266 GR3D_FREQ 99%@1574 VDD_IN 92000mW GPU@60C tj@61C
RAM 24400/125700MB CPU [95%@2016,21%@2016,18%@2016] EMC_FREQ 64%@4266 GR3D_FREQ 97%@1574 VDD_IN 90000mW GPU@61C tj@62C
""".strip()
                + "\n",
                encoding="utf-8",
            )
            summary = parse_tegrastats_log(path)
            self.assertEqual(summary["samples"], 2)
            self.assertAlmostEqual(summary["emc_pct"]["mean"], 63.5, places=1)
            self.assertAlmostEqual(summary["gr3d_pct"]["mean"], 98.0, places=1)
            self.assertGreaterEqual(summary["cpu_hottest_core_p95_pct"], 95.0)
            self.assertAlmostEqual(summary["module_power_w"]["mean"], 91.0, places=1)

    def test_extracts_expected_metrics_thor_style(self):
        fixture = _BENCH_DIR / "test_fixtures" / "tegrastats_thor_sample.log"
        summary = parse_tegrastats_log(fixture)
        self.assertEqual(summary["samples"], 2)
        self.assertIsNone(summary["gr3d_pct"]["mean"])
        self.assertAlmostEqual(summary["gr3d_mhz"]["mean"], 1574.0, places=1)
        self.assertAlmostEqual(summary["module_power_w"]["mean"], 94.064, places=3)
        self.assertAlmostEqual(summary["power_rails_w"]["VDD_GPU"]["mean"], 2.65, places=2)
        self.assertAlmostEqual(summary["power_rails_w"]["VIN_SYS_5V0"]["mean"], 1.35, places=2)


class TestParseTopicHz(unittest.TestCase):
    def test_last_average_rate_is_selected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "topic_hz.log"
            path.write_text(
                "average rate: 7.10\naverage rate: 6.90\n",
                encoding="utf-8",
            )
            rate = parse_topic_hz_log(path)
            self.assertAlmostEqual(rate or 0.0, 6.9, places=2)


class TestSummarizeAndCompare(unittest.TestCase):
    def test_summarize_run_and_recommendation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for mode, emc, gr3d, cpu in (("D", 60, 90, 95), ("E", 45, 70, 80), ("F", 40, 60, 70)):
                run_dir = root / f"run_{mode}"
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "run_config.json").write_text(
                    json.dumps({"mode": mode, "description": mode, "run_id": f"run_{mode}"}) + "\n",
                    encoding="utf-8",
                )
                (run_dir / "tegrastats.log").write_text(
                    f"RAM 1000/8000MB CPU [{cpu}%@2000,10%@2000] EMC_FREQ {emc}%@4266 GR3D_FREQ {gr3d}%@1574 VDD_IN 80000mW GPU@55C tj@58C\n",
                    encoding="utf-8",
                )
                (run_dir / "detections_hz.log").write_text("average rate: 8.0\n", encoding="utf-8")
                (run_dir / "tracked_observation_hz.log").write_text("average rate: 8.0\n", encoding="utf-8")
                (run_dir / "vlm_result_hz.log").write_text("average rate: 1.0\n", encoding="utf-8")

            runs = [summarize_run(root / "run_D"), summarize_run(root / "run_E"), summarize_run(root / "run_F")]
            report = compare_runs(runs)
            self.assertEqual(report["recommendation"]["recommended_mode"], "F")
            self.assertTrue("cpu_single_core_hotspot_present" in report["findings"])

    def test_recommendation_marks_unavailable_metrics(self):
        runs = [
            {
                "mode": "D",
                "description": "D",
                "tegrastats": {
                    "emc_pct": {"mean": 55.0},
                    "gr3d_pct": {"mean": None},
                    "gr3d_mhz": {"mean": None},
                    "cpu_hottest_core_p95_pct": 90.0,
                },
            },
            {
                "mode": "E",
                "description": "E",
                "tegrastats": {
                    "emc_pct": {"mean": 50.0},
                    "gr3d_pct": {"mean": 70.0},
                    "gr3d_mhz": {"mean": 1500.0},
                    "cpu_hottest_core_p95_pct": 80.0,
                },
            },
        ]
        report = compare_runs(runs)
        recommendation = report["recommendation"]
        self.assertEqual(recommendation["recommended_mode"], "E")
        self.assertIn("E", recommendation["ranked_modes"])
        self.assertIn("D", recommendation["unavailable_modes"])

    def test_recommendation_withholds_when_gr3d_units_mixed(self):
        runs = [
            {
                "mode": "D",
                "description": "D",
                "tegrastats": {
                    "emc_pct": {"mean": 50.0},
                    "gr3d_pct": {"mean": 80.0},
                    "gr3d_mhz": {"mean": 1600.0},
                    "cpu_hottest_core_p95_pct": 85.0,
                },
            },
            {
                "mode": "E",
                "description": "E",
                "tegrastats": {
                    "emc_pct": {"mean": 45.0},
                    "gr3d_pct": {"mean": None},
                    "gr3d_mhz": {"mean": 1574.0},
                    "cpu_hottest_core_p95_pct": 80.0,
                },
            },
        ]
        recommendation = compare_runs(runs)["recommendation"]
        self.assertIsNone(recommendation["recommended_mode"])
        self.assertEqual(recommendation["ranked_modes"], [])

    def test_invalid_modes_excluded_from_recommendation(self):
        runs = [
            {
                "mode": "D",
                "description": "D",
                "tegrastats": {
                    "emc_pct": {"mean": 40.0},
                    "gr3d_pct": {"mean": 65.0},
                    "cpu_hottest_core_p95_pct": 70.0,
                },
                "rates_hz": {"detections": 8.0, "tracked_observation": 8.0, "vlm_result": 2.0},
                "vlm": {"result_frames": 10},
                "validation": {"is_valid": True},
            },
            {
                "mode": "E",
                "description": "E",
                "tegrastats": {
                    "emc_pct": {"mean": 38.0},
                    "gr3d_pct": {"mean": 60.0},
                    "cpu_hottest_core_p95_pct": 68.0,
                },
                "rates_hz": {"detections": None, "tracked_observation": None, "vlm_result": None},
                "vlm": {"result_frames": 0},
                "validation": {"is_valid": False},
            },
            {
                "mode": "F",
                "description": "F",
                "tegrastats": {
                    "emc_pct": {"mean": 35.0},
                    "gr3d_pct": {"mean": 55.0},
                    "cpu_hottest_core_p95_pct": 65.0,
                },
                "rates_hz": {"detections": 8.0, "tracked_observation": 8.0, "vlm_result": 1.0},
                "vlm": {"result_frames": 8},
                "validation": {"is_valid": True},
            },
        ]
        recommendation = compare_runs(runs)["recommendation"]
        self.assertEqual(recommendation["recommended_mode"], "F")
        self.assertNotIn("E", recommendation["ranked_modes"])
        self.assertIn("E", recommendation["invalid_modes"])

    def test_summarize_marks_full_pipeline_run_invalid_when_signals_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run_C"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "run_config.json").write_text(
                json.dumps({"mode": "C", "description": "C", "run_id": "run_C"}) + "\n",
                encoding="utf-8",
            )
            (run_dir / "tegrastats.log").write_text(
                "RAM 1000/8000MB CPU [10%@2000] VIN 90000mW GPU@50C\n",
                encoding="utf-8",
            )
            (run_dir / "ros_metrics.json").write_text(
                json.dumps({"aggregate": {"successful_frames": 0, "failed_frames": 0, "total_dropped": 0}}) + "\n",
                encoding="utf-8",
            )
            summary = summarize_run(run_dir)
            self.assertFalse(summary["validation"]["is_valid"])
            self.assertIn("detections", summary["validation"]["missing_signals"])
            self.assertIn("tracked_observation", summary["validation"]["missing_signals"])
            self.assertIn("vlm_output", summary["validation"]["missing_signals"])
            self.assertTrue(summary["telemetry"]["degraded"])


class TestThorRunnerDryRun(unittest.TestCase):
    def test_dry_run_prints_matrix_commands(self):
        script = _BENCH_DIR / "run_thor_pipeline_benchmarks.sh"
        result = subprocess.run(
            [
                "bash",
                str(script),
                "--rosbag-path", "/tmp/fake_bag",
                "--modes", "A,F",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        combined = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, combined)
        self.assertIn("Mode A", combined)
        self.assertIn("Mode F", combined)
        self.assertIn("start_vlm:=false", combined)
        self.assertIn("/image_rect:=/camera0/color/image_raw", combined)
        self.assertIn("/camera_info_rect:=/camera_info", combined)
        remap_lines = [line for line in combined.splitlines() if "ros2 bag play" in line]
        self.assertTrue(remap_lines, combined)
        self.assertEqual(remap_lines[0].count("--remap"), 1, remap_lines[0])
        self.assertIn("comparison_report.json", combined)


if __name__ == "__main__":
    unittest.main()
