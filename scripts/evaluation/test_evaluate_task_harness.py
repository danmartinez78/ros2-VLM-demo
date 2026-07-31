import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import unittest

from evaluate_task_harness import compare_reports, evaluate_run


class EvaluateTaskHarnessTests(unittest.TestCase):
    def setUp(self):
        self.dataset = {
            "dataset_id": "test-dataset",
            "version": "v1",
            "examples": [
                {
                    "id": "example-1",
                    "segment": {"bag": "bag1", "start_seconds": 0, "end_seconds": 1},
                    "rubrics": {
                        "object_presence": {
                            "required_observations": [["pallet", "crate"]]
                        },
                        "scene_state": {"required_observations": [["warehouse"]]},
                    },
                    "unsupported_claim_terms": ["fire"],
                },
                {
                    "id": "example-2",
                    "segment": {"bag": "bag2", "start_seconds": 1, "end_seconds": 2},
                    "rubrics": {
                        "scene_state": {"scoring": "human"},
                    },
                    "unsupported_claim_terms": [],
                },
            ],
        }

    def test_flexible_rubric_matching_and_aggregate(self):
        run = {
            "run_id": "run-1",
            "dataset_id": "test-dataset",
            "mode": "regression",
            "configuration": {"model": "demo"},
            "results": [
                {
                    "example_id": "example-1",
                    "response": "A crate sits in a warehouse aisle.",
                    "success": True,
                    "latency_seconds": 1.2,
                },
                {
                    "example_id": "example-2",
                    "response": "A person is visible.",
                    "success": True,
                    "latency_seconds": 0.8,
                },
            ],
        }

        report = evaluate_run(self.dataset, run)
        self.assertEqual(report["aggregate"]["total_examples"], 2)
        self.assertEqual(report["aggregate"]["correct_examples"], 1)
        self.assertEqual(report["aggregate"]["human_review_required"], 1)
        self.assertAlmostEqual(report["aggregate"]["correctness_rate"], 1.0)

    def test_unsupported_claim_penalty_and_failure_count(self):
        run = {
            "run_id": "run-2",
            "dataset_id": "test-dataset",
            "mode": "regression",
            "configuration": {"model": "demo"},
            "results": [
                {
                    "example_id": "example-1",
                    "response": "There is a pallet and a warehouse fire.",
                    "success": False,
                    "error": "timeout",
                    "latency_seconds": 2.0,
                }
            ],
        }

        report = evaluate_run(self.dataset, run)
        self.assertEqual(report["aggregate"]["failures"], 1)
        self.assertEqual(report["aggregate"]["unsupported_claim_total"], 1)
        self.assertEqual(report["per_example"][0]["unsupported_claim_hits"], ["fire"])
        self.assertFalse(report["per_example"][0]["correctness"])

    def test_comparison_output(self):
        base = evaluate_run(
            self.dataset,
            {
                "run_id": "base",
                "dataset_id": "test-dataset",
                "results": [
                    {
                        "example_id": "example-1",
                        "response": "pallet in warehouse",
                        "success": True,
                        "latency_seconds": 1.0,
                    }
                ],
            },
        )
        candidate = evaluate_run(
            self.dataset,
            {
                "run_id": "candidate",
                "dataset_id": "test-dataset",
                "results": [
                    {
                        "example_id": "example-1",
                        "response": "pallet in warehouse",
                        "success": True,
                        "latency_seconds": 1.5,
                    }
                ],
            },
        )

        comparison = compare_reports(base, candidate)
        self.assertEqual(comparison["dataset_id"], "test-dataset")
        self.assertAlmostEqual(comparison["metric_deltas"]["latency_mean_seconds"], 0.5)


if __name__ == "__main__":
    unittest.main()
