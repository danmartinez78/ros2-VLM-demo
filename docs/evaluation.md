# Task-level evaluation harness

Issue [#11](https://github.com/danmartinez78/ros2-VLM-demo/issues/11) adds a repeatable way to score reasoning quality, not just runtime success.

## Files

- Dataset definition: `scripts/evaluation/dataset_v1.json`
- Evaluation runner: `scripts/evaluation/evaluate_task_harness.py`
- Unit tests: `scripts/evaluation/test_evaluate_task_harness.py`

## Dataset format

Datasets are versioned and machine-readable.

```json
{
  "dataset_id": "thor-image-proc-task-eval",
  "version": "v1",
  "examples": [
    {
      "id": "aisle-static-scene-001",
      "segment": {"bag": "image-proc", "start_seconds": 12.0, "end_seconds": 14.0},
      "rubrics": {
        "object_presence": {"required_observations": [["pallet", "box"]]},
        "hazard_recognition": {"required_observations": [["clear", "no immediate hazard"]]},
        "scene_state": {"scoring": "human"}
      },
      "unsupported_claim_terms": ["fire", "smoke"]
    }
  ]
}
```

Rubric checks are not exact-string equality. `required_observations` supports synonym alternatives by using nested arrays. Set `"scoring": "human"` when reliable automatic judgment is not possible.

## Run-input format

Provide per-example outputs from a model run with the full configuration attached:

```json
{
  "run_id": "run-2026-07-31",
  "mode": "regression",
  "dataset_id": "thor-image-proc-task-eval",
  "configuration": {
    "model": "Cosmos-Reason2-8B",
    "engine": "nvfp4",
    "prompt": "Describe objects and hazards",
    "decoding": {"temperature": 0.2, "top_p": 0.9, "top_k": 20, "max_generate_length": 256},
    "preprocessing": {"image_max_width": 1280, "jpeg_quality": 90}
  },
  "results": [
    {"example_id": "aisle-static-scene-001", "success": true, "latency_seconds": 1.45, "response": "..."}
  ]
}
```

## Usage

Single-run evaluation:

```bash
python3 scripts/evaluation/evaluate_task_harness.py \
  --dataset scripts/evaluation/dataset_v1.json \
  --run /absolute/path/to/run.json \
  --output /absolute/path/to/eval-report.json
```

Two-run comparison on the same dataset:

```bash
python3 scripts/evaluation/evaluate_task_harness.py \
  --dataset scripts/evaluation/dataset_v1.json \
  --run /absolute/path/to/baseline.json \
  --compare-run /absolute/path/to/candidate.json \
  --output /absolute/path/to/eval-comparison.json
```

## Output artifacts

The output JSON includes:

- per-example: correctness (or `null` when human review required), unsupported claims, failures, latency, and rubric-level evidence
- aggregate: correctness rate, unsupported-claim rate, failure rate, latency summary, and rubric summary
- comparison deltas (when `--compare-run` is provided)

Use `mode: regression` for deterministic checks and `mode: exploratory` for model/parameter experiments.

## Human review workflow

Any example with a rubric marked `"scoring": "human"` is flagged as `human_review_required: true` and excluded from automatic correctness-rate denominator.
