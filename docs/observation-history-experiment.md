# Observation-history experiment

This experiment measures whether retaining prior semantic observations improves
single-frame reasoning before adding multi-frame IPC or video-window support.

Observation history is **not** persistent world state and is **not** visual
evidence of motion. Each retained assistant response is unverified context that
may contain errors. Structured delivery keeps system instructions, current user
text, and prior user/assistant turns in their native TensorRT Edge-LLM roles.

## Thor run

Build and source the current branch, then run:

```bash
cd "$HOME/ros2-VLM-demo"
source scripts/edge_vlm_env.sh
source "$HOME/ros2_ws/install/setup.bash"

bash scripts/benchmark/run_observation_history_experiment.sh \
  --history-entries 0,1,3 \
  --success-results 4 \
  --playback-duration 30 \
  --max-generate-length 96 \
  --output-dir "$HOME/tensorrt-edgellm-benchmarks/observation-history-$(date -u +%Y%m%d-%H%M%S)"
```

The runner uses the same bag, prompt, generation limit, and structured delivery
for every configuration. Existing Cosmos deployment processes cause a safe
preflight refusal rather than being killed automatically.

## Artifacts

Each `history_N` directory contains:

- `manifest.json`: commit, engine paths, bag path, and exact run configuration;
- `results.log`: raw `VlmResult` messages;
- `result_summary.json`: success rate, latency summary, errors, and responses;
- `benchmark.jsonl`: per-frame ROS pipeline timings;
- `ros_metrics.json`: computed pipeline metrics;
- `launch.log`: worker and node diagnostics.

The root `experiment.json` combines all configurations without discarding the
raw evidence.

## Review rubric

Compare configurations for:

1. useful continuity across adjacent frames;
2. incorrect observations that persist into later responses;
3. unsupported motion or change claims;
4. omission of currently visible objects because history dominated the image;
5. response length and repetition;
6. inference and end-to-end latency.

Do not select a history length from latency alone. A configuration that is
consistent because it repeats an earlier mistake is worse than the zero-history
baseline.

## Interpretation

The expected first decision is whether one retained observation provides a
measurable benefit over zero. Three entries are included to reveal context
growth and error reinforcement, not because a longer history is presumed better.

If observation history does not materially improve the task-level results,
leave it disabled and proceed to native multi-frame experiments independently.
