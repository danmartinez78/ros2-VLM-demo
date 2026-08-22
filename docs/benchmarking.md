# Performance Benchmarking Guide

This guide explains how to reproduce latency, throughput, and resource
benchmarks for `edge_vlm_ros` on NVIDIA Jetson AGX Thor.

Measurements fall into two layers that must not be conflated:

| Layer | Tooling | What it measures |
|-------|---------|------------------|
| **Native engine/runtime** | NVIDIA `llm_bench`, `llm_inference --dumpProfile` | Prefill/decode/visual latency, token throughput, layer profiling |
| **ROS pipeline overhead** | Repository instrumentation (`benchmark_output_file`) | Image receipt → IPC, encoding, publication latency, dropped frames, cold start |

NVIDIA's tools are authoritative for all engine-level metrics. This repository
measures only what NVIDIA cannot: the overhead introduced by the ROS node,
IPC socket, and image preprocessing code.

---

## Prerequisites

A prepared Jetson AGX Thor with:
- JetPack 7.2 / R39.2 (CUDA 13.2, TensorRT, Edge-LLM built for `sm_110a`)
- Cosmos-Reason2-8B NVFP4 engine bundle
- ROS 2 Jazzy workspace built (see [deployment.md](deployment.md))
- Environment sourced: `source scripts/edge_vlm_env.sh`

Follow [deployment.md](deployment.md) steps 1–6 before running any benchmark.

---

## 1. Native Engine Baseline

Run NVIDIA's tools directly. Store artifacts so results are reproducible across
commits and model configurations.

```bash
cd "$HOME/ros2_ws/src/ros2-VLM-demo"
source scripts/edge_vlm_env.sh

bash scripts/benchmark/run_native_benchmarks.sh \
  --input-vlm-json "$EDGE_VLM_WORKSPACE_DIR/input_vlm.json"
```

Default parameters match the NVIDIA published workload:
`--batch-size 1`, `--input-len 2048`, `--past-kv-len 2048`, `--image-size 1024x2048`,
`--warmup 3`, `--iterations 10`, `--inference-warmup 10`.
Use `--quick` for faster smoke-test runs (128-token lengths, a 320x320
visual input, one warmup, and three measured iterations).

This calls, in order:

```bash
# Prefill latency / throughput
llm_bench --mode prefill \
  --engineDir "$EDGE_VLM_LLM_ENGINE_DIR" \
  --batchSize 1 \
  --inputLen 2048 \
  --warmup 3 --iterations 10 \
  --profile

# Generation (decode) throughput
llm_bench --mode decode \
  --engineDir "$EDGE_VLM_LLM_ENGINE_DIR" \
  --batchSize 1 \
  --pastKVLen 2048 \
  --warmup 3 --iterations 10 \
  --profile

# Vision-encoder latency / throughput
llm_bench --mode visual \
  --engineDir "$EDGE_VLM_MULTIMODAL_ENGINE_DIR/visual" \
  --imageSize 1024x2048 \
  --warmup 3 --iterations 10 \
  --profile

# End-to-end profiling with representative input
llm_inference \
  --engineDir "$EDGE_VLM_LLM_ENGINE_DIR" \
  --multimodalEngineDir "$EDGE_VLM_MULTIMODAL_ENGINE_DIR" \
  --inputFile "$EDGE_VLM_WORKSPACE_DIR/input_vlm.json" \
  --outputFile /tmp/cosmos_native_bench_*/llm_inference_output.json \
  --maxGenerateLength 64 \
  --warmup 10 \
  --dumpProfile \
  --profileOutputFile /tmp/cosmos_native_bench_*/llm_inference_profile.json
```

All native tool outputs are preserved verbatim in a timestamped directory:

```text
/tmp/cosmos_native_bench_YYYYMMDD_HHMMSS/
  manifest.json               ← run metadata and artifact paths
  llm_bench_prefill.txt       ← native prefill output
  llm_bench_decode.txt        ← native decode output
  llm_bench_visual.txt        ← native visual encoder output
  llm_inference_profile.json  ← native end-to-end layer profile
  llm_inference_output.json   ← native inference output JSON
```

> **No TTFT, token throughput, ViT timing, or layer profiling is reimplemented
> in this repository.** Use NVIDIA's output files directly.

---

## 2. ROS Pipeline Overhead

### 2a. Run the pipeline with benchmark output enabled

```bash
source scripts/edge_vlm_env.sh
source "$ROS_WORKSPACE/install/setup.bash"

BENCH_FILE="/tmp/cosmos_ros_bench_$(date +%Y%m%d_%H%M%S).jsonl"

ros2 launch edge_vlm_ros edge_vlm.launch.py \
  image_topic:=/hawk_0_left_rgb_image \
  llm_engine_dir:="$EDGE_VLM_LLM_ENGINE_DIR" \
  multimodal_engine_dir:="$EDGE_VLM_MULTIMODAL_ENGINE_DIR" \
  edge_llm_plugin_path:="$EDGELLM_PLUGIN_PATH" \
  benchmark_output_file:="${BENCH_FILE}" \
  use_sim_time:=true &

LAUNCH_PID=$!

# Play the validated rosbag
ros2 bag play "$HOME/ros2_ws/src/ros2-VLM-demo/scripts/test_data/bags/image-proc" \
  --clock

# Give the pipeline time to finish the last inference
sleep 5
kill ${LAUNCH_PID} 2>/dev/null || true
```

The `benchmark_output_file` parameter writes one JSON line per sampled frame plus
`session_start` and `session_end` records:

```jsonc
{"record_type":"session_start","node_init_wall_ns":1720000000000000000,"worker_ready_wall_ns":1720000002000000000,"task_profile":"scene_description",...}
{"record_type":"frame","frame_seq":1,"subscribe_wall_ns":1720000003000000000,"dequeue_wall_ns":1720000003002000000,"convert_done_ns":1720000003007000000,"infer_done_ns":1720000004510000000,"publish_done_ns":1720000004511000000,"inference_seconds":1.500,"success":true,"error":"","dropped_before":0,...}
...
{"record_type":"session_end","received":120,"sampled":60,"dropped":3,"success":58,"failure":2}
```

### 2b. Compute ROS overhead metrics

```bash
# Collect system metadata
python3 scripts/benchmark/benchmark_metadata.py \
  --model-name "${EDGE_VLM_MODEL_NAME}" \
  --llm-engine-dir "${EDGE_VLM_LLM_ENGINE_DIR}" \
  --multimodal-engine-dir "${EDGE_VLM_MULTIMODAL_ENGINE_DIR}" \
  --edge-llm-root "${TENSORRT_EDGE_LLM_ROOT}" \
  --output /tmp/bench_metadata.json

# Compute ROS pipeline metrics (exclude first 3 frames as warmup)
python3 scripts/benchmark/collect_ros_metrics.py \
  --input "${BENCH_FILE}" \
  --metadata /tmp/bench_metadata.json \
  --warmup 3 \
  --output /tmp/ros_report.json \
  --csv /tmp/ros_report.csv
```

### 2c. Generate comparison report

```bash
NATIVE_DIR=$(ls -d /tmp/cosmos_native_bench_* | tail -1)

python3 scripts/benchmark/generate_benchmark_report.py \
  --ros-report /tmp/ros_report.json \
  --native-dir "${NATIVE_DIR}" \
  --output /tmp/comparison_report.json \
  --text /tmp/comparison_report.txt

cat /tmp/comparison_report.txt
```

Example output:

```
========================================================================
  Cosmos ROS2 VLM Benchmark Report
  Generated: 2025-07-31T23:00:00+00:00
========================================================================

Platform
--------
  GPU:          Jetson AGX Thor
  Compute cap:  8.7
  JetPack:      7.2
  CUDA:         13.2
  TensorRT:     10.x
  Power mode:   MAXN

Native Engine Timing  (authoritative — NVIDIA worker timer)
------------------------------------------------------------
  Inference mean:   1520.0 ms
  Inference p50:    1510.0 ms
  Inference p95:    1580.0 ms

ROS Pipeline Overhead  (repository instrumentation)
----------------------------------------------------
  Image convert mean:  4.2 ms
  IPC overhead mean:   2.8 ms
  Publication mean:    0.9 ms
  Total ROS mean:      7.9 ms
  Ready to first frame: 2100.0 ms

End-to-End Pipeline  (native engine + ROS overhead)
-----------------------------------------------------
  Total mean:       1527.9 ms

Time breakdown (mean)
  Engine:  99.5%  |  ROS overhead:   0.5%
========================================================================
```

---

## 3. Required Metadata

Every benchmark record captures:

| Category | Fields |
|----------|--------|
| Hardware | `arch`, `gpu_name`, `gpu_compute_capability`, `nvpmodel_mode` |
| Platform | `jetpack_version`, `l4t_release`, `cuda_version`, `tensorrt_version` |
| Model | `model_name`, `quantization`, `edge_llm_commit`, `engine_config` |
| Token budget | `max_generate_length`, `warmup_iterations`, `measured_iterations` |
| Image / video | `image_max_width`, `jpeg_quality`, `input_image_path` |
| ROS config | `ros_distro`, `image_topic`, `sample_period_seconds`, `drop_old_frames` |
| Prompt | `task_profile`, `prompt_version`, `prompt_config_hash` |

---

## 4. Comparing Across Commits

To compare two commits:

```bash
# Run the full benchmark on commit A
git checkout <commit-A>
bash scripts/build_workspace.sh
bash scripts/benchmark/run_native_benchmarks.sh ...
python3 scripts/benchmark/collect_ros_metrics.py ... --output /tmp/ros_report_A.json

# Run on commit B
git checkout <commit-B>
bash scripts/build_workspace.sh
bash scripts/benchmark/run_native_benchmarks.sh ...
python3 scripts/benchmark/collect_ros_metrics.py ... --output /tmp/ros_report_B.json

# Compare
python3 scripts/benchmark/generate_benchmark_report.py \
  --ros-report /tmp/ros_report_A.json \
  --output /tmp/report_A.json
python3 scripts/benchmark/generate_benchmark_report.py \
  --ros-report /tmp/ros_report_B.json \
  --output /tmp/report_B.json
```

Profiles are comparable when:
- The same `model_name`, `quantization`, and engine-build parameters are used
- `max_generate_length` and image dimensions match
- `nvpmodel_mode` and clock state are consistent

---

## 5. CI Validation (CPU-only)

Hardware benchmarks do not run in ordinary CI. CI validates:

- Parser logic in `collect_ros_metrics.py`
- Comparison logic in `generate_benchmark_report.py`
- JSON schema structure in `scripts/benchmark/schemas/`
- Metadata collection utilities in `benchmark_metadata.py`

Run CI-safe tests locally:

```bash
python3 scripts/benchmark/test_benchmark_parsers.py -v
```

---

## 6. Schema Files

| File | Purpose |
|------|---------|
| `scripts/benchmark/schemas/ros_benchmark_result.schema.json` | Schema for ROS metrics report |
| `scripts/benchmark/schemas/native_benchmark_result.schema.json` | Schema for native benchmark manifest |

---

## 7. Thor full-pipeline contention benchmark (RT-DETR + adapter + VLM)

Use this when benchmarking the complete Thor perception/reasoning path:

`rosbag -> Isaac ROS RT-DETR -> tracked_observation_adapter -> TensorRT Edge-LLM/VLM -> /vlm/result`

The runner executes the A-F benchmark matrix and records synchronized:
- `tegrastats` telemetry (`tegrastats.log`)
- topic-rate logs for `/detections`, `/tracked_observation`, and `/vlm/result`
- ROS timing JSONL (`benchmark.jsonl`) plus parsed `ros_metrics.json` when available
- per-run manifest with mode, config, and git SHA
- cross-run `comparison_report.json` and `comparison_report.txt`

```bash
cd "$HOME/ros2_ws/src/ros2-VLM-demo"
source scripts/edge_vlm_env.sh
source "$ROS_WORKSPACE/install/setup.bash"

bash scripts/benchmark/run_thor_pipeline_benchmarks.sh \
  --rosbag-path /absolute/path/to/rosbag_dir \
  --duration-seconds 120
```

### Matrix implemented by the runner

| Mode | RT-DETR | VLM cadence |
|------|---------|-------------|
| A | on | off-like (`sample_period_seconds=3600`) |
| B | off | continuous (VLM baseline) |
| C | on | continuous/current (`sample_period_seconds=0`) |
| D | on | 1 Hz (`sample_period_seconds=1`) |
| E | on | 0.5 Hz (`sample_period_seconds=2`) |
| F | on | event/manual baseline (`sample_period_seconds=3600`, optional `--manual-trigger-command`) |

> Notes:
> - `--enable-rviz` is optional and disabled by default for cleaner measurements.
> - `--dry-run` prints the exact commands without launching workloads.

---

## Troubleshooting

| Symptom | Action |
|---------|--------|
| `benchmark_output_file: cannot open` | Check path permissions; the directory must exist |
| Empty JSONL file | Confirm `benchmark_output_file` is set and frames are being sampled |
| `inference_ms` is zero | The worker did not report `inference_seconds`; check TRT backend version |
| `ipc_overhead_ms` is negative | Clock skew between node and worker; `collect_ros_metrics.py` clamps to 0 |
| `ready_to_first_frame_ms` is None | `session_start` record was not written; check node startup succeeded |
| `llm_bench` not found | Build TensorRT Edge-LLM with `--cmake-args -DBUILD_BENCHMARKS=ON` |
| `unrecognized option '--multimodalEngineDir'` | `llm_bench --mode visual` accepts only `--engineDir`; point it directly at `$EDGE_VLM_MULTIMODAL_ENGINE_DIR/visual` |
| `Image data must be 4D [T, H, W, C]` | The pinned `llm_bench` creates a 3D dummy tensor for visual mode. Apply the source workaround below and rebuild `llm_bench` |
| Image dimensions are not divisible by `patchSize * mergeSize` | Use dimensions compatible with the visual model. Cosmos/Qwen3-VL requires multiples of 32; quick mode uses 320x320 |
| Native profile JSON missing | Pass `--profileOutputFile`; verify `--dumpProfile` flag is supported in your Edge-LLM version |

### Cosmos/Qwen3-VL visual benchmark workaround

The TensorRT Edge-LLM revision validated on Thor constructs its synthetic
visual input as `[H, W, C]`, while the Qwen3-VL runner requires
`[T, H, W, C]`. This is an upstream `llm_bench` defect; normal
`llm_inference` is unaffected because its image loader supplies the temporal
dimension.

In `TensorRT-Edge-LLM/examples/llm/llm_bench.cpp`, change the synthetic tensor
near the visual benchmark setup from:

```cpp
rt::Tensor fakeImage({static_cast<int64_t>(args.imageHeight),
    static_cast<int64_t>(args.imageWidth), 3}, ...);
```

to:

```cpp
rt::Tensor fakeImage({1, static_cast<int64_t>(args.imageHeight),
    static_cast<int64_t>(args.imageWidth), 3}, ...);
```

Then rebuild only the native benchmark target:

```bash
cmake --build "$TENSORRT_EDGE_LLM_ROOT/build" --target llm_bench -j"$(nproc)"
```

This repository deliberately does not patch or replace NVIDIA's benchmark
implementation. The wrapper reports the known failure and preserves its native
output for diagnosis.
