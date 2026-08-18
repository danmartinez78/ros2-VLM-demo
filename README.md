# edge_vlm_ros

ROS 2 Jazzy pipeline for NVIDIA Jetson AGX Thor that samples raw camera frames,
runs a TensorRT Edge-LLM VLM through a model-neutral IPC worker, and publishes
structured vision-language results. Supported model configurations include
NVIDIA Cosmos-Reason2 and Qwen3-VL.

The hardware path targets JetPack 7.1 / Jetson Linux R38.4 with a
Cosmos-Reason2-8B NVFP4 engine and NVIDIA Isaac ROS assets.

## Architecture

```mermaid
flowchart LR
    CAMERA[ROS bag or camera] -->|sensor_msgs/Image| ROS[edge_vlm_ros_node]
    ROS -->|bounded BGR8 IPC| IPC[(Unix socket)]
    IPC --> WORKER[edge_vlm_server]
    WORKER -->|in-memory JPEG| TRT[TensorRT Edge-LLM]
    TRT --> WORKER
    WORKER --> IPC
    ROS -->|VlmResult| RESULT[/vlm/result]
```

The ROS and GPU runtimes intentionally run in separate processes. Loading ROS
2's transitive native libraries into the same process as TensorRT Edge-LLM
caused a reproducible fused-attention prefill stall on Thor. Process isolation
removes that interaction and allows a failed worker to be restarted without
restarting the ROS node.

The worker is also a standalone reasoning service. It can remain loaded while
ROS adapters, command-line experiments, and future evaluation tools connect
sequentially through the same versioned IPC contract. See
[the standalone service guide](docs/standalone-reasoning-service.md) for
ROS-free image inference and externally managed worker recipes.

See [docs/architecture.md](docs/architecture.md) for the detailed design and
[docs/thor-edge-llm-prefill-stall-rca.md](docs/thor-edge-llm-prefill-stall-rca.md)
for the investigation and evidence.

## Supported target baseline

| Component | Target configuration |
| --- | --- |
| Hardware | NVIDIA Jetson AGX Thor |
| OS | Ubuntu 24.04, aarch64 |
| JetPack / Jetson Linux | JetPack 7.1 / R38.4 |
| CUDA | 13.0 (Thor profile) |
| ROS 2 | Jazzy |
| TensorRT Edge-LLM | Thor build containing `sm_110a` CUDA images |
| Model | Cosmos-Reason2-8B, NVFP4 |
| Isaac ROS | 4.5 Docker tooling optional (RT-DETR path) |

Other compatible models may work, but they have not yet been hardware-verified here.
Model portability and optimization are tracked in
[#9](https://github.com/danmartinez78/ros2-VLM-demo/issues/9).

## Quick start on a fresh Thor

```bash
mkdir -p "$HOME/ros2_ws/src"
cd "$HOME/ros2_ws/src"
git clone https://github.com/danmartinez78/ros2-VLM-demo.git
cd ros2-VLM-demo

bash scripts/setup_thor_jp71.sh
```

Setup installs dependencies, pins/builds TensorRT Edge-LLM, prepares model/data
layout, installs RT-DETR model assets, and generates `scripts/edge_vlm_env.sh`.
Before first run, accept the gated Cosmos license and authenticate with
Hugging Face (`huggingface-cli login`) on the Thor host.

The deployment verifier can be re-run independently:

```bash
source scripts/edge_vlm_env.sh
source "$ROS_WORKSPACE/install/setup.bash"
bash scripts/verify_thor_jp71.sh --isaac-ros
```

Optional standalone Edge-LLM smoke check:

```bash
bash scripts/verify_thor_jp71.sh --isaac-ros --smoke-image /absolute/path/to/image.jpg
```

The verifier confirms both executables are installed and enforces the process
boundary:

- `edge_vlm_ros_node` must not load CUDA, TensorRT, or the Edge-LLM backend;
- `edge_vlm_server` must not load ROS, RMW, or DDS libraries.

For a fresh system and engine preparation, follow the full
[Thor deployment recipe](docs/deployment.md).

## Run with a ROS bag

Terminal 1:

```bash
cd "$HOME/ros2-VLM-demo"
source scripts/edge_vlm_env.sh
source "$ROS_WORKSPACE/install/setup.bash"

ros2 launch edge_vlm_ros edge_vlm.launch.py \
  image_topic:=/camera/image_raw \
  llm_engine_dir:="$EDGE_VLM_LLM_ENGINE_DIR" \
  multimodal_engine_dir:="$EDGE_VLM_MULTIMODAL_ENGINE_DIR" \
  edge_llm_plugin_path:="$EDGELLM_PLUGIN_PATH" \
  use_sim_time:=true
```

Terminal 2:

```bash
source /opt/ros/jazzy/setup.bash
ros2 bag play /absolute/path/to/bag --clock
```

Terminal 3, optional:

```bash
source /opt/ros/jazzy/setup.bash
source "$HOME/ros2_ws/install/setup.bash"
ros2 topic echo /vlm/result
```

Terminal 4, optional RViz2 panel:

```bash
source /opt/ros/jazzy/setup.bash
source "$HOME/ros2_ws/install/setup.bash"
rviz2 -d "$HOME/ros2-VLM-demo/rviz/vision_reasoning_results.rviz"
```

Paths supplied to ROS must be absolute; `~` is not expanded.

## Thor tracked-observation one-command bring-up

Terminal 1:

```bash
cd "$HOME/ros2-VLM-demo"
source scripts/edge_vlm_env.sh
source "$ROS_WORKSPACE/install/setup.bash"

ros2 launch edge_vlm_ros thor_tracked_observation.launch.py \
  llm_engine_dir:="$EDGE_VLM_LLM_ENGINE_DIR" \
  multimodal_engine_dir:="$EDGE_VLM_MULTIMODAL_ENGINE_DIR" \
  edge_llm_plugin_path:="$EDGELLM_PLUGIN_PATH"
```

This single command starts the existing Edge-LLM worker + ROS VLM node, the
tracked-observation adapter, and RViz2. By default it uses `use_sim_time:=true`,
subscribes to `/camera0/color/image_raw`, accepts detector-neutral
`vision_msgs/msg/Detection2DArray` input on `/detections`, publishes tracked
observations on `/tracked_observation`, and publishes reasoning output on
`/vlm/result`. Rosbag playback remains separate, and a detector can publish to
`/detections` externally.

Terminal 2 (separate rosbag playback):

```bash
source /opt/ros/jazzy/setup.bash
ros2 bag play \
  /home/daniel/ros2-VLM-demo/test_data/rosbags/nvblox/isaac_ros_nvblox/galileo_people_3_2 \
  --clock --loop
```

Optional overrides:
- `image_topic:=...`
- `detections_topic:=...`
- `tracked_observation_topic:=...`
- `start_rtdetr:=true`
- `enable_rviz:=false`
- `use_sim_time:=false`

Set `start_rtdetr:=true` to launch the repository-owned Isaac ROS RT-DETR
backend directly from this entrypoint. When enabled, the launch wires
`/camera0/color/image_raw` (or your `image_topic`) into RT-DETR and remaps its
`vision_msgs/msg/Detection2DArray` output to `/detections`.

The launch fails early with a clear error if RViz2, the RViz config, or any
required engine/plugin path is missing. When `start_rtdetr:=true`, it also
fails early if the supported Isaac ROS RT-DETR packages or launch files are not
installed.

## NVIDIA test data

Download or inspect the supported NVIDIA quickstart assets:

```bash
bash scripts/test_data/download_rosbags.sh list
bash scripts/test_data/download_rosbags.sh download image-proc
bash scripts/test_data/download_rosbags.sh download h264
bash scripts/test_data/inspect_rosbag.sh /path/to/bag
```

Run the validated end-to-end image test:

```bash
source scripts/edge_vlm_env.sh
source "$ROS_WORKSPACE/install/setup.bash"
bash scripts/test_data/run_image_proc_test.sh
```

The smoke test stops after the first successful reasoning result and performs
bounded cleanup. Set `PLAYBACK_DURATION_SECONDS`, `RESULT_TIMEOUT_SECONDS`,
or `MAX_GENERATE_LENGTH` to override its 20-second playback, 120-second
result timeout, and 64-token defaults.

Test worker crash recovery:

```bash
bash scripts/test_data/run_worker_recovery_test.sh
```

That test obtains a successful result, kills only the inference worker,
verifies launch respawns it, and confirms reasoning resumes without restarting
the ROS node.

## Task-level evaluation

Run task-level evaluation on recorded outputs:

```bash
python3 scripts/evaluation/evaluate_task_harness.py \
  --dataset scripts/evaluation/dataset_v1.json \
  --run /absolute/path/to/run.json \
  --output /absolute/path/to/eval-report.json
```

See [docs/evaluation.md](docs/evaluation.md) for dataset and rubric details.

## Observation-history experiment

Prior model outputs can be retained as **unverified semantic observations** and
delivered as native user/assistant message history. This is disabled by default
and is not equivalent to visual motion evidence or persistent scene state.

Compare zero, one, and three retained observations against the same rosbag:

```bash
bash scripts/benchmark/run_observation_history_experiment.sh \
  --history-entries 0,1,3 \
  --success-results 4
```

The runner preserves raw results, ROS timing JSONL, per-run manifests, and a
combined experiment manifest. See
[docs/observation-history-experiment.md](docs/observation-history-experiment.md)
for the Thor recipe and review rubric.

## Performance benchmarking

Benchmarks are separated into two layers that must not be conflated:

1. **Native engine baseline** — NVIDIA `llm_bench` and `llm_inference --dumpProfile`
   are authoritative for prefill/decode/visual latency and token throughput.
2. **ROS pipeline overhead** — repository instrumentation measures image receipt,
   IPC/encoding, publication latency, dropped frames, and cold start.

### Native engine baseline (Thor only)

```bash
source scripts/edge_vlm_env.sh

bash scripts/benchmark/run_native_benchmarks.sh \
  --input-vlm-json "$EDGE_VLM_WORKSPACE_DIR/input_vlm.json"
```

Defaults match the NVIDIA published workload: `--batch-size 1 --input-len 2048
--past-kv-len 2048 --image-size 1024x2048 --warmup 3 --iterations 10
--inference-warmup 10`. Pass `--quick` for a 320x320 visual smoke test and
reduced token lengths/iteration counts. The pinned TensorRT Edge-LLM revision
requires a one-line upstream fix for Cosmos/Qwen3-VL visual benchmarking; see
the [benchmark troubleshooting guide](docs/benchmarking.md#cosmosqwen3-vl-visual-benchmark-workaround).

Artifacts are written to `/tmp/cosmos_native_bench_YYYYMMDD_HHMMSS/`.

### ROS pipeline overhead

Enable benchmark output via the `benchmark_output_file` parameter:

```bash
ros2 launch edge_vlm_ros edge_vlm.launch.py \
  image_topic:=/hawk_0_left_rgb_image \
  llm_engine_dir:="$EDGE_VLM_LLM_ENGINE_DIR" \
  multimodal_engine_dir:="$EDGE_VLM_MULTIMODAL_ENGINE_DIR" \
  edge_llm_plugin_path:="$EDGELLM_PLUGIN_PATH" \
  benchmark_output_file:="/tmp/cosmos_ros_bench.jsonl" \
  use_sim_time:=true
```

Then compute ROS overhead metrics and a comparison report:

```bash
python3 scripts/benchmark/collect_ros_metrics.py \
  --input /tmp/cosmos_ros_bench.jsonl \
  --warmup 3 \
  --output /tmp/ros_report.json

python3 scripts/benchmark/generate_benchmark_report.py \
  --ros-report /tmp/ros_report.json \
  --native-dir /tmp/cosmos_native_bench_* \
  --output /tmp/comparison.json \
  --text /tmp/comparison.txt
```

See [docs/benchmarking.md](docs/benchmarking.md) for the full guide, required
metadata, comparison-across-commits workflow, and troubleshooting.

### Benchmark parsers (CPU-only CI)

Parser logic, schema validation, and metric computation are tested without
hardware:

```bash
python3 scripts/benchmark/test_benchmark_parsers.py -v
```

GitHub Actions runs this automatically in
`.github/workflows/hardware-independent-tests.yml`.

## Automated IPC protocol tests (CPU-only)

The hardware-independent test suite now covers IPC protocol framing, malformed
or truncated worker responses, request/response header validation, timeout
behavior, and reconnect logic using a fake Unix-socket worker.

Run the IPC-focused tests with:

```bash
colcon test --packages-select edge_vlm_ros \
  --ctest-args -R "test_(ipc_protocol|ipc_inference_backend)" --output-on-failure
```

GitHub Actions runs this CPU-only IPC coverage automatically in
`.github/workflows/hardware-independent-tests.yml`.

Thor-only validation is still required for the end-to-end worker respawn launch
path:

```bash
bash scripts/test_data/run_worker_recovery_test.sh
```

The H.264 asset is not directly consumable. Decode it to
`sensor_msgs/msg/Image` with an appropriate ROS/Isaac ROS decoder first.

## Output

Results are published as
`edge_vlm_ros/msg/VlmResult` on
`/vlm/result` by default. Each result includes:

- source image header and topic;
- optional tracked-observation provenance (`detector_id`, `tracker_id`, `source_sequence`,
  tracked-object count, observation age, serialized tracker context) when enabled;
- effective prompt, selected task profile, prompt-version label, and prompt-configuration hash;
- generated response;
- inference duration;
- sampled-frame sequence number;
- success state and error text.

Example console output:

```text
[edge_vlm_ros_node]: [frame 5 | 1.573 s] The scene shows a warehouse aisle...
```

The observed 1.5-second results used a 64-token output limit. Longer responses
increase end-to-end latency.

Tracked-observation mode is optional at runtime. Set
`enable_tracked_observation_input:=true` and publish
`edge_vlm_ros/msg/TrackedObservation` messages to bypass the raw-image subscriber
while preserving continuous latest-only delivery semantics.

## RViz2 visualization panel (optional)

This repository now includes an RViz2 panel plugin named
`edge_vlm_ros/VisionReasoningPanel` is the RViz panel class name for visual debugging and demos.
The panel displays:

- the camera image matched to the result message timestamp;
- prompt and generated response text;
- success/failure/stale status with distinct colors;
- result stamp, latest image stamp, frame sequence, and latency.

The plugin is optional and only builds when these dependencies are present:

- `rviz_common`
- `pluginlib`
- Qt5 (`qtbase5-dev` or equivalent)

If these are unavailable, the core inference package still builds and runs; CMake
prints that the RViz2 panel target was skipped.

When built, load the supplied config:

```bash
rviz2 -d /absolute/path/to/ros2-VLM-demo/rviz/vision_reasoning_results.rviz
```

## Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `image_topic` | string | `/camera/image_raw` | Raw image input topic |
| `result_topic` | string | `/vlm/result` | Result topic |
| `worker_socket_path` | string | `/tmp/edge_vlm.sock` | Worker Unix-socket path |
| `start_worker` | launch argument | `true` | Start and supervise the worker from ROS launch; set `false` to use a standalone service |
| `worker_connect_timeout_seconds` | int | `120` | Initial/reconnect deadline |
| `worker_inference_deadline_seconds` | int | `60` | Worker-side inference deadline; worker self-terminates via watchdog on expiry (must be < `worker_request_timeout_seconds`) |
| `worker_request_timeout_seconds` | int | `90` | Per-request IPC send/receive deadline (must be > `worker_inference_deadline_seconds`) |
| `llm_engine_dir` | string | empty | Required LLM engine/tokenizer directory |
| `multimodal_engine_dir` | string | empty | Required multimodal engine directory |
| `edge_llm_plugin_path` | string | empty | Required Edge-LLM plugin path |
| `prompt` | string | scene-description prompt | Legacy prompt used when `task_profile=legacy_prompt` |
| `task_profile` | string | `legacy_prompt` | Active task profile (`legacy_prompt`, `scene_description`, `hazard_detection`, `inventory`, `navigation_assistance`) |
| `prompt_version` | string | `v1` | Version label attached to every result for reproducibility |
| `system_instruction` | string | empty | Optional system instruction text |
| `task_instruction` | string | empty | Optional task instruction text |
| `instruction_delivery_mode` | string | `inline` | Prompt delivery mode: `inline` (legacy, all text in one user message) or `structured` (system message, user message, and history as native Edge-LLM Message roles) |
| `enable_system_prompt_cache` | bool | `false` | Request system-prompt caching for stable system messages (only valid with `instruction_delivery_mode: structured`; silently ignored when the runtime does not support caching — see Thor validation note in architecture docs) |
| `observation_history_max_entries` | int | `0` | Bound on prior successful responses retained for observation-history injection |
| `observation_history_max_chars` | int | `0` | Maximum total retained observation-history characters (`0` disables size limit) |
| `observation_history_reset_policy` | string | `never` | Observation-history reset policy: `never`, `on_error`, `every_n_requests` |
| `observation_history_reset_interval_requests` | int | `0` | Reset interval when policy is `every_n_requests` |
| `sample_period_seconds` | double | `2.0` | Minimum ROS timestamp interval between samples |
| `max_generate_length` | int | `256` | Maximum generated tokens |
| `temperature` | double | `0.2` | Sampling temperature |
| `top_p` | double | `0.9` | Nucleus-sampling probability |
| `top_k` | int | `20` | Top-k sampling |
| `image_max_width` | int | `1280` | Maximum input width before resizing |
| `jpeg_quality` | int | `90` | In-memory JPEG quality passed to the worker |
| `drop_old_frames` | bool | `true` | Replace a queued stale frame with the newest frame |
| `publish_results` | bool | `true` | Publish result messages |

Prompt behavior is template-driven and validated at startup/launch time.
Unknown template variables, malformed braces, and unsupported instruction
delivery modes fail fast with explicit errors.

## Operational behavior

- The ROS callback never performs inference.
- A single pending-frame slot prevents stale-frame and memory accumulation.
- Engines and the CUDA context remain loaded in the persistent worker.
- Launch respawns a worker that exits unexpectedly after two seconds.
- A failed IPC request is reported once and is not automatically replayed.
- The next sampled frame reconnects to a replacement worker.
- If the worker is alive but wedged in a GPU call, its internal watchdog fires
  after `worker_inference_deadline_seconds` (default 60 s): the worker emits a
  diagnostic, calls `quick_exit`, and launch respawns it. The client IPC timeout
  (`worker_request_timeout_seconds`, default 90 s) is kept longer than the
  worker deadline so the worker exits cleanly before the client-side timeout
  fires. One error result is published; the request is not replayed.

## Troubleshooting

### `device kernel image is invalid`

The final Edge-LLM consumer was not built for Thor. Rebuild Edge-LLM and this
package for `sm_110a`, then verify with `cuobjdump` as shown in the deployment
recipe.

### Missing plugin

Verify the resolved file exists:

```bash
readlink -f "$EDGELLM_PLUGIN_PATH"
```

The unversioned `libNvInfer_edgellm_plugin.so` may be a symlink to the built
versioned library.

### Missing `engine/action/action.engine`

The action runner is optional. Cosmos-Reason2-8B image reasoning works without
an action engine; the same informational message appears in successful native
inference.

### OpenCV package conflicts

Do not install Ubuntu/ROS image packages when APT proposes removing
`nvidia-jetpack`, `nvidia-jetpack-dev`, or `nvidia-opencv-dev`. The production
node intentionally avoids `cv_bridge` so ROS OpenCV 4.6 and NVIDIA OpenCV 4.8
are not loaded into the same process.

### No camera messages

```bash
ros2 bag info /path/to/bag
ros2 topic info /camera/image_raw --verbose
ros2 topic echo /camera/image_raw --once --no-arr
```

The production node accepts `bgr8`, `rgb8`, and `mono8` raw images.

## Known limitations

- Independent sampled images, not temporal video windows
  ([#8](https://github.com/danmartinez78/ros2-VLM-demo/issues/8)).
- Batch size one; batching has not been shown beneficial for this live path.
- Raw `sensor_msgs/msg/Image` only; compressed streams require decoding first.
- Task-level evaluation requires curated run outputs plus rubric review
  ([docs/evaluation.md](docs/evaluation.md)).

## License

Apache 2.0. See [LICENSE](LICENSE).
