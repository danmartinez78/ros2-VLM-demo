# edge_vlm_ros

A generic, model-neutral ROS 2 VLM pipeline demo for NVIDIA Jetson platforms.

The repository demonstrates reusable camera-to-VLM plumbing rather than a single application: process-isolated GPU inference workers, structured ROS results, tracked-observation context, bounded multi-frame history, native-video temporal inference, model/profile management, repeatable benchmarking, and controlled temporal evaluation.

Current hardware validation targets **NVIDIA Jetson AGX Thor**, **JetPack 7.2.x / Jetson Linux R39.2.x**, and **ROS 2 Jazzy**. Runtime work in the repository includes NVIDIA Cosmos-Reason2, Cosmos3 Edge through FlashRT, and Qwen3-VL-compatible temporal paths.

## Architecture

```mermaid
flowchart LR
    CAMERA[ROS bag or camera] -->|sensor_msgs/Image| ROS[ROS 2 VLM node]
    ROS -->|bounded BGR8 IPC| IPC[(Unix socket)]
    IPC --> WORKER[model runtime worker]
    WORKER --> MODEL[VLM runtime / engine]
    MODEL --> WORKER
    WORKER --> IPC
    ROS -->|VlmResult| RESULT[/vlm/result]
```

The ROS and GPU runtimes intentionally run in separate processes. This keeps ROS/DDS dependencies isolated from accelerator/runtime libraries, allows the model worker to remain loaded while clients reconnect, and lets a failed worker restart without restarting the ROS graph.

The versioned IPC contract supports single images, ordered image sequences, native temporal/video requests, exact frame timestamps, prompts, and structured result metadata.

## What this demo includes

- ROS 2 camera or rosbag input through `sensor_msgs/msg/Image`
- latest-only sampling/backpressure for live VLM inference
- isolated TensorRT Edge-LLM worker path
- FlashRT/Cosmos3 native-video worker path
- structured `edge_vlm_ros/msg/VlmResult` output
- optional tracked-object context using `vision_msgs/msg/Detection2DArray`
- bounded rolling temporal windows
- exact per-frame timestamp transport
- controlled forward/reverse/shuffled/static temporal tests
- model/profile management scripts
- native-engine and ROS-pipeline benchmarking
- task-level evaluation and temporal distillation scaffolding
- optional RViz2 visualization panel

## Supported target baseline

| Component | Target configuration |
| --- | --- |
| Hardware | NVIDIA Jetson AGX Thor |
| OS | Ubuntu 24.04, aarch64 |
| JetPack / Jetson Linux | JetPack 7.2.x / R39.2.x |
| CUDA | Host toolkit 13.2 on the validated JP7.2 path |
| ROS 2 | Jazzy |
| Primary Edge-LLM baseline | Cosmos-Reason2-8B NVFP4 |
| Temporal/video development path | Cosmos3 Edge through FlashRT |
| Optional perception input | Isaac ROS / `vision_msgs` detections |

Other compatible VLMs can be integrated behind the same ROS and IPC boundaries.

## Quick start on Thor

```bash
mkdir -p "$HOME/ros2_ws/src"
cd "$HOME/ros2_ws/src"
git clone https://github.com/danmartinez78/ros2-VLM-demo.git
cd ros2-VLM-demo

bash scripts/setup_thor_jp72.sh
```

Before first model use, accept any model-specific gated licenses and authenticate with the required model provider on the Thor host.

Verify the deployment:

```bash
source scripts/edge_vlm_env.sh
source "$ROS_WORKSPACE/install/setup.bash"
bash scripts/verify_thor_jp72.sh --isaac-ros
```

For the full deployment recipe, see [docs/deployment.md](docs/deployment.md).

## Managed model/profile workflow

```bash
./scripts/modelctl list
./scripts/modelctl status cosmos-reason2-8b thor-current
./scripts/modelctl prepare cosmos-reason2-8b
./scripts/modelctl build cosmos-reason2-8b thor-f8
./scripts/modelctl validate cosmos-reason2-8b thor-f8
./scripts/modelctl activate cosmos-reason2-8b thor-f8
./scripts/modelctl current
```

Model/profile state is kept separate from the ROS interface so engine changes do not require rewriting the pipeline.

## Run the standard ROS pipeline with a bag

Terminal 1:

```bash
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

Paths supplied to ROS should be absolute; `~` is not expanded in launch arguments.

## Optional tracked-observation context

The tracked-observation path can combine image reasoning with detector/tracker context while keeping the detector implementation replaceable.

```bash
source scripts/edge_vlm_env.sh
source "$ROS_WORKSPACE/install/setup.bash"

ros2 launch edge_vlm_ros thor_tracked_observation.launch.py \
  llm_engine_dir:="$EDGE_VLM_LLM_ENGINE_DIR" \
  multimodal_engine_dir:="$EDGE_VLM_MULTIMODAL_ENGINE_DIR" \
  edge_llm_plugin_path:="$EDGELLM_PLUGIN_PATH"
```

The adapter accepts detector-neutral `vision_msgs/msg/Detection2DArray` messages and publishes tracked context for the VLM path. Isaac ROS RT-DETR can be launched by the repository or replaced by another detector that publishes the same message contract.

## Native-video temporal path

The `flashrt_temporal/` path demonstrates native Cosmos3 video inference with exact timestamp semantics.

Start the FlashRT worker:

```bash
cd ~/ros2_ws/src/ros2-VLM-demo
sudo bash flashrt_temporal/run_worker_docker.sh
```

Run the temporal ROS node:

```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/local_setup.bash

python3 flashrt_temporal/temporal_ros_node.py --ros-args \
  -p image_topic:=/camera0/color/image_raw \
  -p sample_period_seconds:=0.25 \
  -p temporal_window_frames:=8 \
  -p temporal_max_gap_seconds:=1.25 \
  -p worker_socket_path:=/tmp/edge_vlm_flashrt.sock \
  -p max_generate_length:=256
```

The node resets its temporal window across backward timestamps or large source gaps so one VLM request is never presented as a continuous video when the source stream was discontinuous.

### Controlled chronology test

The saved-capture harness runs the same frame window through controlled temporal variants:

```bash
python3 flashrt_temporal/temporal_chronology_test.py \
  --image-topic /camera0/color/image_raw \
  --sample-period-seconds 0.25 \
  --window-frames 8 \
  --max-gap-seconds 1.25 \
  --capture-candidates 60 \
  --worker-socket-path /tmp/edge_vlm_flashrt.sock \
  --max-generate-length 256
```

It evaluates:

- chronological frames
- reversed frames with the same timestamp schedule
- deterministic shuffled order
- repeated copies of the terminal frame as native video
- a single terminal-frame diagnostic

The first controlled Cosmos3 run on Thor showed clear chronology sensitivity: reversing the same eight frames reversed the inferred lateral motion direction. The shuffled control still produced a plausible motion narrative, so broad temporal-coherence reliability remains an open benchmark question.

See [docs/temporal-chronology-results.md](docs/temporal-chronology-results.md) and [docs/architecture/temporal-results-matrix.md](docs/architecture/temporal-results-matrix.md).

## Output

Results are published as `edge_vlm_ros/msg/VlmResult` on `/vlm/result` by default. Results can include:

- source image header and topic
- inference duration and frame sequence
- prompt/task-profile provenance
- generated response and success/error state
- optional detector/tracker provenance and serialized tracked context
- temporal runtime encoding metadata when using the video path

Example:

```text
[edge_vlm_ros_node]: [frame 5 | 1.573 s] The scene shows a warehouse aisle...
```

## Benchmarking

The repository separates two performance layers:

1. **Native engine baseline** — runtime-native tools measure prefill/decode/vision latency and token throughput.
2. **ROS pipeline overhead** — repository instrumentation measures image receipt, IPC, publication latency, dropped frames, and cold start.

Native baseline example:

```bash
source scripts/edge_vlm_env.sh
bash scripts/benchmark/run_native_benchmarks.sh \
  --input-vlm-json "$EDGE_VLM_WORKSPACE_DIR/input_vlm.json"
```

ROS metrics example:

```bash
python3 scripts/benchmark/collect_ros_metrics.py \
  --input /tmp/cosmos_ros_bench.jsonl \
  --warmup 3 \
  --output /tmp/ros_report.json
```

See [docs/benchmarking.md](docs/benchmarking.md) for the full benchmarking workflow.

## Evaluation and experiments

Task-level evaluation:

```bash
python3 scripts/evaluation/evaluate_task_harness.py \
  --dataset scripts/evaluation/dataset_v1.json \
  --run /absolute/path/to/run.json \
  --output /absolute/path/to/eval-report.json
```

Observation-history experiment:

```bash
bash scripts/benchmark/run_observation_history_experiment.sh \
  --history-entries 0,1,3 \
  --success-results 4
```

Temporal evaluation should preserve the exact representation used by the model: ordered images and native video are different experimental conditions, and exact timestamps are part of the input semantics.

## Documentation map

- [Architecture](docs/architecture.md)
- [Architecture design map](docs/architecture/README.md)
- [Temporal VLM architecture](docs/architecture/temporal-vlm-architecture.md)
- [Temporal evidence/results matrix](docs/architecture/temporal-results-matrix.md)
- [Controlled chronology results](docs/temporal-chronology-results.md)
- [Benchmarking](docs/benchmarking.md)
- [Deployment](docs/deployment.md)
- [Standalone reasoning service](docs/standalone-reasoning-service.md)
- [Distillation pipeline design](docs/distillation-pipeline-design.md)

## License

MIT. See [LICENSE](LICENSE).
