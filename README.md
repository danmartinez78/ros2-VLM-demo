# cosmos_ros2_video_reasoner

ROS 2 Jazzy pipeline for NVIDIA Jetson AGX Thor that samples raw camera frames,
runs NVIDIA Cosmos Reason2 through TensorRT Edge-LLM, and publishes structured
vision-reasoning results.

The hardware path has been validated on JetPack 7.2 / Jetson Linux R39.2 with a
Cosmos-Reason2-8B NVFP4 engine and an NVIDIA Isaac ROS image-proc rosbag.

## Architecture

```mermaid
flowchart LR
    CAMERA[ROS bag or camera] -->|sensor_msgs/Image| ROS[cosmos_reasoner]
    ROS -->|bounded BGR8 IPC| IPC[(Unix socket)]
    IPC --> WORKER[cosmos_inference_worker]
    WORKER -->|in-memory JPEG| TRT[TensorRT Edge-LLM]
    TRT --> WORKER
    WORKER --> IPC
    ROS -->|VisionReasoningResult| RESULT[/cosmos/reasoning]
```

The ROS and GPU runtimes intentionally run in separate processes. Loading ROS
2's transitive native libraries into the same process as TensorRT Edge-LLM
caused a reproducible fused-attention prefill stall on Thor. Process isolation
removes that interaction and allows a failed worker to be restarted without
restarting the ROS node.

See [docs/architecture.md](docs/architecture.md) for the detailed design and
[docs/thor-edge-llm-prefill-stall-rca.md](docs/thor-edge-llm-prefill-stall-rca.md)
for the investigation and evidence.

## Validated platform

| Component | Validated configuration |
| --- | --- |
| Hardware | NVIDIA Jetson AGX Thor |
| OS | Ubuntu 24.04, aarch64 |
| JetPack / Jetson Linux | JetPack 7.2 / R39.2 |
| CUDA | 13.2 |
| ROS 2 | Jazzy |
| TensorRT Edge-LLM | Thor build containing `sm_110a` CUDA images |
| Model | Cosmos-Reason2-8B, NVFP4 |
| Isaac ROS | 4.5 Docker tooling optional; JetPack 7.2 combination remains outside NVIDIA's listed validation matrix |

Other compatible models may work, but they have not yet been validated here.
Model portability and optimization are tracked in
[#9](https://github.com/danmartinez78/ros2-VLM-demo/issues/9).

## Quick start on a prepared Thor

This assumes TensorRT Edge-LLM and the Cosmos engine bundle are already built
on the Thor.

```bash
mkdir -p "$HOME/ros2_ws/src"
cd "$HOME/ros2_ws/src"
git clone https://github.com/danmartinez78/ros2-VLM-demo.git
cd ros2-VLM-demo

bash scripts/setup_deployment.sh
```

On its first run, setup installs the system dependencies and creates:

```text
scripts/cosmos_env.sh
```

Review the paths in that file, then run:

```bash
source scripts/cosmos_env.sh
bash scripts/build_workspace.sh
source "$ROS_WORKSPACE/install/setup.bash"
bash scripts/verify_deployment.sh
```

The verifier confirms both executables are installed and enforces the process
boundary:

- `cosmos_reasoner` must not load CUDA, TensorRT, or the Edge-LLM backend;
- `cosmos_inference_worker` must not load ROS, RMW, or DDS libraries.

For a fresh system and engine preparation, follow the full
[Thor deployment recipe](docs/deployment.md).

## Run with a ROS bag

Terminal 1:

```bash
cd "$HOME/ros2-VLM-demo"
source scripts/cosmos_env.sh
source "$ROS_WORKSPACE/install/setup.bash"

ros2 launch cosmos_ros2_video_reasoner cosmos_reasoner.launch.py \
  image_topic:=/camera/image_raw \
  llm_engine_dir:="$COSMOS_LLM_ENGINE_DIR" \
  multimodal_engine_dir:="$COSMOS_MULTIMODAL_ENGINE_DIR" \
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
ros2 topic echo /cosmos/reasoning
```

Paths supplied to ROS must be absolute; `~` is not expanded.

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
source scripts/cosmos_env.sh
source "$ROS_WORKSPACE/install/setup.bash"
bash scripts/test_data/run_image_proc_test.sh
```

Test worker crash recovery:

```bash
bash scripts/test_data/run_worker_recovery_test.sh
```

That test obtains a successful result, kills only the inference worker,
verifies launch respawns it, and confirms reasoning resumes without restarting
the ROS node.

The H.264 asset is not directly consumable. Decode it to
`sensor_msgs/msg/Image` with an appropriate ROS/Isaac ROS decoder first.

## Output

Results are published as
`cosmos_ros2_video_reasoner/msg/VisionReasoningResult` on
`/cosmos/reasoning` by default. Each result includes:

- source image header and topic;
- effective prompt;
- generated response;
- inference duration;
- sampled-frame sequence number;
- success state and error text.

Example console output:

```text
[cosmos_reasoner]: [frame 5 | 1.573 s] The scene shows a warehouse aisle...
```

The observed 1.5-second results used a 64-token output limit. Longer responses
increase end-to-end latency.

## Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `image_topic` | string | `/camera/image_raw` | Raw image input topic |
| `result_topic` | string | `/cosmos/reasoning` | Result topic |
| `worker_socket_path` | string | `/tmp/cosmos_edge_llm.sock` | Worker Unix-socket path |
| `worker_connect_timeout_seconds` | int | `120` | Initial/reconnect deadline |
| `worker_request_timeout_seconds` | int | `90` | Per-request IPC send/receive deadline |
| `llm_engine_dir` | string | empty | Required LLM engine/tokenizer directory |
| `multimodal_engine_dir` | string | empty | Required multimodal engine directory |
| `edge_llm_plugin_path` | string | empty | Required Edge-LLM plugin path |
| `prompt` | string | scene-description prompt | Prompt applied to every sampled frame |
| `sample_period_seconds` | double | `2.0` | Minimum ROS timestamp interval between samples |
| `max_generate_length` | int | `256` | Maximum generated tokens |
| `temperature` | double | `0.2` | Sampling temperature |
| `top_p` | double | `0.9` | Nucleus-sampling probability |
| `top_k` | int | `20` | Top-k sampling |
| `image_max_width` | int | `1280` | Maximum input width before resizing |
| `jpeg_quality` | int | `90` | In-memory JPEG quality passed to the worker |
| `drop_old_frames` | bool | `true` | Replace a queued stale frame with the newest frame |
| `publish_results` | bool | `true` | Publish result messages |

Prompt and context configuration beyond this single prompt is tracked in
[#12](https://github.com/danmartinez78/ros2-VLM-demo/issues/12).

## Operational behavior

- The ROS callback never performs inference.
- A single pending-frame slot prevents stale-frame and memory accumulation.
- Engines and the CUDA context remain loaded in the persistent worker.
- Launch respawns a worker that exits unexpectedly after two seconds.
- A failed IPC request is reported once and is not automatically replayed.
- The next sampled frame reconnects to a replacement worker.

The current timeout does not kill a worker that remains alive while wedged in a
GPU call. Active watchdog recovery is tracked in
[#6](https://github.com/danmartinez78/ros2-VLM-demo/issues/6).

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
- No RViz2 result display yet
  ([#10](https://github.com/danmartinez78/ros2-VLM-demo/issues/10)).
- No task-level quality evaluation harness yet
  ([#11](https://github.com/danmartinez78/ros2-VLM-demo/issues/11)).

## License

Apache 2.0. See [LICENSE](LICENSE).
