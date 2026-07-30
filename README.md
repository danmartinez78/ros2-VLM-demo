# cosmos_ros2_video_reasoner

ROS 2 Jazzy pipeline for **NVIDIA Jetson AGX Thor** that samples camera frames
from a ROS 2 bag or live camera, sends each frame to a persistent
[NVIDIA Cosmos Reason2](https://developer.nvidia.com/cosmos) TensorRT
Edge-LLM VLM engine, and publishes structured reasoning results.

---

## Architecture

```mermaid
flowchart LR
    BAG[ros2 bag play\n/camera/image_raw] -->|sensor_msgs/Image| SUB
    subgraph cosmos_reasoner node
        SUB[Subscription\ncallback] -->|timestamp\nsampling| QUEUE[(Queue\ndepth 1)]
        QUEUE --> WORKER[Inference\nworker thread]
        WORKER -->|cv_bridge + resize| ENC[JPEG encode]
        ENC --> BACKEND[TensorRTEdgeLLM\nBackend]
        BACKEND --> PUBLISH[Publish\nVisionReasoningResult]
    end
    PUBLISH -->|/cosmos/reasoning| RESULT[Result topic]
    BACKEND -.->|persistent| ENGINE[(TRT engines\nloaded once)]
```

---

## Platform support

| Component | Version |
|---|---|
| Hardware | NVIDIA Jetson AGX Thor |
| JetPack | 7.2 |
| ROS 2 | Jazzy |
| TensorRT Edge-LLM | 0.9.x / current main |
| Cosmos Reason2 model | 2B or 8B |

---

## Prerequisites

### 1 — TensorRT Edge-LLM SDK

```bash
git clone https://github.com/NVIDIA/TensorRT-Edge-LLM.git ~/TensorRT-Edge-LLM
cd ~/TensorRT-Edge-LLM
git submodule update --init --recursive
# Build for Jetson AGX Thor using the same configuration used for the engines
mkdir -p build && cd build
cmake .. \\
  -DCMAKE_BUILD_TYPE=Release \\
  -DTRT_PACKAGE_DIR=/usr \\
  -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \\
  -DEMBEDDED_TARGET=jetson-thor \\
  -DCUDA_CTK_VERSION=13.0 \\
  -DENABLE_CUTE_DSL=ALL
make -j$(nproc)
```

The build produces:
* `build/libedgellmCore.a`
* `build/libNvInfer_edgellm_plugin.so`

### 2 — Cosmos Reason2 TensorRT engines

Follow the Cosmos Reason2 quantisation guide to produce engine directories:

```
~/tensorrt-edgellm-workspace/Cosmos-Reason2-2B/engine/llm/
~/tensorrt-edgellm-workspace/Cosmos-Reason2-2B/engine/
```

or the 8B variant.  The `engine/` directory must contain the visual encoder
engine; `engine/llm/` must contain the language model engine and tokenizer
files (`tokenizer.json`, `tokenizer_config.json`,
`processed_chat_template.json`).

---

## Build

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone <repository-url> cosmos_ros2_video_reasoner

cd ~/ros2_ws
colcon build \
  --symlink-install \
  --cmake-args \
    -DTENSORRT_EDGE_LLM_ROOT=$HOME/TensorRT-Edge-LLM \
    -DTENSORRT_EDGE_LLM_BUILD_DIR=$HOME/TensorRT-Edge-LLM/build \
    -DTRT_PACKAGE_DIR=/path/to/tensorrt

source install/setup.bash
```

### Build without GPU (tests only)

Omit the `-DTENSORRT_EDGE_LLM_*` flags to build the node library and run
hardware-independent tests:

```bash
colcon build --symlink-install
colcon test --packages-select cosmos_ros2_video_reasoner
```

---

## Engine path configuration

Pass engine paths as launch arguments or in the YAML config:

```bash
ros2 launch cosmos_ros2_video_reasoner cosmos_reasoner.launch.py \
  llm_engine_dir:=/absolute/path/to/engine/llm \
  multimodal_engine_dir:=/absolute/path/to/engine \
  edge_llm_plugin_path:=/absolute/path/to/libNvInfer_edgellm_plugin.so
```

> **Note**: Paths must be absolute.  `~` is not expanded inside `rclcpp`.

---

## ROS bag playback

### Discover available topics

```bash
ros2 bag info /path/to/my.bag
ros2 topic list
ros2 topic type /camera/image_raw
```

### Terminal 1 — start the node

```bash
ros2 launch cosmos_ros2_video_reasoner cosmos_reasoner.launch.py \
  image_topic:=/actual/camera/topic \
  llm_engine_dir:=/absolute/path/engine/llm \
  multimodal_engine_dir:=/absolute/path/engine \
  edge_llm_plugin_path:=/absolute/path/libNvInfer_edgellm_plugin.so \
  use_sim_time:=true
```

### Terminal 2 — play the bag

```bash
ros2 bag play /path/to/my.bag --clock
```

### Topic remapping (alternative)

```bash
ros2 launch cosmos_ros2_video_reasoner cosmos_reasoner.launch.py \
  ... \
  --ros-args --remap /camera/image_raw:=/your/actual/topic
```

---

## Example output

```
[INFO] [cosmos_reasoner]: Subscribed to /camera/image_raw
[INFO] [cosmos_reasoner]: Publishing results to /cosmos/reasoning
[INFO] [cosmos_reasoner]: [frame 1 | 1.243 s] The scene shows a busy urban
  intersection. A pedestrian wearing a red jacket is crossing the street.
  Two cars — a white sedan and a dark SUV — are stopped at a traffic light.
  The weather appears overcast. No hazards identified.
[INFO] [cosmos_reasoner]: [frame 2 | 1.187 s] ...
```

---

## Parameter reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image_topic` | string | `/camera/image_raw` | Input image topic |
| `result_topic` | string | `/cosmos/reasoning` | Output result topic |
| `llm_engine_dir` | string | `""` | **Required** — absolute path to LLM engine dir |
| `multimodal_engine_dir` | string | `""` | **Required** — absolute path to visual encoder engine dir |
| `edge_llm_plugin_path` | string | `""` | Absolute path to plugin `.so` (or set `EDGELLM_PLUGIN_PATH`) |
| `prompt` | string | (scene description) | Text prompt sent for every frame |
| `sample_period_seconds` | double | `2.0` | Minimum seconds between sampled frames (message timestamp) |
| `max_generate_length` | int | `256` | Maximum tokens to generate |
| `temperature` | double | `0.2` | Sampling temperature |
| `top_p` | double | `0.9` | Nucleus sampling probability |
| `top_k` | int | `20` | Top-k sampling |
| `image_max_width` | int | `1280` | Resize if wider (aspect preserved) |
| `jpeg_quality` | int | `90` | JPEG quality for VLM ingestion |
| `drop_old_frames` | bool | `true` | Drop pending frame when worker busy |
| `publish_results` | bool | `true` | Publish result messages |
| `dump_profile` | bool | `false` | Enable Edge-LLM profiling |

---

## Troubleshooting

### Missing plugin: `Cannot open plugin library`

```
[ERROR] [cosmos_reasoner]: Failed to load TensorRT Edge-LLM plugin library.
```

* Verify `edge_llm_plugin_path` points to the built
  `libNvInfer_edgellm_plugin.so`.
* Or set `export EDGELLM_PLUGIN_PATH=/path/to/libNvInfer_edgellm_plugin.so`.
* Check `LD_LIBRARY_PATH` includes the TensorRT library directory.

### Incompatible engines: `Failed to deserialize engine`

Engines were built with a different TensorRT version.  Rebuild the Cosmos
engines against the TensorRT version installed with JetPack 7.2.

### CUDA initialisation failure

```
[ERROR] [cosmos_reasoner]: cudaStreamCreate failed
```

* Verify CUDA runtime is installed: `nvcc --version`.
* On Jetson, ensure `nvidia-container-toolkit` or direct GPU access is available.

### Image encoding failure: `cv::imencode failed`

* cv_bridge received a non-standard image encoding.  Check the bag's image
  encoding with `ros2 topic echo /camera/image_raw --once`.
* Supported encodings: `bgr8`, `rgb8`, `mono8`, `16uc1` (cv_bridge will
  convert).

### Wrong bag topic

```
[WARN] [cosmos_reasoner]: No messages received on /camera/image_raw
```

* Use `ros2 bag info` to find the correct topic name and pass it via
  `image_topic:=` or `--ros-args --remap`.

### Slow inference / high frame drop rate

Inference on 2B/8B models at full resolution may take 1–3 seconds per frame.
Increase `sample_period_seconds` to reduce load.  Ensure only one `cosmos_reasoner`
node is running (running two loads two copies of the engines into VRAM).

---

## Performance notes

**Why keep engines loaded?** Deserialising a Cosmos Reason2 TensorRT engine
takes 15–60 seconds.  Loading once at startup and reusing the runtime for all
frames reduces per-frame latency from minutes to seconds.

**Why a bounded queue?** The inference worker is single-threaded and
single-pass (no continuous batching).  Allowing an unbounded queue would
accumulate stale frames and grow memory without bound during a long bag
replay.  With `drop_old_frames = true` the queue remains at depth 1 and the
worker always sees the freshest available frame.

---

## Known limitations

* **Sampled images, not video**: The ROS node currently processes independent sampled frames; its Edge-LLM integration does not yet expose native
  video-stream input.  Frames are treated as independent images.
* **CompressedImage support**: Follow-up task — subscribe to
  `sensor_msgs/msg/CompressedImage` via `image_transport` to avoid
  intermediate decoding when the source is a compressed bag.
* **Batch size 1**: Single-request inference only; no batching across frames.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
