# Jetson AGX Thor deployment recipe

This is the canonical deployment and validation recipe for
`edge_vlm_ros` on NVIDIA Jetson AGX Thor.

## Validated baseline

- Ubuntu 24.04, aarch64
- JetPack 7.2 / Jetson Linux R39.2
- CUDA 13.2
- ROS 2 Jazzy
- TensorRT Edge-LLM built on the target Thor for `sm_110a`
- Cosmos-Reason2-8B NVFP4 engine bundle (model-specific)

Do not treat nearby JetPack, CUDA, TensorRT, or engine versions as binary
compatible. TensorRT engines and CUDA-linked Edge-LLM artifacts should be built
and validated on the target software stack.

## 1. Confirm the base system

```bash
cat /etc/nv_tegra_release
dpkg-query -W nvidia-l4t-core
uname -m
free -h
df -h /
```

Expected key values are R39.2 and `aarch64`.

A fresh flash may contain the BSP without the complete developer toolchain.
Install the JetPack metapackage if `nvcc` and TensorRT headers are absent:

```bash
sudo apt update
sudo apt install nvidia-jetpack
```

Afterward:

```bash
/usr/local/cuda/bin/nvcc --version
dpkg-query -W nvidia-jetpack
```

Do not accept an APT transaction that removes `nvidia-jetpack`,
`nvidia-jetpack-dev`, or `nvidia-opencv-dev` to install an Ubuntu OpenCV
variant.

## 2. Prepare TensorRT Edge-LLM and the model

Follow NVIDIA's TensorRT Edge-LLM instructions and the Jetson AI Lab guide:

- <https://www.jetson-ai-lab.com/tutorials/tensorrt-edge-llm/#tensorrt-edge-llm-on-jetson>
- <https://github.com/NVIDIA/TensorRT-Edge-LLM>

Use the versions appropriate to the installed JetPack release. Build
Edge-LLM on Thor with its Thor/Blackwell target and CuTe DSL artifacts enabled.
The deployment expects:

```text
$HOME/TensorRT-Edge-LLM/build/cpp/libedgellmCore.a
$HOME/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so
```

The plugin may be a symlink to a versioned file:

```bash
cd "$HOME/TensorRT-Edge-LLM/build"
ln -sfn libNvInfer_edgellm_plugin.so.1.0 libNvInfer_edgellm_plugin.so
```

Only create that symlink when the matching versioned library actually exists.

Quantize and build the Cosmos engine following the same guide. The validated
layout is:

```text
$HOME/tensorrt-edgellm-workspace/Cosmos-Reason2-8B/engine/
$HOME/tensorrt-edgellm-workspace/Cosmos-Reason2-8B/engine/llm/
```

The `engine` directory contains the visual engine. `engine/llm` contains
`llm.engine`, its configuration, tokenizer files, embedding data, and the
processed chat template.

The runtime may report a missing `engine/action/action.engine`. That probe is
optional for image reasoning and is not a deployment failure.

## 3. Verify native inference first

Before adding ROS, use NVIDIA's native executable with a known image. This
separates engine/runtime failures from ROS integration failures.

```bash
cd "$HOME/TensorRT-Edge-LLM"
export EDGELLM_PLUGIN_PATH="$PWD/build/libNvInfer_edgellm_plugin.so"
export WORKSPACE_DIR="$HOME/tensorrt-edgellm-workspace"
export MODEL_NAME="Cosmos-Reason2-8B"

./build/examples/llm/llm_inference \
  --engineDir "$WORKSPACE_DIR/$MODEL_NAME/engine/llm" \
  --multimodalEngineDir "$WORKSPACE_DIR/$MODEL_NAME/engine" \
  --inputFile "$WORKSPACE_DIR/input_vlm.json" \
  --outputFile "$WORKSPACE_DIR/output_vlm.json" \
  --maxGenerateLength 64 \
  --dumpOutput
```

Do not continue until the native executable returns a coherent description.

## 4. Install ROS and repository dependencies

Clone the repository inside a ROS workspace:

```bash
mkdir -p "$HOME/ros2_ws/src"
cd "$HOME/ros2_ws/src"
git clone https://github.com/danmartinez78/ros2-VLM-demo.git
cd ros2-VLM-demo
```

Run setup as your normal user. The script uses `sudo` only for system changes:

```bash
bash scripts/setup_deployment.sh
```

Options:

```bash
bash scripts/install_dependencies.sh --desktop
bash scripts/install_dependencies.sh --isaac-ros
```

Isaac ROS is optional. Version 4.5 is configured in Docker mode to avoid
replacing the host CUDA, TensorRT, or NVIDIA OpenCV packages. NVIDIA's listed
Thor validation matrix may lag the JetPack 7.2 host used here, so the verifier
reports that combination as a warning.

If Docker group membership changes, log out and back in before continuing.

## 5. Configure paths

The first setup run creates `scripts/edge_vlm_env.sh` from the tracked example.
Review it:

```bash
cd "$HOME/ros2_ws/src/ros2-VLM-demo"
cp -n scripts/edge_vlm_env.sh.example scripts/edge_vlm_env.sh
${EDITOR:-nano} scripts/edge_vlm_env.sh
```

A typical Thor configuration is:

```bash
export ROS_DISTRO="jazzy"
export ROS_WORKSPACE="$HOME/ros2_ws"
export TENSORRT_EDGE_LLM_ROOT="$HOME/TensorRT-Edge-LLM"
export TENSORRT_EDGE_LLM_BUILD_DIR="$TENSORRT_EDGE_LLM_ROOT/build"
export TRT_PACKAGE_DIR="/usr"
export EDGE_VLM_MODEL_NAME="Cosmos-Reason2-8B"
export EDGE_VLM_WORKSPACE_DIR="$HOME/tensorrt-edgellm-workspace"
export EDGE_VLM_LLM_ENGINE_DIR="$EDGE_VLM_WORKSPACE_DIR/$EDGE_VLM_MODEL_NAME/engine/llm"
export EDGE_VLM_MULTIMODAL_ENGINE_DIR="$EDGE_VLM_WORKSPACE_DIR/$EDGE_VLM_MODEL_NAME/engine"
export EDGELLM_PLUGIN_PATH="$TENSORRT_EDGE_LLM_BUILD_DIR/libNvInfer_edgellm_plugin.so"
```

## 6. Build and verify

```bash
cd "$HOME/ros2_ws/src/ros2-VLM-demo"
source scripts/edge_vlm_env.sh
bash scripts/build_workspace.sh
source "$ROS_WORKSPACE/install/setup.bash"
bash scripts/verify_deployment.sh
```

The verifier checks artifacts, engines, ROS executables, and the required
process isolation.

Confirm Thor CUDA images if diagnosing architecture errors:

```bash
/usr/local/cuda/bin/cuobjdump --list-elf \
  "$ROS_WORKSPACE/install/edge_vlm_ros/lib/edge_vlm_ros/edge_vlm_server" \
  | grep -E 'sm_[0-9]+'
```

The Thor worker should include `sm_110a`. A stale `sm_75` image will fail with
`device kernel image is invalid`.

## 7. Run the validated rosbag test

```bash
cd "$HOME/ros2_ws/src/ros2-VLM-demo"
source scripts/edge_vlm_env.sh
source "$ROS_WORKSPACE/install/setup.bash"
bash scripts/test_data/run_image_proc_test.sh
```

Expected behavior:

1. launch starts `edge_vlm_server` and `edge_vlm_ros_node`;
2. the worker loads the LLM, tokenizer, and visual engine once;
3. the bag publishes `/hawk_0_left_rgb_image`;
4. the ROS process prints coherent results and publishes `/vlm/result`.

Run the crash-recovery integration test:

```bash
bash scripts/test_data/run_worker_recovery_test.sh
```

The final line should be:

```text
PASS: all 6 verifications passed — watchdog fires, worker PID changes, exactly
      one failure published, reasoning resumes, edge_vlm_ros_node PID unchanged,
      and no orphan worker or socket remains after shutdown.
```

### Thor hardware validation: watchdog-triggered recovery (requires hardware)

> **Requires Thor hardware to run. Validated Cosmos configuration reliably
> exceeds the 1-second test deadline, making the watchdog fire deterministically
> without injecting an artificial hang.**

The script `scripts/test_data/run_worker_recovery_test.sh` runs the full
end-to-end test automatically and verifies all six required properties.

Manual recipe (same as the script):

```bash
# The 1-second deadline is shorter than any real Cosmos inference call on Thor.
# worker_request_timeout_seconds (20 s) must exceed the deadline (1 s) — the
# launch file validates this relationship and fails fast if it is violated.
bash scripts/test_data/run_worker_recovery_test.sh
```

The script verifies:
1. The `WATCHDOG: inference deadline` diagnostic appears in the launch log.
2. The worker PID changes after launch respawns it.
3. Exactly one failure result is published for the expired request.
4. A successful reasoning result is received after the replacement worker starts.
5. The `edge_vlm_ros_node` PID does not change (supervisor survives).
6. No orphan worker process or socket file remains after shutdown.

## 8. Run a custom bag or camera

Inspect the source:

```bash
bash scripts/test_data/inspect_rosbag.sh /absolute/path/to/bag
```

Start the pipeline:

```bash
ros2 launch edge_vlm_ros edge_vlm.launch.py \
  image_topic:=/actual/raw/image/topic \
  llm_engine_dir:="$EDGE_VLM_LLM_ENGINE_DIR" \
  multimodal_engine_dir:="$EDGE_VLM_MULTIMODAL_ENGINE_DIR" \
  edge_llm_plugin_path:="$EDGELLM_PLUGIN_PATH" \
  use_sim_time:=true
```

Play a bag in another terminal:

```bash
ros2 bag play /absolute/path/to/bag --clock
```

Use `use_sim_time:=false` for a live camera unless the system publishes a
simulation clock.

Supported raw encodings are `bgr8`, `rgb8`, and `mono8`. Compressed or H.264
streams must be decoded to `sensor_msgs/msg/Image` first.

## Troubleshooting quick reference

| Symptom | Action |
| --- | --- |
| Docker permission denied | Log out and back in after joining the `docker` group |
| `AMENT_TRACE_SETUP_FILES` unbound | Source ROS setup with `nounset` temporarily disabled; repository scripts already do this |
| `device kernel image is invalid` | Rebuild Edge-LLM and the ROS worker for `sm_110a` |
| Missing unversioned plugin | Validate the versioned library, then create the symlink |
| `engine/action/action.engine` missing | Ignore for image-only Cosmos Reason2 inference |
| APT proposes removing JetPack/OpenCV packages | Cancel the transaction; do not downgrade NVIDIA OpenCV |
| No results | Confirm the exact topic, raw encoding, subscriber count, and `--clock`/`use_sim_time` pairing |
| Worker exits | Launch respawns it; one frame may fail before IPC reconnects |
| Worker alive but wedged | Watchdog fires after `worker_inference_deadline_seconds` (default 60 s); worker self-terminates; launch respawns it; one error is published; reasoning resumes on next frame |

For the full historical investigation, see
[thor-edge-llm-prefill-stall-rca.md](thor-edge-llm-prefill-stall-rca.md).

## Migration from `cosmos_ros2_video_reasoner`

This package was renamed from `cosmos_ros2_video_reasoner` to `edge_vlm_ros`.
ROS package renames cannot safely reuse old colcon artifacts.

Run these commands on an existing Thor checkout **before** rebuilding:

```bash
# Remove stale colcon artifacts for the old and new package
rm -rf "${ROS_WORKSPACE}/build/cosmos_ros2_video_reasoner"
rm -rf "${ROS_WORKSPACE}/install/cosmos_ros2_video_reasoner"
rm -rf "${ROS_WORKSPACE}/build/edge_vlm_ros"
rm -rf "${ROS_WORKSPACE}/install/edge_vlm_ros"

# Clear workspace logs (optional but recommended)
rm -rf "${ROS_WORKSPACE}/log"

# Rebuild
cd "${ROS_WORKSPACE}"
colcon build --packages-select edge_vlm_ros
```

Also rename your local environment file:

```bash
# If you have a customised env file from the old name, carry it over
if [[ -f scripts/cosmos_env.sh && ! -f scripts/edge_vlm_env.sh ]]; then
  cp scripts/cosmos_env.sh scripts/edge_vlm_env.sh
fi
```

Update any `COSMOS_*` environment variables in your env file to the new
`EDGE_VLM_*` equivalents (see `scripts/edge_vlm_env.sh.example`).
