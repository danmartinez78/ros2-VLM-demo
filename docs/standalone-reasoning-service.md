# Standalone reasoning service

The TensorRT Edge-LLM runtime is a ROS-free process. ROS 2, command-line
experiments, and future evaluation or web tools connect through the same
versioned Unix-domain-socket protocol.

This separation is intentional:

- CUDA, TensorRT, engines, tokenization, and generation live in the service.
- ROS owns camera subscription, sampling, timestamps, QoS, and result topics.
- Experiment tools can exercise inference without DDS, rosbag playback, or ROS
  process supervision.

The current protocol supports one BGR image, structured or inline instructions,
bounded observation history, generation parameters, and a response. Ordered
frame windows, model sessions, and persistent recurrent state require a future
protocol version.

## Start the service

On a prepared Thor:

```bash
cd "$HOME/ros2-VLM-demo"
source scripts/edge_vlm_env.sh
source "$ROS_WORKSPACE/install/setup.bash"

SOCKET_PATH=/tmp/edge_vlm.sock

"$ROS_WORKSPACE/install/edge_vlm_ros/lib/edge_vlm_ros/edge_vlm_server" \
  "$EDGE_VLM_LLM_ENGINE_DIR" \
  "$EDGE_VLM_MULTIMODAL_ENGINE_DIR" \
  "$EDGELLM_PLUGIN_PATH" \
  "$SOCKET_PATH" \
  90 \
  60
```

The service loads the engines once and then accepts sequential clients. Closing
a CLI or ROS client does not unload the model. SIGINT or SIGTERM removes the
socket during shutdown.

Only one client is served at a time in this milestone. Request scheduling and
concurrent sessions are deliberately deferred.

## Run image inference without ROS

In another terminal:

```bash
source "$HOME/ros2_ws/install/setup.bash"

"$HOME/ros2_ws/install/edge_vlm_ros/lib/edge_vlm_ros/edge_vlm_cli" \
  --socket /tmp/edge_vlm.sock \
  --image /absolute/path/to/image.jpg \
  --sequence-type images \
  --prompt "Describe the current scene concisely." \
  --max-generate-length 64
```

Run the command again with another image to verify that the service remains
loaded across client connections.

For multi-frame temporal contracts:

```bash
"$HOME/ros2_ws/install/edge_vlm_ros/lib/edge_vlm_ros/edge_vlm_cli" \
  --socket /tmp/edge_vlm.sock \
  --image /abs/frame_000.jpg \
  --image /abs/frame_001.jpg \
  --sequence-type temporal_images \
  --fps 8 \
  --frame-timestamps-sec 0.0,0.125 \
  --prompt "Summarize scene changes as compact JSON." \
  --max-generate-length 64
```

At the pinned TensorRT Edge-LLM commit used by this repository:

- `sequence_type=images` uses N independent `image` content items and N image
  buffers;
- `sequence_type=temporal_images|video` uses one native `video` content item
  and one stacked `ImageData` (`[T,H,W,3]`, `isVideo=true`, effective fps,
  optional timestamps).

The server reports effective runtime representation via:

- `Requested sequence type: ...`
- `Runtime temporal encoding: ...`
- `Temporal fallback used: true|false`

For a one-command Thor smoke test that starts the service, invokes two sequential
clients, verifies that the service PID remains unchanged, and cleans up:

```bash
source scripts/edge_vlm_env.sh
source "$ROS_WORKSPACE/install/setup.bash"
bash scripts/test_data/run_standalone_service_smoke.sh /absolute/path/to/image.jpg
```

## Connect the ROS adapter to the existing service

```bash
source /opt/ros/jazzy/setup.bash
source "$HOME/ros2_ws/install/setup.bash"

ros2 launch edge_vlm_ros edge_vlm.launch.py \
  start_worker:=false \
  worker_socket_path:=/tmp/edge_vlm.sock \
  image_topic:=/camera/image_raw
```

With `start_worker:=true` (the default), launch retains the established
behavior: it starts the worker and respawns it after watchdog termination.

## Ownership and cleanup

Do not run a launch-managed worker and a standalone service with the same socket
path. The service unlinks its configured socket before binding, so a second
instance could disrupt the first instance's endpoint.

For experiments, give each independently managed service a unique socket path.
Before deleting a stale socket, confirm that no worker process owns it.

## Next protocol milestones

The standalone boundary is intentionally established before changing transport.
Future versions can add:

- explicit session creation, continuation, reset, and close operations;
- capability and model metadata queries;
- structured result schemas;
- cancellation and richer health reporting;
- shared-memory media transfer when measured payload cost justifies it.

Persistent Mamba/SSM state must be an explicit session capability. It must not
be inferred merely from the model architecture.
