# FlashRT native temporal/video path (Thor)

This directory contains the first repo-integrated Cosmos3 native-video path for
`edge_vlm_ros`. It is intentionally additive: the existing C++ TensorRT
Edge-LLM node/worker remains the single-frame and ordered-multi-image baseline,
while this path uses FlashRT for a rolling temporal window.

For the complete shared IPC request format, field meanings, sequence modes, prompt/message roles, and a side-by-side description of how Edge-LLM and FlashRT invoke inference, see [`docs/inference-request-contract.md`](../docs/inference-request-contract.md).

## Why this exists

Thor validation established that:

- Cosmos3-Edge single-image inference is correct after the documented patch-layout fix.
- TensorRT Edge-LLM 0.10.0 does not expose native video to the Cosmos3 reasoner CLI.
- FlashRT `CosmosReasonerThor` accepts an ordered frame list as a native video media item and successfully reasons about temporal scene changes.
- The existing repo IPC v3 contract already carries ordered frames, `sequence_type`, FPS, and timestamps, so the transport does not need to change.

The remaining integration problem was therefore at the endpoints: buffer a
chronological ROS camera window and provide a worker that interprets
`sequence_type=video` natively instead of degrading to ordered images.

## Components

- `flashrt_ipc_worker.py` - persistent Unix-domain-socket inference worker. It speaks the existing IPC v3 wire protocol and maps `video`/`temporal_images` requests to Cosmos3 native video preprocessing + FlashRT inference.
- `temporal_ros_node.py` - experimental ROS 2 node that subscribes to `sensor_msgs/msg/Image`, keeps a rolling sampled frame window, sends that window as one `sequence_type=video` IPC request, and publishes the existing `edge_vlm_ros/msg/VlmResult` message.
- `run_worker_docker.sh` - launches the FlashRT worker in the validated Thor container while sharing only the checkpoint, this directory, and the Unix socket directory with the host.
- `Dockerfile` - reproducible derived image recipe matching the validated FlashRT + Cosmos Framework environment.

## Pinned validation references

The Thor proof-of-concept used:

- FlashRT commit `f72192b263b267994edd7bbff0a8c62c6da98948`
- NVIDIA cosmos-framework commit `ed8287fd7477113f8ac4f6b84290514d55cf0cdc`
- NVIDIA Cosmos cookbook assets commit `19b2f1b2a8036d31c7a29a66966a52c71c97a56d`

Keep these pins when reproducing the environment until newer revisions are explicitly revalidated.

## Build the derived image on Thor

Use a build context that contains the pinned `cosmos-framework` checkout. For example, from the FlashRT validation directory:

```bash
cp /path/to/ros2-VLM-demo/flashrt_temporal/Dockerfile Dockerfile.cosmos3-video
sudo docker build -f Dockerfile.cosmos3-video -t flashrt:cosmos3-video .
```

The base image `flashrt:cosmos3-thor` must already contain the compiled Thor
FlashRT Cosmos3 kernels (`flash_rt.flash_rt_kernels`).

## Run

Terminal 1, from the repo root. Prefix with `sudo` when Docker socket permissions require it:

```bash
sudo bash flashrt_temporal/run_worker_docker.sh
```

The worker creates `/tmp/edge_vlm_flashrt.sock` with host-connectable permissions.

Terminal 2, after building/sourcing the existing ROS package so
`edge_vlm_ros.msg.VlmResult` is importable:

```bash
source install/setup.bash
python3 flashrt_temporal/temporal_ros_node.py --ros-args \
  -p image_topic:=/camera/image_raw \
  -p sample_period_seconds:=0.5 \
  -p temporal_window_frames:=8 \
  -p temporal_require_full_window:=true \
  -p worker_socket_path:=/tmp/edge_vlm_flashrt.sock
```

This yields an 8-frame rolling window sampled at roughly 2 Hz. Each completed
window is one Cosmos3 inference call. The result is published on `/vlm/result`
using the existing message type; `tracker_context` records the runtime temporal
encoding and frame count.

## Runtime semantics

- ROS message timestamps determine sampling and are sent with every temporal frame window.
- The worker derives effective video FPS from those timestamps when an explicit FPS is absent, avoiding assumptions about camera publish rate or bag replay speed.
- `sequence_type=video` and `sequence_type=temporal_images` both use Cosmos3's native video media path in this FlashRT worker.
- `sequence_type=images` remains an image/multi-image request; timing metadata is not used by the image preprocessing path.
- The generic C++ request validator forbids FPS/timestamps on `sequence_type=images`. The experiment-only `terminal_only` chronology control currently carries a one-element timestamp through the Python IPC client; the FlashRT worker accepts it but does not use it. Do not rely on that as a portable image-mode contract.
- FlashRT's current Thor reasoner path is greedy. The IPC sampling fields remain on the wire for compatibility, but temperature/top-p/top-k are not applied by this worker.
- The worker has a process-level deadline guard. If a CUDA call wedges beyond the configured deadline, the process exits so an external supervisor can restart it.

## Current scope

This is the research/MVP integration, not yet a replacement for the C++ node.
It intentionally does not duplicate the C++ node's tracked-observation adapter,
prompt-history policy, or RViz-specific conveniences. Once temporal experiments
establish the desired window sizes and output schema, the rolling-window logic
can be moved into `VlmReasonerNode` while keeping this same IPC worker and wire
contract.
