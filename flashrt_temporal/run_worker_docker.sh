#!/usr/bin/env bash
set -euo pipefail

# Run from the ros2-VLM-demo repository root. The caller may prefix this script
# with sudo when Docker socket permissions require it on Thor.
CHECKPOINT=${CHECKPOINT:-$HOME/tensorrt-edgellm-workspace/Cosmos3-Edge/hf_checkpoint}
IMAGE=${FLASHRT_IMAGE:-flashrt:cosmos3-video}
SOCKET_PATH=${WORKER_SOCKET_PATH:-/tmp/edge_vlm_flashrt.sock}
QUANT=${FLASHRT_QUANT:-bf16}
DEADLINE=${WORKER_INFERENCE_DEADLINE_SECONDS:-60}

if [[ ! -d "$CHECKPOINT" ]]; then
  echo "Checkpoint directory does not exist: $CHECKPOINT" >&2
  exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
socket_dir=$(dirname "$SOCKET_PATH")
socket_name=$(basename "$SOCKET_PATH")

exec docker run --rm \
  --runtime=nvidia \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$repo_root/flashrt_temporal:/opt/edge_vlm_flashrt:ro" \
  -v "$CHECKPOINT:/model:ro" \
  -v "$socket_dir:/ipc" \
  "$IMAGE" \
  python /opt/edge_vlm_flashrt/flashrt_ipc_worker.py \
    --checkpoint /model \
    --socket-path "/ipc/$socket_name" \
    --quant "$QUANT" \
    --inference-deadline-seconds "$DEADLINE"
