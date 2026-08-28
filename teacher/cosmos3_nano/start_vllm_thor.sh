#!/usr/bin/env bash
set -euo pipefail

IMAGE="${COSMOS3_NANO_VLLM_IMAGE:-vllm/vllm-openai:v0.23.0-aarch64-ubuntu2404}"
MODEL="${COSMOS3_NANO_MODEL:-nvidia/Cosmos3-Nano}"
SERVED_MODEL_NAME="${COSMOS3_NANO_SERVED_NAME:-cosmos3-nano-teacher}"
TOKEN_PATH="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"
DATA_ROOT="${COSMOS3_NANO_DATA_ROOT:-$HOME/cosmos3_teacher}"
PORT="${COSMOS3_NANO_PORT:-8000}"
MAX_MODEL_LEN="${COSMOS3_NANO_MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTILIZATION="${COSMOS3_NANO_GPU_MEMORY_UTILIZATION:-0.55}"

if [[ ! -s "$TOKEN_PATH" ]]; then
  echo "Hugging Face token not found at: $TOKEN_PATH" >&2
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "Docker is not accessible as the current user." >&2
  exit 1
fi

mkdir -p "$DATA_ROOT/videos"
export HF_TOKEN="$(cat "$TOKEN_PATH")"

echo "Starting Cosmos3-Nano teacher server"
echo "  image:        $IMAGE"
echo "  checkpoint:   $MODEL"
echo "  served name:  $SERVED_MODEL_NAME"
echo "  data root:    $DATA_ROOT -> /data"
echo "  port:         $PORT"
echo "  context:      $MAX_MODEL_LEN"
echo "  memory util:  $GPU_MEMORY_UTILIZATION"

# vllm/vllm-openai images use ENTRYPOINT ["vllm", "serve"], so the model is
# intentionally the first argument after the image name.
docker run --rm -it \
  --runtime=nvidia \
  --network host \
  --shm-size=16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e HF_TOKEN="$HF_TOKEN" \
  -e VLLM_USE_DEEP_GEMM=0 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$DATA_ROOT:/data" \
  "$IMAGE" \
  "$MODEL" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --tensor-parallel-size 1 \
    --mm-encoder-tp-mode data \
    --async-scheduling \
    --allowed-local-media-path /data \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --enable-chunked-prefill \
    --mm-processor-cache-gb 0 \
    --media-io-kwargs '{"video":{"num_frames":-1}}' \
    --port "$PORT"
