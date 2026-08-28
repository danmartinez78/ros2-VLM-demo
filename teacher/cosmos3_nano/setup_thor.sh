#!/usr/bin/env bash
set -euo pipefail

IMAGE="${COSMOS3_NANO_VLLM_IMAGE:-vllm/vllm-openai:v0.23.0-aarch64-ubuntu2404}"
TOKEN_PATH="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "warning: validated target is aarch64/Jetson AGX Thor; detected $(uname -m)" >&2
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required before running this setup script." >&2
  exit 1
fi

missing_packages=()
command -v curl >/dev/null 2>&1 || missing_packages+=(curl)
command -v jq >/dev/null 2>&1 || missing_packages+=(jq)
if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  missing_packages+=(ffmpeg)
fi

if ((${#missing_packages[@]})); then
  echo "Installing host utilities: ${missing_packages[*]}"
  sudo apt-get update
  sudo apt-get install -y "${missing_packages[@]}"
fi

if [[ ! -s "$TOKEN_PATH" ]]; then
  echo "Hugging Face token not found at: $TOKEN_PATH" >&2
  echo "Authenticate first or set HF_TOKEN_PATH to the stored token file." >&2
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "Docker is not accessible as the current user." >&2
  echo "Check membership in the docker group and /var/run/docker.sock permissions." >&2
  exit 1
fi

echo "Pulling $IMAGE ..."
docker pull "$IMAGE"

echo "Verifying Cosmos3 support in the container..."
docker run --rm \
  --entrypoint python3 \
  "$IMAGE" \
  -c 'import vllm, transformers; print("vLLM:", vllm.__version__); print("Transformers:", transformers.__version__); from transformers import Cosmos3OmniForConditionalGeneration; print("Cosmos3 Omni: OK")'

mkdir -p "$HOME/cosmos3_teacher/videos"

echo
echo "Setup complete."
echo "Validated image: $IMAGE"
echo "HF token: $TOKEN_PATH"
echo "Teacher data root: $HOME/cosmos3_teacher"
echo "Next: teacher/cosmos3_nano/start_vllm_thor.sh"
