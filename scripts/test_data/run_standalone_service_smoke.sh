#!/usr/bin/env bash
# Smoke-test the standalone service with two sequential ROS-free CLI clients.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /absolute/path/to/image.jpg" >&2
  exit 2
fi

image_path=$1
if [[ ! -f "$image_path" ]]; then
  echo "Image not found: $image_path" >&2
  exit 2
fi

: "${ROS_WORKSPACE:=${HOME}/ros2_ws}"
: "${COSMOS_LLM_ENGINE_DIR:?source scripts/cosmos_env.sh first}"
: "${COSMOS_MULTIMODAL_ENGINE_DIR:?source scripts/cosmos_env.sh first}"
: "${EDGELLM_PLUGIN_PATH:?source scripts/cosmos_env.sh first}"

install_root="${ROS_WORKSPACE}/install/cosmos_ros2_video_reasoner"
service="${install_root}/lib/cosmos_ros2_video_reasoner/cosmos_inference_worker"
client="${install_root}/lib/cosmos_ros2_video_reasoner/cosmos_reasoning_cli"

for executable in "$service" "$client"; do
  if [[ ! -x "$executable" ]]; then
    echo "Executable not found: $executable" >&2
    echo "Build and source the workspace before running this test." >&2
    exit 2
  fi
done

run_dir=$(mktemp -d /tmp/cosmos-standalone-smoke.XXXXXX)
socket_path="${run_dir}/reasoning.sock"
service_log="${run_dir}/service.log"
service_pid=""

cleanup()
{
  if [[ -n "$service_pid" ]] && kill -0 "$service_pid" 2>/dev/null; then
    kill -TERM "$service_pid" 2>/dev/null || true
    wait "$service_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

"$service"   "$COSMOS_LLM_ENGINE_DIR"   "$COSMOS_MULTIMODAL_ENGINE_DIR"   "$EDGELLM_PLUGIN_PATH"   "$socket_path"   90   60 >"$service_log" 2>&1 &
service_pid=$!

echo "Standalone service PID: $service_pid"
echo "Service log: $service_log"

for _ in $(seq 1 600); do
  if [[ -S "$socket_path" ]] && grep -q "worker ready" "$service_log"; then
    break
  fi
  if ! kill -0 "$service_pid" 2>/dev/null; then
    echo "FAIL: service exited during initialization." >&2
    sed -n '1,240p' "$service_log" >&2
    exit 1
  fi
  sleep 0.2
done

if [[ ! -S "$socket_path" ]]; then
  echo "FAIL: service socket was not ready before timeout." >&2
  sed -n '1,240p' "$service_log" >&2
  exit 1
fi

for request_number in 1 2; do
  echo
  echo "==> Standalone inference request $request_number"
  "$client"     --socket "$socket_path"     --image "$image_path"     --prompt "Describe the current scene concisely."     --max-generate-length 64

  if ! kill -0 "$service_pid" 2>/dev/null; then
    echo "FAIL: service exited after client $request_number disconnected." >&2
    sed -n '1,240p' "$service_log" >&2
    exit 1
  fi
done

echo
echo "PASS: two sequential CLI clients completed with service PID $service_pid unchanged."
echo "Artifacts preserved at: $run_dir"

# Preserve artifacts while preventing the EXIT trap from printing shell noise.
cleanup
service_pid=""
trap - EXIT INT TERM
