#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
env_file="${EDGE_VLM_ENV_FILE:-${script_dir}/edge_vlm_env.sh}"
source "${script_dir}/ros_setup_guard.sh"
run_standalone_smoke=0
smoke_image=""
passthrough_args=()
failures=0

usage() {
  cat <<'EOF'
Usage: ./scripts/verify_thor_jp72.sh [--isaac-ros] [--smoke-image /abs/path.jpg]

Verifies the supported Thor JP7.2 deployment path for RT-DETR + TensorRT Edge-LLM.
When --smoke-image is provided, also runs a standalone Edge-LLM inference smoke test.
EOF
}

check() {
  local description="$1"
  shift
  if "$@" >/dev/null 2>&1; then
    printf 'PASS  %s\n' "${description}"
  else
    printf 'FAIL  %s\n' "${description}"
    failures=$((failures + 1))
  fi
}

check_ros_package_available() {
  local package_name="$1"
  local source_workspace_setup="${2:-0}"

  source_ros_setup_nounset_safe "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash" || return 1
  if [[ "${source_workspace_setup}" -eq 1 ]] && [[ -n "${ROS_WORKSPACE:-}" ]] && [[ -f "${ROS_WORKSPACE}/install/setup.bash" ]]; then
    source_ros_setup_nounset_safe "${ROS_WORKSPACE}/install/setup.bash" || return 1
  fi

  ros2 pkg prefix "${package_name}" >/dev/null 2>&1
}

for ((i = 1; i <= $#; i++)); do
  arg="${!i}"
  case "${arg}" in
    --smoke-image)
      next=$((i + 1))
      if [[ ${next} -gt $# ]]; then
        echo "Missing value for --smoke-image" >&2
        exit 2
      fi
      smoke_image="${!next}"
      run_standalone_smoke=1
      i=${next}
      ;;
    --isaac-ros) passthrough_args+=("--isaac-ros") ;;
    -h|--help) usage; exit 0 ;;
    *) passthrough_args+=("${arg}") ;;
  esac
done

if [[ -f "${env_file}" ]]; then
  # shellcheck disable=SC1090
  source "${env_file}"
fi

bash "${script_dir}/verify_deployment.sh" "${passthrough_args[@]}"

check "JP7.2 setup entrypoint present" test -x "${script_dir}/setup_thor_jp72.sh"

check "Thor launch entrypoint present" test -f "${repo_root}/launch/thor_tracked_observation.launch.py"

check "RT-DETR launch wiring present" bash -c \
  'grep -q "_resolve_isaac_rtdetr_launch" "$1" && grep -q "model_file_path" "$1" && grep -q "/detections_output" "$1" && grep -q "/image" "$1"' \
  _ "${repo_root}/launch/thor_tracked_observation.launch.py"

check "Detection timestamp behavior covered by unit test" test -f "${repo_root}/test/test_tracked_observation_adapter.cpp"

check "Rosbag acquisition script present" test -x "${script_dir}/test_data/download_rosbags.sh"

check "LLM engine binary present" test -f "${EDGE_VLM_LLM_ENGINE_DIR:-}/llm.engine"
check "Multimodal engine binary present" bash -c \
  'find "$1" -maxdepth 2 -type f -name "*.engine" -print -quit | grep -q .' \
  _ "${EDGE_VLM_MULTIMODAL_ENGINE_DIR:-}"
for llm_file in \
  embedding.safetensors \
  config.json \
  tokenizer.json \
  tokenizer_config.json \
  processed_chat_template.json; do
  check "LLM artifact present (${llm_file})" test -f "${EDGE_VLM_LLM_ENGINE_DIR:-}/${llm_file}"
done
for visual_file in \
  visual/visual.engine \
  visual/config.json \
  visual/preprocessor_config.json; do
  check "Visual artifact present (${visual_file})" test -f "${EDGE_VLM_MULTIMODAL_ENGINE_DIR:-}/${visual_file}"
done
check "Edge-LLM plugin present" test -f "${EDGELLM_PLUGIN_PATH:-}"

if [[ -d "${repo_root}/test_data/rosbags/image-proc" ]]; then
  check "image-proc rosbag assets present" \
    bash -c 'find "$1" -name metadata.yaml -print -quit | grep -q .' \
    _ "${repo_root}/test_data/rosbags/image-proc"
else
  echo "WARN  image-proc rosbag assets missing (run scripts/test_data/download_rosbags.sh download image-proc)"
fi

if [[ " ${passthrough_args[*]} " == *" --isaac-ros "* ]]; then
  check "Isaac ROS RT-DETR package available" check_ros_package_available isaac_ros_rtdetr 1
  check "Isaac ROS RT-DETR model installer package available" check_ros_package_available isaac_ros_rtdetr_models_install
fi

check "Edge-LLM runtime can load engines and become ready" bash -c '
  set -Eeuo pipefail
  : "${ROS_WORKSPACE:=${HOME}/ros2_ws}"
  server="${ROS_WORKSPACE}/install/edge_vlm_ros/lib/edge_vlm_ros/edge_vlm_server"
  [[ -x "${server}" ]] || exit 1
  run_dir="$(mktemp -d /tmp/edge-vlm-verify-load.XXXXXX)"
  socket_path="${run_dir}/worker.sock"
  log_file="${run_dir}/worker.log"
  cleanup() {
    if [[ -n "${pid:-}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  }
  trap cleanup EXIT
  "${server}" \
    "${EDGE_VLM_LLM_ENGINE_DIR}" \
    "${EDGE_VLM_MULTIMODAL_ENGINE_DIR}" \
    "${EDGELLM_PLUGIN_PATH}" \
    "${socket_path}" \
    90 \
    60 >"${log_file}" 2>&1 &
  pid=$!
  for _ in $(seq 1 300); do
    if [[ -S "${socket_path}" ]] && grep -Eiq "(edge_vlm_server|worker).*ready" "${log_file}"; then
      exit 0
    fi
    if ! kill -0 "${pid}" 2>/dev/null; then
      exit 1
    fi
    sleep 0.2
  done
  exit 1
'

if [[ "${run_standalone_smoke}" -eq 1 ]]; then
check "Edge-LLM semantic text prompt sanity (2+2)" bash -c '
  set -Eeuo pipefail
  : "${ROS_WORKSPACE:=${HOME}/ros2_ws}"
  : "${EDGE_VLM_LLM_ENGINE_DIR:?missing EDGE_VLM_LLM_ENGINE_DIR}"
  : "${EDGE_VLM_MULTIMODAL_ENGINE_DIR:?missing EDGE_VLM_MULTIMODAL_ENGINE_DIR}"
  : "${EDGELLM_PLUGIN_PATH:?missing EDGELLM_PLUGIN_PATH}"
  smoke_image="${1:-}"
  [[ -n "${smoke_image}" ]] || exit 2
  [[ "${smoke_image}" == /* && -f "${smoke_image}" ]] || exit 2
  client="${ROS_WORKSPACE}/install/edge_vlm_ros/lib/edge_vlm_ros/edge_vlm_cli"
  server="${ROS_WORKSPACE}/install/edge_vlm_ros/lib/edge_vlm_ros/edge_vlm_server"
  [[ -x "${client}" && -x "${server}" ]] || exit 1
  run_dir="$(mktemp -d /tmp/edge-vlm-verify-semantic-text.XXXXXX)"
  socket_path="${run_dir}/worker.sock"
  log_file="${run_dir}/worker.log"
  cleanup() {
    if [[ -n "${pid:-}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  }
  trap cleanup EXIT
  "${server}" "${EDGE_VLM_LLM_ENGINE_DIR}" "${EDGE_VLM_MULTIMODAL_ENGINE_DIR}" "${EDGELLM_PLUGIN_PATH}" "${socket_path}" 90 60 >"${log_file}" 2>&1 &
  pid=$!
  for _ in $(seq 1 300); do
    if [[ -S "${socket_path}" ]] && grep -Eiq "(edge_vlm_server|worker).*ready" "${log_file}"; then
      break
    fi
    if ! kill -0 "${pid}" 2>/dev/null; then
      exit 1
    fi
    sleep 0.2
  done
  [[ -S "${socket_path}" ]] || exit 1
  response="$("${client}" --socket "${socket_path}" --image "${smoke_image}" --prompt "What is 2 plus 2? Answer with one short sentence." --max-generate-length 24 2>/dev/null || true)"
  [[ -n "${response}" ]] || exit 1
  [[ "${#response}" -lt 400 ]] || exit 1
  if ! grep -Eiq "(^|[^0-9])4([^0-9]|$)|\\bfour\\b" <<<"${response}"; then
    exit 1
  fi
' _ "${smoke_image}"

check "Edge-LLM semantic VLM prompt sanity (red panda request)" bash -c '
  set -Eeuo pipefail
  : "${ROS_WORKSPACE:=${HOME}/ros2_ws}"
  : "${EDGE_VLM_LLM_ENGINE_DIR:?missing EDGE_VLM_LLM_ENGINE_DIR}"
  : "${EDGE_VLM_MULTIMODAL_ENGINE_DIR:?missing EDGE_VLM_MULTIMODAL_ENGINE_DIR}"
  : "${EDGELLM_PLUGIN_PATH:?missing EDGELLM_PLUGIN_PATH}"
  smoke_image="${1:-}"
  [[ -n "${smoke_image}" ]] || exit 2
  [[ "${smoke_image}" == /* && -f "${smoke_image}" ]] || exit 2
  client="${ROS_WORKSPACE}/install/edge_vlm_ros/lib/edge_vlm_ros/edge_vlm_cli"
  server="${ROS_WORKSPACE}/install/edge_vlm_ros/lib/edge_vlm_ros/edge_vlm_server"
  [[ -x "${client}" && -x "${server}" ]] || exit 1
  run_dir="$(mktemp -d /tmp/edge-vlm-verify-semantic-vlm.XXXXXX)"
  socket_path="${run_dir}/worker.sock"
  log_file="${run_dir}/worker.log"
  cleanup() {
    if [[ -n "${pid:-}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  }
  trap cleanup EXIT
  "${server}" "${EDGE_VLM_LLM_ENGINE_DIR}" "${EDGE_VLM_MULTIMODAL_ENGINE_DIR}" "${EDGELLM_PLUGIN_PATH}" "${socket_path}" 90 60 >"${log_file}" 2>&1 &
  pid=$!
  for _ in $(seq 1 300); do
    if [[ -S "${socket_path}" ]] && grep -Eiq "(edge_vlm_server|worker).*ready" "${log_file}"; then
      break
    fi
    if ! kill -0 "${pid}" 2>/dev/null; then
      exit 1
    fi
    sleep 0.2
  done
  [[ -S "${socket_path}" ]] || exit 1
  response="$("${client}" --socket "${socket_path}" --image "${smoke_image}" --prompt "Describe the red panda in one short sentence." --max-generate-length 48 2>/dev/null || true)"
  [[ -n "${response}" ]] || exit 1
  [[ "${#response}" -lt 500 ]] || exit 1
  if ! grep -Eiq "\\bred panda\\b|\\bpanda\\b" <<<"${response}"; then
    exit 1
  fi
' _ "${smoke_image}"
fi

if [[ "${run_standalone_smoke}" -eq 1 ]]; then
  if [[ "${smoke_image}" != /* ]]; then
    echo "--smoke-image must be an absolute path: ${smoke_image}" >&2
    exit 2
  fi
  bash "${script_dir}/test_data/run_standalone_service_smoke.sh" "${smoke_image}"
else
  echo "WARN  Full request/response smoke skipped (pass --smoke-image /absolute/path.jpg to execute it; semantic checks require this image)"
fi

if [[ "${failures}" -ne 0 ]]; then
  echo "Thor JP7.2 verification failed: ${failures} additional check(s)." >&2
  exit 1
fi

echo "Thor JP7.2 verification checks completed."
