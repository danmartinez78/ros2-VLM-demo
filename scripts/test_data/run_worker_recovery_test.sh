#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
asset_root="${ROSBAG_DIR:-${repo_root}/test_data/rosbags}/image-proc"
env_file="${COSMOS_ENV_FILE:-${repo_root}/scripts/cosmos_env.sh}"
image_topic="/hawk_0_left_rgb_image"
result_topic="/cosmos/reasoning"
launch_pid=""
bag_pid=""

cleanup() {
  if [[ -n "${bag_pid}" ]] && kill -0 "${bag_pid}" 2>/dev/null; then
    kill -INT "${bag_pid}" 2>/dev/null || true
    wait "${bag_pid}" 2>/dev/null || true
  fi
  if [[ -n "${launch_pid}" ]] && kill -0 "${launch_pid}" 2>/dev/null; then
    kill -INT "${launch_pid}" 2>/dev/null || true
    wait "${launch_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if [[ ! -d "${asset_root}" ]]; then
  bash "${script_dir}/download_rosbags.sh" download image-proc
fi
metadata="$(find "${asset_root}" -name metadata.yaml -print -quit)"
if [[ -z "${metadata}" ]]; then
  echo "No ROS 2 bag metadata found under ${asset_root}." >&2
  exit 1
fi
bag_path="$(dirname -- "${metadata}")"

if [[ -f "${env_file}" ]]; then
  # shellcheck disable=SC1090
  source "${env_file}"
fi
ros_distro="${ROS_DISTRO:-jazzy}"
ros_setup="/opt/ros/${ros_distro}/setup.bash"
if [[ ! -f "${ros_setup}" ]]; then
  echo "ROS 2 ${ros_distro} is not installed." >&2
  exit 1
fi
set +u
# shellcheck disable=SC1090
source "${ros_setup}"
if [[ -n "${ROS_WORKSPACE:-}" && -f "${ROS_WORKSPACE}/install/setup.bash" ]]; then
  # shellcheck disable=SC1090
  source "${ROS_WORKSPACE}/install/setup.bash"
elif [[ -f "${repo_root}/../../install/setup.bash" ]]; then
  # shellcheck disable=SC1090
  source "${repo_root}/../../install/setup.bash"
fi
set -u

for variable in COSMOS_LLM_ENGINE_DIR COSMOS_MULTIMODAL_ENGINE_DIR EDGELLM_PLUGIN_PATH; do
  if [[ -z "${!variable:-}" ]]; then
    echo "Missing ${variable}; configure ${env_file} first." >&2
    exit 1
  fi
done

worker_pid() {
  pgrep -n -f '/cosmos_inference_worker($| )' || true
}

wait_for_worker() {
  local excluded_pid="${1:-}"
  local deadline=$((SECONDS + 120))
  local pid=""
  while (( SECONDS < deadline )); do
    pid="$(worker_pid)"
    if [[ -n "${pid}" && "${pid}" != "${excluded_pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      printf '%s\n' "${pid}"
      return 0
    fi
    sleep 1
  done
  return 1
}

wait_for_success_result() {
  local attempts=0
  local output=""
  while (( attempts < 4 )); do
    output="$(mktemp /tmp/cosmos-recovery-result.XXXXXX)"
    if timeout 120 ros2 topic echo "${result_topic}" --once >"${output}" 2>&1; then
      if grep -q '^success: true$' "${output}"; then
        cat "${output}"
        rm -f "${output}"
        return 0
      fi
      echo "Observed an expected transient failed result while reconnecting:"
      cat "${output}"
    fi
    rm -f "${output}"
    attempts=$((attempts + 1))
  done
  return 1
}

echo "Starting isolated worker recovery test..."
ros2 launch cosmos_ros2_video_reasoner cosmos_reasoner.launch.py \
  image_topic:="${image_topic}" \
  result_topic:="${result_topic}" \
  llm_engine_dir:="${COSMOS_LLM_ENGINE_DIR}" \
  multimodal_engine_dir:="${COSMOS_MULTIMODAL_ENGINE_DIR}" \
  edge_llm_plugin_path:="${EDGELLM_PLUGIN_PATH}" \
  sample_period_seconds:=1.0 \
  max_generate_length:=64 \
  use_sim_time:=true &
launch_pid=$!

old_worker_pid="$(wait_for_worker)" || {
  echo "Timed out waiting for the initial inference worker." >&2
  exit 1
}
echo "Initial worker PID: ${old_worker_pid}"

ready=false
deadline=$((SECONDS + 120))
while (( SECONDS < deadline )); do
  if ros2 topic info "${image_topic}" 2>/dev/null \
      | grep -Eq 'Subscription count: [1-9][0-9]*'; then
    ready=true
    break
  fi
  sleep 1
done
if [[ "${ready}" != true ]]; then
  echo "Timed out waiting for the reasoner subscription." >&2
  exit 1
fi

ros2 bag play "${bag_path}" --clock --loop &
bag_pid=$!

echo "Waiting for a successful result before fault injection..."
wait_for_success_result >/tmp/cosmos-recovery-before.txt || {
  echo "No successful result arrived before fault injection." >&2
  exit 1
}
cat /tmp/cosmos-recovery-before.txt
rm -f /tmp/cosmos-recovery-before.txt

echo "Killing inference worker ${old_worker_pid} with SIGKILL..."
kill -KILL "${old_worker_pid}"

new_worker_pid="$(wait_for_worker "${old_worker_pid}")" || {
  echo "Launch did not respawn the inference worker." >&2
  exit 1
}
echo "Respawned worker PID: ${new_worker_pid}"

echo "Waiting for reasoning to resume after reconnect..."
wait_for_success_result >/tmp/cosmos-recovery-after.txt || {
  echo "Reasoning did not recover after the worker restart." >&2
  exit 1
}
cat /tmp/cosmos-recovery-after.txt
rm -f /tmp/cosmos-recovery-after.txt

echo "PASS: worker respawned and reasoning resumed without restarting the ROS node."
