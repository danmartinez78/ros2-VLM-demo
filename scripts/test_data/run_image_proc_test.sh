#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
asset_root="${ROSBAG_DIR:-${repo_root}/test_data/rosbags}/image-proc"
env_file="${COSMOS_ENV_FILE:-${repo_root}/scripts/cosmos_env.sh}"

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
# shellcheck disable=SC1090
source "/opt/ros/${ros_distro}/setup.bash"

if [[ -n "${ROS_WORKSPACE:-}" && -f "${ROS_WORKSPACE}/install/setup.bash" ]]; then
  # shellcheck disable=SC1090
  source "${ROS_WORKSPACE}/install/setup.bash"
elif [[ -f "${repo_root}/../../install/setup.bash" ]]; then
  # Repository is normally checked out at <workspace>/src/<repo>.
  # shellcheck disable=SC1090
  source "${repo_root}/../../install/setup.bash"
fi

for variable in COSMOS_LLM_ENGINE_DIR COSMOS_MULTIMODAL_ENGINE_DIR EDGELLM_PLUGIN_PATH; do
  if [[ -z "${!variable:-}" ]]; then
    echo "Missing ${variable}; configure ${env_file} first." >&2
    exit 1
  fi
done

image_topic="/hawk_0_left_rgb_image"
launch_pid=""
cleanup() {
  if [[ -n "${launch_pid}" ]] && kill -0 "${launch_pid}" 2>/dev/null; then
    kill -INT "${launch_pid}" 2>/dev/null || true
    wait "${launch_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "Starting Cosmos reasoner on ${image_topic}..."
ros2 launch cosmos_ros2_video_reasoner cosmos_reasoner.launch.py   image_topic:="${image_topic}"   llm_engine_dir:="${COSMOS_LLM_ENGINE_DIR}"   multimodal_engine_dir:="${COSMOS_MULTIMODAL_ENGINE_DIR}"   edge_llm_plugin_path:="${EDGELLM_PLUGIN_PATH}"   use_sim_time:=true &
launch_pid=$!

sleep 3
if ! kill -0 "${launch_pid}" 2>/dev/null; then
  echo "The Cosmos reasoner exited before playback started." >&2
  wait "${launch_pid}"
fi

echo "Playing NVIDIA image-proc bag: ${bag_path}"
ros2 bag play "${bag_path}" --clock
