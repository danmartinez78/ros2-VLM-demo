#!/usr/bin/env bash
set -uo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
env_file="${COSMOS_ENV_FILE:-${script_dir}/cosmos_env.sh}"

if [[ -f "${env_file}" ]]; then
  # shellcheck disable=SC1090
  source "${env_file}"
fi

failures=0
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

ros_distro="${ROS_DISTRO:-jazzy}"
edge_root="${TENSORRT_EDGE_LLM_ROOT:-${HOME}/TensorRT-Edge-LLM}"
edge_build="${TENSORRT_EDGE_LLM_BUILD_DIR:-${edge_root}/build}"
model_name="${COSMOS_MODEL_NAME:-Cosmos-Reason2-8B}"
cosmos_workspace="${COSMOS_WORKSPACE_DIR:-${HOME}/tensorrt-edgellm-workspace}"
llm_engine="${COSMOS_LLM_ENGINE_DIR:-${cosmos_workspace}/${model_name}/engine/llm}"
multimodal_engine="${COSMOS_MULTIMODAL_ENGINE_DIR:-${cosmos_workspace}/${model_name}/engine}"
plugin_path="${EDGELLM_PLUGIN_PATH:-${edge_build}/libNvInfer_edgellm_plugin.so}"

check "Ubuntu 24.04" bash -c 'source /etc/os-release && [[ "$ID" == ubuntu && "$VERSION_ID" == 24.04 ]]'
check "aarch64 architecture" bash -c '[[ "$(uname -m)" == aarch64 ]]'
check "Jetson Linux R39.2" grep -q '# R39 (release), REVISION: 2' /etc/nv_tegra_release
check "JetPack metapackage" dpkg-query -W nvidia-jetpack
check "CUDA compiler" bash -c 'export PATH=/usr/local/cuda/bin:$PATH; command -v nvcc && nvcc --version'
check "TensorRT development headers" bash -c 'test -f /usr/include/NvInfer.h || test -f /usr/include/aarch64-linux-gnu/NvInfer.h'
check "ROS 2 Jazzy setup" test -f "/opt/ros/${ros_distro}/setup.bash"
check "rosdep" command -v rosdep
check "colcon" command -v colcon
check "OpenCV development package" dpkg-query -W libopencv-dev
check "Edge-LLM runtime header" test -f "${edge_root}/cpp/runtime/llmInferenceRuntime.h"
check "Edge-LLM core archive" bash -c 'find "$1" -name libedgellmCore.a -print -quit | grep -q .' _ "${edge_build}"
check "Edge-LLM plugin" test -f "${plugin_path}"
check "Cosmos LLM engine directory" test -d "${llm_engine}"
check "Cosmos visual engine directory" test -d "${multimodal_engine}"

if [[ -n "${ROS_WORKSPACE:-}" ]]; then
  ros_workspace="${ROS_WORKSPACE}"
elif [[ "$(basename -- "$(dirname -- "${repo_root}")")" == "src" ]]; then
  ros_workspace="$(cd -- "${repo_root}/../.." && pwd)"
else
  ros_workspace=""
fi

if [[ -n "${ros_workspace}" && -f "${ros_workspace}/install/setup.bash" ]]; then
  # shellcheck disable=SC1090
  source "/opt/ros/${ros_distro}/setup.bash"
  # shellcheck disable=SC1090
  source "${ros_workspace}/install/setup.bash"
  check "Installed ROS package" ros2 pkg prefix cosmos_ros2_video_reasoner
  check "Installed reasoner executable" bash -c "ros2 pkg executables cosmos_ros2_video_reasoner | grep -q cosmos_reasoner"
else
  printf 'FAIL  Built ROS workspace overlay\n'
  failures=$((failures + 1))
fi

echo
echo "Resolved deployment paths:"
echo "  Edge-LLM source:       ${edge_root}"
echo "  Edge-LLM build:        ${edge_build}"
echo "  LLM engine:            ${llm_engine}"
echo "  Multimodal engine:     ${multimodal_engine}"
echo "  Plugin:                ${plugin_path}"
echo

if [[ "${failures}" -ne 0 ]]; then
  echo "Deployment verification failed: ${failures} check(s)." >&2
  exit 1
fi

echo "Deployment verification passed."
