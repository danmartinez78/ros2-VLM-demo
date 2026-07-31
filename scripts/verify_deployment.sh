#!/usr/bin/env bash
set -uo pipefail

VERIFY_ISAAC_ROS=0
for arg in "$@"; do
  case "$arg" in
    --isaac-ros) VERIFY_ISAAC_ROS=1 ;;
    -h|--help)
      echo "Usage: $0 [--isaac-ros]"
      exit 0
      ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

if [[ -f /etc/apt/sources.list.d/nvidia-isaac-ros.list ]]; then
  VERIFY_ISAAC_ROS=1
fi

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
l4t_release="$(sed -n '1p' /etc/nv_tegra_release 2>/dev/null || true)"

check "Ubuntu 24.04" bash -c 'source /etc/os-release && [[ "$ID" == ubuntu && "$VERSION_ID" == 24.04 ]]'
check "aarch64 architecture" bash -c '[[ "$(uname -m)" == aarch64 ]]'
check "Jetson Linux R39.2" grep -q '# R39 (release), REVISION: 2' /etc/nv_tegra_release
check "JetPack metapackage" dpkg-query -W nvidia-jetpack
check "CUDA compiler" bash -c 'export PATH=/usr/local/cuda/bin:$PATH; command -v nvcc && nvcc --version'
check "TensorRT development headers" bash -c 'test -f /usr/include/NvInfer.h || test -f /usr/include/aarch64-linux-gnu/NvInfer.h'
check "ROS 2 Jazzy setup" test -f "/opt/ros/${ros_distro}/setup.bash"

if [[ "${VERIFY_ISAAC_ROS}" -eq 1 ]]; then
  check "Isaac ROS 4.5 Thor APT source" grep -Fxq \
    "deb [signed-by=/usr/share/keyrings/nvidia-isaac-ros.gpg] https://isaac.download.nvidia.com/isaac-ros/release-4.5 noble-jetpack main" \
    /etc/apt/sources.list.d/nvidia-isaac-ros.list
  check "Isaac ROS CLI package" dpkg-query -W isaac-ros-cli
  check "Isaac ROS CLI" isaac-ros --help
  check "NVIDIA container toolkit" dpkg-query -W nvidia-container-toolkit
  check "Docker service" systemctl is-active --quiet docker
  check "Docker CLI" command -v docker

  if [[ "${l4t_release:-}" == *"# R39 (release), REVISION: 2"* ]]; then
    printf 'WARN  Isaac ROS 4.5 is not yet officially validated on JetPack 7.2 / R39.2\n'
    printf '      NVIDIA currently lists JetPack 7.1 / R38.4 for Jetson Thor.\n'
  fi
fi

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
  # ROS-generated setup scripts may read optional variables without defaults.
  set +u
  # shellcheck disable=SC1090
  source "/opt/ros/${ros_distro}/setup.bash"
  # shellcheck disable=SC1090
  source "${ros_workspace}/install/setup.bash"
  set -u

  check "Installed ROS package" ros2 pkg prefix cosmos_ros2_video_reasoner
  check "Installed reasoner executable" bash -c \
    "ros2 pkg executables cosmos_ros2_video_reasoner | grep -q ' cosmos_reasoner$'"
  check "Installed inference worker" bash -c \
    "ros2 pkg executables cosmos_ros2_video_reasoner | grep -q ' cosmos_inference_worker$'"

  package_prefix="$(ros2 pkg prefix cosmos_ros2_video_reasoner)"
  reasoner_executable="${package_prefix}/lib/cosmos_ros2_video_reasoner/cosmos_reasoner"
  worker_executable="${package_prefix}/lib/cosmos_ros2_video_reasoner/cosmos_inference_worker"

  check "ROS process excludes CUDA and TensorRT" bash -c \
    '! ldd "$1" | grep -Eq "lib(cuda|nvinfer|cosmos_trt_backend)"' \
    _ "${reasoner_executable}"
  check "Inference worker excludes ROS libraries" bash -c \
    '! ldd "$1" | grep -Eq "lib(rcl|rmw|rosidl|fastrtps)"' \
    _ "${worker_executable}"
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
