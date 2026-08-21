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
env_file="${EDGE_VLM_ENV_FILE:-${script_dir}/edge_vlm_env.sh}"
source "${script_dir}/apt_transaction_guard.sh"
source "${script_dir}/ros_setup_guard.sh"

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

is_supported_l4t_release() {
  local release_line="${1:-}"
  [[ "${release_line}" =~ ^#\ R39\ \(release\),\ REVISION:\ 2\.[0-9]+([[:space:],].*)?$ ]]
}

ros_distro="${ROS_DISTRO:-jazzy}"
edge_root="${TENSORRT_EDGE_LLM_ROOT:-${HOME}/TensorRT-Edge-LLM}"
edge_build="${TENSORRT_EDGE_LLM_BUILD_DIR:-${edge_root}/build}"
model_name="${EDGE_VLM_MODEL_NAME:-Cosmos-Reason2-8B}"
edge_vlm_workspace="${EDGE_VLM_WORKSPACE_DIR:-${HOME}/tensorrt-edgellm-workspace}"
llm_engine="${EDGE_VLM_LLM_ENGINE_DIR:-${edge_vlm_workspace}/${model_name}/engine/llm}"
multimodal_engine="${EDGE_VLM_MULTIMODAL_ENGINE_DIR:-${edge_vlm_workspace}/${model_name}/engine}"
plugin_path="${EDGELLM_PLUGIN_PATH:-${edge_build}/libNvInfer_edgellm_plugin.so}"
l4t_release="$(sed -n '1p' /etc/nv_tegra_release 2>/dev/null || true)"

if [[ "${EDGE_VLM_L4T_GATE_TEST_MODE:-0}" == "1" ]]; then
  test_release="${EDGE_VLM_L4T_GATE_RELEASE:-${l4t_release}}"
  if is_supported_l4t_release "${test_release}"; then
    echo "L4T gate accepted release: ${test_release}"
    exit 0
  fi
  echo "L4T gate rejected release: ${test_release}" >&2
  exit 1
fi

check "Ubuntu 24.04" bash -c 'source /etc/os-release && [[ "$ID" == ubuntu && "$VERSION_ID" == 24.04 ]]'
check "aarch64 architecture" bash -c '[[ "$(uname -m)" == aarch64 ]]'
check "Jetson Linux R39.2.x (JetPack 7.2.x)" is_supported_l4t_release "${l4t_release}"
check "JetPack metapackage" dpkg-query -W nvidia-jetpack
check "JetPack developer metapackage" dpkg-query -W nvidia-jetpack-dev
check "CUDA compiler" bash -c 'export PATH=/usr/local/cuda/bin:$PATH; command -v nvcc && nvcc --version'
check "TensorRT development headers" bash -c 'test -f /usr/include/NvInfer.h || test -f /usr/include/aarch64-linux-gnu/NvInfer.h'
check "TensorRT development package" dpkg-query -W libnvinfer-dev
check "TensorRT transaction preserves protected NVIDIA packages" \
  assert_safe_apt_transaction "deployment verification TensorRT safety" libnvinfer-dev
check "TensorRT candidate matches installed version" \
  assert_package_candidate_matches_installed libnvinfer-dev "TensorRT development package"
check "ROS 2 Jazzy setup" test -f "/opt/ros/${ros_distro}/setup.bash"

if [[ "${VERIFY_ISAAC_ROS}" -eq 1 ]]; then
  check "Isaac ROS 4.6 Thor APT source" grep -Fxq \
    "deb [signed-by=/usr/share/keyrings/nvidia-isaac-ros.gpg] https://isaac.download.nvidia.com/isaac-ros/release-4.6 noble-jetpack main" \
    /etc/apt/sources.list.d/nvidia-isaac-ros.list
  check "Isaac ROS CLI package" dpkg-query -W isaac-ros-cli
  check "Isaac ROS CLI" isaac-ros --help
  check "NVIDIA container toolkit" dpkg-query -W nvidia-container-toolkit
  check "Docker service" systemctl is-active --quiet docker
  check "Docker CLI" command -v docker

  if ! is_supported_l4t_release "${l4t_release:-}"; then
    printf 'WARN  Isaac ROS Thor support expects JetPack 7.2.x / R39.2.x.\n'
    printf '      This host reports: %s\n' "${l4t_release:-unknown}"
  fi
fi

check "rosdep" command -v rosdep
check "colcon" command -v colcon
check "OpenCV development package" dpkg-query -W libopencv-dev
check "NVIDIA OpenCV development package" dpkg-query -W nvidia-opencv-dev
check "OpenCV transaction preserves protected NVIDIA packages" \
  assert_safe_apt_transaction "deployment verification OpenCV safety" libopencv-dev
check "OpenCV candidate matches installed version" assert_libopencv_candidate_matches_installed
if cuda_owner_pkg="$(resolve_nvcc_owner_package 2>/dev/null)"; then
  check "CUDA transaction preserves protected NVIDIA packages (${cuda_owner_pkg})" \
    assert_safe_apt_transaction "deployment verification CUDA safety" "${cuda_owner_pkg}"
  check "CUDA package candidate matches installed version (${cuda_owner_pkg})" \
    assert_package_candidate_matches_installed "${cuda_owner_pkg}" "CUDA compiler package"
else
  check "CUDA compiler package ownership" false
fi
check "Edge-LLM runtime header" test -f "${edge_root}/cpp/runtime/llmInferenceRuntime.h"
check "Edge-LLM core archive" bash -c 'find "$1" -name libedgellmCore.a -print -quit | grep -q .' _ "${edge_build}"
check "Edge-LLM plugin" test -f "${plugin_path}"
check "LLM engine directory" test -d "${llm_engine}"
check "Visual encoder engine directory" test -d "${multimodal_engine}"

if [[ -n "${ROS_WORKSPACE:-}" ]]; then
  ros_workspace="${ROS_WORKSPACE}"
elif [[ "$(basename -- "$(dirname -- "${repo_root}")")" == "src" ]]; then
  ros_workspace="$(cd -- "${repo_root}/../.." && pwd)"
else
  ros_workspace=""
fi

if [[ -n "${ros_workspace}" && -f "${ros_workspace}/install/setup.bash" ]]; then
  source_ros_setup_nounset_safe "/opt/ros/${ros_distro}/setup.bash" || {
    printf 'FAIL  Source ROS distro setup\n'
    failures=$((failures + 1))
  }
  source_ros_setup_nounset_safe "${ros_workspace}/install/setup.bash" || {
    printf 'FAIL  Source ROS workspace setup\n'
    failures=$((failures + 1))
  }

  check "Installed ROS package" ros2 pkg prefix edge_vlm_ros
  check "Installed reasoner executable" bash -c \
    "ros2 pkg executables edge_vlm_ros | grep -q ' edge_vlm_ros_node$'"
  check "Installed inference worker" bash -c \
    "ros2 pkg executables edge_vlm_ros | grep -q ' edge_vlm_server$'"

  package_prefix="$(ros2 pkg prefix edge_vlm_ros)"
  reasoner_executable="${package_prefix}/lib/edge_vlm_ros/edge_vlm_ros_node"
  worker_executable="${package_prefix}/lib/edge_vlm_ros/edge_vlm_server"

  check "ROS process excludes CUDA and TensorRT" bash -c \
    '! ldd "$1" | grep -Eq "lib(cuda|nvinfer|edge_vlm_trt_backend)"' \
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
