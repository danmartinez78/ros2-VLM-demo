#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
env_file="${EDGE_VLM_ENV_FILE:-${script_dir}/edge_vlm_env.sh}"
source "${script_dir}/apt_transaction_guard.sh"

if [[ -f "${env_file}" ]]; then
  # shellcheck disable=SC1090
  source "${env_file}"
fi

ros_distro="${ROS_DISTRO:-jazzy}"
ros_setup="/opt/ros/${ros_distro}/setup.bash"
if [[ ! -f "${ros_setup}" ]]; then
  echo "ROS 2 ${ros_distro} is not installed. Run scripts/install_dependencies.sh first." >&2
  exit 1
fi
# ROS-generated setup scripts may read optional variables without defaults.
set +u
# shellcheck disable=SC1090
source "${ros_setup}"
set -u

edge_root="${TENSORRT_EDGE_LLM_ROOT:-${HOME}/TensorRT-Edge-LLM}"
edge_build="${TENSORRT_EDGE_LLM_BUILD_DIR:-${edge_root}/build}"
trt_package_dir="${TRT_PACKAGE_DIR:-/usr}"

if [[ ! -f "${edge_root}/cpp/runtime/llmInferenceRuntime.h" ]]; then
  echo "TensorRT Edge-LLM source not found at: ${edge_root}" >&2
  exit 1
fi
if ! find "${edge_build}" -name libedgellmCore.a -print -quit | grep -q .; then
  echo "libedgellmCore.a was not found below: ${edge_build}" >&2
  echo "Build TensorRT Edge-LLM for this Thor before building the ROS package." >&2
  exit 1
fi
if [[ ! -f "${edge_build}/libNvInfer_edgellm_plugin.so" ]]; then
  echo "Edge-LLM plugin not found: ${edge_build}/libNvInfer_edgellm_plugin.so" >&2
  exit 1
fi

if [[ -n "${ROS_WORKSPACE:-}" ]]; then
  ros_workspace="${ROS_WORKSPACE}"
elif [[ "$(basename -- "$(dirname -- "${repo_root}")")" == "src" ]]; then
  ros_workspace="$(cd -- "${repo_root}/../.." && pwd)"
else
  echo "Cannot infer the ROS workspace from ${repo_root}." >&2
  echo "Set ROS_WORKSPACE in scripts/edge_vlm_env.sh." >&2
  exit 1
fi

if [[ ! -d "${ros_workspace}/src" ]]; then
  echo "ROS workspace has no src directory: ${ros_workspace}" >&2
  exit 1
fi

assert_safe_rosdep_install_plan "${repo_root}" "${ros_distro}"
rosdep install \
  --from-paths "${repo_root}" \
  --ignore-src \
  --rosdistro "${ros_distro}" \
  -r -y

cd "${ros_workspace}"
colcon build \
  --symlink-install \
  --packages-select edge_vlm_ros \
  --event-handlers console_direct+ \
  --cmake-args \
    -DCMAKE_BUILD_TYPE=Release \
    -DTENSORRT_EDGE_LLM_ROOT="${edge_root}" \
    -DTENSORRT_EDGE_LLM_BUILD_DIR="${edge_build}" \
    -DTRT_PACKAGE_DIR="${trt_package_dir}"

echo
echo "Build complete. Load the workspace with:"
echo "  source ${ros_workspace}/install/setup.bash"
echo "Then run:"
echo "  ./scripts/verify_deployment.sh"
