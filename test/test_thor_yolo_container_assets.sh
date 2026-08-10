#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
dockerfile="${repo_root}/docker/thor-yolo/Dockerfile"
entrypoint="${repo_root}/docker/thor-yolo/entrypoint.sh"
hpcx_env="${repo_root}/docker/thor-yolo/hpcx-env.sh"
compose_file="${repo_root}/docker/compose.thor-yolo.yml"
launcher_script="${repo_root}/scripts/launch_thor_with_yolo_container.sh"

for required_file in "${dockerfile}" "${entrypoint}" "${hpcx_env}" "${compose_file}" "${launcher_script}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Required Thor YOLO container asset missing: ${required_file}" >&2
    exit 1
  fi
done

grep -Fq 'FROM ${BASE_IMAGE}' "${dockerfile}"
grep -Fq 'ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.05-py3' "${dockerfile}"
grep -Fq 'COPY --from=ghcr.io/astral-sh/uv:0.6.17 /uv /usr/local/bin/uv' "${dockerfile}"
grep -Fq 'COPY docker/thor-yolo/hpcx-env.sh /usr/local/bin/edge-vlm-thor-yolo-hpcx-env' "${dockerfile}"
grep -Fq 'YOLO_VENV=/opt/edge-vlm-yolo-venv' "${dockerfile}"
grep -Fq 'uv venv --system-site-packages "${YOLO_VENV}"' "${dockerfile}"
grep -Fq 'uv pip install --python "${YOLO_VENV}/bin/python" -r /tmp/edge_vlm_yolo_requirements.txt' "${dockerfile}"
grep -Fq 'uv pip install --python "${YOLO_VENV}/bin/python" --no-deps ultralytics==8.4.6' "${dockerfile}"
grep -Fq 'git checkout "${YOLO_ROS_SHA}"' "${dockerfile}"
grep -Fq 'colcon build --merge-install --packages-select yolo_msgs yolo_ros' "${dockerfile}"
grep -Fq 'attempt_download_asset' "${dockerfile}"
grep -Fq "source /usr/local/bin/edge-vlm-thor-yolo-hpcx-env" "${dockerfile}"
grep -Fq "python3 -c 'import torch; print(torch.__version__)'" "${dockerfile}"
grep -Fq "python3 -c 'import torchvision; print(torchvision.__version__)'" "${dockerfile}"
grep -Fq 'import ultralytics' "${dockerfile}"
grep -Fq 'test "$(command -v python3)" = "${YOLO_VENV}/bin/python3"' "${dockerfile}"
grep -Fq 'software-properties-common' "${dockerfile}"
grep -Fq 'add-apt-repository -y universe' "${dockerfile}"
grep -Fq 'packages.ros.org/ros2/ubuntu' "${dockerfile}"
grep -Fq 'ros-dev-tools' "${dockerfile}"
grep -Fq "source /usr/local/bin/edge-vlm-thor-yolo-hpcx-env" "${entrypoint}"
grep -Fq 'export VIRTUAL_ENV="${yolo_venv}"' "${entrypoint}"
grep -Fq 'export PATH="${yolo_venv}/bin:${PATH}"' "${entrypoint}"
grep -Fq "libucs.so" "${hpcx_env}"
grep -Fq "libucc.so" "${hpcx_env}"
grep -Fq "libmpi.so" "${hpcx_env}"

grep -Fq 'runtime: nvidia' "${compose_file}"
grep -Fq 'network_mode: host' "${compose_file}"
grep -Fq 'YOLO_IMAGE_TOPIC' "${compose_file}"
grep -Fq 'YOLO_MODEL:' "${compose_file}"

grep -Fq 'docker compose -f "${compose_file}" up -d thor-yolo-detector' "${launcher_script}"
grep -Fq 'detector_backend:=ultralytics_yolo' "${launcher_script}"

if grep -Eq '(^|[^[:alnum:]_])uv sync([^[:alnum:]_]|$)' "${dockerfile}" "${entrypoint}" "${launcher_script}"; then
  echo "Thor YOLO container assets must not run uv sync at runtime." >&2
  exit 1
fi

if grep -Eq '(^|[^[:alnum:]_])pip install([^[:alnum:]_]|$)' "${entrypoint}" "${launcher_script}"; then
  echo "Thor YOLO runtime assets must not install Python packages at runtime." >&2
  exit 1
fi

if grep -Fq 'uv pip install --system' "${dockerfile}" || \
   grep -Fq -- '--break-system-packages' "${dockerfile}"; then
  echo "Thor YOLO Dockerfile must use the repo-owned venv flow instead of mutating the system interpreter." >&2
  exit 1
fi

if grep -Fq 'python3-colcon-common-extensions' "${dockerfile}" || \
   grep -Fq 'python3-vcstool' "${dockerfile}"; then
  echo "Thor YOLO Dockerfile must use supported Noble/Jazzy tooling package names such as ros-dev-tools." >&2
  exit 1
fi

line_number() {
  local pattern="$1"
  grep -nF "${pattern}" "${dockerfile}" | head -n1 | cut -d: -f1
}

universe_line="$(line_number 'add-apt-repository -y universe')"
ros_source_line="$(line_number 'packages.ros.org/ros2/ubuntu')"
ros_dev_tools_line="$(line_number 'ros-dev-tools')"
hpcx_env_copy_line="$(line_number 'COPY docker/thor-yolo/hpcx-env.sh /usr/local/bin/edge-vlm-thor-yolo-hpcx-env')"
first_torch_import_line="$(line_number "python3 -c 'import torch; print(torch.__version__)'")"
venv_create_line="$(line_number 'uv venv --system-site-packages "${YOLO_VENV}"')"
venv_install_line="$(line_number 'uv pip install --python "${YOLO_VENV}/bin/python" -r /tmp/edge_vlm_yolo_requirements.txt')"
runtime_python_line="$(line_number 'test "$(command -v python3)" = "${YOLO_VENV}/bin/python3"')"

if [[ -z "${universe_line}" || -z "${ros_source_line}" || -z "${ros_dev_tools_line}" || -z "${hpcx_env_copy_line}" || -z "${first_torch_import_line}" || -z "${venv_create_line}" || -z "${venv_install_line}" || -z "${runtime_python_line}" ]]; then
  echo "Thor YOLO Dockerfile is missing required bootstrap steps." >&2
  exit 1
fi

if (( universe_line >= ros_source_line )); then
  echo "Thor YOLO Dockerfile must enable universe before adding the ROS 2 apt source." >&2
  exit 1
fi

if (( ros_source_line >= ros_dev_tools_line )); then
  echo "Thor YOLO Dockerfile must add the ROS 2 apt source before installing ROS development tooling." >&2
  exit 1
fi

if (( hpcx_env_copy_line >= first_torch_import_line )); then
  echo "Thor YOLO Dockerfile must install the HPC-X library-path helper before validating torch imports." >&2
  exit 1
fi

if (( first_torch_import_line >= venv_create_line )); then
  echo "Thor YOLO Dockerfile must validate the NVIDIA torch import before creating the repo-owned venv." >&2
  exit 1
fi

if (( venv_create_line >= venv_install_line )); then
  echo "Thor YOLO Dockerfile must create the repo-owned venv before installing YOLO Python dependencies." >&2
  exit 1
fi

if (( venv_install_line >= runtime_python_line )); then
  echo "Thor YOLO Dockerfile must validate imports from the final venv-backed runtime interpreter after dependency installation." >&2
  exit 1
fi

scratch_dir="$(mktemp -d)"
trap 'rm -rf "${scratch_dir}"' EXIT
mkdir -p \
  "${scratch_dir}/ucx/lib" \
  "${scratch_dir}/ucc/lib" \
  "${scratch_dir}/ompi/lib"
touch \
  "${scratch_dir}/ucx/lib/libucs.so.0" \
  "${scratch_dir}/ucc/lib/libucc.so.1" \
  "${scratch_dir}/ompi/lib/libmpi.so.40"
unset EDGE_VLM_THOR_YOLO_HPCX_LD_LIBRARY_PATH
unset LD_LIBRARY_PATH
export EDGE_VLM_THOR_YOLO_HPCX_ROOT="${scratch_dir}"
# shellcheck disable=SC1090
source "${hpcx_env}"
expected_ld_library_path="${scratch_dir}/ucx/lib:${scratch_dir}/ucc/lib:${scratch_dir}/ompi/lib"
if [[ "${EDGE_VLM_THOR_YOLO_HPCX_LD_LIBRARY_PATH}" != "${expected_ld_library_path}" ]]; then
  echo "Resolved HPC-X library path did not preserve UCX/UCC/MPI ordering." >&2
  exit 1
fi
if [[ "${LD_LIBRARY_PATH}" != "${expected_ld_library_path}"* ]]; then
  echo "LD_LIBRARY_PATH did not inherit the resolved HPC-X library path." >&2
  exit 1
fi
source "${hpcx_env}"
if [[ "${LD_LIBRARY_PATH}" != "${expected_ld_library_path}" ]]; then
  echo "Sourcing the HPC-X helper twice must not duplicate library-path entries." >&2
  exit 1
fi
