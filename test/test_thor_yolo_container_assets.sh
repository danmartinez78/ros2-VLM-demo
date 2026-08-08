#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
dockerfile="${repo_root}/docker/thor-yolo/Dockerfile"
entrypoint="${repo_root}/docker/thor-yolo/entrypoint.sh"
compose_file="${repo_root}/docker/compose.thor-yolo.yml"
launcher_script="${repo_root}/scripts/launch_thor_with_yolo_container.sh"

for required_file in "${dockerfile}" "${entrypoint}" "${compose_file}" "${launcher_script}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Required Thor YOLO container asset missing: ${required_file}" >&2
    exit 1
  fi
done

grep -Fq 'FROM ${BASE_IMAGE}' "${dockerfile}"
grep -Fq 'ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.05-py3' "${dockerfile}"
grep -Fq 'COPY --from=ghcr.io/astral-sh/uv:0.6.17 /uv /usr/local/bin/uv' "${dockerfile}"
grep -Fq 'uv pip install --system --no-deps ultralytics==8.4.6' "${dockerfile}"
grep -Fq 'git checkout "${YOLO_ROS_SHA}"' "${dockerfile}"
grep -Fq 'colcon build --merge-install --packages-select yolo_msgs yolo_ros' "${dockerfile}"
grep -Fq 'attempt_download_asset' "${dockerfile}"
grep -Fq 'torch changed from' "${dockerfile}"

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
