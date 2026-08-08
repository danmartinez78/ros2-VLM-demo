#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
launch_file="${repo_root}/launch/ultralytics_yolo_detector.launch.py"

if [[ ! -f "${launch_file}" ]]; then
  echo "Launch file not found: ${launch_file}" >&2
  exit 1
fi

grep -Fq "package='edge_vlm_ros'" "${launch_file}"
grep -Fq "executable='edge_vlm_yolo_detection2d_adapter'" "${launch_file}"
grep -Fq 'The YOLO Detection2D adapter executable is not installed' "${launch_file}"
grep -Fq "_namespaced_topic" "${launch_file}"

if grep -Fq "package='yolo_ros'" "${launch_file}"; then
  echo "Launch file must not start yolo_ros on the host." >&2
  exit 1
fi

if grep -Eq '(^|[^[:alnum:]_])(uv|pip)([^[:alnum:]_]|$)' "${launch_file}"; then
  echo "Launch file must not reference runtime package managers: ${launch_file}" >&2
  exit 1
fi

if grep -Fq 'IncludeLaunchDescription' "${launch_file}"; then
  echo "Launch file must not include an upstream YOLO launch file." >&2
  exit 1
fi
