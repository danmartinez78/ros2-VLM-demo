#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
launch_file="${repo_root}/launch/ultralytics_yolo_detector.launch.py"

[[ -f "${launch_file}" ]]

grep -Fq "package='yolo_ros'" "${launch_file}"
grep -Fq "executable='yolo_node'" "${launch_file}"
grep -Fq "The yolo_ros package is required" "${launch_file}"

if grep -Eq '(^|[^[:alnum:]_])uv([^[:alnum:]_]|$)' "${launch_file}"; then
  echo "Launch file must not reference uv at runtime: ${launch_file}" >&2
  exit 1
fi

if grep -Fq 'yolo.launch.py' "${launch_file}"; then
  echo "Launch file must not include the upstream yolo.launch.py entrypoint." >&2
  exit 1
fi

if grep -Fq 'IncludeLaunchDescription' "${launch_file}"; then
  echo "Launch file must start yolo_ros directly instead of including another launch file." >&2
  exit 1
fi
