#!/usr/bin/env bash
set -Eeuo pipefail

# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
# shellcheck disable=SC1091
source /opt/edge_vlm_yolo_ws/install/setup.bash

namespace="${YOLO_NAMESPACE:-yolo}"
namespace="${namespace#/}"
namespace="/${namespace}"

model="${YOLO_MODEL:-yolov8m.pt}"
if [[ -f "${model}" ]]; then
  resolved_model="${model}"
elif [[ -f "/opt/models/${model}" ]]; then
  resolved_model="/opt/models/${model}"
else
  echo "YOLO model not found inside the detector container: ${model}" >&2
  echo "Use the preloaded yolov8m.pt model or rebuild the image with a pinned replacement." >&2
  exit 1
fi

exec ros2 run yolo_ros yolo_node --ros-args \
  -r __ns:="${namespace}" \
  -r image_raw:="${YOLO_IMAGE_TOPIC:-/camera0/color/image_raw}" \
  -p model:="${resolved_model}" \
  -p device:="${YOLO_DEVICE:-cuda:0}" \
  -p image_reliability:=2
