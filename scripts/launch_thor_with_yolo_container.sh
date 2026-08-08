#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
compose_file="${repo_root}/docker/compose.thor-yolo.yml"

if [[ ! -f "${compose_file}" ]]; then
  echo "YOLO detector compose file not found: ${compose_file}" >&2
  exit 1
fi
if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required for the Thor YOLO detector container." >&2
  exit 1
fi
if ! docker compose version >/dev/null 2>&1; then
  echo "docker compose is required for the Thor YOLO detector container." >&2
  exit 1
fi
if ! command -v ros2 >/dev/null 2>&1; then
  echo "ros2 must already be available in the current shell before launching." >&2
  exit 1
fi

launch_args=()
yolo_model="yolov8m.pt"
yolo_namespace="yolo"
image_topic="/camera0/color/image_raw"

for arg in "$@"; do
  case "${arg}" in
    detector_backend:=*)
      detector_backend_value="${arg#detector_backend:=}"
      if [[ "${detector_backend_value}" != "ultralytics_yolo" ]]; then
        echo "launch_thor_with_yolo_container.sh only supports detector_backend:=ultralytics_yolo" >&2
        exit 1
      fi
      ;;
    yolo_model:=*)
      yolo_model="${arg#yolo_model:=}"
      ;;
    yolo_namespace:=*)
      yolo_namespace="${arg#yolo_namespace:=}"
      launch_args+=("${arg}")
      ;;
    image_topic:=*)
      image_topic="${arg#image_topic:=}"
      launch_args+=("${arg}")
      ;;
    *)
      launch_args+=("${arg}")
      ;;
  esac
done

export EDGE_VLM_YOLO_MODEL="${yolo_model}"
export EDGE_VLM_YOLO_NAMESPACE="${yolo_namespace}"
export EDGE_VLM_YOLO_IMAGE_TOPIC="${image_topic}"

cleanup() {
  docker compose -f "${compose_file}" stop thor-yolo-detector >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

docker compose -f "${compose_file}" up -d thor-yolo-detector
ros2 launch edge_vlm_ros thor_tracked_observation.launch.py \
  detector_backend:=ultralytics_yolo \
  "${launch_args[@]}"
