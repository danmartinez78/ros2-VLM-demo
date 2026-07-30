#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  echo "Usage: inspect_rosbag.sh /path/to/bag"
}

bag_path="${1:-}"
if [[ -z "${bag_path}" ]]; then
  usage >&2
  exit 2
fi
if [[ ! -e "${bag_path}" ]]; then
  echo "Bag path does not exist: ${bag_path}" >&2
  exit 1
fi

if ! command -v ros2 >/dev/null 2>&1; then
  if [[ -f /opt/ros/jazzy/setup.bash ]]; then
    # shellcheck disable=SC1091
    source /opt/ros/jazzy/setup.bash
  else
    echo "ROS 2 Jazzy is not installed or sourced." >&2
    exit 1
  fi
fi

bag_info="$(ros2 bag info "${bag_path}")"
printf '%s\n' "${bag_info}"

raw_topics="$(sed -n 's/.*Topic: \([^ ]*\).*Type: sensor_msgs\/msg\/Image.*/\1/p' <<<"${bag_info}")"
compressed_topics="$(sed -n 's/.*Topic: \([^ ]*\).*Type: sensor_msgs\/msg\/CompressedImage.*/\1/p' <<<"${bag_info}")"

echo
if [[ -n "${raw_topics}" ]]; then
  echo "Directly compatible sensor_msgs/msg/Image topics:"
  while IFS= read -r topic; do
    [[ -n "${topic}" ]] && echo "  ${topic}"
  done <<<"${raw_topics}"

  first_topic="$(head -n 1 <<<"${raw_topics}")"
  echo
  echo "Suggested test commands:"
  echo "  ros2 launch cosmos_ros2_video_reasoner cosmos_reasoner.launch.py image_topic:=${first_topic} use_sim_time:=true"
  printf '  ros2 bag play %q --clock\n' "${bag_path}"
else
  echo "No sensor_msgs/msg/Image topic was found."
fi

if [[ -n "${compressed_topics}" ]]; then
  echo
  echo "Compressed camera topics (decoder required):"
  while IFS= read -r topic; do
    [[ -n "${topic}" ]] && echo "  ${topic}"
  done <<<"${compressed_topics}"
fi

if [[ -z "${raw_topics}" && -z "${compressed_topics}" ]]; then
  echo "No supported camera topic type was found in this bag." >&2
  exit 1
fi
