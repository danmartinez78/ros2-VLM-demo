#!/usr/bin/env bash
# Reproducible Thor full-pipeline benchmark matrix (A-F).
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"

rosbag_path=""
output_dir=""
modes_csv="A,B,C,D,E,F"
duration_seconds=90
tegrastats_interval_ms=1000
startup_wait_seconds=12
enable_rviz=false
manual_trigger_command=""
bag_image_topic="/image_rect"
pipeline_image_topic="/camera0/color/image_raw"
bag_camera_info_topic="/camera_info_rect"
pipeline_camera_info_topic="/camera_info"
dry_run=false
tegrastats_prefix=(tegrastats)
tegrastats_privilege_mode="unprivileged"
tegrastats_privilege_reason="Using unprivileged tegrastats capture."

usage() {
  cat <<'EOF'
Usage: run_thor_pipeline_benchmarks.sh --rosbag-path /abs/path/to/bag [options]

Options:
  --output-dir DIR              Root artifact directory (default: /tmp/thor_pipeline_bench_TIMESTAMP)
  --modes CSV                   Modes to run (default: A,B,C,D,E,F)
  --duration-seconds N          Per-run benchmark duration (default: 90)
  --tegrastats-interval-ms N    tegrastats polling interval (default: 1000)
  --startup-wait-seconds N      Wait after launch before measurements (default: 12)
  --manual-trigger-command CMD  Optional command for mode F manual trigger
  --bag-image-topic TOPIC        Rosbag image topic (default: /image_rect)
  --pipeline-image-topic TOPIC   Pipeline image topic (default: /camera0/color/image_raw)
  --bag-camera-info-topic TOPIC  Rosbag camera info topic (default: /camera_info_rect)
  --pipeline-camera-info-topic TOPIC
                                 Pipeline camera info topic (default: /camera_info)
  --enable-rviz                 Enable RViz in launched stacks (default: disabled)
  --dry-run                     Print commands without executing
  --help                        Show this help

Benchmark matrix:
  A: RT-DETR + adapter only baseline (VLM disabled)
  B: RT-DETR off, VLM on (edge_vlm.launch baseline)
  C: RT-DETR on, VLM continuous (sample_period_seconds=0)
  D: RT-DETR on, VLM 1 Hz (sample_period_seconds=1)
  E: RT-DETR on, VLM 0.5 Hz (sample_period_seconds=2)
  F: RT-DETR on, VLM event/manual (sample_period_seconds=3600 + optional manual trigger command)
EOF
}

while (($#)); do
  case "$1" in
    --rosbag-path) rosbag_path="$2"; shift 2 ;;
    --output-dir) output_dir="$2"; shift 2 ;;
    --modes) modes_csv="$2"; shift 2 ;;
    --duration-seconds) duration_seconds="$2"; shift 2 ;;
    --tegrastats-interval-ms) tegrastats_interval_ms="$2"; shift 2 ;;
    --startup-wait-seconds) startup_wait_seconds="$2"; shift 2 ;;
    --manual-trigger-command) manual_trigger_command="$2"; shift 2 ;;
    --bag-image-topic) bag_image_topic="$2"; shift 2 ;;
    --pipeline-image-topic) pipeline_image_topic="$2"; shift 2 ;;
    --bag-camera-info-topic) bag_camera_info_topic="$2"; shift 2 ;;
    --pipeline-camera-info-topic) pipeline_camera_info_topic="$2"; shift 2 ;;
    --enable-rviz) enable_rviz=true; shift ;;
    --dry-run) dry_run=true; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "${rosbag_path}" ]] || { echo "--rosbag-path is required" >&2; exit 2; }
[[ "${duration_seconds}" =~ ^[1-9][0-9]*$ ]] || { echo "duration must be positive integer" >&2; exit 2; }
[[ "${tegrastats_interval_ms}" =~ ^[1-9][0-9]*$ ]] || { echo "tegrastats interval must be positive integer" >&2; exit 2; }
[[ "${startup_wait_seconds}" =~ ^[1-9][0-9]*$ ]] || { echo "startup wait must be positive integer" >&2; exit 2; }
[[ -n "${bag_image_topic}" ]] || { echo "--bag-image-topic must be non-empty" >&2; exit 2; }
[[ -n "${pipeline_image_topic}" ]] || { echo "--pipeline-image-topic must be non-empty" >&2; exit 2; }
[[ -n "${bag_camera_info_topic}" ]] || { echo "--bag-camera-info-topic must be non-empty" >&2; exit 2; }
[[ -n "${pipeline_camera_info_topic}" ]] || { echo "--pipeline-camera-info-topic must be non-empty" >&2; exit 2; }

if [[ -z "${output_dir}" ]]; then
  output_dir="/tmp/thor_pipeline_bench_$(date -u +%Y%m%d_%H%M%S)"
fi
mkdir -p "${output_dir}"
output_dir="$(cd -- "${output_dir}" && pwd)"

if [[ "${dry_run}" == "false" ]]; then
  [[ -d "${rosbag_path}" && -f "${rosbag_path}/metadata.yaml" ]] || {
    echo "--rosbag-path must point to a ROS 2 bag directory containing metadata.yaml" >&2
    exit 2
  }
fi

if [[ -f "${repo_root}/scripts/edge_vlm_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${repo_root}/scripts/edge_vlm_env.sh"
fi

if [[ "${dry_run}" == "false" ]]; then
  set +u
  # shellcheck disable=SC1091
  source "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
  if [[ -n "${ROS_WORKSPACE:-}" && -f "${ROS_WORKSPACE}/install/setup.bash" ]]; then
    # shellcheck disable=SC1091
    source "${ROS_WORKSPACE}/install/setup.bash"
  elif [[ -f "${repo_root}/../../install/setup.bash" ]]; then
    # shellcheck disable=SC1091
    source "${repo_root}/../../install/setup.bash"
  fi
  set -u

  for env_name in EDGE_VLM_LLM_ENGINE_DIR EDGE_VLM_MULTIMODAL_ENGINE_DIR EDGELLM_PLUGIN_PATH; do
    if [[ -z "${!env_name:-}" ]]; then
      echo "Missing required environment variable: ${env_name}" >&2
      exit 1
    fi
  done
else
  : "${EDGE_VLM_LLM_ENGINE_DIR:=/tmp/llm_engine}"
  : "${EDGE_VLM_MULTIMODAL_ENGINE_DIR:=/tmp/mm_engine}"
  : "${EDGELLM_PLUGIN_PATH:=/tmp/libNvInfer_edgellm_plugin.so}"
fi

run_cmd() {
  if [[ "${dry_run}" == "true" ]]; then
    echo "[DRY RUN] $*"
    return 0
  fi
  "$@"
}

run_shell() {
  local cmd="$1"
  if [[ "${dry_run}" == "true" ]]; then
    echo "[DRY RUN] ${cmd}"
    return 0
  fi
  bash -lc "${cmd}"
}

configure_tegrastats_capture() {
  if [[ "${dry_run}" == "true" ]]; then
    tegrastats_prefix=(tegrastats)
    tegrastats_privilege_mode="dry-run"
    tegrastats_privilege_reason="Dry run; telemetry capture not executed."
    return
  fi

  if ! command -v tegrastats >/dev/null 2>&1; then
    echo "tegrastats command not found in PATH" >&2
    exit 1
  fi

  if ! command -v sudo >/dev/null 2>&1; then
    tegrastats_prefix=(tegrastats)
    tegrastats_privilege_mode="unprivileged"
    tegrastats_privilege_reason="sudo unavailable; EMC/GR3D may be missing on Thor."
    return
  fi

  if sudo -n true >/dev/null 2>&1; then
    tegrastats_prefix=(sudo -n tegrastats)
    tegrastats_privilege_mode="sudo-noninteractive"
    tegrastats_privilege_reason="Using cached sudo credentials for tegrastats."
    return
  fi

  if [[ -t 0 ]] && sudo -v >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
    tegrastats_prefix=(sudo -n tegrastats)
    tegrastats_privilege_mode="sudo-interactive"
    tegrastats_privilege_reason="sudo -v succeeded; using sudo -n tegrastats."
    return
  fi

  tegrastats_prefix=(tegrastats)
  tegrastats_privilege_mode="unprivileged"
  tegrastats_privilege_reason="sudo privilege unavailable; EMC/GR3D may be missing on Thor."
}

cleanup_pids=()
cleanup() {
  if [[ "${#cleanup_pids[@]}" -eq 0 ]]; then
    return
  fi
  for pid in "${cleanup_pids[@]}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
  sleep 1
  for pid in "${cleanup_pids[@]}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill -KILL "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT

mode_description() {
  case "$1" in
    A) echo "RT-DETR + adapter baseline (VLM disabled)" ;;
    B) echo "RT-DETR off, VLM baseline" ;;
    C) echo "RT-DETR on, VLM continuous" ;;
    D) echo "RT-DETR on, VLM 1 Hz" ;;
    E) echo "RT-DETR on, VLM 0.5 Hz" ;;
    F) echo "RT-DETR on, VLM event/manual" ;;
    *) echo "unknown" ;;
  esac
}

build_launch_args_for_mode() {
  local mode="$1"
  local bench_file="$2"
  local common=(
    "use_sim_time:=true"
    "llm_engine_dir:=${EDGE_VLM_LLM_ENGINE_DIR}"
    "multimodal_engine_dir:=${EDGE_VLM_MULTIMODAL_ENGINE_DIR}"
    "edge_llm_plugin_path:=${EDGELLM_PLUGIN_PATH}"
    "benchmark_output_file:=${bench_file}"
  )
  local thor_common=("enable_rviz:=${enable_rviz}" "${common[@]}")
  case "${mode}" in
    A)
      launch_args=(ros2 launch edge_vlm_ros thor_tracked_observation.launch.py start_rtdetr:=true start_vlm:=false image_topic:="${pipeline_image_topic}" sample_period_seconds:=3600.0 min_vlm_interval_seconds:=0.0 "${thor_common[@]}")
      ;;
    B)
      launch_args=(ros2 launch edge_vlm_ros edge_vlm.launch.py image_topic:="${pipeline_image_topic}" result_topic:=/vlm/result sample_period_seconds:=0.0 min_vlm_interval_seconds:=0.0 "${common[@]}")
      ;;
    C)
      launch_args=(ros2 launch edge_vlm_ros thor_tracked_observation.launch.py start_rtdetr:=true image_topic:="${pipeline_image_topic}" sample_period_seconds:=0.0 min_vlm_interval_seconds:=0.0 "${thor_common[@]}")
      ;;
    D)
      launch_args=(ros2 launch edge_vlm_ros thor_tracked_observation.launch.py start_rtdetr:=true image_topic:="${pipeline_image_topic}" sample_period_seconds:=1.0 min_vlm_interval_seconds:=0.0 "${thor_common[@]}")
      ;;
    E)
      launch_args=(ros2 launch edge_vlm_ros thor_tracked_observation.launch.py start_rtdetr:=true image_topic:="${pipeline_image_topic}" sample_period_seconds:=2.0 min_vlm_interval_seconds:=0.0 "${thor_common[@]}")
      ;;
    F)
      launch_args=(ros2 launch edge_vlm_ros thor_tracked_observation.launch.py start_rtdetr:=true image_topic:="${pipeline_image_topic}" sample_period_seconds:=3600.0 min_vlm_interval_seconds:=0.0 "${thor_common[@]}")
      ;;
    *)
      return 1
      ;;
  esac
}

IFS=',' read -r -a modes <<< "${modes_csv}"
bag_play_remap_args=(
  "--remap" "${bag_image_topic}:=${pipeline_image_topic}"
  "${bag_camera_info_topic}:=${pipeline_camera_info_topic}"
)

configure_tegrastats_capture

echo "Thor pipeline benchmark root: ${output_dir}"
echo "Modes: ${modes_csv}"
echo "tegrastats mode: ${tegrastats_privilege_mode} (${tegrastats_privilege_reason})"

git_sha="$(git -C "${repo_root}" rev-parse HEAD 2>/dev/null || true)"

for mode in "${modes[@]}"; do
  mode="$(echo "${mode}" | xargs)"
  [[ -n "${mode}" ]] || continue
  run_dir="${output_dir}/run_${mode}"
  mkdir -p "${run_dir}"

  launch_log="${run_dir}/launch.log"
  tegra_log="${run_dir}/tegrastats.log"
  bench_file="${run_dir}/benchmark.jsonl"

  launch_args=()
  build_launch_args_for_mode "${mode}" "${bench_file}" || {
    echo "Skipping unknown mode '${mode}'" >&2
    continue
  }

  manual_trigger_json="$(MANUAL_TRIGGER_COMMAND="${manual_trigger_command}" python3 - <<'PY'
import json
import os
value = os.environ.get("MANUAL_TRIGGER_COMMAND", "")
print("null" if value == "" else json.dumps(value))
PY
)"
  RUN_ID="run_${mode}" \
  MODE="${mode}" \
  DESCRIPTION="$(mode_description "${mode}")" \
  GIT_SHA="${git_sha}" \
  ROSBAG_PATH="${rosbag_path}" \
  DURATION_SECONDS="${duration_seconds}" \
  TEGRA_INTERVAL_MS="${tegrastats_interval_ms}" \
  TEGRASTATS_PRIVILEGE_MODE="${tegrastats_privilege_mode}" \
  TEGRASTATS_PRIVILEGE_REASON="${tegrastats_privilege_reason}" \
  TEGRASTATS_COMMAND_PREFIX="$(printf "%q " "${tegrastats_prefix[@]}")" \
  MANUAL_TRIGGER_JSON="${manual_trigger_json}" \
  BAG_IMAGE_TOPIC="${bag_image_topic}" \
  PIPELINE_IMAGE_TOPIC="${pipeline_image_topic}" \
  BAG_CAMERA_INFO_TOPIC="${bag_camera_info_topic}" \
  PIPELINE_CAMERA_INFO_TOPIC="${pipeline_camera_info_topic}" \
  python3 - "${run_dir}/run_config.json" <<'PY'
import json
import os
import sys

manual_raw = os.environ.get("MANUAL_TRIGGER_JSON", "null")
manual = None if manual_raw == "null" else json.loads(manual_raw)

payload = {
    "run_id": os.environ["RUN_ID"],
    "mode": os.environ["MODE"],
    "description": os.environ["DESCRIPTION"],
    "git_sha": os.environ["GIT_SHA"],
    "rosbag_path": os.environ["ROSBAG_PATH"],
    "duration_seconds": int(os.environ["DURATION_SECONDS"]),
    "tegrastats_interval_ms": int(os.environ["TEGRA_INTERVAL_MS"]),
    "tegrastats_capture": {
        "privilege_mode": os.environ.get("TEGRASTATS_PRIVILEGE_MODE", "unknown"),
        "privilege_reason": os.environ.get("TEGRASTATS_PRIVILEGE_REASON", ""),
        "command_prefix": os.environ.get("TEGRASTATS_COMMAND_PREFIX", "").strip(),
    },
    "manual_trigger_command": manual,
    "input_mappings": {
        "image": {
            "bag_topic": os.environ["BAG_IMAGE_TOPIC"],
            "pipeline_topic": os.environ["PIPELINE_IMAGE_TOPIC"],
        },
        "camera_info": {
            "bag_topic": os.environ["BAG_CAMERA_INFO_TOPIC"],
            "pipeline_topic": os.environ["PIPELINE_CAMERA_INFO_TOPIC"],
        },
    },
    "bag_play_remaps": [
        f"{os.environ['BAG_IMAGE_TOPIC']}:={os.environ['PIPELINE_IMAGE_TOPIC']}",
        f"{os.environ['BAG_CAMERA_INFO_TOPIC']}:={os.environ['PIPELINE_CAMERA_INFO_TOPIC']}",
    ],
}
with open(sys.argv[1], "w", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2)
    stream.write("\n")
PY

  echo "==> Mode ${mode}: $(mode_description "${mode}")"
  printf "    Launch: "
  printf "%q " "${launch_args[@]}"
  echo

  cleanup_pids=()

  if [[ "${dry_run}" == "false" ]]; then
    : > "${bench_file}"
    ("${tegrastats_prefix[@]}" --interval "${tegrastats_interval_ms}" > "${tegra_log}" 2>&1) &
    cleanup_pids+=("$!")

    "${launch_args[@]}" > "${launch_log}" 2>&1 &
    launch_pid=$!
    cleanup_pids+=("${launch_pid}")

    sleep "${startup_wait_seconds}"

    ros2 topic hz /detections > "${run_dir}/detections_hz.log" 2>&1 &
    cleanup_pids+=("$!")
    ros2 topic hz /tracked_observation > "${run_dir}/tracked_observation_hz.log" 2>&1 &
    cleanup_pids+=("$!")
    ros2 topic hz /vlm/result > "${run_dir}/vlm_result_hz.log" 2>&1 &
    cleanup_pids+=("$!")

    timeout --signal=INT --kill-after=5 "${duration_seconds}" ros2 bag play "${rosbag_path}" --clock --loop "${bag_play_remap_args[@]}" > "${run_dir}/bag_play.log" 2>&1 &
    bag_pid=$!
    cleanup_pids+=("${bag_pid}")

    if [[ "${mode}" == "F" && -n "${manual_trigger_command}" ]]; then
      echo "Executing manual trigger command for mode F"
      run_shell "${manual_trigger_command}" > "${run_dir}/manual_trigger.log" 2>&1 || true
    fi

    wait "${bag_pid}" || true

    sleep 2
    cleanup
    cleanup_pids=()

    if [[ -s "${bench_file}" ]]; then
      python3 "${script_dir}/collect_ros_metrics.py" \
        --input "${bench_file}" \
        --warmup 3 \
        --output "${run_dir}/ros_metrics.json" || true
    fi
  else
    printf "[DRY RUN] "
    printf "%q " "${tegrastats_prefix[@]}"
    echo "--interval ${tegrastats_interval_ms} > ${tegra_log}"
    printf "[DRY RUN] "
    printf "%q " "${launch_args[@]}"
    echo "> ${launch_log}"
    printf "[DRY RUN] timeout %q ros2 bag play %q --clock --loop " "${duration_seconds}" "${rosbag_path}"
    printf "%q " "${bag_play_remap_args[@]}"
    echo
    echo "[DRY RUN] ros2 topic hz /detections > ${run_dir}/detections_hz.log"
    echo "[DRY RUN] ros2 topic hz /tracked_observation > ${run_dir}/tracked_observation_hz.log"
    echo "[DRY RUN] ros2 topic hz /vlm/result > ${run_dir}/vlm_result_hz.log"
    if [[ "${mode}" == "F" && -n "${manual_trigger_command}" ]]; then
      echo "[DRY RUN] ${manual_trigger_command} > ${run_dir}/manual_trigger.log"
    fi
  fi

done

run_cmd python3 "${script_dir}/thor_pipeline_benchmark_report.py" \
  --run-root "${output_dir}" \
  --modes "${modes_csv}" \
  --output "${output_dir}/comparison_report.json" \
  --text "${output_dir}/comparison_report.txt"

if [[ "${dry_run}" == "false" ]]; then
  invalid_modes="$(
    python3 - "${output_dir}/comparison_report.json" <<'PY'
import json
import sys

report_path = sys.argv[1]
with open(report_path, "r", encoding="utf-8") as stream:
    report = json.load(stream)
modes = report.get("invalid_runs") or []
print(",".join(modes))
PY
  )"
  if [[ -n "${invalid_modes}" ]]; then
    echo "Benchmark invalid runs detected: ${invalid_modes}" >&2
    echo "See ${output_dir}/comparison_report.json for validation details." >&2
    exit 3
  fi
fi

echo "Benchmark complete: ${output_dir}"
echo "Comparison JSON: ${output_dir}/comparison_report.json"
echo "Comparison text: ${output_dir}/comparison_report.txt"
