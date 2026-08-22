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
enable_rviz=false
manual_trigger_command=""
dry_run=false

usage() {
  cat <<'EOF'
Usage: run_thor_pipeline_benchmarks.sh --rosbag-path /abs/path/to/bag [options]

Options:
  --output-dir DIR              Root artifact directory (default: /tmp/thor_pipeline_bench_TIMESTAMP)
  --modes CSV                   Modes to run (default: A,B,C,D,E,F)
  --duration-seconds N          Per-run benchmark duration (default: 90)
  --tegrastats-interval-ms N    tegrastats polling interval (default: 1000)
  --manual-trigger-command CMD  Optional command for mode F manual trigger
  --enable-rviz                 Enable RViz in launched stacks (default: disabled)
  --dry-run                     Print commands without executing
  --help                        Show this help

Benchmark matrix:
  A: RT-DETR on, VLM off-like (sample_period_seconds=3600)
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
    --manual-trigger-command) manual_trigger_command="$2"; shift 2 ;;
    --enable-rviz) enable_rviz=true; shift ;;
    --dry-run) dry_run=true; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "${rosbag_path}" ]] || { echo "--rosbag-path is required" >&2; exit 2; }
[[ "${duration_seconds}" =~ ^[1-9][0-9]*$ ]] || { echo "duration must be positive integer" >&2; exit 2; }
[[ "${tegrastats_interval_ms}" =~ ^[1-9][0-9]*$ ]] || { echo "tegrastats interval must be positive integer" >&2; exit 2; }

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

cleanup_pids=()
cleanup() {
  for pid in "${cleanup_pids[@]:-}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
  sleep 1
  for pid in "${cleanup_pids[@]:-}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill -KILL "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT

mode_description() {
  case "$1" in
    A) echo "RT-DETR on, VLM off-like" ;;
    B) echo "RT-DETR off, VLM baseline" ;;
    C) echo "RT-DETR on, VLM continuous" ;;
    D) echo "RT-DETR on, VLM 1 Hz" ;;
    E) echo "RT-DETR on, VLM 0.5 Hz" ;;
    F) echo "RT-DETR on, VLM event/manual" ;;
    *) echo "unknown" ;;
  esac
}

launch_command_for_mode() {
  local mode="$1"
  local bench_file="$2"
  local common="use_sim_time:=true enable_rviz:=${enable_rviz} llm_engine_dir:=${EDGE_VLM_LLM_ENGINE_DIR} multimodal_engine_dir:=${EDGE_VLM_MULTIMODAL_ENGINE_DIR} edge_llm_plugin_path:=${EDGELLM_PLUGIN_PATH} benchmark_output_file:=${bench_file}"
  case "${mode}" in
    A)
      echo "ros2 launch edge_vlm_ros thor_tracked_observation.launch.py start_rtdetr:=true sample_period_seconds:=3600.0 min_vlm_interval_seconds:=0.0 ${common}"
      ;;
    B)
      echo "ros2 launch edge_vlm_ros edge_vlm.launch.py image_topic:=/camera0/color/image_raw result_topic:=/vlm/result sample_period_seconds:=0.0 min_vlm_interval_seconds:=0.0 llm_engine_dir:=${EDGE_VLM_LLM_ENGINE_DIR} multimodal_engine_dir:=${EDGE_VLM_MULTIMODAL_ENGINE_DIR} edge_llm_plugin_path:=${EDGELLM_PLUGIN_PATH} benchmark_output_file:=${bench_file} use_sim_time:=true"
      ;;
    C)
      echo "ros2 launch edge_vlm_ros thor_tracked_observation.launch.py start_rtdetr:=true sample_period_seconds:=0.0 min_vlm_interval_seconds:=0.0 ${common}"
      ;;
    D)
      echo "ros2 launch edge_vlm_ros thor_tracked_observation.launch.py start_rtdetr:=true sample_period_seconds:=1.0 min_vlm_interval_seconds:=0.0 ${common}"
      ;;
    E)
      echo "ros2 launch edge_vlm_ros thor_tracked_observation.launch.py start_rtdetr:=true sample_period_seconds:=2.0 min_vlm_interval_seconds:=0.0 ${common}"
      ;;
    F)
      echo "ros2 launch edge_vlm_ros thor_tracked_observation.launch.py start_rtdetr:=true sample_period_seconds:=3600.0 min_vlm_interval_seconds:=0.0 ${common}"
      ;;
    *)
      return 1
      ;;
  esac
}

IFS=',' read -r -a modes <<< "${modes_csv}"

echo "Thor pipeline benchmark root: ${output_dir}"
echo "Modes: ${modes_csv}"

for mode in "${modes[@]}"; do
  mode="$(echo "${mode}" | xargs)"
  [[ -n "${mode}" ]] || continue
  run_dir="${output_dir}/run_${mode}"
  mkdir -p "${run_dir}"

  launch_log="${run_dir}/launch.log"
  tegra_log="${run_dir}/tegrastats.log"
  bench_file="${run_dir}/benchmark.jsonl"

  launch_cmd="$(launch_command_for_mode "${mode}" "${bench_file}")" || {
    echo "Skipping unknown mode '${mode}'" >&2
    continue
  }

  git_sha="$(git -C "${repo_root}" rev-parse HEAD 2>/dev/null || true)"
  manual_trigger_json="$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "${manual_trigger_command}")"
  cat > "${run_dir}/run_config.json" <<EOF
{
  "run_id": "run_${mode}",
  "mode": "${mode}",
  "description": "$(mode_description "${mode}")",
  "git_sha": "${git_sha}",
  "rosbag_path": "${rosbag_path}",
  "duration_seconds": ${duration_seconds},
  "tegrastats_interval_ms": ${tegrastats_interval_ms},
  "manual_trigger_command": ${manual_trigger_json}
}
EOF

  echo "==> Mode ${mode}: $(mode_description "${mode}")"
  echo "    Launch: ${launch_cmd}"

  cleanup_pids=()

  if [[ "${dry_run}" == "false" ]]; then
    : > "${bench_file}"
    (tegrastats --interval "${tegrastats_interval_ms}" > "${tegra_log}" 2>&1) &
    cleanup_pids+=("$!")

    run_shell "${launch_cmd} > '${launch_log}' 2>&1" &
    launch_pid=$!
    cleanup_pids+=("${launch_pid}")

    sleep 12

    timeout "${duration_seconds}" ros2 topic hz /detections > "${run_dir}/detections_hz.log" 2>&1 &
    cleanup_pids+=("$!")
    timeout "${duration_seconds}" ros2 topic hz /tracked_observation > "${run_dir}/tracked_observation_hz.log" 2>&1 &
    cleanup_pids+=("$!")
    timeout "${duration_seconds}" ros2 topic hz /vlm/result > "${run_dir}/vlm_result_hz.log" 2>&1 &
    cleanup_pids+=("$!")

    if [[ "${mode}" == "F" && -n "${manual_trigger_command}" ]]; then
      echo "Executing manual trigger command for mode F"
      run_shell "${manual_trigger_command}" > "${run_dir}/manual_trigger.log" 2>&1 || true
    fi

    timeout --signal=INT --kill-after=5 "${duration_seconds}" ros2 bag play "${rosbag_path}" --clock > "${run_dir}/bag_play.log" 2>&1 || true

    sleep 5
    cleanup
    cleanup_pids=()

    if [[ -s "${bench_file}" ]]; then
      python3 "${script_dir}/collect_ros_metrics.py" \
        --input "${bench_file}" \
        --warmup 3 \
        --output "${run_dir}/ros_metrics.json" || true
    fi
  else
    echo "[DRY RUN] tegrastats --interval ${tegrastats_interval_ms} > ${tegra_log}"
    echo "[DRY RUN] ${launch_cmd} > ${launch_log}"
    echo "[DRY RUN] timeout ${duration_seconds} ros2 bag play ${rosbag_path} --clock"
    echo "[DRY RUN] timeout ${duration_seconds} ros2 topic hz /detections > ${run_dir}/detections_hz.log"
    echo "[DRY RUN] timeout ${duration_seconds} ros2 topic hz /tracked_observation > ${run_dir}/tracked_observation_hz.log"
    echo "[DRY RUN] timeout ${duration_seconds} ros2 topic hz /vlm/result > ${run_dir}/vlm_result_hz.log"
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

echo "Benchmark complete: ${output_dir}"
echo "Comparison JSON: ${output_dir}/comparison_report.json"
echo "Comparison text: ${output_dir}/comparison_report.txt"
