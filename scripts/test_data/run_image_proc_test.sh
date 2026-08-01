#!/usr/bin/env bash
# Bounded Thor smoke test using NVIDIA's image-proc rosbag.
#
# Environment overrides:
#   PLAYBACK_DURATION_SECONDS  Maximum wall-clock bag playback (default: 20)
#   RESULT_TIMEOUT_SECONDS     Maximum wait for a successful result (default: 120)
#   MAX_GENERATE_LENGTH        Output token limit used by the smoke test (default: 64)
#   SUCCESS_RESULTS_REQUIRED    Successful results required before exit (default: 1)
#   INSTRUCTION_DELIVERY_MODE  inline or structured (default: inline)
#   OBSERVATION_HISTORY_MAX_ENTRIES  Retained successful observations (default: 0)
#   OBSERVATION_HISTORY_MAX_CHARS    Retained observation character budget (default: 0)
#   SYSTEM_INSTRUCTION         Optional structured system message
#   TEST_PROMPT                Optional per-frame prompt override
#   ARTIFACT_DIR               Preserve logs, timing JSONL, and run manifest here
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
asset_root="${ROSBAG_DIR:-${repo_root}/test_data/rosbags}/image-proc"
env_file="${COSMOS_ENV_FILE:-${repo_root}/scripts/cosmos_env.sh}"
image_topic="/hawk_0_left_rgb_image"
result_topic="/cosmos/reasoning"
worker_socket="${WORKER_SOCKET_PATH:-/tmp/cosmos_edge_llm.sock}"
playback_duration="${PLAYBACK_DURATION_SECONDS:-20}"
result_timeout="${RESULT_TIMEOUT_SECONDS:-120}"
max_generate_length="${MAX_GENERATE_LENGTH:-64}"
success_results_required="${SUCCESS_RESULTS_REQUIRED:-1}"
instruction_delivery_mode="${INSTRUCTION_DELIVERY_MODE:-inline}"
observation_history_max_entries="${OBSERVATION_HISTORY_MAX_ENTRIES:-0}"
observation_history_max_chars="${OBSERVATION_HISTORY_MAX_CHARS:-0}"
system_instruction="${SYSTEM_INSTRUCTION:-}"
test_prompt="${TEST_PROMPT:-}"
artifact_dir="${ARTIFACT_DIR:-}"
benchmark_output_file=""

for value_name in playback_duration result_timeout max_generate_length success_results_required; do
  value="${!value_name}"
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${value_name} must be a positive integer; got '${value}'." >&2
    exit 2
  fi
done
for value_name in observation_history_max_entries observation_history_max_chars; do
  value="${!value_name}"
  if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
    echo "${value_name} must be a non-negative integer; got '${value}'." >&2
    exit 2
  fi
done
if [[ "${instruction_delivery_mode}" != "inline" && "${instruction_delivery_mode}" != "structured" ]]; then
  echo "INSTRUCTION_DELIVERY_MODE must be inline or structured." >&2
  exit 2
fi
if [[ -n "${artifact_dir}" ]]; then
  mkdir -p "${artifact_dir}"
  artifact_dir="$(cd -- "${artifact_dir}" && pwd)"
  benchmark_output_file="${artifact_dir}/benchmark.jsonl"
  : > "${benchmark_output_file}"
fi

launch_pid=""
bag_pid=""
result_echo_pid=""
test_passed=false
launch_log="$(mktemp /tmp/cosmos-image-proc-launch.XXXXXX.log)"
result_log="$(mktemp /tmp/cosmos-image-proc-results.XXXXXX.log)"

wait_for_pid_exit() {
  local pid="$1"
  local deadline=$((SECONDS + ${2:-3}))
  while kill -0 "${pid}" 2>/dev/null && (( SECONDS < deadline )); do
    sleep 0.2
  done
  ! kill -0 "${pid}" 2>/dev/null
}

wait_for_group_exit() {
  local pgid="$1"
  local deadline=$((SECONDS + ${2:-3}))
  while kill -0 -- "-${pgid}" 2>/dev/null && (( SECONDS < deadline )); do
    sleep 0.2
  done
  ! kill -0 -- "-${pgid}" 2>/dev/null
}

stop_isolated_group() {
  local leader_pid="$1"
  [[ -z "${leader_pid}" ]] && return 0

  local pgid own_pgid
  pgid="$(ps -o pgid= -p "${leader_pid}" 2>/dev/null | tr -d ' ')" || true
  own_pgid="$(ps -o pgid= -p "$$" 2>/dev/null | tr -d ' ')" || true

  if [[ -n "${pgid}" && "${pgid}" != "0" && -n "${own_pgid}" && "${pgid}" != "${own_pgid}" ]]; then
    kill -INT -- "-${pgid}" 2>/dev/null || true
    wait_for_group_exit "${pgid}" 3 || kill -TERM -- "-${pgid}" 2>/dev/null || true
    wait_for_group_exit "${pgid}" 3 || kill -KILL -- "-${pgid}" 2>/dev/null || true
  else
    kill -INT "${leader_pid}" 2>/dev/null || true
    wait_for_pid_exit "${leader_pid}" 3 || kill -TERM "${leader_pid}" 2>/dev/null || true
    wait_for_pid_exit "${leader_pid}" 3 || kill -KILL "${leader_pid}" 2>/dev/null || true
  fi
  wait "${leader_pid}" 2>/dev/null || true
}

cleanup() {
  if [[ -n "${result_echo_pid}" ]] && kill -0 "${result_echo_pid}" 2>/dev/null; then
    kill -TERM "${result_echo_pid}" 2>/dev/null || true
    wait_for_pid_exit "${result_echo_pid}" 2 || kill -KILL "${result_echo_pid}" 2>/dev/null || true
    wait "${result_echo_pid}" 2>/dev/null || true
  fi

  stop_isolated_group "${bag_pid}"
  stop_isolated_group "${launch_pid}"

  if [[ "${test_passed}" == true ]]; then
    rm -f "${launch_log}" "${result_log}"
  else
    echo "Diagnostic launch log: ${launch_log}" >&2
    echo "Diagnostic result log: ${result_log}" >&2
  fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

if [[ ! -d "${asset_root}" ]]; then
  bash "${script_dir}/download_rosbags.sh" download image-proc
fi

metadata="$(find "${asset_root}" -name metadata.yaml -print -quit)"
if [[ -z "${metadata}" ]]; then
  echo "No ROS 2 bag metadata found under ${asset_root}." >&2
  exit 1
fi
bag_path="$(dirname -- "${metadata}")"

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

set +u
# shellcheck disable=SC1090
source "${ros_setup}"
if [[ -n "${ROS_WORKSPACE:-}" && -f "${ROS_WORKSPACE}/install/setup.bash" ]]; then
  # shellcheck disable=SC1090
  source "${ROS_WORKSPACE}/install/setup.bash"
elif [[ -f "${repo_root}/../../install/setup.bash" ]]; then
  # shellcheck disable=SC1090
  source "${repo_root}/../../install/setup.bash"
fi
set -u

for variable in COSMOS_LLM_ENGINE_DIR COSMOS_MULTIMODAL_ENGINE_DIR EDGELLM_PLUGIN_PATH; do
  if [[ -z "${!variable:-}" ]]; then
    echo "Missing ${variable}; configure ${env_file} first." >&2
    exit 1
  fi
done

# Refuse to attach assertions or automatic cleanup to an existing deployment.
# Print process-group commands because killing only the worker may let ros2 launch respawn it.
existing_reasoners="$(pgrep -f '/cosmos_reasoner($| )' 2>/dev/null || true)"
existing_workers="$(pgrep -f '/cosmos_inference_worker($| )' 2>/dev/null || true)"
if [[ -n "${existing_reasoners}" || -n "${existing_workers}" ]]; then
  detected_pids="$(
    printf '%s\n%s\n' "${existing_reasoners}" "${existing_workers}" |
      sed '/^[[:space:]]*$/d' |
      sort -nu
  )"
  detected_pid_csv="$(printf '%s\n' "${detected_pids}" | paste -sd, -)"
  own_pgid="$(ps -o pgid= -p "$$" 2>/dev/null | tr -d ' ')" || true
  detected_pgids="$(
    ps -o pgid= -p "${detected_pid_csv}" 2>/dev/null |
      tr -d ' ' |
      sed '/^$/d' |
      sort -nu |
      awk -v own="${own_pgid}" '$1 != 0 && $1 != own'
  )"

  echo "Existing Cosmos deployment processes detected; stop them before testing." >&2
  ps -o pid,ppid,pgid,sid,etime,stat,cmd -p "${detected_pid_csv}" >&2 || true
  echo >&2
  echo "Run the following commands, then rerun this test:" >&2
  if [[ -n "${detected_pgids}" ]]; then
    while IFS= read -r pgid; do
      printf 'kill -TERM -- -%q\n' "${pgid}" >&2
    done <<< "${detected_pgids}"
    echo "sleep 3" >&2
    while IFS= read -r pgid; do
      printf 'kill -KILL -- -%q 2>/dev/null || true\n' "${pgid}" >&2
    done <<< "${detected_pgids}"
  else
    echo "# Could not identify a safe process group; inspect the process table above." >&2
  fi
  printf 'rm -f -- %q\n' "${worker_socket}" >&2

  # A preflight refusal is not a launched-test failure, so discard empty logs.
  test_passed=true
  exit 1
fi
if [[ -e "${worker_socket}" ]]; then
  echo "Stale worker socket exists: ${worker_socket}" >&2
  echo "No Cosmos process was found. Remove the stale socket with:" >&2
  printf 'rm -f -- %q\n' "${worker_socket}" >&2
  test_passed=true
  exit 1
fi

echo "Starting Cosmos reasoner on ${image_topic}..."
echo "  playback duration:    ${playback_duration} s maximum"
echo "  result timeout:       ${result_timeout} s"
echo "  max generate length:  ${max_generate_length}"
echo "  successful results:   ${success_results_required}"
echo "  delivery mode:        ${instruction_delivery_mode}"
echo "  observation history:  ${observation_history_max_entries} entries / ${observation_history_max_chars} chars"

launch_args=(
  image_topic:="${image_topic}"
  result_topic:="${result_topic}"
  llm_engine_dir:="${COSMOS_LLM_ENGINE_DIR}"
  multimodal_engine_dir:="${COSMOS_MULTIMODAL_ENGINE_DIR}"
  edge_llm_plugin_path:="${EDGELLM_PLUGIN_PATH}"
  max_generate_length:="${max_generate_length}"
  instruction_delivery_mode:="${instruction_delivery_mode}"
  observation_history_max_entries:="${observation_history_max_entries}"
  observation_history_max_chars:="${observation_history_max_chars}"
  use_sim_time:=true
)
[[ -n "${system_instruction}" ]] && launch_args+=(system_instruction:="${system_instruction}")
[[ -n "${test_prompt}" ]] && launch_args+=(prompt:="${test_prompt}")
[[ -n "${benchmark_output_file}" ]] && launch_args+=(benchmark_output_file:="${benchmark_output_file}")

setsid ros2 launch cosmos_ros2_video_reasoner cosmos_reasoner.launch.py \
  "${launch_args[@]}" >"${launch_log}" 2>&1 &
launch_pid=$!

echo "Waiting for Cosmos reasoner initialization..."
ready=false
deadline=$((SECONDS + 120))
while (( SECONDS < deadline )); do
  if ! kill -0 "${launch_pid}" 2>/dev/null; then
    echo "The Cosmos reasoner exited before playback started." >&2
    tail -40 "${launch_log}" >&2
    exit 1
  fi
  if ros2 topic info "${image_topic}" 2>/dev/null \
      | grep -Eq 'Subscription count: [1-9][0-9]*'; then
    ready=true
    break
  fi
  sleep 1
done
if [[ "${ready}" != true ]]; then
  echo "Timed out waiting for the reasoner subscription." >&2
  tail -40 "${launch_log}" >&2
  exit 1
fi

ros2 topic echo "${result_topic}" >"${result_log}" 2>&1 &
result_echo_pid=$!

subscriber_ready=false
deadline=$((SECONDS + 30))
while (( SECONDS < deadline )); do
  if ros2 topic info "${result_topic}" 2>/dev/null \
      | grep -Eq 'Subscription count: [1-9][0-9]*'; then
    subscriber_ready=true
    break
  fi
  sleep 0.5
done
if [[ "${subscriber_ready}" != true ]]; then
  echo "Timed out waiting for the result-topic subscriber." >&2
  exit 1
fi

echo "Playing NVIDIA image-proc bag: ${bag_path}"
setsid timeout --signal=INT --kill-after=5 "${playback_duration}" \
  ros2 bag play "${bag_path}" --clock > /dev/null 2>&1 &
bag_pid=$!

success=false
successful_results=0
deadline=$((SECONDS + result_timeout))
while (( SECONDS < deadline )); do
  successful_results="$(grep -c '^success: true
  if ! kill -0 "${launch_pid}" 2>/dev/null; then
    echo "The Cosmos reasoner exited before producing a successful result." >&2
    tail -40 "${launch_log}" >&2
    exit 1
  fi
  if ! kill -0 "${bag_pid}" 2>/dev/null && (( successful_results < success_results_required )); then
    echo "Bag playback ended after ${successful_results}/${success_results_required} successful results." >&2
    tail -40 "${launch_log}" >&2
    exit 1
  fi
  sleep 0.5
done

if [[ "${success}" != true ]]; then
  echo "Timed out after ${successful_results}/${success_results_required} successful results." >&2
  tail -40 "${launch_log}" >&2
  exit 1
fi

echo "--- First successful result ---"
awk 'BEGIN {RS="---\\n"; ORS="---\n"} /success: true/ {print; exit}' "${result_log}"

if [[ -n "${artifact_dir}" ]]; then
  cp "${launch_log}" "${artifact_dir}/launch.log"
  cp "${result_log}" "${artifact_dir}/results.log"
  python3 - "${artifact_dir}/manifest.json" <<PY
import json
import subprocess
import sys
from datetime import datetime, timezone

def git_value(*args):
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return None

manifest = {
    "schema_version": 1,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "git_commit": git_value("rev-parse", "HEAD"),
    "git_branch": git_value("branch", "--show-current"),
    "bag_path": """${bag_path}""",
    "image_topic": """${image_topic}""",
    "result_topic": """${result_topic}""",
    "llm_engine_dir": """${COSMOS_LLM_ENGINE_DIR}""",
    "multimodal_engine_dir": """${COSMOS_MULTIMODAL_ENGINE_DIR}""",
    "instruction_delivery_mode": """${instruction_delivery_mode}""",
    "observation_history_max_entries": int("""${observation_history_max_entries}"""),
    "observation_history_max_chars": int("""${observation_history_max_chars}"""),
    "max_generate_length": int("""${max_generate_length}"""),
    "successful_results_required": int("""${success_results_required}"""),
    "successful_results_observed": int("""${successful_results}"""),
}
with open(sys.argv[1], "w", encoding="utf-8") as stream:
    json.dump(manifest, stream, indent=2)
    stream.write("\\n")
PY
  echo "Artifacts preserved at: ${artifact_dir}"
fi

test_passed=true
cleanup
trap - EXIT INT TERM

orphan_workers="$(pgrep -f "cosmos_inference_worker.*${worker_socket}" 2>/dev/null || true)"
if [[ -n "${orphan_workers}" ]]; then
  echo "FAIL: orphan inference worker(s) remain: ${orphan_workers}" >&2
  exit 1
fi
if [[ -e "${worker_socket}" ]]; then
  echo "FAIL: worker socket remains after shutdown: ${worker_socket}" >&2
  exit 1
fi

echo "PASS: successful reasoning result received and all test processes cleaned up."
 "${result_log}" 2>/dev/null || true)"
  if (( successful_results >= success_results_required )); then
    success=true
    break
  fi
  if ! kill -0 "${launch_pid}" 2>/dev/null; then
    echo "The Cosmos reasoner exited before producing a successful result." >&2
    tail -40 "${launch_log}" >&2
    exit 1
  fi
  if ! kill -0 "${bag_pid}" 2>/dev/null && ! grep -q '^success: true$' "${result_log}"; then
    echo "Bag playback ended without a successful reasoning result." >&2
    tail -40 "${launch_log}" >&2
    exit 1
  fi
  sleep 0.5
done

if [[ "${success}" != true ]]; then
  echo "Timed out waiting for a successful reasoning result." >&2
  tail -40 "${launch_log}" >&2
  exit 1
fi

echo "--- First successful result ---"
awk 'BEGIN {RS="---\\n"; ORS="---\\n"} /success: true/ {print; exit}' "${result_log}"

test_passed=true
cleanup
trap - EXIT INT TERM

orphan_workers="$(pgrep -f "cosmos_inference_worker.*${worker_socket}" 2>/dev/null || true)"
if [[ -n "${orphan_workers}" ]]; then
  echo "FAIL: orphan inference worker(s) remain: ${orphan_workers}" >&2
  exit 1
fi
if [[ -e "${worker_socket}" ]]; then
  echo "FAIL: worker socket remains after shutdown: ${worker_socket}" >&2
  exit 1
fi

echo "PASS: successful reasoning result received and all test processes cleaned up."
