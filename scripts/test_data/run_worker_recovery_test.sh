#!/usr/bin/env bash
# Thor hardware validation: watchdog-triggered recovery test.
#
# Uses a 1-second inference deadline (worker_inference_deadline_seconds=1) to
# force the worker-side watchdog to fire deterministically — Cosmos inference
# on the validated Thor configuration reliably exceeds 1 second, so the
# watchdog fires on the first request without injecting any artificial hang.
#
# Verifications (all 6 required):
#   1. WATCHDOG diagnostic appears in worker stderr.
#   2. Worker PID changes after respawn.
#   3. Exactly one failure is published for the expired request.
#   4. A later successful reasoning result is received.
#   5. cosmos_reasoner PID does not change.
#   6. No orphan worker process or socket file remains after shutdown.
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
asset_root="${ROSBAG_DIR:-${repo_root}/test_data/rosbags}/image-proc"
env_file="${COSMOS_ENV_FILE:-${repo_root}/scripts/cosmos_env.sh}"
image_topic="/hawk_0_left_rgb_image"
result_topic="/cosmos/reasoning"
worker_socket="${WORKER_SOCKET_PATH:-/tmp/cosmos_edge_llm.sock}"
# 1-second deadline is below any real Cosmos inference time on Thor hardware;
# the watchdog fires deterministically without needing an injected hang.
watchdog_deadline=1
# Client timeout must exceed the watchdog deadline so the client sees a clean
# EOF instead of a SO_RCVTIMEO error.
client_timeout=20

launch_pid=""
bag_pid=""
launch_log="$(mktemp /tmp/cosmos-recovery-launch.XXXXXX.log)"

cleanup() {
  if [[ -n "${bag_pid}" ]] && kill -0 "${bag_pid}" 2>/dev/null; then
    kill -INT "${bag_pid}" 2>/dev/null || true
    wait "${bag_pid}" 2>/dev/null || true
  fi
  if [[ -n "${launch_pid}" ]] && kill -0 "${launch_pid}" 2>/dev/null; then
    kill -INT "${launch_pid}" 2>/dev/null || true
    wait "${launch_pid}" 2>/dev/null || true
  fi
  rm -f "${launch_log}"
}
trap cleanup EXIT INT TERM

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
  echo "ROS 2 ${ros_distro} is not installed." >&2
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

# ── PID helpers ──────────────────────────────────────────────────────────────
# Scope the worker lookup to the specific socket path to avoid matching
# unrelated processes that happen to share a similar command name.
worker_pid() {
  # Full-command match ("-f") scoped to the exact socket path to avoid
  # collisions with other test runs or unrelated processes.
  pgrep -f "cosmos_inference_worker.*${worker_socket}" || true
}

reasoner_pid() {
  pgrep -f "cosmos_reasoner" || true
}

wait_for_worker() {
  local excluded_pid="${1:-}"
  local deadline=$((SECONDS + 120))
  local pid=""
  while (( SECONDS < deadline )); do
    pid="$(worker_pid)"
    if [[ -n "${pid}" && "${pid}" != "${excluded_pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      printf '%s\n' "${pid}"
      return 0
    fi
    sleep 1
  done
  return 1
}

wait_for_success_result() {
  local max_attempts="${1:-6}"
  local attempts=0
  local output=""
  while (( attempts < max_attempts )); do
    output="$(mktemp /tmp/cosmos-recovery-result.XXXXXX)"
    if timeout 120 ros2 topic echo "${result_topic}" --once >"${output}" 2>&1; then
      if grep -q '^success: true$' "${output}"; then
        cat "${output}"
        rm -f "${output}"
        return 0
      fi
      echo "Observed transient failed result (expected during reconnect):"
      cat "${output}"
    fi
    rm -f "${output}"
    attempts=$((attempts + 1))
  done
  return 1
}

count_failure_results() {
  # Count consecutive failed results before the first success.  Returns the
  # count in the caller's "failure_count" variable.
  failure_count=0
  local output=""
  while true; do
    output="$(mktemp /tmp/cosmos-recovery-count.XXXXXX)"
    if timeout 30 ros2 topic echo "${result_topic}" --once >"${output}" 2>&1; then
      if grep -q '^success: true$' "${output}"; then
        cat "${output}"
        rm -f "${output}"
        return 0
      fi
      failure_count=$(( failure_count + 1 ))
    fi
    rm -f "${output}"
    if (( failure_count >= 5 )); then
      return 1   # more than expected — something is wrong
    fi
  done
}

# ── Launch ───────────────────────────────────────────────────────────────────
echo "Starting watchdog-triggered recovery test..."
echo "  worker_inference_deadline_seconds = ${watchdog_deadline}"
echo "  worker_request_timeout_seconds    = ${client_timeout}"
echo "  worker_socket_path                = ${worker_socket}"

ros2 launch cosmos_ros2_video_reasoner cosmos_reasoner.launch.py \
  image_topic:="${image_topic}" \
  result_topic:="${result_topic}" \
  llm_engine_dir:="${COSMOS_LLM_ENGINE_DIR}" \
  multimodal_engine_dir:="${COSMOS_MULTIMODAL_ENGINE_DIR}" \
  edge_llm_plugin_path:="${EDGELLM_PLUGIN_PATH}" \
  worker_socket_path:="${worker_socket}" \
  worker_inference_deadline_seconds:="${watchdog_deadline}" \
  worker_request_timeout_seconds:="${client_timeout}" \
  sample_period_seconds:=1.0 \
  max_generate_length:=64 \
  use_sim_time:=true 2>&1 | tee "${launch_log}" &
launch_pid=$!

# ── Wait for initial worker and reasoner ─────────────────────────────────────
old_worker_pid="$(wait_for_worker)" || {
  echo "Timed out waiting for the initial inference worker." >&2
  exit 1
}
echo "Initial worker PID:   ${old_worker_pid}"

# Record cosmos_reasoner PID for verification 5.
old_reasoner_pid="$(reasoner_pid)" || true
echo "cosmos_reasoner PID:  ${old_reasoner_pid}"

# Wait for the reasoner subscription to come up before publishing frames.
ready=false
deadline=$((SECONDS + 120))
while (( SECONDS < deadline )); do
  if ros2 topic info "${image_topic}" 2>/dev/null \
      | grep -Eq 'Subscription count: [1-9][0-9]*'; then
    ready=true
    break
  fi
  sleep 1
done
if [[ "${ready}" != true ]]; then
  echo "Timed out waiting for the reasoner subscription." >&2
  exit 1
fi

ros2 bag play "${bag_path}" --clock --loop &
bag_pid=$!

# ── Watchdog fires on first inference request ─────────────────────────────────
# The 1-second deadline is shorter than any real Cosmos inference call on Thor;
# the watchdog fires deterministically and the worker self-terminates.
# We wait for exactly one failure (verification 3) followed by a success.
echo "Waiting for watchdog expiry and worker self-termination..."

failure_count=0
success_output="$(mktemp /tmp/cosmos-recovery-success.XXXXXX)"
# Drain results: count failures before the first success (verif. 3 + 4).
count_failure_results >"${success_output}" || {
  echo "Did not receive a successful result after the watchdog fired." >&2
  exit 1
}

echo "Failures before recovery: ${failure_count}"
if (( failure_count != 1 )); then
  echo "FAIL (verification 3): expected exactly 1 failure for the expired request," \
       "got ${failure_count}." >&2
  exit 1
fi
echo "PASS (verification 3): exactly 1 failure for the expired request."

echo "--- First successful result after recovery ---"
cat "${success_output}"
rm -f "${success_output}"
echo "PASS (verification 4): successful reasoning result received after recovery."

# ── Verify worker PID changed (verification 2) ───────────────────────────────
new_worker_pid="$(wait_for_worker "${old_worker_pid}")" || {
  echo "FAIL (verification 2): launch did not respawn the inference worker." >&2
  exit 1
}
echo "Respawned worker PID: ${new_worker_pid}"
if [[ "${old_worker_pid}" == "${new_worker_pid}" ]]; then
  echo "FAIL (verification 2): worker PID did not change after self-termination." >&2
  exit 1
fi
echo "PASS (verification 2): worker PID changed (${old_worker_pid} → ${new_worker_pid})."

# ── Verify WATCHDOG diagnostic appeared in the launch log (verification 1) ──
if grep -q 'WATCHDOG: inference deadline' "${launch_log}"; then
  echo "PASS (verification 1): WATCHDOG diagnostic found in launch log."
else
  echo "FAIL (verification 1): WATCHDOG diagnostic not found in launch log." >&2
  echo "Launch log tail:" >&2
  tail -20 "${launch_log}" >&2
  exit 1
fi

# ── Verify cosmos_reasoner PID did not change (verification 5) ───────────────
new_reasoner_pid="$(reasoner_pid)" || true
if [[ -n "${old_reasoner_pid}" && "${old_reasoner_pid}" != "${new_reasoner_pid}" ]]; then
  echo "FAIL (verification 5): cosmos_reasoner PID changed" \
       "(${old_reasoner_pid} → ${new_reasoner_pid})." >&2
  exit 1
fi
echo "PASS (verification 5): cosmos_reasoner PID unchanged (${new_reasoner_pid})."

# ── Shutdown: verify no orphan worker or socket (verification 6) ─────────────
cleanup
trap - EXIT INT TERM

# After shutdown, no worker process or socket file should remain.
orphan_pids="$(worker_pid)"
if [[ -n "${orphan_pids}" ]]; then
  echo "FAIL (verification 6): orphan worker process(es) remain: ${orphan_pids}" >&2
  exit 1
fi
echo "PASS (verification 6, worker): no orphan worker process after shutdown."

if [[ -e "${worker_socket}" ]]; then
  echo "FAIL (verification 6, socket): socket file ${worker_socket} still exists." >&2
  exit 1
fi
echo "PASS (verification 6, socket): socket file removed after shutdown."

echo ""
echo "PASS: all 6 verifications passed — watchdog fires, worker PID changes, exactly"
echo "      one failure published, reasoning resumes, cosmos_reasoner PID unchanged,"
echo "      and no orphan worker or socket remains after shutdown."

