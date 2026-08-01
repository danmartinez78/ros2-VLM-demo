#!/usr/bin/env bash
# Thor hardware validation: watchdog-triggered recovery test.
#
# Injects a deterministic one-shot hang using the
# COSMOS_TEST_INJECT_HANG_ONCE_SENTINEL environment variable:
#   - The first inference worker creates a sentinel file and then sleeps for
#     worker_inference_deadline_seconds + 30 s.  The InferenceWatchdog fires
#     at the configured deadline and calls std::_Exit(1).
#   - The respawned worker finds the sentinel present and runs normally
#     (normal inference, no injected hang), so a successful result arrives.
#
# The normal production deadline (worker_inference_deadline_seconds=60) is
# preserved.  The client timeout (worker_request_timeout_seconds=90) exceeds
# the deadline so the client observes a clean EOF rather than SO_RCVTIMEO.
#
# This test is HARDWARE-ONLY; COSMOS_TEST_INJECT_HANG_ONCE_SENTINEL must
# never be set in production deployments.
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
# Normal production deadline; long enough for real Cosmos inference on Thor.
watchdog_deadline="${WORKER_INFERENCE_DEADLINE_SECONDS:-60}"
# Client timeout must exceed the watchdog deadline to observe a clean EOF
# (client sees the worker exit) rather than a SO_RCVTIMEO socket error.
client_timeout="${WORKER_REQUEST_TIMEOUT_SECONDS:-90}"

launch_pid=""
bag_pid=""
launch_log="$(mktemp /tmp/cosmos-recovery-launch.XXXXXX.log)"

# One-shot injected hang: the first inference request creates this file and
# then sleeps past the watchdog deadline; the watchdog fires and calls
# std::_Exit(1).  The respawned worker finds the sentinel present and skips
# the hang, allowing normal inference to proceed.
# Never set in production — this variable is only exported by this script.
test_sentinel="$(mktemp -u /tmp/cosmos-test-sentinel-XXXXXX.flag)"
export COSMOS_TEST_INJECT_HANG_ONCE_SENTINEL="${test_sentinel}"

cleanup() {
  if [[ -n "${bag_pid}" ]] && kill -0 "${bag_pid}" 2>/dev/null; then
    kill -INT "${bag_pid}" 2>/dev/null || true
    wait "${bag_pid}" 2>/dev/null || true
  fi
  if [[ -n "${launch_pid}" ]]; then
    # Kill the entire process group so ros2 launch and all child workers
    # (inference worker, etc.) shut down cleanly together.
    # Safety guard: only send a negative-PGID signal when the target PGID is
    # confirmed to differ from the test script's own PGID — prevents
    # accidentally terminating the calling shell and its parents.
    local pgid own_pgid
    pgid="$(ps -o pgid= -p "${launch_pid}" 2>/dev/null | tr -d ' ')" || true
    own_pgid="$(ps -o pgid= -p "$$" 2>/dev/null | tr -d ' ')" || true
    if [[ -n "${pgid}" ]] && [[ "${pgid}" != "0" ]] && \
       [[ -n "${own_pgid}" ]] && [[ "${pgid}" != "${own_pgid}" ]]; then
      kill -INT -- "-${pgid}" 2>/dev/null || true
      sleep 3
      kill -TERM -- "-${pgid}" 2>/dev/null || true
    else
      # PGID matches own group or cannot be determined — signal only the
      # launch process directly rather than the shared group.
      kill -INT "${launch_pid}" 2>/dev/null || true
    fi
    wait "${launch_pid}" 2>/dev/null || true
  fi
  rm -f "${launch_log}" "${test_sentinel}"
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
worker_pid() {
  # Escape ERE metacharacters in the socket path (primarily '.' in filenames)
  # so that pgrep -f treats it as a literal component, not a regex wildcard.
  local escaped_socket
  escaped_socket="$(printf '%s' "${worker_socket}" | sed 's/[.[\*^$()|+?{}]/\\&/g')"
  local pids
  pids="$(pgrep -f "cosmos_inference_worker.*${escaped_socket}" 2>/dev/null)" || true
  # Require exactly one match — guards against misreporting 0 (not yet
  # started) or 2+ PIDs (brief overlap during respawn) as a valid single PID.
  local count=0
  [[ -n "${pids}" ]] && count="$(printf '%s\n' "${pids}" | wc -l | tr -d ' ')"
  [[ "${count}" -eq 1 ]] && printf '%s\n' "${pids}" || true
}

reasoner_pid() {
  # Match the installed executable path ending in '/cosmos_reasoner' followed
  # by end-of-command or a space — avoids matching cosmos_reasoner_node or
  # other commands that merely contain the substring "cosmos_reasoner".
  local pids
  pids="$(pgrep -f '/cosmos_reasoner($| )' 2>/dev/null)" || true
  # Require exactly one match for this test instance.
  local count=0
  [[ -n "${pids}" ]] && count="$(printf '%s\n' "${pids}" | wc -l | tr -d ' ')"
  [[ "${count}" -eq 1 ]] && printf '%s\n' "${pids}" || true
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
  # Overall bound: fail fast if no success arrives within 5 minutes.
  local deadline=$((SECONDS + 300))
  local output=""
  while (( SECONDS < deadline )); do
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
  echo "count_failure_results: overall 5-minute deadline exceeded." >&2
  return 1
}

# ── Launch ───────────────────────────────────────────────────────────────────
echo "Starting watchdog-triggered recovery test..."
echo "  worker_inference_deadline_seconds = ${watchdog_deadline}"
echo "  worker_request_timeout_seconds    = ${client_timeout}"
echo "  worker_socket_path                = ${worker_socket}"
echo "  COSMOS_TEST_INJECT_HANG_ONCE_SENTINEL = ${test_sentinel}"

# Start ros2 launch in its own session so setsid gives it a new PGID that
# differs from the test script's PGID.  This makes the negative-PGID kill in
# cleanup() safe: it can never reach the test script's own process group.
setsid ros2 launch cosmos_ros2_video_reasoner cosmos_reasoner.launch.py \
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
  use_sim_time:=true > "${launch_log}" 2>&1 &
launch_pid=$!

# ── Wait for initial worker and reasoner ─────────────────────────────────────
old_worker_pid="$(wait_for_worker)" || {
  echo "Timed out waiting for the initial inference worker." >&2
  exit 1
}
echo "Initial worker PID:   ${old_worker_pid}"

# Record cosmos_reasoner PID for verification 5.
# Require exactly one nonempty PID — an empty result here means the reasoner
# has not started yet or there are multiple matches.  Do not proceed: a false
# empty would silently bypass the "reasoner PID unchanged" assertion.
old_reasoner_pid="$(reasoner_pid)" || true
if [[ -z "${old_reasoner_pid}" ]]; then
  echo "FAIL: could not find exactly one cosmos_reasoner process." \
       "Ensure no other cosmos_reasoner instance is running." >&2
  exit 1
fi
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
# The sentinel mechanism causes only the first worker to hang past its deadline.
# The watchdog fires at ${watchdog_deadline} s, calls std::_Exit(1), and the OS
# closes all file descriptors.  The client observes a clean EOF (well within
# the ${client_timeout} s client timeout) and reports exactly one failure.
echo "Waiting for watchdog expiry and worker self-termination..."
echo "(This will take approximately ${watchdog_deadline} s while the injected hang runs.)"

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
if [[ -z "${new_reasoner_pid}" ]]; then
  echo "FAIL (verification 5): cosmos_reasoner is no longer running after recovery." >&2
  exit 1
fi
if [[ "${old_reasoner_pid}" != "${new_reasoner_pid}" ]]; then
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

