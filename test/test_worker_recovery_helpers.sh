#!/usr/bin/env bash
# Copyright 2025 cosmos_ros2_video_reasoner contributors
#
# Shell-level unit tests for the PID-match helpers and PGID isolation safety
# checks used in scripts/test_data/run_worker_recovery_test.sh.
#
# Tests run without ROS, hardware, or any network access.  They exercise the
# exact logic that is copy-pasted into the recovery test script so that logic
# regressions are caught by the hardware-independent CI job.
set -Eeuo pipefail

pass=0
fail=0

check() {
  local desc="$1" expected="$2" actual="$3"
  if [[ "${expected}" == "${actual}" ]]; then
    printf 'PASS: %s\n' "${desc}"
    pass=$(( pass + 1 ))
  else
    printf 'FAIL: %s\n' "${desc}" >&2
    printf '  expected: "%s"\n' "${expected}" >&2
    printf '  actual:   "%s"\n' "${actual}" >&2
    fail=$(( fail + 1 ))
  fi
}

# ── Helper under test ─────────────────────────────────────────────────────────
# Mirrors the exact require-one logic from run_worker_recovery_test.sh.
# Both worker_pid() and reasoner_pid() rely on this pattern.

require_one_pid() {
  local pids="$1"
  local count=0
  [[ -n "${pids}" ]] && count="$(printf '%s\n' "${pids}" | wc -l | tr -d ' ')"
  [[ "${count}" -eq 1 ]] && printf '%s\n' "${pids}" || true
}

# ── Tests: require_one_pid ────────────────────────────────────────────────────

check "require_one_pid: empty input returns empty" \
  "" \
  "$(require_one_pid "")"

check "require_one_pid: single PID is returned unchanged" \
  "12345" \
  "$(require_one_pid "12345")"

check "require_one_pid: two PIDs return empty (ambiguous)" \
  "" \
  "$(require_one_pid "$(printf '%s\n' 12345 67890)")"

check "require_one_pid: three PIDs return empty (ambiguous)" \
  "" \
  "$(require_one_pid "$(printf '%s\n' 1 2 3)")"

# ── Tests: PGID safety-guard logic ───────────────────────────────────────────
# Mirrors cleanup()'s safety check: only signal with -PGID when target PGID is
# non-empty, non-zero, and differs from the test script's own PGID.

pgid_safety_would_signal() {
  local target_pgid="$1"
  local self_pgid="$2"
  if [[ -n "${target_pgid}" ]] && [[ "${target_pgid}" != "0" ]] && \
     [[ -n "${self_pgid}" ]] && [[ "${target_pgid}" != "${self_pgid}" ]]; then
    printf 'yes'
  else
    printf 'no'
  fi
}

check "PGID safety: signals when target PGID differs from own PGID" \
  "yes" \
  "$(pgid_safety_would_signal "9999" "1234")"

check "PGID safety: does not signal when target PGID equals own PGID" \
  "no" \
  "$(pgid_safety_would_signal "1234" "1234")"

check "PGID safety: does not signal when target PGID is 0" \
  "no" \
  "$(pgid_safety_would_signal "0" "1234")"

check "PGID safety: does not signal when target PGID is empty" \
  "no" \
  "$(pgid_safety_would_signal "" "1234")"

check "PGID safety: does not signal when own PGID is empty (indeterminate)" \
  "no" \
  "$(pgid_safety_would_signal "9999" "")"

# ── Tests: setsid PGID isolation ─────────────────────────────────────────────
# setsid starts the child in a new session; its PGID equals its own PID, which
# differs from the test script's PGID.

own_pgid="$(ps -o pgid= -p "$$" 2>/dev/null | tr -d ' ')"

setsid sleep 30 &
setsid_pid=$!
# Give the kernel a moment to assign the new session before querying.
sleep 0.1
child_pgid="$(ps -o pgid= -p "${setsid_pid}" 2>/dev/null | tr -d ' ')" || true
kill "${setsid_pid}" 2>/dev/null || true
wait "${setsid_pid}" 2>/dev/null || true

check "setsid: child PGID is non-empty" \
  "true" \
  "$( [[ -n "${child_pgid}" ]] && printf 'true' || printf 'false' )"

check "setsid: child PGID differs from test script PGID" \
  "true" \
  "$( [[ "${child_pgid}" != "${own_pgid}" ]] && printf 'true' || printf 'false' )"

check "setsid: pgid_safety_would_signal is 'yes' for the setsid child" \
  "yes" \
  "$(pgid_safety_would_signal "${child_pgid}" "${own_pgid}")"

# ── Summary ───────────────────────────────────────────────────────────────────
printf '\nResults: %d passed, %d failed.\n' "${pass}" "${fail}"
[[ "${fail}" -eq 0 ]]
