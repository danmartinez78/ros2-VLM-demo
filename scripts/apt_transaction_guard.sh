#!/usr/bin/env bash

PROTECTED_NVIDIA_PACKAGES=(
  nvidia-jetpack
  nvidia-jetpack-dev
  nvidia-opencv-dev
)

apt_guard_fail() {
  if declare -F fail >/dev/null 2>&1; then
    fail "$*"
  fi
  printf 'ERROR: %s\n' "$*" >&2
  return 1
}

simulate_apt_install_output() {
  if [[ -n "${EDGE_VLM_APT_SIMULATION_OUTPUT:-}" ]]; then
    printf '%s\n' "${EDGE_VLM_APT_SIMULATION_OUTPUT}"
    return "${EDGE_VLM_APT_SIMULATION_EXIT_CODE:-0}"
  fi
  sudo apt-get -s install -y "$@"
}

assert_safe_apt_transaction() {
  local description="$1"
  shift
  local simulation_output

  if ! simulation_output="$(simulate_apt_install_output "$@" 2>&1)"; then
    apt_guard_fail "Unable to simulate APT transaction for ${description}. Output:"$'\n'"${simulation_output}"
    return 1
  fi

  local protected_pkg
  local removed_pkg
  local -a removed_pkgs=()
  mapfile -t removed_pkgs < <(printf '%s\n' "${simulation_output}" | awk '/^Remv[[:space:]]/{print $2}')
  for protected_pkg in "${PROTECTED_NVIDIA_PACKAGES[@]}"; do
    for removed_pkg in "${removed_pkgs[@]}"; do
      if [[ "${removed_pkg}" == "${protected_pkg}" || "${removed_pkg}" == "${protected_pkg}:"* ]]; then
        apt_guard_fail "Refusing to continue: planned APT transaction for ${description} removes protected package '${protected_pkg}'. Keep the NVIDIA Jetson stack intact."
        return 1
      fi
    done
  done
}

libopencv_policy_output() {
  if [[ -n "${EDGE_VLM_APT_POLICY_LIBOPENCV_DEV_OUTPUT:-}" ]]; then
    printf '%s\n' "${EDGE_VLM_APT_POLICY_LIBOPENCV_DEV_OUTPUT}"
    return 0
  fi
  apt-cache policy libopencv-dev
}

assert_libopencv_candidate_matches_installed() {
  local policy_output
  local installed_version
  local candidate_version

  policy_output="$(libopencv_policy_output 2>&1)"

  installed_version="$(printf '%s\n' "${policy_output}" | awk '/^[[:space:]]*Installed:/{print $2; exit}')"
  candidate_version="$(printf '%s\n' "${policy_output}" | awk '/^[[:space:]]*Candidate:/{print $2; exit}')"

  [[ -n "${installed_version}" && "${installed_version}" != "(none)" ]] || {
    apt_guard_fail "libopencv-dev is not installed on the host."
    return 1
  }
  [[ -n "${candidate_version}" && "${candidate_version}" != "(none)" ]] || {
    apt_guard_fail "No install candidate is available for libopencv-dev."
    return 1
  }

  if [[ "${installed_version}" != "${candidate_version}" ]]; then
    apt_guard_fail \
      "Candidate version for libopencv-dev (${candidate_version}) differs from installed (${installed_version}). Refusing to continue because this indicates host OpenCV downgrade/replacement pressure."
    return 1
  fi

  return 0
}
