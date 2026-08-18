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
