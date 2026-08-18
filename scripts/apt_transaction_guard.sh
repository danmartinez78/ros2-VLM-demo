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
  if [[ -n "${EDGE_VLM_APT_SIMULATION_OUTPUT_COMMAND:-}" ]]; then
    if [[ "${EDGE_VLM_ALLOW_APT_SIMULATION_COMMAND:-0}" != "1" && \
          "${EDGE_VLM_APT_GUARD_TEST_MODE:-0}" != "1" && \
          "${EDGE_VLM_ISAAC_PREF_GUARD_TEST_MODE:-0}" != "1" ]]; then
      apt_guard_fail "EDGE_VLM_APT_SIMULATION_OUTPUT_COMMAND is test-only. Set EDGE_VLM_ALLOW_APT_SIMULATION_COMMAND=1 to override deliberately."
      return 1
    fi
    "${EDGE_VLM_APT_SIMULATION_OUTPUT_COMMAND}" "$@"
    return
  fi
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

package_policy_output() {
  local package_name="$1"
  local override_var
  local override_var_suffix
  local override_value=""

  override_var_suffix="$(printf '%s' "${package_name}" | tr '[:lower:]-.+/' '[:upper:]_____')"
  override_var="EDGE_VLM_APT_POLICY_${override_var_suffix}_OUTPUT"
  override_value="${!override_var:-}"

  if [[ -n "${override_value}" ]]; then
    printf '%s\n' "${override_value}"
    return 0
  fi

  apt-cache policy "${package_name}"
}

assert_package_candidate_matches_installed() {
  local package_name="$1"
  local package_label="${2:-${package_name}}"
  local policy_output
  local installed_version
  local candidate_version

  policy_output="$(package_policy_output "${package_name}" 2>&1)"

  installed_version="$(printf '%s\n' "${policy_output}" | awk '/^[[:space:]]*Installed:/{print $2; exit}')"
  candidate_version="$(printf '%s\n' "${policy_output}" | awk '/^[[:space:]]*Candidate:/{print $2; exit}')"

  [[ -n "${installed_version}" && "${installed_version}" != "(none)" ]] || {
    apt_guard_fail "${package_label} (${package_name}) is not installed on the host."
    return 1
  }
  [[ -n "${candidate_version}" && "${candidate_version}" != "(none)" ]] || {
    apt_guard_fail "No install candidate is available for ${package_label} (${package_name})."
    return 1
  }

  if [[ "${installed_version}" != "${candidate_version}" ]]; then
    apt_guard_fail \
      "Candidate version for ${package_label} (${package_name}) (${candidate_version}) differs from installed (${installed_version}). Refusing to continue because this indicates host stack downgrade/replacement pressure."
    return 1
  fi

  return 0
}

assert_libopencv_candidate_matches_installed() {
  assert_package_candidate_matches_installed "libopencv-dev" "OpenCV development package"
}

resolve_nvcc_owner_package() {
  local nvcc_bin=""
  local nvcc_path
  local owner_pkg

  if command -v nvcc >/dev/null 2>&1; then
    nvcc_bin="$(command -v nvcc)"
  elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
    nvcc_bin="/usr/local/cuda/bin/nvcc"
  else
    return 1
  fi

  nvcc_path="$(readlink -f -- "${nvcc_bin}")"
  owner_pkg="$(dpkg-query -S "${nvcc_path}" 2>/dev/null | awk -F: 'NR==1{print $1}')"
  [[ -n "${owner_pkg}" ]] || return 1
  printf '%s\n' "${owner_pkg}"
}

simulate_rosdep_install_output() {
  local from_paths="$1"
  local ros_distro="$2"

  if [[ -n "${EDGE_VLM_ROSDEP_SIMULATION_OUTPUT:-}" ]]; then
    printf '%s\n' "${EDGE_VLM_ROSDEP_SIMULATION_OUTPUT}"
    return "${EDGE_VLM_ROSDEP_SIMULATION_EXIT_CODE:-0}"
  fi

  rosdep install \
    --simulate \
    --from-paths "${from_paths}" \
    --ignore-src \
    --rosdistro "${ros_distro}" \
    -r -y
}

assert_safe_rosdep_install_plan() {
  local from_paths="$1"
  local ros_distro="$2"
  local simulation_output
  local simulation_stderr=""
  local stderr_file
  local -a tokens=()
  local -a packages=()
  local token
  local in_install_args
  stderr_file="$(mktemp)"

  if ! simulation_output="$(simulate_rosdep_install_output "${from_paths}" "${ros_distro}" 2>"${stderr_file}")"; then
    simulation_stderr="$(cat -- "${stderr_file}" 2>/dev/null || true)"
    rm -f -- "${stderr_file}"
    apt_guard_fail "Unable to simulate rosdep install plan. Stdout:"$'\n'"${simulation_output}"$'\n'"Stderr:"$'\n'"${simulation_stderr}"
    return 1
  fi
  rm -f -- "${stderr_file}"

  while IFS= read -r line; do
    [[ "${line}" == *"apt-get install"* ]] || continue

    tokens=()
    packages=()
    in_install_args=0
    read -r -a tokens <<<"${line}"

    for token in "${tokens[@]}"; do
      if [[ "${in_install_args}" -eq 0 ]]; then
        [[ "${token}" == "install" ]] && in_install_args=1
        continue
      fi

      [[ -z "${token}" || "${token}" == -* ]] && continue
      token="${token//;/}"
      token="${token//,/}"
      [[ -n "${token}" ]] && packages+=("${token}")
    done

    if [[ "${#packages[@]}" -gt 0 ]]; then
      assert_safe_apt_transaction "rosdep-managed packages" "${packages[@]}" || return 1
    fi
  done <<<"${simulation_output}"

  return 0
}
