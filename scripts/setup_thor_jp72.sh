#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
install_script="${EDGE_VLM_INSTALL_DEPENDENCIES_SCRIPT:-${script_dir}/install_dependencies.sh}"
prepare_script="${EDGE_VLM_PREPARE_THOR_ASSETS_SCRIPT:-${script_dir}/prepare_thor_jp72_assets.sh}"
build_script="${EDGE_VLM_BUILD_WORKSPACE_SCRIPT:-${script_dir}/build_workspace.sh}"
verify_script="${EDGE_VLM_VERIFY_THOR_SCRIPT:-${script_dir}/verify_thor_jp72.sh}"
download_rosbags=1
install_args=(--isaac-ros)
prepare_args=()
dry_run=0

usage() {
  cat <<'EOF'
Usage: ./scripts/setup_thor_jp72.sh [--desktop] [--force] [--dry-run] [--skip-rosbag-download]

Fresh-machine bootstrap for Jetson AGX Thor on JetPack 7.2.x / R39.2.x.
Runs dependency installation, environment bootstrap, optional test-data download,
workspace build, and deployment verification.

When --dry-run is set, this wrapper performs non-mutating planning only across
dependency and asset setup phases, then exits before source/build/verify.
EOF
}

for arg in "$@"; do
  case "$arg" in
    --desktop|--force) install_args+=("$arg") ;;
    --skip-rosbag-download) download_rosbags=0 ;;
    --dry-run)
      dry_run=1
      install_args+=(--dry-run)
      prepare_args+=(--dry-run)
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; usage >&2; exit 2 ;;
  esac
done

ensure_docker_session_ready() {
  command -v docker >/dev/null 2>&1 || {
    echo "ERROR: docker is required for Thor asset preparation, but it is not installed." >&2
    exit 1
  }

  docker info >/dev/null 2>&1 && return 0

  local current_shell_has_docker_group=0
  if id -nG | tr ' ' '\n' | grep -Fxq docker; then
    current_shell_has_docker_group=1
  fi

  local configured_for_user=0
  if getent group docker >/dev/null 2>&1; then
    if getent group docker | awk -F: '{print $4}' | tr ',' '\n' | grep -Fxq "${USER}"; then
      configured_for_user=1
    fi
  fi

  if [[ "${configured_for_user}" -eq 1 && "${current_shell_has_docker_group}" -eq 0 ]]; then
    cat >&2 <<EOF
ERROR: Docker group membership was configured for ${USER}, but this shell has not picked it up yet.
Log out and back in, then rerun this same command before continuing:
  bash scripts/setup_thor_jp72.sh
EOF
    exit 1
  fi

  cat >&2 <<'EOF'
ERROR: Unable to run Docker as the current user (`docker info` failed).
Ensure Docker is running and that this shell has active docker-group membership.
EOF
  exit 1
}

bash "${install_script}" "${install_args[@]}"

if [[ "${dry_run}" -ne 1 ]]; then
  ensure_docker_session_ready
fi

if [[ "${download_rosbags}" -eq 0 ]]; then
  prepare_args+=(--skip-data)
fi

bash "${prepare_script}" "${prepare_args[@]}"

if [[ "${dry_run}" -eq 1 ]]; then
  echo "Dry-run mode requested; skipping environment source, build, and verification."
  exit 0
fi

# shellcheck disable=SC1090
source "${script_dir}/edge_vlm_env.sh"

bash "${build_script}"
bash "${verify_script}" --isaac-ros
