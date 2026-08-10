#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
download_rosbags=1
install_args=(--isaac-ros)
prepare_args=()
dry_run=0

usage() {
  cat <<'EOF'
Usage: ./scripts/setup_thor_jp71.sh [--desktop] [--force] [--dry-run] [--skip-rosbag-download]

Fresh-machine bootstrap for Jetson AGX Thor on JetPack 7.1 / R38.4.
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

bash "${script_dir}/install_dependencies.sh" "${install_args[@]}"

if [[ "${download_rosbags}" -eq 0 ]]; then
  prepare_args+=(--skip-data)
fi

bash "${script_dir}/prepare_thor_jp71_assets.sh" "${prepare_args[@]}"

if [[ "${dry_run}" -eq 1 ]]; then
  echo "Dry-run mode requested; skipping environment source, build, and verification."
  exit 0
fi

# shellcheck disable=SC1090
source "${script_dir}/edge_vlm_env.sh"

bash "${script_dir}/build_workspace.sh"
bash "${script_dir}/verify_thor_jp71.sh" --isaac-ros
