#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
download_rosbags=1
install_args=(--isaac-ros)

usage() {
  cat <<'EOF'
Usage: ./scripts/setup_thor_jp71.sh [--desktop] [--force] [--skip-rosbag-download]

Fresh-machine bootstrap for Jetson AGX Thor on JetPack 7.1 / R38.4.
Runs dependency installation, environment bootstrap, optional test-data download,
workspace build, and deployment verification.
EOF
}

for arg in "$@"; do
  case "$arg" in
    --desktop|--force) install_args+=("$arg") ;;
    --skip-rosbag-download) download_rosbags=0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; usage >&2; exit 2 ;;
  esac
done

bash "${script_dir}/install_dependencies.sh" "${install_args[@]}"

if [[ ! -f "${script_dir}/edge_vlm_env.sh" ]]; then
  cp "${script_dir}/edge_vlm_env.sh.example" "${script_dir}/edge_vlm_env.sh"
  echo
  echo "Created ${script_dir}/edge_vlm_env.sh"
  echo "Set local TensorRT Edge-LLM/model paths in that file, then rerun:"
  echo "  ${script_dir}/setup_thor_jp71.sh"
  exit 0
fi

# shellcheck disable=SC1090
source "${script_dir}/edge_vlm_env.sh"

if [[ "${download_rosbags}" -eq 1 ]]; then
  bash "${script_dir}/test_data/download_rosbags.sh" download image-proc
fi

bash "${script_dir}/build_workspace.sh"
bash "${script_dir}/verify_thor_jp71.sh" --isaac-ros
