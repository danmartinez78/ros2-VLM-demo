#!/usr/bin/env bash
set -Eeuo pipefail

ISAAC_ROS_MAJOR=4
ISAAC_ROS_MINOR=5
NGC_ORG="nvidia"
NGC_TEAM="isaac"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
output_root="${ROSBAG_DIR:-${repo_root}/test_data/rosbags}"

usage() {
  cat <<'EOF'
Usage: download_rosbags.sh list
       download_rosbags.sh download <image-proc|h264|nvblox|rtdetr>
       download_rosbags.sh all

Downloads version-matched NVIDIA Isaac ROS 4.5 quickstart assets from NGC.
Set ROSBAG_DIR to override the default test_data/rosbags directory.
EOF
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    echo "Install prerequisites with: sudo apt-get install -y curl jq tar" >&2
    exit 1
  }
}

asset_resource() {
  case "$1" in
    image-proc) echo "isaac_ros_image_proc_assets" ;;
    h264) echo "isaac_ros_h264_decoder_assets" ;;
    nvblox) echo "isaac_ros_nvblox_assets" ;;
    rtdetr) echo "isaac_ros_rtdetr_assets" ;;
    *) return 1 ;;
  esac
}

asset_description() {
  case "$1" in
    image-proc) echo "Raw RGB + camera info; directly usable by the Cosmos node" ;;
    h264) echo "Dual H.264 CompressedImage streams; requires the Isaac ROS decoder" ;;
    nvblox) echo "RGB-D bag used by the Isaac ROS nvblox quickstart and tracker demos" ;;
    rtdetr) echo "RT-DETR quickstart bag (expects isaac_ros_rtdetr/quickstart.bag)" ;;
    *) return 1 ;;
  esac
}

latest_compatible_version() {
  local resource="$1"
  local versions_url="https://api.ngc.nvidia.com/v2/resources/${NGC_ORG}/${NGC_TEAM}/${resource}/versions"
  local response

  response="$(curl -fsSL --retry 3 --retry-delay 2     -H "Accept: application/json" "${versions_url}")"

  jq -r --argjson major "${ISAAC_ROS_MAJOR}" --argjson minor "${ISAAC_ROS_MINOR}" '
    .recipeVersions[].versionId
    | select(test("^\\d+\\.\\d+\\.\\d+$"))
    | . as $version
    | split(".") | map(tonumber)
    | select(.[0] == $major and .[1] <= $minor)
    | $version
  ' <<<"${response}" | sort -V | tail -n 1
}

download_asset() {
  local key="$1"
  local resource version target archive staging download_url

  resource="$(asset_resource "${key}")" || {
    echo "Unknown dataset: ${key}" >&2
    usage >&2
    exit 2
  }
  version="$(latest_compatible_version "${resource}")"
  if [[ -z "${version}" ]]; then
    echo "No Isaac ROS ${ISAAC_ROS_MAJOR}.${ISAAC_ROS_MINOR}-compatible version found for ${resource}." >&2
    echo "NGC versions returned:" >&2
    curl -fsSL --retry 3 --retry-delay 2 \
      -H "Accept: application/json" \
      "https://api.ngc.nvidia.com/v2/resources/${NGC_ORG}/${NGC_TEAM}/${resource}/versions" \
      | jq -r '.recipeVersions[].versionId' >&2 || true
    exit 1
  fi

  target="${output_root}/${key}"
  if [[ -f "${target}/.ngc-version" ]] &&
     [[ "$(cat "${target}/.ngc-version")" == "${version}" ]]; then
    echo "${key}: already installed (NGC version ${version}) at ${target}"
    return
  fi
  if [[ -e "${target}" ]]; then
    echo "${target} already exists but does not match NGC version ${version}." >&2
    echo "Move or remove that directory, then rerun the download." >&2
    exit 1
  fi

  mkdir -p "${output_root}"
  archive="${output_root}/.${key}-${version}.tar.gz.part"
  staging="${output_root}/.${key}-${version}.staging"
  download_url="https://api.ngc.nvidia.com/v2/resources/${NGC_ORG}/${NGC_TEAM}/${resource}/versions/${version}/files/quickstart.tar.gz"

  if [[ -e "${staging}" ]]; then
    echo "Stale staging directory exists: ${staging}" >&2
    echo "Move or remove it before retrying." >&2
    exit 1
  fi

  trap 'if [[ -n "${staging:-}" && -d "${staging}" ]]; then rm -rf -- "${staging}"; fi' ERR
  echo "Downloading ${key} (NGC ${resource} ${version})..."
  curl -fL --retry 3 --retry-delay 2 -C - -o "${archive}" "${download_url}"
  tar -tzf "${archive}" >/dev/null
  mkdir "${staging}"
  tar -xzf "${archive}" -C "${staging}"
  printf '%s\n' "${version}" >"${staging}/.ngc-version"
  mv "${staging}" "${target}"
  trap - ERR
  rm -f -- "${archive}"

  echo "${key}: installed at ${target}"
  find "${target}" -name metadata.yaml -printf '  bag: %h\n' | sort
}

main() {
  require_command curl
  require_command jq
  require_command tar

  case "${1:-}" in
    list)
      printf '%-12s %s\n' "DATASET" "DESCRIPTION"
      for key in image-proc h264 nvblox rtdetr; do
        printf '%-12s %s\n' "${key}" "$(asset_description "${key}")"
      done
      ;;
    download)
      [[ -n "${2:-}" ]] || { usage >&2; exit 2; }
      download_asset "$2"
      ;;
    all)
      download_asset image-proc
      download_asset h264
      download_asset nvblox
      download_asset rtdetr
      ;;
    -h|--help|"")
      usage
      ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
}

main "$@"
