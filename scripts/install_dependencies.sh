#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

ROS_DISTRO="jazzy"
INSTALL_DESKTOP=0
INSTALL_ISAAC_ROS=0
FORCE_UNSUPPORTED=0
DRY_RUN=0
APT_PREFERENCES_DIR="${EDGE_VLM_APT_PREFERENCES_DIR:-/etc/apt/preferences.d}"

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

source "${script_dir}/apt_transaction_guard.sh"

collect_isaac_ros_pref_files() {
  shopt -s nullglob
  local pref_files=("${APT_PREFERENCES_DIR}"/isaac-ros-*.pref)
  shopt -u nullglob
  printf '%s\n' "${pref_files[@]}"
}

assert_isaac_ros_preferences_compatible() {
  local stage="$1"
  local pref_file
  local pref_files=()
  while IFS= read -r pref_file; do
    [[ -n "${pref_file}" ]] || continue
    pref_files+=("${pref_file}")
  done < <(collect_isaac_ros_pref_files)

  if [[ "${#pref_files[@]}" -gt 0 ]]; then
    echo "Detected Isaac ROS host APT preference files during ${stage}:"
    printf '  %s\n' "${pref_files[@]}"
  fi

  if ! assert_host_jetpack_stack_safe; then
    fail \
      "APT candidate simulation indicates incompatible host package pinning during ${stage}. Review Isaac ROS host preferences under ${APT_PREFERENCES_DIR} and ensure they do not downgrade or remove protected JetPack packages (${PROTECTED_NVIDIA_PACKAGES[*]})."
  fi
}

assert_host_jetpack_stack_safe() {
  local cuda_owner_pkg="${EDGE_VLM_CUDA_PACKAGE_FOR_TEST:-}"

  assert_safe_apt_transaction "host OpenCV safety check" libopencv-dev
  assert_package_candidate_matches_installed libopencv-dev "OpenCV development package"

  assert_safe_apt_transaction "host TensorRT safety check" libnvinfer-dev
  assert_package_candidate_matches_installed libnvinfer-dev "TensorRT development package"

  if [[ -z "${cuda_owner_pkg}" ]]; then
    cuda_owner_pkg="$(resolve_nvcc_owner_package || true)"
  fi
  [[ -n "${cuda_owner_pkg}" ]] || fail \
    "Unable to resolve the installed CUDA package owning nvcc. Ensure the JP7.2 CUDA toolkit is installed before enabling Isaac ROS Docker mode."

  assert_safe_apt_transaction "host CUDA safety check (${cuda_owner_pkg})" "${cuda_owner_pkg}"
  assert_package_candidate_matches_installed "${cuda_owner_pkg}" "CUDA compiler package"
}

usage() {
  cat <<'EOF'
Usage: sudo -v && ./scripts/install_dependencies.sh [--desktop] [--isaac-ros] [--force] [--dry-run]

Installs the JetPack 7.2 development stack, ROS 2 Jazzy, rosdep/colcon,
OpenCV, rosbag2, and all system packages needed by this repository.

Options:
  --desktop   Install ros-jazzy-desktop instead of the headless ros-base variant.
  --isaac-ros Configure NVIDIA Isaac ROS 4.6 using its recommended Docker mode.
              This repository's supported Thor path targets JetPack 7.2 / R39.2.
  --force    Continue on an unsupported OS, architecture, or Jetson Linux release.
  --dry-run  Print planned installation actions without mutating the system.
EOF
}

for arg in "$@"; do
  case "$arg" in
    --desktop) INSTALL_DESKTOP=1 ;;
    --isaac-ros) INSTALL_ISAAC_ROS=1 ;;
    --force) FORCE_UNSUPPORTED=1 ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "${EUID}" -eq 0 ]]; then
  echo "Run this script as your normal user. It invokes sudo only where needed." >&2
  exit 2
fi

source /etc/os-release
machine_arch="$(uname -m)"
l4t_release="$(sed -n '1p' /etc/nv_tegra_release 2>/dev/null || true)"

unsupported=0
[[ "${ID}" == "ubuntu" && "${VERSION_ID}" == "24.04" ]] || unsupported=1
[[ "${machine_arch}" == "aarch64" ]] || unsupported=1
[[ "${l4t_release}" == *"# R39 (release), REVISION: 2.0"* ]] || unsupported=1

if [[ "${unsupported}" -ne 0 && "${FORCE_UNSUPPORTED}" -ne 1 ]]; then
  cat >&2 <<EOF
Unsupported deployment target.
Expected: Ubuntu 24.04, aarch64, Jetson Linux R39.2 (JetPack 7.2)
Detected: ${PRETTY_NAME:-unknown}, ${machine_arch}, ${l4t_release:-no /etc/nv_tegra_release}
Use --force only if you understand that this repository targets JetPack 7.2.
EOF
  exit 1
fi

ros_variant="ros-${ROS_DISTRO}-ros-base"
if [[ "${INSTALL_DESKTOP}" -eq 1 ]]; then
  ros_variant="ros-${ROS_DISTRO}-desktop"
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
  cat <<EOF
DRY-RUN  install_dependencies plan:
  Supported target baseline: Ubuntu 24.04, aarch64, Jetson Linux R39.2
  Selected ROS variant: ${ros_variant}
  Isaac ROS setup requested: ${INSTALL_ISAAC_ROS}
  Planned actions:
    - sudo apt-get update
    - simulate APT transactions and block removal of protected NVIDIA packages
      (${PROTECTED_NVIDIA_PACKAGES[*]})
    - install baseline packages + nvidia-jetpack
    - install ROS apt source + ${ros_variant}
    - install build/rosdep dependencies
    - optionally configure Isaac ROS Docker mode (if --isaac-ros)
      - before and after "isaac-ros init docker", verify host CUDA/TensorRT/OpenCV package candidates
        remain aligned with installed JP7.2 packages and protected NVIDIA metapackages
      - if Isaac ROS host preference files are present, fail safely when candidate simulation indicates
        downgrade/removal pressure instead of mutating supported JP7.2 pin files
    - initialize/update rosdep
EOF
  exit 0
fi

if [[ "${EDGE_VLM_APT_GUARD_TEST_MODE:-0}" == "1" ]]; then
  [[ -n "${EDGE_VLM_APT_SIMULATION_OUTPUT:-}" ]] || fail \
    "EDGE_VLM_APT_GUARD_TEST_MODE requires EDGE_VLM_APT_SIMULATION_OUTPUT."
  assert_safe_apt_transaction "guard test transaction" guard-test-package
  echo "APT guard test transaction passed."
  exit 0
fi

if [[ "${EDGE_VLM_ISAAC_PREF_GUARD_TEST_MODE:-0}" == "1" ]]; then
  assert_isaac_ros_preferences_compatible "Isaac preference guard test mode"
  echo "Isaac ROS host preference guard test passed."
  exit 0
fi

sudo -v
if [[ "${INSTALL_ISAAC_ROS}" -eq 1 ]]; then
  assert_isaac_ros_preferences_compatible "pre-Isaac initialization host check"
fi
sudo apt-get update
bootstrap_packages=(
  ca-certificates
  curl
  jq
  locales
  software-properties-common
  gnupg
)
assert_safe_apt_transaction "bootstrap packages" "${bootstrap_packages[@]}"
sudo apt-get install -y --no-install-recommends "${bootstrap_packages[@]}"

sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

sudo add-apt-repository -y universe

# The Jetson ISO installs the BSP. The full JetPack SDK adds CUDA development
# tools, TensorRT headers/libraries, cuDNN, and the remaining developer packages.
sudo apt-get update
jetpack_packages=(nvidia-jetpack)
assert_safe_apt_transaction "JetPack SDK packages" "${jetpack_packages[@]}"
sudo apt-get install -y "${jetpack_packages[@]}"

ros_source_version="$(
  curl -fsSL https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest |
    sed -n 's/.*"tag_name":[[:space:]]*"\([^"]*\)".*/\1/p' |
    head -n 1
)"
if [[ -z "${ros_source_version}" ]]; then
  echo "Unable to determine the latest ros-apt-source release." >&2
  exit 1
fi

ubuntu_codename="${UBUNTU_CODENAME:-${VERSION_CODENAME:-}}"
ros_source_deb="ros2-apt-source_${ros_source_version}.${ubuntu_codename}_all.deb"
setup_tmp_dir="$(mktemp -d)"
cleanup() {
  if [[ -n "${setup_tmp_dir:-}" && -d "${setup_tmp_dir}" ]]; then
    rm -rf -- "${setup_tmp_dir}"
  fi
}
trap cleanup EXIT

curl -fL \
  -o "${setup_tmp_dir}/${ros_source_deb}" \
  "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ros_source_version}/${ros_source_deb}"
sudo dpkg -i "${setup_tmp_dir}/${ros_source_deb}"

sudo apt-get update
# JetPack installs NVIDIA's OpenCV development packages (nvidia-opencv-dev),
# so avoid pulling Ubuntu's generic libopencv-dev on Thor.
ros_and_build_packages=(
  "${ros_variant}"
  ros-dev-tools
  python3-rosdep
  python3-colcon-common-extensions
  python3-vcstool
  build-essential
  cmake
  ninja-build
  git
  pkg-config
  libnvinfer-dev
  libnvonnxparsers-dev
  "ros-${ROS_DISTRO}-image-transport"
  "ros-${ROS_DISTRO}-image-transport-plugins"
  "ros-${ROS_DISTRO}-rosbag2"
  "ros-${ROS_DISTRO}-rosbag2-storage-mcap"
  "ros-${ROS_DISTRO}-rclcpp"
  "ros-${ROS_DISTRO}-rcl-interfaces"
  "ros-${ROS_DISTRO}-sensor-msgs"
  "ros-${ROS_DISTRO}-std-msgs"
)
assert_safe_apt_transaction "ROS and build dependencies" "${ros_and_build_packages[@]}"
sudo apt-get install -y "${ros_and_build_packages[@]}"

if [[ "${INSTALL_ISAAC_ROS}" -eq 1 ]]; then
  isaac_keyring="/usr/share/keyrings/nvidia-isaac-ros.gpg"
  isaac_source_file="/etc/apt/sources.list.d/nvidia-isaac-ros.list"
  isaac_source="deb [signed-by=${isaac_keyring}] https://isaac.download.nvidia.com/isaac-ros/release-4.6 noble-jetpack main"

  curl -fsSL https://isaac.download.nvidia.com/isaac-ros/repos.key |
    sudo gpg --dearmor --yes -o "${isaac_keyring}"
  echo "${isaac_source}" | sudo tee "${isaac_source_file}" >/dev/null

  sudo apt-get update
  isaac_docker_packages=(isaac-ros-cli nvidia-container-toolkit)
  assert_safe_apt_transaction "Isaac ROS Docker dependencies" "${isaac_docker_packages[@]}"
  sudo apt-get install -y "${isaac_docker_packages[@]}"
  sudo usermod -aG docker "${USER}"
  sudo systemctl enable --now docker
  sudo systemctl restart docker

  # Docker mode is NVIDIA's recommended isolation strategy and protects the
  # host-side TensorRT Edge-LLM stack from Isaac ROS package version pins.
  # Initialization writes system configuration and therefore requires root.
  sudo isaac-ros init docker
  assert_isaac_ros_preferences_compatible "post-isaac-ros init docker host check"

  echo
  echo "Isaac ROS Docker mode initialized."
  echo "Log out and back in before using Docker without sudo."
fi

if [[ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]]; then
  sudo rosdep init
fi
rosdep update --rosdistro "${ROS_DISTRO}"

echo
echo "Dependency installation complete."
echo "Open a new shell or run:"
echo "  source /opt/ros/${ROS_DISTRO}/setup.bash"
echo "Next:"
echo "  ./scripts/setup_thor_jp72.sh"
if [[ "${INSTALL_ISAAC_ROS}" -eq 1 ]]; then
  echo "Isaac ROS verification:"
  echo "  ./scripts/verify_thor_jp72.sh --isaac-ros"
fi
