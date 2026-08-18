#!/usr/bin/env bash
set -Eeuo pipefail

ROS_DISTRO="jazzy"
INSTALL_DESKTOP=0
INSTALL_ISAAC_ROS=0
FORCE_UNSUPPORTED=0
DRY_RUN=0
PROTECTED_NVIDIA_PACKAGES=(
  nvidia-jetpack
  nvidia-jetpack-dev
  nvidia-opencv-dev
)

fail() {
  printf 'ERROR: %b\n' "$*" >&2
  exit 1
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
    fail "Unable to simulate APT transaction for ${description}. Output:\n${simulation_output}"
  fi

  local protected_pkg
  local removed_pkg
  mapfile -t removed_pkgs < <(printf '%s\n' "${simulation_output}" | awk '/^Remv[[:space:]]/{print $2}')
  for protected_pkg in "${PROTECTED_NVIDIA_PACKAGES[@]}"; do
    for removed_pkg in "${removed_pkgs[@]}"; do
      if [[ "${removed_pkg}" == "${protected_pkg}" || "${removed_pkg}" == "${protected_pkg}:"* ]]; then
        fail "Refusing to continue: planned APT transaction for ${description} removes protected package '${protected_pkg}'. Keep the JP7.1 NVIDIA stack intact."
      fi
    done
  done
}

usage() {
  cat <<'EOF'
Usage: sudo -v && ./scripts/install_dependencies.sh [--desktop] [--isaac-ros] [--force] [--dry-run]

Installs the JetPack 7.1 development stack, ROS 2 Jazzy, rosdep/colcon,
OpenCV, rosbag2, and all system packages needed by this repository.

Options:
  --desktop   Install ros-jazzy-desktop instead of the headless ros-base variant.
  --isaac-ros Configure NVIDIA Isaac ROS 4.5 using its recommended Docker mode.
              This repository's supported Thor path targets JetPack 7.1 / R38.4.
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
[[ "${l4t_release}" == *"# R38 (release), REVISION: 4"* ]] || unsupported=1

if [[ "${unsupported}" -ne 0 && "${FORCE_UNSUPPORTED}" -ne 1 ]]; then
  cat >&2 <<EOF
Unsupported deployment target.
Expected: Ubuntu 24.04, aarch64, Jetson Linux R38.4 (JetPack 7.1)
Detected: ${PRETTY_NAME:-unknown}, ${machine_arch}, ${l4t_release:-no /etc/nv_tegra_release}
Use --force only if you understand that this repository targets JetPack 7.1.
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
  Supported target baseline: Ubuntu 24.04, aarch64, Jetson Linux R38.4
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
    - initialize/update rosdep
EOF
  exit 0
fi

if [[ "${EDGE_VLM_APT_GUARD_TEST_MODE:-0}" == "1" ]]; then
  assert_safe_apt_transaction "guard test transaction" guard-test-package
  echo "APT guard test transaction passed."
  exit 0
fi

sudo -v
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
  cat >&2 <<'EOF'

WARNING: NVIDIA Isaac ROS 4.5 currently validates Jetson Thor on JetPack 7.1
(Jetson Linux R38.4). Isaac ROS is being configured in Docker isolation mode to
avoid replacing the host CUDA, TensorRT, or OpenCV packages.
EOF

  isaac_keyring="/usr/share/keyrings/nvidia-isaac-ros.gpg"
  isaac_source_file="/etc/apt/sources.list.d/nvidia-isaac-ros.list"
  isaac_source="deb [signed-by=${isaac_keyring}] https://isaac.download.nvidia.com/isaac-ros/release-4.5 noble-jetpack main"

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
echo "  ./scripts/setup_thor_jp71.sh"
if [[ "${INSTALL_ISAAC_ROS}" -eq 1 ]]; then
  echo "Isaac ROS verification:"
  echo "  ./scripts/verify_thor_jp71.sh --isaac-ros"
fi
