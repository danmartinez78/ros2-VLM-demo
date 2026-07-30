#!/usr/bin/env bash
set -Eeuo pipefail

ROS_DISTRO="jazzy"
INSTALL_DESKTOP=0
INSTALL_ISAAC_ROS=0
FORCE_UNSUPPORTED=0

usage() {
  cat <<'EOF'
Usage: sudo -v && ./scripts/install_dependencies.sh [--desktop] [--isaac-ros] [--force]

Installs the JetPack 7.2 development stack, ROS 2 Jazzy, rosdep/colcon,
OpenCV, rosbag2, and all system packages needed by this repository.

Options:
  --desktop   Install ros-jazzy-desktop instead of the headless ros-base variant.
  --isaac-ros Configure NVIDIA Isaac ROS 4.5 using its recommended Docker mode.
              JetPack 7.2 / R39.2 is newer than NVIDIA's validated JetPack 7.1 / R38.4 target.
  --force    Continue on an unsupported OS, architecture, or Jetson Linux release.
EOF
}

for arg in "$@"; do
  case "$arg" in
    --desktop) INSTALL_DESKTOP=1 ;;
    --isaac-ros) INSTALL_ISAAC_ROS=1 ;;
    --force) FORCE_UNSUPPORTED=1 ;;
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
[[ "${l4t_release}" == *"# R39 (release), REVISION: 2"* ]] || unsupported=1

if [[ "${unsupported}" -ne 0 && "${FORCE_UNSUPPORTED}" -ne 1 ]]; then
  cat >&2 <<EOF
Unsupported deployment target.
Expected: Ubuntu 24.04, aarch64, Jetson Linux R39.2
Detected: ${PRETTY_NAME:-unknown}, ${machine_arch}, ${l4t_release:-no /etc/nv_tegra_release}
Use --force only if you understand that this repository targets JetPack 7.2.
EOF
  exit 1
fi

sudo -v
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  ca-certificates curl locales software-properties-common gnupg

sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

sudo add-apt-repository -y universe

# The Jetson ISO installs the BSP. The full JetPack SDK adds CUDA development
# tools, TensorRT headers/libraries, cuDNN, and the remaining developer packages.
sudo apt-get update
sudo apt-get install -y nvidia-jetpack

ros_source_version="$(
  curl -fsSL https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest |
    sed -n 's/.*"tag_name":[[:space:]]*"\([^"]*\)".*/\1/p' |
    head -n 1
)"
if [[ -z "${ros_source_version}" ]]; then
  echo "Unable to determine the latest ros-apt-source release." >&2
  exit 1
fi

ubuntu_codename="${UBUNTU_CODENAME:-${VERSION_CODENAME}}"
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

ros_variant="ros-${ROS_DISTRO}-ros-base"
if [[ "${INSTALL_DESKTOP}" -eq 1 ]]; then
  ros_variant="ros-${ROS_DISTRO}-desktop"
fi

sudo apt-get update
sudo apt-get install -y \
  "${ros_variant}" \
  ros-dev-tools \
  python3-rosdep \
  python3-colcon-common-extensions \
  python3-vcstool \
  build-essential \
  cmake \
  ninja-build \
  git \
  pkg-config \
  libopencv-dev \
  libnvinfer-dev \
  libnvonnxparsers-dev \
  "ros-${ROS_DISTRO}-cv-bridge" \
  "ros-${ROS_DISTRO}-image-transport" \
  "ros-${ROS_DISTRO}-image-transport-plugins" \
  "ros-${ROS_DISTRO}-rosbag2" \
  "ros-${ROS_DISTRO}-rosbag2-storage-mcap" \
  "ros-${ROS_DISTRO}-rclcpp" \
  "ros-${ROS_DISTRO}-rcl-interfaces" \
  "ros-${ROS_DISTRO}-sensor-msgs" \
  "ros-${ROS_DISTRO}-std-msgs"

if [[ "${INSTALL_ISAAC_ROS}" -eq 1 ]]; then
  cat >&2 <<'EOF'

WARNING: NVIDIA Isaac ROS 4.5 currently validates Jetson Thor on JetPack 7.1
(Jetson Linux R38.4). This host targets JetPack 7.2 / R39.2 for TensorRT
Edge-LLM. Isaac ROS is being configured in Docker isolation mode to avoid
replacing the host CUDA, TensorRT, or OpenCV packages. This combination is
experimental until NVIDIA adds JetPack 7.2 to its support matrix.
EOF

  isaac_keyring="/usr/share/keyrings/nvidia-isaac-ros.gpg"
  isaac_source_file="/etc/apt/sources.list.d/nvidia-isaac-ros.list"
  isaac_source="deb [signed-by=${isaac_keyring}] https://isaac.download.nvidia.com/isaac-ros/release-4.5 noble-jetpack main"

  curl -fsSL https://isaac.download.nvidia.com/isaac-ros/repos.key |
    sudo gpg --dearmor --yes -o "${isaac_keyring}"
  echo "${isaac_source}" | sudo tee "${isaac_source_file}" >/dev/null

  sudo apt-get update
  sudo apt-get install -y isaac-ros-cli nvidia-container-toolkit
  sudo usermod -aG docker "${USER}"
  sudo systemctl enable --now docker
  sudo systemctl restart docker

  # Docker mode is NVIDIA's recommended isolation strategy and protects the
  # host-side TensorRT Edge-LLM stack from Isaac ROS package version pins.
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
echo "  cp scripts/cosmos_env.sh.example scripts/cosmos_env.sh"
echo "  edit scripts/cosmos_env.sh"
echo "  ./scripts/build_workspace.sh"
if [[ "${INSTALL_ISAAC_ROS}" -eq 1 ]]; then
  echo "Isaac ROS verification:"
  echo "  ./scripts/verify_deployment.sh --isaac-ros"
fi
