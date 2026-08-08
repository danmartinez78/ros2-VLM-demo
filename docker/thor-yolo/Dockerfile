# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.05-py3
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG ROS_DISTRO=jazzy
ARG YOLO_ROS_SHA=e712bdee9c5fcffb5e42a17b28792de004224964
ARG YOLO_MODEL_ASSET=yolov8m.pt

ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    ROS_DISTRO=${ROS_DISTRO} \
    ROS_WS=/opt/edge_vlm_yolo_ws \
    YOLO_MODEL_ASSET=${YOLO_MODEL_ASSET}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

COPY --from=ghcr.io/astral-sh/uv:0.6.17 /uv /usr/local/bin/uv

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      curl \
      git \
      gnupg \
      locales \
      python3-colcon-common-extensions \
      python3-rosdep \
      python3-vcstool && \
    locale-gen en_US en_US.UTF-8 && \
    update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 && \
    curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo \"${UBUNTU_CODENAME}\") main" \
      > /etc/apt/sources.list.d/ros2.list && \
    apt-get update && apt-get install -y --no-install-recommends \
      ros-${ROS_DISTRO}-cv-bridge \
      ros-${ROS_DISTRO}-geometry-msgs \
      ros-${ROS_DISTRO}-message-filters \
      ros-${ROS_DISTRO}-ros-base \
      ros-${ROS_DISTRO}-sensor-msgs \
      ros-${ROS_DISTRO}-std-srvs \
      ros-${ROS_DISTRO}-tf2-ros \
      ros-${ROS_DISTRO}-visualization-msgs && \
    rm -rf /var/lib/apt/lists/*

COPY docker/thor-yolo/requirements.txt /tmp/edge_vlm_yolo_requirements.txt
RUN python3 - <<'PY' > /tmp/torch_versions.json
import json
import torch
import torchvision
print(json.dumps({
    'torch': torch.__version__,
    'torchvision': torchvision.__version__,
}))
PY
RUN uv pip install --system -r /tmp/edge_vlm_yolo_requirements.txt && \
    uv pip install --system --no-deps ultralytics==8.4.6 && \
    python3 - <<'PY'
import json
from pathlib import Path
import torch
import torchvision
versions = json.loads(Path('/tmp/torch_versions.json').read_text())
if torch.__version__ != versions['torch']:
    raise SystemExit(f"torch changed from {versions['torch']} to {torch.__version__}")
if torchvision.__version__ != versions['torchvision']:
    raise SystemExit(
        f"torchvision changed from {versions['torchvision']} to {torchvision.__version__}")
PY

RUN mkdir -p "${ROS_WS}/src" /opt/models && \
    cd "${ROS_WS}/src" && \
    git clone https://github.com/mgonzs13/yolo_ros.git && \
    cd yolo_ros && \
    git checkout "${YOLO_ROS_SHA}" && \
    source "/opt/ros/${ROS_DISTRO}/setup.bash" && \
    cd "${ROS_WS}" && \
    colcon build --merge-install --packages-select yolo_msgs yolo_ros

RUN python3 - <<'PY'
from pathlib import Path
from ultralytics.utils.downloads import attempt_download_asset
import os
model_asset = os.environ.get('YOLO_MODEL_ASSET', 'yolov8m.pt')
source = Path(attempt_download_asset(model_asset))
target = Path('/opt/models') / Path(model_asset).name
if source.resolve() != target.resolve():
    target.write_bytes(source.read_bytes())
PY

COPY docker/thor-yolo/entrypoint.sh /usr/local/bin/edge-vlm-thor-yolo-entrypoint
RUN chmod +x /usr/local/bin/edge-vlm-thor-yolo-entrypoint

WORKDIR ${ROS_WS}
ENTRYPOINT ["/usr/local/bin/edge-vlm-thor-yolo-entrypoint"]
