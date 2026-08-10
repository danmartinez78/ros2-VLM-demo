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
COPY docker/thor-yolo/hpcx-env.sh /usr/local/bin/edge-vlm-thor-yolo-hpcx-env
RUN chmod +x /usr/local/bin/edge-vlm-thor-yolo-hpcx-env

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      curl \
      git \
      gnupg \
      locales \
      software-properties-common && \
    add-apt-repository -y universe && \
    locale-gen en_US en_US.UTF-8 && \
    update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 && \
    curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo \"${VERSION_CODENAME}\") main" \
      > /etc/apt/sources.list.d/ros2.list && \
    apt-get update && apt-get install -y --no-install-recommends \
      ros-dev-tools \
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
RUN source /usr/local/bin/edge-vlm-thor-yolo-hpcx-env && \
    python3 -c 'import torch; print(torch.__version__)' > /tmp/torch_version.txt && \
    python3 -c 'import torchvision; print(torchvision.__version__)' > /tmp/torchvision_version.txt
RUN source /usr/local/bin/edge-vlm-thor-yolo-hpcx-env && \
    uv pip install --system -r /tmp/edge_vlm_yolo_requirements.txt && \
    uv pip install --system --no-deps ultralytics==8.4.6 && \
    python3 -c 'import torch; print(torch.__version__)' > /tmp/torch_version_after.txt && \
    python3 -c 'import torchvision; print(torchvision.__version__)' > /tmp/torchvision_version_after.txt && \
    test "$(cat /tmp/torch_version.txt)" = "$(cat /tmp/torch_version_after.txt)" && \
    test "$(cat /tmp/torchvision_version.txt)" = "$(cat /tmp/torchvision_version_after.txt)"

RUN mkdir -p "${ROS_WS}/src" /opt/models && \
    cd "${ROS_WS}/src" && \
    git clone https://github.com/mgonzs13/yolo_ros.git && \
    cd yolo_ros && \
    git checkout "${YOLO_ROS_SHA}" && \
    source "/opt/ros/${ROS_DISTRO}/setup.bash" && \
    cd "${ROS_WS}" && \
    colcon build --merge-install --packages-select yolo_msgs yolo_ros

RUN source /usr/local/bin/edge-vlm-thor-yolo-hpcx-env && \
    python3 - <<'PY'
import shutil
from pathlib import Path
from ultralytics.utils.downloads import attempt_download_asset
import os
model_asset = os.environ.get('YOLO_MODEL_ASSET', 'yolov8m.pt')
source = Path(attempt_download_asset(model_asset))
if not source.exists():
    raise SystemExit(
        f'Failed to download the pinned YOLO model asset during image build: {model_asset}')
target = Path('/opt/models') / Path(model_asset).name
if source.resolve() != target.resolve():
    shutil.copy2(source, target)
PY

COPY docker/thor-yolo/entrypoint.sh /usr/local/bin/edge-vlm-thor-yolo-entrypoint
RUN chmod +x /usr/local/bin/edge-vlm-thor-yolo-entrypoint

WORKDIR ${ROS_WS}
ENTRYPOINT ["/usr/local/bin/edge-vlm-thor-yolo-entrypoint"]
