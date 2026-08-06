#!/usr/bin/env python3
# Copyright 2025 edge_vlm_ros contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Thor tracked-observation bring-up with optional detector backends, adapter, VLM, and RViz2."""

import os

from ament_index_python.packages import PackageNotFoundError, get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, SetLaunchConfiguration
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

_REPO_YOLO_INCLUDE_PATH = None
_RTDETR_LAUNCH_PATH = None


def _truthy(value: str) -> bool:
    return value.lower() in ('1', 'true', 'yes', 'on')


def _require_existing_path(label: str, path: str) -> None:
    if not path:
        raise RuntimeError(f'{label} must be set to an existing absolute path.')
    if not os.path.isabs(path):
        raise RuntimeError(f'{label} must be an absolute path: {path}')
    if not os.path.exists(path):
        raise RuntimeError(f'{label} does not exist: {path}')


def _resolve_isaac_rtdetr_launch() -> str:
    candidates = [
        ('isaac_ros_rtdetr', os.path.join('launch', 'isaac_ros_rtdetr.launch.py')),
        ('isaac_ros_examples', os.path.join('launch', 'isaac_ros_examples.launch.py')),
    ]
    attempted = []
    for package_name, rel_path in candidates:
        try:
            share_dir = get_package_share_directory(package_name)
        except PackageNotFoundError:
            attempted.append(f'{package_name} (package missing)')
            continue
        launch_path = os.path.join(share_dir, rel_path)
        if os.path.exists(launch_path):
            return launch_path
        attempted.append(f'{package_name}:{launch_path}')
    joined = ', '.join(attempted) if attempted else 'no candidate packages checked'
    raise RuntimeError(
        'Supported Isaac ROS RT-DETR launch file not found. Install the Isaac ROS RT-DETR packages '
        f'and ensure one of these launch files exists: {joined}')


def _resolve_repo_yolo_include() -> str:
    share_dir = get_package_share_directory('edge_vlm_ros')
    launch_path = os.path.join(share_dir, 'launch', 'ultralytics_yolo_detector.launch.py')
    if not os.path.exists(launch_path):
        raise RuntimeError(f'Repo-owned YOLO detector include not found: {launch_path}')
    return launch_path


def _validate_thor_launch(context, *args, **kwargs):
    global _REPO_YOLO_INCLUDE_PATH
    global _RTDETR_LAUNCH_PATH
    share_dir = get_package_share_directory('edge_vlm_ros')
    rviz_config = os.path.join(share_dir, 'rviz', 'vision_reasoning_results.rviz')
    _require_existing_path('RViz config', rviz_config)

    llm_engine_dir = LaunchConfiguration('llm_engine_dir').perform(context)
    multimodal_engine_dir = LaunchConfiguration('multimodal_engine_dir').perform(context)
    edge_llm_plugin_path = LaunchConfiguration('edge_llm_plugin_path').perform(context)
    _require_existing_path('llm_engine_dir', llm_engine_dir)
    _require_existing_path('multimodal_engine_dir', multimodal_engine_dir)
    _require_existing_path('edge_llm_plugin_path', edge_llm_plugin_path)

    if _truthy(LaunchConfiguration('enable_rviz').perform(context)):
        try:
            rviz_prefix = get_package_prefix('rviz2')
        except PackageNotFoundError as exc:
            raise RuntimeError('rviz2 package is required when enable_rviz:=true') from exc
        rviz_executable = os.path.join(rviz_prefix, 'lib', 'rviz2', 'rviz2')
        if not os.path.exists(rviz_executable):
            raise RuntimeError(f'rviz2 executable not found: {rviz_executable}')

    detector_backend = LaunchConfiguration('detector_backend').perform(context)
    start_rtdetr = _truthy(LaunchConfiguration('start_rtdetr').perform(context))

    if detector_backend == 'isaac_ros_rtdetr' or (detector_backend == 'none' and start_rtdetr):
        _RTDETR_LAUNCH_PATH = _resolve_isaac_rtdetr_launch()
    elif detector_backend == 'ultralytics_yolo':
        _REPO_YOLO_INCLUDE_PATH = _resolve_repo_yolo_include()
    elif detector_backend != 'none':
        raise RuntimeError(
            f'Unsupported detector_backend: {detector_backend}. '
            'Supported values: none, isaac_ros_rtdetr, ultralytics_yolo')
    return []


def _configure_detector_provenance(context, *args, **kwargs):
    detector_id = LaunchConfiguration('detector_id').perform(context).strip()
    if detector_id:
        return [SetLaunchConfiguration('detector_id', detector_id)]

    detector_backend = LaunchConfiguration('detector_backend').perform(context)
    start_rtdetr = _truthy(LaunchConfiguration('start_rtdetr').perform(context))
    if detector_backend == 'ultralytics_yolo':
        detector_id = 'ultralytics_yolo'
    elif detector_backend == 'isaac_ros_rtdetr' or (detector_backend == 'none' and start_rtdetr):
        detector_id = 'isaac_ros_rtdetr'
    else:
        detector_id = 'detector_external'
    return [SetLaunchConfiguration('detector_id', detector_id)]


def generate_launch_description() -> LaunchDescription:
    base_launch = PathJoinSubstitution([
        FindPackageShare('edge_vlm_ros'), 'launch', 'edge_vlm.launch.py'
    ])

    use_sim_time = LaunchConfiguration('use_sim_time')
    image_topic = LaunchConfiguration('image_topic')
    detections_topic = LaunchConfiguration('detections_topic')
    tracked_observation_topic = LaunchConfiguration('tracked_observation_topic')
    llm_engine_dir = LaunchConfiguration('llm_engine_dir')
    multimodal_engine_dir = LaunchConfiguration('multimodal_engine_dir')
    edge_llm_plugin_path = LaunchConfiguration('edge_llm_plugin_path')
    enable_rviz = LaunchConfiguration('enable_rviz')
    start_rtdetr = LaunchConfiguration('start_rtdetr')
    detector_backend = LaunchConfiguration('detector_backend')
    yolo_namespace = LaunchConfiguration('yolo_namespace')
    yolo_model = LaunchConfiguration('yolo_model')
    rviz_config = PathJoinSubstitution([
        FindPackageShare('edge_vlm_ros'), 'rviz', 'vision_reasoning_results.rviz'
    ])

    def launch_selected_detector(context, *args, **kwargs):
        global _REPO_YOLO_INCLUDE_PATH
        global _RTDETR_LAUNCH_PATH
        actions = []
        detector_backend_value = detector_backend.perform(context)
        start_rtdetr_value = _truthy(start_rtdetr.perform(context))
        if detector_backend_value == 'ultralytics_yolo':
            actions.append(
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(_REPO_YOLO_INCLUDE_PATH),
                    launch_arguments={
                        'image_topic': image_topic,
                        'detections_topic': detections_topic,
                        'yolo_namespace': yolo_namespace,
                        'yolo_model': yolo_model,
                    }.items(),
                )
            )
        if detector_backend_value == 'isaac_ros_rtdetr' or (
            detector_backend_value == 'none' and start_rtdetr_value
        ):
            if _RTDETR_LAUNCH_PATH is None:
                raise RuntimeError('RT-DETR launch path was not validated before backend startup.')
            actions.append(
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(_RTDETR_LAUNCH_PATH),
                    launch_arguments={
                        'use_sim_time': use_sim_time,
                        'image_topic': image_topic,
                        'input_image_topic': image_topic,
                        'camera_image_topic': image_topic,
                        'detections_topic': detections_topic,
                        'output_detections_topic': detections_topic,
                        'detection2_d_array_topic': detections_topic,
                    }.items(),
                )
            )
        return actions

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('image_topic', default_value='/camera0/color/image_raw'),
        DeclareLaunchArgument('detections_topic', default_value='/detections'),
        DeclareLaunchArgument('tracked_observation_topic', default_value='/tracked_observation'),
        DeclareLaunchArgument('result_topic', default_value='/vlm/result'),
        DeclareLaunchArgument('start_rtdetr', default_value='false'),
        DeclareLaunchArgument('detector_backend', default_value='none'),
        DeclareLaunchArgument(
            'detector_id',
            default_value='',
            description='Optional detector identity stamped into tracked observations. Defaults to the selected managed backend, or detector_external when no backend is launched here.',
        ),
        DeclareLaunchArgument('tracker_id', default_value='iou_tracker'),
        DeclareLaunchArgument('yolo_namespace', default_value='yolo'),
        DeclareLaunchArgument('yolo_model', default_value='yolov8m.pt'),
        DeclareLaunchArgument('enable_rviz', default_value='true'),
        DeclareLaunchArgument('llm_engine_dir', default_value=''),
        DeclareLaunchArgument('multimodal_engine_dir', default_value=''),
        DeclareLaunchArgument('edge_llm_plugin_path', default_value=''),
        OpaqueFunction(function=_validate_thor_launch),
        OpaqueFunction(function=_configure_detector_provenance),
        OpaqueFunction(function=launch_selected_detector),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(base_launch),
            launch_arguments={
                'use_sim_time': use_sim_time,
                'enable_tracked_observation_input': 'true',
                'tracked_observation_topic': tracked_observation_topic,
                'image_topic': image_topic,
                'result_topic': LaunchConfiguration('result_topic'),
                'sample_period_seconds': '0.0',
                'min_vlm_interval_seconds': '0.0',
                'llm_engine_dir': llm_engine_dir,
                'multimodal_engine_dir': multimodal_engine_dir,
                'edge_llm_plugin_path': edge_llm_plugin_path,
            }.items(),
        ),
        Node(
            package='edge_vlm_ros',
            executable='edge_vlm_tracked_observation_adapter',
            name='tracked_observation_adapter',
            output='screen',
            remappings=[
                ('/detections', detections_topic),
                ('/tracked_observation', tracked_observation_topic),
            ],
            parameters=[{
                'use_sim_time': use_sim_time,
                'image_topic': image_topic,
                'detections_topic': detections_topic,
                'tracked_observation_topic': tracked_observation_topic,
                'detector_id': LaunchConfiguration('detector_id'),
                'tracker_id': LaunchConfiguration('tracker_id'),
            }],
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='edge_vlm_rviz2',
            output='screen',
            arguments=['-d', rviz_config],
            parameters=[{'use_sim_time': use_sim_time}],
            condition=IfCondition(enable_rviz),
        ),
    ])
