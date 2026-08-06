#!/usr/bin/env python3
# Copyright 2025 edge_vlm_ros contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Repo-owned include for the external Ultralytics YOLO backend plus Detection2D bridge."""

import os

from ament_index_python.packages import PackageNotFoundError, get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_YOLO_LAUNCH_PATH = None


def _resolve_yolo_launch() -> str:
    candidates = [
        ('yolo_bringup', os.path.join('launch', 'yolo.launch.py')),
        ('yolo_ros', os.path.join('launch', 'yolo.launch.py')),
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
        'Supported Ultralytics YOLO launch file not found. Install the ROS 2 Jazzy '
        f'YOLO packages and ensure one of these launch files exists: {joined}')


def _validate_launch(context, *args, **kwargs):
    global _YOLO_LAUNCH_PATH
    prefix = get_package_prefix('edge_vlm_ros')
    adapter_executable = os.path.join(prefix, 'lib', 'edge_vlm_ros', 'edge_vlm_yolo_detection2d_adapter')
    if not os.path.exists(adapter_executable):
        raise RuntimeError(
            'The YOLO Detection2D adapter executable is not installed. Rebuild edge_vlm_ros '
            'in an environment where yolo_msgs is available before using this launch include.')
    _YOLO_LAUNCH_PATH = _resolve_yolo_launch()
    return []


def _namespaced_topic(namespace: str, topic_name: str) -> str:
    cleaned_namespace = namespace.strip().strip('/')
    if not cleaned_namespace:
        return f'/{topic_name}'
    return f'/{cleaned_namespace}/{topic_name}'


def _launch_yolo_backend(context, *args, **kwargs):
    global _YOLO_LAUNCH_PATH
    if _YOLO_LAUNCH_PATH is None:
        raise RuntimeError('YOLO launch path was not validated before backend startup.')
    image_topic = LaunchConfiguration('image_topic')
    detections_topic = LaunchConfiguration('detections_topic')
    yolo_namespace = LaunchConfiguration('yolo_namespace')
    yolo_model = LaunchConfiguration('yolo_model')
    yolo_detections_topic = _namespaced_topic(yolo_namespace.perform(context), 'detections')
    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(_YOLO_LAUNCH_PATH),
            launch_arguments={
                'input_image_topic': image_topic,
                'image_reliability': '2',
                'model': yolo_model,
                'namespace': yolo_namespace,
            }.items(),
        ),
        Node(
            package='edge_vlm_ros',
            executable='edge_vlm_yolo_detection2d_adapter',
            name='yolo_detection2d_adapter',
            output='screen',
            parameters=[{
                'input_topic': yolo_detections_topic,
                'output_topic': detections_topic,
            }],
        ),
    ]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument('image_topic', default_value='/camera0/color/image_raw'),
        DeclareLaunchArgument('detections_topic', default_value='/detections'),
        DeclareLaunchArgument('yolo_namespace', default_value='yolo'),
        DeclareLaunchArgument('yolo_model', default_value='yolov8m.pt'),
        OpaqueFunction(function=_validate_launch),
        OpaqueFunction(function=_launch_yolo_backend),
    ])
