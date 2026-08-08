#!/usr/bin/env python3
# Copyright 2025 edge_vlm_ros contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Repo-owned host-side Detection2D bridge for the containerized Ultralytics YOLO backend."""

import os

from ament_index_python.packages import get_package_prefix
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _validate_launch(context, *args, **kwargs):
    prefix = get_package_prefix('edge_vlm_ros')
    adapter_executable = os.path.join(
        prefix, 'lib', 'edge_vlm_ros', 'edge_vlm_yolo_detection2d_adapter')
    if not os.path.exists(adapter_executable):
        raise RuntimeError(
            'The YOLO Detection2D adapter executable is not installed. Rebuild edge_vlm_ros '
            'in an environment where yolo_msgs is available before using this launch include.')

    yolo_detections_topic = _namespaced_topic(
        LaunchConfiguration('yolo_namespace').perform(context), 'detections')
    detections_topic = LaunchConfiguration('detections_topic').perform(context).strip()
    if yolo_detections_topic == detections_topic:
        raise RuntimeError(
            'The YOLO backend detections topic matches the adapted output topic. '
            'Use a non-empty yolo_namespace or set detections_topic to a different topic '
            'to avoid republishing the adapter output back into its own input.')
    return []


def _namespaced_topic(namespace: str, topic_name: str) -> str:
    cleaned_namespace = namespace.strip().strip('/')
    if not cleaned_namespace:
        return f'/{topic_name}'
    return f'/{cleaned_namespace}/{topic_name}'


def _launch_yolo_backend(context, *args, **kwargs):
    detections_topic = LaunchConfiguration('detections_topic')
    yolo_namespace = LaunchConfiguration('yolo_namespace')
    yolo_detections_topic = _namespaced_topic(yolo_namespace.perform(context), 'detections')
    return [
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
