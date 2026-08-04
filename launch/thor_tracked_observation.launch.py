#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    base_launch = PathJoinSubstitution([
        FindPackageShare('edge_vlm_ros'), 'launch', 'edge_vlm.launch.py'
    ])

    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(base_launch),
            launch_arguments={
                'enable_tracked_observation_input': 'true',
                'tracked_observation_topic': '/tracked_observation',
                'image_topic': '/camera0/color/image_raw',
                'sample_period_seconds': '0.0',
                'min_vlm_interval_seconds': '0.0',
            }.items(),
        )
    ])
