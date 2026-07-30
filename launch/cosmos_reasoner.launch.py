#!/usr/bin/env python3
# Copyright 2025 cosmos_ros2_video_reasoner contributors
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

"""Launch cosmos_reasoner with configurable engine paths and topics."""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory('cosmos_ros2_video_reasoner')
    default_config = os.path.join(pkg_share, 'config', 'cosmos_reasoner.yaml')

    # ── Launch arguments ──────────────────────────────────────────────────────
    args = [
        DeclareLaunchArgument(
            'image_topic',
            default_value='/camera/image_raw',
            description='Input image topic (sensor_msgs/msg/Image)',
        ),
        DeclareLaunchArgument(
            'result_topic',
            default_value='/cosmos/reasoning',
            description='Output result topic (VisionReasoningResult)',
        ),
        DeclareLaunchArgument(
            'llm_engine_dir',
            default_value='',
            description='Absolute path to the LLM TensorRT engine directory',
        ),
        DeclareLaunchArgument(
            'multimodal_engine_dir',
            default_value='',
            description='Absolute path to the multimodal TensorRT engine directory',
        ),
        DeclareLaunchArgument(
            'edge_llm_plugin_path',
            default_value='',
            description='Absolute path to libNvInfer_edgellm_plugin.so',
        ),
        DeclareLaunchArgument(
            'prompt',
            default_value=(
                'Describe the scene in this camera frame. '
                'Identify important objects, people, animals, vehicles, '
                'terrain, hazards, and unusual conditions. '
                'Do not claim details that are not visually supported.'
            ),
            description='Text prompt for the VLM',
        ),
        DeclareLaunchArgument(
            'sample_period_seconds',
            default_value='2.0',
            description='Seconds between sampled frames (uses message timestamp)',
        ),
        DeclareLaunchArgument(
            'max_generate_length',
            default_value='256',
            description='Maximum number of tokens to generate per frame',
        ),
        DeclareLaunchArgument(
            'temperature',
            default_value='0.2',
            description='Sampling temperature',
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time (set true when playing a bag with --clock)',
        ),
    ]

    # ── Node ──────────────────────────────────────────────────────────────────
    cosmos_node = Node(
        package='cosmos_ros2_video_reasoner',
        executable='cosmos_reasoner',
        name='cosmos_reasoner',
        output='screen',
        parameters=[
            default_config,
            {
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'image_topic': LaunchConfiguration('image_topic'),
                'result_topic': LaunchConfiguration('result_topic'),
                'llm_engine_dir': LaunchConfiguration('llm_engine_dir'),
                'multimodal_engine_dir': LaunchConfiguration('multimodal_engine_dir'),
                'edge_llm_plugin_path': LaunchConfiguration('edge_llm_plugin_path'),
                'prompt': LaunchConfiguration('prompt'),
                'sample_period_seconds': LaunchConfiguration('sample_period_seconds'),
                'max_generate_length': LaunchConfiguration('max_generate_length'),
                'temperature': LaunchConfiguration('temperature'),
            },
        ],
    )

    return LaunchDescription(args + [cosmos_node])
