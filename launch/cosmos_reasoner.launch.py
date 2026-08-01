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
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory('cosmos_ros2_video_reasoner')
    default_config = os.path.join(pkg_share, 'config', 'cosmos_reasoner.yaml')
    package_prefix = os.path.dirname(os.path.dirname(pkg_share))
    worker_executable = os.path.join(
        package_prefix, 'lib', 'cosmos_ros2_video_reasoner', 'cosmos_inference_worker')

    # ── Launch arguments ──────────────────────────────────────────────────────
    args = [
        DeclareLaunchArgument(
            'worker_socket_path',
            default_value='/tmp/cosmos_edge_llm.sock',
            description='Unix-domain socket used by the isolated inference worker',
        ),
        DeclareLaunchArgument(
            'worker_request_timeout_seconds',
            default_value='90',
            description='Maximum seconds to wait for one worker response',
        ),
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
            'task_profile',
            default_value='legacy_prompt',
            description='Named task profile used to render the prompt',
        ),
        DeclareLaunchArgument(
            'prompt_version',
            default_value='v1',
            description='Version label recorded with each reasoning result',
        ),
        DeclareLaunchArgument(
            'system_instruction',
            default_value='',
            description='Optional system-level instruction text',
        ),
        DeclareLaunchArgument(
            'task_instruction',
            default_value='',
            description='Optional task-level instruction text',
        ),
        DeclareLaunchArgument(
            'instruction_delivery_mode',
            default_value='inline',
            description='Instruction delivery mode (currently only inline is supported)',
        ),
        DeclareLaunchArgument(
            'prompt_history_max_entries',
            default_value='0',
            description='Count of prior successful responses retained for prompt-history injection',
        ),
        DeclareLaunchArgument(
            'prompt_history_max_chars',
            default_value='0',
            description='Maximum total retained prompt-history characters (0 disables size limit)',
        ),
        DeclareLaunchArgument(
            'prompt_history_reset_policy',
            default_value='never',
            description='Prompt-history reset policy: never, on_error, every_n_requests',
        ),
        DeclareLaunchArgument(
            'prompt_history_reset_interval_requests',
            default_value='0',
            description='Reset interval used when policy is every_n_requests',
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
            'jpeg_quality',
            default_value='90',
            description='JPEG quality used by the inference worker (1-100)',
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
        DeclareLaunchArgument(
            'benchmark_output_file',
            default_value='',
            description=(
                'When non-empty, append per-frame timing JSON lines to this file for '
                'ROS overhead benchmarking (zero overhead when empty)'
            ),
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
                'worker_socket_path': LaunchConfiguration('worker_socket_path'),
                'worker_request_timeout_seconds': LaunchConfiguration(
                    'worker_request_timeout_seconds'),
                'image_topic': LaunchConfiguration('image_topic'),
                'result_topic': LaunchConfiguration('result_topic'),
                'llm_engine_dir': LaunchConfiguration('llm_engine_dir'),
                'multimodal_engine_dir': LaunchConfiguration('multimodal_engine_dir'),
                'edge_llm_plugin_path': LaunchConfiguration('edge_llm_plugin_path'),
                'prompt': LaunchConfiguration('prompt'),
                'task_profile': LaunchConfiguration('task_profile'),
                'prompt_version': LaunchConfiguration('prompt_version'),
                'system_instruction': LaunchConfiguration('system_instruction'),
                'task_instruction': LaunchConfiguration('task_instruction'),
                'instruction_delivery_mode': LaunchConfiguration('instruction_delivery_mode'),
                'prompt_history_max_entries': LaunchConfiguration('prompt_history_max_entries'),
                'prompt_history_max_chars': LaunchConfiguration('prompt_history_max_chars'),
                'prompt_history_reset_policy': LaunchConfiguration('prompt_history_reset_policy'),
                'prompt_history_reset_interval_requests': LaunchConfiguration(
                    'prompt_history_reset_interval_requests'),
                'sample_period_seconds': LaunchConfiguration('sample_period_seconds'),
                'max_generate_length': LaunchConfiguration('max_generate_length'),
                'temperature': LaunchConfiguration('temperature'),
                'jpeg_quality': LaunchConfiguration('jpeg_quality'),
                'benchmark_output_file': LaunchConfiguration('benchmark_output_file'),
            },
        ],
    )

    worker = ExecuteProcess(
        cmd=[
            worker_executable,
            LaunchConfiguration('llm_engine_dir'),
            LaunchConfiguration('multimodal_engine_dir'),
            LaunchConfiguration('edge_llm_plugin_path'),
            LaunchConfiguration('worker_socket_path'),
            LaunchConfiguration('jpeg_quality'),
        ],
        output='screen',
        respawn=True,
        respawn_delay=2.0,
    )

    return LaunchDescription(args + [worker, cosmos_node])
