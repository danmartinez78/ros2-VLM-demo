#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    base_launch = PathJoinSubstitution([
        FindPackageShare('edge_vlm_ros'), 'launch', 'edge_vlm.launch.py'
    ])

    use_sim_time = LaunchConfiguration('use_sim_time')
    image_topic = LaunchConfiguration('image_topic')
    detections_topic = LaunchConfiguration('detections_topic')
    tracked_observation_topic = LaunchConfiguration('tracked_observation_topic')

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('image_topic', default_value='/camera0/color/image_raw'),
        DeclareLaunchArgument('detections_topic', default_value='/detections'),
        DeclareLaunchArgument('tracked_observation_topic', default_value='/tracked_observation'),
        DeclareLaunchArgument('detector_id', default_value='isaac_ros_rtdetr'),
        DeclareLaunchArgument('tracker_id', default_value='iou_tracker'),
        DeclareLaunchArgument(
            'rosbag_path',
            default_value='/home/daniel/ros2-VLM-demo/test_data/rosbags/nvblox/isaac_ros_nvblox/galileo_people_3_2',
        ),
        DeclareLaunchArgument(
            'play_rosbag',
            default_value='false',
        ),
        DeclareLaunchArgument(
            'rtdetr_launch_file',
            default_value='',
        ),
        DeclareLaunchArgument(
            'start_rtdetr',
            default_value='false',
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(base_launch),
            launch_arguments={
                'use_sim_time': use_sim_time,
                'enable_tracked_observation_input': 'true',
                'tracked_observation_topic': tracked_observation_topic,
                'image_topic': image_topic,
                'sample_period_seconds': '0.0',
                'min_vlm_interval_seconds': '0.0',
            }.items(),
        ),
        Node(
            package='edge_vlm_ros',
            executable='edge_vlm_tracked_observation_adapter',
            name='tracked_observation_adapter',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'image_topic': image_topic,
                'detections_topic': detections_topic,
                'tracked_observation_topic': tracked_observation_topic,
                'detector_id': LaunchConfiguration('detector_id'),
                'tracker_id': LaunchConfiguration('tracker_id'),
            }],
        ),
        ExecuteProcess(
            cmd=['ros2', 'bag', 'play', LaunchConfiguration('rosbag_path'), '--clock'],
            output='screen',
            condition=IfCondition(LaunchConfiguration('play_rosbag')),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(LaunchConfiguration('rtdetr_launch_file')),
            condition=IfCondition(LaunchConfiguration('start_rtdetr')),
            launch_arguments={
                'use_sim_time': use_sim_time,
                'image_topic': image_topic,
            }.items(),
        ),
    ])
