#!/usr/bin/env python3

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    pkg_pioneer_nav = get_package_share_directory('pioneer_nav')

    use_sim_time = LaunchConfiguration('use_sim_time')
    rviz = LaunchConfiguration('rviz')
    slam_params = LaunchConfiguration('slam_params')
    scan_frame = LaunchConfiguration('scan_frame')
    odom_tf_stamp_with_current_time = LaunchConfiguration('odom_tf_stamp_with_current_time')
    rviz_config = LaunchConfiguration('rviz_config')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use Gazebo /clock when mapping in simulation.',
        ),
        DeclareLaunchArgument(
            'rviz',
            default_value='true',
            description='Start RViz for viewing the live map.',
        ),
        DeclareLaunchArgument(
            'slam_params',
            default_value=PathJoinSubstitution([
                pkg_pioneer_nav,
                'config',
                'slam_toolbox_mapping.yaml',
            ]),
            description='Path to slam_toolbox mapping parameters.',
        ),
        DeclareLaunchArgument(
            'scan_frame',
            default_value='pioneer/base_link/laser',
            description='LaserScan frame_id to connect to base_link.',
        ),
        DeclareLaunchArgument(
            'odom_tf_stamp_with_current_time',
            default_value='true',
            description='Stamp odom TF with current ROS time instead of the /odom header stamp.',
        ),
        DeclareLaunchArgument(
            'rviz_config',
            default_value=PathJoinSubstitution([
                pkg_pioneer_nav,
                'config',
                'slam_mapping.rviz',
            ]),
            description='RViz config to load for mapping.',
        ),
        Node(
            package='pioneer_nav',
            executable='odom_tf_broadcaster',
            name='odom_tf_broadcaster',
            output='screen',
            parameters=[
                {'use_sim_time': use_sim_time},
                {'odom_topic': '/odom'},
                {'odom_frame': 'odom'},
                {'base_frame': 'base_link'},
                {'stamp_with_current_time': odom_tf_stamp_with_current_time},
            ],
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='scan_frame_static_tf',
            output='screen',
            arguments=[
                '--x', '0.2',
                '--y', '0.0',
                '--z', '0.281',
                '--roll', '0.0',
                '--pitch', '0.0',
                '--yaw', '0.0',
                '--frame-id', 'base_link',
                '--child-frame-id', scan_frame,
            ],
        ),
        Node(
            package='slam_toolbox',
            executable='async_slam_toolbox_node',
            name='slam_toolbox',
            output='screen',
            parameters=[
                slam_params,
                {'use_sim_time': use_sim_time},
            ],
        ),
        TimerAction(
            period=5.0,
            actions=[
                Node(
                    package='nav2_lifecycle_manager',
                    executable='lifecycle_manager',
                    name='lifecycle_manager_slam',
                    output='screen',
                    parameters=[
                        {'use_sim_time': use_sim_time},
                        {'autostart': True},
                        {'node_names': ['slam_toolbox']},
                        # slam_toolbox does not form the Nav2 bond on this setup,
                        # so let the manager configure/activate it without waiting
                        # for a bond heartbeat.
                        {'bond_timeout': 0.0},
                    ],
                ),
            ],
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            condition=IfCondition(rviz),
            arguments=['-d', rviz_config],
            parameters=[
                {'use_sim_time': use_sim_time},
            ],
        ),
    ])
