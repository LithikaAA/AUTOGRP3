#!/usr/bin/env python3
"""
Nav2 navigation stack for the Pioneer robot.

Brings up:
  - controller_server  (DWB local planner + local costmap)
  - planner_server     (NavFn global planner + global costmap)
  - behavior_server    (spin / backup recovery behaviors)
  - bt_navigator       (NavigateToPose / FollowWaypoints actions)
  - lifecycle_manager  (activates all of the above)

Requires slam_mapping.launch.py to already be running so that the
/map topic and map→odom TF are available.
"""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('pioneer_nav')

    use_sim_time = LaunchConfiguration('use_sim_time')
    nav2_params_file = LaunchConfiguration('nav2_params_file')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use /clock from Gazebo when running in simulation.',
        ),
        DeclareLaunchArgument(
            'nav2_params_file',
            default_value=PathJoinSubstitution([pkg, 'config', 'nav2_params.yaml']),
            description='Full path to the Nav2 parameters YAML file.',
        ),

        # ── DWB local planner + local costmap ──────────────────────────────
        Node(
            package='nav2_controller',
            executable='controller_server',
            output='screen',
            parameters=[nav2_params_file, {'use_sim_time': use_sim_time}],
            remappings=[('/cmd_vel', '/cmd_vel')],
        ),

        # ── NavFn global planner + global costmap ──────────────────────────
        Node(
            package='nav2_planner',
            executable='planner_server',
            name='planner_server',
            output='screen',
            parameters=[nav2_params_file, {'use_sim_time': use_sim_time}],
        ),

        # ── Recovery behaviours (spin, backup) ─────────────────────────────
        Node(
            package='nav2_behaviors',
            executable='behavior_server',
            name='behavior_server',
            output='screen',
            parameters=[nav2_params_file, {'use_sim_time': use_sim_time}],
        ),

        # ── BT Navigator: provides NavigateToPose action server ───────────
        Node(
            package='nav2_bt_navigator',
            executable='bt_navigator',
            name='bt_navigator',
            output='screen',
            parameters=[nav2_params_file, {'use_sim_time': use_sim_time}],
        ),

        # ── Lifecycle manager activates all Nav2 nodes in order ───────────
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_navigation',
            output='screen',
            parameters=[
                {'use_sim_time': use_sim_time},
                {'autostart': True},
                {'bond_timeout': 4.0},
                {'node_names': [
                    'controller_server',
                    'planner_server',
                    'behavior_server',
                    'bt_navigator',
                ]},
            ],
        ),
    ])
