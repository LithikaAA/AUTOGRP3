#!/usr/bin/env python3

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    pkg_pioneer_nav = get_package_share_directory('pioneer_nav')

    gui = LaunchConfiguration('gui')
    estop = LaunchConfiguration('estop')
    waypoint_driver = LaunchConfiguration('waypoint_driver')
    slam_start_delay = LaunchConfiguration('slam_start_delay')

    control_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                pkg_pioneer_nav,
                'launch',
                'control_node_gazebo.launch.py',
            ])
        ),
        launch_arguments={
            'rviz': 'false',
        }.items(),
    )

    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                pkg_pioneer_nav,
                'launch',
                'slam_mapping.launch.py',
            ])
        ),
        launch_arguments={
            'use_sim_time': 'true',
            'rviz': 'false',
            'scan_frame': 'pioneer/base_link/laser',
            'odom_tf_stamp_with_current_time': 'true',
            'slam_start_delay': slam_start_delay,
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'gui',
            default_value='true',
            description='Start the PyQt robot monitor GUI.',
        ),
        DeclareLaunchArgument(
            'estop',
            default_value='true',
            description='Start LiDAR e-stop status node.',
        ),
        DeclareLaunchArgument(
            'waypoint_driver',
            default_value='true',
            description='Start distbug waypoint command listener.',
        ),
        DeclareLaunchArgument(
            'slam_start_delay',
            default_value='8.0',
            description='Seconds to wait before starting slam_toolbox.',
        ),
        control_launch,
        TimerAction(
            period=2.0,
            actions=[
                Node(
                    package='pioneer_nav',
                    executable='estopconnect',
                    name='estopconnect',
                    output='screen',
                    condition=IfCondition(estop),
                    parameters=[
                        {'bag_directory': '/tmp/pioneer_estop/bags'},
                        {'incident_log': '/tmp/pioneer_estop/incidents.txt'},
                    ],
                ),
            ],
        ),
        TimerAction(
            period=2.0,
            actions=[slam_launch],
        ),
        TimerAction(
            period=4.0,
            actions=[
                Node(
                    package='pioneer_nav',
                    executable='distbug_controller',
                    name='distbug_controller',
                    output='screen',
                    condition=IfCondition(waypoint_driver),
                    parameters=[
                        {'image_topic': '/oak/rgb/image_raw'},
                        {'imu_topic': '/imu'},
                    ],
                ),
                Node(
                    package='pioneer_nav',
                    executable='robot_gui',
                    name='robot_gui',
                    output='screen',
                    condition=IfCondition(gui),
                ),
            ],
        ),
    ])
