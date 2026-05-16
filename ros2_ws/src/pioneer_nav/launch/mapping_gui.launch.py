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

    use_sim_time = LaunchConfiguration('use_sim_time')
    scan_frame = LaunchConfiguration('scan_frame')
    gui = LaunchConfiguration('gui')
    estop = LaunchConfiguration('estop')
    waypoint_driver = LaunchConfiguration('waypoint_driver')
    slam_start_delay = LaunchConfiguration('slam_start_delay')
    odom_tf_stamp_with_current_time = LaunchConfiguration('odom_tf_stamp_with_current_time')

    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                pkg_pioneer_nav,
                'launch',
                'slam_mapping.launch.py',
            ])
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'rviz': 'false',
            'scan_frame': scan_frame,
            'odom_tf_stamp_with_current_time': odom_tf_stamp_with_current_time,
            'slam_start_delay': slam_start_delay,
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('scan_frame', default_value='sick_laser'),
        DeclareLaunchArgument('gui', default_value='true'),
        DeclareLaunchArgument('estop', default_value='true'),
        DeclareLaunchArgument('waypoint_driver', default_value='true'),
        DeclareLaunchArgument('slam_start_delay', default_value='8.0'),
        DeclareLaunchArgument('odom_tf_stamp_with_current_time', default_value='true'),
        slam_launch,
        Node(
            package='pioneer_nav',
            executable='estopconnect',
            name='estopconnect',
            output='screen',
            condition=IfCondition(estop),
        ),
        TimerAction(
            period=2.0,
            actions=[
                Node(
                    package='pioneer_nav',
                    executable='distbug_controller',
                    name='distbug_controller',
                    output='screen',
                    condition=IfCondition(waypoint_driver),
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
