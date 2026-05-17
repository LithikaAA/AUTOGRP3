#!/usr/bin/env python3

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_pioneer_nav = get_package_share_directory('pioneer_nav')

    gui = LaunchConfiguration('gui')
    estop = LaunchConfiguration('estop')
    detector = LaunchConfiguration('detector')
    camera_topic = LaunchConfiguration('camera_topic')
    depth_topic = LaunchConfiguration('depth_topic')
    waypoint_driver = LaunchConfiguration('waypoint_driver')
    obstacle_waypoint_file = LaunchConfiguration('obstacle_waypoint_file')
    waypoint_goal_tolerance = LaunchConfiguration('waypoint_goal_tolerance')
    waypoint_linear_speed = LaunchConfiguration('waypoint_linear_speed')
    waypoint_slow_linear_speed = LaunchConfiguration('waypoint_slow_linear_speed')
    waypoint_obstacle_linear_speed = LaunchConfiguration('waypoint_obstacle_linear_speed')
    waypoint_obstacle_turn_speed = LaunchConfiguration('waypoint_obstacle_turn_speed')
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
        DeclareLaunchArgument('detector', default_value='true'),
        DeclareLaunchArgument('camera_topic', default_value='/camera/image'),
        DeclareLaunchArgument('depth_topic', default_value='/oak/stereo/image_raw'),
        DeclareLaunchArgument(
            'waypoint_driver',
            default_value='true',
            description='Start distbug waypoint command listener.',
        ),
        DeclareLaunchArgument('obstacle_waypoint_file', default_value=''),
        DeclareLaunchArgument('waypoint_goal_tolerance', default_value='0.8'),
        DeclareLaunchArgument('waypoint_linear_speed', default_value='0.18'),
        DeclareLaunchArgument('waypoint_slow_linear_speed', default_value='0.04'),
        DeclareLaunchArgument('waypoint_obstacle_linear_speed', default_value='0.04'),
        DeclareLaunchArgument('waypoint_obstacle_turn_speed', default_value='0.45'),
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
                    package='ros_gz_bridge',
                    executable='parameter_bridge',
                    name='camera_bridge',
                    output='screen',
                    arguments=[
                        '/camera/image@sensor_msgs/msg/Image@gz.msgs.Image',
                    ],
                ),
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
                Node(
                    package='pioneer_nav',
                    executable='unified_detector',
                    name='unified_detector',
                    output='screen',
                    condition=IfCondition(detector),
                    parameters=[
                        {'topic': camera_topic},
                        {'depth_topic': depth_topic},
                        {'require_mapping_state': False},
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
                    executable='waypoint_controller',
                    name='waypoint_controller',
                    output='screen',
                    condition=IfCondition(waypoint_driver),
                    parameters=[
                        {'scan_topic': '/scan'},
                        {'image_topic': '/oak/rgb/image_raw'},
                        {'imu_topic': '/imu'},
                        {'odom_topic': '/odom'},
                        {'cmd_vel_topic': '/cmd_vel'},
                        {'mission_command_topic': '/mission_command'},
                        {'robot_state_topic': '/robot_state'},
                        {'use_gazebo_tf_pose': True},
                        {'gazebo_tf_topic': '/world/pioneer_world/dynamic_pose/info'},
                        {'gazebo_tf_frame_match': 'pioneer'},
                        {'gazebo_tf_allow_unmatched': True},
                        {'obstacle_waypoint_file': obstacle_waypoint_file},
                        {'relative_to_start': True},
                        {'use_test_waypoint': False},
                        {'waypoint_goal_tolerance': ParameterValue(waypoint_goal_tolerance, value_type=float)},
                        {'waypoint_linear_speed': ParameterValue(waypoint_linear_speed, value_type=float)},
                        {'waypoint_slow_linear_speed': ParameterValue(waypoint_slow_linear_speed, value_type=float)},
                        {'waypoint_obstacle_linear_speed': ParameterValue(waypoint_obstacle_linear_speed, value_type=float)},
                        {'waypoint_obstacle_turn_speed': ParameterValue(waypoint_obstacle_turn_speed, value_type=float)},
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
