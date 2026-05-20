"""
AUTO4508 Part 3 — Full System Launch File
==========================================
Starts all nodes needed for Part 3 in the correct order.

On the ROBOT run:
    ros2 launch pioneer_nav part3_mission.launch.py

On the LAPTOP run separately:
    python3 ros2_ws/src/pioneer_nav/pioneer_nav/GUI.py

Node startup order:
    1. PS4 joystick          — must be first so deadman is ready
    2. OAK-D camera          — detector depends on this
    3. Lidar                 — control node depends on this
    4. Odom TF broadcaster   — control node depends on this
    5. Control node          — depends on odom, lidar, joystick
    6. Unified detector      — depends on camera
    7. SLAM toolbox          — depends on odom TF
    8. Mission manager       — top level coordinator, starts last
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    # ── Launch arguments ────────────────────────────────────────────────
    # These can be overridden on the command line:
    #   ros2 launch pioneer_nav part3_mission.launch.py scan_topic:=/scan2

    return LaunchDescription([

        DeclareLaunchArgument('joy_topic',    default_value='/joy'),
        DeclareLaunchArgument('scan_topic',   default_value='/scan'),
        DeclareLaunchArgument('odom_topic',   default_value='/odom'),
        DeclareLaunchArgument('cmd_vel_topic', default_value='/cmd_vel'),
        DeclareLaunchArgument('camera_topic', default_value='/oak/rgb/image_raw'),

        # ── 1. PS4 Joystick ─────────────────────────────────────────────
        # Reads the controller and publishes /joy
        # The deadman switch and mode buttons depend on this
        Node(
            package='pioneer_nav',
            executable='ps4_joystick',
            name='ps4_joystick',
            output='screen',
        ),

        # ── 2. OAK-D Camera ─────────────────────────────────────────────
        # Publishes /oak/rgb/image_raw
        # The unified detector subscribes to this
        # Uses depthai_ros_driver if installed, otherwise use your own node
        Node(
            package='depthai_ros_driver',
            executable='camera',
            name='oak_camera',
            output='screen',
            parameters=[{
                'camera_model': 'OAK-D',
                'rgb_fps': 15,
            }],
        ),

        # ── 3. Lidar ────────────────────────────────────────────────────
        # Publishes /scan
        # control_node uses this for obstacle detection and estop
        Node(
            package='sick_scan_xd',
            executable='sick_generic_caller',
            name='lidar',
            output='screen',
            parameters=[{
                'scanner_type': 'sick_tim_7xx',
            }],
        ),

        # ── 4. Odom TF Broadcaster ──────────────────────────────────────
        # Publishes odometry and TF transforms
        # control_node and SLAM toolbox both need this
        Node(
            package='pioneer_nav',
            executable='odom_tf_broadcaster',
            name='odom_tf_broadcaster',
            output='screen',
        ),

        # ── 5. Control Node ─────────────────────────────────────────────
        # Handles autonomous driving, obstacle avoidance, boundary detection
        # Publishes /robot/pose and /arena_status for the GUI
        # Subscribes to /joy, /scan, /odom
        # NOTE: state_pub removed — mission_manager owns /robot_state
        TimerAction(
            period=2.0,   # wait 2s for odom and lidar to be ready
            actions=[
                Node(
                    package='pioneer_nav',
                    executable='control_node',
                    name='pioneer_control_node',
                    output='screen',
                    parameters=[{
                        'joy_topic':     LaunchConfiguration('joy_topic'),
                        'scan_topic':    LaunchConfiguration('scan_topic'),
                        'odom_topic':    LaunchConfiguration('odom_topic'),
                        'cmd_vel_topic': LaunchConfiguration('cmd_vel_topic'),
                        'external_manual_control': True,
                        'center_arena_on_start':   True,
                    }],
                ),
            ]
        ),

        # ── 6. Unified Detector ─────────────────────────────────────────
        # Handles both greek letter detection and colour obstacle detection
        # Single node = single camera subscription = no "camera busy" error
        # Publishes /detected_letter and /detections/colour for the GUI
        TimerAction(
            period=3.0,   # wait for camera to be ready
            actions=[
                Node(
                    package='pioneer_nav',
                    executable='unified_detector_node',
                    name='unified_detector',
                    output='screen',
                    parameters=[{
                        'topic':                    LaunchConfiguration('camera_topic'),
                        'confidence_threshold':     0.5,
                        'confirmations_required':   3,
                        'process_every_n_frames':   3,
                        'require_mapping_state':    False,
                    }],
                ),
            ]
        ),

        # ── 7. SLAM Toolbox ─────────────────────────────────────────────
        # Builds the map during the mapping phase
        # Publishes /map for the GUI map panel
        # Needs odom TF to be running first
        TimerAction(
            period=3.0,
            actions=[
                Node(
                    package='slam_toolbox',
                    executable='async_slam_toolbox_node',
                    name='slam_toolbox',
                    output='screen',
                    parameters=[{
                        'use_sim_time': False,
                        'odom_frame':   'odom',
                        'map_frame':    'map',
                        'base_frame':   'base_link',
                    }],
                ),
            ]
        ),

        # ── 8. Mission Manager ──────────────────────────────────────────
        # Top-level coordinator — starts last
        # Owns /robot_state (feeds the GUI state badge)
        # Publishes /mapping/enable and /waypoint/enable to control phases
        # Services: /mission/start_mapping, /mission/start_waypoint, /mission/stop
        # To start mapping from terminal:
        #   ros2 service call /mission/start_mapping std_srvs/srv/Trigger {}
        TimerAction(
            period=5.0,   # wait for everything else to be ready
            actions=[
                Node(
                    package='pioneer_nav',
                    executable='mission_manager',
                    name='mission_manager',
                    output='screen',
                ),
            ]
        ),

    ])