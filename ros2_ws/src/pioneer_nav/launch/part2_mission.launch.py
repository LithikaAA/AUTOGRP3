import math
from launch import LaunchDescription
from launch_ros.actions import Node

# =============================================================================
# WORLD ORIGIN — set this to match where your robot spawns in the SDF world.
# All GPS waypoints below are calculated as offsets from this origin.
# If you move the robot spawn point in the SDF, update these two values.
# =============================================================================
ORIGIN_LAT = -31.980000
ORIGIN_LON = 115.820000


def xy_to_gps(x: float, y: float):
    """
    Convert a local XY position (metres) in the Gazebo world
    to a GPS lat/lon, relative to ORIGIN_LAT / ORIGIN_LON.
    x = east/west,  y = north/south
    """
    lat = ORIGIN_LAT + y / 111320.0
    lon = ORIGIN_LON + x / (111320.0 * math.cos(math.radians(ORIGIN_LAT)))
    return round(lat, 8), round(lon, 8)


# =============================================================================
# WAYPOINTS — defined as (x, y) positions in your Gazebo world (metres).
# These match the waypoint_cone positions in your SDF:
#   waypoint_cone_1  ->  (4,  4)
#   waypoint_cone_2  ->  (-4, 4)
#   waypoint_cone_3  ->  (-4, -4)
# The last waypoint returns the robot to the start.
# Change these x/y values if you move the cones in the SDF.
# =============================================================================
WAYPOINTS_XY = [
    (0.0,   0.0),   # start / origin
    (4.0,   4.0),   # waypoint_cone_1
    (-4.0,  4.0),   # waypoint_cone_2
    (-4.0, -4.0),   # waypoint_cone_3
    (0.0,   0.0),   # return to start
]

# Convert to flat [lat, lon, lat, lon, ...] list for the ROS parameter
gps_waypoints = []
for x, y in WAYPOINTS_XY:
    lat, lon = xy_to_gps(x, y)
    gps_waypoints.extend([lat, lon])


def generate_launch_description():
    return LaunchDescription([

        # -------------------------
        # Gazebo <-> ROS 2 bridges
        # -------------------------

        # cmd_vel: ROS -> Gazebo (drive commands)
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='cmd_vel_bridge',
            output='screen',
            arguments=['/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist']
        ),

        # LiDAR: Gazebo -> ROS
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='scan_bridge',
            output='screen',
            arguments=['/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan']
        ),

        # Camera: Gazebo -> ROS
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='camera_bridge',
            output='screen',
            arguments=['/camera/image@sensor_msgs/msg/Image@gz.msgs.Image']
        ),

        # IMU: Gazebo -> ROS
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='imu_bridge',
            output='screen',
            arguments=['/imu@sensor_msgs/msg/Imu@gz.msgs.IMU']
        ),

        # Odometry: Gazebo -> ROS (needed by fake GPS)
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='odom_bridge',
            output='screen',
            arguments=['/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry']
        ),

        # NOTE: No GPS bridge — Gazebo has no NavSat sensor in this world.
        # The fake_gps node below converts /odom -> /fix instead.

        # -------------------------
        # Fake GPS node
        # Converts /odom (Gazebo odometry) into /fix (NavSatFix)
        # so the mission controller gets real-looking GPS coordinates.
        # -------------------------
        Node(
            package='pioneer_nav',
            executable='fake_gps',
            name='fake_gps',
            output='screen',
            parameters=[{
                'origin_lat': ORIGIN_LAT,
                'origin_lon': ORIGIN_LON,
                'odom_topic': '/odom',
                'fix_topic':  '/fix',
            }]
        ),

        # -------------------------
        # Joystick node
        # -------------------------
        Node(
            package='joy',
            executable='joy_node',
            name='joy_node',
            output='screen'
        ),

        # -------------------------
        # Part 2 mission controller
        # -------------------------
        Node(
            package='pioneer_nav',
            executable='part2_mission_controller',
            name='part2_mission_controller',
            output='screen',
            parameters=[{
                # --- topics ---
                'gps_topic':    '/fix',
                'scan_topic':   '/scan',
                'image_topic':  '/camera/image',
                'joy_topic':    '/joy',
                'imu_topic':    '/imu',
                'cmd_vel_topic': '/cmd_vel',

                # --- GPS waypoints (auto-calculated from WAYPOINTS_XY above) ---
                'gps_waypoints': gps_waypoints,

                # --- joystick mapping (PS4/PS5 default) ---
                # axis 1 = left stick vertical  (forward/back)
                # axis 0 = left stick horizontal (turn)
                # axis 5 = right trigger         (dead-man)
                # button 2 = X                   (enable AUTO)
                # button 1 = O                   (enable MANUAL)
                'joy_axis_linear':   3
                'joy_axis_angular':  2,
                'joy_deadman_axis':  5,
                'joy_auto_button':   0
                'joy_manual_button': 1,

                # --- speed limits ---
                'max_linear_speed':  0.5,
                'max_angular_speed': 1.0,

                # --- navigation tuning ---
                'goal_radius_m':          1.2,   # how close = "waypoint reached"
                'cone_stop_distance_m':   1.4,   # how close to stop at cone
                'object_search_radius_m': 4.0,   # max range to look for object
                'front_obstacle_dist_m':  0.9,   # soft obstacle avoidance threshold
                'critical_obstacle_dist_m': 0.5, # hard stop threshold

                # --- camera / vision tuning ---
                'camera_hfov_rad': 1.089,   # matches SDF camera horizontal_fov
                'cone_min_area':   500.0,   # min pixel area to count as a cone
                'object_min_area': 350.0,   # min pixel area to count as an object

                # --- output ---
                'photos_dir': 'mission_photos',
            }]
        ),
    ])