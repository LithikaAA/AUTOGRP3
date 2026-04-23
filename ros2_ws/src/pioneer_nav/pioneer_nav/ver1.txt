#!/usr/bin/env python3

import math
import time
import rclpy

# for nav files in ws
import os
from ament_index_python.packages import get_package_share_directory

from rclpy.node import Node

# from geometry_msgs.msg import Twist, PoseArray
from geometry_msgs.msg import Twist
from sensor_msgs.msg import NavSatFix
from nav_msgs.msg import Odometry

from sensor_msgs.msg import LaserScan



# this helper is so i can limit values like speeds/turn rates
# and stop the controller from asking the robot to do something too extreme.
def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


# I need angles to stay between -pi and pi, otherwise the robot can try to
# turn the long way around when really it only needs a small correction.
def wrap_to_pi(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


# Gazebo gives orientation as a quaternion, but for my controller I only really
# care about yaw because this is a 2D ground robot.
def quaternion_to_yaw(x, y, z, w):
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class WaypointController(Node):
    def __init__(self):
        super().__init__('distbug_controller')
        
        # ---------------- PARAMETERS ----------------
        # I made these parameters so I can change topic names/indexes without
        # rewriting the whole file if Gazebo or the bridge setup changes.
        self.declare_parameter('world_pose_topic', '/world/pioneer_world/dynamic_pose/info')
        self.declare_parameter('robot_pose_index', 0)
        self.declare_parameter('scan_topic', '/scan')
        
        self.world_pose_topic = self.get_parameter(
            'world_pose_topic'
        ).get_parameter_value().string_value
        
        self.robot_pose_index = self.get_parameter(
            'robot_pose_index'
        ).get_parameter_value().integer_value
        
        self.scan_topic = self.get_parameter(
            'scan_topic'
        ).get_parameter_value().string_value
        
        # ---------------- WAYPOINTS ----------------
        # I load the waypoints from a text file so I can test different paths
        # easily without touching the actual controller logic.
        self.waypoints = []
        # file_path = "/mnt/c/Users/Lithi/Desktop/AUTO4408/Project 1/ros2_ws/src/pioneer_nav/waypoint.txt"

        file_path = os.path.join(
            get_package_share_directory('pioneer_nav'),
            'waypoint.txt'
        )

        try:
            with open(file_path, "r") as f:
                for line in f:
                    # Skip blank lines so the file can still be neat/readable.
                    if line.strip() == "":
                        continue
                    
                    # Each line is x y yaw, so I unpack them straight into floats.
                    # x, y, yaw = map(float, line.split())
                    # self.waypoints.append((x, y, yaw))
                    lat, lon, yaw = map(float, line.split())
                    self.waypoints.append((lat, lon, yaw))
                    
            self.get_logger().info(f"Loaded {len(self.waypoints)} waypoints.")
            
        except Exception as e:
            self.get_logger().error(f"Failed to load waypoints: {e}")
            
        # If nothing loads, there is no point running the node because the robot
        # would not know where to go.
        if len(self.waypoints) == 0:
            self.get_logger().error("No waypoints loaded. Stopping node.")
            return
        
        # Start from the first waypoint.
        self.current_waypoint_index = 0
        self.goal_x, self.goal_y, self.goal_yaw = self.waypoints[self.current_waypoint_index]
        self.current_waypoint_start_time = self.get_clock().now()

        # Pause at waypoints 
        self.pause_duration = 10.0
        self.pause_start_time = None
        self.is_pausing = False
        # Only print mode when it changes so terminal is readable.
        self.last_logged_mode = None
        
        # ---------------- CONTROL GAINS ----------------
        # These are just the simple proportional gains I used for movement.
        # I kept it simple because I wanted something easy to tune first before
        # making the controller more complicated.
        self.k_linear = 0.6
        self.k_angular = 1.5
        self.k_final_yaw = 1.2
        
        # ---------------- SPEED LIMITS ----------------
        # These caps are here so the robot does not drive or spin too aggressively.
        # The min speed helps stop it from crawling forever when it's nearly moving
        # but not quite enough to actually make progress.
        self.max_linear_speed = 0.20
        self.max_angular_speed = 1.0
        self.min_linear_speed = 0.05
        
        # ---------------- TOLERANCES ----------------
        # Position tolerance = how close is “good enough” for waypoint position.
        # Yaw tolerance = how close is “good enough” for final orientation.
        self.position_tolerance = 0.05
        self.yaw_tolerance = 0.05
        # If I get near a waypoint and there is still something sitting right there,
        # I treat that waypoint as blocked instead of waiting on a timeout.
        self.waypoint_block_check_dist = 0.6
        
        # ---------------- OBSTACLE AVOIDANCE SETTINGS ----------------
        # These values are for the DistBug-style behaviour.
        # I split them out so I could tune obstacle behaviour separately from
        # waypoint tracking behaviour.
        self.front_obstacle_dist = 0.90       # once something is this close ahead, start avoiding
        self.emergency_stop_dist = 0.30       # never drive forward if something is dangerously close
        self.goal_clearance_dist = 1.20       # I only leave wall follow if there is enough free space ahead
        self.wall_target_dist = 0.75          # this is the distance I try to keep from the wall
        self.wall_kp = 1.2                    # proportional steering gain while following a wall
        self.follow_speed = 0.08              # slower speed near walls so it behaves more safely
        self.turn_speed = 0.65                # fixed turning speed while avoiding obstacles
        
        # ---------------- BLOCKED WAYPOINT SETTINGS ----------------
        # instead of giving up because of time, only say a waypoint is unreachable
        # when the robot gets near it and there is clearly an object sitting there.
        self.waypoint_block_check_dist = 0.6
        
        # ---------------- FAIL-SAFE SETTINGS ----------------
        # I added these because in simulation sometimes topics freeze or the robot
        # can get stuck against something, and I did not want it to keep blindly pushing.
        self.sensor_timeout = 5.0             # stop if scan/pose data goes stale
        self.stuck_timeout = 10.0              # stop if it has been “trying” to move for too long with no progress
        self.stuck_distance_threshold = 0.01  # if it moves less than this, I count that as basically stuck
        
        # ---------------- UNREACHABLE WAYPOINT SETTINGS ----------------
        # If the robot gets within this distance of the waypoint but it is still blocked by an obstacle, I assume it is unreachable
        # or not realistically reachable and move on to the next one.
        self.waypoint_block_check_dist = 0.6
        
        # ---------------- ROBOT STATE ----------------
        # These start as None until I get my first pose message.
        self.current_x = None
        self.current_y = None
        self.current_yaw = None
        
        # Lidar info gets filled in once scans start coming through.
        self.scan_ranges = []
        self.scan_angle_min = 0.0
        self.scan_angle_increment = 0.0
        self.scan_range_max = 0.0
        
        # track message times so I can tell if the sensor data is fresh.
        self.last_pose_time = None
        self.last_scan_time = None
        
        # ---------------- CONTROLLER MODES ----------------
        # GO_TO_POINT = normal waypoint driving
        # WALL_FOLLOW = obstacle avoidance mode
        # FIX_YAW = rotate in place to match final heading
        # STOPPED = fail-safe stop
        # DONE = finished all waypoints
        self.mode = "GO_TO_POINT"
        # Print mode changes once, not spam the terminal every loop.
        self.last_logged_mode = None
        # This stores how far I was from the goal when I first hit an obstacle.
        # I use it later to decide if I’ve actually made enough progress to leave the wall.
        self.hit_distance = None
        
        # Variables for stuck detection.
        self.prev_progress_x = None
        self.prev_progress_y = None
        self.stuck_start_time = None
        
        # Store the last command mostly for tracking/debugging.
        self.last_cmd_linear = 0.0
        self.last_cmd_angular = 0.0
        
        # ---------------- ROS PUB/SUB ----------------
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Subscribe to Gazebo world pose instead of odom because I wanted the controller
        # to use the actual Gazebo position directly.
        # origin for converting GPS to x,y
        self.origin_lat = None
        self.origin_lon = None

        self.gps_sub = self.create_subscription(
            NavSatFix, '/fix', self.gps_callback, 10)

        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        
        # Subscribe to lidar scan for obstacle detection.
        self.scan_sub = self.create_subscription(
            LaserScan,
            self.scan_topic,
            self.scan_callback,
            10
        )
        
        # Run the control loop every 0.1 seconds.
        # This felt fast enough to respond but not so fast that logs became unbearable.
        self.timer = self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info("DistBug controller with fail-safes started")
        self.get_logger().info(f"Using world pose topic: {self.world_pose_topic}")
        self.get_logger().info(f"Using robot pose index: {self.robot_pose_index}")
        self.get_logger().info(f"Using scan topic: {self.scan_topic}")
        self.log_current_waypoint()
        
    def log_current_waypoint(self):
        # Just a cleaner way to print the current target waypoint.
        self.get_logger().info(
            f"Waypoint {self.current_waypoint_index + 1}/{len(self.waypoints)} -> "
            f"x={self.goal_x:.2f}, y={self.goal_y:.2f}, yaw={self.goal_yaw:.2f}"
        )
        
    def advance_to_next_waypoint(self):
        # Move to the next waypoint once the current one is fully complete.
        self.current_waypoint_index += 1
        
        if self.current_waypoint_index >= len(self.waypoints):
            # If there are no more waypoints left, I switch to DONE and stop.
            self.mode = "DONE"
            self.stop_robot()
            self.get_logger().info("All waypoints completed successfully.")
            return
        
        # Otherwise load the next waypoint and go back to normal drive mode.
        self.goal_x, self.goal_y, self.goal_yaw = self.waypoints[self.current_waypoint_index]
        self.current_waypoint_start_time = self.get_clock().now()
        self.mode = "GO_TO_POINT"
        self.last_logged_mode = None
        self.hit_distance = None
        self.is_pausing = False
        self.pause_start_time = None
        self.reset_stuck_detector()
        
        self.get_logger().info("Moving to next waypoint.")
        self.log_current_waypoint()
        
    def mark_waypoint_unreachable(self, reason="Unknown reason"):
        # If the robot cannot reasonably reach this waypoint, log it and move on.
        if self.mode == "DONE" or self.current_waypoint_index >= len(self.waypoints):
            return

        self.stop_robot()
        self.get_logger().warn(
            f"Unable to reach waypoint {self.current_waypoint_index + 1}: {reason}"
        )
        
        self.get_logger().warn(
            f"Blocked target was x={self.goal_x:.2f}, y={self.goal_y:.2f}, yaw={self.goal_yaw:.2f}"
        )
        self.get_logger().warn("Unable to reach location. Moving to next waypoint.")
        self.advance_to_next_waypoint()
        
    def waypoint_is_blocked(self, distance_error):
        # I only care about this when I am already fairly near the waypoint.
        if distance_error > self.waypoint_block_check_dist:
            return False

        # Check the space in front and slightly to the front-left because
        # when the robot is curving around an obstacle, the blockage might not
        # be perfectly centred straight ahead anymore.
        front = self.get_sector_min(-20, 20)
        front_left = self.get_sector_min(20, 50)

        # If I am near the goal and something is still very close in the goal area,
        # I treat the waypoint as blocked.
        blocked_front = front < 0.30
        blocked_front_left = front_left < 0.25

        return blocked_front or blocked_front_left
    
    def gps_callback(self, msg):
        if self.origin_lat is None:
            # first fix becomes our origin
            self.origin_lat = msg.latitude
            self.origin_lon = msg.longitude
            self.get_logger().info(f"GPS origin set: {self.origin_lat}, {self.origin_lon}")

        R = 6371000
        dlat = math.radians(msg.latitude - self.origin_lat)
        dlon = math.radians(msg.longitude - self.origin_lon)
        self.current_x = dlon * R * math.cos(math.radians(self.origin_lat))
        self.current_y = dlat * R
        self.last_pose_time = self.get_clock().now()

    def odom_callback(self, msg):
        # use odom just for yaw since GPS has no heading
        q = msg.pose.pose.orientation
        self.last_pose_time = self.get_clock().now()
        self.current_yaw = quaternion_to_yaw(q.x, q.y, q.z, q.w)

    def gps_to_local(self, lat, lon):
        R = 6371000
        dlat = math.radians(lat - self.origin_lat)
        dlon = math.radians(lon - self.origin_lon)
        x = dlon * R * math.cos(math.radians(self.origin_lat))
        y = dlat * R
        return x, y
        
    def scan_callback(self, msg):
        # Store all the scan info I need for sector checks.
        self.scan_ranges = list(msg.ranges)
        self.scan_angle_min = msg.angle_min
        self.scan_angle_increment = msg.angle_increment
        self.scan_range_max = msg.range_max
        self.last_scan_time = self.get_clock().now()
        
    def stop_robot(self):
        # Send a zero Twist to stop movement.
        cmd = Twist()
        self.cmd_pub.publish(cmd)
        self.last_cmd_linear = 0.0
        self.last_cmd_angular = 0.0
        
    def set_stopped_mode(self, reason):
        # I made this helper so any fail-safe stop goes through one place.
        # That keeps it easier to debug why the robot stopped.
        if self.mode != "STOPPED":
            self.get_logger().warn(f"Fail-safe stop triggered: {reason}")
        self.mode = "STOPPED"
        self.stop_robot()
        
    def get_sector_min(self, start_deg, end_deg):
        # This checks a chosen angle sector of the lidar and returns the minimum
        # valid distance there. I use sectors instead of the whole scan because
        # I care about specific regions like directly ahead or to the left.
        if not self.scan_ranges or self.scan_angle_increment == 0.0:
            return float('inf')
        
        values = []
        
        for i, r in enumerate(self.scan_ranges):
            angle = self.scan_angle_min + i * self.scan_angle_increment
            angle_deg = math.degrees(angle)
            
            if start_deg <= angle_deg <= end_deg:
                # Ignore invalid/too-small values because those can mess up obstacle logic.
                if math.isfinite(r) and r > 0.05:
                    values.append(r)
                    
        if not values:
            return float('inf')
        
        return min(values)
    
    def obstacle_ahead(self):
        # Small front window to decide whether I should stop trying to go straight
        # to the goal and switch to avoidance.
        front = self.get_sector_min(-20, 20)
        return front < self.front_obstacle_dist
    
    def emergency_blocked(self):
        # Narrower front window for the emergency rule.
        # This is the “absolutely do not drive forward” check.
        front = self.get_sector_min(-15, 15)
        return front < self.emergency_stop_dist
    
    def path_to_goal_clear(self):
        # I made this leave condition a bit stricter on purpose.
        # It is not enough for the front to be clear, I also want front-left to
        # look decent so the robot does not leave the wall too early and hit again.
        front = self.get_sector_min(-20, 20)
        front_left = self.get_sector_min(20, 60)
        return front > self.goal_clearance_dist and front_left > 0.70
    
    def sensor_data_fresh(self):
        # This checks whether pose and scan messages are still updating.
        # If one topic freezes, I’d rather stop than keep driving on old data.
        now = self.get_clock().now()
        
        if self.last_pose_time is None or self.last_scan_time is None:
            return False
        
        pose_age = (now - self.last_pose_time).nanoseconds / 1e9
        scan_age = (now - self.last_scan_time).nanoseconds / 1e9
        
        return pose_age <= self.sensor_timeout and scan_age <= self.sensor_timeout
    
    def reset_stuck_detector(self):
        # Whenever the mode changes or I start fresh behaviour, I reset the
        # progress tracking so the stuck detector does not carry old info across modes.
        self.prev_progress_x = self.current_x
        self.prev_progress_y = self.current_y
        self.stuck_start_time = None
        
    def check_stuck(self, cmd_linear):
        # Idea here: if I’m commanding forward motion but the robot position is
        # barely changing, then it is probably stuck on something.
        if self.current_x is None or self.current_y is None:
            return False
        
        now = self.get_clock().now()
        
        # Only use stuck detection when I’m actually trying to move forward.
        # If I’m rotating in place, low translation is normal.
        if cmd_linear < 0.03:
            self.prev_progress_x = self.current_x
            self.prev_progress_y = self.current_y
            self.stuck_start_time = None
            return False
        
        if self.prev_progress_x is None or self.prev_progress_y is None:
            self.prev_progress_x = self.current_x
            self.prev_progress_y = self.current_y
            return False
        
        movement = math.hypot(
            self.current_x - self.prev_progress_x,
            self.current_y - self.prev_progress_y
        )
        
        if movement < self.stuck_distance_threshold:
            # If movement stays tiny for long enough, I count that as stuck.
            if self.stuck_start_time is None:
                self.stuck_start_time = now
            else:
                stuck_time = (now - self.stuck_start_time).nanoseconds / 1e9
                if stuck_time > self.stuck_timeout:
                    return True
        else:
            # If I do see proper motion, reset the stuck timer.
            self.prev_progress_x = self.current_x
            self.prev_progress_y = self.current_y
            self.stuck_start_time = None
            
        return False
    
    def control_loop(self):
        # ---------------- MODE 1: DONE ----------------
        if self.mode == "DONE":
            # All waypoints complete, so just keep the robot stopped.
            self.stop_robot()
            return
        
        # If taking too long → skip waypoint
        elapsed = (self.get_clock().now() - self.current_waypoint_start_time).nanoseconds / 1e9
            
        # If pose has not arrived yet, I stop rather than guessing.
        if self.current_x is None or self.current_y is None or self.current_yaw is None:
            self.stop_robot()
            return
        
        # Same idea for scan data.
        if not self.scan_ranges:
            self.stop_robot()
            return
        
        # Fail-safe for stale data.
        if not self.sensor_data_fresh():
            self.stop_robot()
            self.get_logger().warn("Pose or scan data timed out")
            return
        
        # Convert GPS waypoint (lat, lon) → local x,y
        goal_x, goal_y = self.gps_to_local(self.goal_x, self.goal_y)

        dx = goal_x - self.current_x
        dy = goal_y - self.current_y

        
        distance_error = math.hypot(dx, dy)
        target_heading = math.atan2(dy, dx)
        heading_error = wrap_to_pi(target_heading - self.current_yaw)
        final_yaw_error = wrap_to_pi(self.goal_yaw - self.current_yaw)
                
                
        # These sector checks are used a lot, so I compute them once per loop.
        front = self.get_sector_min(-20, 20)
        front_left = self.get_sector_min(20, 70)
        left = self.get_sector_min(70, 110)
        
        cmd = Twist()
        
        # ---------------- MODE 2: DRIVE TO TARGET ----------------
        if self.mode == "GO_TO_POINT":
            
            # First priority: if I’m close enough to the waypoint position,
            # stop chasing position and switch to final yaw alignment.
            if distance_error <= self.position_tolerance:
                self.mode = "FIX_YAW"
                self.reset_stuck_detector()
                self.get_logger().info(
                    f"Reached waypoint {self.current_waypoint_index + 1} position. Fixing final orientation."
                )
                
            
            elif self.obstacle_ahead():
                self.mode = "WALL_FOLLOW"
                self.hit_distance = distance_error
                self.stop_robot()
                self.reset_stuck_detector()
                self.get_logger().info(
                    f"Obstacle detected. Switching to WALL_FOLLOW. "
                    f"Hit distance = {self.hit_distance:.2f}"
                )
                return
            
            else:
                # If heading is pretty wrong, I rotate first before driving.
                # I did this because trying to drive and turn a lot at once made
                # it cut corners badly and look messy near waypoints.
                if abs(heading_error) > 0.35:
                    cmd.linear.x = 0.0
                    cmd.angular.z = clamp(
                        self.k_angular * heading_error,
                        -self.max_angular_speed,
                        self.max_angular_speed
                    )
                else:
                    # Once the heading is reasonable, I drive forward.
                    # I slow down near the goal so it does not overshoot as much.
                    if distance_error < 0.5:
                        speed = self.k_linear * distance_error * 0.5
                    else:
                        speed = self.k_linear * distance_error
                        
                    cmd.linear.x = clamp(speed, 0.0, self.max_linear_speed)
                    
                    # Extra slowdown if something is getting a bit close ahead,
                    # even if it is not yet close enough to fully switch modes.
                    if front < 1.20:
                        cmd.linear.x = min(cmd.linear.x, 0.10)
                        
                    # Keep a minimum forward speed so the robot actually moves
                    # instead of hovering at useless tiny values.
                    if 0.0 < cmd.linear.x < self.min_linear_speed:
                        cmd.linear.x = self.min_linear_speed
                        
                    # Still allow some steering while moving toward the goal.
                    cmd.angular.z = clamp(
                        self.k_angular * heading_error,
                        -self.max_angular_speed,
                        self.max_angular_speed
                    )
                    
        # ---------------- MODE 3: WALL FOLLOW ----------------
        elif self.mode == "WALL_FOLLOW":
            # If I have gone around the obstacle enough and the front is open again,
            # go back to normal waypoint chasing.
            if (
                self.hit_distance is not None
                and distance_error < (self.hit_distance - 0.20)
                and front > 1.0
            ):
                self.mode = "GO_TO_POINT"
                self.reset_stuck_detector()
                self.get_logger().info("Leaving wall and returning to GO_TO_POINT.")
                return

            # Hard safety rule: if something is way too close in front,
            # do not drive forward at all, just turn away.
            elif self.emergency_blocked():
                cmd.linear.x = 0.0
                cmd.angular.z = -self.turn_speed

            else:
                # If there is still something in front, prioritise turning right.
                if front < self.front_obstacle_dist:
                    cmd.linear.x = 0.0
                    cmd.angular.z = -self.turn_speed

                else:
                    # If the wall disappears on the left, I gently curve left
                    # to try to find it again.
                    if left == float('inf'):
                        cmd.linear.x = self.follow_speed
                        cmd.angular.z = 0.30
                    else:
                        # Standard wall following.
                        wall_error = self.wall_target_dist - left
                        cmd.linear.x = self.follow_speed
                        cmd.angular.z = clamp(
                            -self.wall_kp * wall_error,
                            -0.6,
                            0.6
                        )

                    # Extra front-left protection so it does not clip corners.
                    if front_left < 0.60:
                        cmd.linear.x = 0.0
                        cmd.angular.z = -self.turn_speed
                        
                        
        # ---------------- MODE 4: FIX FINAL YAW ----------------
        elif self.mode == "FIX_YAW":
            # Once I’m at the waypoint position, I separately fix the final heading.
            # I split this into its own mode because trying to fix position and yaw
            # at the exact same time made the finish less clean.
            if abs(final_yaw_error) > self.yaw_tolerance:
                cmd.linear.x = 0.0
                cmd.angular.z = clamp(
                    self.k_final_yaw * final_yaw_error,
                    -self.max_angular_speed,
                    self.max_angular_speed
                )
            else:
                # Only after both position and final yaw are good do I count the waypoint as done.
                self.stop_robot()
                
                # Print the final pose when I fully reach the waypoint (position + orientation)
                # Start pause if not already pausing
                if not self.is_pausing:
                    self.get_logger().info(
                        f"Final position: x={self.current_x:.2f}, y={self.current_y:.2f}, yaw={self.current_yaw:.2f}"
                    )
                    self.get_logger().info(
                        f"Waypoint {self.current_waypoint_index + 1} reached. Pausing for 10 seconds..."
                    )
                    self.is_pausing = True
                    self.pause_start_time = time.time()
                    return

                
                # Stay paused
                if time.time() - self.pause_start_time < self.pause_duration:
                    self.stop_robot()
                    return
                
                # Done pausing → move to next waypoint
                self.get_logger().info(
                    f"Moving to waypoint {self.current_waypoint_index + 2}"
                )
                
                self.is_pausing = False
                self.advance_to_next_waypoint()
                return
            
        # ---------------- MODE 5: STOPPED ----------------
        elif self.mode == "STOPPED":
            # Stay stopped until I manually restart or rerun the node.
            self.stop_robot()
            return
        
        
        # Final fail-safe check before publishing.
        if self.check_stuck(cmd.linear.x):
            self.mark_waypoint_unreachable("Robot appears stuck while commanded forward")
            return
        
        # Log ONLY when mode changes
        if self.mode != self.last_logged_mode:
            self.get_logger().info(
                f"[MODE] → {self.mode} | Waypoint {self.current_waypoint_index + 1}/{len(self.waypoints)}"
            )
            self.last_logged_mode = self.mode
        self.cmd_pub.publish(cmd)
        self.last_cmd_linear = cmd.linear.x
        self.last_cmd_angular = cmd.angular.z
        
        
def main(args=None):
    rclpy.init(args=args)
    node = WaypointController()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Make sure the robot stops cleanly if I kill the node.
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()
        
        
if __name__ == '__main__':
    main()
