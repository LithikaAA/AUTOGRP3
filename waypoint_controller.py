#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

"""
Multi-Waypoint Controller Overview

This node implements a simple autonomous waypoint controller for a mobile robot using ROS2.
The robot receives its current position and orientation from odometry and calculates the
error relative to a target waypoint (x, y, yaw). For Task 6, this was extended so the robot
can move through a list of waypoints instead of stopping after only one. The controller uses
a state-based approach with three modes: driving to the current waypoint position, correcting
final orientation, and stopping once all waypoints are completed. Proportional control is used
to generate linear and angular velocity commands, which are continuously updated in a timed
control loop.

Realism in this controller is introduced through several aspects. Firstly, all motion is
based on odometry feedback, which in real systems is subject to noise and drift, meaning the
robot cannot rely on perfect positioning. Secondly, velocity commands are clamped to realistic
limits, preventing unrealistically fast or instant movements. The controller also accounts for
angular wrapping, ensuring the robot takes the shortest rotation path similar to real robotic
systems. Additionally, the use of tolerances for position and orientation reflects real-world
behaviour where exact precision is not achievable. A slowdown region is also used near each
waypoint so the robot approaches more smoothly and reduces overshoot, which makes the behaviour
less idealised and more physically realistic.

The controller gains were selected to achieve a balance between responsiveness and stability.
The linear gain determines how aggressively the robot moves toward the waypoint, while the
angular gains control how quickly the robot corrects its heading and final orientation. Higher
gains result in faster responses but can lead to oscillations or overshoot, whereas lower gains
produce smoother but slower motion. The chosen values were tuned experimentally within the
simulation to ensure the robot approaches each waypoint efficiently without excessive turning
or instability. Separate gains were used for general heading correction and final orientation
to allow more precise alignment at each goal.
"""


def clamp(value, min_value, max_value):
    """Keep a value between a minimum and maximum."""
    return max(min_value, min(value, max_value))


def wrap_to_pi(angle):
    """Wrap any angle into the range [-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def quaternion_to_yaw(x, y, z, w):
    """Convert quaternion orientation into yaw angle."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class WaypointController(Node):
    def __init__(self):
        super().__init__('waypoint_controller')

        # ---------------- WAYPOINT LIST ----------------
        # Each waypoint is written as (x, y, final_yaw).
        # ----------- LOAD WAYPOINTS FROM FILE -----------
        self.waypoints = []

        file_path = "/mnt/c/Users/Lithi/Desktop/AUTO4408/Project 1/waypoints.txt"

        try:
            with open(file_path, "r") as f:
                for line in f:
                    # skip empty lines
                    if line.strip() == "":
                        continue

                    x, y, yaw = map(float, line.split())
                    self.waypoints.append((x, y, yaw))

            self.get_logger().info(f"Loaded {len(self.waypoints)} waypoints.")

        except Exception as e:
            self.get_logger().error(f"Failed to load waypoints: {e}")
        '''The robot does not exactly reach the waypoint position due to the use of a finite positional 
        tolerance and velocity-based control. Once within tolerance, the controller prioritises orientation 
        correction, which can introduce small positional drift due to non-ideal motion.'''

        # Track which waypoint we are currently driving to.
        self.current_waypoint_index = 0

        # Load the first waypoint as the active target.
        self.goal_x, self.goal_y, self.goal_yaw = self.waypoints[self.current_waypoint_index]

        if len(self.waypoints) == 0:
            self.get_logger().error("No waypoints loaded. Stopping node.")
            return
        
        # ---------------- CONTROL GAINS ----------------
        # Linear speed gain for driving to the point.
        self.k_linear = 0.6

        # Angular speed gain for turning toward the point.
        self.k_angular = 1.5

        # Angular speed gain for final orientation correction.
        self.k_final_yaw = 1.2

        # ---------------- SPEED LIMITS ----------------
        # Max forward speed.
        self.max_linear_speed = 0.30

        # Max turning speed.
        self.max_angular_speed = 1.0

        # Small minimum speed so it does not crawl too slowly near the goal.
        self.min_linear_speed = 0.05

        # ---------------- TOLERANCES ----------------
        # How close the robot needs to be to count as "at the point".
        self.position_tolerance = 0.10  # metres

        # How close the final angle needs to be.
        self.yaw_tolerance = 0.08       # radians (~4.6 deg)

        # ---------------- ROBOT STATE ----------------
        self.current_x = None
        self.current_y = None
        self.current_yaw = None

        # ---------------- CONTROLLER MODE ----------------
        # GO_TO_POINT = move to x, y
        # FIX_YAW = rotate to final heading
        # DONE = stop after all waypoints are finished
        self.mode = "GO_TO_POINT"

        # Publisher for robot velocity commands.
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscriber for odometry feedback.
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        # Timer for control loop.
        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info("Multi-waypoint controller started")
        self.log_current_waypoint()

    def log_current_waypoint(self):
        """Print the current target waypoint."""
        self.get_logger().info(
            f"Waypoint {self.current_waypoint_index + 1}/{len(self.waypoints)} -> "
            f"x={self.goal_x:.2f}, y={self.goal_y:.2f}, yaw={self.goal_yaw:.2f}"
        )

    def advance_to_next_waypoint(self):
        """Move to the next waypoint or finish if all are done."""
        self.current_waypoint_index += 1

        if self.current_waypoint_index >= len(self.waypoints):
            self.mode = "DONE"
            self.stop_robot()
            self.get_logger().info("All waypoints completed successfully.")
            return

        # Load the next waypoint and continue driving.
        self.goal_x, self.goal_y, self.goal_yaw = self.waypoints[self.current_waypoint_index]
        self.mode = "GO_TO_POINT"
        self.get_logger().info("Moving to next waypoint.")
        self.log_current_waypoint()

    def odom_callback(self, msg):
        """Update the robot pose from odometry."""
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        self.current_yaw = quaternion_to_yaw(q.x, q.y, q.z, q.w)

    def stop_robot(self):
        """Publish zero velocity to stop the robot."""
        cmd = Twist()
        self.cmd_pub.publish(cmd)

    def control_loop(self):
        """Main waypoint control logic."""
        if self.current_x is None or self.current_y is None or self.current_yaw is None:
            return

        # Position error relative to current target waypoint.
        dx = self.goal_x - self.current_x
        dy = self.goal_y - self.current_y

        # Straight-line distance to target point.
        distance_error = math.hypot(dx, dy)

        # Heading angle from robot to target point.
        target_heading = math.atan2(dy, dx)

        # Difference between where robot is facing and where it should face to reach the point.
        heading_error = wrap_to_pi(target_heading - self.current_yaw)

        # Difference between final required yaw and current yaw.
        final_yaw_error = wrap_to_pi(self.goal_yaw - self.current_yaw)

        cmd = Twist()

        # ---------------- MODE 1: DRIVE TO TARGET POSITION ----------------
        if self.mode == "GO_TO_POINT":

            if distance_error > self.position_tolerance:

                # If robot is facing too far away from the target direction,
                # rotate first before driving forward.
                if abs(heading_error) > 0.35:
                    cmd.linear.x = 0.0
                    cmd.angular.z = clamp(
                        self.k_angular * heading_error,
                        -self.max_angular_speed,
                        self.max_angular_speed
                    )
                else:
                    # Slow down near the waypoint so the robot does not rush it.
                    if distance_error < 0.5:
                        speed = self.k_linear * distance_error * 0.5
                    else:
                        speed = self.k_linear * distance_error

                    cmd.linear.x = clamp(
                        speed,
                        0.0,
                        self.max_linear_speed
                    )

                    # Keep a tiny minimum speed so it still moves properly.
                    if 0.0 < cmd.linear.x < self.min_linear_speed:
                        cmd.linear.x = self.min_linear_speed

                    # Drive forward and gently steer toward the point.
                    cmd.angular.z = clamp(
                        self.k_angular * heading_error,
                        -self.max_angular_speed,
                        self.max_angular_speed
                    )

            else:
                self.mode = "FIX_YAW"
                self.get_logger().info(
                    f"Reached waypoint {self.current_waypoint_index + 1} position. Now fixing final orientation."
                )

        # ---------------- MODE 2: FIX FINAL ORIENTATION ----------------
        elif self.mode == "FIX_YAW":

            if abs(final_yaw_error) > self.yaw_tolerance:
                cmd.linear.x = 0.0
                cmd.angular.z = clamp(
                    self.k_final_yaw * final_yaw_error,
                    -self.max_angular_speed,
                    self.max_angular_speed
                )
            else:
                self.stop_robot()
                self.get_logger().info(
                    f"Waypoint {self.current_waypoint_index + 1} reached successfully."
                )
                self.get_logger().info(
                    f"Pose: x={self.current_x:.2f}, y={self.current_y:.2f}, yaw={self.current_yaw:.2f}"
                )
                self.advance_to_next_waypoint()
                return

        # ---------------- MODE 3: DONE ----------------
        elif self.mode == "DONE":
            self.stop_robot()
            return

        # Publish command.
        self.cmd_pub.publish(cmd)

        # Print debug info so you can check what the robot is doing.
        self.get_logger().info(
            f"Mode: {self.mode} | "
            f"WP: {self.current_waypoint_index + 1}/{len(self.waypoints)} | "
            f"Current pose: ({self.current_x:.2f}, {self.current_y:.2f}, {self.current_yaw:.2f}) | "
            f"Distance error: {distance_error:.2f} | "
            f"Heading error: {heading_error:.2f} | "
            f"Final yaw error: {final_yaw_error:.2f}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = WaypointController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()