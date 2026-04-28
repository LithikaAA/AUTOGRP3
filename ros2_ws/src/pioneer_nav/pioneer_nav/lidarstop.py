#!/usr/bin/env python3
"""
Bare-bones LiDAR stop node.
- Drives forward on /cmd_vel
- Stops when anything is within STOP_DISTANCE on /scan
- Then exits cleanly

Run:
    python3 lidar_stop.py
"""

import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

STOP_DISTANCE = 0.5   # metres
FORWARD_SPEED = 0.2   # m/s


class LidarStop(Node):

    def __init__(self):
        super().__init__("lidar_stop")
        self.stopped = False

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_subscription(LaserScan, "/scan", self.scan_cb, 10)

        # Drive forward immediately
        self.get_logger().info("Moving forward...")
        self.send_velocity(FORWARD_SPEED)

    def scan_cb(self, msg: LaserScan):
        if self.stopped:
            return

        # Find closest valid reading in the entire scan
        closest = min(
            (r for r in msg.ranges if not math.isnan(r) and not math.isinf(r)),
            default=float("inf")
        )

        if closest <= STOP_DISTANCE:
            self.get_logger().info(f"Obstacle at {closest:.2f} m — stopping.")
            self.send_velocity(0.0)
            self.stopped = True
            raise SystemExit  # clean shutdown

    def send_velocity(self, speed: float):
        twist = Twist()
        twist.linear.x = speed
        self.cmd_pub.publish(twist)


def main():
    rclpy.init()
    node = LidarStop()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.send_velocity(0.0)  # safety: ensure stop on exit
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
