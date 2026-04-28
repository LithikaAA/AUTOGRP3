#!/usr/bin/env python3
"""
Bare-bones LiDAR stop node.
- Drives forward on /cmd_vel
- Stops when anything is within stopdist on /scan
- Then exits cleanly

Run:
-> ARIA, then LIDAR (can check scan too)
    ros2 run pioneer_nav lidarstop
"""

import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

stopdist  = 0.5
fwdspeed  = 0.2
conespan = math.radians(30) # ±30° cone either side of dead ahead


class LidarStop(Node):

    def __init__(self):
        super().__init__("lidar_stop")
        self.stopped = False

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_subscription(LaserScan, "/scan", self.scan_cb, 10)

        # Drive forward immediately
        self.get_logger().info("Moving forward...")
        self.send_velocity(fwdspeed)

    def scan_cb(self, msg: LaserScan):
        if self.stopped:
            return

        # Find closest valid reading within the forward cone only
        closest = float("inf")
        for i, r in enumerate(msg.ranges):
            if math.isnan(r) or math.isinf(r) or r <= 0.0:
                continue
            bearing = msg.angle_min + i * msg.angle_increment
            if abs(bearing) <= conespan:
                closest = min(closest, r)

        if closest <= stopdist:
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
