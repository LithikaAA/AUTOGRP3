#!/usr/bin/env python3

"""
LiDAR E-STOP -> moving obstacle detection

- Robot drives forward continuously
- Checks ALL directions for moving obstacles
- If a moving obstacle comes within 1m:
      -> robot e-stops

How moving detection works:
- We save the previous lidar scan
- We compare it to the current scan
- If multiple readings changed significantly, something moved
- If that moving thing is also within 1m, we stop

Topics used:
- Subscribes to: /scan
- Publishes to: /cmd_vel
"""

import math
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

# settings
stopdist = 1.0  # metres
fwdspeed = 0.2  # m/s

# upgraded threshold (more stable against noise)
movethres = 0.18  # metres


class LidarEstop(Node):

    def __init__(self):
        # create ROS2 node
        super().__init__("estoplidar")

        # stores whether a moving obstacle is currently detected
        self.obstacle_detected = False

        # stores the previous scan so we can compare
        self.prev_ranges = None

        # create publisher for robot velocity
        self.cmd_pub = self.create_publisher(
            Twist,
            "/cmd_vel",
            10
        )

        # subscribe to lidar scan topic
        self.create_subscription(
            LaserScan,
            "/scan",
            self.lidarcb,
            10
        )

        # timer: runs every 0.1 seconds (10 Hz)
        # continuously sends velocity commands
        self.create_timer(0.1, self.controloop)

        self.get_logger().info("LiDAR e-stop node started. Watching for moving obstacles...")

    # lidar callback, runs every time a new scan arrives
    def lidarcb(self, msg: LaserScan):

        curr = msg.ranges

        # first scan ever
        # nothing to compare against yet, just save and wait
        if self.prev_ranges is None:
            self.prev_ranges = curr
            return
        
        # make sure scan lengths match (they always should, but just in case)
        if len(curr) != len(self.prev_ranges):
            self.prev_ranges = curr
            return

        # check every ray in the scan (full 360 deg)
        obbnear = False

        # count how many rays agree motion exists
        movehits = 0

        for i in range(len(curr)):

            rangenow = curr[i]
            rangeprev = self.prev_ranges[i]

            # skip invalid readings (esp 0 readings)
            if (
                math.isnan(rangenow) or math.isinf(rangenow) or rangenow <= 0.1 or
                math.isnan(rangeprev) or math.isinf(rangeprev) or rangeprev <= 0.1
            ):
                continue

            # how much did this ray change since last scan?
            change = abs(rangenow - rangeprev)

            # ignore tiny jitter completely
            if change < 0.05:
                continue

            # detect meaningful motion near robot
            if change >= movethres and rangenow <= stopdist:
                movehits += 1

        # needs multiple rays to agree, so its not just like a particle moving
        if movehits >= 5:
            obbnear = True
            print(f"moving object detected (hits={movehits})")

        # # update state -> only log when it changes
        if obbnear and not self.obstacle_detected:
            self.get_logger().info("Moving obstacle detected within 1m. E-STOP triggered.")
            self.obstacle_detected = True

        elif not obbnear and self.obstacle_detected:
            self.get_logger().info("Path clear.")
            self.obstacle_detected = False

        # save current scan for next comparison
        self.prev_ranges = curr

    # control loop, runs on the timer
    def controloop(self):
        # e-stop: moving obstacle detected close by
        if self.obstacle_detected:
            self.sendvelo(0.0)

        # all clear, go forward
        else:
            self.sendvelo(fwdspeed)

    # sends a velocity command
    def sendvelo(self, speed: float):
        twist = Twist()
        twist.linear.x = speed
        self.cmd_pub.publish(twist)

# main function
def main():
    rclpy.init()
    node = LidarEstop()

    try:
        # keep node running
        rclpy.spin(node)

    except KeyboardInterrupt:
        # ctrl+c
        pass

    finally:
        # safety stop before shutdown
        node.sendvelo(0.0)
        # clean up
        node.destroy_node()
        rclpy.shutdown()

# run program
if __name__ == "__main__":
    main()