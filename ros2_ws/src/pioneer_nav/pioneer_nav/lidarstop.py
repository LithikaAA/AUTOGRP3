#!/usr/bin/env python3

"""
LiDAR front obstacle detection node.

- Robot drives forward continuously
- Robot checks only the front LiDAR region
- If an obstacle is too close:
      -> robot stops
- If path becomes clear again:
      -> robot moves forward again

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
stopdist = 0.5 # metres
fwdspeed = 0.2 # m/s
# how many degrees left or right of front to check
frontcone = 30 # eg. 30 here means -30° to +30°


class LidarStop(Node):

    def __init__(self):
        # create ROS2 node
        super().__init__("lidarstop")

        # stores whether obstacle currently exists
        self.obstacle_detected = False

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

        # make timer: runs every 0.1 seconds (10 Hz)
        # this continuously sends velocity commands. may req continuous cmd_vel updates.
        self.create_timer(0.1, self.control_loop)

        self.get_logger().info("LiDAR stop node started.")

    # lidar callback - runs every time a new scan is received
    def lidarcb(self, msg: LaserScan):

        # list to store only front scan readings
        frontranges = []

        # check angles in front cone
        for angle in range(-frontcone, frontcone + 1):

            # convert degrees -> radians
            anglerad = math.radians(angle)

            # convert angle -> index in lidar array
            # index = (desired_angle - angle_min) / angle_increment
            index = int(
                (anglerad - msg.angle_min) / msg.angle_increment
            )

            # safety check: make sure index is valid
            if 0 <= index < len(msg.ranges):
                distance = msg.ranges[index]

                # ignore invalid values
                if (
                    not math.isnan(distance)
                    and not math.isinf(distance)
                ):
                    frontranges.append(distance)

        # if no valid readings exist
        if not frontranges:
            return

        # find closest object in front area
        closest = min(frontranges)

        # obstacle exist? t/f
        if closest <= stopdist:
            # obstacle detected
            self.obstacle_detected = True

            self.get_logger().info(
                f"Obstacle ahead: {closest:.2f} m"
            )

        else:
            # front is clear
            self.obstacle_detected = False

    # control loop (runs on the timer)
    def control_loop(self):
        
        # if obstacle ahead, stop
        if self.obstacle_detected:
            self.send_velocity(0.0)

        # else go forward
        # later this will be everything else
        else:
            self.send_velocity(fwdspeed)

    # send velocity functions: sends movement commands
    def send_velocity(self, speed: float):
        twist = Twist()
        # forward speed
        twist.linear.x = speed
        # publish command
        self.cmd_pub.publish(twist)

# main function
def main():
    # start ROS2
    rclpy.init()
    # create node
    node = LidarStop()

    try:
        # keep node running
        rclpy.spin(node)

    except KeyboardInterrupt:
        # ctrl+c
        pass

    finally:
        # safety stop before shutdown
        node.send_velocity(0.0)
        # clean up node
        node.destroy_node()
        # shutdown ROS2
        rclpy.shutdown()

# run program
if __name__ == "__main__":
    main()
