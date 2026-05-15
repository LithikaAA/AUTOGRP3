#!/usr/bin/env python3

"""
Simple motion sequence node (ROS2)

Sequence:
1. Forward
2. Turn right
3. Forward
4. Turn left
5. Spin 360
6. Backward
7. Stop

Listens to /estop_status (Int8) from the lidar estop node:
    0 = all clear, run normally
    1 = warning, pause and wait (will resume when clear)
    2 = EMERGENCY STOP, halt forever, human must reset
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Int8

# mirror the same constants from estoplidar (we use estoptest for now) so its obvious what we're checking
clearstate = 0
warnstate  = 1
estopstate = 2


class MoveItMoveIt(Node):

    def __init__(self):
        super().__init__("moveitmoveit")

        self.cmdpub = self.create_publisher(Twist, "/cmd_vel", 10)

        # current status from the lidar estop node
        # starts as clear, gets updated whenever /estop_status arrives
        self.status = clearstate

        self.create_subscription(Int8, "/estop_status", self.estatuscb, 10)

        # step control
        self.step = 0
        self.starttime = self.get_clock().now()

        self.create_timer(0.1, self.controloop)

        self.get_logger().info("MoveItMoveIt node started")


    # estop status callback - just update our local status
    def estatuscb(self, msg: Int8):
        prev = self.status
        self.status = msg.data

        if self.status == estopstate and prev != estopstate:
            self.get_logger().info("EMERGENCY STOP received - halting permanently")

        elif self.status == warnstate and prev != warnstate:
            self.get_logger().info("Warning received - pausing sequence")

        elif self.status == clearstate and prev != clearstate:
            self.get_logger().info("All clear - resuming sequence")
            # reset step timer so we don't think time passed while we were paused
            self.starttime = self.get_clock().now()


    # control loop
    def controloop(self):

        # full estop - never move again, publish nothing
        if self.status == estopstate:
            return

        # warning - paused, publish nothing, estop node handles cmd_vel
        if self.status == warnstate:
            return

        # all clear, run the sequence
        now = self.get_clock().now()
        elapsed = (now - self.starttime).nanoseconds / 1e9

        twist = Twist()

        # forward
        if self.step == 0:
            twist.linear.x = 0.2
            if elapsed > 3.0:
                self.nextstep()

        # turn right
        elif self.step == 1:
            twist.angular.z = -0.8
            if elapsed > 1.5:
                self.nextstep()

        # forward
        elif self.step == 2:
            twist.linear.x = 0.2
            if elapsed > 2.5:
                self.nextstep()

        # turn left
        elif self.step == 3:
            twist.angular.z = 0.8
            if elapsed > 1.5:
                self.nextstep()

        # spin 360
        elif self.step == 4:
            twist.angular.z = 1.2
            # approx 360 deg (may need tuning per robot)
            if elapsed > 3.2:
                self.nextstep()

        # backward
        elif self.step == 5:
            twist.linear.x = -0.2
            if elapsed > 2.5:
                self.nextstep()

        # done
        elif self.step == 6:
            twist.linear.x = 0.0
            twist.angular.z = 0.0

        self.cmdpub.publish(twist)


    def nextstep(self):
        self.step += 1
        self.starttime = self.get_clock().now()
        self.get_logger().info(f"Switching to step {self.step}")


# main

def main():
    rclpy.init()
    node = MoveItMoveIt()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop = Twist()
        node.cmdpub.publish(stop)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
