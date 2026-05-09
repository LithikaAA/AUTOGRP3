#!/usr/bin/env python3

"""
Simple motion practice node (ROS2)

Sequence:
1. Forward
2. Turn right
3. Forward
4. Turn left
5. Spin 360
6. Backward
7. Stop
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class MoveItMoveIt(Node):

    def __init__(self):
        super().__init__("moveitmoveit")

        # Publisher to robot velocity
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # Step control (tracks where we are in sequence)
        self.step = 0

        # Timer runs every 0.1s
        self.create_timer(0.1, self.control_loop)

        # Time tracking for each action
        self.start_time = self.get_clock().now()

        self.get_logger().info("MoveItMoveIt node started")

    # control loop
    def control_loop(self):

        now = self.get_clock().now()
        elapsed = (now - self.start_time).nanoseconds / 1e9

        twist = Twist()

        # forward
        if self.step == 0:
            twist.linear.x = 0.2

            if elapsed > 3.0:
                self.next_step()

        # 1. right
        elif self.step == 1:
            twist.angular.z = -0.8

            if elapsed > 1.5:
                self.next_step()

        # 2. forward
        elif self.step == 2:
            twist.linear.x = 0.2

            if elapsed > 2.5:
                self.next_step()

        # 3. left
        elif self.step == 3:
            twist.angular.z = 0.8

            if elapsed > 1.5:
                self.next_step()

        # 4. spin 360
        elif self.step == 4:
            twist.angular.z = 1.2

            # approx 360 degrees (depends on robot, may need tuning)
            if elapsed > 3.2:
                self.next_step()

        # 5. backward
        elif self.step == 5:
            twist.linear.x = -0.2

            if elapsed > 2.5:
                self.next_step()

        # 6. stop
        elif self.step == 6:
            twist.linear.x = 0.0
            twist.angular.z = 0.0

        # Publish command
        self.cmd_pub.publish(twist)

    # next step
    def next_step(self):
        self.step += 1
        self.start_time = self.get_clock().now()

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
        # safety stop
        stop = Twist()
        node.cmd_pub.publish(stop)

        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()