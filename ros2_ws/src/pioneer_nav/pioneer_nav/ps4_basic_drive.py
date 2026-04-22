#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class Ps4BasicDrive(Node):
    def __init__(self):
        super().__init__('ps4_basic_drive')

        self.declare_parameter('joy_topic', '/joy')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel_manual')
        self.declare_parameter('axis_linear', 1)
        self.declare_parameter('axis_angular', 0)
        self.declare_parameter('enable_button', 7)
        self.declare_parameter('turbo_button', 1)
        self.declare_parameter('require_enable_button', False)
        self.declare_parameter('linear_scale', 0.25)
        self.declare_parameter('angular_scale', 0.8)
        self.declare_parameter('linear_turbo_scale', 0.45)
        self.declare_parameter('angular_turbo_scale', 1.2)

        joy_topic = self.get_parameter('joy_topic').value
        cmd_vel_topic = self.get_parameter('cmd_vel_topic').value

        self.axis_linear = int(self.get_parameter('axis_linear').value)
        self.axis_angular = int(self.get_parameter('axis_angular').value)
        self.enable_button = int(self.get_parameter('enable_button').value)
        self.turbo_button = int(self.get_parameter('turbo_button').value)
        self.require_enable_button = bool(
            self.get_parameter('require_enable_button').value
        )
        self.linear_scale = float(self.get_parameter('linear_scale').value)
        self.angular_scale = float(self.get_parameter('angular_scale').value)
        self.linear_turbo_scale = float(
            self.get_parameter('linear_turbo_scale').value
        )
        self.angular_turbo_scale = float(
            self.get_parameter('angular_turbo_scale').value
        )

        self.cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.create_subscription(Joy, joy_topic, self.joy_callback, 10)

        self.get_logger().info(f'Listening for PS4 input on {joy_topic}')
        self.get_logger().info(f'Publishing drive commands to {cmd_vel_topic}')

    def _axis(self, msg: Joy, index: int) -> float:
        if 0 <= index < len(msg.axes):
            return float(msg.axes[index])
        return 0.0

    def _button(self, msg: Joy, index: int) -> int:
        if 0 <= index < len(msg.buttons):
            return int(msg.buttons[index])
        return 0

    def joy_callback(self, msg: Joy):
        enabled = True
        if self.require_enable_button:
            enabled = self._button(msg, self.enable_button) == 1

        cmd = Twist()

        if enabled:
            turbo = self._button(msg, self.turbo_button) == 1
            linear_scale = self.linear_turbo_scale if turbo else self.linear_scale
            angular_scale = self.angular_turbo_scale if turbo else self.angular_scale

            linear = clamp(self._axis(msg, self.axis_linear), -1.0, 1.0)
            angular = clamp(self._axis(msg, self.axis_angular), -1.0, 1.0)

            cmd.linear.x = linear * linear_scale
            cmd.angular.z = angular * angular_scale

        self.cmd_pub.publish(cmd)

    def destroy_node(self):
        self.cmd_pub.publish(Twist())
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = Ps4BasicDrive()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
