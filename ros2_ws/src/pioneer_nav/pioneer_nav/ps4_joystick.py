#!/usr/bin/env python3

# PS4 Joystick Controller
# Standalone node for PS4 controller input to cmd_vel

import math
import sys
import termios
import tty
import threading

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy


# =========================
# helper functions
# =========================

def clamp(value: float, low: float, high: float) -> float:
    """Clamp a value between low and high bounds."""
    return max(low, min(high, value))


# =========================
# main node
# =========================

class PS4JoystickController(Node):
    def __init__(self):
        super().__init__("ps4_joystick_controller")

        # -------------------------
        # params
        # -------------------------
        self.declare_parameter("joy_topic",     "/joy")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")

        # -------------------------------------------------
        # joystick mapping (PS4 defaults)
        #   axes[0]    = left stick horizontal  (turn)
        #   axes[1]    = left stick vertical    (forward)
        #   buttons[0] = X (cross)              -> not used in standalone
        #   buttons[1] = O (circle)             -> not used in standalone
        #   buttons[3] = Triangle               -> deadman (optional, default off)
        # -------------------------------------------------
        self.declare_parameter("joy_axis_linear",    1)
        self.declare_parameter("joy_axis_angular",   0)
        self.declare_parameter("joy_deadman_button", 3)   # Triangle
        self.declare_parameter("use_deadman",        False)  # Enable deadman switch

        # control tuning
        self.declare_parameter("max_linear_speed",  0.5)
        self.declare_parameter("max_angular_speed", 1.0)

        # read params
        joy_topic     = self.get_parameter("joy_topic").value
        cmd_vel_topic = self.get_parameter("cmd_vel_topic").value

        self.joy_axis_linear   = int(self.get_parameter("joy_axis_linear").value)
        self.joy_axis_angular  = int(self.get_parameter("joy_axis_angular").value)
        self.joy_deadman_button = int(self.get_parameter("joy_deadman_button").value)
        self.use_deadman       = bool(self.get_parameter("use_deadman").value)

        self.max_linear_speed  = float(self.get_parameter("max_linear_speed").value)
        self.max_angular_speed = float(self.get_parameter("max_angular_speed").value)

        # -------------------------
        # state
        # -------------------------
        self.manual_linear   = 0.0
        self.manual_angular  = 0.0
        self.deadman_pressed = not self.use_deadman  # Default ON if deadman not used

        # -------------------------
        # pubs / subs
        # -------------------------
        self.cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)

        self.create_subscription(Joy, joy_topic, self.joy_callback, 10)

        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info("PS4 Joystick controller started")
        if self.use_deadman:
            self.get_logger().info(f"Hold Triangle (button {self.joy_deadman_button}) to enable movement")
        else:
            self.get_logger().info("Deadman switch disabled - joystick will control movement directly")
        self.get_logger().info(f"Linear axis: {self.joy_axis_linear}, Angular axis: {self.joy_axis_angular}")
        self.get_logger().info("Keyboard: w/s/q/e/x for manual control, ESC/d to toggle deadman")

        # Start keyboard fallback thread
        self._keyboard_thread = threading.Thread(target=self.keyboard_thread, daemon=True)
        self._keyboard_thread.start()

    # =========================
    # callbacks
    # =========================

    def joy_callback(self, msg: Joy):
        """
        PS4 controller input handling.

        PS4 default mapping:
            axes[1]    = left stick up/down    (forward/back)
            axes[0]    = left stick left/right (turn)
            buttons[3] = Triangle              -> deadman (if enabled)
        """
        n_axes    = len(msg.axes)
        n_buttons = len(msg.buttons)

        # guard: log clearly if axis indices are out of range
        max_axis = max(self.joy_axis_linear, self.joy_axis_angular)
        if max_axis >= n_axes:
            self.get_logger().warn(
                f"Axis index {max_axis} out of range (controller has {n_axes} axes). "
                f"Run 'ros2 topic echo /joy' to check available axes.",
                throttle_duration_sec=5.0
            )
            return

        # Update deadman state if enabled
        if self.use_deadman:
            if self.joy_deadman_button < n_buttons:
                self.deadman_pressed = bool(msg.buttons[self.joy_deadman_button])
            else:
                self.get_logger().warn(
                    f"Deadman button {self.joy_deadman_button} out of range "
                    f"(controller has {n_buttons} buttons).",
                    throttle_duration_sec=5.0
                )

        # Extract and clamp joystick input
        lin_axis = msg.axes[self.joy_axis_linear]  if self.joy_axis_linear  < len(msg.axes) else 0.0
        ang_axis = msg.axes[self.joy_axis_angular] if self.joy_axis_angular < len(msg.axes) else 0.0

        self.manual_linear  = clamp(lin_axis, -1.0, 1.0) * self.max_linear_speed
        self.manual_angular = clamp(ang_axis, -1.0, 1.0) * self.max_angular_speed

    # =========================
    # keyboard thread (fallback)
    # =========================

    def keyboard_thread(self):
        """Background keyboard fallback - useful for testing or when controller unavailable."""
        settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            self.get_logger().info("Keyboard input thread started (w/a/s/d to move, ESC to exit)")
            while rclpy.ok():
                key = sys.stdin.read(1)
                if key == '\x1b':  # ESC
                    break
                elif key == 'd':
                    if self.use_deadman:
                        self.deadman_pressed = not self.deadman_pressed
                        self.get_logger().info(
                            f"Keyboard: deadman={'ON' if self.deadman_pressed else 'OFF'}"
                        )
                elif key == 'w':
                    self.manual_linear  =  0.3
                    self.manual_angular =  0.0
                elif key == 's':
                    self.manual_linear  = -0.3
                    self.manual_angular =  0.0
                elif key == 'q':
                    self.manual_linear  =  0.0
                    self.manual_angular =  0.5
                elif key == 'e':
                    self.manual_linear  =  0.0
                    self.manual_angular = -0.5
                elif key == 'x':
                    self.manual_linear  = 0.0
                    self.manual_angular = 0.0
        except Exception as e:
            self.get_logger().warn(f"Keyboard thread error: {e}")
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)

    # =========================
    # control loop
    # =========================

    def control_loop(self):
        """Main control loop - publishes cmd_vel based on joystick input."""
        # Check deadman before publishing
        if self.use_deadman and not self.deadman_pressed:
            self.publish_cmd(0.0, 0.0)
        else:
            self.publish_cmd(self.manual_linear, self.manual_angular)

    def publish_cmd(self, linear_x: float, angular_z: float):
        """Publish Twist command to cmd_vel topic."""
        msg = Twist()
        msg.linear.x  = linear_x
        msg.angular.z = angular_z
        self.cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PS4JoystickController()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
