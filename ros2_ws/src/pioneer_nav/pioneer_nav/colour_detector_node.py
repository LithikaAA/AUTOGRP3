#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Pose

import cv2
import numpy as np
import json
import math
import os
import time


# ---------------- HSV COLOUR RANGES ----------------
# Red wraps around HSV, so it needs two ranges
RED_LOWER_1 = np.array([0, 120, 80], dtype=np.uint8)
RED_UPPER_1 = np.array([10, 255, 255], dtype=np.uint8)

RED_LOWER_2 = np.array([165, 120, 80], dtype=np.uint8)
RED_UPPER_2 = np.array([179, 255, 255], dtype=np.uint8)

YELLOW_LOWER = np.array([18, 120, 80], dtype=np.uint8)
YELLOW_UPPER = np.array([35, 255, 255], dtype=np.uint8)


class ColourDetectorNode(Node):
    def __init__(self):
        super().__init__("colour_detector")

        # Match the letter detector style
        self.declare_parameter("topic", "/oak/rgb/image_raw")
        self.declare_parameter("process_every_n_frames", 3)
        self.declare_parameter("min_area", 800.0)
        self.declare_parameter("require_mapping_state", False)

        self.topic = self.get_parameter("topic").value
        self.process_every = int(self.get_parameter("process_every_n_frames").value)
        self.min_area = float(self.get_parameter("min_area").value)
        self.require_mapping_state = bool(self.get_parameter("require_mapping_state").value)

        # Robot position, if /robot/pose exists
        self.robot_x = 0.0
        self.robot_y = 0.0

        # If require_mapping_state=False, detector runs straight away
        self.active = not self.require_mapping_state

        self.frame_count = 0
        self.last_photo_time = {}
        self.photo_cooldown_s = 5.0

        self.save_dir = os.path.expanduser("~/part3_logs/colour_detections")
        os.makedirs(self.save_dir, exist_ok=True)

        # Publishers
        self.det_pub = self.create_publisher(String, "/detections/colour", 10)

        # Subscribers
        self.sub = self.create_subscription(
            Image,
            self.topic,
            self.image_callback,
            10
        )

        self.create_subscription(String, "/robot_state", self.state_callback, 10)
        self.create_subscription(Pose, "/robot/pose", self.pose_callback, 10)

        self.get_logger().info(f"Colour detector listening on: {self.topic}")
        self.get_logger().info("Publishing colour detections on: /detections/colour")

        if self.require_mapping_state:
            self.get_logger().info("Waiting for /robot_state == MAPPING before detecting")
        else:
            self.get_logger().info("Running immediately, no /robot_state needed")

    # ---------------- CALLBACKS ----------------

    def state_callback(self, msg):
        self.active = msg.data == "MAPPING"

    def pose_callback(self, msg):
        self.robot_x = msg.position.x
        self.robot_y = msg.position.y

    def image_callback(self, msg):
        if not self.active:
            return

        self.frame_count += 1
        if self.frame_count % self.process_every != 0:
            return

        # Same image conversion style as the working letter detector
        frame = np.frombuffer(msg.data, dtype=np.uint8)
        frame = frame.reshape((msg.height, msg.width, -1))

        # OAK image should usually be BGR8
        bgr = frame.copy()
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        detections = []

        red = self.detect_colour(
            hsv,
            RED_LOWER_1,
            RED_UPPER_1,
            "red_obstacle",
            RED_LOWER_2,
            RED_UPPER_2
        )

        yellow = self.detect_colour(
            hsv,
            YELLOW_LOWER,
            YELLOW_UPPER,
            "yellow_obstacle"
        )

        if red is not None:
            detections.append(red)

        if yellow is not None:
            detections.append(yellow)

        annotated = bgr.copy()

        for detection in detections:
            label, contour, bbox = detection
            x, y, w, h = bbox

            area = cv2.contourArea(contour)
            cx = x + w / 2.0
            cy = y + h / 2.0

            bearing_rad = self.bearing_from_x(cx, msg.width)
            distance_m = self.estimate_distance_from_area(area)

            colour = (0, 0, 255) if "red" in label else (0, 255, 255)

            cv2.rectangle(
                annotated,
                (x, y),
                (x + w, y + h),
                colour,
                3
            )

            cv2.putText(
                annotated,
                f"{label} {distance_m:.2f}m",
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                colour,
                2
            )

            photo_path = self.save_photo_if_needed(
                annotated,
                label
            )

            msg_out = String()
            msg_out.data = json.dumps({
                "label": label,
                "center_x": cx,
                "center_y": cy,
                "area": area,
                "bearing_rad": bearing_rad,
                "bearing_deg": math.degrees(bearing_rad),
                "distance_m": distance_m,
                "robot_x": self.robot_x,
                "robot_y": self.robot_y,
                "photo_path": photo_path,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            })

            self.det_pub.publish(msg_out)

            self.get_logger().info(
                f"Detected {label}: area={area:.1f}, "
                f"bearing={math.degrees(bearing_rad):.1f} deg, "
                f"distance≈{distance_m:.2f} m"
            )

        # Save latest debug image every processed frame
        cv2.imwrite("/tmp/colour_detection_result.png", annotated)

    # ---------------- DETECTION HELPERS ----------------

    def detect_colour(self, hsv, lower1, upper1, label, lower2=None, upper2=None):
        mask = cv2.inRange(hsv, lower1, upper1)

        if lower2 is not None and upper2 is not None:
            mask2 = cv2.inRange(hsv, lower2, upper2)
            mask = cv2.bitwise_or(mask, mask2)

        # Clean up noisy dots
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        biggest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(biggest)

        if area < self.min_area:
            return None

        bbox = cv2.boundingRect(biggest)

        return label, biggest, bbox

    def bearing_from_x(self, x_px, image_width):
        # OAK-D horizontal FOV is roughly 71 degrees
        hfov_rad = math.radians(71.0)

        # Left side = negative, right side = positive
        norm = (x_px - image_width / 2.0) / (image_width / 2.0)

        return norm * (hfov_rad / 2.0)

    def estimate_distance_from_area(self, area_px):
        # This is only a rough fallback estimate because this version does not use depth.
        # Bigger area = closer object.
        if area_px <= 0:
            return float("inf")

        known_width_m = 0.30
        focal_px = 600.0

        return (known_width_m * focal_px) / math.sqrt(area_px)

    def save_photo_if_needed(self, frame, label):
        now = time.time()
        last = self.last_photo_time.get(label, 0)

        if now - last < self.photo_cooldown_s:
            return ""

        self.last_photo_time[label] = now

        filename = f"{label}_{time.strftime('%Y%m%d_%H%M%S')}.png"
        path = os.path.join(self.save_dir, filename)

        cv2.imwrite(path, frame)

        return path


def main(args=None):
    rclpy.init(args=args)
    node = ColourDetectorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()