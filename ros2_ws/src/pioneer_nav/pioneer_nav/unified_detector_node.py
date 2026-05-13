#!/usr/bin/env python3
"""
unified_detector_node.py
------------------------
Combines Greek letter detection (ONNX CNN) with red/yellow colour object
detection into a single ROS2 node.

Publishes
---------
/detected_letter        std_msgs/String  — confirmed letter name
/detections/colour      std_msgs/String  — JSON blob for each colour detection
/detections/image       sensor_msgs/Image — annotated BGR frame

Subscribes
----------
/oak/rgb/image_raw      sensor_msgs/Image  — camera feed
/odom                   nav_msgs/Odometry  — robot pose + heading
/robot_state            std_msgs/String    — optional; "MAPPING" gates detection

Odometry logging
----------------
A detection is "confident" when the same label has been seen continuously
for at least `confident_duration_s` seconds (default 1.0).  At that moment
the current odometry (x, y, yaw_deg) is appended to
~/part3_logs/detections_log.jsonl  — one JSON record per line.
Photos are also saved to ~/part3_logs/colour_detections/ for colour hits.
"""

import json
import math
import os
import time

import cv2
import numpy as np
import onnxruntime as ort
import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

# ── HSV colour ranges ────────────────────────────────────────────────────────
RED_LOWER_1  = np.array([  0, 120,  80], dtype=np.uint8)
RED_UPPER_1  = np.array([ 10, 255, 255], dtype=np.uint8)
RED_LOWER_2  = np.array([165, 120,  80], dtype=np.uint8)
RED_UPPER_2  = np.array([179, 255, 255], dtype=np.uint8)
YELLOW_LOWER = np.array([ 18, 120,  80], dtype=np.uint8)
YELLOW_UPPER = np.array([ 35, 255, 255], dtype=np.uint8)

# OAK-D horizontal field of view
HFOV_RAD = math.radians(71.0)


def yaw_from_quaternion(q) -> float:
    """Extract yaw (radians) from a geometry_msgs/Quaternion."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class UnifiedDetectorNode(Node):

    def __init__(self):
        super().__init__("unified_detector")

        # ── Parameters ───────────────────────────────────────────────────────
        self.declare_parameter("topic",                    "/oak/rgb/image_raw")
        self.declare_parameter("brightness_threshold",     170)
        self.declare_parameter("confidence_threshold",     0.5)
        self.declare_parameter("process_every_n_frames",   3)
        self.declare_parameter("confirmations_required",   3)   # letter: frame count
        self.declare_parameter("min_colour_area",          800.0)
        self.declare_parameter("require_mapping_state",    False)
        self.declare_parameter("confident_duration_s",     1.0)  # seconds before logging odom
        self.declare_parameter("photo_cooldown_s",         5.0)

        topic               = self.get_parameter("topic").value
        self.bright_thresh  = self.get_parameter("brightness_threshold").value
        self.conf_thresh    = self.get_parameter("confidence_threshold").value
        self.process_every  = int(self.get_parameter("process_every_n_frames").value)
        self.confirms_req   = int(self.get_parameter("confirmations_required").value)
        self.min_colour_area = float(self.get_parameter("min_colour_area").value)
        self.require_mapping = bool(self.get_parameter("require_mapping_state").value)
        self.confident_dur  = float(self.get_parameter("confident_duration_s").value)
        self.photo_cooldown = float(self.get_parameter("photo_cooldown_s").value)

        # ── ONNX model ────────────────────────────────────────────────────────
        model_path = os.path.join(os.path.dirname(__file__), "greek_classifier.onnx")
        self.session    = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.get_logger().info("ONNX model loaded")

        # Load class map from greek_classes.txt next to this file
        self.classes: dict[int, str] = {}
        class_file = os.path.join(os.path.dirname(__file__), "greek_classes.txt")
        with open(class_file) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    self.classes[int(parts[0])] = parts[1]

        # ── State ─────────────────────────────────────────────────────────────
        self.active         = not self.require_mapping
        self.frame_count    = 0

        # Odometry
        self.robot_x   = 0.0
        self.robot_y   = 0.0
        self.robot_yaw = 0.0   # radians

        # Letter tracking
        self.letter_last_label  : str | None = None
        self.letter_frame_count : int        = 0
        self.letter_first_seen  : float      = 0.0   # wall time
        self.letter_logged      : bool       = False  # odom saved for this event?

        # Colour tracking  {label: {first_seen, logged, last_photo}}
        self.colour_state: dict[str, dict] = {
            "red_obstacle":    {"first_seen": 0.0, "logged": False, "last_photo": 0.0},
            "yellow_obstacle": {"first_seen": 0.0, "logged": False, "last_photo": 0.0},
        }

        # ── Logging dirs ──────────────────────────────────────────────────────
        self.log_dir   = os.path.expanduser("~/part3_logs")
        self.photo_dir = os.path.join(self.log_dir, "colour_detections")
        os.makedirs(self.photo_dir, exist_ok=True)
        self.log_file  = os.path.join(self.log_dir, "detections_log.jsonl")

        # ── Publishers ────────────────────────────────────────────────────────
        self.letter_pub = self.create_publisher(String, "/detected_letter",   10)
        self.colour_pub = self.create_publisher(String, "/detections/colour", 10)
        self.image_pub  = self.create_publisher(Image,  "/detections/image",  10)

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(Image,    topic,          self.image_callback, 10)
        self.create_subscription(Odometry, "/odom",        self.odom_callback,  10)
        self.create_subscription(String,   "/robot_state", self.state_callback, 10)

        self.get_logger().info(f"Unified detector listening on: {topic}")
        if self.require_mapping:
            self.get_logger().info("Waiting for /robot_state == MAPPING")

    # ── ROS callbacks ────────────────────────────────────────────────────────

    def state_callback(self, msg: String):
        self.active = (msg.data == "MAPPING")

    def odom_callback(self, msg: Odometry):
        self.robot_x   = msg.pose.pose.position.x
        self.robot_y   = msg.pose.pose.position.y
        self.robot_yaw = yaw_from_quaternion(msg.pose.pose.orientation)

    def image_callback(self, msg: Image):
        if not self.active:
            return

        self.frame_count += 1
        if self.frame_count % self.process_every != 0:
            return

        # Convert raw message to numpy BGR
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            (msg.height, msg.width, -1)
        )
        bgr  = frame.copy()
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        now = time.monotonic()

        # ── Letter detection ──────────────────────────────────────────────────
        self._process_letter(gray, bgr, now)

        # ── Colour detection ──────────────────────────────────────────────────
        self._process_colour(hsv, bgr, msg.width, now)

        # ── Publish annotated image ───────────────────────────────────────────
        self._publish_image(bgr, msg)

        # Debug dump
        cv2.imwrite("/tmp/unified_detection.png", bgr)

    # ── Letter pipeline ───────────────────────────────────────────────────────

    def _process_letter(self, gray: np.ndarray, bgr: np.ndarray, now: float):
        region_rect = self.find_sign_region(gray)

        if region_rect is None:
            # Reset tracking — object gone
            self.letter_last_label  = None
            self.letter_frame_count = 0
            self.letter_logged      = False
            return

        x, y, cw, ch = region_rect
        cv2.rectangle(bgr, (x, y), (x + cw, y + ch), (255, 0, 0), 2)

        region = gray[y:y + ch, x:x + cw]
        letter = self.extract_letter(region)
        if letter.size == 0:
            return

        name, confidence = self.classify(letter)

        if confidence <= self.conf_thresh:
            self.letter_last_label  = None
            self.letter_frame_count = 0
            self.letter_logged      = False
            return

        # Update frame-count streak
        if name == self.letter_last_label:
            self.letter_frame_count += 1
        else:
            self.letter_last_label  = name
            self.letter_frame_count = 1
            self.letter_first_seen  = now
            self.letter_logged      = False

        # Publish once confirmed (frame count threshold — fast feedback)
        if self.letter_frame_count >= self.confirms_req:
            self.get_logger().info(f"Letter: {name}  conf={confidence:.3f}")
            out = String()
            out.data = name
            self.letter_pub.publish(out)
            cv2.rectangle(bgr, (x, y), (x + cw, y + ch), (0, 255, 0), 2)
            cv2.putText(
                bgr, f"{name} ({confidence:.2f})",
                (x, max(y - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
            )

        # Log odometry once after confident_duration_s
        elapsed = now - self.letter_first_seen
        if elapsed >= self.confident_dur and not self.letter_logged:
            self._log_detection("letter", name, confidence)
            self.letter_logged = True

    # ── Colour pipeline ───────────────────────────────────────────────────────

    def _process_colour(
        self, hsv: np.ndarray, bgr: np.ndarray, img_width: int, now: float
    ):
        candidates = [
            self._detect_colour(hsv, RED_LOWER_1, RED_UPPER_1,
                                "red_obstacle",    RED_LOWER_2, RED_UPPER_2),
            self._detect_colour(hsv, YELLOW_LOWER, YELLOW_UPPER,
                                "yellow_obstacle"),
        ]

        seen_labels = set()

        for result in candidates:
            if result is None:
                continue

            label, contour, (bx, by, bw, bh) = result
            seen_labels.add(label)

            area  = cv2.contourArea(contour)
            cx_px = bx + bw / 2.0
            bearing_rad = self._bearing_from_x(cx_px, img_width)
            distance_m  = self._estimate_distance(area)

            # Annotate
            colour = (0, 0, 255) if "red" in label else (0, 255, 255)
            cv2.rectangle(bgr, (bx, by), (bx + bw, by + bh), colour, 3)
            cv2.putText(
                bgr, f"{label} {distance_m:.2f}m",
                (bx, max(by - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2,
            )

            # Publish JSON
            payload = String()
            payload.data = json.dumps({
                "label":       label,
                "center_x":    cx_px,
                "center_y":    by + bh / 2.0,
                "area":        area,
                "bearing_rad": bearing_rad,
                "bearing_deg": math.degrees(bearing_rad),
                "distance_m":  distance_m,
                "robot_x":     self.robot_x,
                "robot_y":     self.robot_y,
                "robot_yaw_deg": math.degrees(self.robot_yaw),
                "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%S"),
            })
            self.colour_pub.publish(payload)

            # Confidence timing and odom logging
            state = self.colour_state[label]
            if state["first_seen"] == 0.0:
                state["first_seen"] = now

            elapsed = now - state["first_seen"]
            if elapsed >= self.confident_dur and not state["logged"]:
                self._log_detection(label, label, confidence=1.0,
                                    extra={"distance_m": distance_m,
                                           "bearing_deg": math.degrees(bearing_rad)})
                state["logged"] = True
                self.get_logger().info(
                    f"Confident {label}: odom logged  "
                    f"x={self.robot_x:.2f} y={self.robot_y:.2f} "
                    f"yaw={math.degrees(self.robot_yaw):.1f}°"
                )

            # Save photo (rate limited)
            if now - state["last_photo"] >= self.photo_cooldown:
                fname = f"{label}_{time.strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(os.path.join(self.photo_dir, fname), bgr)
                state["last_photo"] = now
                self.get_logger().info(f"Saved photo: {fname}")

        # Reset labels that have disappeared
        for label, state in self.colour_state.items():
            if label not in seen_labels:
                state["first_seen"] = 0.0
                state["logged"]     = False

    # ── Odom logger ───────────────────────────────────────────────────────────

    def _log_detection(
        self, kind: str, name: str, confidence: float, extra: dict | None = None
    ):
        record = {
            "type":        kind,
            "name":        name,
            "confidence":  round(confidence, 4),
            "robot_x":     round(self.robot_x, 4),
            "robot_y":     round(self.robot_y, 4),
            "robot_yaw_deg": round(math.degrees(self.robot_yaw), 2),
            "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        if extra:
            record.update(extra)

        with open(self.log_file, "a") as f:
            f.write(json.dumps(record) + "\n")

        self.get_logger().info(
            f"Logged: {kind}/{name}  "
            f"x={self.robot_x:.2f} y={self.robot_y:.2f} "
            f"yaw={math.degrees(self.robot_yaw):.1f}°"
        )

    # ── Image publisher ───────────────────────────────────────────────────────

    def _publish_image(self, bgr: np.ndarray, original_msg: Image):
        out = Image()
        out.header         = original_msg.header
        out.height         = bgr.shape[0]
        out.width          = bgr.shape[1]
        out.encoding       = "bgr8"
        out.is_bigendian   = 0
        out.step           = bgr.shape[1] * 3
        out.data           = bgr.tobytes()
        self.image_pub.publish(out)

    # ── Letter helpers (unchanged from working version) ───────────────────────

    def find_sign_region(self, gray: np.ndarray):
        h, w   = gray.shape
        img_cx = w // 2
        img_cy = h // 2

        _, bright = cv2.threshold(gray, self.bright_thresh, 255, cv2.THRESH_BINARY)
        edges     = cv2.Canny(gray, 50, 150)

        k1              = np.ones((20, 20), np.uint8)
        bright_dilated  = cv2.dilate(bright, k1)
        edges_in_bright = cv2.bitwise_and(edges, bright_dilated)

        k2     = np.ones((15, 15), np.uint8)
        filled = cv2.dilate(edges_in_bright, k2)
        filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, k2)

        contours, _ = cv2.findContours(
            filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        best       = None
        best_score = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (h * w * 0.01 < area < h * w * 0.6):
                continue

            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect = cw / ch if ch > 0 else 0
            if not (0.3 < aspect < 2.5):
                continue

            region = gray[y:y + ch, x:x + cw]
            if region.mean() < 150:
                continue

            dark_ratio = np.sum(region < 100) / region.size
            if not (0.02 <= dark_ratio <= 0.5):
                continue

            cx_r  = x + cw // 2
            cy_r  = y + ch // 2
            dist  = math.hypot(cx_r - img_cx, cy_r - img_cy)
            max_d = math.hypot(img_cx, img_cy)
            score = (area / (h * w)) * (1 - 0.6 * dist / max_d)

            if score > best_score:
                best_score = score
                best       = (x, y, cw, ch)

        return best

    def extract_letter(self, region: np.ndarray) -> np.ndarray:
        IMG_SIZE = 64
        blur   = cv2.GaussianBlur(region, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 4,
        )

        conts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        lx = region.shape[1]
        ly = region.shape[0]
        lw = lh = 0

        for cnt in conts:
            if cv2.contourArea(cnt) > 20:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                lx = min(lx, bx)
                ly = min(ly, by)
                lw = max(lw, bx + bw)
                lh = max(lh, by + bh)

        lw -= lx
        lh -= ly

        letter = region[ly:ly + lh, lx:lx + lw] if (lw > 0 and lh > 0) else region

        scale = (IMG_SIZE - 8) / max(letter.shape)
        new_w = int(letter.shape[1] * scale)
        new_h = int(letter.shape[0] * scale)
        resized = cv2.resize(letter, (new_w, new_h))

        result              = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.uint8) * 255
        y_off               = (IMG_SIZE - new_h) // 2
        x_off               = (IMG_SIZE - new_w) // 2
        result[y_off:y_off + new_h, x_off:x_off + new_w] = resized
        return result

    def classify(self, letter_region: np.ndarray):
        inp     = letter_region.astype(np.float32) / 255.0
        inp     = inp[np.newaxis, np.newaxis, :, :]
        outputs = self.session.run(None, {self.input_name: inp})
        probs   = outputs[0][0]
        probs   = np.exp(probs - probs.max())
        probs  /= probs.sum()
        class_id   = int(np.argmax(probs))
        confidence = float(probs[class_id])
        return self.classes[class_id], confidence

    # ── Colour helpers ────────────────────────────────────────────────────────

    def _detect_colour(self, hsv, lower1, upper1, label, lower2=None, upper2=None):
        mask = cv2.inRange(hsv, lower1, upper1)
        if lower2 is not None:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower2, upper2))

        k = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        biggest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(biggest) < self.min_colour_area:
            return None

        return label, biggest, cv2.boundingRect(biggest)

    @staticmethod
    def _bearing_from_x(x_px: float, img_width: int) -> float:
        norm = (x_px - img_width / 2.0) / (img_width / 2.0)
        return norm * (HFOV_RAD / 2.0)

    @staticmethod
    def _estimate_distance(area_px: float) -> float:
        if area_px <= 0:
            return float("inf")
        return (0.30 * 600.0) / math.sqrt(area_px)


def main(args=None):
    rclpy.init(args=args)
    node = UnifiedDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
