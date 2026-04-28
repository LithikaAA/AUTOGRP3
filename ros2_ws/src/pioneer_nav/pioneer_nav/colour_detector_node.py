#!/usr/bin/env python3
"""
AUTO4508 Part 3 - Colour Detector Node (OAK-D V2)
Detects red and yellow obstacles, saves photos + logs location.

Publishes:
  /detections/colour   (std_msgs/String)  JSON with label, distance, bearing, location
  /detections/image    (sensor_msgs/Image) annotated frame for UI

Subscribes:
  /robot_state         (std_msgs/String)   only runs when MAPPING
  /robot/pose          (geometry_msgs/Pose) current robot position for logging

Saves photos to ~/part3_logs/colour_detections/
"""

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

# DepthAI — OAK-D V2
try:
    import depthai as dai
    DEPTHAI_AVAILABLE = True
except ImportError:
    DEPTHAI_AVAILABLE = False
    print("[colour_detector] WARNING: depthai not installed. Run: pip install depthai")


# =============================================================================
# HSV ranges for Part 3 targets (outdoor daylight)
# Tune these first if detection is off — run debug mode standalone
# =============================================================================

# Red has two ranges because it wraps around 0/180 in HSV
RED_LOWER_1  = np.array([0,   120,  80], dtype=np.uint8)
RED_UPPER_1  = np.array([10,  255, 255], dtype=np.uint8)
RED_LOWER_2  = np.array([165, 120,  80], dtype=np.uint8)
RED_UPPER_2  = np.array([179, 255, 255], dtype=np.uint8)

YELLOW_LOWER = np.array([18,  120,  80], dtype=np.uint8)
YELLOW_UPPER = np.array([35,  255, 255], dtype=np.uint8)

# Minimum contour area in pixels — increase if getting false positives outdoors
MIN_AREA = 800.0

# OAK-D V2 horizontal FOV in radians (~71 degrees)
HFOV_RAD = math.radians(71.0)

# How often to save a photo of the same object (seconds) — avoids duplicates
PHOTO_COOLDOWN_S = 5.0


# =============================================================================
# Detection result
# =============================================================================

@dataclass
class ColourDetection:
    label: str           # "red_obstacle" or "yellow_obstacle"
    center_x: float      # pixels
    center_y: float      # pixels
    area: float          # pixels squared
    bearing_rad: float   # negative = left, positive = right
    distance_m: float    # from OAK-D depth (or area estimate if depth unavailable)
    robot_x: float       # robot world position when detected
    robot_y: float
    photo_path: str      # path to saved photo
    timestamp: str


# =============================================================================
# OAK-D pipeline setup
# =============================================================================

def build_oakd_pipeline():
    """
    Build a DepthAI v3 pipeline that gives us:
    - RGB frame (640x480 for detection)
    - Depth map aligned to RGB (for real distance measurement)
    """
    pipeline = dai.Pipeline()

    # RGB camera
    cam_rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)

    # Stereo depth — v3 correct signature
    stereo = pipeline.create(dai.node.StereoDepth).build(
        autoCreateCameras=True,
        presetMode=dai.node.StereoDepth.PresetMode.FAST_ACCURACY,
        size=(640, 400),
        fps=15.0,
    )

    # Output queues
    q_rgb   = cam_rgb.requestOutput((640, 480), dai.ImgFrame.Type.BGR888p).createOutputQueue()
    q_depth = stereo.depth.createOutputQueue()

    return pipeline, q_rgb, q_depth


# =============================================================================
# Detection helpers
# =============================================================================

def apply_morphology(mask: np.ndarray, kernel_size: int = 7) -> np.ndarray:
    k = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def bearing_from_x(x_px: float, width: int) -> float:
    norm = (x_px - width / 2.0) / (width / 2.0)
    return norm * (HFOV_RAD / 2.0)


def depth_at_bbox(depth_frame: np.ndarray, bbox, margin: int = 10) -> float:
    """
    Get median depth in metres within a bounding box.
    Uses median to ignore noise/holes in depth map.
    Returns inf if depth unavailable.
    """
    if depth_frame is None:
        return float("inf")
    x, y, w, h = bbox
    x1 = max(0, x + margin)
    y1 = max(0, y + margin)
    x2 = min(depth_frame.shape[1], x + w - margin)
    y2 = min(depth_frame.shape[0], y + h - margin)
    roi = depth_frame[y1:y2, x1:x2]
    valid = roi[roi > 0]
    if valid.size == 0:
        return float("inf")
    # OAK-D depth is in mm, convert to metres
    return float(np.median(valid)) / 1000.0


def area_distance_estimate(area_px: float, known_width_m: float = 0.3,
                            focal_px: float = 600.0) -> float:
    """Fallback distance estimate when depth is unavailable."""
    if area_px <= 0:
        return float("inf")
    return (known_width_m * focal_px) / math.sqrt(area_px)


def detect_colour(hsv: np.ndarray, depth: np.ndarray,
                  lower1, upper1, label: str,
                  lower2=None, upper2=None) -> Optional[tuple]:
    """
    Detect a colour blob. Returns (contour, bbox, distance_m) or None.
    Supports two HSV ranges (needed for red).
    """
    mask = cv2.inRange(hsv, lower1, upper1)
    if lower2 is not None:
        mask |= cv2.inRange(hsv, lower2, upper2)

    mask = apply_morphology(mask, kernel_size=7)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < MIN_AREA:
        return None

    bbox = cv2.boundingRect(cnt)
    dist = depth_at_bbox(depth, bbox)
    if math.isinf(dist):
        dist = area_distance_estimate(area)

    return cnt, bbox, dist


def save_photo(frame: np.ndarray, bbox, label: str, dist: float,
               save_dir: str) -> str:
    os.makedirs(save_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(save_dir, f"{label}_{ts}.png")

    annotated = frame.copy()
    x, y, w, h = bbox
    colour = (0, 0, 255) if "red" in label else (0, 255, 255)
    cv2.rectangle(annotated, (x, y), (x + w, y + h), colour, 3)
    text = f"{label} {dist:.1f}m"
    cv2.putText(annotated, text, (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
    cv2.imwrite(path, annotated)
    return path


# =============================================================================
# ROS2 Node
# =============================================================================

class ColourDetectorNode(Node):
    def __init__(self):
        super().__init__("colour_detector")

        self.bridge = CvBridge()
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.active = False          # only run during MAPPING phase
        self.last_photo_time = {}    # label -> timestamp, for cooldown

        self.save_dir = os.path.expanduser("~/part3_logs/colour_detections")

        # Publishers
        self.det_pub = self.create_publisher(String, "/detections/colour", 10)
        self.img_pub = self.create_publisher(Image, "/detections/image", 10)

        # Subscribers
        self.create_subscription(String, "/robot_state", self.state_cb, 10)
        self.create_subscription(Pose, "/robot/pose", self.pose_cb, 10)

        # Start OAK-D
        if DEPTHAI_AVAILABLE:
            self._start_oakd()
        else:
            self.get_logger().warn("DepthAI not available — using fallback webcam")
            self._start_webcam_fallback()

        self.create_timer(0.1, self.process_frame)   # 10 Hz
        self.get_logger().info("Colour detector ready")

    # ------------------------------------------------------------------
    # Camera init
    # ------------------------------------------------------------------

    def _start_oakd(self):
        # v3 API — queues are returned directly from build_oakd_pipeline
        pipeline, self.q_rgb, self.q_depth = build_oakd_pipeline()
        self.device = dai.Device(pipeline)
        self.use_oakd = True
        self.get_logger().info("OAK-D pipeline started (depthai v3)")

    def _start_webcam_fallback(self):
        """Fallback for testing on laptop without OAK-D."""
        self.cap = cv2.VideoCapture(0)
        self.use_oakd = False
        self.get_logger().warn("Using webcam fallback — no real depth available")

    def _get_frames(self):
        """Returns (bgr_frame, depth_frame). depth_frame may be None."""
        if self.use_oakd:
            rgb_msg   = self.q_rgb.tryGet()
            depth_msg = self.q_depth.tryGet()
            if rgb_msg is None:
                return None, None
            bgr   = rgb_msg.getCvFrame()
            depth = depth_msg.getFrame() if depth_msg else None
            return bgr, depth
        else:
            ret, frame = self.cap.read()
            return (frame if ret else None), None

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------

    def state_cb(self, msg: String):
        self.active = (msg.data == "MAPPING")

    def pose_cb(self, msg: Pose):
        self.robot_x = msg.position.x
        self.robot_y = msg.position.y

    # ------------------------------------------------------------------
    # Main detection loop
    # ------------------------------------------------------------------

    def process_frame(self):
        if not self.active:
            return

        bgr, depth = self._get_frames()
        if bgr is None:
            return

        h, w = bgr.shape[:2]
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        results = []

        # --- Red detection ---
        red = detect_colour(hsv, depth,
                            RED_LOWER_1, RED_UPPER_1, "red_obstacle",
                            RED_LOWER_2, RED_UPPER_2)
        if red:
            cnt, bbox, dist = red
            results.append(("red_obstacle", cnt, bbox, dist))

        # --- Yellow detection ---
        yellow = detect_colour(hsv, depth,
                               YELLOW_LOWER, YELLOW_UPPER, "yellow_obstacle")
        if yellow:
            cnt, bbox, dist = yellow
            results.append(("yellow_obstacle", cnt, bbox, dist))

        # --- Process results ---
        annotated = bgr.copy()

        for label, cnt, bbox, dist in results:
            x, y, bw, bh = bbox
            cx = x + bw / 2.0
            cy = y + bh / 2.0
            bearing = bearing_from_x(cx, w)
            area = cv2.contourArea(cnt)

            # Draw on annotated frame
            colour = (0, 0, 255) if "red" in label else (0, 255, 255)
            cv2.rectangle(annotated, (x, y), (x + bw, y + bh), colour, 3)
            cv2.putText(annotated, f"{label} {dist:.1f}m",
                        (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)

            # Photo + log (with cooldown)
            now = time.time()
            last = self.last_photo_time.get(label, 0)
            photo_path = ""
            if now - last > PHOTO_COOLDOWN_S:
                photo_path = save_photo(bgr, bbox, label, dist, self.save_dir)
                self.last_photo_time[label] = now
                self.get_logger().info(
                    f"Detected {label} at {dist:.1f}m, bearing {math.degrees(bearing):.1f}° — photo: {photo_path}"
                )

            # Publish JSON detection
            det = ColourDetection(
                label=label,
                center_x=cx,
                center_y=cy,
                area=area,
                bearing_rad=bearing,
                distance_m=dist,
                robot_x=self.robot_x,
                robot_y=self.robot_y,
                photo_path=photo_path,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            )
            msg = String()
            msg.data = json.dumps(asdict(det))
            self.det_pub.publish(msg)

        # Publish annotated image for UI node
        try:
            img_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            self.img_pub.publish(img_msg)
        except Exception as e:
            self.get_logger().warn(f"Image publish failed: {e}")

    def destroy_node(self):
        if self.use_oakd and hasattr(self, "device"):
            self.device.close()
        elif hasattr(self, "cap"):
            self.cap.release()
        super().destroy_node()


# =============================================================================
# Standalone debug mode — run without ROS to tune HSV
# python3 colour_detector_node.py
# =============================================================================

def debug_standalone():
    print("Colour detector debug mode")
    print("  R = red mask    Y = yellow mask    Q = quit")

    if DEPTHAI_AVAILABLE:
        pipeline = build_oakd_pipeline()
        with dai.Device(pipeline) as device:
            q_rgb   = device.getOutputQueue("rgb",   maxSize=1, blocking=False)
            q_depth = device.getOutputQueue("depth", maxSize=1, blocking=False)
            _debug_loop_oakd(q_rgb, q_depth)
    else:
        cap = cv2.VideoCapture(0)
        _debug_loop_webcam(cap)
        cap.release()


def _debug_loop_oakd(q_rgb, q_depth):
    while True:
        rgb_pkt   = q_rgb.tryGet()
        depth_pkt = q_depth.tryGet()
        if rgb_pkt is None:
            continue
        bgr   = rgb_pkt.getCvFrame()
        depth = depth_pkt.getFrame() if depth_pkt else None
        if not _debug_show(bgr, depth):
            break


def _debug_loop_webcam(cap):
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        if not _debug_show(bgr, None):
            break


def _debug_show(bgr, depth) -> bool:
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    red    = detect_colour(hsv, depth, RED_LOWER_1, RED_UPPER_1, "red", RED_LOWER_2, RED_UPPER_2)
    yellow = detect_colour(hsv, depth, YELLOW_LOWER, YELLOW_UPPER, "yellow")

    display = bgr.copy()

    if red:
        _, bbox, dist = red
        x, y, bw, bh = bbox
        cv2.rectangle(display, (x, y), (x+bw, y+bh), (0, 0, 255), 2)
        cv2.putText(display, f"RED {dist:.1f}m", (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if yellow:
        _, bbox, dist = yellow
        x, y, bw, bh = bbox
        cv2.rectangle(display, (x, y), (x+bw, y+bh), (0, 255, 255), 2)
        cv2.putText(display, f"YELLOW {dist:.1f}m", (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Small red mask overlay top-left
    red_mask = cv2.inRange(hsv, RED_LOWER_1, RED_UPPER_1)
    red_mask |= cv2.inRange(hsv, RED_LOWER_2, RED_UPPER_2)
    small = cv2.resize(red_mask, (w//4, h//4))
    display[0:h//4, 0:w//4] = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
    cv2.putText(display, "red mask", (5, h//4 + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    cv2.imshow("Colour Detector Debug", display)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        cv2.destroyAllWindows()
        return False
    return True


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
    debug_standalone()