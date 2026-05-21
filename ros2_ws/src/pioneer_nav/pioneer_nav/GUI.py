#!/usr/bin/env python3
"""
AUTO4508 Part 3 - Robot Monitor GUI  (merged)
=============================================
Runs on a LAPTOP connected to the same network as the robot.
Subscribes to ROS2 topics published by the robot's nodes and displays
everything in one window. It never publishes anything — it is read-only.

HOW THE NODES CONNECT TO THIS GUI:
-----------------------------------
  mission_manager.py  →  /robot_state        →  state badge + action label
  control_node.py     →  /robot/pose         →  map arrow + position display
  control_node.py     →  /arena_status       →  arena debug panel
  unified_detector    →  /detected_letter    →  detection log + map marker
  unified_detector    →  /detections/colour  →  detection log entry
  unified_detector    →  /detections/image   →  bottom-right annotated photo
  slam_toolbox        →  /map                →  map panel background
  mission_manager.py  →  /planned_path       →  path overlay on map
  OAK-D driver        →  /oak/rgb/image_raw  →  camera feed panel
  detections_log.jsonl (polled every 2 s)    →  map markers (one per obstacle)

Requirements:
    sudo apt install python3-pyqt5 python3-opencv
    Source ROS2 before running:
        source /opt/ros/humble/setup.bash
    Set matching ROS_DOMAIN_ID on robot and laptop:
        export ROS_DOMAIN_ID=42

Run:
    cd ros2_ws/src/pioneer_nav/pioneer_nav
    python3 GUI.py
"""

import sys
import json
import math
import os
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path as FilePath

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String
from geometry_msgs.msg import Pose
from nav_msgs.msg import OccupancyGrid, Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QFrame, QScrollArea, QSizePolicy, QProgressBar, QPushButton, QLineEdit
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QPointF
from PyQt5.QtGui import (
    QImage, QPixmap, QFont, QColor, QPainter,
    QPen, QBrush, QPolygonF
)


# ──────────────────────────────────────────────
#  CONSTANTS
# ──────────────────────────────────────────────
BG        = "#0d1117"
PANEL     = "#161b22"
BORDER    = "#30363d"
ACCENT    = "#58a6ff"
GREEN     = "#3fb950"
YELLOW    = "#d29922"
RED       = "#f85149"
TEXT      = "#e6edf3"
TEXT_DIM  = "#8b949e"

FONT_UI   = "Ubuntu Mono"
FONT_BODY = "DejaVu Sans"

# Path to the detection log written by unified_detector_node
DETECTION_LOG_PATH = os.path.expanduser("~/part3_logs/detections_log.jsonl")

# Directory where mission_manager saves e-stop event files
ESTOP_LOG_DIR = os.path.expanduser("~/part3_logs/estop_events")

# Directory where mission_manager saves rosbags
BAG_DIR = os.path.expanduser("~/part3_logs/bags")

# Directory where colour detector saves photos
PHOTO_DIR = os.path.expanduser("~/part3_logs/colour_detections")

# Directory for recorded journey videos
VIDEO_DIR = os.path.expanduser("~/part3_logs/videos")

# Minimum lidar range to ignore (filters out the robot's own body)
LIDAR_SELF_MASK_MIN_RANGE = float(os.environ.get("PIONEER_GUI_LIDAR_SELF_MASK_MIN_RANGE", "0.18"))

# Arena size in metres
ARENA_SIZE_M = float(os.environ.get("PIONEER_GUI_ARENA_SIZE_M", "15.0"))


# ──────────────────────────────────────────────
#  SIGNALS
# ──────────────────────────────────────────────
class Signals(QObject):
    camera_frame    = pyqtSignal(np.ndarray)           # raw frame → top-left camera panel
    colour_image    = pyqtSignal(np.ndarray)           # annotated frame → bottom-right panel
    robot_state     = pyqtSignal(str)
    robot_pose      = pyqtSignal(float, float, float)  # x, y, yaw (radians)
    letter_detected = pyqtSignal(str)                  # JSON string from /detected_letter
    colour_detected = pyqtSignal(dict)                 # JSON dict from /detections/colour
    map_updated     = pyqtSignal(object)               # OccupancyGrid message
    scan_updated    = pyqtSignal(object)               # LaserScan message
    path_updated    = pyqtSignal(object)               # Path message
    arena_updated   = pyqtSignal(dict)                 # JSON dict from /arena_status
    save_status     = pyqtSignal(str, bool)
    mission_command = pyqtSignal(str)


# ──────────────────────────────────────────────
#  ROS2 NODE
# ──────────────────────────────────────────────
class GUINode(Node):
    def __init__(self, signals: Signals):
        super().__init__("robot_gui")
        self.signals = signals
        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        live_map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        # Main camera feed
        self.create_subscription(Image,         "/oak/rgb/image_raw", self._cb_camera,        10)
        self.create_subscription(Image,         "/camera/image",      self._cb_camera,        10)

        # Annotated detection image — separate from the main feed
        self.create_subscription(Image,         "/detections/image",  self._cb_colour_image,  10)

        self.create_subscription(String,        "/robot_state",       self._cb_state,         10)
        self.create_subscription(Pose,          "/robot/pose",        self._cb_pose,          10)
        self.create_subscription(String,        "/detected_letter",   self._cb_letter,        10)
        self.create_subscription(String,        "/detections/colour", self._cb_colour,        10)
        self.create_subscription(OccupancyGrid, "/map",               self._cb_map,           map_qos)
        self.create_subscription(OccupancyGrid, "/map",               self._cb_map,           live_map_qos)
        self.create_subscription(LaserScan,     "/scan",              self._cb_scan,          10)
        self.create_subscription(Path,          "/planned_path",      self._cb_path,          10)
        self.create_subscription(String,        "/arena_status",      self._cb_arena,         10)
        self.command_pub = self.create_publisher(String, "/mission_command", 10)

    def publish_command(self, command: str):
        self.command_pub.publish(String(data=command))
        self.get_logger().info(f"GUI command published: {command}")

    def _cb_camera(self, msg):
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, -1))
        if msg.encoding != "bgr8":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.signals.camera_frame.emit(frame)

    def _cb_colour_image(self, msg):
        """Annotated detection frame — goes to the bottom-right photo panel."""
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, -1))
        if msg.encoding != "bgr8":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.signals.colour_image.emit(frame)

    def _cb_state(self, msg):
        self.signals.robot_state.emit(msg.data)

    def _cb_pose(self, msg):
        q    = msg.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw  = math.atan2(siny, cosy)
        self.signals.robot_pose.emit(msg.position.x, msg.position.y, yaw)

    def _cb_letter(self, msg):
        self.signals.letter_detected.emit(msg.data)

    def _cb_colour(self, msg):
        try:
            self.signals.colour_detected.emit(json.loads(msg.data))
        except Exception:
            pass

    def _cb_map(self, msg):
        self.signals.map_updated.emit(msg)

    def _cb_scan(self, msg):
        self.signals.scan_updated.emit(msg)

    def _cb_path(self, msg):
        self.signals.path_updated.emit(msg)

    def _cb_arena(self, msg):
        try:
            self.signals.arena_updated.emit(json.loads(msg.data))
        except Exception:
            pass


# ──────────────────────────────────────────────
#  UI HELPER FUNCTIONS
# ──────────────────────────────────────────────

def make_panel(title: str) -> tuple:
    frame = QFrame()
    frame.setStyleSheet(f"""
        QFrame {{
            background: {PANEL};
            border: 1px solid {BORDER};
            border-radius: 8px;
        }}
    """)
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(10, 8, 10, 10)
    layout.setSpacing(6)

    if title:
        lbl = QLabel(title.upper())
        lbl.setFont(QFont(FONT_UI, 8, QFont.Bold))
        lbl.setStyleSheet(f"color: {TEXT_DIM}; border: none; background: transparent;")
        layout.addWidget(lbl)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet(f"color: {BORDER}; border: none; background: {BORDER}; max-height: 1px;")
        layout.addWidget(line)

    return frame, layout


def status_row(grid: QGridLayout, row: int, key: str, val: str = "—", val_colour: str = TEXT):
    k = QLabel(key)
    k.setFont(QFont(FONT_UI, 9))
    k.setStyleSheet(f"color: {TEXT_DIM}; border: none; background: transparent;")
    v = QLabel(val)
    v.setFont(QFont(FONT_UI, 9, QFont.Bold))
    v.setStyleSheet(f"color: {val_colour}; border: none; background: transparent;")
    grid.addWidget(k, row, 0)
    grid.addWidget(v, row, 1)
    return v


# ──────────────────────────────────────────────
#  MAP WIDGET  (original coverage-grid version)
#  Kept from the original — LiDAR coverage grid,
#  path trace, robot dot, detection markers,
#  and detection markers.
# ──────────────────────────────────────────────
class MapWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(300, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(f"background: {BG};")

        self._map_img   = None
        self._map_res   = 0.05
        self._map_ox    = 0.0
        self._map_oy    = 0.0
        self._map_w     = 0
        self._map_h     = 0
        self._map_data  = None
        self._last_map_time = None

        self._robot_x   = 0.0
        self._robot_y   = 0.0
        self._robot_yaw = 0.0
        self._have_pose  = False
        self._arena_origin_x = None
        self._arena_origin_y = None

        self._arena_size = ARENA_SIZE_M
        self._arena_half = self._arena_size / 2.0
        self._coverage_res = 0.25
        self._coverage_n = int(self._arena_size / self._coverage_res)
        self._free_cells = np.zeros((self._coverage_n, self._coverage_n), dtype=bool)
        self._obstacle_cells = np.zeros((self._coverage_n, self._coverage_n), dtype=bool)

        self._path_trace = []
        self._live_scan_rays = []
        self._max_trace_points = 2000

        # Each entry: (world_x, world_y, label, colour_str)
        # Rebuilt every 2 seconds from detections_log.jsonl — never appended directly
        self._detections = []

        self._path = []

    def update_map(self, msg):
        self._map_res = msg.info.resolution
        self._map_ox  = msg.info.origin.position.x
        self._map_oy  = msg.info.origin.position.y
        self._map_w   = msg.info.width
        self._map_h   = msg.info.height

        data = np.array(msg.data, dtype=np.int8).reshape((self._map_h, self._map_w))
        self._map_data = data.copy()
        img = np.full((self._map_h, self._map_w, 3), 205, dtype=np.uint8)
        img[data == 0] = [254, 254, 254]
        img[data > 50] = [0, 0, 0]
        img = np.flipud(img)
        h, w, _ = img.shape
        self._map_img = QImage(img.tobytes(), w, h, 3*w, QImage.Format_RGB888).copy()
        self._last_map_time = time.time()
        self.update()

    def update_pose(self, x, y, yaw):
        self._robot_x   = x
        self._robot_y   = y
        self._robot_yaw = yaw
        self._have_pose = True
        if self._arena_origin_x is None or self._arena_origin_y is None:
            self._arena_origin_x = x
            self._arena_origin_y = y
        if not self._path_trace or math.hypot(
            x - self._path_trace[-1][0],
            y - self._path_trace[-1][1],
        ) > 0.03:
            self._path_trace.append((x, y))
            if len(self._path_trace) > self._max_trace_points:
                self._path_trace = self._path_trace[-self._max_trace_points:]
        self.update()

    def reset_centre_reference(self):
        if self._have_pose:
            self._arena_origin_x = self._robot_x
            self._arena_origin_y = self._robot_y
        else:
            self._arena_origin_x = None
            self._arena_origin_y = None
        self._free_cells.fill(False)
        self._obstacle_cells.fill(False)
        self._path_trace.clear()
        self._live_scan_rays.clear()
        self.update()

    def update_scan(self, msg):
        if not self._have_pose:
            return

        rays = []
        min_usable_range = max(msg.range_min, LIDAR_SELF_MASK_MIN_RANGE)
        for i, raw_range in enumerate(msg.ranges):
            r = raw_range
            if math.isnan(r) or r <= min_usable_range:
                continue
            hit_obstacle = math.isfinite(r)
            if not hit_obstacle:
                r = msg.range_max
            elif r > msg.range_max:
                r = msg.range_max
                hit_obstacle = False
            angle = self._robot_yaw + msg.angle_min + i * msg.angle_increment
            wx = self._robot_x + r * math.cos(angle)
            wy = self._robot_y + r * math.sin(angle)
            rays.append((self._robot_x, self._robot_y, wx, wy, hit_obstacle))

        self._live_scan_rays = rays
        self.update()

    def _world_to_arena(self, wx, wy):
        if self._arena_origin_x is None or self._arena_origin_y is None:
            return 0.0, 0.0
        return wx - self._arena_origin_x, wy - self._arena_origin_y

    def _arena_to_cell(self, ax, ay):
        col = int((ax + self._arena_half) / self._coverage_res)
        row = int((self._arena_half - ay) / self._coverage_res)
        if 0 <= row < self._coverage_n and 0 <= col < self._coverage_n:
            return row, col
        return None

    def _world_to_cell(self, wx, wy):
        ax, ay = self._world_to_arena(wx, wy)
        return self._arena_to_cell(ax, ay)

    def _ray_distance_to_arena_edge(self, angle):
        ax, ay = self._world_to_arena(self._robot_x, self._robot_y)
        dx = math.cos(angle)
        dy = math.sin(angle)
        distances = []
        if abs(dx) > 1e-6:
            distances.append((self._arena_half - ax) / dx)
            distances.append((-self._arena_half - ax) / dx)
        if abs(dy) > 1e-6:
            distances.append((self._arena_half - ay) / dy)
            distances.append((-self._arena_half - ay) / dy)
        positive = [dist for dist in distances if dist > 0.0]
        return min(positive) if positive else self._arena_half * 2.0

    def _paint_lidar_ray(self, angle, distance, sensor_max_range, hit_obstacle=True):
        arena_exit_dist = self._ray_distance_to_arena_edge(angle)
        max_dist = min(distance, sensor_max_range, arena_exit_dist)
        step = max(self._coverage_res * 0.5, 0.05)
        travelled = 0.0
        while travelled < max_dist:
            wx = self._robot_x + travelled * math.cos(angle)
            wy = self._robot_y + travelled * math.sin(angle)
            cell = self._world_to_cell(wx, wy)
            if cell is not None:
                self._free_cells[cell] = True
            travelled += step

        if hit_obstacle and distance < sensor_max_range * 0.97:
            wx = self._robot_x + distance * math.cos(angle)
            wy = self._robot_y + distance * math.sin(angle)
            cell = self._world_to_cell(wx, wy)
            if cell is not None:
                self._obstacle_cells[cell] = True

    def set_detections(self, detections: list):
        """Replace all detection markers. Called by the log poller."""
        self._detections = detections
        self.update()

    def update_path(self, poses):
        self._path = [(p.pose.position.x, p.pose.position.y) for p in poses]
        if self._path:
            coverage_home = self._coverage_home_from_path()
            if coverage_home is not None:
                # The control node appends the arena home/start pose to the end
                # of coverage paths. Keep that as the fallback GUI centre for
                # pre-SLAM drawing, while /map remains the primary live view.
                self._arena_origin_x, self._arena_origin_y = coverage_home
            elif self._arena_origin_x is None or self._arena_origin_y is None:
                if self._have_pose:
                    self._arena_origin_x = self._robot_x
                    self._arena_origin_y = self._robot_y
                else:
                    self._arena_origin_x, self._arena_origin_y = self._path[-1]
        self.update()

    def _coverage_home_from_path(self):
        if len(self._path) < 4:
            return None
        xs = [p[0] for p in self._path]
        ys = [p[1] for p in self._path]
        span_x = max(xs) - min(xs)
        span_y = max(ys) - min(ys)
        if span_x < self._arena_size * 0.4 and span_y < self._arena_size * 0.4:
            return None
        home_x, home_y = self._path[-1]
        centre_x = (min(xs) + max(xs)) / 2.0
        centre_y = (min(ys) + max(ys)) / 2.0
        # A coverage route's final pose is home, near the centre of its sweep.
        if math.hypot(home_x - centre_x, home_y - centre_y) <= self._arena_size * 0.25:
            return home_x, home_y
        return None

    def _arena_view_rect(self):
        size = max(10, min(self.width(), self.height()) - 20)
        dx = (self.width() - size) // 2
        dy = (self.height() - size) // 2
        return dx, dy, size

    def _world_to_px(self, wx, wy):
        if self._map_w > 0 and self._map_h > 0:
            dx, dy, scale = self._map_view_transform()
            gx = (wx - self._map_ox) / self._map_res
            gy = self._map_h - (wy - self._map_oy) / self._map_res
            return (int(dx + gx * scale), int(dy + gy * scale))

        if self._arena_origin_x is not None and self._arena_origin_y is not None:
            dx, dy, size = self._arena_view_rect()
            ax, ay = self._world_to_arena(wx, wy)
            px = int(dx + (ax + self._arena_half) / self._arena_size * size)
            py = int(dy + (self._arena_half - ay) / self._arena_size * size)
            return (px, py)

        if self._map_w == 0 or self._map_h == 0:
            scan_points = [(x2, y2) for _x1, _y1, x2, y2, _hit in self._live_scan_rays]
            points = self._path_trace + scan_points + [(self._robot_x, self._robot_y)]
            if not points:
                return (self.width()//2, self.height()//2)
            min_x = min(p[0] for p in points)
            max_x = max(p[0] for p in points)
            min_y = min(p[1] for p in points)
            max_y = max(p[1] for p in points)
            span = max(max_x - min_x, max_y - min_y, 1.0)
            pad = max(1.0, span * 0.15)
            min_x -= pad; max_x += pad; min_y -= pad; max_y += pad
            sx = self.width() / max(max_x - min_x, 0.1)
            sy = self.height() / max(max_y - min_y, 0.1)
            scale = min(sx, sy)
            used_w = (max_x - min_x) * scale
            used_h = (max_y - min_y) * scale
            px = int((wx - min_x) * scale + (self.width() - used_w) / 2)
            py = int((max_y - wy) * scale + (self.height() - used_h) / 2)
            return (px, py)

        return (self.width() // 2, self.height() // 2)

    def _map_view_transform(self):
        if self._map_w <= 0 or self._map_h <= 0:
            size = max(10, min(self.width(), self.height()) - 20)
            return (self.width() - size) // 2, (self.height() - size) // 2, 1.0
        scale = min(
            max(1, self.width() - 20) / self._map_w,
            max(1, self.height() - 20) / self._map_h,
        )
        used_w = self._map_w * scale
        used_h = self._map_h * scale
        return (self.width() - used_w) / 2, (self.height() - used_h) / 2, scale

    def _draw_coverage_grid(self, painter):
        dx, dy, size = self._arena_view_rect()
        cell = size / self._coverage_n

        painter.fillRect(dx, dy, size, size, QColor(205, 205, 205))

        for row in range(self._coverage_n):
            y = int(dy + row * cell)
            h = max(1, int(math.ceil(cell)))
            for col in range(self._coverage_n):
                if not self._free_cells[row, col] and not self._obstacle_cells[row, col]:
                    continue
                x = int(dx + col * cell)
                w = max(1, int(math.ceil(cell)))
                colour = QColor(20, 20, 20) if self._obstacle_cells[row, col] else QColor(255, 255, 255)
                painter.fillRect(x, y, w, h, colour)

        painter.setPen(QPen(QColor(120, 120, 120, 90), 1))
        for i in range(self._coverage_n + 1):
            pos = int(dx + i * cell)
            painter.drawLine(pos, dy, pos, dy + size)
            painter.drawLine(dx, pos, dx + size, pos)

        metre_step = self._coverage_n / self._arena_size
        painter.setPen(QPen(QColor(88, 166, 255, 100), 1))
        for metre in range(int(self._arena_size) + 1):
            offset = int(metre * metre_step * cell)
            painter.drawLine(dx + offset, dy, dx + offset, dy + size)
            painter.drawLine(dx, dy + offset, dx + size, dy + offset)

        painter.setPen(QPen(QColor(88, 166, 255, 210), 2, Qt.DashLine))
        painter.drawRect(dx, dy, size, size)

        painter.setPen(QPen(QColor(70, 70, 70), 1))
        painter.setFont(QFont(FONT_UI, 9))
        painter.drawText(dx + 8, dy + 18, f"{self._arena_size:g} x {self._arena_size:g} m LiDAR scan")

    def _draw_empty_arena(self, painter):
        size = min(self.width(), self.height()) - 20
        size = max(10, size)
        dx = (self.width() - size) // 2
        dy = (self.height() - size) // 2
        painter.fillRect(dx, dy, size, size, QColor(205, 205, 205))
        painter.setPen(QPen(QColor(120, 120, 120), 1, Qt.DashLine))
        painter.drawRect(dx, dy, size, size)

        painter.setPen(QPen(QColor(70, 70, 70), 1))
        painter.setFont(QFont(FONT_UI, 9))
        label = "Waiting for /map from SLAM..." if self._map_img is None else f"{self._arena_size:g} x {self._arena_size:g} m SLAM map"
        painter.drawText(dx + 8, dy + 18, label)

    def _draw_slam_map(self, painter):
        if self._map_img is None or self._map_w <= 0 or self._map_h <= 0:
            self._draw_empty_arena(painter)
            return

        dx, dy, scale = self._map_view_transform()
        painter.drawImage(
            int(dx), int(dy),
            self._map_img.scaled(
                int(self._map_w * scale),
                int(self._map_h * scale),
                Qt.KeepAspectRatio,
                Qt.FastTransformation,
            )
        )

    def _draw_arena_boundary(self, painter):
        if self._arena_origin_x is None or self._arena_origin_y is None:
            return
        left = self._arena_origin_x - self._arena_half
        right = self._arena_origin_x + self._arena_half
        bottom = self._arena_origin_y - self._arena_half
        top = self._arena_origin_y + self._arena_half
        corners = [
            self._world_to_px(left, bottom),
            self._world_to_px(right, bottom),
            self._world_to_px(right, top),
            self._world_to_px(left, top),
        ]
        painter.setPen(QPen(QColor(88, 166, 255, 180), 2, Qt.DashLine))
        for i in range(len(corners)):
            painter.drawLine(*corners[i], *corners[(i + 1) % len(corners)])

    def _draw_live_scan(self, painter):
        if not self._live_scan_rays:
            return

        clear_pen = QPen(QColor(255, 255, 255, 115), 1)
        hit_pen = QPen(QColor(35, 35, 35, 180), 1)
        hit_brush = QBrush(QColor(35, 35, 35, 180))
        for x1, y1, x2, y2, hit_obstacle in self._live_scan_rays:
            p1 = self._world_to_px(x1, y1)
            p2 = self._world_to_px(x2, y2)
            painter.setPen(hit_pen if hit_obstacle else clear_pen)
            painter.drawLine(*p1, *p2)
            if hit_obstacle:
                painter.setBrush(hit_brush)
                painter.drawEllipse(p2[0] - 1, p2[1] - 1, 2, 2)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(205, 205, 205))
        self._draw_slam_map(painter)
        self._draw_live_scan(painter)

        # Draw detection markers — one per unique obstacle from the log file
        for (wx, wy, label, col) in self._detections:
            px, py = self._world_to_px(wx, wy)
            colour = QColor(RED) if "red" in col else \
                     QColor(YELLOW) if "yellow" in col else \
                     QColor(ACCENT) if col == "letter" else QColor(GREEN)
            painter.setBrush(QBrush(colour))
            painter.setPen(QPen(QColor(TEXT), 1))
            painter.drawEllipse(px - 8, py - 8, 16, 16)
            painter.setFont(QFont(FONT_UI, 7, QFont.Bold))
            painter.setPen(QColor(TEXT))
            display = label.replace("_obstacle", "").replace("_", " ")
            painter.drawText(px + 10, py + 4, display)

        rx, ry = self._world_to_px(self._robot_x, self._robot_y)
        painter.setBrush(QBrush(QColor(ACCENT)))
        painter.setPen(QPen(QColor(TEXT), 2))
        painter.drawEllipse(rx - 5, ry - 5, 10, 10)
        painter.end()


# ──────────────────────────────────────────────
#  DETECTION LOG WIDGET
# ──────────────────────────────────────────────
class DetectionLog(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet(f"""
            QScrollArea {{ border: none; background: transparent; }}
            QScrollBar:vertical {{ background: {PANEL}; width: 6px; }}
            QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 3px; }}
        """)

        self._inner = QWidget()
        self._inner.setStyleSheet("background: transparent;")
        self._inner_layout = QVBoxLayout(self._inner)
        self._inner_layout.setContentsMargins(0, 0, 0, 0)
        self._inner_layout.setSpacing(3)
        self._inner_layout.addStretch()

        self._scroll.setWidget(self._inner)
        layout.addWidget(self._scroll)

    def add_entry(self, text: str, colour: str = TEXT):
        ts  = datetime.now().strftime("%H:%M:%S")
        lbl = QLabel(f"<span style='color:{TEXT_DIM}'>{ts}</span>  "
                     f"<span style='color:{colour}'>{text}</span>")
        lbl.setFont(QFont(FONT_UI, 9))
        lbl.setStyleSheet("background: transparent; border: none;")
        lbl.setWordWrap(True)
        count = self._inner_layout.count()
        self._inner_layout.insertWidget(count - 1, lbl)
        QTimer.singleShot(50, lambda: self._scroll.verticalScrollBar().setValue(
            self._scroll.verticalScrollBar().maximum()))


# ──────────────────────────────────────────────
#  ARENA DEBUG PANEL
# ──────────────────────────────────────────────
class ArenaPanel(QWidget):
    ARENA_HALF = ARENA_SIZE_M / 2.0

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        grid = QGridLayout()
        grid.setSpacing(4)
        grid.setColumnStretch(1, 1)

        self._lbl_state  = status_row(grid, 0, "Drive state",    "—",      ACCENT)
        self._lbl_pos    = status_row(grid, 1, "Position",       "(—, —)", TEXT)
        self._lbl_home   = status_row(grid, 2, "Dist from home", "— m",    TEXT)
        self._lbl_edge   = status_row(grid, 3, "Edge clearance", "— m",    TEXT)
        self._lbl_yaw    = status_row(grid, 4, "Heading",        "—°",     TEXT)
        self._lbl_source = status_row(grid, 5, "Pose source",    "—",      TEXT_DIM)
        layout.addLayout(grid)

        bar_label = QLabel("Edge clearance")
        bar_label.setFont(QFont(FONT_UI, 8))
        bar_label.setStyleSheet(f"color: {TEXT_DIM}; border: none; background: transparent;")
        layout.addWidget(bar_label)

        self._edge_bar = QProgressBar()
        self._edge_bar.setRange(0, 100)
        self._edge_bar.setValue(100)
        self._edge_bar.setTextVisible(False)
        self._edge_bar.setFixedHeight(8)
        self._set_bar_colour(GREEN)
        layout.addWidget(self._edge_bar)

    def _set_bar_colour(self, colour: str):
        self._edge_bar.setStyleSheet(f"""
            QProgressBar {{ background: {BORDER}; border: none; border-radius: 4px; }}
            QProgressBar::chunk {{ background: {colour}; border-radius: 4px; }}
        """)

    def update(self, data: dict):
        state  = data.get('state',         '—')
        rel_x  = data.get('rel_x',          0.0)
        rel_y  = data.get('rel_y',          0.0)
        home   = data.get('center_dist',    0.0)
        edge   = data.get('edge_clearance', 0.0)
        yaw    = data.get('yaw',            0.0)
        source = data.get('pose_source',    '—')

        state_colours = {
            'WANDERING_DRIVE':       GREEN,  'WANDERING_TURN':        ACCENT,
            'OBSTACLE_REVERSE':      YELLOW, 'OBSTACLE_TURN':         YELLOW,
            'BOUNDARY_REVERSE':      RED,    'BOUNDARY_ESCAPE_TURN':  RED,
            'BOUNDARY_ESCAPE_DRIVE': RED,    'RETURN_TO_CENTER':      YELLOW,
            'INITIAL_TURN':          TEXT_DIM, 'INITIAL_DRIVE':       TEXT_DIM,
        }
        sc = state_colours.get(state, TEXT)
        self._lbl_state.setText(state.replace('_', ' '))
        self._lbl_state.setStyleSheet(
            f"color: {sc}; border: none; background: transparent; font-weight: bold;")
        self._lbl_pos.setText(f"({rel_x:+.2f}, {rel_y:+.2f}) m")
        self._lbl_home.setText(f"{home:.2f} m")
        self._lbl_edge.setText(f"{edge:.2f} m")
        self._lbl_yaw.setText(f"{yaw:.1f}°")
        self._lbl_source.setText(source)

        pct = min(100, int(edge / self.ARENA_HALF * 100))
        self._edge_bar.setValue(pct)
        self._set_bar_colour(GREEN if pct > 50 else YELLOW if pct > 20 else RED)


# ──────────────────────────────────────────────
#  MAIN WINDOW
# ──────────────────────────────────────────────
class RobotGUI(QMainWindow):
    def __init__(self, signals: Signals):
        super().__init__()
        self.signals = signals
        self.setWindowTitle("AUTO4508 — Pioneer 3-AT Monitor")
        self.setMinimumSize(1400, 800)
        self.setStyleSheet(f"background: {BG}; color: {TEXT};")
        self._latest_map_msg = None

        # Tracking for detection log spam prevention
        self._last_detection_event_times: dict[str, float] = {}

        # Track last file-modified time so we don't re-read unchanged log
        self._last_log_mtime: float = 0.0
        # Dedup state log entries
        self._last_display_state: str = ""

        self._build_ui()
        self._connect_signals()

        # Poll detections_log.jsonl every 2 s to update map markers
        self._log_poll_timer = QTimer()
        self._log_poll_timer.timeout.connect(self._poll_detection_log)
        self._log_poll_timer.start(2000)

        # Poll photo directory every 2s to show latest saved detection photo
        self._last_photo_path: str = ""
        self._video_writer = None
        self._video_path: str = ""
        self._video_recording: bool = False
        self._photo_poll_timer = QTimer()
        self._photo_poll_timer.timeout.connect(self._poll_latest_photo)
        self._photo_poll_timer.start(2000)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # Header
        header = QHBoxLayout()
        title  = QLabel("AUTO4508  ·  Pioneer 3-AT")
        title.setFont(QFont(FONT_BODY, 13, QFont.Bold))
        title.setStyleSheet(f"color: {ACCENT};")
        header.addWidget(title)
        header.addStretch()
        self._state_badge = QLabel("IDLE")
        self._state_badge.setFont(QFont(FONT_UI, 11, QFont.Bold))
        self._state_badge.setAlignment(Qt.AlignCenter)
        self._state_badge.setFixedHeight(32)
        self._state_badge.setStyleSheet(self._badge_style(YELLOW))
        header.addWidget(self._state_badge)

        # Reset E-Stop button — always visible at top, red when active
        self._reset_estop_btn = QPushButton("✕  RESET E-STOP")
        self._reset_estop_btn.setFont(QFont(FONT_UI, 10, QFont.Bold))
        self._reset_estop_btn.setCursor(Qt.PointingHandCursor)
        self._reset_estop_btn.setFixedHeight(32)
        self._reset_estop_btn.setStyleSheet(f"""
            QPushButton {{
                background: {RED};
                color: white;
                border: 2px solid {RED};
                border-radius: 6px;
                padding: 2px 16px;
            }}
            QPushButton:hover {{
                background: #c0392b;
                border-color: #c0392b;
            }}
        """)
        header.addWidget(self._reset_estop_btn)
        root.addLayout(header)

        content = QHBoxLayout()
        content.setSpacing(10)

        # ── LEFT COLUMN ──────────────────────────────────────────────────────
        left = QVBoxLayout()
        left.setSpacing(10)

        cam_frame, cam_layout = make_panel("Camera Feed")
        self._cam_label = QLabel()
        self._cam_label.setAlignment(Qt.AlignCenter)
        self._cam_label.setMinimumSize(480, 270)
        self._cam_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._cam_label.setStyleSheet("background: #000; border: none;")
        self._cam_label.setText("No camera feed")
        cam_layout.addWidget(self._cam_label)
        left.addWidget(cam_frame, 3)

        status_frame, status_layout = make_panel("Robot Status")
        sg = QGridLayout()
        sg.setSpacing(6)
        self._status_labels = {}
        for i, (key, val) in enumerate([
            ("State",       "IDLE"),
            ("Action",      "—"),
            ("Pos X",       "0.00 m"),
            ("Pos Y",       "0.00 m"),
            ("Last Letter", "—"),
        ]):
            self._status_labels[key] = status_row(sg, i, key, val)
        status_layout.addLayout(sg)
        left.addWidget(status_frame, 1)

        arena_frame, arena_layout = make_panel("Arena Debug")
        self._arena_panel = ArenaPanel()
        arena_layout.addWidget(self._arena_panel)
        left.addWidget(arena_frame, 1)

        content.addLayout(left, 5)

        # ── MIDDLE COLUMN — map ───────────────────────────────────────────────
        map_frame, map_layout = make_panel("Map")
        self._map_widget = MapWidget()
        map_layout.addWidget(self._map_widget)

        map_controls = QHBoxLayout()
        self._start_wandering_btn = QPushButton("Start Mapping")
        self._start_wandering_btn.setFont(QFont(FONT_UI, 9, QFont.Bold))
        self._start_wandering_btn.setCursor(Qt.PointingHandCursor)
        self._start_wandering_btn.setStyleSheet(f"""
            QPushButton {{ background: {GREEN}22; color: {GREEN};
                border: 1px solid {GREEN}; border-radius: 6px; padding: 6px 12px; }}
        """)
        self._go_home_btn = QPushButton("Go Home")
        self._go_home_btn.setFont(QFont(FONT_UI, 9, QFont.Bold))
        self._go_home_btn.setCursor(Qt.PointingHandCursor)
        self._go_home_btn.setStyleSheet(f"""
            QPushButton {{ background: {ACCENT}22; color: {ACCENT};
                border: 1px solid {ACCENT}; border-radius: 6px; padding: 6px 12px; }}
        """)
        self._drive_waypoints_btn = QPushButton("Drive Waypoints")
        self._drive_waypoints_btn.setFont(QFont(FONT_UI, 9, QFont.Bold))
        self._drive_waypoints_btn.setCursor(Qt.PointingHandCursor)
        self._drive_waypoints_btn.setStyleSheet(f"""
            QPushButton {{ background: {YELLOW}22; color: {YELLOW};
                border: 1px solid {YELLOW}; border-radius: 6px; padding: 6px 12px; }}
        """)
        self._save_map_btn = QPushButton("Save Map")
        self._save_map_btn.setFont(QFont(FONT_UI, 9, QFont.Bold))
        self._save_map_btn.setCursor(Qt.PointingHandCursor)
        self._save_map_btn.setStyleSheet(f"""
            QPushButton {{ background: {GREEN}22; color: {GREEN};
                border: 1px solid {GREEN}; border-radius: 6px; padding: 6px 12px; }}
            QPushButton:disabled {{ background: {BORDER}; color: {TEXT_DIM};
                border: 1px solid {BORDER}; }}
        """)
        self._replay_btn = QPushButton("▶  Replay Journey")
        self._replay_btn.setFont(QFont(FONT_UI, 9, QFont.Bold))
        self._replay_btn.setCursor(Qt.PointingHandCursor)
        self._replay_btn.setStyleSheet(f"""
            QPushButton {{ background: {ACCENT}22; color: {ACCENT};
                border: 1px solid {ACCENT}; border-radius: 6px; padding: 6px 12px; }}
            QPushButton:disabled {{ background: {BORDER}; color: {TEXT_DIM};
                border: 1px solid {BORDER}; }}
        """)
        self._save_map_label = QLabel("Maps save to ros2_ws/maps")
        self._save_map_label.setFont(QFont(FONT_UI, 8))
        self._save_map_label.setStyleSheet(
            f"color: {TEXT_DIM}; border: none; background: transparent;")
        map_controls.addWidget(self._start_wandering_btn)
        map_controls.addWidget(self._go_home_btn)
        map_controls.addWidget(self._drive_waypoints_btn)
        map_controls.addWidget(self._save_map_btn)
        map_controls.addWidget(self._replay_btn)
        map_controls.addWidget(self._save_map_label, 1)
        map_layout.addLayout(map_controls)

        # Drive-to row — type object names, click Drive To
        drive_to_row = QHBoxLayout()
        drive_to_lbl = QLabel("Drive to:")
        drive_to_lbl.setFont(QFont(FONT_UI, 9))
        drive_to_lbl.setStyleSheet(f"color: {TEXT_DIM}; border: none; background: transparent;")
        self._drive_to_input = QLineEdit()
        self._drive_to_input.setFont(QFont(FONT_UI, 9))
        self._drive_to_input.setPlaceholderText("e.g.  mu, red_obstacle, tau, yellow_obstacle")
        self._drive_to_input.setStyleSheet(f"""
            QLineEdit {{
                background: {PANEL}; color: {TEXT};
                border: 1px solid {BORDER}; border-radius: 4px; padding: 4px 8px;
            }}
            QLineEdit:focus {{ border-color: {ACCENT}; }}
        """)
        self._drive_to_opt_btn = QPushButton("Drive To (optimal order)")
        self._drive_to_opt_btn.setFont(QFont(FONT_UI, 9, QFont.Bold))
        self._drive_to_opt_btn.setCursor(Qt.PointingHandCursor)
        self._drive_to_opt_btn.setStyleSheet(f"""
            QPushButton {{ background: {ACCENT}22; color: {ACCENT};
                border: 1px solid {ACCENT}; border-radius: 6px; padding: 6px 12px; }}
        """)
        self._drive_to_ord_btn = QPushButton("Drive To (my order)")
        self._drive_to_ord_btn.setFont(QFont(FONT_UI, 9, QFont.Bold))
        self._drive_to_ord_btn.setCursor(Qt.PointingHandCursor)
        self._drive_to_ord_btn.setStyleSheet(f"""
            QPushButton {{ background: {YELLOW}22; color: {YELLOW};
                border: 1px solid {YELLOW}; border-radius: 6px; padding: 6px 12px; }}
        """)
        drive_to_row.addWidget(drive_to_lbl)
        drive_to_row.addWidget(self._drive_to_input, 1)
        drive_to_row.addWidget(self._drive_to_opt_btn)
        drive_to_row.addWidget(self._drive_to_ord_btn)
        map_layout.addLayout(drive_to_row)
        content.addWidget(map_frame, 4)

        # ── RIGHT COLUMN ──────────────────────────────────────────────────────
        right = QVBoxLayout()
        right.setSpacing(10)

        log_frame, log_layout = make_panel("Detection Log")
        self._det_log = DetectionLog()
        self._det_log.setMinimumHeight(200)
        log_layout.addWidget(self._det_log)
        right.addWidget(log_frame, 3)

        # E-stop event panel — loads last 5s of data when estop fires
        estop_frame, estop_layout = make_panel("⚠  Last E-Stop Event")
        self._estop_log = DetectionLog()
        self._estop_log.setMinimumHeight(80)
        estop_layout.addWidget(self._estop_log)
        right.addWidget(estop_frame, 1)

        # Bottom-right photo panel — fed by /detections/image (annotated frames)
        photo_frame, photo_layout = make_panel("Last Detection  (/detections/image)")
        self._photo_label = QLabel()
        self._photo_label.setAlignment(Qt.AlignCenter)
        self._photo_label.setMinimumHeight(160)
        self._photo_label.setStyleSheet("background: #000; border: none;")
        self._photo_label.setText("Waiting for detection...")
        photo_layout.addWidget(self._photo_label)
        right.addWidget(photo_frame, 2)

        content.addLayout(right, 3)
        root.addLayout(content, 1)

        # Bottom status bar
        bottom = QHBoxLayout()
        self._action_label = QLabel("Waiting for robot...")
        self._action_label.setFont(QFont(FONT_UI, 9))
        self._action_label.setStyleSheet(f"color: {TEXT_DIM};")
        bottom.addWidget(self._action_label)
        bottom.addStretch()
        self._clock_label = QLabel()
        self._clock_label.setFont(QFont(FONT_UI, 9))
        self._clock_label.setStyleSheet(f"color: {TEXT_DIM};")
        bottom.addWidget(self._clock_label)
        root.addLayout(bottom)

        self._clock_timer = QTimer()
        self._clock_timer.timeout.connect(self._tick_clock)
        self._clock_timer.start(1000)
        self._tick_clock()

    def _badge_style(self, colour: str) -> str:
        return (f"background: {colour}22; color: {colour}; "
                f"border: 1px solid {colour}; border-radius: 6px; padding: 2px 14px;")

    def _connect_signals(self):
        self.signals.camera_frame.connect(self._on_camera)
        self.signals.colour_image.connect(self._on_colour_image)
        self.signals.robot_state.connect(self._on_state)
        self.signals.robot_pose.connect(self._on_pose)
        self.signals.letter_detected.connect(self._on_letter)
        self.signals.colour_detected.connect(self._on_colour)
        self.signals.map_updated.connect(self._on_map)
        self.signals.scan_updated.connect(self._on_scan)
        self.signals.path_updated.connect(self._on_path)
        self.signals.arena_updated.connect(self._on_arena)
        self.signals.save_status.connect(self._on_save_status)
        self._start_wandering_btn.clicked.connect(
            lambda: self._send_mission_command("drive_coverage"))
        self._go_home_btn.clicked.connect(
            lambda: self._send_mission_command("go_home"))
        self._drive_waypoints_btn.clicked.connect(
            lambda: self._send_mission_command("drive_waypoints"))
        self._save_map_btn.clicked.connect(self._save_map)
        self._replay_btn.clicked.connect(self._play_latest_video)
        self._reset_estop_btn.clicked.connect(self._reset_estop)
        self._drive_to_opt_btn.clicked.connect(self._drive_to_optimal)
        self._drive_to_ord_btn.clicked.connect(self._drive_to_ordered)

    # ── Detection log poller ──────────────────────────────────────────────────

    def _poll_latest_photo(self):
        """Find the most recently saved photo (colour or letter) and display it."""
        search_dirs = [
            PHOTO_DIR,
            os.path.expanduser("~/part3_logs/letter_detections"),
        ]
        try:
            photos = []
            for d in search_dirs:
                if os.path.isdir(d):
                    photos.extend([
                        os.path.join(d, f)
                        for f in os.listdir(d)
                        if f.lower().endswith((".png", ".jpg", ".jpeg"))
                    ])
            if not photos:
                return
            latest = max(photos, key=os.path.getmtime)
            if latest == self._last_photo_path:
                return
            self._last_photo_path = latest
            img = cv2.imread(latest)
            if img is None:
                return
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.tobytes(), w, h, ch * w, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg).scaled(
                self._photo_label.width(), self._photo_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self._photo_label.setPixmap(pix)
            self._photo_label.setToolTip(os.path.basename(latest))
        except Exception:
            pass

    def _poll_detection_log(self):
        """
        Read detections_log.jsonl every 2 s.
        Keeps the closest reading per label so each obstacle appears once on the map.
        Skips re-reading if the file hasn't changed since the last poll.
        """
        if not os.path.exists(DETECTION_LOG_PATH):
            return
        try:
            mtime = os.path.getmtime(DETECTION_LOG_PATH)
        except OSError:
            return
        if mtime == self._last_log_mtime:
            return
        self._last_log_mtime = mtime

        best: dict[str, dict] = {}
        try:
            with open(DETECTION_LOG_PATH) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        name   = record.get("name", "")
                        dist   = record.get("distance_m") or 999.0
                        if name not in best or dist < (best[name].get("distance_m") or 999.0):
                            best[name] = record
                    except Exception:
                        continue
        except Exception:
            return

        detections = []
        for record in best.values():
            obj_x = record.get("object_x")
            obj_y = record.get("object_y")
            name  = record.get("name", "")
            kind  = record.get("type", "")
            shape = record.get("shape", "")
            if obj_x is None or obj_y is None:
                continue
            colour  = "red"    if "red"    in name else \
                      "yellow" if "yellow" in name else \
                      "letter" if kind == "letter" else "green"
            # Include shape in the label if available (e.g. "cone red")
            display = f"{shape} {name}".strip() if shape else name
            detections.append((obj_x, obj_y, display, colour))

        self._map_widget.set_detections(detections)
        self._save_map_label.setText(
            f"Maps → ros2_ws/maps  ·  {len(detections)} detection(s) logged")

    # ── Slot handlers ─────────────────────────────────────────────────────────

    def _on_camera(self, frame: np.ndarray):
        # Write frame to video if recording
        if self._video_recording and self._video_writer is not None:
            try:
                self._video_writer.write(frame)
            except Exception:
                pass
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.tobytes(), w, h, ch * w, QImage.Format_RGB888)
        pix  = QPixmap.fromImage(qimg).scaled(
            self._cam_label.width(), self._cam_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._cam_label.setPixmap(pix)

    def _on_colour_image(self, frame: np.ndarray):
        """Bottom-right panel — annotated detection frame from /detections/image."""
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.tobytes(), w, h, ch * w, QImage.Format_RGB888)
        pix  = QPixmap.fromImage(qimg).scaled(
            self._photo_label.width(), self._photo_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._photo_label.setPixmap(pix)

    def _on_state(self, state: str):
        self._status_labels["State"].setText(state)
        self._state_badge.setText(state)
        colour_map = {
            "MAPPING":          GREEN,  "WAYPOINT":         ACCENT,
            "GOAL_ACHIEVED":    GREEN,  "REACHED_HOME":     GREEN,
            "IDLE":             YELLOW, "STOPPED":          RED,
            "ESTOP":            RED,    "RETURN_TO_CENTER": YELLOW,
        }
        colour = colour_map.get(state, TEXT_DIM)
        self._state_badge.setStyleSheet(self._badge_style(colour))
        action_map = {
            "MAPPING":          "Exploring area and building map...",
            "WAYPOINT":         "Driving to waypoints at maximum speed...",
            "GOAL_ACHIEVED":    "Goal achieved. Press Go Home to return to centre.",
            "REACHED_HOME":     "Reached home. Waiting for next command.",
            "IDLE":             "Standing by.",
            "STOPPED":          "EMERGENCY STOP — all motion halted!",
            "ESTOP":            "EMERGENCY STOP — obstacle detected!",
            "RETURN_TO_CENTER": "Returning to arena centre...",
        }
        action = action_map.get(state, state)
        self._status_labels["Action"].setText(action)
        self._action_label.setText(action)

        # Only log when state actually changes — prevents 2 Hz spam when stuck in ESTOP
        if state != getattr(self, "_last_display_state", None):
            self._last_display_state = state
            self._det_log.add_entry(f"State → {state}", colour)
            if state in ("STOPPED", "ESTOP"):
                self._load_latest_estop_event()
            # Auto-record video during active states
            if state == "MAPPING" and not self._video_recording:
                self._start_video_recording()
            elif state in ("REACHED_HOME", "IDLE", "GOAL_ACHIEVED") and self._video_recording:
                self._stop_video_recording()

    def _on_pose(self, x: float, y: float, yaw: float):
        self._status_labels["Pos X"].setText(f"{x:.2f} m")
        self._status_labels["Pos Y"].setText(f"{y:.2f} m")
        self._map_widget.update_pose(x, y, yaw)

    def _should_log_detection_event(self, key: str, cooldown_s: float = 2.0) -> bool:
        """Rate-limits detection log entries so the same thing doesn't spam the log."""
        now  = time.monotonic()
        last = self._last_detection_event_times.get(key, 0.0)
        if now - last < cooldown_s:
            return False
        self._last_detection_event_times[key] = now
        return True

    def _on_letter(self, raw: str):
        """
        Parse JSON payload from /detected_letter.
        Logs to the detection widget; map markers come from the log poller.
        """
        try:
            data    = json.loads(raw)
            name    = data.get("name", raw)
            dist    = data.get("distance_m")
            bearing = math.radians(data.get("bearing_deg", 0.0))
            rx      = data.get("robot_x", 0.0)
            ry      = data.get("robot_y", 0.0)
            dist_str = f"  dist={dist:.2f}m" if dist else ""
        except (json.JSONDecodeError, TypeError):
            name     = raw
            dist_str = ""

        self._status_labels["Last Letter"].setText(name)
        if self._should_log_detection_event(f"letter:{name}", cooldown_s=8.0):
            self._det_log.add_entry(f"Greek letter: {name}{dist_str}", GREEN)

    def _on_colour(self, data: dict):
        """
        Log a colour detection entry including shape if the detector provides it.
        Map markers are handled by _poll_detection_log — not placed here
        to avoid duplicates with the log-file poller.
        """
        label   = data.get("label",       "unknown")
        dist    = data.get("distance_m",  0.0)
        bearing = data.get("bearing_deg", 0.0)
        shape   = data.get("shape",       "")        # e.g. "cone", "cylinder", "box"
        colour  = RED if "red" in label else YELLOW

        shape_str = f"  shape={shape}" if shape else ""
        if self._should_log_detection_event(f"colour:{label}", cooldown_s=8.0):
            self._det_log.add_entry(
                f"{label}{shape_str}  dist={dist:.2f}m  bearing={bearing:.1f}°",
                colour)

    def _on_map(self, msg):
        self._latest_map_msg = msg
        self._map_widget.update_map(msg)

    def _on_scan(self, msg):
        self._map_widget.update_scan(msg)

    def _on_path(self, msg):
        self._map_widget.update_path(msg.poses)
        self._det_log.add_entry(f"Path updated — {len(msg.poses)} waypoints", ACCENT)

    def _on_arena(self, data: dict):
        self._arena_panel.update(data)

    def _send_mission_command(self, command: str):
        if command in {"start_wandering", "drive_coverage"}:
            self._map_widget.reset_centre_reference()
            self._on_state("MAPPING")
        elif command == "drive_waypoints":
            self._on_state("WAYPOINT")
        elif command == "go_home":
            self._on_state("RETURN_TO_CENTER")
        self.signals.mission_command.emit(command)
        self._det_log.add_entry(f"Command → {command}", ACCENT)

    def _load_latest_estop_event(self):
        """
        Display estop event data. Checks two sources:
        1. estop_*.jsonl files written by mission_manager.py
        2. incidents.txt written by control_node.py
        """
        if not os.path.isdir(ESTOP_LOG_DIR):
            self._estop_log.add_entry("No e-stop directory yet.", TEXT_DIM)
            return

        # Try JSONL files first (mission_manager format)
        jsonl_files = sorted(
            [f for f in os.listdir(ESTOP_LOG_DIR)
             if f.startswith("estop_") and f.endswith(".jsonl")],
            reverse=True)

        if jsonl_files:
            path = os.path.join(ESTOP_LOG_DIR, jsonl_files[0])
            self._estop_log.add_entry(f"\u2500\u2500 {jsonl_files[0]} \u2500\u2500", RED)
            try:
                with open(path) as f:
                    entries = [json.loads(l) for l in f if l.strip()]
                pose_entries = [e for e in entries if e.get("type") == "pose"]
                scan_entries = [e for e in entries if e.get("type") == "scan"]
                self._estop_log.add_entry(
                    f"{len(entries)} records | {len(pose_entries)} pose | "
                    f"{len(scan_entries)} scan", TEXT_DIM)
                if pose_entries:
                    p0, p1 = pose_entries[0], pose_entries[-1]
                    self._estop_log.add_entry(
                        f"({p0['x']:.2f},{p0['y']:.2f}) \u2192 "
                        f"({p1['x']:.2f},{p1['y']:.2f})  yaw={p1['yaw']:.1f}\u00b0", YELLOW)
                if scan_entries:
                    ranges = [r for r in scan_entries[-1].get("ranges", []) if r > 0.05]
                    if ranges:
                        self._estop_log.add_entry(
                            f"Closest obstacle: {min(ranges):.2f} m", RED)
            except Exception as exc:
                self._estop_log.add_entry(f"Error: {exc}", RED)
            return

        # Fallback: incidents.txt written by control_node.py
        incidents_path = os.path.join(ESTOP_LOG_DIR, "incidents.txt")
        if os.path.exists(incidents_path):
            try:
                with open(incidents_path) as f:
                    lines = [l.strip() for l in f if l.strip()]
                if lines:
                    self._estop_log.add_entry("\u2500\u2500 incidents.txt \u2500\u2500", RED)
                    for line in lines[-3:]:
                        self._estop_log.add_entry(line, YELLOW)
                else:
                    self._estop_log.add_entry("incidents.txt empty.", TEXT_DIM)
            except Exception as exc:
                self._estop_log.add_entry(f"Error: {exc}", RED)
        else:
            self._estop_log.add_entry("No estop files yet — will appear when triggered.", TEXT_DIM)

    def _replay_journey(self):
        """
        Play ALL bags from this session in chronological order so the replay
        is smooth and continuous rather than jumping between snapshots.
        Bags are sorted oldest-first so they play back in the order recorded.
        """
        if not os.path.isdir(BAG_DIR):
            self._det_log.add_entry("No bags directory found.", RED)
            return

        # Sort oldest-first (ascending) so we replay in recording order
        # Use only directories that contain a metadata.yaml (valid bags)
        bags = sorted(
            [os.path.join(BAG_DIR, d) for d in os.listdir(BAG_DIR)
             if os.path.isdir(os.path.join(BAG_DIR, d))
             and os.path.exists(os.path.join(BAG_DIR, d, "metadata.yaml"))])

        if not bags:
            self._det_log.add_entry("No recorded bags found.", RED)
            return

        self._det_log.add_entry(
            f"Replaying {len(bags)} bag(s) in order...", ACCENT)
        self._replay_btn.setEnabled(False)
        self._replay_btn.setText(f"Replaying 1 / {len(bags)}…")

        def _run():
            for i, bag_path in enumerate(bags):
                name = os.path.basename(bag_path)
                # Update button label on Qt thread
                QTimer.singleShot(0, lambda n=name, idx=i: (
                    self._replay_btn.setText(f"Replaying {idx+1}/{len(bags)}: {n[-12:]}")
                ))
                subprocess.run(
                    ["ros2", "bag", "play", bag_path,
                     "--clock",
                     "--rate", "1.0"],   # change to 0.5 to slow down
                    check=False)
            QTimer.singleShot(0, self._on_replay_done)

        threading.Thread(target=_run, daemon=True).start()

    def _on_replay_done(self):
        self._replay_btn.setEnabled(True)
        self._replay_btn.setText("▶  Replay Journey")
        self._det_log.add_entry("Replay finished.", TEXT_DIM)

    def _drive_to_optimal(self):
        """Send drive_to command — waypoint controller will optimise visit order."""
        names = self._drive_to_input.text().strip()
        if not names:
            self._det_log.add_entry("Type object names before clicking Drive To", YELLOW)
            return
        command = f"drive_to:{names}"
        self.signals.mission_command.emit(command)
        self._det_log.add_entry(f"Drive to (optimal order): {names}", ACCENT)
        self._on_state("WAYPOINT")

    def _drive_to_ordered(self):
        """Send drive_to_ordered command — visits in the exact order typed."""
        names = self._drive_to_input.text().strip()
        if not names:
            self._det_log.add_entry("Type object names before clicking Drive To", YELLOW)
            return
        command = f"drive_to_ordered:{names}"
        self.signals.mission_command.emit(command)
        self._det_log.add_entry(f"Drive to (your order): {names}", YELLOW)
        self._on_state("WAYPOINT")

    def _start_video_recording(self):
        """Start recording camera feed to an mp4 file."""
        os.makedirs(VIDEO_DIR, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._video_path = os.path.join(VIDEO_DIR, f"journey_{stamp}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._video_writer = cv2.VideoWriter(self._video_path, fourcc, 20.0, (640, 480))
        self._video_recording = True
        self._det_log.add_entry(f"Recording video: journey_{stamp}.mp4", GREEN)
        self.get_logger().info(f"Video recording started: {self._video_path}")

    def _stop_video_recording(self):
        """Stop recording and save the video file."""
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
        self._video_recording = False
        if self._video_path:
            self._det_log.add_entry(
                f"Video saved: {os.path.basename(self._video_path)}", GREEN)
            self.get_logger().info(f"Video saved: {self._video_path}")

    def _play_latest_video(self):
        """Open the most recent journey video with the system video player."""
        if not os.path.isdir(VIDEO_DIR):
            self._det_log.add_entry("No videos recorded yet.", TEXT_DIM)
            return
        videos = sorted(
            [os.path.join(VIDEO_DIR, f) for f in os.listdir(VIDEO_DIR)
             if f.endswith(".mp4")],
            reverse=True)
        if not videos:
            self._det_log.add_entry("No video files found.", RED)
            return
        latest = videos[0]
        self._det_log.add_entry(
            f"Playing: {os.path.basename(latest)}", ACCENT)
        # Try multiple players
        for player in ["xdg-open", "vlc", "ffplay", "mpv"]:
            try:
                subprocess.Popen([player, latest])
                return
            except FileNotFoundError:
                continue
        self._det_log.add_entry(
            f"No video player found. File at: {latest}", YELLOW)

    def _reset_estop(self):
        """Clear the e-stop state and return to IDLE so the robot can be commanded again."""
        self._last_display_state = ""   # force next state log entry through
        self.signals.mission_command.emit("reset_estop")
        self._det_log.add_entry("E-Stop reset sent — robot returning to IDLE", YELLOW)
        self._on_state("IDLE")

    # ── Map save ──────────────────────────────────────────────────────────────

    def _save_map(self):
        if self._latest_map_msg is None:
            self.signals.save_status.emit("No /map received yet", False)
            return
        self._save_map_btn.setEnabled(False)
        self._save_map_label.setText("Saving current /map...")
        self._save_map_label.setStyleSheet(
            f"color: {TEXT_DIM}; border: none; background: transparent;")

        def worker():
            try:
                prefix = self._save_occupancy_grid(self._latest_map_msg)
                self.signals.save_status.emit(f"Saved {prefix}.png", True)
            except Exception as exc:
                self.signals.save_status.emit(str(exc), False)

        threading.Thread(target=worker, daemon=True).start()

    def _save_occupancy_grid(self, msg) -> str:
        out_dir = self._map_output_dir()
        os.makedirs(out_dir, exist_ok=True)
        stamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = os.path.join(out_dir, f"pioneer_map_{stamp}")

        data = np.array(msg.data, dtype=np.int16).reshape((msg.info.height, msg.info.width))
        img  = np.full((msg.info.height, msg.info.width), 205, dtype=np.uint8)
        img[data == 0]   = 254
        img[data >= 65]  = 0
        img = np.flipud(img)

        image_path                     = f"{prefix}.png"
        yaml_path                      = f"{prefix}.yaml"
        obstacle_wp_path               = f"{prefix}_obstacle_waypoints.txt"
        latest_obstacle_wp_path        = os.path.join(out_dir, "latest_obstacle_waypoints.txt")

        if not cv2.imwrite(image_path, img):
            raise RuntimeError(f"Could not write {image_path}")

        yaw = self._yaw_from_quaternion(msg.info.origin.orientation)
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(
                f"image: {os.path.basename(image_path)}\n"
                f"mode: trinary\n"
                f"resolution: {msg.info.resolution:.8f}\n"
                f"origin: [{msg.info.origin.position.x:.8f}, "
                f"{msg.info.origin.position.y:.8f}, {yaw:.8f}]\n"
                f"negate: 0\n"
                f"occupied_thresh: 0.65\n"
                f"free_thresh: 0.25\n"
            )

        waypoints = self._coverage_obstacle_standoff_waypoints()
        for path in (obstacle_wp_path, latest_obstacle_wp_path):
            self._write_obstacle_waypoints(path, waypoints)

        return prefix

    def _write_obstacle_waypoints(self, path, waypoints):
        with open(path, "w", encoding="utf-8") as f:
            f.write("# target_rel_x_m target_rel_y_m obstacle_rel_x_m obstacle_rel_y_m\n")
            for tx, ty, ox, oy in waypoints:
                f.write(f"{tx:.3f} {ty:.3f} {ox:.3f} {oy:.3f}\n")

    def _coverage_obstacle_standoff_waypoints(self):
        grid     = self._map_widget
        occupied = grid._obstacle_cells.copy()
        visited  = np.zeros_like(occupied, dtype=bool)
        h, w     = occupied.shape
        clusters = []

        for sr in range(h):
            for sc in range(w):
                if not occupied[sr, sc] or visited[sr, sc]:
                    continue
                stack = [(sr, sc)]; visited[sr, sc] = True; cells = []
                while stack:
                    r, c = stack.pop(); cells.append((r, c))
                    for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
                        nr, nc = r+dr, c+dc
                        if 0<=nr<h and 0<=nc<w and occupied[nr,nc] and not visited[nr,nc]:
                            visited[nr,nc] = True; stack.append((nr,nc))
                if len(cells) >= 3:
                    clusters.append(cells)

        waypoints   = []
        standoff_m  = 1.0
        min_spacing = 0.75
        edge_margin = 0.35
        for cells in clusters:
            avg_row    = sum(r for r, _ in cells) / len(cells)
            avg_col    = sum(c for _, c in cells) / len(cells)
            ox = (avg_col + 0.5) * grid._coverage_res - grid._arena_half
            oy = grid._arena_half - (avg_row + 0.5) * grid._coverage_res
            d  = math.hypot(ox, oy)
            if d < standoff_m + 0.2:
                continue
            tx = ox - (ox/d)*standoff_m
            ty = oy - (oy/d)*standoff_m
            tx = max(-grid._arena_half+edge_margin, min(grid._arena_half-edge_margin, tx))
            ty = max(-grid._arena_half+edge_margin, min(grid._arena_half-edge_margin, ty))
            if all(math.hypot(tx-px, ty-py) >= min_spacing for px, py, *_ in waypoints):
                waypoints.append((tx, ty, ox, oy))

        waypoints.sort(key=lambda item: math.hypot(item[0], item[1]))
        return waypoints[:12]

    def _yaw_from_quaternion(self, q) -> float:
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny, cosy)

    def _map_output_dir(self) -> str:
        env_dir = os.environ.get("PIONEER_MAP_DIR")
        if env_dir:
            return os.path.expanduser(env_dir)
        here = FilePath(__file__).resolve()
        for parent in [here] + list(here.parents):
            if parent.name == "ros2_ws":
                return str(parent / "maps")
        cwd = FilePath.cwd().resolve()
        if cwd.name == "ros2_ws":
            return str(cwd / "maps")
        if FilePath("/ros2_ws").exists():
            return "/ros2_ws/maps"
        return str(cwd / "maps")

    def _on_save_status(self, message: str, ok: bool):
        self._save_map_btn.setEnabled(True)
        colour = GREEN if ok else RED
        self._save_map_label.setText(message)
        self._save_map_label.setStyleSheet(
            f"color: {colour}; border: none; background: transparent;")
        self._det_log.add_entry(message, colour)

    def _tick_clock(self):
        self._clock_label.setText(datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))


# ──────────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────────
def main():
    rclpy.init()
    signals = Signals()
    node    = GUINode(signals)
    signals.mission_command.connect(node.publish_command)

    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = RobotGUI(signals)
    win.show()

    exit_code = app.exec_()
    node.destroy_node()
    rclpy.shutdown()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
