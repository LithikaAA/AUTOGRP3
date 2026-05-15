#!/usr/bin/env python3
"""
AUTO4508 Part 3 - Robot Monitor GUI
====================================
Runs on a LAPTOP connected to the same network as the robot.
Subscribes to ROS2 topics published by the robot's nodes and displays
everything in one window. It never publishes anything — it is read-only.

HOW THE NODES CONNECT TO THIS GUI:
-----------------------------------
  mission_manager.py  →  /robot_state        →  state badge + action label
  control_node.py     →  /robot/pose         →  map arrow + position display
  control_node.py     →  /arena_status       →  arena debug panel
  unified_detector    →  /detected_letter    →  detection log + status panel
  unified_detector    →  /detections/colour  →  detection log + map markers + photo
  slam_toolbox        →  /map                →  map panel background
  mission_manager.py  →  /planned_path       →  path overlay on map
  OAK-D driver        →  /oak/rgb/image_raw  →  camera feed panel

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
import threading
from datetime import datetime

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Pose
from nav_msgs.msg import OccupancyGrid, Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QFrame, QScrollArea, QSizePolicy, QProgressBar
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QPointF
from PyQt5.QtGui import (
    QImage, QPixmap, QFont, QColor, QPainter,
    QPen, QBrush, QPolygonF
)


# ──────────────────────────────────────────────
#  COLOUR PALETTE
#  All colours defined once here so they're
#  easy to change if needed.
# ──────────────────────────────────────────────
BG        = "#0d1117"   # main background
PANEL     = "#161b22"   # panel background
BORDER    = "#30363d"   # panel borders
ACCENT    = "#58a6ff"   # blue — used for headings and neutral highlights
GREEN     = "#3fb950"   # good / mapping
YELLOW    = "#d29922"   # warning / idle
RED       = "#f85149"   # danger / estop
TEXT      = "#e6edf3"   # primary text
TEXT_DIM  = "#8b949e"   # secondary / label text

# Changed from Courier New — using a clean sans-serif
FONT_UI   = "Ubuntu Mono"   # monospaced but less obviously "AI generated"
FONT_BODY = "DejaVu Sans"


# ──────────────────────────────────────────────
#  SIGNALS
#  PyQt5 requires that UI updates happen on the
#  main thread. ROS callbacks run on a background
#  thread. Signals are the thread-safe bridge
#  between them — each signal carries data from
#  a ROS callback to a Qt slot on the main thread.
# ──────────────────────────────────────────────
class Signals(QObject):
    camera_frame    = pyqtSignal(np.ndarray)       # raw BGR frame from OAK-D
    robot_state     = pyqtSignal(str)              # e.g. "MAPPING", "ESTOP"
    robot_pose      = pyqtSignal(float, float, float)  # x, y, yaw (radians)
    letter_detected = pyqtSignal(str)              # e.g. "alpha"
    colour_detected = pyqtSignal(dict)             # JSON dict from /detections/colour
    map_updated     = pyqtSignal(object)           # OccupancyGrid message
    path_updated    = pyqtSignal(object)           # Path message
    arena_updated   = pyqtSignal(dict)             # JSON dict from /arena_status


# ──────────────────────────────────────────────
#  ROS2 NODE
#  This is the only ROS2 node in the GUI.
#  It runs on a background thread (see main())
#  and converts every incoming ROS message into
#  a Qt signal so the UI can update safely.
# ──────────────────────────────────────────────
class GUINode(Node):
    def __init__(self, signals: Signals):
        super().__init__("robot_gui")
        self.signals = signals

        # Each subscription maps one ROS topic to one callback.
        # The callback converts the message and emits a signal.
        self.create_subscription(Image,         "/oak/rgb/image_raw", self._cb_camera,  10)
        self.create_subscription(String,        "/robot_state",       self._cb_state,   10)
        self.create_subscription(Pose,          "/robot/pose",        self._cb_pose,    10)
        self.create_subscription(String,        "/detected_letter",   self._cb_letter,  10)
        self.create_subscription(String,        "/detections/colour", self._cb_colour,  10)
        self.create_subscription(OccupancyGrid, "/map",               self._cb_map,     10)
        self.create_subscription(Path,          "/planned_path",      self._cb_path,    10)
        self.create_subscription(String,        "/arena_status",      self._cb_arena,   10)

    def _cb_camera(self, msg):
        # Convert raw bytes to numpy array, fix channel order if needed
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            (msg.height, msg.width, -1))
        if msg.encoding != "bgr8":
            # OAK-D publishes rgb8 by default — swap to BGR for OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.signals.camera_frame.emit(frame)

    def _cb_state(self, msg):
        # Published by mission_manager.py at 2Hz
        self.signals.robot_state.emit(msg.data)

    def _cb_pose(self, msg):
        # Published by control_node.py every odom tick
        # Extract yaw angle from quaternion for the map arrow rotation
        q    = msg.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw  = math.atan2(siny, cosy)
        self.signals.robot_pose.emit(msg.position.x, msg.position.y, yaw)

    def _cb_letter(self, msg):
        # Published by unified_detector_node.py when a greek letter is confirmed
        self.signals.letter_detected.emit(msg.data)

    def _cb_colour(self, msg):
        # Published by unified_detector_node.py as a JSON string
        # Contains label, distance, bearing, robot position, photo path
        try:
            self.signals.colour_detected.emit(json.loads(msg.data))
        except Exception:
            pass

    def _cb_map(self, msg):
        # Published by slam_toolbox during mapping phase
        self.signals.map_updated.emit(msg)

    def _cb_path(self, msg):
        # Published by mission_manager.py during waypoint phase
        self.signals.path_updated.emit(msg)

    def _cb_arena(self, msg):
        # Published by control_node.py — contains drive state, distances etc.
        try:
            self.signals.arena_updated.emit(json.loads(msg.data))
        except Exception:
            pass


# ──────────────────────────────────────────────
#  UI HELPER FUNCTIONS
# ──────────────────────────────────────────────

def make_panel(title: str) -> tuple:
    """
    Creates a styled panel with a title bar.
    Returns (QFrame, QVBoxLayout) — add widgets to the layout.
    """
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
    """
    Adds a key/value row to a grid layout.
    Returns the value QLabel so it can be updated later.
    """
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
#  MAP WIDGET
#  Draws the occupancy grid map, the robot's
#  position as an arrow, detection markers,
#  and the planned path as a dashed line.
# ──────────────────────────────────────────────
class MapWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(300, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(f"background: {BG};")

        # Map metadata — updated when /map is received
        self._map_img   = None   # QImage of the occupancy grid
        self._map_res   = 0.05   # metres per cell
        self._map_ox    = 0.0    # map origin x (world coords)
        self._map_oy    = 0.0    # map origin y (world coords)
        self._map_w     = 0      # map width in cells
        self._map_h     = 0      # map height in cells

        # Robot pose — updated from /robot/pose
        self._robot_x   = 0.0
        self._robot_y   = 0.0
        self._robot_yaw = 0.0

        # Detection markers — accumulated over the run
        # Each entry: (world_x, world_y, label_string, colour_string)
        self._detections = []

        # Planned path — list of (world_x, world_y) tuples
        self._path = []

    def update_map(self, msg):
        """Convert OccupancyGrid message to a QImage for painting."""
        self._map_res = msg.info.resolution
        self._map_ox  = msg.info.origin.position.x
        self._map_oy  = msg.info.origin.position.y
        self._map_w   = msg.info.width
        self._map_h   = msg.info.height

        data = np.array(msg.data, dtype=np.int8).reshape((self._map_h, self._map_w))
        img  = np.zeros((self._map_h, self._map_w, 3), dtype=np.uint8)

        # -1 = unknown (grey), 0 = free (light), >50 = occupied (dark)
        img[data == -1] = [40,  40,  40]
        img[data == 0]  = [200, 200, 200]
        img[data > 50]  = [20,  20,  20]

        # ROS map origin is bottom-left, Qt is top-left — flip vertically
        img = np.flipud(img)
        h, w, _ = img.shape
        self._map_img = QImage(img.tobytes(), w, h, 3*w, QImage.Format_RGB888)
        self.update()   # trigger repaint

    def update_pose(self, x, y, yaw):
        self._robot_x   = x
        self._robot_y   = y
        self._robot_yaw = yaw
        self.update()

    def add_detection(self, x, y, label, colour):
        self._detections.append((x, y, label, colour))
        self.update()

    def update_path(self, poses):
        self._path = [(p.pose.position.x, p.pose.position.y) for p in poses]
        self.update()

    def _world_to_px(self, wx, wy):
        """
        Convert world coordinates (metres) to widget pixel coordinates.
        Accounts for map scale and centres the map in the widget.
        """
        if self._map_w == 0 or self._map_h == 0:
            return (self.width()//2, self.height()//2)

        # World → grid cell
        gx    = (wx - self._map_ox) / self._map_res
        gy    = self._map_h - (wy - self._map_oy) / self._map_res  # flip y

        # Scale to fit widget while keeping aspect ratio
        scale = min(self.width() / self._map_w, self.height() / self._map_h)
        px = int(gx * scale + (self.width()  - self._map_w * scale) / 2)
        py = int(gy * scale + (self.height() - self._map_h * scale) / 2)
        return (px, py)

    def paintEvent(self, event):
        """Called by Qt whenever the widget needs to be redrawn."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(BG))

        # Draw occupancy grid map
        if self._map_img:
            scale = min(self.width() / self._map_w, self.height() / self._map_h)
            dw    = int(self._map_w * scale)
            dh    = int(self._map_h * scale)
            dx    = (self.width()  - dw) // 2
            dy    = (self.height() - dh) // 2
            pix   = QPixmap.fromImage(self._map_img).scaled(
                dw, dh, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            painter.drawPixmap(dx, dy, pix)
        else:
            painter.setPen(QColor(TEXT_DIM))
            painter.setFont(QFont(FONT_UI, 10))
            painter.drawText(self.rect(), Qt.AlignCenter, "Waiting for /map...")

        # Draw planned path as dashed line
        if len(self._path) >= 2:
            painter.setPen(QPen(QColor(ACCENT), 2, Qt.DashLine))
            for i in range(len(self._path) - 1):
                p1 = self._world_to_px(*self._path[i])
                p2 = self._world_to_px(*self._path[i+1])
                painter.drawLine(*p1, *p2)

        # Draw detection markers as coloured dots
        for (wx, wy, label, col) in self._detections:
            px, py = self._world_to_px(wx, wy)
            colour = QColor(RED) if "red" in col else \
                     QColor(YELLOW) if "yellow" in col else QColor(GREEN)
            painter.setBrush(QBrush(colour))
            painter.setPen(QPen(QColor(TEXT), 1))
            painter.drawEllipse(px-6, py-6, 12, 12)
            painter.setFont(QFont(FONT_UI, 7))
            painter.setPen(QColor(TEXT))
            painter.drawText(px+8, py+4, label[:3])

        # Draw robot as a directional arrow
        rx, ry = self._world_to_px(self._robot_x, self._robot_y)
        painter.save()
        painter.translate(rx, ry)
        painter.rotate(-math.degrees(self._robot_yaw))
        arrow = QPolygonF([
            QPointF(0, -12), QPointF(-7, 8),
            QPointF(0, 4),   QPointF(7, 8),
        ])
        painter.setBrush(QBrush(QColor(ACCENT)))
        painter.setPen(QPen(QColor(TEXT), 1))
        painter.drawPolygon(arrow)
        painter.restore()
        painter.end()


# ──────────────────────────────────────────────
#  DETECTION LOG WIDGET
#  A scrolling list of timestamped log entries.
#  Each entry is colour-coded by type.
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
        # Insert before the bottom stretch so new entries appear at the bottom
        count = self._inner_layout.count()
        self._inner_layout.insertWidget(count - 1, lbl)
        # Auto-scroll to latest entry
        QTimer.singleShot(50, lambda: self._scroll.verticalScrollBar().setValue(
            self._scroll.verticalScrollBar().maximum()))


# ──────────────────────────────────────────────
#  ARENA DEBUG PANEL
#  Shows the live arena status data published
#  by control_node.py on /arena_status.
#  Includes a colour-changing progress bar for
#  edge clearance (how close the robot is to
#  the boundary of the 15x15m area).
# ──────────────────────────────────────────────
class ArenaPanel(QWidget):
    ARENA_HALF = 7.5   # half of 15m arena

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        grid = QGridLayout()
        grid.setSpacing(4)
        grid.setColumnStretch(1, 1)

        # These labels are updated every time /arena_status is received
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

        # Progress bar: full = at centre, empty = at boundary
        # Colour changes green → yellow → red as robot approaches boundary
        self._edge_bar = QProgressBar()
        self._edge_bar.setRange(0, 100)
        self._edge_bar.setValue(100)
        self._edge_bar.setTextVisible(False)
        self._edge_bar.setFixedHeight(8)
        self._set_bar_colour(GREEN)
        layout.addWidget(self._edge_bar)

    def _set_bar_colour(self, colour: str):
        self._edge_bar.setStyleSheet(f"""
            QProgressBar {{
                background: {BORDER};
                border: none;
                border-radius: 4px;
            }}
            QProgressBar::chunk {{
                background: {colour};
                border-radius: 4px;
            }}
        """)

    def update(self, data: dict):
        """Called every time a new /arena_status message arrives."""
        state  = data.get('state', '—')
        rel_x  = data.get('rel_x', 0.0)
        rel_y  = data.get('rel_y', 0.0)
        home   = data.get('center_dist', 0.0)
        edge   = data.get('edge_clearance', 0.0)
        yaw    = data.get('yaw', 0.0)
        source = data.get('pose_source', '—')

        # Colour-code the drive state label
        state_colours = {
            'WANDERING_DRIVE':       GREEN,
            'WANDERING_TURN':        ACCENT,
            'OBSTACLE_REVERSE':      YELLOW,
            'OBSTACLE_TURN':         YELLOW,
            'BOUNDARY_REVERSE':      RED,
            'BOUNDARY_ESCAPE_TURN':  RED,
            'BOUNDARY_ESCAPE_DRIVE': RED,
            'RETURN_TO_CENTER':      YELLOW,
            'INITIAL_TURN':          TEXT_DIM,
            'INITIAL_DRIVE':         TEXT_DIM,
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

        # Update progress bar — green when safe, yellow when close, red when critical
        pct = min(100, int(edge / self.ARENA_HALF * 100))
        self._edge_bar.setValue(pct)
        if pct > 50:
            self._set_bar_colour(GREEN)
        elif pct > 20:
            self._set_bar_colour(YELLOW)
        else:
            self._set_bar_colour(RED)


# ──────────────────────────────────────────────
#  MAIN WINDOW
#  Layout:
#    LEFT   — camera feed, robot status, arena debug
#    MIDDLE — map
#    RIGHT  — detection log, last photo
# ──────────────────────────────────────────────
class RobotGUI(QMainWindow):
    def __init__(self, signals: Signals):
        super().__init__()
        self.signals = signals
        self.setWindowTitle("AUTO4508 — Pioneer 3-AT Monitor")
        self.setMinimumSize(1400, 800)
        self.setStyleSheet(f"background: {BG}; color: {TEXT};")

        self._build_ui()
        self._connect_signals()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # ── Header bar ──────────────────────────────
        header = QHBoxLayout()
        title  = QLabel("AUTO4508  ·  Pioneer 3-AT")
        title.setFont(QFont(FONT_BODY, 13, QFont.Bold))
        title.setStyleSheet(f"color: {ACCENT};")
        header.addWidget(title)
        header.addStretch()

        # State badge — updates colour based on robot state
        self._state_badge = QLabel("IDLE")
        self._state_badge.setFont(QFont(FONT_UI, 11, QFont.Bold))
        self._state_badge.setAlignment(Qt.AlignCenter)
        self._state_badge.setFixedHeight(32)
        self._state_badge.setStyleSheet(self._badge_style(YELLOW))
        header.addWidget(self._state_badge)
        root.addLayout(header)

        # ── Main content row ────────────────────────
        content = QHBoxLayout()
        content.setSpacing(10)

        # LEFT COLUMN
        left = QVBoxLayout()
        left.setSpacing(10)

        # Camera feed — from /oak/rgb/image_raw via unified_detector
        cam_frame, cam_layout = make_panel("Camera Feed")
        self._cam_label = QLabel()
        self._cam_label.setAlignment(Qt.AlignCenter)
        self._cam_label.setMinimumSize(480, 270)
        self._cam_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._cam_label.setStyleSheet("background: #000; border: none;")
        self._cam_label.setText("No camera feed")
        cam_layout.addWidget(self._cam_label)
        left.addWidget(cam_frame, 3)

        # Robot status — position/heading from control_node, state from mission_manager
        status_frame, status_layout = make_panel("Robot Status")
        sg = QGridLayout()
        sg.setSpacing(6)
        self._status_labels = {}
        for i, (key, val) in enumerate([
            ("State",       "IDLE"),
            ("Action",      "—"),
            ("Pos X",       "0.00 m"),
            ("Pos Y",       "0.00 m"),
            ("Heading",     "0.0°"),
            ("Last Letter", "—"),
        ]):
            self._status_labels[key] = status_row(sg, i, key, val)
        status_layout.addLayout(sg)
        left.addWidget(status_frame, 1)

        # Arena debug — from control_node /arena_status
        arena_frame, arena_layout = make_panel("Arena Debug")
        self._arena_panel = ArenaPanel()
        arena_layout.addWidget(self._arena_panel)
        left.addWidget(arena_frame, 1)

        content.addLayout(left, 5)

        # MIDDLE COLUMN — map from slam_toolbox /map
        map_frame, map_layout = make_panel("Map")
        self._map_widget = MapWidget()
        map_layout.addWidget(self._map_widget)
        content.addWidget(map_frame, 4)

        # RIGHT COLUMN
        right = QVBoxLayout()
        right.setSpacing(10)

        # Detection log — entries from /detected_letter and /detections/colour
        log_frame, log_layout = make_panel("Detection Log")
        self._det_log = DetectionLog()
        self._det_log.setMinimumHeight(200)
        log_layout.addWidget(self._det_log)
        right.addWidget(log_frame, 3)

        # Last photo — saved by unified_detector_node to ~/part3_logs/
        photo_frame, photo_layout = make_panel("Last Detection Photo")
        self._photo_label = QLabel()
        self._photo_label.setAlignment(Qt.AlignCenter)
        self._photo_label.setMinimumHeight(160)
        self._photo_label.setStyleSheet("background: #000; border: none;")
        self._photo_label.setText("No photo yet")
        photo_layout.addWidget(self._photo_label)
        right.addWidget(photo_frame, 2)

        content.addLayout(right, 3)
        root.addLayout(content, 1)

        # ── Bottom status bar ───────────────────────
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

        # Clock updates every second
        self._clock_timer = QTimer()
        self._clock_timer.timeout.connect(self._tick_clock)
        self._clock_timer.start(1000)
        self._tick_clock()

    def _badge_style(self, colour: str) -> str:
        return (f"background: {colour}22; color: {colour}; "
                f"border: 1px solid {colour}; border-radius: 6px; padding: 2px 14px;")

    def _connect_signals(self):
        """Wire each signal from GUINode to its handler slot."""
        self.signals.camera_frame.connect(self._on_camera)
        self.signals.robot_state.connect(self._on_state)
        self.signals.robot_pose.connect(self._on_pose)
        self.signals.letter_detected.connect(self._on_letter)
        self.signals.colour_detected.connect(self._on_colour)
        self.signals.map_updated.connect(self._on_map)
        self.signals.path_updated.connect(self._on_path)
        self.signals.arena_updated.connect(self._on_arena)

    # ── Slot handlers ────────────────────────────
    # These run on the Qt main thread, safe to update UI

    def _on_camera(self, frame: np.ndarray):
        """Display latest camera frame in the camera panel."""
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.tobytes(), w, h, ch*w, QImage.Format_RGB888)
        pix  = QPixmap.fromImage(qimg).scaled(
            self._cam_label.width(), self._cam_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._cam_label.setPixmap(pix)

    def _on_state(self, state: str):
        """Update state badge colour and action description."""
        self._status_labels["State"].setText(state)
        self._state_badge.setText(state)

        colour_map = {
            "MAPPING":          GREEN,
            "WAYPOINT":         ACCENT,
            "IDLE":             YELLOW,
            "STOPPED":          RED,
            "ESTOP":            RED,
            "RETURN_TO_CENTER": YELLOW,
        }
        colour = colour_map.get(state, TEXT_DIM)
        self._state_badge.setStyleSheet(self._badge_style(colour))

        action_map = {
            "MAPPING":  "Exploring area and building map...",
            "WAYPOINT": "Driving to waypoints at maximum speed...",
            "IDLE":     "Standing by.",
            "STOPPED":  "EMERGENCY STOP — all motion halted!",
            "ESTOP":    "EMERGENCY STOP — obstacle detected!",
            "RETURN_TO_CENTER": "Returning to arena centre...",
        }
        action = action_map.get(state, state)
        self._status_labels["Action"].setText(action)
        self._action_label.setText(action)
        self._det_log.add_entry(f"State → {state}", colour)

    def _on_pose(self, x: float, y: float, yaw: float):
        """Update position display and move robot arrow on map."""
        self._status_labels["Pos X"].setText(f"{x:.2f} m")
        self._status_labels["Pos Y"].setText(f"{y:.2f} m")
        self._status_labels["Heading"].setText(f"{math.degrees(yaw):.1f}°")
        self._map_widget.update_pose(x, y, yaw)

    def _on_letter(self, name: str):
        """Log a detected greek letter."""
        self._status_labels["Last Letter"].setText(name)
        self._det_log.add_entry(f"Greek letter detected: {name}", GREEN)

    def _on_colour(self, data: dict):
        """Log a colour detection, place marker on map, show photo."""
        label      = data.get("label", "unknown")
        dist       = data.get("distance_m", 0.0)
        bearing    = data.get("bearing_deg", 0.0)
        rx         = data.get("robot_x", 0.0)
        ry         = data.get("robot_y", 0.0)
        photo_path = data.get("photo_path", "")

        colour = RED if "red" in label else YELLOW
        self._det_log.add_entry(
            f"{label}  dist={dist:.2f}m  bearing={bearing:.1f}°", colour)

        # Estimate world position of detected object from robot pose + bearing
        wx = rx + dist * math.cos(math.radians(bearing))
        wy = ry + dist * math.sin(math.radians(bearing))
        self._map_widget.add_detection(wx, wy, label, label)

        # Load and display photo if the detector saved one
        if photo_path:
            img = cv2.imread(photo_path)
            if img is not None:
                rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qimg = QImage(rgb.tobytes(), w, h, ch*w, QImage.Format_RGB888)
                pix  = QPixmap.fromImage(qimg).scaled(
                    self._photo_label.width(), self._photo_label.height(),
                    Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self._photo_label.setPixmap(pix)

    def _on_map(self, msg):
        self._map_widget.update_map(msg)

    def _on_path(self, msg):
        self._map_widget.update_path(msg.poses)
        self._det_log.add_entry(
            f"Path updated — {len(msg.poses)} waypoints", ACCENT)

    def _on_arena(self, data: dict):
        self._arena_panel.update(data)

    def _tick_clock(self):
        self._clock_label.setText(datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))


# ──────────────────────────────────────────────
#  ENTRY POINT
#  ROS2 runs on a background daemon thread.
#  Qt must run on the main thread.
#  Signals safely bridge data between them.
# ──────────────────────────────────────────────
def main():
    rclpy.init()
    signals = Signals()
    node    = GUINode(signals)

    # Spin ROS2 in background — this processes all incoming messages
    ros_thread = threading.Thread(
        target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    # Qt takes over the main thread
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
