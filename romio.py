#!/usr/bin/env python3
"""
Shoulder Rotation Tracker
Measures interior (medial/internal) and anterior (lateral/external) shoulder rotation
using MediaPipe Pose + Intel RealSense depth camera
"""


# lots of dependencies
import sys
import time
import math
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import PoseLandmarkerOptions, RunningMode
import urllib.request
import os
from collections import deque

# runs locally! Tell a friend!! 

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("[WARN] pyrealsense2 not found — falling back to RGB-only mode")

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QFrame, QPushButton, QSizePolicy, QGraphicsDropShadowEffect
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QPropertyAnimation, QEasingCurve, QRectF, QPointF
from PyQt6.QtGui import (
    QImage, QPixmap, QPainter, QPen, QBrush, QColor, QFont,
    QFontDatabase, QLinearGradient, QRadialGradient, QPainterPath, QConicalGradient
)


PALETTE = {
    "bg":          "#0a0d14",
    "panel":       "#0f1420",
    "border":      "#1c2540",
    "accent_blue": "#3d7fff",
    "accent_cyan": "#00e5ff",
    "accent_left": "#ff6b6b",
    "accent_right":"#6bffb8",
    "text":        "#c8d8ff",
    "text_dim":    "#4a5880",
    "warning":     "#ffcc00",
    "good":        "#00ff9d",
    "neutral":     "#3d7fff",
}

# places landmarks according to the guide at https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
class LANDMARK:
    LEFT_SHOULDER  = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW     = 13
    RIGHT_ELBOW    = 14
    LEFT_WRIST     = 15
    RIGHT_WRIST    = 16
    LEFT_HIP       = 23
    RIGHT_HIP      = 24

MODEL_PATH = os.path.expanduser("~/.cache/mediapipe/pose_landmarker_heavy.task")

def ensure_model(): # grabs model if we don't already have it. thank you linux we say in unison. openpose hates us 
    if os.path.exists(MODEL_PATH):
        return
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    url = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/" #straight up APIing it. and by API let's just say...... it's a link
           "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task")
    print(f"[INFO] Downloading pose model to {MODEL_PATH} …")
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("[INFO] Model downloaded.")

# Pose skeleton connections for manual drawing (landmark index pairs)
POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),(11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
    (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28),
    (27,29),(28,30),(29,31),(30,32),(27,31),(28,32),
]

# Angle smoothing window to reduce jittering blaaaahh can be adjusted as needed
SMOOTH_N = 8

# Angle thresholds for color coding ! 
THRESH_WARNING = 45
THRESH_DANGER  = 90


# math that i dont fully understand if im being so fr 

def vec3(a, b):
    #sets a vector from a to b 
    return np.array([b[0]-a[0], b[1]-a[1], b[2]-a[2]], dtype=float)

def angle_between(v1, v2):
    #computes angle in degrees between two 3d vectors 
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return math.degrees(math.acos(cos_a))

def signed_angle(v1, v2, axis):
    # signed angle around a given axis 
    cross = np.cross(v1, v2)
    sign = np.sign(np.dot(cross, axis))
    return sign * angle_between(v1, v2)

def get_landmark_3d(landmarks, idx, depth_frame=None, w=1, h=1):
    # extract coords from landmarks that were set earlier, grabs body coords basically
    lm = landmarks[idx]
    x_world = lm.x
    y_world = lm.y
    z_world = lm.z if lm.z is not None else 0.0

    if depth_frame is not None:
        px = min(max(int(x_world * w), 0), depth_frame.get_width()-1)
        py = min(max(int(y_world * h), 0), depth_frame.get_height()-1)
        try:
            d = depth_frame.get_distance(px, py)
            if d > 0:
                z_world = d
        except Exception:
            pass

    return np.array([x_world, y_world, z_world])


def compute_shoulder_angles(landmarks, depth_frame=None, w=1, h=1):
    """
    Compute interior (medial/internal) and anterior (lateral/external)
    rotation angles for left and right shoulders.

    Interior rotation  = rotation of the arm *toward* the body midline
    Anterior rotation  = rotation of the arm *forward* from the coronal plane

    Returns dict with keys: left_interior, left_anterior, right_interior, right_anterior
    """
    lm = landmarks

    # grabs landmarks
    L_SHOULDER = get_landmark_3d(lm, LANDMARK.LEFT_SHOULDER, depth_frame, w, h)
    R_SHOULDER = get_landmark_3d(lm, LANDMARK.RIGHT_SHOULDER, depth_frame, w, h)
    L_ELBOW    = get_landmark_3d(lm, LANDMARK.LEFT_ELBOW,    depth_frame, w, h)
    R_ELBOW    = get_landmark_3d(lm, LANDMARK.RIGHT_ELBOW,   depth_frame, w, h)
    L_WRIST    = get_landmark_3d(lm, LANDMARK.LEFT_WRIST,    depth_frame, w, h)
    R_WRIST    = get_landmark_3d(lm, LANDMARK.RIGHT_WRIST,   depth_frame, w, h)
    L_HIP      = get_landmark_3d(lm, LANDMARK.LEFT_HIP,      depth_frame, w, h)
    R_HIP      = get_landmark_3d(lm, LANDMARK.RIGHT_HIP,     depth_frame, w, h)

    # places vectors for directions
    spine    = vec3(((L_HIP+R_HIP)/2), ((L_SHOULDER+R_SHOULDER)/2))  # up
    lateral  = vec3(R_SHOULDER, L_SHOULDER)                            # left to right
    anterior = np.cross(lateral, spine)                                 # forward

    # normalizeeeeeee
    spine    = spine    / (np.linalg.norm(spine)    + 1e-9)
    lateral  = lateral  / (np.linalg.norm(lateral)  + 1e-9)
    anterior = anterior / (np.linalg.norm(anterior) + 1e-9)

    results = {}

    for side, shoulder, elbow, wrist in [
        ("left",  L_SHOULDER, L_ELBOW, L_WRIST),
        ("right", R_SHOULDER, R_ELBOW, R_WRIST),
    ]:
        upper_arm = vec3(shoulder, elbow)
        forearm   = vec3(elbow, wrist)

        # anterior (forward) rotation 
        # Project upper arm onto the coronal plane (lateral times spine)
        # and measure how far it deviates forward/backward.
        ua_norm = upper_arm / (np.linalg.norm(upper_arm) + 1e-9)
        ant_angle = math.degrees(math.asin(np.clip(np.dot(ua_norm, anterior), -1, 1)))

        # interior (medial) rotation 
        # With elbow at 90 degrees, the forearm sweeps inward = internal rotation.
        #  compute the rotation of the forearm around the upper-arm axis.
        ua_axis = ua_norm
        # Reference direction: lateral axis projected to the plane
        ref = lateral - np.dot(lateral, ua_axis) * ua_axis
        ref_n = ref / (np.linalg.norm(ref) + 1e-9)
        # Forearm projected to that same plane
        fa_proj = forearm - np.dot(forearm, ua_axis) * ua_axis
        fa_n = fa_proj / (np.linalg.norm(fa_proj) + 1e-9)

        int_angle = signed_angle(ref_n, fa_n, ua_axis)
        # Flip sign convention for left vs right so that "interior" means
        # toward midline for both arms
        if side == "left":
            int_angle = -int_angle

        results[f"{side}_interior"] = float(int_angle)
        results[f"{side}_anterior"] = float(ant_angle)

    return results # im pretty sure this is how it's done but im confused



def draw_skeleton(bgr, landmarks, w, h): # drawing pose landmarks for skeleton
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in POSE_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(bgr, pts[a], pts[b], (61, 127, 255), 2, cv2.LINE_AA)
    for x, y in pts:
        cv2.circle(bgr, (x, y), 3, (0, 230, 255), -1, cv2.LINE_AA)


class CameraThread(QThread):
    frame_ready = pyqtSignal(object, object, object)  # bgr, depth_frame, angles

    def __init__(self, use_realsense=True):
        super().__init__()
        self.use_realsense = use_realsense and REALSENSE_AVAILABLE
        self.running = False
        self._landmarker = None

    def _make_landmarker(self):
        ensure_model()
        options = PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )
        return mp_vision.PoseLandmarker.create_from_options(options)

    def run(self):
        self.running = True
        self._landmarker = self._make_landmarker()

        if self.use_realsense:
            pipeline = rs.pipeline()
            config   = rs.config()
            config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16,  30)
            align    = rs.align(rs.stream.color)
            pipeline.start(config)
        else:
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  848)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        frame_idx = 0
        t0 = time.perf_counter()

        while self.running:
            depth_frame = None
            try:
                if self.use_realsense:
                    frames      = pipeline.wait_for_frames(timeout_ms=1000)
                    aligned     = align.process(frames)
                    color_frame = aligned.get_color_frame()
                    depth_frame = aligned.get_depth_frame()
                    if not color_frame:
                        continue
                    bgr = np.asanyarray(color_frame.get_data())
                else:
                    ret, bgr = cap.read()
                    if not ret:
                        continue

                h, w = bgr.shape[:2]
                rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                # Build MediaPipe Image
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int((time.perf_counter() - t0) * 1000)
                result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

                angles = None
                if result.pose_landmarks and len(result.pose_landmarks) > 0:
                    lms = result.pose_landmarks[0]
                    draw_skeleton(bgr, lms, w, h)
                    angles = compute_shoulder_angles(lms, depth_frame, w, h)

                self.frame_ready.emit(bgr, depth_frame, angles)
                frame_idx += 1

            except Exception as e:
                print(f"[CameraThread] {e}")
                time.sleep(0.05)

        if self.use_realsense:
            pipeline.stop()
        else:
            cap.release()
        if self._landmarker:
            self._landmarker.close()

    def stop(self):
        self.running = False
        self.wait()


# gauges 
class GaugeWidget(QWidget):

    def __init__(self, label, color_hex, parent=None):
        super().__init__(parent)
        self.label     = label
        self.color     = QColor(color_hex)
        self._value    = 0.0
        self._max_val  = 180.0
        self.setMinimumSize(160, 160)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_value(self, v):
        self._value = float(v)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        side  = min(w, h) - 16
        cx, cy = w / 2, h / 2
        r_outer = side / 2
        r_inner = r_outer * 0.72
        r_track = (r_outer + r_inner) / 2
        arc_w   = r_outer - r_inner

        rect = QRectF(cx - r_outer, cy - r_outer, r_outer*2, r_outer*2)
        rect_i = QRectF(cx - r_inner, cy - r_inner, r_inner*2, r_inner*2)

        # Track arc (background)
        pen_track = QPen(QColor(PALETTE["border"]))
        pen_track.setWidthF(arc_w)
        pen_track.setCapStyle(Qt.PenCapStyle.FlatCap)
        p.setPen(pen_track)
        p.drawArc(
            QRectF(cx - r_track, cy - r_track, r_track*2, r_track*2),
            -210 * 16, -300 * 16
        )

        # Value arc
        frac = min(abs(self._value) / self._max_val, 1.0)
        span = int(frac * 300 * 16)

        grad = QConicalGradient(QPointF(cx, cy), 120)
        c1 = self.color.lighter(80)
        c2 = self.color
        c3 = self.color.lighter(140)
        grad.setColorAt(0.0, c1)
        grad.setColorAt(0.5, c2)
        grad.setColorAt(1.0, c3)

        pen_val = QPen(QBrush(grad), arc_w)
        pen_val.setCapStyle(Qt.PenCapStyle.FlatCap)
        p.setPen(pen_val)
        if span > 0:
            p.drawArc(
                QRectF(cx - r_track, cy - r_track, r_track*2, r_track*2),
                -210 * 16, -span
            )

        # Inner glow circle
        rad_grad = QRadialGradient(cx, cy, r_inner * 0.9)
        rad_grad.setColorAt(0.0, QColor(self.color.red(), self.color.green(), self.color.blue(), 18))
        rad_grad.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(rad_grad))
        p.drawEllipse(QRectF(cx - r_inner, cy - r_inner, r_inner*2, r_inner*2))

        # Value text
        val_txt = f"{self._value:+.1f}°"
        p.setPen(QColor(PALETTE["text"]))
        font = QFont("monospace", int(side * 0.13), QFont.Weight.Bold)
        p.setFont(font)
        fm = p.fontMetrics()
        p.drawText(
            int(cx - fm.horizontalAdvance(val_txt)/2),
            int(cy + fm.ascent()/2 - fm.descent() * 0.2),
            val_txt
        )

        # Label below value
        p.setPen(QColor(PALETTE["text_dim"]))
        lbl_font = QFont("monospace", int(side * 0.07))
        p.setFont(lbl_font)
        fm2 = p.fontMetrics()
        p.drawText(
            int(cx - fm2.horizontalAdvance(self.label)/2),
            int(cy + r_inner * 0.55),
            self.label
        )

        p.end()


# charts for angle history which could be downloaded and sent to records. 
class HistoryChart(QWidget):

    def __init__(self, color_hex, label, parent=None):
        super().__init__(parent)
        self.color  = QColor(color_hex)
        self.label  = label
        self.data   = deque(maxlen=120)
        self.setMinimumHeight(60)

    def push(self, v):
        self.data.append(float(v))
        self.update()

    def paintEvent(self, event):
        if len(self.data) < 2:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        data = list(self.data)
        mn, mx = -180, 180
        rng = mx - mn or 1

        def yx(i, v):
            x = i / (len(data)-1) * w
            y = h - (v - mn) / rng * h
            return x, y

        # Zero line
        p.setPen(QPen(QColor(PALETTE["border"]), 1, Qt.PenStyle.DashLine))
        zy = h - (0 - mn) / rng * h
        p.drawLine(0, int(zy), w, int(zy))

        # Fill
        path_fill = QPainterPath()
        x0, y0 = yx(0, data[0])
        path_fill.moveTo(x0, h)
        path_fill.lineTo(x0, y0)
        for i, v in enumerate(data[1:], 1):
            path_fill.lineTo(*yx(i, v))
        path_fill.lineTo(yx(len(data)-1, data[-1])[0], h)
        path_fill.closeSubpath()

        fill_grad = QLinearGradient(0, 0, 0, h)
        c = self.color
        fill_grad.setColorAt(0.0, QColor(c.red(), c.green(), c.blue(), 80))
        fill_grad.setColorAt(1.0, QColor(c.red(), c.green(), c.blue(), 0))
        p.setBrush(QBrush(fill_grad))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawPath(path_fill)

        # Line
        path = QPainterPath()
        path.moveTo(*yx(0, data[0]))
        for i, v in enumerate(data[1:], 1):
            path.lineTo(*yx(i, v))
        p.setPen(QPen(self.color, 2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)

        # Label
        p.setPen(QColor(PALETTE["text_dim"]))
        p.setFont(QFont("monospace", 8))
        p.drawText(4, 12, self.label)
        p.end()



class ShoulderPanel(QFrame):

    def __init__(self, side: str, color_hex: str, parent=None):
        super().__init__(parent)
        self.side   = side
        self.color  = color_hex
        self._int_buf = deque(maxlen=SMOOTH_N)
        self._ant_buf = deque(maxlen=SMOOTH_N)

        self.setStyleSheet(f"""
            QFrame {{
                background: {PALETTE['panel']};
                border: 1px solid {PALETTE['border']};
                border-radius: 16px;
            }}
        """)
        self.setMinimumWidth(260)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(10)

        # Header
        header = QLabel(f"{'◀ LEFT' if side=='left' else 'RIGHT ▶'}  SHOULDER")
        header.setFont(QFont("monospace", 11, QFont.Weight.Bold))
        header.setStyleSheet(f"color: {color_hex}; border: none; letter-spacing: 2px;")
        layout.addWidget(header)

        # Gauges row
        gauges_row = QHBoxLayout()
        self.gauge_int = GaugeWidget("INTERIOR", color_hex)
        self.gauge_ant = GaugeWidget("ANTERIOR", color_hex)
        gauges_row.addWidget(self.gauge_int)
        gauges_row.addWidget(self.gauge_ant)
        layout.addLayout(gauges_row)

        # Charts
        self.chart_int = HistoryChart(color_hex, "Interior°")
        self.chart_ant = HistoryChart(color_hex, "Anterior°")
        layout.addWidget(self.chart_int)
        layout.addWidget(self.chart_ant)

        # Numeric labels
        nums = QHBoxLayout()
        self.lbl_int = QLabel("INT  ---°")
        self.lbl_ant = QLabel("ANT  ---°")
        for lbl in (self.lbl_int, self.lbl_ant):
            lbl.setFont(QFont("monospace", 9))
            lbl.setStyleSheet(f"color: {PALETTE['text']}; border: none;")
            nums.addWidget(lbl)
        layout.addLayout(nums)

    def update_angles(self, interior: float, anterior: float):
        self._int_buf.append(interior)
        self._ant_buf.append(anterior)
        si = np.mean(self._int_buf)
        sa = np.mean(self._ant_buf)

        self.gauge_int.set_value(si)
        self.gauge_ant.set_value(sa)
        self.chart_int.push(si)
        self.chart_ant.push(sa)
        self.lbl_int.setText(f"INT  {si:+.1f}°")
        self.lbl_ant.setText(f"ANT  {sa:+.1f}°")


# display live
class VideoDisplay(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(f"""
            background: #060810;
            border: 1px solid {PALETTE['border']};
            border-radius: 12px;
        """)
        self.setMinimumSize(480, 320)
        self.setText("Waiting for camera…")
        self.setFont(QFont("monospace", 12))
        self.setStyleSheet(self.styleSheet() + f"color: {PALETTE['text_dim']};")

    def show_frame(self, bgr: np.ndarray):
        h, w, ch = bgr.shape
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        qi  = QImage(rgb.data, w, h, ch*w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qi).scaled(
            self.width(), self.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.setPixmap(pix)


# status
class StatusDot(QWidget):
    def __init__(self, label, parent=None):
        super().__init__(parent)
        self._on = False
        self._color = QColor(PALETTE["neutral"])
        self._label = label
        self.setFixedSize(110, 22)

    def set_status(self, on, color=None):
        self._on = on
        if color:
            self._color = QColor(color)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        c = self._color if self._on else QColor(PALETTE["text_dim"])
        p.setBrush(QBrush(c))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(2, 5, 12, 12)
        p.setPen(c)
        p.setFont(QFont("monospace", 9))
        p.drawText(20, 16, self._label)
        p.end()


# draws main window with colours and graphics
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Shoulder Rotation Tracker")
        self.resize(1200, 720)
        self.setStyleSheet(f"background: {PALETTE['bg']};")

        self._fps_times = deque(maxlen=30)
        self._pose_detected = False

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(10)

        # top bar 
        topbar = QHBoxLayout()
        title = QLabel("SHOULDER ROTATION TRACKER")
        title.setFont(QFont("monospace", 15, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {PALETTE['accent_cyan']}; letter-spacing: 4px;")
        topbar.addWidget(title)
        topbar.addStretch()

        self.dot_pose   = StatusDot("POSE")
        self.dot_depth  = StatusDot("DEPTH")
        self.lbl_fps    = QLabel("FPS: --")
        self.lbl_fps.setFont(QFont("monospace", 9))
        self.lbl_fps.setStyleSheet(f"color: {PALETTE['text_dim']};")
        topbar.addWidget(self.dot_pose)
        topbar.addWidget(self.dot_depth)
        topbar.addWidget(self.lbl_fps)
        root.addLayout(topbar)

        
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"color: {PALETTE['border']};")
        root.addWidget(sep)

        # content
        content = QHBoxLayout()
        content.setSpacing(14)

        self.left_panel  = ShoulderPanel("left",  PALETTE["accent_left"])
        self.right_panel = ShoulderPanel("right", PALETTE["accent_right"])
        self.video       = VideoDisplay()

        content.addWidget(self.left_panel,  3)
        content.addWidget(self.video,       5)
        content.addWidget(self.right_panel, 3)
        root.addLayout(content, 1)

        # bottom info
        info = QLabel(
            "Interior rotation = medial (toward midline) | "
            "Anterior rotation = forward from coronal plane | "
            "Positive = exterior/posterior  Negative = interior/anterior"
        )
        info.setFont(QFont("monospace", 8))
        info.setStyleSheet(f"color: {PALETTE['text_dim']};")
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(info)

        # camera
        self.camera = CameraThread(use_realsense=REALSENSE_AVAILABLE)
        self.camera.frame_ready.connect(self.on_frame)
        self.camera.start()

        # Depth status
        self.dot_depth.set_status(REALSENSE_AVAILABLE, PALETTE["good"] if REALSENSE_AVAILABLE else PALETTE["text_dim"])

    def on_frame(self, bgr, depth_frame, angles):
        # FPS
        now = time.perf_counter()
        self._fps_times.append(now)
        if len(self._fps_times) >= 2:
            fps = (len(self._fps_times)-1) / (self._fps_times[-1] - self._fps_times[0])
            self.lbl_fps.setText(f"FPS: {fps:.0f}")

        # Video
        self.video.show_frame(bgr)

        # Pose status
        detected = angles is not None
        if detected != self._pose_detected:
            self._pose_detected = detected
            self.dot_pose.set_status(detected, PALETTE["good"] if detected else PALETTE["text_dim"])

        # Angles
        if angles:
            self.left_panel.update_angles(
                angles["left_interior"], angles["left_anterior"]
            )
            self.right_panel.update_angles(
                angles["right_interior"], angles["right_anterior"]
            )

    def closeEvent(self, event):
        self.camera.stop()
        event.accept()

# entry
def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Shoulder Rotation Tracker")

    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
