"""
hyoid_tracking_main_code.py

Real-time multi-target hyoid-tracking tool for ultrasound research.

Provides a dual-window GUI:
  * Experimenter View  - setup, control panel, annotated feed with bounding boxes/FPS.
  * Participant View    - configurable overlay with five display modes.

Core capabilities
-----------------
  * Screen-region capture (multi-monitor aware) as the live video source.
  * Multi-target tracking with OpenCV CSRT trackers in a local search region.
  * Optional Kalman filter smoothing (6-state: position, velocity, size).
  * ORB-based automatic re-initialisation of lost trackers (homography).
  * Manual re-initialisation by drawing a fresh ROI.
  * Per-tracker motion trails + swallow trajectory recording.
  * Draggable stepped-gradient reference box, synced across both views.
  * Swallow strength meter (mode 4) and speedometer (mode 5).
  * Participant-view zoom (auto or custom region).
  * CSV export of per-frame/per-tracker data and annotated MP4 recording.
  * Keyboard shortcuts for all common actions.hyoid_tracking_main_code.py

Dependencies: opencv-contrib-python, PySide6, numpy, pandas
"""

import sys
import time
import math

import cv2
import numpy as np
import pandas as pd

from PySide6.QtCore import QThread, Signal, Slot, Qt, QRect, QPoint, QTimer
from PySide6.QtGui import (
    QImage, QPixmap, QPainter, QPen, QColor, QGuiApplication,
    QKeySequence, QShortcut,
)
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QSpinBox, QComboBox, QMessageBox, QCheckBox, QSizePolicy,
    QSlider, QDialog, QDialogButtonBox, QScrollArea, QLineEdit,
)

# Output file names
OUTPUT_VIDEO = "tracked_output_gui.mp4"

# Secondary-view mode identifiers
MODE_COPY = 1          # mirrored frame only
MODE_BLACK_BOX = 2     # black background + gradient box + tracker dots
MODE_FRAME_BOX = 3     # frame background + gradient box + tracker dots
MODE_STRENGTH = 4      # swallow strength meter
MODE_SPEED = 5         # swallow speedometer

MODE_NAMES = {
    MODE_COPY: "Copy",
    MODE_BLACK_BOX: "Black+Box",
    MODE_FRAME_BOX: "Frame+Box",
    MODE_STRENGTH: "Strength Meter",
    MODE_SPEED: "Speedometer",
}


# =====================================================================
# Geometry / Kalman helper functions
# =====================================================================
def create_kalman_from_roi(roi, dt=1.0):
    """Build a 6-state (cx, cy, vx, vy, w, h) constant-velocity Kalman filter.

    Measurement vector is (cx, cy, w, h).
    """
    kf = cv2.KalmanFilter(6, 4)
    kf.transitionMatrix = np.array([
        [1, 0, dt, 0, 0, 0],
        [0, 1, 0, dt, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ], np.float32)
    kf.measurementMatrix = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ], np.float32)
    kf.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-2
    kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
    kf.errorCovPost = np.eye(6, dtype=np.float32)
    x, y, w, h = roi
    cx = x + w / 2.0
    cy = y + h / 2.0
    kf.statePost = np.array([[cx], [cy], [0.], [0.], [w], [h]], dtype=np.float32)
    return kf


def expand_roi(roi, frac, frame_w, frame_h):
    """Pad a ROI by ``frac`` of its size on each side, clamped to the frame."""
    x, y, w, h = roi
    pad_x = int(w * frac)
    pad_y = int(h * frac)
    nx = max(0, x - pad_x)
    ny = max(0, y - pad_y)
    nw = min(frame_w - nx, w + 2 * pad_x)
    nh = min(frame_h - ny, h + 2 * pad_y)
    return (nx, ny, nw, nh)


def clamp_roi_to_frame(roi, frame_w, frame_h):
    """Clamp a ROI so it lies fully inside the frame with positive size."""
    x, y, w, h = map(int, roi)
    x = max(0, min(x, frame_w - 1))
    y = max(0, min(y, frame_h - 1))
    w = max(1, min(w, frame_w - x))
    h = max(1, min(h, frame_h - y))
    return (x, y, w, h)


def centered_search_region(roi, frame_w, frame_h, area_fraction=0.25):
    """Return a search window centred on the ROI covering ``area_fraction``
    of the frame area (never smaller than the ROI itself)."""
    x, y, w, h = map(float, roi)
    cx = x + w / 2.0
    cy = y + h / 2.0
    side_scale = max(0.05, float(area_fraction)) ** 0.5
    search_w = max(int(round(frame_w * side_scale)), int(round(w)))
    search_h = max(int(round(frame_h * side_scale)), int(round(h)))
    sx = int(round(cx - search_w / 2.0))
    sy = int(round(cy - search_h / 2.0))
    sx = max(0, min(sx, frame_w - search_w))
    sy = max(0, min(sy, frame_h - search_h))
    search_w = min(search_w, frame_w - sx)
    search_h = min(search_h, frame_h - sy)
    return (sx, sy, search_w, search_h)


def bgr_to_qimage(bgr):
    """Convert an OpenCV BGR ndarray to a deep-copied RGB888 QImage."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()


# =====================================================================
# Drawable label - lets the experimenter draw ROIs and drag the box
# =====================================================================
class DrawableLabel(QLabel):
    box_offset_changed = Signal(int)   # absolute box_x_offset in secondary-image coords
    measure_points_ready = Signal(QPoint, QPoint)  # two clicked points (widget coords)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = None
        self.drawing = False
        self.start_point = QPoint()
        self.end_point = QPoint()
        self.rects = []
        self.temp_rect = None
        self.draw_mode = False
        self.setMouseTracking(True)
        # point-to-point measurement state
        self.measure_mode = False
        self._measure_pts = []       # up to 2 QPoint in widget coords
        self._measure_preview = None  # live cursor point before the 2nd click
        self._measure_text = ""      # distance label drawn near the line
        # box-drag state
        self._box_drag_mode = False
        self._box_drag_start_x = None
        self._box_drag_start_offset = 0
        self._box_current_offset = 0
        self._pix_w = 1   # width of last displayed pixmap (coord mapping)
        self._sec_w = 1   # secondary image width (coord mapping)

    # --- pixmap / coordinate bookkeeping ---
    def setPixmap(self, pixmap: QPixmap):
        super().setPixmap(pixmap)
        self._pixmap = pixmap
        self._pix_w = max(1, pixmap.width())

    def set_sec_w(self, w: int):
        self._sec_w = max(1, w)

    # --- box drag ---
    def set_box_drag_mode(self, enabled: bool):
        self._box_drag_mode = enabled
        self.setCursor(Qt.SizeHorCursor if enabled else Qt.ArrowCursor)

    def sync_offset(self, offset: int):
        """Adopt an offset set elsewhere without re-emitting."""
        self._box_current_offset = offset
        self._box_drag_start_offset = offset

    # --- ROI drawing ---
    def enter_draw_mode(self):
        self.draw_mode = True
        self.rects = []
        self.temp_rect = None
        self.update()

    def exit_draw_mode(self):
        self.draw_mode = False
        self.temp_rect = None
        self.update()

    def get_rects_display(self):
        return list(self.rects)

    def clear_rects(self):
        self.rects = []
        self.temp_rect = None
        self.update()

    # --- point-to-point measurement ---
    def enter_measure_mode(self):
        self.measure_mode = True
        self._measure_pts = []
        self._measure_preview = None
        self._measure_text = ""
        self.setCursor(Qt.CrossCursor)
        self.update()

    def exit_measure_mode(self):
        self.measure_mode = False
        self._measure_pts = []
        self._measure_preview = None
        self._measure_text = ""
        self.setCursor(Qt.ArrowCursor)
        self.update()

    def set_measure_text(self, text: str):
        self._measure_text = text
        self.update()

    # --- mouse events ---
    def mousePressEvent(self, event):
        if self.measure_mode and event.button() == Qt.LeftButton:
            if len(self._measure_pts) >= 2:      # start a fresh measurement
                self._measure_pts = []
                self._measure_text = ""
            self._measure_pts.append(event.pos())
            self._measure_preview = event.pos()
            if len(self._measure_pts) == 2:
                self.measure_points_ready.emit(self._measure_pts[0], self._measure_pts[1])
            self.update()
            return
        if self._box_drag_mode and event.button() == Qt.LeftButton:
            self._box_drag_start_x = event.pos().x()
            self._box_drag_start_offset = self._box_current_offset
            return
        if self.draw_mode and event.button() == Qt.LeftButton:
            self.drawing = True
            self.start_point = event.pos()
            self.end_point = event.pos()
            self.temp_rect = QRect(self.start_point, self.end_point)
            self.update()

    def mouseMoveEvent(self, event):
        if self.measure_mode:
            if len(self._measure_pts) == 1:
                self._measure_preview = event.pos()
                self.update()
            return
        if self._box_drag_mode and self._box_drag_start_x is not None:
            dx = event.pos().x() - self._box_drag_start_x
            dx_image = int(dx * self._sec_w / self._pix_w)
            self._box_current_offset = self._box_drag_start_offset + dx_image
            self.box_offset_changed.emit(self._box_current_offset)
            return
        if self.draw_mode and self.drawing:
            self.end_point = event.pos()
            self.temp_rect = QRect(self.start_point, self.end_point).normalized()
            self.update()

    def mouseReleaseEvent(self, event):
        if self._box_drag_mode and event.button() == Qt.LeftButton:
            self._box_drag_start_x = None
            return
        if self.draw_mode and event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            self.end_point = event.pos()
            rect = QRect(self.start_point, self.end_point).normalized()
            if rect.width() > 5 and rect.height() > 5:
                self.rects.append(rect)
            self.temp_rect = None
            self.update()

    def mouseDoubleClickEvent(self, event):
        if self._box_drag_mode and event.button() == Qt.LeftButton:
            self._box_current_offset = 0
            self._box_drag_start_offset = 0
            self.box_offset_changed.emit(0)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._pixmap is None:
            return
        painter = QPainter(self)
        painter.setPen(QPen(Qt.green, 2))
        for r in self.rects:
            painter.drawRect(r)
        if self.temp_rect is not None:
            painter.setPen(QPen(Qt.yellow, 2))
            painter.drawRect(self.temp_rect)
        if self.measure_mode and self._measure_pts:
            self._paint_measurement(painter)
        painter.end()

    def _paint_measurement(self, painter):
        p0 = self._measure_pts[0]
        p1 = self._measure_pts[1] if len(self._measure_pts) >= 2 else self._measure_preview
        # connecting line
        if p1 is not None:
            painter.setPen(QPen(QColor(0, 220, 255), 2))
            painter.drawLine(p0, p1)
        # endpoint markers
        painter.setPen(QPen(Qt.yellow, 2))
        painter.setBrush(QColor(0, 220, 255))
        for pt in self._measure_pts:
            painter.drawEllipse(pt, 4, 4)
        painter.setBrush(Qt.NoBrush)
        # distance label near the midpoint
        if self._measure_text and p1 is not None:
            mx = (p0.x() + p1.x()) // 2
            my = (p0.y() + p1.y()) // 2
            metrics = painter.fontMetrics()
            tw = metrics.horizontalAdvance(self._measure_text)
            th = metrics.height()
            bx, by = mx + 8, my - th - 4
            painter.fillRect(bx - 3, by - 2, tw + 6, th + 4, QColor(0, 0, 0, 180))
            painter.setPen(QPen(QColor(0, 220, 255), 1))
            painter.drawText(bx, by + th - 4, self._measure_text)


# =====================================================================
# Fullscreen overlay used to select the screen-capture region
# =====================================================================
class ScreenRegionSelector(QWidget):
    def __init__(self, screenshot, screen_geometry, parent=None):
        super().__init__(parent)
        self.screenshot = screenshot
        self.screen_geometry = screen_geometry
        self.start_point = None
        self.end_point = None
        self.selected_rect = None
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setCursor(Qt.CrossCursor)
        self.setGeometry(screen_geometry)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start_point = event.pos()
            self.end_point = event.pos()
            self.update()

    def mouseMoveEvent(self, event):
        if self.start_point is not None:
            self.end_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.start_point is not None:
            self.end_point = event.pos()
            rect = QRect(self.start_point, self.end_point).normalized()
            if rect.width() > 5 and rect.height() > 5:
                # Map widget-proportional coords -> logical screen coords so the
                # selection is correct regardless of DPI scaling or window size.
                w = max(1, self.width())
                h = max(1, self.height())
                geo = self.screen_geometry
                self.selected_rect = QRect(
                    geo.x() + int(rect.x() * geo.width() / w),
                    geo.y() + int(rect.y() * geo.height() / h),
                    int(rect.width() * geo.width() / w),
                    int(rect.height() * geo.height() / h),
                )
            self.close()

    def paintEvent(self, event):
        painter = QPainter(self)
        widget_rect = self.rect()
        w = max(1, widget_rect.width())
        h = max(1, widget_rect.height())
        ss_w = max(1, self.screenshot.width())
        ss_h = max(1, self.screenshot.height())
        painter.fillRect(widget_rect, Qt.black)
        painter.setOpacity(0.65)
        painter.drawPixmap(widget_rect, self.screenshot)
        painter.setOpacity(1.0)
        if self.start_point is not None and self.end_point is not None:
            rect = QRect(self.start_point, self.end_point).normalized()
            # Map selection (widget coords) -> screenshot raw pixel coords.
            src_rect = QRect(
                int(rect.x() * ss_w / w),
                int(rect.y() * ss_h / h),
                int(rect.width() * ss_w / w),
                int(rect.height() * ss_h / h),
            )
            painter.drawPixmap(rect, self.screenshot, src_rect)
            painter.setPen(QPen(Qt.green, 2))
            painter.drawRect(rect)
        painter.end()


# =====================================================================
# Participant View - shows the secondary image; supports box dragging
# =====================================================================
class SecondaryWindow(QWidget):
    box_offset_changed = Signal(int)   # absolute box_x_offset in image coords
    size_changed = Signal(int, int)    # (w, h) on every resize

    def __init__(self, width=320, height=240):
        super().__init__()
        self.setWindowTitle("Participant View")
        screens = QGuiApplication.screens()
        last_screen = screens[-1] if screens else None
        if last_screen is not None:
            self.move(last_screen.geometry().topLeft())
        else:
            self.move(1020, 30)
        self.label = QLabel()
        self.label.setMinimumSize(80, 60)
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label)
        self.setLayout(layout)
        self.resize(width, height)
        self.mode = MODE_COPY
        # box-drag state
        self._drag_mode = False
        self._drag_start_x = None
        self._drag_start_offset = 0
        self._current_offset = 0
        self._image_w = 1
        self.setMouseTracking(True)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.size_changed.emit(event.size().width(), event.size().height())

    def update_image(self, qt_img):
        self._image_w = max(1, qt_img.width())
        pix = QPixmap.fromImage(qt_img).scaled(self.label.size(), Qt.KeepAspectRatio)
        self.label.setPixmap(pix)

    def set_mode(self, m):
        self.mode = int(m)

    def set_drag_mode(self, enabled: bool):
        self._drag_mode = enabled
        self.setCursor(Qt.SizeHorCursor if enabled else Qt.ArrowCursor)

    def reset_box_offset(self):
        self._current_offset = 0
        self._drag_start_offset = 0
        self.box_offset_changed.emit(0)

    def sync_offset(self, offset: int):
        self._current_offset = offset
        self._drag_start_offset = offset

    def mousePressEvent(self, event):
        if self._drag_mode and event.button() == Qt.LeftButton:
            self._drag_start_x = event.pos().x()
            self._drag_start_offset = self._current_offset

    def mouseMoveEvent(self, event):
        if self._drag_mode and self._drag_start_x is not None:
            dx_display = event.pos().x() - self._drag_start_x
            label_w = max(1, self.label.width())
            dx_image = int(dx_display * self._image_w / label_w)
            self._current_offset = self._drag_start_offset + dx_image
            self.box_offset_changed.emit(self._current_offset)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_start_x = None

    def mouseDoubleClickEvent(self, event):
        if self._drag_mode and event.button() == Qt.LeftButton:
            self.reset_box_offset()


# =====================================================================
# Background worker - capture, track, render, log
# =====================================================================
class VideoThread(QThread):
    change_pixmap = Signal(QImage)        # main (experimenter) display
    change_secondary = Signal(QImage)     # participant display
    status_msg = Signal(str)
    finished_processing = Signal()
    swallow_count_changed = Signal(int)

    def __init__(self):
        super().__init__()
        # capture / scaling config
        self.capture_region = None
        self.scale_fx = 1.0
        self.scale_fy = 1.0
        self.num_trackers = 1
        self.fps_video = 60.0

        # tracker state (parallel lists, indexed by tracker)
        self.trackers = []
        self.rois = []
        self.colors = []
        self.trails = []
        self.csv_rows = []

        # ORB re-init
        self.orb = cv2.ORB_create(1000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.templates_kp = []
        self.templates_des = []
        self.templates_size = []
        self.match_thresh = 8
        self.template_pad_fraction = 0.20
        self.search_area_fraction = 1.0

        # Kalman
        self.kalman_filters = []
        self.use_kf = True

        # control
        self.paused = True
        self.stop_requested = False
        self.manual_reinit_request = None
        self.frame_idx = 0
        self.frames_since_reinit = 0

        # secondary view
        self.secondary_mode = MODE_FRAME_BOX
        self.secondary_manual_override = False
        self.secondary_img_w = 320
        self.secondary_img_h = 240
        self.show_participant_labels = True

        # gradient box
        self.box_x_offset = 0
        self.box_width_fraction = 1.0 / 16.67   # ~6 %
        self.box_alpha_max = 0.80
        self.box_alpha_min = 0.20
        self.box_num_shades = 5

        # tracker-dot color feedback (modes 2 & 3)
        self._circle_entered_box_time = []   # None or float entry timestamp
        self._circle_yellow = []             # bool: draw yellow instead of red
        self.tracker_circle_radius = 6       # tracker-dot radius in px (diameter/2)

        # swallow recording
        self.swallow_active = False
        self.current_swallow_trail = []      # [[(cx,cy),...per tracker], ...per frame]
        self.swallow_trails = []             # completed trails
        self.swallow_count = 0
        self.n_swallow_display = 3
        self.show_swallow_trails = True

        # participant-view zoom
        self.zoom_participant = False
        self.zoom_region = None              # None=auto, else (x,y,w,h) sec coords

        # mode-4 strength meter
        self.last_swallow_excursion = 0.0
        self.strength_metric = "displacement"   # or "arc_length"
        self.strength_scale_max_displacement = 30.0
        self.strength_scale_max_arc_length = 500.0
        self.auto_expand_strength = True

        # mode-5 speedometer
        self.last_swallow_peak_speed = 0.0
        self.speed_scale_max = 2500.0
        self.auto_expand_speed = True
        self.speed_show_max = True    # show running max speed during a swallow (vs. live)

        # cached last frame for paused re-renders
        self._last_sec_frame = None
        self._last_tracker_centers = []
        self._request_secondary_redraw = False

        # output
        self.video_writer = None

    # ------------------------------------------------------------------
    # Configuration setters
    # ------------------------------------------------------------------
    def use_kalman_filtering(self, choice: bool):
        self.use_kf = bool(choice)
        self.status_msg.emit(f"Kalman filtering {'enabled' if self.use_kf else 'disabled'}")
        if self.use_kf and self.rois:
            for i in range(len(self.rois)):
                while i >= len(self.kalman_filters):
                    self.kalman_filters.append(None)
                if self.kalman_filters[i] is None:
                    try:
                        self.kalman_filters[i] = create_kalman_from_roi(self.rois[i])
                    except Exception:
                        self.kalman_filters[i] = None

    def set_scaling_factor(self, fx, fy):
        self.scale_fx = fx
        self.scale_fy = fy

    def set_capture_region(self, rect):
        self.capture_region = (
            int(rect.x()), int(rect.y()), int(rect.width()), int(rect.height()),
        )
        self.status_msg.emit(
            f"Capture region set: {self.capture_region[2]}x{self.capture_region[3]} at "
            f"({self.capture_region[0]}, {self.capture_region[1]})"
        )

    def set_num_trackers(self, n):
        self.num_trackers = n

    # ------------------------------------------------------------------
    # Screen capture
    # ------------------------------------------------------------------
    def grab_capture_frame(self):
        """Grab the configured capture region as a BGR frame (multi-monitor aware)."""
        if self.capture_region is None:
            return None
        x, y, w, h = self.capture_region
        cap_rect = QRect(x, y, w, h)
        # Pick the screen with the largest intersection with the capture region.
        screen = None
        best_area = 0
        for s in QGuiApplication.screens():
            inter = s.geometry().intersected(cap_rect)
            area = inter.width() * inter.height()
            if area > best_area:
                best_area = area
                screen = s
        if screen is None:
            screen = QGuiApplication.primaryScreen()
        if screen is None:
            return None
        sg = screen.geometry()
        full_pixmap = screen.grabWindow(0)
        if full_pixmap.isNull():
            return None
        # Map the region into the chosen screen's local (possibly DPI-scaled) coords.
        sx = full_pixmap.width() / max(1, sg.width())
        sy = full_pixmap.height() / max(1, sg.height())
        local_x = x - sg.x()
        local_y = y - sg.y()
        pixmap = full_pixmap.copy(
            int(local_x * sx), int(local_y * sy),
            max(1, int(w * sx)), max(1, int(h * sy)),
        )
        if pixmap.isNull():
            return None
        image = pixmap.toImage().convertToFormat(QImage.Format_RGB888)
        width = image.width()
        height = image.height()
        bytes_per_line = image.bytesPerLine()
        buffer = np.frombuffer(image.bits(), dtype=np.uint8, count=bytes_per_line * height)
        rgb = buffer.reshape((height, bytes_per_line))[:, :width * 3].reshape((height, width, 3)).copy()
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if self.scale_fx != 1.0 or self.scale_fy != 1.0:
            target_w = max(1, int(w * self.scale_fx))
            target_h = max(1, int(h * self.scale_fy))
            frame = cv2.resize(frame, (target_w, target_h))
        return frame

    # ------------------------------------------------------------------
    # Tracker construction / update (local-search-region based)
    # ------------------------------------------------------------------
    def create_local_tracker(self, frame, roi):
        frame_h, frame_w = frame.shape[:2]
        roi = clamp_roi_to_frame(roi, frame_w, frame_h)
        sx, sy, sw, sh = centered_search_region(roi, frame_w, frame_h, self.search_area_fraction)
        search_frame = frame[sy:sy + sh, sx:sx + sw]
        local_roi = (roi[0] - sx, roi[1] - sy, roi[2], roi[3])
        tr = cv2.legacy.TrackerCSRT_create()
        tr.init(search_frame, local_roi)
        return tr

    def update_tracker_in_local_region(self, tracker, frame, roi):
        frame_h, frame_w = frame.shape[:2]
        roi = clamp_roi_to_frame(roi, frame_w, frame_h)
        sx, sy, sw, sh = centered_search_region(roi, frame_w, frame_h, self.search_area_fraction)
        search_frame = frame[sy:sy + sh, sx:sx + sw]
        ok, local_roi = tracker.update(search_frame)
        if not ok:
            return False, roi
        lx, ly, lw, lh = map(int, local_roi)
        global_roi = clamp_roi_to_frame((sx + lx, sy + ly, lw, lh), frame_w, frame_h)
        return True, global_roi

    def _compute_template(self, frame, roi):
        """Compute ORB keypoints/descriptors and size for a padded ROI template."""
        h, w = frame.shape[:2]
        ex_x, ex_y, ex_w, ex_h = map(int, expand_roi(roi, self.template_pad_fraction, w, h))
        templ = frame[ex_y:ex_y + ex_h, ex_x:ex_x + ex_w]
        templ_gray = cv2.cvtColor(templ, cv2.COLOR_BGR2GRAY) if templ.size else None
        if templ_gray is not None and templ_gray.size > 0:
            kp, des = self.orb.detectAndCompute(templ_gray, None)
        else:
            kp, des = [], None
        return kp, des, (ex_w, ex_h)

    def init_trackers_from_rois(self, frame, rois_list):
        self.trackers = []
        self.rois = []
        self.colors = []
        self.trails = []
        self.csv_rows = []
        self.templates_kp = []
        self.templates_des = []
        self.templates_size = []
        self.kalman_filters = []
        rng = np.random.default_rng(42)
        h_frame, w_frame = frame.shape[:2]
        for roi in rois_list:
            roi = clamp_roi_to_frame(roi, w_frame, h_frame)
            self.rois.append(roi)
            self.trackers.append(self.create_local_tracker(frame, roi))
            self.colors.append(tuple(int(c) for c in rng.integers(50, 255, 3)))
            self.trails.append([])
            kp, des, size = self._compute_template(frame, roi)
            self.templates_kp.append(kp)
            self.templates_des.append(des)
            self.templates_size.append(size)
            kf = None
            if self.use_kf:
                try:
                    kf = create_kalman_from_roi(roi)
                except Exception:
                    kf = None
            self.kalman_filters.append(kf)

    def try_orb_reinit(self, frame_gray, templ_kp, templ_des, templ_size):
        """Attempt to relocate a lost target via ORB matching + homography.
        Returns a new ROI or None."""
        if templ_des is None or len(templ_des) < 4:
            return None
        kp_scene, des_scene = self.orb.detectAndCompute(frame_gray, None)
        if des_scene is None or len(des_scene) < 4:
            return None
        try:
            matches = self.bf.match(templ_des, des_scene)
        except Exception:
            return None
        if not matches:
            return None
        matches = sorted(matches, key=lambda m: m.distance)
        good = matches[: max(10, int(len(matches) * 0.25))]
        if len(good) < self.match_thresh:
            return None
        pts_template = np.float32([templ_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts_scene = np.float32([kp_scene[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        try:
            H, _ = cv2.findHomography(pts_template, pts_scene, cv2.RANSAC, 5.0)
            if H is None:
                return None
        except Exception:
            return None
        tw, th = templ_size[0], templ_size[1]
        corners = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]]).reshape(-1, 1, 2)
        try:
            transformed = cv2.perspectiveTransform(corners, H)
        except Exception:
            return None
        pts = transformed.reshape(-1, 2)
        min_x = max(0, int(np.min(pts[:, 0])))
        min_y = max(0, int(np.min(pts[:, 1])))
        max_x = min(frame_gray.shape[1] - 1, int(np.max(pts[:, 0])))
        max_y = min(frame_gray.shape[0] - 1, int(np.max(pts[:, 1])))
        return (min_x, min_y, max(1, max_x - min_x), max(1, max_y - min_y))

    # ------------------------------------------------------------------
    # Secondary (participant) image rendering
    # ------------------------------------------------------------------
    def _draw_gradient_box(self, sec, half_w, half_h, w, h, tracker_centers):
        """Draw the stepped-gradient green box + tracker dots onto ``sec``.

        Dots are red, turning yellow after 3 continuous seconds at/left of the
        box, reverting to red once they pass to the right of the box.
        """
        xmid = half_w // 2
        box_w = max(1, int(half_w * self.box_width_fraction))
        box_x = max(0, min(half_w - box_w, int(xmid - box_w / 2) + self.box_x_offset))
        x_end = min(half_w, box_x + box_w)
        actual_w = x_end - box_x

        n = self.box_num_shades
        step = (self.box_alpha_max - self.box_alpha_min) / n
        alphas_full = np.empty(box_w, dtype=np.float32)
        for i in range(n):
            alphas_full[int(i * box_w / n):int((i + 1) * box_w / n)] = self.box_alpha_max - i * step
        alphas = alphas_full[:actual_w][np.newaxis, :, np.newaxis]
        green = np.array([0, 255, 0], dtype=np.float32)
        sec[:, box_x:x_end] = (
            sec[:, box_x:x_end].astype(np.float32) * (1 - alphas) + green * alphas
        ).astype(np.uint8)

        # tracker dots with red/yellow feedback
        n_c = len(tracker_centers)
        while len(self._circle_entered_box_time) < n_c:
            self._circle_entered_box_time.append(None)
            self._circle_yellow.append(False)
        now = time.time()
        for idx, (cx, cy) in enumerate(tracker_centers):
            sx = int(cx * half_w / float(w))
            sy = int(cy * half_h / float(h))
            if sx <= x_end:
                if self._circle_entered_box_time[idx] is None:
                    self._circle_entered_box_time[idx] = now
                if now - self._circle_entered_box_time[idx] >= 3.0:
                    self._circle_yellow[idx] = True
            else:
                self._circle_entered_box_time[idx] = None
                self._circle_yellow[idx] = False
            clr = (0, 255, 255) if self._circle_yellow[idx] else (0, 0, 255)
            cv2.circle(sec, (sx, sy), max(1, self.tracker_circle_radius), clr, -1)

    def _trail_excursion(self, trail, metric):
        """Max over trackers of displacement (start->any) or arc length."""
        if len(trail) < 2:
            return 0.0
        n_tr = max(len(fp) for fp in trail)
        best = 0.0
        if metric == "displacement":
            first = trail[0]
            for fr in trail[1:]:
                for ti in range(n_tr):
                    if ti < len(first) and ti < len(fr):
                        ddx = fr[ti][0] - first[ti][0]
                        ddy = fr[ti][1] - first[ti][1]
                        best = max(best, math.hypot(ddx, ddy))
        else:  # arc_length
            for ti in range(n_tr):
                tot = 0.0
                for fi in range(1, len(trail)):
                    pp, cp = trail[fi - 1], trail[fi]
                    if ti < len(pp) and ti < len(cp):
                        tot += math.hypot(cp[ti][0] - pp[ti][0], cp[ti][1] - pp[ti][1])
                best = max(best, tot)
        return best

    def _trail_peak_speed(self, trail):
        """Max frame-to-frame speed (px/s) over all trackers across the trail."""
        if len(trail) < 2:
            return 0.0
        peak = 0.0
        for fi in range(1, len(trail)):
            prev, curr = trail[fi - 1], trail[fi]
            for ti in range(min(len(prev), len(curr))):
                dx = curr[ti][0] - prev[ti][0]
                dy = curr[ti][1] - prev[ti][1]
                peak = max(peak, math.hypot(dx, dy) * self.fps_video)
        return peak

    def _render_strength_meter(self, half_w, half_h):
        sec = np.zeros((half_h, half_w, 3), dtype=np.uint8)
        # current excursion (live during a swallow, else last completed)
        if self.swallow_active and len(self.current_swallow_trail) >= 2:
            cur_exc = self._trail_excursion(self.current_swallow_trail, self.strength_metric)
        elif self.swallow_active:
            cur_exc = 0.0
        else:
            cur_exc = self.last_swallow_excursion
        scale_max = max(1.0, self.strength_scale_max_displacement
                        if self.strength_metric == "displacement"
                        else self.strength_scale_max_arc_length)
        ratio = min(1.0, cur_exc / scale_max)

        # bar layout
        bar_cx = int(half_w * 0.38)
        bar_w_px = max(6, int(half_w * 0.14))
        bar_x = bar_cx - bar_w_px // 2
        bar_top = int(half_h * 0.12)
        bar_bot = int(half_h * 0.84)
        bar_h = max(1, bar_bot - bar_top)

        # gradient background (red top -> green bottom), dimmed above the needle
        ys = np.linspace(0.0, 1.0, bar_h, dtype=np.float32)   # 0=bottom,1=top
        bar_B = np.zeros(bar_h, dtype=np.uint8)
        bar_R = np.where(ys < 0.5, 200, np.clip(200 - (ys - 0.5) * 400, 0, 200)).astype(np.uint8)
        bar_G = np.where(ys < 0.5, np.clip(ys * 400, 0, 200), 200).astype(np.uint8)
        bar_colors = np.stack([bar_B, bar_G, bar_R], axis=1)[::-1]   # row 0 = top
        needle_row = int((1.0 - ratio) * bar_h)
        bar_colors[:needle_row] = bar_colors[:needle_row] // 5
        sec[bar_top:bar_bot, bar_x + 1:bar_x + bar_w_px - 1] = bar_colors[:, np.newaxis, :]

        cv2.rectangle(sec, (bar_x, bar_top), (bar_x + bar_w_px, bar_bot), (160, 160, 160), 1)

        # needle line + side triangles
        needle_y = bar_top + needle_row
        cv2.line(sec, (bar_x - 14, needle_y), (bar_x + bar_w_px + 14, needle_y), (255, 255, 255), 3)
        pts_l = np.array([[bar_x - 2, needle_y], [bar_x - 14, needle_y - 6],
                          [bar_x - 14, needle_y + 6]], np.int32)
        cv2.fillPoly(sec, [pts_l], (255, 255, 255))
        pts_r = np.array([[bar_x + bar_w_px + 2, needle_y], [bar_x + bar_w_px + 14, needle_y - 6],
                          [bar_x + bar_w_px + 14, needle_y + 6]], np.int32)
        cv2.fillPoly(sec, [pts_r], (255, 255, 255))

        tick_x = bar_x + bar_w_px + 2
        lbl_x = tick_x + 10
        font_sc = max(0.25, half_h / 900)
        if self.show_participant_labels:
            for pct in (0, 25, 50, 75, 100):
                ty = int(bar_bot - (pct / 100.0) * bar_h)
                cv2.line(sec, (tick_x, ty), (tick_x + 7, ty), (180, 180, 180), 1)
                cv2.putText(sec, f"{pct / 100.0 * scale_max:.0f}", (lbl_x, ty + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, font_sc, (180, 180, 180), 1)

        # previous-swallow markers (left of bar, using the selected metric)
        for pi, trail in enumerate(self.swallow_trails):
            pexc = self._trail_excursion(trail, self.strength_metric)
            pr = min(1.0, pexc / scale_max)
            py = int(bar_bot - pr * bar_h)
            dim = max(60, 220 - pi * 50)
            cv2.line(sec, (bar_x - 18, py), (bar_x - 3, py), (dim, dim, dim), 2)

        # title (always shown)
        title_x = max(4, bar_x - int(half_w * 0.15))
        lh = max(16, int(font_sc * 1.2 * 28))
        cv2.putText(sec, "SWALLOW", (title_x, bar_top - lh - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_sc * 1.2, (220, 220, 220), 1)
        cv2.putText(sec, "STRENGTH", (title_x, bar_top - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_sc * 1.2, (220, 220, 220), 1)

        if self.show_participant_labels:
            cv2.putText(sec, f"{cur_exc:.1f} px", (bar_x, bar_bot + int(half_h * 0.055)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_sc * 1.1, (255, 255, 255), 1)
            metric_lbl = "Disp." if self.strength_metric == "displacement" else "Arc Len."
            cv2.putText(sec, metric_lbl, (bar_x, bar_bot + int(half_h * 0.10)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_sc * 0.85, (160, 160, 160), 1)
            cv2.putText(sec, f"Swallows: {self.swallow_count}",
                        (int(half_w * 0.55), int(half_h * 0.94)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_sc, (200, 200, 200), 1)
            if self.swallow_active:
                cv2.putText(sec, "LIVE", (int(half_w * 0.72), bar_top + int(half_h * 0.04)),
                            cv2.FONT_HERSHEY_SIMPLEX, font_sc * 1.4, (0, 80, 255), 2)
        return sec

    def _render_speedometer(self, half_w, half_h):
        sec = np.zeros((half_h, half_w, 3), dtype=np.uint8)
        # current speed during a swallow: running max over the swallow (speed_show_max)
        # or rolling max over the last 5 frames (live); else the last completed peak.
        if self.swallow_active and len(self.current_swallow_trail) >= 2:
            if self.speed_show_max:
                cur_spd = self._trail_peak_speed(self.current_swallow_trail)
            else:
                win = min(5, len(self.current_swallow_trail) - 1)
                live_spd = 0.0
                for wi in range(win):
                    fi = len(self.current_swallow_trail) - 1 - wi
                    ppts = self.current_swallow_trail[fi - 1]
                    cpts = self.current_swallow_trail[fi]
                    for ti in range(min(len(ppts), len(cpts))):
                        dx = cpts[ti][0] - ppts[ti][0]
                        dy = cpts[ti][1] - ppts[ti][1]
                        live_spd = max(live_spd, math.hypot(dx, dy) * self.fps_video)
                cur_spd = live_spd
        else:
            cur_spd = self.last_swallow_peak_speed
        spd_scale = max(1.0, self.speed_scale_max)
        ratio = min(1.0, cur_spd / spd_scale)

        cx = half_w // 2
        cy = int(half_h * 0.62)
        r = int(min(half_w, half_h) * 0.40)
        athk = max(3, r // 8)
        fsc = max(0.28, half_h / 900)

        # arc from 135 deg to 405 deg, green->yellow->red, dimmed past the needle
        n_segs = 90
        for s in range(n_segs):
            t0 = s / n_segs
            a0 = math.radians(135.0 + t0 * 270.0)
            a1 = math.radians(135.0 + (s + 1) / n_segs * 270.0)
            x0 = int(cx + r * math.cos(a0)); y0 = int(cy + r * math.sin(a0))
            x1 = int(cx + r * math.cos(a1)); y1 = int(cy + r * math.sin(a1))
            if t0 < 0.5:
                col = (0, int(t0 * 2 * 200), 200)
            else:
                col = (0, 200, int((1.0 - (t0 - 0.5) * 2) * 200))
            if t0 > ratio:
                col = (col[0] // 5, col[1] // 5, col[2] // 5)
            cv2.line(sec, (x0, y0), (x1, y1), col, athk)

        if self.show_participant_labels:
            for pct in (0, 25, 50, 75, 100):
                ta = math.radians(135.0 + pct / 100.0 * 270.0)
                cos_ta, sin_ta = math.cos(ta), math.sin(ta)
                ox = int(cx + r * 1.04 * cos_ta); oy = int(cy + r * 1.04 * sin_ta)
                ix = int(cx + r * 0.82 * cos_ta); iy = int(cy + r * 0.82 * sin_ta)
                cv2.line(sec, (ix, iy), (ox, oy), (200, 200, 200), 2)
                lx = int(cx + r * 1.22 * cos_ta) - 14
                ly = int(cy + r * 1.22 * sin_ta) + 4
                cv2.putText(sec, f"{pct / 100.0 * spd_scale:.0f}", (lx, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, fsc * 0.85, (160, 160, 160), 1)
            for pct10 in range(0, 101, 10):
                if pct10 % 25 == 0:
                    continue
                ta = math.radians(135.0 + pct10 / 100.0 * 270.0)
                ox = int(cx + r * 1.04 * math.cos(ta)); oy = int(cy + r * 1.04 * math.sin(ta))
                ix = int(cx + r * 0.93 * math.cos(ta)); iy = int(cy + r * 0.93 * math.sin(ta))
                cv2.line(sec, (ix, iy), (ox, oy), (110, 110, 110), 1)

        # previous-swallow markers: peak speed of each prior swallow, drawn as
        # radial ticks just outside the arc, dimmed by age (newest brightest)
        n_prev = len(self.swallow_trails)
        for pi, trail in enumerate(self.swallow_trails):
            pspd = self._trail_peak_speed(trail)
            pr = min(1.0, pspd / spd_scale)
            pa = math.radians(135.0 + pr * 270.0)
            cos_pa, sin_pa = math.cos(pa), math.sin(pa)
            dim = max(60, 220 - (n_prev - 1 - pi) * 50)
            ix = int(cx + (r - athk) * cos_pa); iy = int(cy + (r - athk) * sin_pa)
            ox = int(cx + (r + athk) * cos_pa); oy = int(cy + (r + athk) * sin_pa)
            cv2.line(sec, (ix, iy), (ox, oy), (dim, dim, dim), 2)

        # needle + hub
        na = math.radians(135.0 + ratio * 270.0)
        tip_x = int(cx + r * 0.80 * math.cos(na)); tip_y = int(cy + r * 0.80 * math.sin(na))
        bas_x = int(cx - r * 0.14 * math.cos(na)); bas_y = int(cy - r * 0.14 * math.sin(na))
        cv2.line(sec, (bas_x, bas_y), (tip_x, tip_y), (255, 255, 255), 3)
        cv2.circle(sec, (cx, cy), max(5, r // 10), (180, 180, 180), -1)

        # title (always shown)
        ttl = "SWALLOW SPEED"
        tw = cv2.getTextSize(ttl, cv2.FONT_HERSHEY_SIMPLEX, fsc * 1.15, 1)[0][0]
        cv2.putText(sec, ttl, (cx - tw // 2, int(half_h * 0.09)),
                    cv2.FONT_HERSHEY_SIMPLEX, fsc * 1.15, (220, 220, 220), 1)

        if self.show_participant_labels:
            vstr = f"{cur_spd:.1f} px/s"
            vw = cv2.getTextSize(vstr, cv2.FONT_HERSHEY_SIMPLEX, fsc * 1.1, 1)[0][0]
            cv2.putText(sec, vstr, (cx - vw // 2, int(cy + r * 0.36)),
                        cv2.FONT_HERSHEY_SIMPLEX, fsc * 1.1, (255, 255, 255), 1)
            if not self.swallow_active:
                lbl2 = "(peak)"
            else:
                lbl2 = "(max)" if self.speed_show_max else "(live)"
            lw = cv2.getTextSize(lbl2, cv2.FONT_HERSHEY_SIMPLEX, fsc * 0.8, 1)[0][0]
            cv2.putText(sec, lbl2, (cx - lw // 2, int(cy + r * 0.52)),
                        cv2.FONT_HERSHEY_SIMPLEX, fsc * 0.8, (160, 160, 160), 1)
            cv2.putText(sec, f"Swallows: {self.swallow_count}",
                        (int(half_w * 0.05), int(half_h * 0.95)),
                        cv2.FONT_HERSHEY_SIMPLEX, fsc, (200, 200, 200), 1)
            if self.swallow_active:
                cv2.putText(sec, "LIVE", (int(half_w * 0.76), int(half_h * 0.10)),
                            cv2.FONT_HERSHEY_SIMPLEX, fsc * 1.4, (0, 80, 255), 2)
        return sec

    def _draw_swallow_trails(self, sec, half_w, half_h, w, h):
        palette = [
            (0, 140, 255), (255, 0, 255), (0, 255, 255), (255, 215, 0),
            (255, 100, 0), (100, 255, 50), (180, 0, 180), (0, 200, 150),
            (255, 50, 50), (50, 150, 255),
        ]
        for si, trail in enumerate(self.swallow_trails[-self.n_swallow_display:]):
            col = palette[si % len(palette)]
            for fi in range(1, len(trail)):
                prev, curr = trail[fi - 1], trail[fi]
                for ti in range(min(len(prev), len(curr))):
                    p1 = (int(prev[ti][0] * half_w / float(w)), int(prev[ti][1] * half_h / float(h)))
                    p2 = (int(curr[ti][0] * half_w / float(w)), int(curr[ti][1] * half_h / float(h)))
                    cv2.line(sec, p1, p2, col, 2)
        if self.swallow_active and len(self.current_swallow_trail) > 1:
            for fi in range(1, len(self.current_swallow_trail)):
                prev, curr = self.current_swallow_trail[fi - 1], self.current_swallow_trail[fi]
                for ti in range(min(len(prev), len(curr))):
                    p1 = (int(prev[ti][0] * half_w / float(w)), int(prev[ti][1] * half_h / float(h)))
                    p2 = (int(curr[ti][0] * half_w / float(w)), int(curr[ti][1] * half_h / float(h)))
                    cv2.line(sec, p1, p2, (255, 255, 255), 2)

    def _apply_zoom(self, sec, half_w, half_h):
        if self.zoom_region is not None:
            zx, zy, zw, zh = self.zoom_region
            zx = min(zx, half_w - 1)
            zy = min(zy, half_h - 1)
            zw = min(zw, half_w - zx)
            zh = min(zh, half_h - zy)
            if zw > 0 and zh > 0:
                return cv2.resize(sec[zy:zy + zh, zx:zx + zw], (half_w, half_h))
            return sec
        xmid = half_w // 2
        bw = max(1, int(half_w * self.box_width_fraction))
        bx = max(0, min(half_w - bw, int(xmid - bw / 2) + self.box_x_offset))
        zoom_x = max(0, bx - bw)
        if zoom_x < half_w - 1:
            return cv2.resize(sec[:, zoom_x:], (half_w, half_h))
        return sec

    def emit_secondary_image(self, frame, tracker_centers):
        """Build and emit the participant image for the current secondary mode."""
        h, w = frame.shape[:2]
        half_w = max(1, self.secondary_img_w)
        half_h = max(1, self.secondary_img_h)
        mode = int(self.secondary_mode)

        if mode == MODE_COPY:
            sec = cv2.resize(frame, (half_w, half_h))
        elif mode == MODE_BLACK_BOX:
            sec = np.zeros((half_h, half_w, 3), dtype=np.uint8)
            self._draw_gradient_box(sec, half_w, half_h, w, h, tracker_centers)
        elif mode == MODE_FRAME_BOX:
            sec = cv2.resize(frame, (half_w, half_h))
            self._draw_gradient_box(sec, half_w, half_h, w, h, tracker_centers)
        elif mode == MODE_STRENGTH:
            sec = self._render_strength_meter(half_w, half_h)
        elif mode == MODE_SPEED:
            sec = self._render_speedometer(half_w, half_h)
        else:
            sec = cv2.resize(frame, (half_w, half_h))

        # swallow trajectories + zoom apply only to the image-based modes (1-3)
        if mode in (MODE_COPY, MODE_BLACK_BOX, MODE_FRAME_BOX):
            if self.show_swallow_trails:
                self._draw_swallow_trails(sec, half_w, half_h, w, h)
            if self.zoom_participant:
                sec = self._apply_zoom(sec, half_w, half_h)

        self.change_secondary.emit(bgr_to_qimage(sec))

    # ------------------------------------------------------------------
    # Main processing loop
    # ------------------------------------------------------------------
    def run(self):
        if self.capture_region is None:
            self.status_msg.emit("Select a capture region first")
            return
        self.stop_requested = False
        self.paused = False
        self.frame_idx = 0
        self.frames_since_reinit = 0
        frame_w = max(1, int(self.capture_region[2] * self.scale_fx))
        frame_h = max(1, int(self.capture_region[3] * self.scale_fy))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, self.fps_video, (frame_w, frame_h))

        while not self.stop_requested:
            if self.paused:
                if self._request_secondary_redraw and self._last_sec_frame is not None:
                    self._request_secondary_redraw = False
                    try:
                        self.emit_secondary_image(self._last_sec_frame, self._last_tracker_centers)
                    except Exception:
                        pass
                time.sleep(0.03)
                continue

            t0 = time.time()
            frame = self.grab_capture_frame()
            if frame is None:
                time.sleep(0.03)
                continue
            vis = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            reinit_attempted = [False] * self.num_trackers
            reinit_failed = [False] * self.num_trackers
            tracker_centers = []

            for i in range(self.num_trackers):
                if i >= len(self.trackers):
                    continue
                ok, new_roi = self.update_tracker_in_local_region(self.trackers[i], frame, self.rois[i])
                reinit_success = False
                measured = None

                if ok:
                    self.rois[i] = new_roi
                    x, y, w, h = map(int, new_roi)
                    try:
                        self.trackers[i] = self.create_local_tracker(frame, (x, y, w, h))
                    except Exception:
                        pass
                    cx = x + w / 2.0
                    cy = y + h / 2.0
                    measured = np.array([cx, cy, w, h], dtype=np.float32)
                    kp, des, size = self._compute_template(frame, (x, y, w, h))
                    self.templates_kp[i] = kp
                    self.templates_des[i] = des
                    self.templates_size[i] = size
                    if self.use_kf:
                        if self.kalman_filters[i] is None:
                            self.kalman_filters[i] = create_kalman_from_roi((x, y, w, h))
                        else:
                            try:
                                self.kalman_filters[i].correct(measured)
                            except Exception:
                                self.kalman_filters[i] = create_kalman_from_roi((x, y, w, h))
                else:
                    reinit_attempted[i] = True
                    des_t = self.templates_des[i]
                    if des_t is not None and len(des_t) >= 4:
                        cand = self.try_orb_reinit(gray, self.templates_kp[i], des_t, self.templates_size[i])
                        if cand is not None:
                            try:
                                self.trackers[i] = self.create_local_tracker(frame, cand)
                                self.rois[i] = cand
                                kp, des, size = self._compute_template(frame, cand)
                                self.templates_kp[i] = kp
                                self.templates_des[i] = des
                                self.templates_size[i] = size
                                if self.use_kf:
                                    cx = cand[0] + cand[2] / 2.0
                                    cy = cand[1] + cand[3] / 2.0
                                    meas = np.array([cx, cy, cand[2], cand[3]], dtype=np.float32)
                                    if self.kalman_filters[i] is None:
                                        self.kalman_filters[i] = create_kalman_from_roi(cand)
                                    else:
                                        try:
                                            self.kalman_filters[i].correct(meas)
                                        except Exception:
                                            self.kalman_filters[i] = create_kalman_from_roi(cand)
                                reinit_success = True
                            except Exception:
                                reinit_failed[i] = True
                        else:
                            reinit_failed[i] = True
                    else:
                        reinit_failed[i] = True

                # visualization box (Kalman prediction if available)
                if self.use_kf and i < len(self.kalman_filters) and self.kalman_filters[i] is not None:
                    try:
                        pred = self.kalman_filters[i].predict()
                        pred_cx, pred_cy = float(pred[0]), float(pred[1])
                        pred_w, pred_h = float(pred[4]), float(pred[5])
                        vis_w = int(max(1, pred_w)); vis_h = int(max(1, pred_h))
                        vis_x = int(pred_cx - vis_w / 2.0); vis_y = int(pred_cy - vis_h / 2.0)
                    except Exception:
                        try:
                            vis_x, vis_y, vis_w, vis_h = map(int, self.rois[i])
                        except Exception:
                            continue
                else:
                    try:
                        vis_x, vis_y, vis_w, vis_h = map(int, self.rois[i])
                    except Exception:
                        continue

                if ok or reinit_success:
                    # use raw ROI right after (re)init while the KF settles
                    if self.frames_since_reinit <= 15:
                        try:
                            vis_x, vis_y, vis_w, vis_h = map(int, self.rois[i])
                        except Exception:
                            pass
                    cv2.rectangle(vis, (vis_x, vis_y), (vis_x + vis_w, vis_y + vis_h), self.colors[i], 2)
                    cv2.putText(vis, f"T{i+1}", (vis_x, max(12, vis_y - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors[i], 2)
                    center = (vis_x + vis_w // 2, vis_y + vis_h // 2)
                    tracker_centers.append(center)
                    self.trails[i].append(center)
                    if len(self.trails[i]) > 40:
                        self.trails[i].pop(0)
                    for t in range(1, len(self.trails[i])):
                        cv2.line(vis, self.trails[i][t - 1], self.trails[i][t], self.colors[i], 2)
                else:
                    cv2.rectangle(vis, (vis_x, vis_y), (vis_x + vis_w, vis_y + vis_h), (0, 255, 255), 1)
                    cv2.putText(vis, f"T{i+1} LOST", (vis_x, max(12, vis_y - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                    if reinit_failed[i]:
                        cv2.putText(vis, f"T{i+1} RE-INIT FAILED", (20, 60 + i * 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # CSV logging
                if measured is not None:
                    meas_cx, meas_cy = float(measured[0]), float(measured[1])
                    meas_w, meas_h = float(measured[2]), float(measured[3])
                    meas_x = meas_cx - meas_w / 2.0
                    meas_y = meas_cy - meas_h / 2.0
                else:
                    meas_x = meas_y = meas_w = meas_h = np.nan

                if self.use_kf and i < len(self.kalman_filters) and self.kalman_filters[i] is not None:
                    state = self.kalman_filters[i].statePost.flatten()
                    smooth_cx, smooth_cy = float(state[0]), float(state[1])
                    smooth_w, smooth_h = float(state[4]), float(state[5])
                    smooth_x = smooth_cx - smooth_w / 2.0
                    smooth_y = smooth_cy - smooth_h / 2.0
                else:
                    smooth_x, smooth_y = float(vis_x), float(vis_y)
                    smooth_w, smooth_h = float(vis_w), float(vis_h)

                self.csv_rows.append({
                    "frame": self.frame_idx,
                    "tracker_id": i + 1,
                    "meas_x": meas_x, "meas_y": meas_y, "meas_w": meas_w, "meas_h": meas_h,
                    "smooth_x": smooth_x, "smooth_y": smooth_y,
                    "smooth_w": smooth_w, "smooth_h": smooth_h,
                    "ok": bool(ok),
                    "reinit_attempted": bool(reinit_attempted[i]),
                    "reinit_success": bool(reinit_success),
                    "reinit_failed": bool(reinit_failed[i]),
                })

            # record swallow trail snapshot for this frame
            if self.swallow_active and tracker_centers:
                self.current_swallow_trail.append(list(tracker_centers))

            self.frame_idx += 1
            self.frames_since_reinit += 1

            elapsed = time.time() - t0
            fps = int(1 / elapsed) if elapsed > 0 else 0
            if self.manual_reinit_request is None:
                cv2.putText(vis, f"FPS: {fps}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                if self.video_writer:
                    self.video_writer.write(vis)
            else:
                vis = frame.copy()
                cv2.putText(vis, "Manual reinit active", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # emit main image
            self.change_pixmap.emit(bgr_to_qimage(vis))

            # cache for paused re-renders + emit secondary
            self._last_sec_frame = frame
            self._last_tracker_centers = list(tracker_centers)
            self._request_secondary_redraw = False
            try:
                self.emit_secondary_image(frame, tracker_centers)
            except Exception:
                pass

            # handle a queued manual reinit
            if self.manual_reinit_request is not None:
                idx, roi = self.manual_reinit_request
                self.manual_reinit_request = None
                self.paused = True
                self.status_msg.emit(f"Manual reinit for tracker {idx+1}")
                if roi is not None:
                    try:
                        self.trackers[idx] = self.create_local_tracker(frame, roi)
                        self.rois[idx] = roi
                        kp, des, size = self._compute_template(frame, roi)
                        self.templates_kp[idx] = kp
                        self.templates_des[idx] = des
                        self.templates_size[idx] = size
                        self.kalman_filters[idx] = create_kalman_from_roi(roi) if self.use_kf else None
                        self.status_msg.emit(f"Manual reinit tracker {idx+1} done")
                    except Exception as e:
                        self.status_msg.emit(f"Manual reinit failed: {e}")
                self.paused = False
                self.frames_since_reinit = 0

            time.sleep(0.002)

        if self.video_writer:
            self.video_writer.release()
        self.status_msg.emit("Processing finished")
        self.finished_processing.emit()

    # ------------------------------------------------------------------
    # Slots / control API
    # ------------------------------------------------------------------
    @Slot()
    def pause_toggle(self):
        self.paused = not self.paused
        self.status_msg.emit("Paused" if self.paused else "Resumed")

    @Slot()
    def stop(self):
        self.stop_requested = True
        self.paused = True

    @Slot(int)
    def request_manual_reinit(self, idx):
        if 0 <= idx < self.num_trackers:
            self.manual_reinit_request = (idx, None)
        else:
            self.status_msg.emit("Invalid tracker index for reinit")

    @Slot(tuple)
    def request_manual_reinit_with_roi(self, args):
        idx, roi = args[0], args[1]
        if 0 <= idx < self.num_trackers:
            self.manual_reinit_request = (idx, roi)
        else:
            self.status_msg.emit("Invalid tracker index for reinit")

    @Slot(str)
    def save_csv(self, outpath):
        try:
            pd.DataFrame(self.csv_rows).to_csv(outpath, index=False)
            self.status_msg.emit(f"CSV saved to {outpath}")
        except Exception as e:
            self.status_msg.emit(f"CSV save failed: {e}")

    # --- gradient-box parameters ---
    @Slot(int)
    def set_box_offset(self, offset):
        self.box_x_offset = offset

    @Slot(float)
    def set_box_width_fraction(self, fraction: float):
        self.box_width_fraction = max(0.01, min(0.99, fraction))

    @Slot(float)
    def set_box_alpha_max(self, value: float):
        self.box_alpha_max = max(0.0, min(1.0, value))

    @Slot(float)
    def set_box_alpha_min(self, value: float):
        self.box_alpha_min = max(0.0, min(1.0, value))

    @Slot(int)
    def set_box_num_shades(self, n: int):
        self.box_num_shades = max(5, min(50, n))

    @Slot(int)
    def set_tracker_circle_diameter(self, diameter: int):
        self.tracker_circle_radius = max(1, int(diameter) // 2)
        self._request_secondary_redraw = True

    @Slot(int, int)
    def set_secondary_size(self, w: int, h: int):
        self.secondary_img_w = max(1, w)
        self.secondary_img_h = max(1, h)

    # --- swallow recording ---
    @Slot()
    def start_swallow(self):
        self.current_swallow_trail = []
        self.swallow_active = True

    @Slot()
    def end_swallow(self):
        if not self.swallow_active:
            return
        self.swallow_active = False
        if self.current_swallow_trail:
            self.swallow_trails.append(self.current_swallow_trail)
            self.swallow_trails = self.swallow_trails[-self.n_swallow_display:]
            self.swallow_count += 1
            self.swallow_count_changed.emit(self.swallow_count)
            # strength meter: excursion for the chosen metric
            exc = self._trail_excursion(self.current_swallow_trail, self.strength_metric)
            self.last_swallow_excursion = exc
            if self.auto_expand_strength:
                if self.strength_metric == "displacement":
                    if exc > self.strength_scale_max_displacement:
                        self.strength_scale_max_displacement = exc * 1.2
                else:
                    if exc > self.strength_scale_max_arc_length:
                        self.strength_scale_max_arc_length = exc * 1.2
            # speedometer: peak frame-to-frame speed
            if len(self.current_swallow_trail) >= 2:
                peak = self._trail_peak_speed(self.current_swallow_trail)
                self.last_swallow_peak_speed = peak
                if self.auto_expand_speed and peak > self.speed_scale_max:
                    self.speed_scale_max = peak * 1.2
        self.current_swallow_trail = []

    @Slot(int)
    def set_n_swallow_display(self, n: int):
        self.n_swallow_display = max(1, n)
        self.swallow_trails = self.swallow_trails[-self.n_swallow_display:]

    @Slot(str)
    def set_strength_metric(self, metric: str):
        self.strength_metric = metric

    @Slot(bool)
    def set_auto_expand_strength(self, enabled: bool):
        self.auto_expand_strength = bool(enabled)

    @Slot(bool)
    def set_auto_expand_speed(self, enabled: bool):
        self.auto_expand_speed = bool(enabled)

    @Slot(bool)
    def set_speed_show_max(self, enabled: bool):
        self.speed_show_max = bool(enabled)

    @Slot(bool)
    def set_show_participant_labels(self, enabled: bool):
        self.show_participant_labels = bool(enabled)

    @Slot(int)
    def set_strength_scale_displacement(self, value: int):
        self.strength_scale_max_displacement = float(max(1, value))

    @Slot(int)
    def set_strength_scale_arc_length(self, value: int):
        self.strength_scale_max_arc_length = float(max(1, value))

    @Slot(int)
    def set_speed_scale_max(self, value: int):
        self.speed_scale_max = float(max(1, value))

    @Slot(bool)
    def set_show_swallow_trails(self, enabled: bool):
        self.show_swallow_trails = bool(enabled)
        self._request_secondary_redraw = True

    @Slot()
    def clear_swallow_trails(self):
        self.swallow_trails = []
        self.current_swallow_trail = []
        self._request_secondary_redraw = True

    # --- zoom ---
    @Slot(bool)
    def set_zoom_participant(self, enabled: bool):
        self.zoom_participant = bool(enabled)

    @Slot(int, int, int, int)
    def set_zoom_region(self, x: int, y: int, w: int, h: int):
        self.zoom_region = (max(0, x), max(0, y), max(1, w), max(1, h))

    @Slot()
    def clear_zoom_region(self):
        self.zoom_region = None


# =====================================================================
# Experimenter View - control panel + main display
# =====================================================================
class MainWindow(QWidget):
    def __init__(self, num_of_tracker=2, use_kf=True, moveExperimenterViewToSecondScreen=True):
        super().__init__()
        self.setWindowTitle("Experimenter View")
        self.resize(900, 585)
        screens = QGuiApplication.screens()
        if moveExperimenterViewToSecondScreen and len(screens) >= 2:
            self.move(screens[1].geometry().topLeft())
        else:
            self.move(10, 10)

        # --- main image label ---
        self.image_label = DrawableLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.display_w = 600
        self.display_h = 530
        self.image_label.setFixedSize(self.display_w, self.display_h)

        self._build_controls(num_of_tracker, use_kf)
        self._build_layout()
        self._build_worker_and_secondary()
        self._wire_signals()

        # internal state
        self.last_frame = None
        self.selecting_roi = False
        self.secondary_manual_override = False
        self.measuring = False
        self._measure_img_size = None   # (w, h) of the frozen source frame

        # start in copy mode until tracking begins
        self.worker.secondary_mode = MODE_COPY
        self.slider.setValue(MODE_COPY)
        self.slider_label.setText(f"Mode {MODE_COPY}: {MODE_NAMES[MODE_COPY]}")

    # ------------------------------------------------------------------
    # Widget construction
    # ------------------------------------------------------------------
    def _build_controls(self, num_of_tracker, use_kf):
        self.spin_num = QSpinBox()
        self.spin_num.setMinimum(1)
        self.spin_num.setMaximum(20)
        self.spin_num.setValue(num_of_tracker)
        self.chk_kf = QCheckBox("Use Kalman Filter")
        self.chk_kf.setChecked(bool(use_kf))

        self.btn_load = QPushButton("Screen Mirror Region (Ctrl+M)")
        self.btn_select_rois = QPushButton("Select ROIs (Ctrl+I)")
        self.btn_start = QPushButton("Start Tracking (Ctrl+T)")
        self.btn_pause = QPushButton("Pause/Resume (Ctrl+P)")
        self.combo_reinit = QComboBox()
        self.combo_reinit.addItems([f"Tracker {i+1}" for i in range(num_of_tracker)])
        self.btn_reinit = QPushButton("Reinit Selected (draw) (Ctrl+R)")
        self.btn_export = QPushButton("Export CSV")
        self.btn_exit = QPushButton("Exit")

        # mode slider (1..5)
        self.slider_label = QLabel("Mode 1: Copy")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(MODE_COPY)
        self.slider.setMaximum(MODE_SPEED)
        self.slider.setValue(MODE_COPY)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(1)

        self.chk_move_box = QCheckBox("Move Box (drag in participant view)")
        self.chk_move_box.setToolTip(
            "Check to drag the gradient box horizontally in the Participant View.\n"
            "Double-click to reset its position."
        )

        self._box_width_pct = 8
        self.btn_box_width = QPushButton(f"Adjust Shaded Box Width: {self._box_width_pct}%")
        self.btn_box_shading = QPushButton("Box Shading...")

        # tracker-dot diameter (participant view, modes 2 & 3)
        self.slider_circle_diam = QSlider(Qt.Horizontal)
        self.slider_circle_diam.setRange(2, 60)
        self.slider_circle_diam.setValue(12)
        self.slider_circle_diam.setTickPosition(QSlider.TicksBelow)
        self.slider_circle_diam.setTickInterval(5)
        self.slider_circle_diam.setToolTip(
            "Diameter of the tracker dots shown on the participant view (modes 2 & 3)."
        )
        self.lbl_circle_diam = QLabel("12 px")

        self.chk_show_box_main = QCheckBox("Show box on main view")
        self.chk_show_box_main.setChecked(True)
        self.chk_show_participant_labels = QCheckBox("Show participant labels")
        self.chk_show_participant_labels.setChecked(True)
        self.chk_show_participant_labels.setToolTip(
            "Show/hide scale marks, live/peak values, swallow count,\n"
            "metric label, and LIVE badge on the participant screen\n"
            "(mode 4 and 5). Title text is always visible."
        )

        # swallow controls
        self.btn_swallow = QPushButton("Mark Swallow Start (Ctrl+S)")
        self.btn_swallow.setCheckable(True)
        self.lbl_swallow_count = QLabel("Swallows: 0")
        self.lbl_swallow_count.setStyleSheet("font-weight: bold;")
        self.spin_swallow_n = QSpinBox()
        self.spin_swallow_n.setMinimum(1)
        self.spin_swallow_n.setMaximum(20)
        self.spin_swallow_n.setValue(3)
        self.chk_show_trails = QCheckBox("Show swallow trajectories")
        self.chk_show_trails.setChecked(True)
        self.btn_clear_trails = QPushButton("Clear Trajectories (Ctrl+C)")
        self.combo_strength_metric = QComboBox()
        self.combo_strength_metric.addItem("Displacement", "displacement")
        self.combo_strength_metric.addItem("Arc Length", "arc_length")

        # point-to-point measurement
        self.btn_measure = QPushButton("Measure Distance (Ctrl+D)")
        self.btn_measure.setCheckable(True)
        self.btn_measure.setToolTip(
            "Freeze the experimenter view and click two points to measure\n"
            "the distance between them. Click again for a new measurement;\n"
            "uncheck to resume the live feed."
        )
        self.measure_result = QLineEdit()
        self.measure_result.setReadOnly(True)
        self.measure_result.setPlaceholderText("Distance: -- px")

        self.chk_zoom_participant = QCheckBox("Zoom participant view")
        self.chk_zoom_participant.setToolTip(
            "Zooms the participant view to the region from one box-width\n"
            "left of the gradient box to the right edge of the image."
        )
        self.btn_set_zoom_region = QPushButton("Set Zoom Region")
        self.btn_set_zoom_region.setToolTip(
            "Draw a rectangle on the experimenter view to define the zoom region."
        )
        self.btn_reset_zoom_region = QPushButton("Reset to Auto")
        self.btn_reset_zoom_region.setToolTip("Revert to the automatic zoom region.")
        self.lbl_zoom_mode = QLabel("Auto")
        self.lbl_zoom_mode.setStyleSheet("color: gray; font-style: italic;")

        # scale settings (modes 4 & 5)
        self.chk_auto_expand_strength = QCheckBox("Auto-expand strength scale")
        self.chk_auto_expand_strength.setChecked(True)
        self.slider_disp_scale = QSlider(Qt.Horizontal)
        self.slider_disp_scale.setRange(1, 500)
        self.slider_disp_scale.setValue(30)
        self.lbl_disp_scale_val = QLabel("30 px")
        self.slider_arc_scale = QSlider(Qt.Horizontal)
        self.slider_arc_scale.setRange(1, 5000)
        self.slider_arc_scale.setValue(500)
        self.lbl_arc_scale_val = QLabel("500 px")
        self.chk_speed_show_max = QCheckBox("Show max speed during swallow")
        self.chk_speed_show_max.setChecked(True)
        self.chk_auto_expand_speed = QCheckBox("Auto-expand speed scale")
        self.chk_auto_expand_speed.setChecked(True)
        self.slider_speed_scale = QSlider(Qt.Horizontal)
        self.slider_speed_scale.setRange(100, 10000)
        self.slider_speed_scale.setValue(2500)
        self.lbl_speed_scale_val = QLabel("2500 px/s")

        self.status_label = QLabel("")

    @staticmethod
    def _labeled_row(label, *widgets):
        row = QHBoxLayout()
        if label is not None:
            row.addWidget(QLabel(label))
        for wdg in widgets:
            row.addWidget(wdg)
        return row

    def _build_layout(self):
        vbox = QVBoxLayout()
        vbox.addWidget(QLabel("Number of trackers:"))
        vbox.addWidget(self.spin_num)
        vbox.addWidget(self.chk_kf)
        vbox.addSpacing(6)
        vbox.addWidget(self.btn_load)
        vbox.addWidget(self.btn_select_rois)
        vbox.addWidget(self.btn_start)
        vbox.addWidget(QLabel("Manual Reinit:"))
        vbox.addWidget(self.combo_reinit)
        vbox.addWidget(self.btn_reinit)
        vbox.addSpacing(6)
        vbox.addWidget(self.slider_label)
        vbox.addWidget(self.slider)
        vbox.addWidget(self.chk_move_box)
        vbox.addWidget(self.btn_box_width)
        vbox.addWidget(self.btn_box_shading)
        vbox.addLayout(self._labeled_row("Tracker dot diameter:",
                                         self.slider_circle_diam, self.lbl_circle_diam))
        vbox.addWidget(self.chk_show_box_main)
        vbox.addWidget(self.chk_show_participant_labels)
        vbox.addSpacing(6)
        vbox.addWidget(QLabel("Swallow Marking:"))
        vbox.addWidget(self.btn_swallow)
        vbox.addWidget(self.lbl_swallow_count)
        vbox.addLayout(self._labeled_row("Show last N:", self.spin_swallow_n))
        vbox.addWidget(self.chk_show_trails)
        vbox.addWidget(self.btn_clear_trails)
        vbox.addLayout(self._labeled_row("Strength metric:", self.combo_strength_metric))
        vbox.addSpacing(6)
        vbox.addWidget(QLabel("Measurement:"))
        vbox.addWidget(self.btn_measure)
        vbox.addWidget(self.measure_result)
        vbox.addSpacing(6)
        vbox.addWidget(self.chk_zoom_participant)
        vbox.addLayout(self._labeled_row(None, self.btn_set_zoom_region,
                                         self.btn_reset_zoom_region, self.lbl_zoom_mode))
        vbox.addSpacing(6)
        vbox.addWidget(QLabel("Scale Settings (Mode 4 & 5):"))
        vbox.addWidget(self.chk_auto_expand_strength)
        vbox.addLayout(self._labeled_row("Disp. max:", self.slider_disp_scale, self.lbl_disp_scale_val))
        vbox.addLayout(self._labeled_row("Arc max:", self.slider_arc_scale, self.lbl_arc_scale_val))
        vbox.addWidget(self.chk_speed_show_max)
        vbox.addWidget(self.chk_auto_expand_speed)
        vbox.addLayout(self._labeled_row("Speed max:", self.slider_speed_scale, self.lbl_speed_scale_val))
        vbox.addSpacing(6)
        vbox.addWidget(self.btn_pause)
        vbox.addWidget(self.btn_export)
        vbox.addWidget(self.btn_exit)
        vbox.addStretch(1)
        vbox.addWidget(self.status_label)

        ctrl_widget = QWidget()
        ctrl_widget.setLayout(vbox)
        ctrl_scroll = QScrollArea()
        ctrl_scroll.setWidget(ctrl_widget)
        ctrl_scroll.setWidgetResizable(True)
        ctrl_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        hbox = QHBoxLayout()
        hbox.addWidget(self.image_label)
        hbox.addWidget(ctrl_scroll)
        self.setLayout(hbox)

    def _build_worker_and_secondary(self):
        self.worker = VideoThread()
        self.worker.use_kalman_filtering(self.chk_kf.isChecked())
        self.worker.change_pixmap.connect(self.on_frame)
        self.worker.change_secondary.connect(self.on_secondary_image)
        self.worker.status_msg.connect(self.show_status)
        self.worker.finished_processing.connect(self.on_finished)

        self.secondary = SecondaryWindow(width=self.display_w // 2, height=self.display_h // 2)
        self.secondary.box_offset_changed.connect(self.worker.set_box_offset)
        self.secondary.size_changed.connect(self.worker.set_secondary_size)
        self.secondary.size_changed.connect(lambda w, _h: self.image_label.set_sec_w(w))
        self.secondary.showMaximized()
        # seed sizes
        self.worker.set_secondary_size(self.secondary.width(), self.secondary.height())
        self.image_label.set_sec_w(self.secondary.width())
        # keep box offset synced both directions
        self.image_label.box_offset_changed.connect(self.worker.set_box_offset)
        self.image_label.box_offset_changed.connect(self.secondary.sync_offset)
        self.secondary.box_offset_changed.connect(self.image_label.sync_offset)

        # live preview of the capture region before tracking starts
        self.preview_timer = QTimer(self)
        self.preview_timer.setInterval(40)   # ~25 fps
        self.preview_timer.timeout.connect(self._update_preview)

    def _wire_signals(self):
        # hotkeys
        QShortcut(QKeySequence("Ctrl+M"), self).activated.connect(self.btn_load.click)
        QShortcut(QKeySequence("Ctrl+I"), self).activated.connect(self.btn_select_rois.click)
        QShortcut(QKeySequence("Ctrl+R"), self).activated.connect(self.btn_reinit.click)
        QShortcut(QKeySequence("Ctrl+T"), self).activated.connect(self.btn_start.click)
        QShortcut(QKeySequence("Ctrl+S"), self).activated.connect(self.btn_swallow.toggle)
        QShortcut(QKeySequence("Ctrl+P"), self).activated.connect(self.btn_pause.click)
        QShortcut(QKeySequence("Ctrl+C"), self).activated.connect(self.btn_clear_trails.click)
        QShortcut(QKeySequence("Ctrl+D"), self).activated.connect(self.btn_measure.toggle)

        # buttons / inputs
        self.btn_load.clicked.connect(self.on_load)
        self.btn_select_rois.clicked.connect(self.on_select_rois_gui)
        self.btn_start.clicked.connect(self.on_start_tracking)
        self.btn_pause.clicked.connect(self.on_pause)
        self.btn_reinit.clicked.connect(self.on_manual_reinit_gui)
        self.btn_export.clicked.connect(self.on_export)
        self.btn_exit.clicked.connect(self.close)
        self.spin_num.valueChanged.connect(self.on_num_changed)
        self.chk_kf.stateChanged.connect(self.on_kf_toggled)
        self.slider.valueChanged.connect(self.on_slider_changed)
        self.chk_move_box.toggled.connect(self.on_move_box_toggled)
        self.btn_box_width.clicked.connect(self.on_box_width_clicked)
        self.btn_box_shading.clicked.connect(self.on_box_shading_clicked)
        self.slider_circle_diam.valueChanged.connect(self.worker.set_tracker_circle_diameter)
        self.slider_circle_diam.valueChanged.connect(
            lambda v: self.lbl_circle_diam.setText(f"{v} px"))
        self.btn_swallow.toggled.connect(self.on_swallow_toggled)
        self.btn_measure.toggled.connect(self.on_measure_toggled)
        self.image_label.measure_points_ready.connect(self.on_measure_points)
        self.spin_swallow_n.valueChanged.connect(self.worker.set_n_swallow_display)
        self.chk_show_trails.toggled.connect(self.worker.set_show_swallow_trails)
        self.chk_show_participant_labels.toggled.connect(self.worker.set_show_participant_labels)
        self.btn_clear_trails.clicked.connect(self.worker.clear_swallow_trails)
        self.combo_strength_metric.currentIndexChanged.connect(
            lambda: self.worker.set_strength_metric(self.combo_strength_metric.currentData()))
        self.chk_auto_expand_strength.toggled.connect(self.worker.set_auto_expand_strength)
        self.slider_disp_scale.valueChanged.connect(self.worker.set_strength_scale_displacement)
        self.slider_disp_scale.valueChanged.connect(lambda v: self.lbl_disp_scale_val.setText(f"{v} px"))
        self.slider_arc_scale.valueChanged.connect(self.worker.set_strength_scale_arc_length)
        self.slider_arc_scale.valueChanged.connect(lambda v: self.lbl_arc_scale_val.setText(f"{v} px"))
        self.chk_speed_show_max.toggled.connect(self.worker.set_speed_show_max)
        self.chk_auto_expand_speed.toggled.connect(self.worker.set_auto_expand_speed)
        self.slider_speed_scale.valueChanged.connect(self.worker.set_speed_scale_max)
        self.slider_speed_scale.valueChanged.connect(lambda v: self.lbl_speed_scale_val.setText(f"{v} px/s"))
        self.chk_zoom_participant.toggled.connect(self.worker.set_zoom_participant)
        self.btn_set_zoom_region.clicked.connect(self.on_set_zoom_region)
        self.btn_reset_zoom_region.clicked.connect(self.on_reset_zoom_region)
        self.worker.swallow_count_changed.connect(
            lambda n: self.lbl_swallow_count.setText(f"Swallows: {n}"))

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------
    def _display_frame(self, frame):
        self._measure_img_size = (frame.shape[1], frame.shape[0])
        if self.measuring:      # experimenter view is frozen for measurement
            return
        self.image_label.setPixmap(QPixmap.fromImage(bgr_to_qimage(frame)))

    @Slot()
    def _update_preview(self):
        """Stream the raw capture region into the experimenter view until the
        tracking thread takes over (it then supplies the annotated feed)."""
        if self.worker.capture_region is None:
            return
        # The ROI/reinit/zoom draw loops drive the display themselves.
        if self.selecting_roi:
            return
        # Once tracking is running, on_frame provides the annotated feed.
        if self.worker.isRunning():
            return
        frame = self.worker.grab_capture_frame()
        if frame is not None:
            self.last_frame = frame.copy()
            self._display_frame(frame)

    # ------------------------------------------------------------------
    # Capture region selection
    # ------------------------------------------------------------------
    @Slot()
    def on_load(self):
        screens = QGuiApplication.screens()
        if not screens:
            QMessageBox.critical(self, "Error", "No screen is available for capture")
            return
        try:
            self.hide()
            QApplication.processEvents()
            time.sleep(0.05)
            # one fullscreen selector per monitor so any screen can be captured
            selectors = []
            for s in screens:
                pix = s.grabWindow(0)
                sel = ScreenRegionSelector(pix, s.geometry())
                sel.winId()
                sel.windowHandle().setScreen(s)
                sel.showFullScreen()
                selectors.append(sel)
            while all(sel.isVisible() for sel in selectors):
                QApplication.processEvents()
                time.sleep(0.01)
            self.show()
            self.raise_()
            self.activateWindow()
            rect = None
            for sel in selectors:
                if sel.selected_rect is not None:
                    rect = sel.selected_rect
                    break
            for sel in selectors:
                if sel.isVisible():
                    sel.close()
            if rect is None or rect.width() <= 0 or rect.height() <= 0:
                return

            self.worker.set_capture_region(rect)
            aspect_ratio = rect.width() / rect.height()
            self.display_h = self.display_w / aspect_ratio
            fx = self.display_w / rect.width()
            fy = self.display_h / rect.height()
            self.worker.set_scaling_factor(fx, fy)
            w = max(1, int(rect.width() * self.worker.scale_fx))
            h = max(1, int(rect.height() * self.worker.scale_fy))
            self.display_w, self.display_h = w, h
            self.image_label.setFixedSize(w, h)
            self.image_label.clear_rects()
            self.image_label.setPixmap(QPixmap(w, h))
            self.secondary.showMaximized()
            self.show_status("Capture region selected. Click Select ROIs to draw.")
            frame = self.worker.grab_capture_frame()
            if frame is not None:
                self.last_frame = frame.copy()
                self._display_frame(frame)
            # start a continuous live preview of the region until tracking begins
            self.preview_timer.start()
            if not self.secondary_manual_override:
                self.worker.secondary_mode = MODE_COPY
                self.slider.blockSignals(True)
                self.slider.setValue(MODE_COPY)
                self.slider_label.setText(f"Mode {MODE_COPY}: {MODE_NAMES[MODE_COPY]}")
                self.slider.blockSignals(False)
        except Exception as e:
            self.show()
            QMessageBox.critical(self, "Error", f"Cannot capture screen region: {e}")

    # ------------------------------------------------------------------
    # ROI selection
    # ------------------------------------------------------------------
    @Slot()
    def on_select_rois_gui(self):
        self.chk_move_box.setChecked(False)
        n = self.spin_num.value()
        if self.worker.capture_region is None:
            QMessageBox.warning(self, "Warning", "Select a capture region first")
            return
        # Pause the running thread so its annotated frames (boxes/trails) stop
        # overwriting the clean feed while the user draws new ROIs.
        if self.worker.isRunning() and not self.worker.paused:
            self.worker.paused = True
        frame = self.worker.grab_capture_frame()
        if frame is None:
            QMessageBox.warning(self, "Warning", "Could not capture frame for ROI selection")
            return
        self.selecting_roi = True
        self.last_frame = frame.copy()
        self._display_frame(self.last_frame)
        self.image_label.enter_draw_mode()
        self.show_status(f"Draw {n} ROIs on the image (drag).")
        while True:
            QApplication.processEvents()
            frame = self.worker.grab_capture_frame()
            if frame is None:
                QMessageBox.warning(self, "Warning", "Could not capture frame for ROI selection")
                return
            self.last_frame = frame.copy()
            self._display_frame(self.last_frame)
            if len(self.image_label.get_rects_display()) >= n:
                break
            time.sleep(0.05)
        self.image_label.exit_draw_mode()
        rects = self.image_label.get_rects_display()[:n]
        rois = [(int(r.x()), int(r.y()), int(r.width()), int(r.height())) for r in rects]
        self.worker.set_num_trackers(n)
        self.worker.init_trackers_from_rois(self.last_frame, rois)
        self.combo_reinit.clear()
        self.combo_reinit.addItems([f"Tracker {i+1}" for i in range(n)])
        self.show_status("ROIs set and trackers initialized (GUI).")
        self.image_label.clear_rects()
        self.selecting_roi = False
        self.on_start_tracking()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_N and self.selecting_roi:
            frame = self.worker.grab_capture_frame()
            if frame is None:
                QMessageBox.warning(self, "Warning", "Could not capture frame for ROI selection")
                return
            self.last_frame = frame.copy()
            cv2.putText(self.last_frame, "Press N for next frame", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            self._display_frame(self.last_frame)

    # ------------------------------------------------------------------
    # Tracking control
    # ------------------------------------------------------------------
    @Slot()
    def on_start_tracking(self):
        if self.worker.capture_region is None:
            QMessageBox.warning(self, "Warning", "Select a capture region first")
            return
        if len(self.worker.trackers) < self.worker.num_trackers:
            QMessageBox.warning(self, "Warning", "Select ROIs first")
            return
        self.worker.use_kalman_filtering(self.chk_kf.isChecked())
        if not self.secondary_manual_override:
            self.worker.secondary_mode = MODE_FRAME_BOX
            self.slider.blockSignals(True)
            self.slider.setValue(MODE_FRAME_BOX)
            self.slider_label.setText(f"Mode {MODE_FRAME_BOX}: Frame+Box")
            self.slider.blockSignals(False)
        if not self.worker.isRunning():
            self.preview_timer.stop()
            self.worker.start()
            self.show_status("Processing started")
        else:
            self.worker.paused = False
            self.show_status("Resumed processing")

    @Slot()
    def on_pause(self):
        if self.worker.isRunning():
            self.worker.pause_toggle()

    @Slot()
    def on_manual_reinit_gui(self):
        self.chk_move_box.setChecked(False)
        idx = self.combo_reinit.currentIndex()
        if self.worker.capture_region is None:
            QMessageBox.warning(self, "Warning", "Select a capture region first")
            return
        was_running = self.worker.isRunning() and not self.worker.paused
        if was_running:
            self.worker.paused = True
        frame = self.worker.grab_capture_frame()
        if frame is None:
            QMessageBox.warning(self, "Warning", "Could not capture frame for manual reinit")
            return
        self.last_frame = frame.copy()
        self._display_frame(self.last_frame)
        self.image_label.clear_rects()
        self.image_label.enter_draw_mode()
        self.show_status(f"Draw ROI for tracker {idx+1}")
        while True:
            QApplication.processEvents()
            frame = self.worker.grab_capture_frame()
            if frame is None:
                QMessageBox.warning(self, "Warning", "Could not capture frame for manual reinit")
                return
            self.last_frame = frame.copy()
            self._display_frame(self.last_frame)
            if len(self.image_label.get_rects_display()) >= 1:
                break
            time.sleep(0.05)
        self.image_label.exit_draw_mode()
        r = self.image_label.get_rects_display()[0]
        new_roi = (int(r.x()), int(r.y()), int(r.width()), int(r.height()))
        self.worker.request_manual_reinit_with_roi((idx, new_roi))
        self.show_status(f"Manual reinit requested for tracker {idx+1}")
        self.image_label.clear_rects()
        if was_running:
            self.worker.paused = False

    @Slot()
    def on_export(self):
        if not self.worker.csv_rows:
            QMessageBox.information(self, "Info", "No data to export")
            return
        fname, _ = QFileDialog.getSaveFileName(self, "Save CSV", "tracking_output.csv", "CSV Files (*.csv)")
        if fname:
            self.worker.save_csv(fname)

    # ------------------------------------------------------------------
    # Frame display slots
    # ------------------------------------------------------------------
    @Slot(QImage)
    def on_frame(self, qt_img):
        self._measure_img_size = (qt_img.width(), qt_img.height())
        if self.measuring:      # experimenter view is frozen for measurement
            return
        pix = QPixmap.fromImage(qt_img).scaled(self.image_label.size(), Qt.KeepAspectRatio)
        disp_w = pix.width()
        disp_h = pix.height()
        img_w = max(1, qt_img.width())
        img_h = max(1, qt_img.height())

        show_box = self.chk_show_box_main.isChecked() and int(self.worker.secondary_mode) in (MODE_BLACK_BOX, MODE_FRAME_BOX)
        show_trails = self.chk_show_trails.isChecked()
        if show_box or show_trails:
            painter = QPainter(pix)
            if show_box:
                sec_w = max(1, self.worker.secondary_img_w)
                main_box_w = max(1, int(disp_w * self.worker.box_width_fraction))
                main_xmid = disp_w // 2
                main_offset = int(self.worker.box_x_offset * disp_w / sec_w)
                main_box_x = max(0, min(disp_w - main_box_w,
                                        int(main_xmid - main_box_w / 2) + main_offset))
                n = self.worker.box_num_shades
                a_max = self.worker.box_alpha_max
                step = (a_max - self.worker.box_alpha_min) / n
                green = QColor(0, 255, 0)
                for i in range(n):
                    seg_x = main_box_x + int(i * main_box_w / n)
                    seg_w = int((i + 1) * main_box_w / n) - int(i * main_box_w / n)
                    painter.setOpacity(a_max - i * step)
                    painter.fillRect(seg_x, 0, seg_w, disp_h, green)
            if show_trails:
                self._paint_trails(painter, disp_w, disp_h, img_w, img_h)
            painter.end()

        self.image_label.setPixmap(pix)

    def _paint_trails(self, painter, disp_w, disp_h, img_w, img_h):
        palette = [
            QColor(255, 140, 0), QColor(255, 0, 255), QColor(0, 255, 255),
            QColor(255, 215, 0), QColor(255, 100, 0), QColor(100, 255, 50),
            QColor(180, 0, 180), QColor(0, 200, 150), QColor(255, 50, 50),
            QColor(50, 150, 255),
        ]
        painter.setOpacity(1.0)
        trails_snap = self.worker.swallow_trails[-self.worker.n_swallow_display:]
        for si, trail in enumerate(trails_snap):
            painter.setPen(QPen(palette[si % len(palette)], 2))
            for fi in range(1, len(trail)):
                prev, curr = trail[fi - 1], trail[fi]
                for ti in range(min(len(prev), len(curr))):
                    painter.drawLine(int(prev[ti][0] * disp_w / img_w), int(prev[ti][1] * disp_h / img_h),
                                     int(curr[ti][0] * disp_w / img_w), int(curr[ti][1] * disp_h / img_h))
        if self.worker.swallow_active and len(self.worker.current_swallow_trail) > 1:
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            trail = self.worker.current_swallow_trail
            for fi in range(1, len(trail)):
                prev, curr = trail[fi - 1], trail[fi]
                for ti in range(min(len(prev), len(curr))):
                    painter.drawLine(int(prev[ti][0] * disp_w / img_w), int(prev[ti][1] * disp_h / img_h),
                                     int(curr[ti][0] * disp_w / img_w), int(curr[ti][1] * disp_h / img_h))

    @Slot(QImage)
    def on_secondary_image(self, qt_img):
        if self.secondary:
            self.secondary.update_image(qt_img)

    @Slot(str)
    def show_status(self, msg):
        self.status_label.setText(msg)

    @Slot()
    def on_finished(self):
        self.close()

    # ------------------------------------------------------------------
    # Control slots
    # ------------------------------------------------------------------
    @Slot(int)
    def on_num_changed(self, val):
        self.combo_reinit.clear()
        self.combo_reinit.addItems([f"Tracker {i+1}" for i in range(val)])
        self.worker.set_num_trackers(val)

    @Slot(int)
    def on_kf_toggled(self, state):
        self.worker.use_kalman_filtering(state == Qt.Checked)

    @Slot(int)
    def on_slider_changed(self, v):
        self.secondary_manual_override = True
        v = int(v)
        self.slider_label.setText(f"Mode {v}: {MODE_NAMES.get(v, '')}")
        self.worker.secondary_mode = v
        self.worker.secondary_manual_override = True
        if self.secondary:
            self.secondary.set_mode(v)

    @Slot(bool)
    def on_move_box_toggled(self, checked):
        if self.secondary:
            self.secondary.set_drag_mode(checked)
        self.image_label.set_box_drag_mode(checked)

    @Slot(bool)
    def on_swallow_toggled(self, checked):
        if checked:
            self.worker.start_swallow()
            self.btn_swallow.setText("Mark Swallow End (Ctrl+S)")
            self.btn_swallow.setStyleSheet("background-color: #cc2222; color: white; font-weight: bold;")
        else:
            self.worker.end_swallow()
            self.btn_swallow.setText("Mark Swallow Start (Ctrl+S)")
            self.btn_swallow.setStyleSheet("")

    # ------------------------------------------------------------------
    # Point-to-point measurement
    # ------------------------------------------------------------------
    @Slot(bool)
    def on_measure_toggled(self, checked):
        if checked:
            if self.selecting_roi:
                self.btn_measure.setChecked(False)
                return
            pm = self.image_label.pixmap()
            if pm is None or pm.isNull():
                QMessageBox.information(self, "Measure",
                                        "No frame to measure yet. Start the feed first.")
                self.btn_measure.setChecked(False)
                return
            if self.chk_move_box.isChecked():   # avoid conflicting mouse handling
                self.chk_move_box.setChecked(False)
            self.measuring = True
            self.measure_result.clear()
            self.image_label.enter_measure_mode()
            self.btn_measure.setText("Resume Live View (Ctrl+D)")
            self.btn_measure.setStyleSheet("background-color: #2277cc; color: white; font-weight: bold;")
            self.show_status("View frozen – click two points to measure the distance.")
        else:
            self.measuring = False
            self.image_label.exit_measure_mode()
            self.btn_measure.setText("Measure Distance (Ctrl+D)")
            self.btn_measure.setStyleSheet("")
            self.show_status("Live view resumed.")

    @Slot(QPoint, QPoint)
    def on_measure_points(self, p0, p1):
        pm = self.image_label.pixmap()
        if pm is None or pm.isNull():
            return
        pix_w, pix_h = pm.width(), pm.height()
        # the pixmap is centered inside the fixed-size label (Qt.AlignCenter)
        off_x = (self.image_label.width() - pix_w) / 2.0
        off_y = (self.image_label.height() - pix_h) / 2.0
        # map displayed-pixmap coords back to source-frame pixels
        if self._measure_img_size is not None:
            img_w, img_h = self._measure_img_size
            sx = img_w / max(1, pix_w)
            sy = img_h / max(1, pix_h)
        else:
            sx = sy = 1.0

        def to_frame(p):
            return (p.x() - off_x) * sx, (p.y() - off_y) * sy

        x0, y0 = to_frame(p0)
        x1, y1 = to_frame(p1)
        dist = math.hypot(x1 - x0, y1 - y0)
        self.measure_result.setText(f"Distance: {dist:.1f} px")
        self.image_label.set_measure_text(f"{dist:.1f} px")
        self.show_status(f"Distance: {dist:.1f} px  – click two new points to re-measure.")

    @Slot()
    def on_set_zoom_region(self):
        if self.worker.capture_region is None:
            QMessageBox.warning(self, "Warning", "Select a capture region first")
            return
        self.chk_move_box.setChecked(False)
        self.image_label.clear_rects()
        self.image_label.enter_draw_mode()
        self.show_status("Draw zoom region on the experimenter view, then release.")
        while True:
            QApplication.processEvents()
            if len(self.image_label.get_rects_display()) >= 1:
                break
            time.sleep(0.05)
        self.image_label.exit_draw_mode()
        r = self.image_label.get_rects_display()[0]
        self.image_label.clear_rects()
        # map experimenter-view (frame) space -> secondary-image space
        frame_w = max(1, self.display_w)
        frame_h = max(1, self.display_h)
        sec_w = max(1, self.worker.secondary_img_w)
        sec_h = max(1, self.worker.secondary_img_h)
        zx = int(r.x() * sec_w / frame_w)
        zy = int(r.y() * sec_h / frame_h)
        zw = max(1, int(r.width() * sec_w / frame_w))
        zh = max(1, int(r.height() * sec_h / frame_h))
        self.worker.set_zoom_region(zx, zy, zw, zh)
        self.lbl_zoom_mode.setText("Custom")
        self.lbl_zoom_mode.setStyleSheet("color: #cc6600; font-style: italic; font-weight: bold;")
        self.show_status("Custom zoom region set.")

    @Slot()
    def on_reset_zoom_region(self):
        self.worker.clear_zoom_region()
        self.lbl_zoom_mode.setText("Auto")
        self.lbl_zoom_mode.setStyleSheet("color: gray; font-style: italic;")
        self.show_status("Zoom region reset to auto.")

    @Slot()
    def on_box_width_clicked(self):
        prev_pct = self._box_width_pct
        dlg = QDialog(self)
        dlg.setWindowTitle("Adjust Shaded Box Width")
        dlg.setMinimumWidth(300)
        lbl = QLabel(f"Width: {prev_pct}% of view")
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(1)
        slider.setMaximum(50)
        slider.setValue(prev_pct)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(5)
        slider.valueChanged.connect(lambda v: (
            lbl.setText(f"Width: {v}% of view"),
            self.worker.set_box_width_fraction(v / 100.0),
        ))
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout = QVBoxLayout()
        layout.addWidget(lbl)
        layout.addWidget(slider)
        layout.addWidget(buttons)
        dlg.setLayout(layout)
        if dlg.exec() == QDialog.Accepted:
            self._box_width_pct = slider.value()
            self.btn_box_width.setText(f"Box Width: {self._box_width_pct}%")
        else:
            self.worker.set_box_width_fraction(prev_pct / 100.0)

    @Slot()
    def on_box_shading_clicked(self):
        prev_amax = self.worker.box_alpha_max
        prev_amin = self.worker.box_alpha_min
        prev_shades = self.worker.box_num_shades

        dlg = QDialog(self)
        dlg.setWindowTitle("Box Shading")
        dlg.setMinimumWidth(320)

        lbl_amax = QLabel(f"Max opacity (left edge): {int(prev_amax * 100)}%")
        sl_amax = QSlider(Qt.Horizontal)
        sl_amax.setMinimum(0); sl_amax.setMaximum(100)
        sl_amax.setValue(int(prev_amax * 100))
        sl_amax.setTickPosition(QSlider.TicksBelow); sl_amax.setTickInterval(10)
        sl_amax.valueChanged.connect(lambda v: (
            lbl_amax.setText(f"Max opacity (left edge): {v}%"),
            self.worker.set_box_alpha_max(v / 100.0),
        ))

        lbl_amin = QLabel(f"Min opacity (right edge): {int(prev_amin * 100)}%")
        sl_amin = QSlider(Qt.Horizontal)
        sl_amin.setMinimum(0); sl_amin.setMaximum(100)
        sl_amin.setValue(int(prev_amin * 100))
        sl_amin.setTickPosition(QSlider.TicksBelow); sl_amin.setTickInterval(10)
        sl_amin.valueChanged.connect(lambda v: (
            lbl_amin.setText(f"Min opacity (right edge): {v}%"),
            self.worker.set_box_alpha_min(v / 100.0),
        ))

        lbl_shades = QLabel(f"Number of shades: {prev_shades}")
        sl_shades = QSlider(Qt.Horizontal)
        sl_shades.setMinimum(5); sl_shades.setMaximum(50)
        sl_shades.setValue(prev_shades)
        sl_shades.setTickPosition(QSlider.TicksBelow); sl_shades.setTickInterval(5)
        sl_shades.valueChanged.connect(lambda v: (
            lbl_shades.setText(f"Number of shades: {v}"),
            self.worker.set_box_num_shades(v),
        ))

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)

        layout = QVBoxLayout()
        layout.addWidget(lbl_amax); layout.addWidget(sl_amax)
        layout.addSpacing(6)
        layout.addWidget(lbl_amin); layout.addWidget(sl_amin)
        layout.addSpacing(6)
        layout.addWidget(lbl_shades); layout.addWidget(sl_shades)
        layout.addSpacing(6)
        layout.addWidget(buttons)
        dlg.setLayout(layout)

        if dlg.exec() != QDialog.Accepted:
            self.worker.set_box_alpha_max(prev_amax)
            self.worker.set_box_alpha_min(prev_amin)
            self.worker.set_box_num_shades(prev_shades)

    def closeEvent(self, event):
        if self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(1000)
        if self.secondary:
            self.secondary.close()
        event.accept()


# =====================================================================
# Entry point
# =====================================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    moveExperimenterViewToSecondScreen = True
    win = MainWindow(
        num_of_tracker=1,
        use_kf=False,
        moveExperimenterViewToSecondScreen=moveExperimenterViewToSecondScreen,
    )
    win.show()
    sys.exit(app.exec())
