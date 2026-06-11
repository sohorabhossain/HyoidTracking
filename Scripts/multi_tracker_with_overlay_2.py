# multi_tracker_gui_secondary.py
import sys
import time
import math
import cv2
import numpy as np
import pandas as pd

from PySide6.QtCore import QThread, Signal, Slot, Qt, QRect, QPoint, QTimer
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QGuiApplication, QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QSpinBox, QComboBox, QMessageBox, QCheckBox, QSizePolicy,
    QSlider, QFrame, QDialog, QDialogButtonBox, QScrollArea,
)

# ----------------------------
# Multi-monitor screen grab helper
# ----------------------------
def _grab_all_screens():
    """Composite all connected screens into one pixmap spanning the virtual desktop.
    Returns (composite_pixmap, vdesktop_rect) in logical pixel coordinates,
    or (None, None) if no screens are available.
    """
    screens = QGuiApplication.screens()
    if not screens:
        return None, None
    vdesktop = screens[0].geometry()
    for s in screens[1:]:
        vdesktop = vdesktop.united(s.geometry())
    composite = QPixmap(vdesktop.width(), vdesktop.height())
    composite.fill(Qt.black)
    p = QPainter(composite)
    for s in screens:
        pix = s.grabWindow(0)
        if not pix.isNull():
            sg = s.geometry()
            dst = QRect(sg.x() - vdesktop.x(), sg.y() - vdesktop.y(), sg.width(), sg.height())
            p.drawPixmap(dst, pix)
    p.end()
    return composite, vdesktop

# ----------------------------
# Utility functions
# ----------------------------
def create_kalman_from_roi(roi, dt=1.0):
    kf = cv2.KalmanFilter(6, 4)
    kf.transitionMatrix = np.array([
        [1, 0, dt, 0, 0, 0],
        [0, 1, 0, dt, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ], np.float32)
    kf.measurementMatrix = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
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
    x, y, w, h = roi
    pad_x = int(w * frac)
    pad_y = int(h * frac)
    nx = max(0, x - pad_x)
    ny = max(0, y - pad_y)
    nw = min(frame_w - nx, w + 2 * pad_x)
    nh = min(frame_h - ny, h + 2 * pad_y)
    return (nx, ny, nw, nh)

def clamp_roi_to_frame(roi, frame_w, frame_h):
    x, y, w, h = map(int, roi)
    x = max(0, min(x, frame_w - 1))
    y = max(0, min(y, frame_h - 1))
    w = max(1, min(w, frame_w - x))
    h = max(1, min(h, frame_h - y))
    return (x, y, w, h)

def centered_search_region(roi, frame_w, frame_h, area_fraction=0.25):
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

# ----------------------------
# Drawable QLabel for GUI ROI drawing
# ----------------------------
class DrawableLabel(QLabel):
    box_offset_changed = Signal(int)  # emits absolute box_x_offset in secondary-image coords

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = None
        self.drawing = False
        self.start_point = QPoint()
        self.end_point = QPoint()
        self.rects = []
        self.temp_rect = None
        self.setMouseTracking(True)
        self.draw_mode = False
        # box drag state
        self._box_drag_mode = False
        self._box_drag_start_x = None
        self._box_drag_start_offset = 0
        self._box_current_offset = 0
        self._pix_w = 1   # width of last displayed pixmap (for coord mapping)
        self._sec_w = 1   # secondary image width (for coord mapping)

    def setPixmap(self, pixmap: QPixmap):
        super().setPixmap(pixmap)
        self._pixmap = pixmap
        self._pix_w = max(1, pixmap.width())

    def set_box_drag_mode(self, enabled: bool):
        self._box_drag_mode = enabled
        self.setCursor(Qt.SizeHorCursor if enabled else Qt.ArrowCursor)

    def set_sec_w(self, w: int):
        self._sec_w = max(1, w)

    def sync_offset(self, offset: int):
        """Update local offset from an external source without re-emitting."""
        self._box_current_offset = offset
        self._box_drag_start_offset = offset

    def enter_draw_mode(self):
        self.draw_mode = True
        self.rects = []
        self.temp_rect = None
        self.update()

    def exit_draw_mode(self):
        self.draw_mode = False
        self.temp_rect = None
        self.update()

    def mousePressEvent(self, event):
        if self._box_drag_mode and event.button() == Qt.LeftButton:
            self._box_drag_start_x = event.pos().x()
            self._box_drag_start_offset = self._box_current_offset
            return
        if not self.draw_mode:
            return
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.start_point = event.pos()
            self.end_point = event.pos()
            self.temp_rect = QRect(self.start_point, self.end_point)
            self.update()

    def mouseMoveEvent(self, event):
        if self._box_drag_mode and self._box_drag_start_x is not None:
            dx = event.pos().x() - self._box_drag_start_x
            dx_image = int(dx * self._sec_w / self._pix_w)
            self._box_current_offset = self._box_drag_start_offset + dx_image
            self.box_offset_changed.emit(self._box_current_offset)
            return
        if not self.draw_mode:
            return
        if self.drawing:
            self.end_point = event.pos()
            self.temp_rect = QRect(self.start_point, self.end_point).normalized()
            self.update()

    def mouseReleaseEvent(self, event):
        if self._box_drag_mode and event.button() == Qt.LeftButton:
            self._box_drag_start_x = None
            return
        if not self.draw_mode:
            return
        if event.button() == Qt.LeftButton and self.drawing:
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
        pen = QPen(Qt.green, 2)
        painter.setPen(pen)
        for r in self.rects:
            painter.drawRect(r)
        if self.temp_rect is not None:
            pen = QPen(Qt.yellow, 2)
            painter.setPen(pen)
            painter.drawRect(self.temp_rect)
        painter.end()

    def get_rects_display(self):
        return list(self.rects)

    def clear_rects(self):
        self.rects = []
        self.temp_rect = None
        self.update()

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
                # Map widget-proportional coords to logical screen coords so the
                # selected rect is correct regardless of DPI or window sizing.
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
        # Scale screenshot to fill the widget regardless of DPI scaling.
        painter.fillRect(widget_rect, Qt.black)
        painter.setOpacity(0.65)
        painter.drawPixmap(widget_rect, self.screenshot)
        painter.setOpacity(1.0)
        if self.start_point is not None and self.end_point is not None:
            rect = QRect(self.start_point, self.end_point).normalized()
            # Map selection (widget coords) → screenshot raw pixel coords.
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

# ----------------------------
# Secondary window (half-size image)
# ----------------------------
class SecondaryWindow(QWidget):
    box_offset_changed = Signal(int)   # emits absolute box_x_offset in image coords
    size_changed = Signal(int, int)    # emits (w, h) whenever the window is resized

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
        self.mode = 1  # 1=copy frame,2=black+line+circles,3=frame+line+circles
        self._drag_mode = False
        self._drag_start_x = None
        self._drag_start_offset = 0
        self._current_offset = 0
        self._image_w = 1  # last received image width for coord mapping
        self.setMouseTracking(True)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.size_changed.emit(event.size().width(), event.size().height())

    def update_image(self, qt_img):
        self._image_w = max(1, qt_img.width())
        pix = QPixmap.fromImage(qt_img)
        pix = pix.scaled(self.label.size(), Qt.KeepAspectRatio)
        self.label.setPixmap(pix)

    def set_mode(self, m):
        self.mode = int(m)

    def set_size(self, w, h):
        self.resize(int(w), int(h))

    def set_drag_mode(self, enabled: bool):
        self._drag_mode = enabled
        self.setCursor(Qt.SizeHorCursor if enabled else Qt.ArrowCursor)

    def reset_box_offset(self):
        self._current_offset = 0
        self._drag_start_offset = 0
        self.box_offset_changed.emit(0)

    def sync_offset(self, offset: int):
        """Update local offset from an external source without re-emitting."""
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

# ----------------------------
# Video processing thread (emits main frame and secondary QImage)
# ----------------------------
class VideoThread(QThread):
    change_pixmap = Signal(QImage)   # main display
    change_secondary = Signal(QImage)  # secondary window display
    status_msg = Signal(str)
    finished_processing = Signal()
    swallow_count_changed = Signal(int)  # emits updated swallow count

    def __init__(self):
        super().__init__()
        # config
        self.video_path = None
        self.cap = None
        self.capture_region = None
        self.scale_fx = 1.0 #0.65
        self.scale_fy = 1.0 #0.65
        self.num_trackers = 1
        self.fps_video = 60.0 #30.0

        # trackers
        self.trackers = []
        self.rois = []
        self.colors = []
        self.trails = []
        self.csv_rows = []

        # ORB
        self.orb = cv2.ORB_create(1000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.templates_kp = []
        self.templates_des = []
        self.templates_size = []

        # KF
        self.kalman_filters = []
        self.use_kf = True

        # reinit params
        self.match_thresh = 8
        self.template_pad_fraction = 0.20
        self.search_area_fraction = 1.0 # 0.25

        # control
        self.paused = True
        self.stop_requested = False
        self.manual_reinit_request = None
        self.frame_idx = 0
        self.frames_since_reinit = 0

        # secondary view mode (1,2,3) and manual override flag controlled by MainWindow
        self.secondary_mode = 3
        self.secondary_manual_override = False
        # horizontal pixel offset for gradient box in modes 2 and 3
        self.box_x_offset = 0
        # box width as fraction of half_w (default 1/16.67 ≈ 6%)
        self.box_width_fraction = 1.0 / 16.67
        # stepped-shading parameters
        self.box_alpha_max = 0.80
        self.box_alpha_min = 0.20
        self.box_num_shades = 5
        # target secondary image size — updated via set_secondary_size when window resizes
        self.secondary_img_w = 320
        self.secondary_img_h = 240
        # swallow tracking
        self.swallow_active = False
        self.current_swallow_trail = []   # [(cx,cy),...] per frame while recording
        self.swallow_trails = []          # list of completed swallow trails
        self.swallow_count = 0
        self.n_swallow_display = 3        # how many recent swallows to show
        self.show_swallow_trails = True   # whether to render trajectories
        self.zoom_participant = False     # zoom participant view around the gradient box
        self.zoom_region = None          # None = auto; (x,y,w,h) in secondary-image coords = manual
        # mode-4 strength meter
        self.last_swallow_excursion = 0.0       # excursion of last completed swallow (frame px)
        self.strength_scale_max_displacement = 30.0   # auto-expands independently per metric
        self.strength_scale_max_arc_length   = 500.0
        self.strength_metric = "displacement"  # "displacement" or "arc_length"
        # mode-5 speedometer
        self.last_swallow_peak_speed = 0.0  # peak speed of last completed swallow (frame px/s)
        self.speed_scale_max = 2500.0        # top of speedometer scale (frame px/s); auto-expands
        self.auto_expand_strength = True
        self.auto_expand_speed    = True
        self.show_participant_labels = True
        # cached last frame for forced re-renders (e.g. when trails toggled while paused)
        self._last_sec_frame = None
        self._last_tracker_centers = []
        self._request_secondary_redraw = False
        # per-tracker circle color state for modes 2 & 3
        self._circle_entered_box_time = []  # None or float timestamp when circle entered box
        self._circle_yellow = []            # bool: True → draw yellow instead of red
        self.circle_radius = 6              # tracker circle radius (px) in secondary modes 2 & 3

        # writer
        self.video_writer = None

    def use_kalman_filtering(self, choice: bool):
        self.use_kf = bool(choice)
        self.status_msg.emit(f"Kalman filtering {'enabled' if self.use_kf else 'disabled'}")
        if self.use_kf and len(self.rois) > 0:
            for i in range(len(self.rois)):
                if i >= len(self.kalman_filters) or self.kalman_filters[i] is None:
                    try:
                        kf = create_kalman_from_roi(self.rois[i])
                        if i >= len(self.kalman_filters):
                            self.kalman_filters.extend([None] * (i - len(self.kalman_filters) + 1))
                        self.kalman_filters[i] = kf
                    except Exception:
                        if i >= len(self.kalman_filters):
                            self.kalman_filters.extend([None] * (i - len(self.kalman_filters) + 1))
                        self.kalman_filters[i] = None

    def set_scaling_factor(self, fx, fy):
        self.scale_fx = fx
        self.scale_fy = fy

    def set_capture_region(self, rect):
        self.capture_region = (
            int(rect.x()),
            int(rect.y()),
            int(rect.width()),
            int(rect.height()),
        )
        self.status_msg.emit(
            f"Capture region set: {self.capture_region[2]}x{self.capture_region[3]} at "
            f"({self.capture_region[0]}, {self.capture_region[1]})"
        )

    def grab_capture_frame(self):
        if self.capture_region is None:
            return None
        x, y, w, h = self.capture_region
        # Find the screen with the largest intersection with the capture region.
        # This allows captures on secondary/USB/WiFi monitors.
        cap_rect = QRect(x, y, w, h)
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
        # Grab that screen and map the capture region into its local coordinates.
        full_pixmap = screen.grabWindow(0)
        if full_pixmap.isNull():
            return None
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
            tr = self.create_local_tracker(frame, roi)
            self.trackers.append(tr)
            self.colors.append(tuple(int(c) for c in rng.integers(50, 255, 3)))
            self.trails.append([])

            ex_roi = expand_roi(roi, self.template_pad_fraction, w_frame, h_frame)
            x2, y2, w2, h2 = map(int, ex_roi)
            templ = frame[y2:y2 + h2, x2:x2 + w2]
            templ_gray = cv2.cvtColor(templ, cv2.COLOR_BGR2GRAY)
            kp, des = self.orb.detectAndCompute(templ_gray, None) if templ_gray.size > 0 else ([], None)
            self.templates_kp.append(kp)
            self.templates_des.append(des)
            self.templates_size.append((w2, h2))

            if self.use_kf:
                try:
                    kf = create_kalman_from_roi(roi)
                except Exception:
                    kf = None
            else:
                kf = None
            self.kalman_filters.append(kf)

    def create_local_tracker(self, frame, roi):
        frame_h, frame_w = frame.shape[:2]
        roi = clamp_roi_to_frame(roi, frame_w, frame_h)
        search_x, search_y, search_w, search_h = centered_search_region(
            roi, frame_w, frame_h, self.search_area_fraction
        )
        search_frame = frame[search_y:search_y + search_h, search_x:search_x + search_w]
        local_roi = (
            roi[0] - search_x,
            roi[1] - search_y,
            roi[2],
            roi[3],
        )
        tr = cv2.legacy.TrackerCSRT_create()
        tr.init(search_frame, local_roi)
        return tr

    def update_tracker_in_local_region(self, tracker, frame, roi):
        frame_h, frame_w = frame.shape[:2]
        roi = clamp_roi_to_frame(roi, frame_w, frame_h)
        search_x, search_y, search_w, search_h = centered_search_region(
            roi, frame_w, frame_h, self.search_area_fraction
        )
        search_frame = frame[search_y:search_y + search_h, search_x:search_x + search_w]
        ok, local_roi = tracker.update(search_frame)
        if not ok:
            return False, roi
        lx, ly, lw, lh = map(int, local_roi)
        global_roi = clamp_roi_to_frame(
            (search_x + lx, search_y + ly, lw, lh),
            frame_w,
            frame_h,
        )
        return True, global_roi

    def try_orb_reinit(self, frame_gray, templ_kp, templ_des, templ_size):
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
        matches = sorted(matches, key=lambda x: x.distance)
        good = matches[: max(10, int(len(matches)*0.25))]
        if len(good) < self.match_thresh:
            return None
        pts_template = np.float32([templ_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts_scene = np.float32([kp_scene[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        try:
            H, mask = cv2.findHomography(pts_template, pts_scene, cv2.RANSAC, 5.0)
            if H is None:
                return None
        except Exception:
            return None
        th, tw = templ_size[1], templ_size[0]
        corners = np.float32([[0,0],[tw,0],[tw,th],[0,th]]).reshape(-1,1,2)
        try:
            transformed = cv2.perspectiveTransform(corners, H)
        except Exception:
            return None
        pts = transformed.reshape(-1,2)
        min_x = max(0, int(np.min(pts[:,0])))
        min_y = max(0, int(np.min(pts[:,1])))
        max_x = min(frame_gray.shape[1]-1, int(np.max(pts[:,0])))
        max_y = min(frame_gray.shape[0]-1, int(np.max(pts[:,1])))
        w_new = max(1, max_x - min_x)
        h_new = max(1, max_y - min_y)
        return (min_x, min_y, w_new, h_new)

    def emit_secondary_image(self, frame, tracker_centers):
        """
        Build and emit secondary image based on self.secondary_mode.
        frame: processed (resized) BGR frame
        tracker_centers: list of (cx,cy) in processed-frame coords
        """
        h, w = frame.shape[:2]
        half_w = max(1, self.secondary_img_w)
        half_h = max(1, self.secondary_img_h)
        mode = int(self.secondary_mode)
        # Option 1: copy resized frame scaled to half
        if mode == 1:
            sec = cv2.resize(frame, (half_w, half_h))
        elif mode == 2:
            sec = np.zeros((half_h, half_w, 3), dtype=np.uint8)
            xmid = half_w // 2
            box_w = max(1, int(half_w * self.box_width_fraction))
            box_x = max(0, min(half_w - box_w, int(xmid - box_w / 2) + self.box_x_offset))
            x_end = min(half_w, box_x + box_w)
            actual_w = x_end - box_x
            _n = self.box_num_shades
            _step = (self.box_alpha_max - self.box_alpha_min) / _n
            _alphas_full = np.empty(box_w, dtype=np.float32)
            for _i in range(_n):
                _alphas_full[int(_i * box_w / _n):int((_i + 1) * box_w / _n)] = self.box_alpha_max - _i * _step
            alphas = _alphas_full[:actual_w][np.newaxis, :, np.newaxis]
            green = np.array([0, 255, 0], dtype=np.float32)
            sec[:, box_x:x_end] = (
                sec[:, box_x:x_end].astype(np.float32) * (1 - alphas) + green * alphas
            ).astype(np.uint8)
            # draw circles; turn yellow after 3 s inside/left of box, red once right of box
            _n_c = len(tracker_centers)
            while len(self._circle_entered_box_time) < _n_c:
                self._circle_entered_box_time.append(None)
                self._circle_yellow.append(False)
            _now = time.time()
            for _idx, (cx, cy) in enumerate(tracker_centers):
                sx = int(cx * half_w / float(w))
                sy = int(cy * half_h / float(h))
                if sx <= x_end:
                    if self._circle_entered_box_time[_idx] is None:
                        self._circle_entered_box_time[_idx] = _now
                    if _now - self._circle_entered_box_time[_idx] >= 3.0:
                        self._circle_yellow[_idx] = True
                else:
                    self._circle_entered_box_time[_idx] = None
                    self._circle_yellow[_idx] = False
                _clr = (0, 255, 255) if self._circle_yellow[_idx] else (0, 0, 255)
                cv2.circle(sec, (sx, sy), self.circle_radius, _clr, -1)
        elif mode == 3:  # frame background + gradient box + circles
            sec = cv2.resize(frame, (half_w, half_h))
            xmid = half_w // 2
            box_w = max(1, int(half_w * self.box_width_fraction))
            box_x = max(0, min(half_w - box_w, int(xmid - box_w / 2) + self.box_x_offset))
            x_end = min(half_w, box_x + box_w)
            actual_w = x_end - box_x
            _n = self.box_num_shades
            _step = (self.box_alpha_max - self.box_alpha_min) / _n
            _alphas_full = np.empty(box_w, dtype=np.float32)
            for _i in range(_n):
                _alphas_full[int(_i * box_w / _n):int((_i + 1) * box_w / _n)] = self.box_alpha_max - _i * _step
            alphas = _alphas_full[:actual_w][np.newaxis, :, np.newaxis]
            green = np.array([0, 255, 0], dtype=np.float32)
            sec[:, box_x:x_end] = (
                sec[:, box_x:x_end].astype(np.float32) * (1 - alphas) + green * alphas
            ).astype(np.uint8)
            _n_c = len(tracker_centers)
            while len(self._circle_entered_box_time) < _n_c:
                self._circle_entered_box_time.append(None)
                self._circle_yellow.append(False)
            _now = time.time()
            for _idx, (cx, cy) in enumerate(tracker_centers):
                sx = int(cx * half_w / float(w))
                sy = int(cy * half_h / float(h))
                if sx <= x_end:
                    if self._circle_entered_box_time[_idx] is None:
                        self._circle_entered_box_time[_idx] = _now
                    if _now - self._circle_entered_box_time[_idx] >= 3.0:
                        self._circle_yellow[_idx] = True
                else:
                    self._circle_entered_box_time[_idx] = None
                    self._circle_yellow[_idx] = False
                _clr = (0, 255, 255) if self._circle_yellow[_idx] else (0, 0, 255)
                cv2.circle(sec, (sx, sy), self.circle_radius, _clr, -1)
        elif mode == 4:  # swallow strength meter
            sec = np.zeros((half_h, half_w, 3), dtype=np.uint8)
            # --- current excursion to display ---
            if self.swallow_active and self.current_swallow_trail:
                if len(self.current_swallow_trail) >= 2:
                    _n_tr = max(len(_fp) for _fp in self.current_swallow_trail)
                    _cur_exc = 0.0
                    if self.strength_metric == "displacement":
                        _first_fr = self.current_swallow_trail[0]
                        _last_fr  = self.current_swallow_trail[-1]
                        for _ti in range(_n_tr):
                            if _ti < len(_first_fr) and _ti < len(_last_fr):
                                _ddx = _last_fr[_ti][0] - _first_fr[_ti][0]
                                _ddy = _last_fr[_ti][1] - _first_fr[_ti][1]
                                _cur_exc = max(_cur_exc, math.sqrt(_ddx * _ddx + _ddy * _ddy))
                    else:  # arc_length
                        for _ti in range(_n_tr):
                            _tot = 0.0
                            for _fi in range(1, len(self.current_swallow_trail)):
                                _pp = self.current_swallow_trail[_fi - 1]
                                _cp = self.current_swallow_trail[_fi]
                                if _ti < len(_pp) and _ti < len(_cp):
                                    _ddx = _cp[_ti][0] - _pp[_ti][0]
                                    _ddy = _cp[_ti][1] - _pp[_ti][1]
                                    _tot += math.sqrt(_ddx * _ddx + _ddy * _ddy)
                            _cur_exc = max(_cur_exc, _tot)
                else:
                    _cur_exc = 0.0
            else:
                _cur_exc = self.last_swallow_excursion
            _scale_max = max(1.0, self.strength_scale_max_displacement
                            if self.strength_metric == "displacement"
                            else self.strength_scale_max_arc_length)
            _ratio = min(1.0, _cur_exc / _scale_max)

            # --- layout ---
            _bar_cx   = int(half_w * 0.38)          # horizontal centre of bar
            _bar_w_px = max(6, int(half_w * 0.14))  # bar width
            _bar_x    = _bar_cx - _bar_w_px // 2
            _bar_top  = int(half_h * 0.12)
            _bar_bot  = int(half_h * 0.84)
            _bar_h    = max(1, _bar_bot - _bar_top)

            # --- draw full gradient background of bar (red→yellow→green bottom→top) ---
            _full_ys = np.linspace(0.0, 1.0, _bar_h, dtype=np.float32)  # 0=bottom,1=top
            _bar_B = np.zeros(_bar_h, dtype=np.uint8)
            _bar_R = np.where(_full_ys < 0.5,
                              200,
                              np.clip(200 - (_full_ys - 0.5) * 400, 0, 200)).astype(np.uint8)
            _bar_G = np.where(_full_ys < 0.5,
                              np.clip(_full_ys * 400, 0, 200),
                              200).astype(np.uint8)
            _bar_colors = np.stack([_bar_B, _bar_G, _bar_R], axis=1)  # (H,3)
            # flip so row-0 of array = bar_top = high value (red)
            _bar_colors = _bar_colors[::-1]
            _needle_row = int((1.0 - _ratio) * _bar_h)  # row index of needle within bar
            # dim the unfilled portion above the needle
            _bar_colors[:_needle_row] = (_bar_colors[:_needle_row] // 5)
            _bar_slice = sec[_bar_top:_bar_bot, _bar_x + 1:_bar_x + _bar_w_px - 1]
            _bar_slice[:] = _bar_colors[:, np.newaxis, :]

            # --- bar outline ---
            cv2.rectangle(sec, (_bar_x, _bar_top), (_bar_x + _bar_w_px, _bar_bot),
                          (160, 160, 160), 1)

            # --- needle (white horizontal line + small side triangles) ---
            _needle_y = _bar_top + _needle_row
            _nx1, _nx2 = _bar_x - 14, _bar_x + _bar_w_px + 14
            cv2.line(sec, (_nx1, _needle_y), (_nx2, _needle_y), (255, 255, 255), 3)
            # left pointer triangle
            pts_l = np.array([[_bar_x - 2, _needle_y],
                               [_bar_x - 14, _needle_y - 6],
                               [_bar_x - 14, _needle_y + 6]], np.int32)
            cv2.fillPoly(sec, [pts_l], (255, 255, 255))
            # right pointer triangle
            pts_r = np.array([[_bar_x + _bar_w_px + 2, _needle_y],
                               [_bar_x + _bar_w_px + 14, _needle_y - 6],
                               [_bar_x + _bar_w_px + 14, _needle_y + 6]], np.int32)
            cv2.fillPoly(sec, [pts_r], (255, 255, 255))

            # --- scale ticks and labels (right of bar) ---
            _tick_x  = _bar_x + _bar_w_px + 2
            _lbl_x   = _tick_x + 10
            _font_sc = max(0.25, half_h / 900)
            if self.show_participant_labels:
                for _pct in [0, 25, 50, 75, 100]:
                    _ty = int(_bar_bot - (_pct / 100.0) * _bar_h)
                    cv2.line(sec, (_tick_x, _ty), (_tick_x + 7, _ty), (180, 180, 180), 1)
                    cv2.putText(sec, f"{_pct / 100.0 * _scale_max:.0f}",
                                (_lbl_x, _ty + 4), cv2.FONT_HERSHEY_SIMPLEX,
                                _font_sc, (180, 180, 180), 1)

            # --- previous swallow markers on left of bar ---
            _prev_excs = []
            for _tr in self.swallow_trails:
                if len(_tr) >= 2:
                    _n_tr2 = max(len(_fp) for _fp in _tr)
                    _tr_exc = 0.0
                    for _ti in range(_n_tr2):
                        _tot2 = 0.0
                        for _fi in range(1, len(_tr)):
                            _pp2 = _tr[_fi - 1]
                            _cp2 = _tr[_fi]
                            if _ti < len(_pp2) and _ti < len(_cp2):
                                _ddx2 = _cp2[_ti][0] - _pp2[_ti][0]
                                _ddy2 = _cp2[_ti][1] - _pp2[_ti][1]
                                _tot2 += math.sqrt(_ddx2 * _ddx2 + _ddy2 * _ddy2)
                        _tr_exc = max(_tr_exc, _tot2)
                    _prev_excs.append(_tr_exc)
            for _pi, _pexc in enumerate(_prev_excs):
                _pr = min(1.0, _pexc / _scale_max)
                _py = int(_bar_bot - _pr * _bar_h)
                _dim = max(60, 220 - _pi * 50)
                cv2.line(sec, (_bar_x - 18, _py), (_bar_x - 3, _py), (_dim, _dim, _dim), 2)

            # --- title and labels ---
            _title_x = max(4, _bar_x - int(half_w * 0.15))
            _lh = max(16, int(_font_sc * 1.2 * 28))
            cv2.putText(sec, "SWALLOW", (_title_x, _bar_top - _lh - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, _font_sc * 1.2, (220, 220, 220), 1)
            cv2.putText(sec, "STRENGTH", (_title_x, _bar_top - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, _font_sc * 1.2, (220, 220, 220), 1)
            if self.show_participant_labels:
                cv2.putText(sec, f"{_cur_exc:.1f} px",
                            (_bar_x, _bar_bot + int(half_h * 0.055)),
                            cv2.FONT_HERSHEY_SIMPLEX, _font_sc * 1.1, (255, 255, 255), 1)
                _metric_lbl = "Disp." if self.strength_metric == "displacement" else "Arc Len."
                cv2.putText(sec, _metric_lbl,
                            (_bar_x, _bar_bot + int(half_h * 0.10)),
                            cv2.FONT_HERSHEY_SIMPLEX, _font_sc * 0.85, (160, 160, 160), 1)
                cv2.putText(sec, f"Swallows: {self.swallow_count}",
                            (int(half_w * 0.55), int(half_h * 0.94)),
                            cv2.FONT_HERSHEY_SIMPLEX, _font_sc, (200, 200, 200), 1)
                if self.swallow_active:
                    cv2.putText(sec, "LIVE",
                                (int(half_w * 0.72), _bar_top + int(half_h * 0.04)),
                                cv2.FONT_HERSHEY_SIMPLEX, _font_sc * 1.4, (0, 80, 255), 2)
        elif mode == 5:  # speedometer
            sec = np.zeros((half_h, half_w, 3), dtype=np.uint8)
            # --- current speed to display (rolling max over last 5 frames while live) ---
            if self.swallow_active and len(self.current_swallow_trail) >= 2:
                _win = min(5, len(self.current_swallow_trail) - 1)
                _live_spd = 0.0
                for _wi in range(_win):
                    _fi = len(self.current_swallow_trail) - 1 - _wi
                    _ppts = self.current_swallow_trail[_fi - 1]
                    _cpts = self.current_swallow_trail[_fi]
                    for _ti in range(min(len(_ppts), len(_cpts))):
                        _dx = _cpts[_ti][0] - _ppts[_ti][0]
                        _dy = _cpts[_ti][1] - _ppts[_ti][1]
                        _live_spd = max(_live_spd,
                                        math.sqrt(_dx * _dx + _dy * _dy) * self.fps_video)
                _cur_spd = _live_spd
            else:
                _cur_spd = self.last_swallow_peak_speed
            _spd_scale = max(1.0, self.speed_scale_max)
            _ratio = min(1.0, _cur_spd / _spd_scale)

            # --- layout ---
            _cx   = half_w // 2
            _cy   = int(half_h * 0.62)
            _r    = int(min(half_w, half_h) * 0.40)
            _athk = max(3, _r // 8)          # arc stroke thickness
            _fsc  = max(0.28, half_h / 900)  # font scale

            # Speedometer arc: cv2 angle 135° (lower-left) to 405° (=45°, lower-right)
            # going clockwise in cv2 (visually counterclockwise — left→top→right).
            # color gradient: green→yellow→red along the arc.
            _n_segs = 90
            for _s in range(_n_segs):
                _t0 = _s / _n_segs
                _t1 = (_s + 1) / _n_segs
                _a0 = math.radians(135.0 + _t0 * 270.0)
                _a1 = math.radians(135.0 + _t1 * 270.0)
                _x0 = int(_cx + _r * math.cos(_a0))
                _y0 = int(_cy + _r * math.sin(_a0))
                _x1 = int(_cx + _r * math.cos(_a1))
                _y1 = int(_cy + _r * math.sin(_a1))
                # BGR color along arc: red(0,0,200)→yellow(0,200,200)→green(0,200,0)
                if _t0 < 0.5:
                    _sc = (_B, _G, _R) = (0, int(_t0 * 2 * 200), 200)
                else:
                    _sc = (_B, _G, _R) = (0, 200, int((1.0 - (_t0 - 0.5) * 2) * 200))
                if _t0 > _ratio:
                    _sc = (_B // 5, _G // 5, _R // 5)
                cv2.line(sec, (_x0, _y0), (_x1, _y1), _sc, _athk)

            # --- major ticks and labels (5 marks: 0 / 25 / 50 / 75 / 100 %) ---
            if self.show_participant_labels:
                for _pct in [0, 25, 50, 75, 100]:
                    _ta = math.radians(135.0 + _pct / 100.0 * 270.0)
                    _cos_ta, _sin_ta = math.cos(_ta), math.sin(_ta)
                    _ox = int(_cx + _r * 1.04 * _cos_ta)
                    _oy = int(_cy + _r * 1.04 * _sin_ta)
                    _ix = int(_cx + _r * 0.82 * _cos_ta)
                    _iy = int(_cy + _r * 0.82 * _sin_ta)
                    cv2.line(sec, (_ix, _iy), (_ox, _oy), (200, 200, 200), 2)
                    _lx = int(_cx + _r * 1.22 * _cos_ta) - 14
                    _ly = int(_cy + _r * 1.22 * _sin_ta) + 4
                    cv2.putText(sec, f"{_pct / 100.0 * _spd_scale:.0f}",
                                (_lx, _ly), cv2.FONT_HERSHEY_SIMPLEX,
                                _fsc * 0.85, (160, 160, 160), 1)

            # --- minor ticks (every 10 %) ---
            if self.show_participant_labels:
                for _pct10 in range(0, 101, 10):
                    if _pct10 % 25 == 0:
                        continue
                    _ta = math.radians(135.0 + _pct10 / 100.0 * 270.0)
                    _ox = int(_cx + _r * 1.04 * math.cos(_ta))
                    _oy = int(_cy + _r * 1.04 * math.sin(_ta))
                    _ix = int(_cx + _r * 0.93 * math.cos(_ta))
                    _iy = int(_cy + _r * 0.93 * math.sin(_ta))
                    cv2.line(sec, (_ix, _iy), (_ox, _oy), (110, 110, 110), 1)

            # --- needle ---
            _na    = math.radians(135.0 + _ratio * 270.0)
            _tip_x = int(_cx + _r * 0.80 * math.cos(_na))
            _tip_y = int(_cy + _r * 0.80 * math.sin(_na))
            _bas_x = int(_cx - _r * 0.14 * math.cos(_na))
            _bas_y = int(_cy - _r * 0.14 * math.sin(_na))
            cv2.line(sec, (_bas_x, _bas_y), (_tip_x, _tip_y), (255, 255, 255), 3)
            cv2.circle(sec, (_cx, _cy), max(5, _r // 10), (180, 180, 180), -1)

            # --- title ---
            _ttl = "SWALLOW SPEED"
            _tw  = cv2.getTextSize(_ttl, cv2.FONT_HERSHEY_SIMPLEX, _fsc * 1.15, 1)[0][0]
            cv2.putText(sec, _ttl, (_cx - _tw // 2, int(half_h * 0.09)),
                        cv2.FONT_HERSHEY_SIMPLEX, _fsc * 1.15, (220, 220, 220), 1)

            # --- value inside gauge ---
            if self.show_participant_labels:
                _vstr = f"{_cur_spd:.1f} px/s"
                _vw   = cv2.getTextSize(_vstr, cv2.FONT_HERSHEY_SIMPLEX, _fsc * 1.1, 1)[0][0]
                cv2.putText(sec, _vstr, (_cx - _vw // 2, int(_cy + _r * 0.36)),
                            cv2.FONT_HERSHEY_SIMPLEX, _fsc * 1.1, (255, 255, 255), 1)
                _lbl2 = "(peak)" if not self.swallow_active else "(live)"
                _lw   = cv2.getTextSize(_lbl2, cv2.FONT_HERSHEY_SIMPLEX, _fsc * 0.8, 1)[0][0]
                cv2.putText(sec, _lbl2, (_cx - _lw // 2, int(_cy + _r * 0.52)),
                            cv2.FONT_HERSHEY_SIMPLEX, _fsc * 0.8, (160, 160, 160), 1)

            # --- swallow count & LIVE badge ---
            if self.show_participant_labels:
                cv2.putText(sec, f"Swallows: {self.swallow_count}",
                            (int(half_w * 0.05), int(half_h * 0.95)),
                            cv2.FONT_HERSHEY_SIMPLEX, _fsc, (200, 200, 200), 1)
                if self.swallow_active:
                    cv2.putText(sec, "LIVE",
                                (int(half_w * 0.76), int(half_h * 0.10)),
                                cv2.FONT_HERSHEY_SIMPLEX, _fsc * 1.4, (0, 80, 255), 2)
        else:
            sec = cv2.resize(frame, (half_w, half_h))

        # draw swallow trajectories over secondary image (modes 1-3 only)
        if self.show_swallow_trails and mode in (1, 2, 3):
            _trail_palette = [
                (0, 140, 255), (255, 0, 255), (0, 255, 255), (255, 215, 0),
                (255, 100, 0), (100, 255, 50), (180, 0, 180), (0, 200, 150),
                (255, 50, 50),  (50, 150, 255),
            ]
            for _si, _trail in enumerate(self.swallow_trails[-self.n_swallow_display:]):
                _col = _trail_palette[_si % len(_trail_palette)]
                for _fi in range(1, len(_trail)):
                    _prev, _curr = _trail[_fi - 1], _trail[_fi]
                    for _ti in range(min(len(_prev), len(_curr))):
                        _p1 = (int(_prev[_ti][0] * half_w / float(w)),
                               int(_prev[_ti][1] * half_h / float(h)))
                        _p2 = (int(_curr[_ti][0] * half_w / float(w)),
                               int(_curr[_ti][1] * half_h / float(h)))
                        cv2.line(sec, _p1, _p2, _col, 2)
            # active (in-progress) swallow drawn in white
            if self.swallow_active and len(self.current_swallow_trail) > 1:
                for _fi in range(1, len(self.current_swallow_trail)):
                    _prev = self.current_swallow_trail[_fi - 1]
                    _curr = self.current_swallow_trail[_fi]
                    for _ti in range(min(len(_prev), len(_curr))):
                        _p1 = (int(_prev[_ti][0] * half_w / float(w)),
                               int(_prev[_ti][1] * half_h / float(h)))
                        _p2 = (int(_curr[_ti][0] * half_w / float(w)),
                               int(_curr[_ti][1] * half_h / float(h)))
                        cv2.line(sec, _p1, _p2, (255, 255, 255), 2)

        # zoom into the region around the gradient box if enabled (modes 1-3 only)
        if self.zoom_participant and mode in (1, 2, 3):
            if self.zoom_region is not None:
                _zx, _zy, _zw, _zh = self.zoom_region
                _zx = min(_zx, half_w - 1)
                _zy = min(_zy, half_h - 1)
                _zw = min(_zw, half_w - _zx)
                _zh = min(_zh, half_h - _zy)
                if _zw > 0 and _zh > 0:
                    sec = cv2.resize(sec[_zy:_zy + _zh, _zx:_zx + _zw], (half_w, half_h))
            else:
                _xmid = half_w // 2
                _bw = max(1, int(half_w * self.box_width_fraction))
                _bx = max(0, min(half_w - _bw, int(_xmid - _bw / 2) + self.box_x_offset))
                _zoom_x = max(0, _bx - _bw)
                if _zoom_x < half_w - 1:
                    sec = cv2.resize(sec[:, _zoom_x:], (half_w, half_h))

        # convert BGR->RGB->QImage
        rgb = cv2.cvtColor(sec, cv2.COLOR_BGR2RGB)
        h2, w2, ch = rgb.shape
        bytes_per_line = ch * w2
        qt_img = QImage(rgb.data, w2, h2, bytes_per_line, QImage.Format_RGB888).copy()
        self.change_secondary.emit(qt_img)

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
        self.video_writer = cv2.VideoWriter("tracked_output_gui.mp4", fourcc, self.fps_video, (frame_w, frame_h))

        while not self.stop_requested:
            if self.paused:
                if self._request_secondary_redraw and self._last_sec_frame is not None:
                    self._request_secondary_redraw = False
                    try:
                        self.emit_secondary_image(self._last_sec_frame,
                                                   self._last_tracker_centers)
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
                tracker = self.trackers[i]
                ok, new_roi = self.update_tracker_in_local_region(tracker, frame, self.rois[i])
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
                    measured = np.array([np.float32(cx), np.float32(cy), np.float32(w), np.float32(h)], dtype=np.float32)
                    ex_roi = expand_roi((x,y,w,h), self.template_pad_fraction, gray.shape[1], gray.shape[0])
                    ex_x, ex_y, ex_w, ex_h = map(int, ex_roi)
                    templ_img = frame[ex_y:ex_y+ex_h, ex_x:ex_x+ex_w]
                    templ_gray = cv2.cvtColor(templ_img, cv2.COLOR_BGR2GRAY)
                    kp, des = self.orb.detectAndCompute(templ_gray, None) if templ_gray.size>0 else ([], None)
                    self.templates_kp[i] = kp
                    self.templates_des[i] = des
                    self.templates_size[i] = (ex_w, ex_h)
                    if self.use_kf:
                        if self.kalman_filters[i] is None:
                            self.kalman_filters[i] = create_kalman_from_roi((x,y,w,h))
                        else:
                            try:
                                self.kalman_filters[i].correct(measured)
                            except Exception:
                                self.kalman_filters[i] = create_kalman_from_roi((x,y,w,h))
                else:
                    reinit_attempted[i] = True
                    kp_t = self.templates_kp[i]
                    des_t = self.templates_des[i]
                    tsize = self.templates_size[i]
                    if des_t is not None and len(des_t) >= 4:
                        new_roi = self.try_orb_reinit(gray, kp_t, des_t, tsize)
                        if new_roi is not None:
                            try:
                                new_tracker = self.create_local_tracker(frame, new_roi)
                                self.trackers[i] = new_tracker
                                self.rois[i] = new_roi
                                ex_x, ex_y, ex_w, ex_h = map(int, expand_roi(new_roi, self.template_pad_fraction, gray.shape[1], gray.shape[0]))
                                templ_img = frame[ex_y:ex_y+ex_h, ex_x:ex_x+ex_w]
                                templ_gray = cv2.cvtColor(templ_img, cv2.COLOR_BGR2GRAY)
                                kp, des = self.orb.detectAndCompute(templ_gray, None) if templ_gray.size>0 else ([], None)
                                self.templates_kp[i] = kp
                                self.templates_des[i] = des
                                self.templates_size[i] = (ex_w, ex_h)
                                if self.use_kf:
                                    cx = new_roi[0] + new_roi[2]/2.0
                                    cy = new_roi[1] + new_roi[3]/2.0
                                    meas = np.array([[np.float32(cx)], [np.float32(cy)], [np.float32(new_roi[2])], [np.float32(new_roi[3])]], dtype=np.float32)
                                    if self.kalman_filters[i] is None:
                                        self.kalman_filters[i] = create_kalman_from_roi(new_roi)
                                    else:
                                        try:
                                            self.kalman_filters[i].correct(meas)
                                        except Exception:
                                            self.kalman_filters[i] = create_kalman_from_roi(new_roi)
                                reinit_success = True
                            except Exception:
                                reinit_failed[i] = True
                        else:
                            reinit_failed[i] = True
                    else:
                        reinit_failed[i] = True

                # visualization coordinates
                if self.use_kf and (i < len(self.kalman_filters) and self.kalman_filters[i] is not None):
                    try:
                        pred = self.kalman_filters[i].predict()
                        pred_cx = float(pred[0]); pred_cy = float(pred[1])
                        pred_w = float(pred[4]); pred_h = float(pred[5])
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
                    if self.frames_since_reinit <= 15 and (ok or reinit_success):
                        try:
                            vis_x, vis_y, vis_w, vis_h = map(int, self.rois[i])
                        except Exception:
                            pass
                    cv2.rectangle(vis, (vis_x, vis_y), (vis_x + vis_w, vis_y + vis_h), self.colors[i], 2)
                    cv2.putText(vis, f"T{i+1}", (vis_x, max(12, vis_y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors[i], 2)
                    center = (vis_x + vis_w // 2, vis_y + vis_h // 2)
                    tracker_centers.append(center)
                    self.trails[i].append(center)
                    if len(self.trails[i]) > 40:
                        self.trails[i].pop(0)
                    for t in range(1, len(self.trails[i])):
                        cv2.line(vis, self.trails[i][t-1], self.trails[i][t], self.colors[i], 2)
                else:
                    cv2.rectangle(vis, (vis_x, vis_y), (vis_x + vis_w, vis_y + vis_h), (0, 255, 255), 1)
                    cv2.putText(vis, f"T{i+1} LOST", (vis_x, max(12, vis_y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                    if reinit_failed[i]:
                        cv2.putText(vis, f"T{i+1} RE-INIT FAILED", (20, 60 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # logging (same as earlier - omitted for brevity in comments)
                if measured is not None:
                    meas_cx = float(measured[0]); meas_cy = float(measured[1]); meas_w = float(measured[2]); meas_h = float(measured[3])
                    meas_x = meas_cx - meas_w / 2.0; meas_y = meas_cy - meas_h / 2.0
                else:
                    meas_x = meas_y = meas_w = meas_h = np.nan

                if self.use_kf and (i < len(self.kalman_filters) and self.kalman_filters[i] is not None):
                    state = self.kalman_filters[i].statePost.flatten()
                    smooth_cx = float(state[0]); smooth_cy = float(state[1])
                    smooth_w = float(state[4]); smooth_h = float(state[5])
                    smooth_x = smooth_cx - smooth_w / 2.0; smooth_y = smooth_cy - smooth_h / 2.0
                else:
                    smooth_x = float(vis_x); smooth_y = float(vis_y); smooth_w = float(vis_w); smooth_h = float(vis_h)

                self.csv_rows.append({
                    "frame": self.frame_idx,
                    "tracker_id": i + 1,
                    "meas_x": meas_x, "meas_y": meas_y, "meas_w": meas_w, "meas_h": meas_h,
                    "smooth_x": smooth_x, "smooth_y": smooth_y, "smooth_w": smooth_w, "smooth_h": smooth_h,
                    "ok": bool(ok),
                    "reinit_attempted": bool(reinit_attempted[i]),
                    "reinit_success": bool(reinit_success),
                    "reinit_failed": bool(reinit_failed[i])
                })

            # record swallow trail snapshot
            if self.swallow_active and tracker_centers:
                self.current_swallow_trail.append(list(tracker_centers))

            self.frame_idx += 1
            self.frames_since_reinit += 1

            elapsed = (time.time() - t0)
            fps = int(1 / elapsed) if elapsed > 0 else 0
            if self.manual_reinit_request is None:
                cv2.putText(vis, f"FPS: {fps}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                if self.video_writer:
                    self.video_writer.write(vis)
            else:
                vis = frame.copy()
                cv2.putText(vis, f"Manual reinit active", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # emit main image
            rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            h2, w2, ch = rgb.shape
            bytes_per_line = ch * w2
            qt_img = QImage(rgb.data, w2, h2, bytes_per_line, QImage.Format_RGB888).copy()
            self.change_pixmap.emit(qt_img)

            # cache for potential re-renders (e.g. trail toggle while paused)
            self._last_sec_frame = frame
            self._last_tracker_centers = list(tracker_centers)
            self._request_secondary_redraw = False
            # emit secondary based on current secondary_mode
            try:
                self.emit_secondary_image(frame, tracker_centers)
            except Exception:
                pass

            # manual reinit handling
            if self.manual_reinit_request is not None:
                idx, roi = self.manual_reinit_request
                self.manual_reinit_request = None
                self.paused = True
                self.status_msg.emit(f"Manual reinit for tracker {idx+1}")
                if roi is None:
                    # wait for UI to provide via request_manual_reinit_with_roi
                    pass
                else:
                    try:
                        new_tracker = self.create_local_tracker(frame, roi)
                        self.trackers[idx] = new_tracker
                        self.rois[idx] = roi
                        ex_roi = expand_roi(roi, self.template_pad_fraction, gray.shape[1], gray.shape[0])
                        ex_x, ex_y, ex_w, ex_h = map(int, ex_roi)
                        templ_img = frame[ex_y:ex_y+ex_h, ex_x:ex_x+ex_w]
                        templ_gray = cv2.cvtColor(templ_img, cv2.COLOR_BGR2GRAY)
                        kp, des = self.orb.detectAndCompute(templ_gray, None) if templ_gray.size>0 else ([], None)
                        self.templates_kp[idx] = kp
                        self.templates_des[idx] = des
                        self.templates_size[idx] = (ex_w, ex_h)
                        if self.use_kf:
                            self.kalman_filters[idx] = create_kalman_from_roi(roi)
                        else:
                            self.kalman_filters[idx] = None
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
            df = pd.DataFrame(self.csv_rows)
            df.to_csv(outpath, index=False)
            self.status_msg.emit(f"CSV saved to {outpath}")
        except Exception as e:
            self.status_msg.emit(f"CSV save failed: {e}")

    def set_num_trackers(self, n):
        self.num_trackers = n

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

    @Slot(int, int)
    def set_secondary_size(self, w: int, h: int):
        self.secondary_img_w = max(1, w)
        self.secondary_img_h = max(1, h)

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
            # compute excursion for mode-4 strength meter (displacement or arc length)
            if len(self.current_swallow_trail) >= 2:
                _n_tr = max(len(_fp) for _fp in self.current_swallow_trail)
                _max_exc = 0.0
                if self.strength_metric == "displacement":
                    _first_fr = self.current_swallow_trail[0]
                    for _fi in range(1, len(self.current_swallow_trail)):
                        _cur_fr = self.current_swallow_trail[_fi]
                        for _ti in range(_n_tr):
                            if _ti < len(_first_fr) and _ti < len(_cur_fr):
                                _ddx = _cur_fr[_ti][0] - _first_fr[_ti][0]
                                _ddy = _cur_fr[_ti][1] - _first_fr[_ti][1]
                                _max_exc = max(_max_exc, math.sqrt(_ddx * _ddx + _ddy * _ddy))
                else:  # arc_length
                    for _ti in range(_n_tr):
                        _tot = 0.0
                        for _fi in range(1, len(self.current_swallow_trail)):
                            _pp = self.current_swallow_trail[_fi - 1]
                            _cp = self.current_swallow_trail[_fi]
                            if _ti < len(_pp) and _ti < len(_cp):
                                _ddx = _cp[_ti][0] - _pp[_ti][0]
                                _ddy = _cp[_ti][1] - _pp[_ti][1]
                                _tot += math.sqrt(_ddx * _ddx + _ddy * _ddy)
                        _max_exc = max(_max_exc, _tot)
                self.last_swallow_excursion = _max_exc
                if self.auto_expand_strength:
                    if self.strength_metric == "displacement":
                        if _max_exc > self.strength_scale_max_displacement:
                            self.strength_scale_max_displacement = _max_exc * 1.2
                    else:
                        if _max_exc > self.strength_scale_max_arc_length:
                            self.strength_scale_max_arc_length = _max_exc * 1.2
            # compute peak speed for mode-5 speedometer
            if len(self.current_swallow_trail) >= 2:
                _peak = 0.0
                for _fi in range(1, len(self.current_swallow_trail)):
                    _prev = self.current_swallow_trail[_fi - 1]
                    _curr = self.current_swallow_trail[_fi]
                    for _ti in range(min(len(_prev), len(_curr))):
                        _dx = _curr[_ti][0] - _prev[_ti][0]
                        _dy = _curr[_ti][1] - _prev[_ti][1]
                        _peak = max(_peak, math.sqrt(_dx * _dx + _dy * _dy) * self.fps_video)
                self.last_swallow_peak_speed = _peak
                if self.auto_expand_speed and _peak > self.speed_scale_max:
                    self.speed_scale_max = _peak * 1.2
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

    @Slot(int)
    def set_circle_radius(self, value: int):
        self.circle_radius = max(1, int(value))
        self._request_secondary_redraw = True

    @Slot(bool)
    def set_show_swallow_trails(self, enabled: bool):
        self.show_swallow_trails = bool(enabled)
        self._request_secondary_redraw = True

    @Slot()
    def clear_swallow_trails(self):
        self.swallow_trails = []
        self.current_swallow_trail = []
        self._request_secondary_redraw = True

    @Slot(bool)
    def set_zoom_participant(self, enabled: bool):
        self.zoom_participant = bool(enabled)

    @Slot(int, int, int, int)
    def set_zoom_region(self, x: int, y: int, w: int, h: int):
        self.zoom_region = (max(0, x), max(0, y), max(1, w), max(1, h))

    @Slot()
    def clear_zoom_region(self):
        self.zoom_region = None

# ----------------------------
# MainWindow with slider + SecondaryWindow
# ----------------------------
class MainWindow(QWidget):
    def __init__(self, num_of_tracker=2, use_kf=True, moveExperimenterViewToSecondScreen=True):
        super().__init__()
        self.setWindowTitle("Experimenter View")
        self.resize(900, 585) #self.resize(1000, 650)
        _screens = QGuiApplication.screens()
        if moveExperimenterViewToSecondScreen and len(_screens) >= 2:
            self.move(_screens[1].geometry().topLeft())
        else:
            self.move(10, 10)

        # left image label
        self.image_label = DrawableLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.display_w = 600 #700
        self.display_h = 530 #620
        self.image_label.setFixedSize(self.display_w, self.display_h)

        # controls on right
        self.spin_num = QSpinBox(); self.spin_num.setMinimum(1); self.spin_num.setValue(num_of_tracker); self.spin_num.setMaximum(20)
        self.chk_kf = QCheckBox("Use Kalman Filter"); self.chk_kf.setChecked(bool(use_kf))
        self.btn_load = QPushButton("Screen Mirror Region (Ctrl+M)")
        self.btn_select_rois = QPushButton("Select ROIs (Ctrl+I)")
        self.btn_start = QPushButton("Start Tracking (Ctrl+T)")
        self.btn_pause = QPushButton("Pause/Resume (Ctrl+P)")
        self.combo_reinit = QComboBox(); self.combo_reinit.addItems([f"Tracker {i+1}" for i in range(num_of_tracker)])
        self.btn_reinit = QPushButton("Reinit Selected (draw) (Ctrl+R)")
        self.btn_export = QPushButton("Export CSV")
        self.btn_exit = QPushButton("Exit")

        # Slider for secondary mode (1..3)
        self.slider_label = QLabel("Mode 1: Copy")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(5)
        self.slider.setValue(1)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(1)

        # Move-box toggle (modes 2/3 only); double-click resets position
        self.chk_move_box = QCheckBox("Move Box (drag in participant view)")
        self.chk_move_box.setToolTip(
            "Check to drag the gradient box horizontally in the Participant View.\n"
            "Double-click to reset its position."
        )

        # Button to open box-width adjustment dialog
        self._box_width_pct = 8  # default ≈ 8%
        self.btn_box_width = QPushButton(f"Adjust Shaded Box Width: {self._box_width_pct}%")

        # Button to open box shading dialog (alpha range + number of shades)
        self.btn_box_shading = QPushButton("Box Shading...")

        # Toggle to mirror gradient box onto the main (experimenter) view
        self.chk_show_box_main = QCheckBox("Show box on main view")
        self.chk_show_box_main.setChecked(True)
        self.chk_show_participant_labels = QCheckBox("Show participant labels")
        self.chk_show_participant_labels.setChecked(True)
        self.chk_show_participant_labels.setToolTip(
            "Show/hide scale marks, live/peak values, swallow count,\n"
            "metric label, and LIVE badge on the participant screen\n"
            "(mode 4 and 5). Title text is always visible."
        )

        # Swallow marking controls
        self.btn_swallow = QPushButton("Mark Swallow Start (Ctrl+S)")
        self.btn_swallow.setCheckable(True)
        self.lbl_swallow_count = QLabel("Swallows: 0")
        self.lbl_swallow_count.setStyleSheet("font-weight: bold;")
        swallow_n_row = QHBoxLayout()
        swallow_n_row.addWidget(QLabel("Show last N:"))
        self.spin_swallow_n = QSpinBox()
        self.spin_swallow_n.setMinimum(1)
        self.spin_swallow_n.setMaximum(20)
        self.spin_swallow_n.setValue(3)
        swallow_n_row.addWidget(self.spin_swallow_n)
        self.chk_show_trails = QCheckBox("Show swallow trajectories")
        self.chk_show_trails.setChecked(True)
        self.btn_clear_trails = QPushButton("Clear Trajectories (Ctrl+C)")
        strength_metric_row = QHBoxLayout()
        strength_metric_row.addWidget(QLabel("Strength metric:"))
        self.combo_strength_metric = QComboBox()
        self.combo_strength_metric.addItem("Displacement", "displacement")
        self.combo_strength_metric.addItem("Arc Length", "arc_length")
        strength_metric_row.addWidget(self.combo_strength_metric)
        self.chk_zoom_participant = QCheckBox("Zoom participant view")
        self.chk_zoom_participant.setToolTip(
            "Zooms the participant view to the region from one box-width\n"
            "left of the gradient box to the right edge of the image."
        )
        zoom_region_row = QHBoxLayout()
        self.btn_set_zoom_region = QPushButton("Set Zoom Region")
        self.btn_set_zoom_region.setToolTip(
            "Draw a rectangle on the experimenter view to define the zoom region."
        )
        self.btn_reset_zoom_region = QPushButton("Reset to Auto")
        self.btn_reset_zoom_region.setToolTip("Revert to the automatic zoom region.")
        self.lbl_zoom_mode = QLabel("Auto")
        self.lbl_zoom_mode.setStyleSheet("color: gray; font-style: italic;")
        zoom_region_row.addWidget(self.btn_set_zoom_region)
        zoom_region_row.addWidget(self.btn_reset_zoom_region)
        zoom_region_row.addWidget(self.lbl_zoom_mode)

        # scale settings widgets
        self.chk_auto_expand_strength = QCheckBox("Auto-expand strength scale")
        self.chk_auto_expand_strength.setChecked(True)
        disp_scale_row = QHBoxLayout()
        disp_scale_row.addWidget(QLabel("Disp. max:"))
        self.slider_disp_scale = QSlider(Qt.Horizontal)
        self.slider_disp_scale.setRange(1, 500)
        self.slider_disp_scale.setValue(30)
        self.lbl_disp_scale_val = QLabel("30 px")
        disp_scale_row.addWidget(self.slider_disp_scale)
        disp_scale_row.addWidget(self.lbl_disp_scale_val)
        arc_scale_row = QHBoxLayout()
        arc_scale_row.addWidget(QLabel("Arc max:"))
        self.slider_arc_scale = QSlider(Qt.Horizontal)
        self.slider_arc_scale.setRange(1, 5000)
        self.slider_arc_scale.setValue(500)
        self.lbl_arc_scale_val = QLabel("500 px")
        arc_scale_row.addWidget(self.slider_arc_scale)
        arc_scale_row.addWidget(self.lbl_arc_scale_val)
        self.chk_auto_expand_speed = QCheckBox("Auto-expand speed scale")
        self.chk_auto_expand_speed.setChecked(True)
        spd_scale_row = QHBoxLayout()
        spd_scale_row.addWidget(QLabel("Speed max:"))
        self.slider_speed_scale = QSlider(Qt.Horizontal)
        self.slider_speed_scale.setRange(100, 10000)
        self.slider_speed_scale.setValue(2500)
        self.lbl_speed_scale_val = QLabel("2500 px/s")
        spd_scale_row.addWidget(self.slider_speed_scale)
        spd_scale_row.addWidget(self.lbl_speed_scale_val)

        # tracker circle diameter (modes 2 & 3)
        circle_dia_row = QHBoxLayout()
        circle_dia_row.addWidget(QLabel("Circle dia.:"))
        self.slider_circle_dia = QSlider(Qt.Horizontal)
        self.slider_circle_dia.setRange(2, 60)
        self.slider_circle_dia.setValue(12)  # diameter = 2 * default radius (6)
        self.slider_circle_dia.setTickPosition(QSlider.TicksBelow)
        self.slider_circle_dia.setTickInterval(5)
        self.lbl_circle_dia_val = QLabel("12 px")
        circle_dia_row.addWidget(self.slider_circle_dia)
        circle_dia_row.addWidget(self.lbl_circle_dia_val)

        # layout
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
        vbox.addWidget(self.chk_show_box_main)
        vbox.addWidget(self.chk_show_participant_labels)
        vbox.addLayout(circle_dia_row)
        vbox.addSpacing(6)
        vbox.addWidget(QLabel("Swallow Marking:"))
        vbox.addWidget(self.btn_swallow)
        vbox.addWidget(self.lbl_swallow_count)
        vbox.addLayout(swallow_n_row)
        vbox.addWidget(self.chk_show_trails)
        vbox.addWidget(self.btn_clear_trails)
        vbox.addLayout(strength_metric_row)
        vbox.addWidget(self.chk_zoom_participant)
        vbox.addLayout(zoom_region_row)
        vbox.addSpacing(6)
        vbox.addWidget(QLabel("Scale Settings (Mode 4 & 5):"))
        vbox.addWidget(self.chk_auto_expand_strength)
        vbox.addLayout(disp_scale_row)
        vbox.addLayout(arc_scale_row)
        vbox.addWidget(self.chk_auto_expand_speed)
        vbox.addLayout(spd_scale_row)
        vbox.addSpacing(6)
        vbox.addWidget(self.btn_pause)
        vbox.addWidget(self.btn_export)
        vbox.addWidget(self.btn_exit)
        vbox.addStretch(1)

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

        # status
        self.status_label = QLabel("")
        vbox.addWidget(self.status_label)

        # worker and secondary window
        self.worker = VideoThread()
        self.worker.use_kalman_filtering(self.chk_kf.isChecked())
        self.worker.change_pixmap.connect(self.on_frame)
        self.worker.change_secondary.connect(self.on_secondary_image)
        self.worker.status_msg.connect(self.show_status)
        self.worker.finished_processing.connect(self.on_finished)
        # secondary window initial size half of placeholder
        self.secondary = SecondaryWindow(width=self.display_w//2, height=self.display_h//2)
        self.secondary.box_offset_changed.connect(self.worker.set_box_offset)
        self.secondary.size_changed.connect(self.worker.set_secondary_size)
        self.secondary.size_changed.connect(lambda w, _h: self.image_label.set_sec_w(w))
        self.secondary.showMaximized()
        # seed worker and image_label with the initial secondary window size
        self.worker.set_secondary_size(self.secondary.width(), self.secondary.height())
        self.image_label.set_sec_w(self.secondary.width())
        # keep both views' local offsets in sync when either one drags the box
        self.image_label.box_offset_changed.connect(self.worker.set_box_offset)
        self.image_label.box_offset_changed.connect(self.secondary.sync_offset)
        self.secondary.box_offset_changed.connect(self.image_label.sync_offset)

        # manual override flag
        self.secondary_manual_override = False

        # connections
        # hotkeys
        QShortcut(QKeySequence("Ctrl+M"), self).activated.connect(self.btn_load.click)
        QShortcut(QKeySequence("Ctrl+I"), self).activated.connect(self.btn_select_rois.click)
        QShortcut(QKeySequence("Ctrl+R"), self).activated.connect(self.btn_reinit.click)
        QShortcut(QKeySequence("Ctrl+T"), self).activated.connect(self.btn_start.click)
        QShortcut(QKeySequence("Ctrl+S"), self).activated.connect(self.btn_swallow.toggle)
        QShortcut(QKeySequence("Ctrl+P"), self).activated.connect(self.btn_pause.click)
        QShortcut(QKeySequence("Ctrl+C"), self).activated.connect(self.btn_clear_trails.click)

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
        self.btn_swallow.toggled.connect(self.on_swallow_toggled)
        self.spin_swallow_n.valueChanged.connect(self.worker.set_n_swallow_display)
        self.chk_show_trails.toggled.connect(self.worker.set_show_swallow_trails)
        self.chk_show_participant_labels.toggled.connect(self.worker.set_show_participant_labels)
        self.btn_clear_trails.clicked.connect(self.worker.clear_swallow_trails)
        self.combo_strength_metric.currentIndexChanged.connect(
            lambda: self.worker.set_strength_metric(self.combo_strength_metric.currentData())
        )
        self.chk_auto_expand_strength.toggled.connect(self.worker.set_auto_expand_strength)
        self.slider_disp_scale.valueChanged.connect(self.worker.set_strength_scale_displacement)
        self.slider_disp_scale.valueChanged.connect(
            lambda v: self.lbl_disp_scale_val.setText(f"{v} px"))
        self.slider_arc_scale.valueChanged.connect(self.worker.set_strength_scale_arc_length)
        self.slider_arc_scale.valueChanged.connect(
            lambda v: self.lbl_arc_scale_val.setText(f"{v} px"))
        self.chk_auto_expand_speed.toggled.connect(self.worker.set_auto_expand_speed)
        self.slider_speed_scale.valueChanged.connect(self.worker.set_speed_scale_max)
        self.slider_speed_scale.valueChanged.connect(
            lambda v: self.lbl_speed_scale_val.setText(f"{v} px/s"))
        self.slider_circle_dia.valueChanged.connect(
            lambda v: self.worker.set_circle_radius(max(1, v // 2)))
        self.slider_circle_dia.valueChanged.connect(
            lambda v: self.lbl_circle_dia_val.setText(f"{v} px"))
        self.chk_zoom_participant.toggled.connect(self.worker.set_zoom_participant)
        self.btn_set_zoom_region.clicked.connect(self.on_set_zoom_region)
        self.btn_reset_zoom_region.clicked.connect(self.on_reset_zoom_region)
        self.worker.swallow_count_changed.connect(
            lambda n: self.lbl_swallow_count.setText(f"Swallows: {n}")
        )

        # internal
        self.last_frame = None
        self.selecting_roi = False

        # live preview of the capture region (shown before tracking starts)
        self.preview_timer = QTimer(self)
        self.preview_timer.setInterval(40)  # ~25 fps
        self.preview_timer.timeout.connect(self._update_preview)

        # ensure worker.secondary_mode starts as 1 (copy) until tracking starts
        self.worker.secondary_mode = 1
        self.slider.setValue(1)
        self.slider_label.setText("Mode 1: Copy")

    def _update_preview(self):
        # Continuously show the live capture region until tracking owns the view.
        # Skip while the worker is running (it emits processed frames itself) or
        # while a draw/ROI-selection loop is driving the display.
        if self.worker.isRunning():
            return
        if self.selecting_roi or self.image_label.draw_mode:
            return
        if self.worker.capture_region is None:
            return
        frame = self.worker.grab_capture_frame()
        if frame is not None:
            self.last_frame = frame.copy()
            self._display_frame(frame)

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
            # Show one fullscreen selector per monitor so any screen can be captured.
            selectors = []
            for s in screens:
                pix = s.grabWindow(0)
                sel = ScreenRegionSelector(pix, s.geometry())
                sel.winId()  # ensure native handle exists before setScreen
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
            self.secondary.showMaximized()  # keep participant view maximized on its screen
            self.show_status("Capture region selected. Click Select ROIs to draw.")
            frame = self.worker.grab_capture_frame()
            if frame is not None:
                self.last_frame = frame.copy()
                self._display_frame(frame)
            # start the live preview so the experimenter view shows a continuous
            # feed of the region (not just a single screenshot) before tracking
            self.preview_timer.start()
            # Auto-select mode 1 until tracking starts, unless manual override
            if not self.secondary_manual_override:
                self.worker.secondary_mode = 1
                self.slider.blockSignals(True)
                self.slider.setValue(1)
                self.slider_label.setText("Mode 1: Copy")
                self.slider.blockSignals(False)
        except Exception as e:
            self.show()
            QMessageBox.critical(self, "Error", f"Cannot capture screen region: {e}")

    def _display_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(qt_img)
        self.image_label.setPixmap(pix)

    @Slot()
    def on_select_rois_gui(self):
        self.chk_move_box.setChecked(False)  # disable box drag while drawing ROIs
        n = self.spin_num.value()
        if self.worker.capture_region is None:
            QMessageBox.warning(self, "Warning", "Select a capture region first")
            return
        # pause the worker so it stops emitting overlaid frames (boxes/trails),
        # otherwise they keep redrawing over the clean ROI-selection frame
        if self.worker.isRunning() and not self.worker.paused:
            self.worker.paused = True
        frame = self.worker.grab_capture_frame()
        if frame is None:
            QMessageBox.warning(self, "Warning", "Could not capture frame for ROI selection")
            return
        self.selecting_roi = True
        self.last_frame = frame.copy()
        # cv2.putText(self.last_frame, f"Press N for next frame", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
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
            # cv2.putText(self.last_frame, f"Press N for next frame", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            #             (0, 255, 0), 2)
            self._display_frame(self.last_frame)

            rects = self.image_label.get_rects_display()
            if len(rects) >= n:
                break
            time.sleep(0.05)
        self.image_label.exit_draw_mode()
        rects = self.image_label.get_rects_display()[:n]
        rois = []
        for r in rects:
            rois.append((int(r.x()), int(r.y()), int(r.width()), int(r.height())))
        self.worker.set_num_trackers(n)
        self.worker.init_trackers_from_rois(self.last_frame, rois)
        self.combo_reinit.clear()
        self.combo_reinit.addItems([f"Tracker {i+1}" for i in range(n)])
        self.show_status("ROIs set and trackers initialized (GUI).")
        self.image_label.clear_rects()
        self.selecting_roi = False
        # After selecting ROIs, still keep secondary mode auto behavior until user toggles slider.
        self.on_start_tracking()

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_N and self.selecting_roi:
            frame = self.worker.grab_capture_frame()
            if frame is None:
                QMessageBox.warning(self, "Warning", "Could not capture frame for ROI selection")
                return
            self.last_frame = frame.copy()
            cv2.putText(self.last_frame, f"Press N for next frame", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            self._display_frame(self.last_frame)

    @Slot()
    def on_start_tracking(self):
        if self.worker.capture_region is None:
            QMessageBox.warning(self, "Warning", "Select a capture region first")
            return
        if len(self.worker.trackers) < self.worker.num_trackers:
            QMessageBox.warning(self, "Warning", "Select ROIs first")
            return
        self.worker.use_kalman_filtering(self.chk_kf.isChecked())
        # If user has not manually changed slider, auto-switch to mode 3 at start
        if not self.secondary_manual_override:
            self.worker.secondary_mode = 3
            self.slider.blockSignals(True)
            self.slider.setValue(3)
            self.slider_label.setText("Mode 3: Frame+Box")
            self.slider.blockSignals(False)
        if not self.worker.isRunning():
            self.worker.start()
            self.show_status("Processing started")
        else:
            self.worker.paused = False
            self.show_status("Resumed processing")

    @Slot()
    def on_pause(self):
        if not self.worker.isRunning():
            return
        self.worker.pause_toggle()

    @Slot()
    def on_manual_reinit_gui(self):
        self.chk_move_box.setChecked(False)  # disable box drag while drawing reinit ROI
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
            rects = self.image_label.get_rects_display()
            if len(rects) >= 1:
                break
            time.sleep(0.05)
        self.image_label.exit_draw_mode()
        r = rects[0]
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
        if not fname:
            return
        self.worker.save_csv(fname)

    @Slot(QImage)
    def on_frame(self, qt_img):
        pix = QPixmap.fromImage(qt_img)
        pix = pix.scaled(self.image_label.size(), Qt.KeepAspectRatio)
        disp_w = pix.width()
        disp_h = pix.height()
        img_w = max(1, qt_img.width())
        img_h = max(1, qt_img.height())

        need_painter = (
            (self.chk_show_box_main.isChecked() and int(self.worker.secondary_mode) in (2, 3))
            or self.chk_show_trails.isChecked()
        )
        if need_painter:
            painter = QPainter(pix)

            if self.chk_show_box_main.isChecked() and int(self.worker.secondary_mode) in (2, 3):
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

            if self.chk_show_trails.isChecked():
                _trail_palette = [
                    QColor(255, 140, 0), QColor(255, 0, 255), QColor(0, 255, 255),
                    QColor(255, 215, 0), QColor(255, 100, 0), QColor(100, 255, 50),
                    QColor(180, 0, 180), QColor(0, 200, 150), QColor(255, 50, 50),
                    QColor(50, 150, 255),
                ]
                painter.setOpacity(1.0)
                n_disp = self.worker.n_swallow_display
                trails_snap = self.worker.swallow_trails[-n_disp:]
                for _si, _trail in enumerate(trails_snap):
                    col = _trail_palette[_si % len(_trail_palette)]
                    pen = QPen(col, 2)
                    painter.setPen(pen)
                    for _fi in range(1, len(_trail)):
                        _prev, _curr = _trail[_fi - 1], _trail[_fi]
                        for _ti in range(min(len(_prev), len(_curr))):
                            x1 = int(_prev[_ti][0] * disp_w / img_w)
                            y1 = int(_prev[_ti][1] * disp_h / img_h)
                            x2 = int(_curr[_ti][0] * disp_w / img_w)
                            y2 = int(_curr[_ti][1] * disp_h / img_h)
                            painter.drawLine(x1, y1, x2, y2)
                if self.worker.swallow_active and len(self.worker.current_swallow_trail) > 1:
                    pen = QPen(QColor(255, 255, 255), 2)
                    painter.setPen(pen)
                    trail = self.worker.current_swallow_trail
                    for _fi in range(1, len(trail)):
                        _prev, _curr = trail[_fi - 1], trail[_fi]
                        for _ti in range(min(len(_prev), len(_curr))):
                            x1 = int(_prev[_ti][0] * disp_w / img_w)
                            y1 = int(_prev[_ti][1] * disp_h / img_h)
                            x2 = int(_curr[_ti][0] * disp_w / img_w)
                            y2 = int(_curr[_ti][1] * disp_h / img_h)
                            painter.drawLine(x1, y1, x2, y2)

            painter.end()

        self.image_label.setPixmap(pix)

    @Slot(QImage)
    def on_secondary_image(self, qt_img):
        # show only if secondary window exists
        if self.secondary:
            self.secondary.update_image(qt_img)

    @Slot(str)
    def show_status(self, msg):
        self.status_label.setText(msg)

    @Slot()
    def on_finished(self):
        #QMessageBox.information(self, "Finished", "Processing finished.")
        # keep windows open
        self.close()

    @Slot(int)
    def on_num_changed(self, val):
        self.combo_reinit.clear()
        self.combo_reinit.addItems([f"Tracker {i+1}" for i in range(val)])
        self.worker.set_num_trackers(val)

    @Slot(int)
    def on_kf_toggled(self, state):
        enabled = True if state == Qt.Checked else False
        self.worker.use_kalman_filtering(enabled)

    @Slot(int)
    def on_slider_changed(self, v):
        # user moved slider -> manual override enabled
        self.secondary_manual_override = True
        v = int(v)
        _names = {1: "Copy", 2: "Black+Box", 3: "Frame+Box", 4: "Strength Meter", 5: "Speedometer"}
        self.slider_label.setText(f"Mode {v}: {_names.get(v, '')}")
        self.worker.secondary_mode = v
        self.worker.secondary_manual_override = True
        # update secondary window mode
        if self.secondary:
            self.secondary.set_mode(v)

    @Slot(bool)
    def on_move_box_toggled(self, checked: bool):
        if self.secondary:
            self.secondary.set_drag_mode(checked)
        self.image_label.set_box_drag_mode(checked)

    @Slot(bool)
    def on_swallow_toggled(self, checked: bool):
        if checked:
            self.worker.start_swallow()
            self.btn_swallow.setText("Mark Swallow End (Ctrl+S)")
            self.btn_swallow.setStyleSheet("background-color: #cc2222; color: white; font-weight: bold;")
        else:
            self.worker.end_swallow()
            self.btn_swallow.setText("Mark Swallow Start (Ctrl+S)")
            self.btn_swallow.setStyleSheet("")

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
        # Map from experimenter-view (frame) space to secondary-image space
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
        _prev_pct = self._box_width_pct

        dlg = QDialog(self)
        dlg.setWindowTitle("Adjust Shaded Box Width")
        dlg.setMinimumWidth(300)

        lbl = QLabel(f"Width: {_prev_pct}% of view")
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(1)
        slider.setMaximum(50)
        slider.setValue(_prev_pct)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(5)
        slider.valueChanged.connect(lambda v: (
            lbl.setText(f"Width: {v}% of view"),
            self.worker.set_box_width_fraction(v / 100.0)
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
            self.worker.set_box_width_fraction(_prev_pct / 100.0)

    @Slot()
    def on_box_shading_clicked(self):
        # snapshot current values so Cancel can restore them
        _prev_amax = self.worker.box_alpha_max
        _prev_amin = self.worker.box_alpha_min
        _prev_shades = self.worker.box_num_shades

        dlg = QDialog(self)
        dlg.setWindowTitle("Box Shading")
        dlg.setMinimumWidth(320)

        # --- Alpha max (0–100 integer = 0.0–1.0) ---
        lbl_amax = QLabel(f"Max opacity (left edge): {int(_prev_amax * 100)}%")
        sl_amax = QSlider(Qt.Horizontal)
        sl_amax.setMinimum(0); sl_amax.setMaximum(100)
        sl_amax.setValue(int(_prev_amax * 100))
        sl_amax.setTickPosition(QSlider.TicksBelow); sl_amax.setTickInterval(10)
        sl_amax.valueChanged.connect(lambda v: (
            lbl_amax.setText(f"Max opacity (left edge): {v}%"),
            self.worker.set_box_alpha_max(v / 100.0)
        ))

        # --- Alpha min (0–100 integer = 0.0–1.0) ---
        lbl_amin = QLabel(f"Min opacity (right edge): {int(_prev_amin * 100)}%")
        sl_amin = QSlider(Qt.Horizontal)
        sl_amin.setMinimum(0); sl_amin.setMaximum(100)
        sl_amin.setValue(int(_prev_amin * 100))
        sl_amin.setTickPosition(QSlider.TicksBelow); sl_amin.setTickInterval(10)
        sl_amin.valueChanged.connect(lambda v: (
            lbl_amin.setText(f"Min opacity (right edge): {v}%"),
            self.worker.set_box_alpha_min(v / 100.0)
        ))

        # --- Number of shades (5–50) ---
        lbl_shades = QLabel(f"Number of shades: {_prev_shades}")
        sl_shades = QSlider(Qt.Horizontal)
        sl_shades.setMinimum(5); sl_shades.setMaximum(50)
        sl_shades.setValue(_prev_shades)
        sl_shades.setTickPosition(QSlider.TicksBelow); sl_shades.setTickInterval(5)
        sl_shades.valueChanged.connect(lambda v: (
            lbl_shades.setText(f"Number of shades: {v}"),
            self.worker.set_box_num_shades(v)
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
            # restore previous values on Cancel
            self.worker.set_box_alpha_max(_prev_amax)
            self.worker.set_box_alpha_min(_prev_amin)
            self.worker.set_box_num_shades(_prev_shades)

    def closeEvent(self, event):
        if self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(1000)
        if self.secondary:
            self.secondary.close()
        event.accept()

# ----------------------------
# Run application
# ----------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    moveExperimenterViewToSecondScreen = True
    w = MainWindow(num_of_tracker=1, use_kf=False, moveExperimenterViewToSecondScreen=moveExperimenterViewToSecondScreen)
    w.show()
    sys.exit(app.exec())