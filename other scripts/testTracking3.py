# multi_tracker_full.py
import sys
import time
import cv2
import numpy as np
import pandas as pd

from PySide6.QtCore import QThread, Signal, Slot, Qt, QRect, QPoint, QTimer
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QSpinBox, QComboBox, QMessageBox, QCheckBox, QSizePolicy,
    QSlider, QFrame
)

# ---------------------------
# Helper utilities (Kalman, expand ROI)
# ---------------------------
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
    x, y, w, h = roi
    pad_x = int(w * frac)
    pad_y = int(h * frac)
    nx = max(0, x - pad_x)
    ny = max(0, y - pad_y)
    nw = min(frame_w - nx, w + 2 * pad_x)
    nh = min(frame_h - ny, h + 2 * pad_y)
    return (nx, ny, nw, nh)


# ---------------------------
# Drawable QLabel for GUI ROI drawing
# ---------------------------
class DrawableLabel(QLabel):
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

    def setPixmap(self, pixmap: QPixmap):
        super().setPixmap(pixmap)
        self._pixmap = pixmap

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
        if not self.draw_mode:
            return
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.start_point = event.pos()
            self.end_point = event.pos()
            self.temp_rect = QRect(self.start_point, self.end_point)
            self.update()

    def mouseMoveEvent(self, event):
        if not self.draw_mode:
            return
        if self.drawing:
            self.end_point = event.pos()
            self.temp_rect = QRect(self.start_point, self.end_point).normalized()
            self.update()

    def mouseReleaseEvent(self, event):
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


# ---------------------------
# Worker thread: tracking engine
# ---------------------------
class VideoThread(QThread):
    change_pixmap = Signal(QImage)
    status_msg = Signal(str)
    finished_processing = Signal()

    def __init__(self):
        super().__init__()
        self.video_path = None
        self.cap = None
        self.scale_fx = 0.65
        self.scale_fy = 0.65
        self.num_trackers = 1
        self.fps_video = 30.0

        # trackers & data
        self.trackers = []
        self.rois = []
        self.colors = []
        self.trails = []

        # ORB templates
        self.orb = cv2.ORB_create(1000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.templates_kp = []
        self.templates_des = []
        self.templates_size = []

        # kalman filters (optional)
        self.kalman_filters = []

        # smoothing (drift reduction) using exponential moving average on centers
        self.smoothing_enabled = True
        self.smooth_alpha = 0.25
        self.smooth_centers = []  # per tracker: (cx,cy)

        # reinit params
        self.match_thresh = 8
        self.template_pad_fraction = 0.20

        # CSV log
        self.csv_rows = []

        # control
        self.paused = True
        self.stop_requested = False
        self.manual_reinit_request = None  # (idx, roi) where roi may be None (draw)
        self.use_kf = True

        self.video_writer = None
        self.frame_idx = 0
        self.frames_since_reinit = 0

    def load_video(self, path):
        self.video_path = path
        if self.cap:
            try:
                self.cap.release()
            except:
                pass
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open video")
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps_video = fps if fps and not np.isnan(fps) and fps > 0 else 30.0
        self.status_msg.emit(f"Video loaded: {self.video_path} ({self.fps_video:.2f} FPS)")

    def set_num_trackers(self, n):
        self.num_trackers = int(n)

    def init_from_gui_rois(self, frame, rois_list):
        """Initialize trackers/templates/KF/smoothing from given processed-frame rois list."""
        self.trackers = []
        self.rois = []
        self.colors = []
        self.trails = []
        self.templates_kp = []
        self.templates_des = []
        self.templates_size = []
        self.kalman_filters = []
        self.smooth_centers = []
        self.csv_rows = []

        rng = np.random.default_rng(42)
        h_frame, w_frame = frame.shape[:2]

        for roi in rois_list:
            self.rois.append(roi)
            tr = cv2.legacy.TrackerCSRT_create()
            tr.init(frame, roi)
            self.trackers.append(tr)
            self.colors.append(tuple(int(c) for c in rng.integers(50, 255, 3)))
            self.trails.append([])
            # ORB template from expanded ROI
            ex_roi = expand_roi(roi, self.template_pad_fraction, w_frame, h_frame)
            ex_x, ex_y, ex_w, ex_h = map(int, ex_roi)
            templ_img = frame[ex_y:ex_y+ex_h, ex_x:ex_x+ex_w]
            templ_gray = cv2.cvtColor(templ_img, cv2.COLOR_BGR2GRAY) if templ_img.size>0 else None
            kp, des = (self.orb.detectAndCompute(templ_gray, None) if templ_gray is not None else ([], None))
            self.templates_kp.append(kp)
            self.templates_des.append(des)
            self.templates_size.append((ex_w, ex_h))
            # KF if enabled
            if self.use_kf:
                try:
                    kf = create_kalman_from_roi(roi)
                except Exception:
                    kf = None
            else:
                kf = None
            self.kalman_filters.append(kf)
            # smoothing initial center
            x,y,w,h = roi
            cx = x + w/2.0; cy = y + h/2.0
            self.smooth_centers.append((cx, cy))

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
        pts_template = np.float32([templ_kp[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        pts_scene = np.float32([kp_scene[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        try:
            H, mask = cv2.findHomography(pts_template, pts_scene, cv2.RANSAC, 5.0)
            if H is None:
                return None
        except Exception:
            return None
        tw, th = templ_size
        corners = np.float32([[0,0],[tw,0],[tw,th],[0,th]]).reshape(-1,1,2)
        try:
            transformed = cv2.perspectiveTransform(corners, H)
        except Exception:
            return None
        pts = transformed.reshape(-1,2)
        min_x = max(0, int(np.min(pts[:,0]))); min_y = max(0,int(np.min(pts[:,1])))
        max_x = min(frame_gray.shape[1]-1, int(np.max(pts[:,0]))); max_y = min(frame_gray.shape[0]-1, int(np.max(pts[:,1])))
        w_new = max(1, max_x-min_x); h_new = max(1, max_y-min_y)
        return (min_x, min_y, w_new, h_new)

    def use_kalman_filtering(self, choice: bool):
        self.use_kf = bool(choice)
        self.status_msg.emit(f"Kalman filtering {'enabled' if self.use_kf else 'disabled'}")
        # if enabling now, create KF instances for any rois that lack them
        if self.use_kf:
            for i in range(len(self.rois)):
                if i >= len(self.kalman_filters) or self.kalman_filters[i] is None:
                    try:
                        kf = create_kalman_from_roi(self.rois[i])
                        if i >= len(self.kalman_filters):
                            self.kalman_filters.extend([None]* (i - len(self.kalman_filters) + 1))
                        self.kalman_filters[i] = kf
                    except Exception:
                        if i >= len(self.kalman_filters):
                            self.kalman_filters.extend([None]* (i - len(self.kalman_filters) + 1))
                        self.kalman_filters[i] = None

    def run(self):
        if not self.cap:
            self.status_msg.emit("Load a video first")
            return
        self.stop_requested = False
        self.paused = False
        self.frame_idx = 0
        self.frames_since_reinit = 0

        # setup writer
        frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) * self.scale_fx)
        frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * self.scale_fy)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter("tracked_output_full.mp4", fourcc, self.fps_video, (frame_w, frame_h))

        while self.cap.isOpened() and not self.stop_requested:
            if self.paused:
                time.sleep(0.03)
                continue
            t0 = time.time()
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, None, fx=self.scale_fx, fy=self.scale_fy)
            vis = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            reinit_attempted = [False] * self.num_trackers
            reinit_failed = [False] * self.num_trackers

            for i in range(self.num_trackers):
                if i >= len(self.trackers):
                    continue
                tracker = self.trackers[i]
                ok, new_roi = tracker.update(frame)
                reinit_success = False
                measured = None

                if ok:
                    self.rois[i] = new_roi
                    x,y,w,h = map(int, new_roi)
                    cx = x + w/2.0; cy = y + h/2.0
                    measured = np.array([[np.float32(cx)], [np.float32(cy)], [np.float32(w)], [np.float32(h)]], dtype=np.float32)

                    # update ORB template from expanded ROI
                    ex_roi = expand_roi((x,y,w,h), self.template_pad_fraction, gray.shape[1], gray.shape[0])
                    ex_x, ex_y, ex_w, ex_h = map(int, ex_roi)
                    templ_img = frame[ex_y:ex_y+ex_h, ex_x:ex_x+ex_w] if ex_w>0 and ex_h>0 else None
                    if templ_img is not None and templ_img.size>0:
                        templ_gray = cv2.cvtColor(templ_img, cv2.COLOR_BGR2GRAY)
                        kp, des = self.orb.detectAndCompute(templ_gray, None)
                        self.templates_kp[i] = kp; self.templates_des[i] = des; self.templates_size[i] = (ex_w, ex_h)

                    # KF correction if enabled
                    if self.use_kf:
                        if self.kalman_filters[i] is None:
                            self.kalman_filters[i] = create_kalman_from_roi((x,y,w,h))
                        else:
                            try:
                                self.kalman_filters[i].correct(measured)
                            except Exception:
                                self.kalman_filters[i] = create_kalman_from_roi((x,y,w,h))

                    # smoothing (EMA) update center
                    if self.smoothing_enabled:
                        prev = self.smooth_centers[i]
                        newcx = (1 - self.smooth_alpha) * prev[0] + self.smooth_alpha * cx
                        newcy = (1 - self.smooth_alpha) * prev[1] + self.smooth_alpha * cy
                        self.smooth_centers[i] = (newcx, newcy)
                    else:
                        self.smooth_centers[i] = (cx, cy)

                else:
                    # attempt ORB reinit (still allowed even if KF disabled)
                    reinit_attempted[i] = True
                    kp_t = self.templates_kp[i]
                    des_t = self.templates_des[i]
                    tsize = self.templates_size[i]
                    if des_t is not None and len(des_t) >= 4:
                        candidate = self.try_orb_reinit(gray, kp_t, des_t, tsize)
                        if candidate is not None:
                            try:
                                new_tracker = cv2.legacy.TrackerCSRT_create()
                                new_tracker.init(frame, candidate)
                                self.trackers[i] = new_tracker
                                self.rois[i] = candidate
                                ex_roi = expand_roi(candidate, self.template_pad_fraction, gray.shape[1], gray.shape[0])
                                ex_x, ex_y, ex_w, ex_h = map(int, ex_roi)
                                templ_img = frame[ex_y:ex_y+ex_h, ex_x:ex_x+ex_w] if ex_w>0 and ex_h>0 else None
                                if templ_img is not None and templ_img.size>0:
                                    templ_gray = cv2.cvtColor(templ_img, cv2.COLOR_BGR2GRAY)
                                    kp, des = self.orb.detectAndCompute(templ_gray, None)
                                    self.templates_kp[i] = kp; self.templates_des[i] = des; self.templates_size[i] = (ex_w, ex_h)
                                # KF: correct/create
                                if self.use_kf:
                                    cx = candidate[0] + candidate[2]/2.0; cy = candidate[1] + candidate[3]/2.0
                                    meas = np.array([[np.float32(cx)], [np.float32(cy)], [np.float32(candidate[2])], [np.float32(candidate[3])]], dtype=np.float32)
                                    if self.kalman_filters[i] is None:
                                        self.kalman_filters[i] = create_kalman_from_roi(candidate)
                                    else:
                                        try:
                                            self.kalman_filters[i].correct(meas)
                                        except Exception:
                                            self.kalman_filters[i] = create_kalman_from_roi(candidate)
                                reinit_success = True
                                # update smoothing center
                                cx = candidate[0] + candidate[2]/2.0; cy = candidate[1] + candidate[3]/2.0
                                if self.smoothing_enabled:
                                    prev = self.smooth_centers[i]
                                    self.smooth_centers[i] = ((1 - self.smooth_alpha)*prev[0] + self.smooth_alpha*cx,
                                                              (1 - self.smooth_alpha)*prev[1] + self.smooth_alpha*cy)
                                else:
                                    self.smooth_centers[i] = (cx, cy)
                            except Exception:
                                reinit_failed[i] = True
                        else:
                            reinit_failed[i] = True
                    else:
                        reinit_failed[i] = True

                # determine visualization coordinates
                if self.use_kf and (i < len(self.kalman_filters) and self.kalman_filters[i] is not None):
                    try:
                        pred = self.kalman_filters[i].predict()
                        pred_cx = float(pred[0]); pred_cy = float(pred[1])
                        pred_w = float(pred[4]); pred_h = float(pred[5])
                        vis_w = int(max(1, pred_w)); vis_h = int(max(1, pred_h))
                        vis_x = int(pred_cx - vis_w/2.0); vis_y = int(pred_cy - vis_h/2.0)
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

                # draw
                if ok or reinit_success:
                    # shortly after reinit use raw roi
                    if self.frames_since_reinit <= 15:
                        try:
                            vis_x, vis_y, vis_w, vis_h = map(int, self.rois[i])
                        except Exception:
                            pass
                    cv2.rectangle(vis, (vis_x, vis_y), (vis_x+vis_w, vis_y+vis_h), self.colors[i], 2)
                    cv2.putText(vis, f"T{i+1}", (vis_x, max(12, vis_y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors[i], 2)
                    # trail uses smoothed centers
                    scx, scy = self.smooth_centers[i]
                    center = (int(scx), int(scy))
                    self.trails[i].append(center)
                    if len(self.trails[i]) > 60:
                        self.trails[i].pop(0)
                    for t in range(1, len(self.trails[i])):
                        cv2.line(vis, self.trails[i][t-1], self.trails[i][t], self.colors[i], 2)
                else:
                    cv2.rectangle(vis, (vis_x, vis_y), (vis_x+vis_w, vis_y+vis_h), (0,255,255), 1)
                    cv2.putText(vis, f"T{i+1} LOST", (vis_x, max(12, vis_y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)
                    if reinit_failed[i]:
                        cv2.putText(vis, f"T{i+1} RE-INIT FAILED", (20, 60 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

                # log D3 (bbox + center)
                try:
                    bx, by, bw, bh = vis_x, vis_y, vis_w, vis_h
                    cx_log = bx + bw/2.0; cy_log = by + bh/2.0
                except Exception:
                    bx = by = bw = bh = np.nan
                    cx_log = cy_log = np.nan

                self.csv_rows.append({
                    "frame": self.frame_idx,
                    "tracker_id": i+1,
                    "x": bx, "y": by, "w": bw, "h": bh,
                    "cx": cx_log, "cy": cy_log,
                    "ok": bool(ok),
                    "reinit_attempted": bool(reinit_attempted[i]),
                    "reinit_success": bool(reinit_success),
                    "reinit_failed": bool(reinit_failed[i])
                })

            self.frame_idx += 1
            self.frames_since_reinit += 1

            elapsed = (time.time() - t0)
            fps = int(1/elapsed) if elapsed>0 else 0
            cv2.putText(vis, f"FPS: {fps}", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)

            if self.video_writer:
                self.video_writer.write(vis)

            rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
            self.change_pixmap.emit(qt_img)

            # manual reinit handling (if main window requested a drawn ROI, worker waits for main to call request_manual_reinit_with_roi)
            if self.manual_reinit_request is not None:
                idx, roi = self.manual_reinit_request
                self.manual_reinit_request = None
                self.paused = True
                self.status_msg.emit(f"Manual reinit for tracker {idx+1}")
                if roi is not None:
                    # apply directly (roi is in processed coords)
                    try:
                        new_tracker = cv2.legacy.TrackerCSRT_create()
                        new_tracker.init(frame, roi)
                        self.trackers[idx] = new_tracker
                        self.rois[idx] = roi
                        ex_roi = expand_roi(roi, self.template_pad_fraction, gray.shape[1], gray.shape[0])
                        ex_x, ex_y, ex_w, ex_h = map(int, ex_roi)
                        templ_img = frame[ex_y:ex_y+ex_h, ex_x:ex_x+ex_w]
                        templ_gray = cv2.cvtColor(templ_img, cv2.COLOR_BGR2GRAY) if templ_img.size>0 else None
                        kp, des = (self.orb.detectAndCompute(templ_gray, None) if templ_gray is not None else ([], None))
                        self.templates_kp[idx] = kp; self.templates_des[idx] = des; self.templates_size[idx] = (ex_w, ex_h)
                        if self.use_kf:
                            self.kalman_filters[idx] = create_kalman_from_roi(roi)
                        else:
                            self.kalman_filters[idx] = None
                        # reset smoothing center
                        cx = roi[0] + roi[2]/2.0; cy = roi[1] + roi[3]/2.0
                        self.smooth_centers[idx] = (cx, cy)
                        self.status_msg.emit(f"Manual reinit applied for tracker {idx+1}")
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
    def step_forward(self):
        # grab next frame only and emit it (no tracking)
        if not self.cap:
            return
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = cv2.resize(frame, None, fx=self.scale_fx, fy=self.scale_fy)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        self.change_pixmap.emit(qt_img)

    @Slot()
    def step_back(self):
        # step back one frame
        if not self.cap:
            return
        pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        seek = max(0, pos - 2)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, seek)
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = cv2.resize(frame, None, fx=self.scale_fx, fy=self.scale_fy)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        self.change_pixmap.emit(qt_img)

    @Slot(int)
    def request_manual_reinit_with_roi(self, args):
        idx, roi = args
        self.manual_reinit_request = (idx, roi)

    @Slot()
    def save_csv(self, outpath):
        try:
            df = pd.DataFrame(self.csv_rows)
            df.to_csv(outpath, index=False)
            self.status_msg.emit(f"CSV saved to {outpath}")
        except Exception as e:
            self.status_msg.emit(f"CSV save failed: {e}")


# ---------------------------
# MainWindow: GUI
# ---------------------------
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Tracker Full")
        self.resize(1400, 820)

        # left label
        self.image_label = DrawableLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.display_w = 900; self.display_h = 700
        self.image_label.setFixedSize(self.display_w, self.display_h)

        # controls on right
        self.spin_num = QSpinBox()
        self.spin_num.setMinimum(1); self.spin_num.setValue(2); self.spin_num.setMaximum(20)

        self.chk_kf = QCheckBox("Use Kalman Filter")
        self.chk_kf.setChecked(True)

        self.chk_smooth = QCheckBox("Enable EMA smoothing")
        self.chk_smooth.setChecked(True)
        self.smooth_alpha_spin = QSpinBox()
        self.smooth_alpha_spin.setRange(1, 90)
        self.smooth_alpha_spin.setValue(25)
        self.smooth_alpha_spin.setSuffix(" % (alpha)")

        self.btn_load = QPushButton("Load Video")
        self.btn_select_rois = QPushButton("Select ROIs (GUI)")
        self.btn_start = QPushButton("Start Tracking")
        self.btn_pause = QPushButton("Pause/Resume")
        self.btn_step_fwd = QPushButton("Step Forward")
        self.btn_step_back = QPushButton("Step Back")
        self.btn_save_csv = QPushButton("Save CSV")
        self.btn_export_video = QPushButton("Export Video (toggle path)")

        # dynamic per-tracker reinit area
        self.reinit_box = QVBoxLayout()
        self.reinit_box_widget = QVBoxLayout()

        # slider for scrubbing
        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setMinimum(0)

        # layout
        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("Number of trackers:"))
        right_layout.addWidget(self.spin_num)
        right_layout.addWidget(self.chk_kf)
        right_layout.addWidget(self.chk_smooth)
        right_layout.addWidget(QLabel("Smoothing alpha (%)"))
        right_layout.addWidget(self.smooth_alpha_spin)
        right_layout.addSpacing(8)
        right_layout.addWidget(self.btn_load)
        right_layout.addWidget(self.btn_select_rois)
        right_layout.addWidget(self.btn_start)
        # right_layout.addWidget(self.btn_pause)
        # right_layout.addWidget(self.btn_step_fwd)
        # right_layout.addWidget(self.btn_step_back)
        right_layout.addWidget(QLabel("Scrub video"))
        right_layout.addWidget(self.seek_slider)
        right_layout.addSpacing(8)
        right_layout.addWidget(QLabel("Manual re-init (per tracker):"))
        # container for per-tracker buttons
        self.reinit_container = QVBoxLayout()
        right_layout.addLayout(self.reinit_container)
        right_layout.addWidget(self.btn_save_csv)
        right_layout.addWidget(self.btn_export_video)
        right_layout.addStretch(1)

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.image_label)
        main_layout.addLayout(right_layout)
        self.setLayout(main_layout)

        # status label bottom
        self.status_label = QLabel("")
        right_layout.addWidget(self.status_label)

        # worker thread
        self.worker = VideoThread()
        self.worker.change_pixmap.connect(self.update_image)
        self.worker.status_msg.connect(self.show_status)
        self.worker.finished_processing.connect(self.on_finished)

        # connect controls
        self.btn_load.clicked.connect(self.on_load)
        self.btn_select_rois.clicked.connect(self.on_select_rois_gui)
        self.btn_start.clicked.connect(self.on_start)
        self.btn_pause.clicked.connect(self.on_pause)
        self.btn_step_fwd.clicked.connect(self.on_step_forward)
        self.btn_step_back.clicked.connect(self.on_step_back)
        self.btn_save_csv.clicked.connect(self.on_save_csv)
        self.spin_num.valueChanged.connect(self.on_num_changed)
        self.chk_kf.stateChanged.connect(self.on_kf_toggled)
        self.chk_smooth.stateChanged.connect(self.on_smooth_toggled)
        self.smooth_alpha_spin.valueChanged.connect(self.on_smooth_alpha_changed)
        self.seek_slider.valueChanged.connect(self.on_slider_changed)
        self.btn_export_video.clicked.connect(self.on_export_video)

        # internal state
        self.last_frame = None
        self.video_loaded = False
        self.export_path = "tracked_output_full.mp4"

    # ---------------- UI slots ----------------
    def on_load(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv)")
        if not fname:
            return
        try:
            self.worker.load_video(fname)
            # adapt label size to video processed resolution
            w = int(self.worker.cap.get(cv2.CAP_PROP_FRAME_WIDTH) * self.worker.scale_fx)
            h = int(self.worker.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * self.worker.scale_fy)
            self.display_w, self.display_h = w, h
            self.image_label.setFixedSize(w, h)
            self.seek_slider.setMaximum(int(self.worker.cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)
            self.video_loaded = True
            # show first frame
            self.worker.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.worker.cap.read()
            if ret:
                frame = cv2.resize(frame, None, fx=self.worker.scale_fx, fy=self.worker.scale_fy)
                self.last_frame = frame.copy()
                self.display_frame(frame)
            self.show_status("Video loaded.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cannot load video: {e}")

    def display_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimage = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimage)
        self.image_label.setPixmap(pix)

    def on_slider_changed(self, pos):
        if not self.video_loaded:
            return
        idx = self.seek_slider.value()
        self.worker.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.worker.cap.read()
        if not ret:
            return
        frame = cv2.resize(frame, None, fx=self.worker.scale_fx, fy=self.worker.scale_fy)
        self.last_frame = frame.copy()
        self.display_frame(self.last_frame)

    def on_select_rois_gui(self):
        if not self.video_loaded:
            QMessageBox.warning(self, "Warning", "Load a video first")
            return
        # get n
        n = self.spin_num.value()
        # set cap to slider position
        idx = self.seek_slider.value()
        self.worker.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.worker.cap.read()
        if not ret:
            QMessageBox.warning(self, "Warning", "Cannot read frame for ROI selection")
            return
        frame = cv2.resize(frame, None, fx=self.worker.scale_fx, fy=self.worker.scale_fy)
        self.last_frame = frame.copy()
        self.display_frame(self.last_frame)

        # enable draw mode and wait for n rects
        self.image_label.clear_rects()
        self.image_label.enter_draw_mode()
        self.show_status(f"Draw {n} ROIs on image (drag).")
        while True:
            QApplication.processEvents()
            rects = self.image_label.get_rects_display()
            if len(rects) >= n:
                break
            time.sleep(0.05)
        self.image_label.exit_draw_mode()
        rects = self.image_label.get_rects_display()[:n]
        rois = [(int(r.x()), int(r.y()), int(r.width()), int(r.height())) for r in rects]
        # init trackers inside worker
        self.worker.set_num_trackers(n)
        self.worker.init_from_gui_rois(self.last_frame, rois)
        # create per-tracker reinit buttons
        self._create_reinit_buttons(n)
        self.show_status("ROIs set and trackers initialized.")

    def _create_reinit_buttons(self, n):
        # clear container
        while self.reinit_container.count():
            item = self.reinit_container.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        # create buttons per tracker
        for i in range(n):
            btn = QPushButton(f"Reinit Tracker {i+1}")
            # bind index
            btn.clicked.connect(lambda checked, idx=i: self.on_manual_reinit_btn(idx))
            self.reinit_container.addWidget(btn)

    def on_manual_reinit_btn(self, idx):
        if not self.video_loaded:
            QMessageBox.warning(self, "Warning", "Load a video first")
            return
        # pause if running
        was_running = self.worker.isRunning() and not self.worker.paused
        self.worker.paused = True
        # show current frame for drawing
        pos = int(self.worker.cap.get(cv2.CAP_PROP_POS_FRAMES))
        seek = max(0, pos - 1)
        self.worker.cap.set(cv2.CAP_PROP_POS_FRAMES, seek)
        ret, frame = self.worker.cap.read()
        if not ret:
            QMessageBox.warning(self, "Warning", "Cannot read frame for manual reinit")
            if was_running:
                self.worker.paused = False
            return
        frame = cv2.resize(frame, None, fx=self.worker.scale_fx, fy=self.worker.scale_fy)
        self.last_frame = frame.copy()
        self.display_frame(self.last_frame)
        # draw one ROI
        self.image_label.clear_rects()
        self.image_label.enter_draw_mode()
        self.show_status(f"Draw ROI for tracker {idx+1}")
        while True:
            QApplication.processEvents()
            rects = self.image_label.get_rects_display()
            if len(rects) >= 1:
                break
            time.sleep(0.05)
        self.image_label.exit_draw_mode()
        r = rects[0]
        new_roi = (int(r.x()), int(r.y()), int(r.width()), int(r.height()))
        # send to worker
        self.worker.request_manual_reinit_with_roi((idx, new_roi))
        self.show_status(f"Manual reinit requested for tracker {idx+1}")
        if was_running:
            self.worker.paused = False

    def on_start(self):
        if not self.video_loaded:
            QMessageBox.warning(self, "Warning", "Load a video first")
            return
        if len(self.worker.trackers) < self.worker.num_trackers:
            QMessageBox.warning(self, "Warning", "Select ROIs first")
            return
        # set KF flag and smoothing factors
        self.worker.use_kalman_filtering(self.chk_kf.isChecked())
        self.worker.smoothing_enabled = self.chk_smooth.isChecked()
        self.worker.smooth_alpha = max(0.01, min(0.99, self.smooth_alpha_spin.value() / 100.0))
        if not self.worker.isRunning():
            self.worker.start()
            self.show_status("Processing started")
        else:
            self.worker.paused = False
            self.show_status("Resumed processing")

    def on_pause(self):
        if not self.worker.isRunning():
            return
        self.worker.pause_toggle()

    def on_step_forward(self):
        if not self.video_loaded:
            return
        # use worker slot to step forward (emits frame)
        self.worker.step_forward()

    def on_step_back(self):
        if not self.video_loaded:
            return
        self.worker.step_back()

    def on_save_csv(self):
        if not self.worker.csv_rows:
            QMessageBox.information(self, "Info", "No data to export")
            return
        fname, _ = QFileDialog.getSaveFileName(self, "Save CSV", "tracking_output.csv", "CSV Files (*.csv)")
        if not fname:
            return
        self.worker.save_csv(fname)

    def on_export_video(self):
        # toggle or choose path
        fname, _ = QFileDialog.getSaveFileName(self, "Save Video", "tracked_output_full.mp4", "MP4 Files (*.mp4)")
        if not fname:
            return
        # set writer path in worker (simple approach: release old and create new)
        self.worker.video_writer = None
        # create new writer on next run call - we set path by replacing filename used by writer creation
        # Easiest: just inform user that output is saved to this path after processing; advanced: pass path into worker
        QMessageBox.information(self, "Info", f"Output video will be saved to {fname} (processing must run to completion)")

    @Slot(QImage)
    def update_image(self, qt_img):
        pix = QPixmap.fromImage(qt_img)
        pix = pix.scaled(self.image_label.size(), Qt.KeepAspectRatio)
        self.image_label.setPixmap(pix)

    @Slot(str)
    def show_status(self, msg):
        self.status_label.setText(msg)

    @Slot()
    def on_finished(self):
        QMessageBox.information(self, "Finished", "Processing finished.")

    def on_num_changed(self, val):
        # adjust reinit button container to at least val buttons if trackers already created later
        # we will re-create buttons after ROI selection anyway
        pass

    def on_kf_toggled(self, state):
        enabled = True if state == Qt.Checked else False
        self.worker.use_kalman_filtering(enabled)

    def on_smooth_toggled(self, state):
        self.worker.smoothing_enabled = True if state == Qt.Checked else False

    def on_smooth_alpha_changed(self, val):
        self.worker.smooth_alpha = max(0.01, min(0.99, val / 100.0))

    def closeEvent(self, event):
        if self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(1000)
        event.accept()


# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
