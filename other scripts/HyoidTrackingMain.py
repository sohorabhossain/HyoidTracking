# multi_tracker_gui_orb_kf_guiroi.py
import sys
import time
import cv2
import numpy as np
import pandas as pd

from PySide6.QtCore import QThread, Signal, Slot, Qt, QRect, QPoint
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QKeyEvent
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QSpinBox, QComboBox, QMessageBox, QCheckBox, QSizePolicy
)

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

# ----------------------------
# QLabel subclass for drawing/selecting ROIs
# ----------------------------
class DrawableLabel(QLabel):
    """
    QLabel that supports drawing a rectangle with mouse.
    Emits no signal; instead MainWindow calls get_drawn_rois() after drawing mode finishes.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = None
        self.drawing = False
        self.start_point = QPoint()
        self.end_point = QPoint()
        self.rects = []  # list of QRect in display coords
        self.temp_rect = None
        self.setMouseTracking(True)
        self.draw_mode = False
        self.scale_x = 1.0
        self.scale_y = 1.0

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
        # draw existing rects
        for r in self.rects:
            painter.drawRect(r)
        # draw temp rect
        if self.temp_rect is not None:
            pen = QPen(Qt.yellow, 2)
            painter.setPen(pen)
            painter.drawRect(self.temp_rect)
        painter.end()

    def get_rects_display(self):
        """Return list of QRect in display coordinates."""
        return list(self.rects)

    def clear_rects(self):
        self.rects = []
        self.temp_rect = None
        self.update()

# ----------------------------
# Video processing thread
# ----------------------------
class VideoThread(QThread):
    change_pixmap = Signal(QImage)
    status_msg = Signal(str)
    finished_processing = Signal()

    def __init__(self):
        super().__init__()
        # video/config
        self.video_path = None
        self.cap = None
        self.scale_fx = 0.65
        self.scale_fy = 0.65
        self.num_trackers = 1
        self.fps_video = 30.0

        # tracker data
        self.trackers = []
        self.rois = []            # raw ROIs (x,y,w,h) in processed frame coords
        self.colors = []
        self.trails = []
        self.csv_rows = []

        # ORB templates: store keypoints & descriptors and template size
        self.templates_kp = []
        self.templates_des = []
        self.templates_size = []

        # Kalman filters (optional)
        self.kalman_filters = []

        # ORB matcher & detector
        self.orb = cv2.ORB_create(1000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # reinit parameters
        self.match_thresh = 8  # minimal number of good matches required
        self.template_pad_fraction = 0.20  # expand template by 20%

        # control flags
        self.paused = True
        self.stop_requested = False
        self.manual_reinit_request = None  # (idx, roi) or (idx, None) to pop-up drawing
        self.use_kf = True

        # writer
        self.video_writer = None

        # frame index
        self.frame_idx = 0
        self.frames_since_reinit = 0

    def use_kalman_filtering(self, choice: bool):
        self.use_kf = bool(choice)
        self.status_msg.emit(f"Kalman filtering {'enabled' if self.use_kf else 'disabled'}")
        # If enabling, create KFs for existing ROIs immediately
        if self.use_kf and len(self.rois) > 0:
            # create KF objects for any trackers missing them
            for i in range(len(self.rois)):
                if i >= len(self.kalman_filters) or self.kalman_filters[i] is None:
                    # create and store
                    if i >= len(self.kalman_filters):
                        self.kalman_filters.extend([None] * (i - len(self.kalman_filters) + 1))
                    try:
                        kf = create_kalman_from_roi(self.rois[i])
                        self.kalman_filters[i] = kf
                    except Exception:
                        # ignore creation errors
                        self.kalman_filters[i] = None

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
        if fps and not np.isnan(fps) and fps > 0:
            self.fps_video = fps
        else:
            self.fps_video = 30.0
        self.status_msg.emit(f"Video loaded ({self.fps_video:.2f} FPS)")

    def init_trackers_from_rois(self, frame, rois_list):
        """Initialize trackers, templates, KFs and other containers from a provided frame and rois list.
           frame: BGR image already resized by scale_fx/fy (processed frame coords).
           rois_list: list of (x,y,w,h) in processed frame coords.
        """
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
            self.rois.append(roi)
            tr = cv2.legacy.TrackerCSRT_create()
            try:
                tr.init(frame, roi)
            except Exception:
                # Initialize may fail if ROI out of frame - clamp
                x, y, w, h = roi
                x = max(0, min(x, w_frame - 1))
                y = max(0, min(y, h_frame - 1))
                w = max(1, min(w, w_frame - x))
                h = max(1, min(h, h_frame - y))
                roi = (x, y, w, h)
                tr.init(frame, roi)
            self.trackers.append(tr)
            self.colors.append(tuple(int(c) for c in rng.integers(50, 255, 3)))
            self.trails.append([])

            # create ORB template for the expanded ROI
            ex_roi = expand_roi(roi, self.template_pad_fraction, w_frame, h_frame)
            x2, y2, w2, h2 = map(int, ex_roi)
            templ = frame[y2:y2 + h2, x2:x2 + w2]
            templ_gray = cv2.cvtColor(templ, cv2.COLOR_BGR2GRAY)
            kp, des = self.orb.detectAndCompute(templ_gray, None) if templ_gray.size > 0 else ([], None)
            self.templates_kp.append(kp)
            self.templates_des.append(des)
            self.templates_size.append((w2, h2))

            # create KF only if enabled
            if self.use_kf:
                try:
                    kf = create_kalman_from_roi(roi)
                except Exception:
                    kf = None
            else:
                kf = None
            self.kalman_filters.append(kf)

    def try_orb_reinit(self, frame_gray, templ_kp, templ_des, templ_size):
        """
        Try to re-detect template area in frame using ORB feature matching.
        Returns new_roi (x,y,w,h) in frame coords if successful, else None.
        """
        if templ_des is None or len(templ_des) < 4:
            return None

        kp_scene, des_scene = self.orb.detectAndCompute(frame_gray, None)
        if des_scene is None or len(des_scene) < 4:
            return None

        # match descriptors
        try:
            matches = self.bf.match(templ_des, des_scene)
        except Exception:
            return None
        if not matches:
            return None

        # sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)
        # take good matches (heuristic)
        good = matches[: max(10, int(len(matches)*0.25))]

        if len(good) < self.match_thresh:
            return None

        pts_template = np.float32([templ_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts_scene = np.float32([kp_scene[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        # compute homography
        try:
            H, mask = cv2.findHomography(pts_template, pts_scene, cv2.RANSAC, 5.0)
            if H is None:
                return None
        except Exception:
            return None

        th, tw = templ_size[1], templ_size[0]
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
        w_new = max(1, max_x - min_x)
        h_new = max(1, max_y - min_y)
        return (min_x, min_y, w_new, h_new)

    def run(self):
        if not self.cap:
            self.status_msg.emit("Load a video first")
            return

        self.stop_requested = False
        self.paused = False
        self.frame_idx = 0
        self.frames_since_reinit = 0

        # Setup video writer
        frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) * self.scale_fx)
        frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * self.scale_fy)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter("tracked_output_gui.mp4", fourcc, self.fps_video, (frame_w, frame_h))

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
                    x, y, w, h = map(int, new_roi)
                    cx = x + w / 2.0
                    cy = y + h / 2.0
                    measured = np.array([[np.float32(cx)], [np.float32(cy)], [np.float32(w)], [np.float32(h)]], dtype=np.float32)

                    # update ORB template with expanded ROI
                    ex_roi = expand_roi((x, y, w, h), self.template_pad_fraction, gray.shape[1], gray.shape[0])
                    ex_x, ex_y, ex_w, ex_h = map(int, ex_roi)
                    templ_img = frame[ex_y:ex_y+ex_h, ex_x:ex_x+ex_w]
                    templ_gray = cv2.cvtColor(templ_img, cv2.COLOR_BGR2GRAY)
                    kp, des = self.orb.detectAndCompute(templ_gray, None) if templ_gray.size>0 else ([], None)
                    self.templates_kp[i] = kp
                    self.templates_des[i] = des
                    self.templates_size[i] = (ex_w, ex_h)

                    # Kalman correction only if enabled
                    if self.use_kf:
                        if self.kalman_filters[i] is None:
                            self.kalman_filters[i] = create_kalman_from_roi((x, y, w, h))
                        else:
                            try:
                                self.kalman_filters[i].correct(measured)
                            except Exception:
                                self.kalman_filters[i] = create_kalman_from_roi((x, y, w, h))
                else:
                    # attempt ORB-based reinit
                    reinit_attempted[i] = True
                    kp_t = self.templates_kp[i]
                    des_t = self.templates_des[i]
                    tsize = self.templates_size[i]
                    if des_t is not None and len(des_t) >= 4:
                        new_roi = self.try_orb_reinit(gray, kp_t, des_t, tsize)
                        if new_roi is not None:
                            try:
                                new_tracker = cv2.legacy.TrackerCSRT_create()
                                new_tracker.init(frame, new_roi)
                                self.trackers[i] = new_tracker
                                self.rois[i] = new_roi
                                ex_x, ex_y, ex_w, ex_h = map(int, expand_roi(new_roi, self.template_pad_fraction, gray.shape[1], gray.shape[0]))
                                templ_img = frame[ex_y:ex_y+ex_h, ex_x:ex_x+ex_w]
                                templ_gray = cv2.cvtColor(templ_img, cv2.COLOR_BGR2GRAY)
                                kp, des = self.orb.detectAndCompute(templ_gray, None) if templ_gray.size>0 else ([], None)
                                self.templates_kp[i] = kp
                                self.templates_des[i] = des
                                self.templates_size[i] = (ex_w, ex_h)
                                # if KF enabled, correct/create
                                if self.use_kf:
                                    cx = new_roi[0] + new_roi[2] / 2.0
                                    cy = new_roi[1] + new_roi[3] / 2.0
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

                # Visualization coordinates
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

                # Draw results
                if ok or reinit_success:
                    # shortly after reinit draw actual roi rather than prediction
                    if self.frames_since_reinit <= 15 and (ok or reinit_success):
                        try:
                            vis_x, vis_y, vis_w, vis_h = map(int, self.rois[i])
                        except Exception:
                            pass
                    cv2.rectangle(vis, (vis_x, vis_y), (vis_x + vis_w, vis_y + vis_h), self.colors[i], 2)
                    cv2.putText(vis, f"T{i+1}", (vis_x, max(12, vis_y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors[i], 2)
                    center = (vis_x + vis_w // 2, vis_y + vis_h // 2)
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

                # Logging
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
                cv2.putText(vis, f"Press N for next frame",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # emit frame
            rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
            self.change_pixmap.emit(qt_img)

            # check manual reinit request
            if self.manual_reinit_request is not None:
                idx, roi = self.manual_reinit_request
                self.manual_reinit_request = None
                self.paused = True
                self.status_msg.emit(f"Manual reinit for tracker {idx+1}")
                if roi is None:
                    # UI will supply ROI drawing, so just wait until MainWindow calls request_manual_reinit_roi
                    pass
                else:
                    # roi provided in processed-frame coords; apply immediately
                    try:
                        new_tracker = cv2.legacy.TrackerCSRT_create()
                        new_tracker.init(frame, roi)
                        self.trackers[idx] = new_tracker
                        self.rois[idx] = roi
                        # update ORB template
                        ex_roi = expand_roi(roi, self.template_pad_fraction, gray.shape[1], gray.shape[0])
                        ex_x, ex_y, ex_w, ex_h = map(int, ex_roi)
                        templ_img = frame[ex_y:ex_y+ex_h, ex_x:ex_x+ex_w]
                        templ_gray = cv2.cvtColor(templ_img, cv2.COLOR_BGR2GRAY)
                        kp, des = self.orb.detectAndCompute(templ_gray, None) if templ_gray.size>0 else ([], None)
                        self.templates_kp[idx] = kp
                        self.templates_des[idx] = des
                        self.templates_size[idx] = (ex_w, ex_h)
                        # create KF if enabled
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
            # queue a manual reinit request with roi=None (MainWindow will provide roi via request_manual_reinit_roi)
            self.manual_reinit_request = (idx, None)
        else:
            self.status_msg.emit("Invalid tracker index for reinit")

    @Slot(tuple)
    def request_manual_reinit_with_roi(self, args):
        """args = (idx, roi, frame) where roi is in processed-frame coords, frame is image"""
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


# ----------------------------
# Main window GUI
# ----------------------------
class MainWindow(QWidget):
    def __init__(self, num_of_tracker=2, use_kf=True):
        super().__init__()
        self.setWindowTitle("Multi-Tracker GUI (ORB reinit + KF toggle + GUI ROI)")
        self.resize(1200, 760)

        # left panel: DrawableLabel
        self.image_label = DrawableLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        # placeholder size until video loaded
        self.display_w = 900
        self.display_h = 700
        self.image_label.setFixedSize(self.display_w, self.display_h)

        # right controls
        self.spin_num = QSpinBox()
        self.spin_num.setMinimum(1)
        self.spin_num.setValue(num_of_tracker)
        self.spin_num.setMaximum(20)

        self.chk_kf = QCheckBox("Use Kalman Filter")
        self.chk_kf.setChecked(bool(use_kf))

        self.btn_load = QPushButton("Start Video Rendering")
        self.btn_select_rois = QPushButton("Select ROIs (GUI)")
        self.btn_start = QPushButton("Start Tracking")
        self.btn_pause = QPushButton("Pause/Resume")
        self.combo_reinit = QComboBox()
        self.combo_reinit.addItems([f"Tracker {i+1}" for i in range(num_of_tracker)])
        self.btn_reinit = QPushButton("Reinit Selected (draw)")
        self.btn_export = QPushButton("Export CSV")
        self.btn_exit = QPushButton("Exit")

        vbox = QVBoxLayout()
        vbox.addWidget(QLabel("Number of trackers:"))
        vbox.addWidget(self.spin_num)
        vbox.addWidget(self.chk_kf)
        vbox.addSpacing(10)
        vbox.addWidget(self.btn_load)
        vbox.addWidget(self.btn_select_rois)
        # vbox.addWidget(self.btn_start)
        # vbox.addWidget(self.btn_pause)
        vbox.addWidget(QLabel("Manual Reinit:"))
        vbox.addWidget(self.combo_reinit)
        vbox.addWidget(self.btn_reinit)
        vbox.addWidget(self.btn_export)
        vbox.addWidget(self.btn_exit)
        vbox.addStretch(1)

        hbox = QHBoxLayout()
        hbox.addWidget(self.image_label)
        hbox.addLayout(vbox)
        self.setLayout(hbox)

        # status label
        self.status_label = QLabel("")
        vbox.addWidget(self.status_label)

        # worker thread
        self.worker = VideoThread()
        self.worker.use_kalman_filtering(self.chk_kf.isChecked())
        self.worker.change_pixmap.connect(self.on_frame)
        self.worker.status_msg.connect(self.show_status)
        self.worker.finished_processing.connect(self.on_finished)

        # connect signals
        self.btn_load.clicked.connect(self.on_load)
        self.btn_select_rois.clicked.connect(self.on_select_rois_gui)
        self.btn_start.clicked.connect(self.on_start_tracking)
        self.btn_pause.clicked.connect(self.on_pause)
        self.btn_reinit.clicked.connect(self.on_manual_reinit_gui)
        self.btn_export.clicked.connect(self.on_export)
        self.btn_exit.clicked.connect(self.close)
        self.spin_num.valueChanged.connect(self.on_num_changed)
        self.chk_kf.stateChanged.connect(self.on_kf_toggled)

        # internal: last frame shown (processed size)
        self.last_frame = None  # numpy BGR image in processed coords
        self.selecting_roi = False

    @Slot()
    def on_load(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv)")
        if not fname:
            return
        try:
            self.worker.load_video(fname)
            # set display label to match processed frame size
            w = int(self.worker.cap.get(cv2.CAP_PROP_FRAME_WIDTH) * self.worker.scale_fx)
            h = int(self.worker.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * self.worker.scale_fy)
            self.display_w, self.display_h = w, h
            self.image_label.setFixedSize(w, h)
            self.image_label.clear_rects()
            self.image_label.setPixmap(QPixmap(w, h))
            self.show_status("Video loaded. Click Select ROIs to draw.")
            # read first frame to show
            self.worker.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.worker.cap.read()
            if ret:
                frame = cv2.resize(frame, None, fx=self.worker.scale_fx, fy=self.worker.scale_fy)
                self.last_frame = frame.copy()
                self._display_frame(frame)
                if not self.worker.isRunning():
                    self.worker.start()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cannot load video: {e}")

    def _display_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(qt_img)
        self.image_label.setPixmap(pix)

    @Slot()
    def on_select_rois_gui(self):
        """Enable GUI ROI drawing on the current displayed frame, collect N ROIs, and initialize trackers."""
        n = self.spin_num.value()
        if self.worker.cap is None:
            QMessageBox.warning(self, "Warning", "Load a video first")
            return
        # ensure we have a recent frame
        #self.worker.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.worker.cap.read()
        if not ret:
            QMessageBox.warning(self, "Warning", "Could not read frame for ROI selection")
            return
        self.selecting_roi = True
        frame = cv2.resize(frame, None, fx=self.worker.scale_fx, fy=self.worker.scale_fy)
        self.last_frame = frame.copy()
        cv2.putText(self.last_frame, f"Press N for next frame",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        self._display_frame(self.last_frame)

        # start draw mode
        self.image_label.enter_draw_mode()
        self.show_status(f"Draw {n} ROIs on the image (drag). Press Escape to cancel.")
        # Wait for user to draw n rectangles (blocking UI loop but allowed)
        # To keep UI responsive, we'll use a simple event loop polling rect count.
        while True:
            QApplication.processEvents()
            rects = self.image_label.get_rects_display()
            if len(rects) >= n:
                break
            # allow user to cancel via Escape key
            # we can't capture Escape easily here; rely on Cancel button in future

            time.sleep(0.05)

        # done drawing
        self.image_label.exit_draw_mode()
        rects = self.image_label.get_rects_display()[:n]

        # convert display QRect to processed-frame coords (they are the same because we fixed label size to frame size)
        rois = []
        for r in rects:
            x = r.x(); y = r.y(); w = r.width(); h = r.height()
            rois.append((int(x), int(y), int(w), int(h)))

        # initialize trackers inside worker
        self.worker.set_num_trackers(n)
        self.worker.init_trackers_from_rois(self.last_frame, rois)
        self.combo_reinit.clear()
        self.combo_reinit.addItems([f"Tracker {i+1}" for i in range(n)])
        self.show_status("ROIs set and trackers initialized (GUI).")
        self.image_label.clear_rects()
        self.selecting_roi = False
        self.on_start_tracking()

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_N and self.selecting_roi:
            ret, frame = self.worker.cap.read()
            if not ret:
                QMessageBox.warning(self, "Warning", "Could not read frame for ROI selection")
                return
            frame = cv2.resize(frame, None, fx=self.worker.scale_fx, fy=self.worker.scale_fy)
            self.last_frame = frame.copy()
            cv2.putText(self.last_frame, f"Press N for next frame",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            self._display_frame(self.last_frame)


    @Slot()
    def on_start_tracking(self):
        if not self.worker.cap:
            QMessageBox.warning(self, "Warning", "Load a video first")
            return
        if len(self.worker.trackers) < self.worker.num_trackers:
            QMessageBox.warning(self, "Warning", "Select ROIs first")
            return
        # set KF flag from checkbox and ensure worker creates KFs if enabling now
        self.worker.use_kalman_filtering(self.chk_kf.isChecked())
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
        idx = self.combo_reinit.currentIndex()
        if not self.worker.cap:
            QMessageBox.warning(self, "Warning", "Load a video first")
            return
        # Pause worker if running
        was_running = self.worker.isRunning() and not self.worker.paused
        if was_running:
            self.worker.paused = True
        # show current frame (grab fresh)
        pos = int(self.worker.cap.get(cv2.CAP_PROP_POS_FRAMES))
        # step back one frame to get current
        seek = max(0, pos - 1)
        # self.worker.cap.set(cv2.CAP_PROP_POS_FRAMES, seek)
        ret, frame = self.worker.cap.read()
        if not ret:
            QMessageBox.warning(self, "Warning", "Could not read frame for manual reinit")
            return
        self.selecting_roi = True

        # allow user to draw one ROI
        self.image_label.clear_rects()
        self.image_label.enter_draw_mode()
        self.show_status(f"Draw ROI for tracker {idx+1}")

        while True:
            QApplication.processEvents()
            rects = self.image_label.get_rects_display()
            if len(rects) >= 1:
                break
            time.sleep(0.05)
        self.image_label.clear_rects()
        self.image_label.exit_draw_mode()
        r = rects[0]
        new_roi = (int(r.x()), int(r.y()), int(r.width()), int(r.height()))

        # send to worker to apply
        self.worker.request_manual_reinit_with_roi((idx, new_roi))
        self.show_status(f"Manual reinit requested for tracker {idx+1}")
        # resume if was running
        if was_running:
            self.worker.paused = False
        self.selecting_roi = False

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
        # ensure label size consistent
        pix = pix.scaled(self.image_label.size(), Qt.KeepAspectRatio)
        self.image_label.setPixmap(pix)
        # convert back to last_frame for manual ROI usage if needed
        # we won't convert back here; last_frame updated when selecting ROIs manually

    @Slot(str)
    def show_status(self, msg):
        self.status_label.setText(msg)

    @Slot()
    def on_finished(self):
        # QMessageBox.information(self, "Finished", "Processing finished.")
        self.close()    #check it back. Add code if anything needes to be done before closing everything

    @Slot(int)
    def on_num_changed(self, val):
        self.combo_reinit.clear()
        self.combo_reinit.addItems([f"Tracker {i+1}" for i in range(val)])
        self.worker.set_num_trackers(val)

    @Slot(int)
    def on_kf_toggled(self, state):
        enabled = True if state == Qt.Checked else False
        self.worker.use_kalman_filtering(enabled)

    def closeEvent(self, event):
        if self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(1000)
        event.accept()

# ----------------------------
# Run application
# ----------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow(num_of_tracker=1, use_kf=False)
    w.show()
    sys.exit(app.exec())
