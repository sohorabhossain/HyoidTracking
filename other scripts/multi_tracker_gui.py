import sys
import time
import cv2
import numpy as np
import pandas as pd

from PySide6.QtCore import QThread, Signal, Slot, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QSpinBox, QComboBox, QMessageBox
)

# ----------------------------
# Utility: Kalman & Template functions (similar to previous)
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
        [1, 0, 0, 0, 0, 0],  # cx
        [0, 1, 0, 0, 0, 0],  # cy
        [0, 0, 0, 0, 1, 0],  # w
        [0, 0, 0, 0, 0, 1],  # h
    ], np.float32)
    kf.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-2
    kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
    kf.errorCovPost = np.eye(6, dtype=np.float32)
    x, y, w, h = roi
    cx = x + w / 2.0
    cy = y + h / 2.0
    kf.statePost = np.array([[cx], [cy], [0.], [0.], [w], [h]], dtype=np.float32)
    return kf

def extract_template_from_frame(frame, roi, pad=8):
    h, w = frame.shape[:2]
    x, y, rw, rh = roi
    x0 = max(0, int(x - pad))
    y0 = max(0, int(y - pad))
    x1 = min(w, int(x + rw + pad))
    y1 = min(h, int(y + rh + pad))
    templ = frame[y0:y1, x0:x1]
    if templ.size == 0:
        return None, (0, 0)
    templ_gray = cv2.cvtColor(templ, cv2.COLOR_BGR2GRAY)
    return templ_gray, (templ.shape[1], templ.shape[0])

# ----------------------------
# Video processing thread
# ----------------------------
class VideoThread(QThread):
    change_pixmap = Signal(QImage)
    status_msg = Signal(str)
    finished_processing = Signal()

    def __init__(self):
        super().__init__()
        # config and state
        self.video_path = None
        self.scale_fx = 0.65
        self.scale_fy = 0.65
        self.num_trackers = 1

        # internal containers
        self.cap = None
        self.trackers = []
        self.rois = []
        self.colors = []
        self.templates = []
        self.template_sizes = []
        self.kalman_filters = []
        self.trails = []
        self.csv_rows = []

        self.paused = True
        self.stop_requested = False
        # auto reinit params
        self.match_method = cv2.TM_CCOEFF_NORMED
        self.match_thresh = 0.60
        self.template_pad = 8

        self.fps_video = 30.0
        self.frame_idx = 0
        self.frames_since_reinit = 0
        self.use_kf = True #use kalman filtering

        # manual reinit requests: (index) or None
        self.manual_reinit_request = None

    def use_kalman_filtering(self, choice):
        self.use_kf = choice

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

    def init_trackers_containers(self):
        # create colors and empty containers
        self.trackers = []
        self.rois = []
        self.colors = []
        self.templates = []
        self.template_sizes = []
        self.kalman_filters = []
        self.trails = []
        self.csv_rows = []
        rng = np.random.default_rng(42)
        for _ in range(self.num_trackers):
            self.colors.append(tuple(int(c) for c in rng.integers(50, 255, 3)))
            self.trails.append([])

    @Slot(int)
    def set_num_trackers(self, n):
        self.num_trackers = int(n)

    @Slot()
    def select_rois_interactively(self):
        """Reads frames and allows user to choose ROIs using OpenCV windows.
           This runs in the worker thread to avoid blocking GUI.
        """
        if not self.cap:
            self.status_msg.emit("No video loaded")
            return

        # ensure starting from first frame for selection
        self.frame_idx = 0
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_idx)
        self.init_trackers_containers()

        for i in range(self.num_trackers):
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    self.status_msg.emit("Failed to read frame during ROI selection")
                    return
                frame = cv2.resize(frame, None, fx=self.scale_fx, fy=self.scale_fy)
                disp = frame.copy()
                cv2.putText(disp, f"Press N for next frame, R to select ROI {i+1}/{self.num_trackers}",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.imshow("Select Objects", disp)
                key = cv2.waitKey(0) & 0xFF
                if key in [ord("n"), ord("N")]:
                    continue
                elif key in [ord("r"), ord("R")]:
                    roi = cv2.selectROI("Select Objects", frame, False, False)
                    if roi[2] == 0 or roi[3] == 0:
                        self.status_msg.emit("Invalid ROI; try again")
                        continue
                    self.rois.append(roi)
                    tr = cv2.legacy.TrackerCSRT_create()
                    tr.init(frame, roi)
                    self.trackers.append(tr)

                    # Kalman + template
                    kf = create_kalman_from_roi(roi)
                    self.kalman_filters.append(kf)
                    templ_gray, tsize = extract_template_from_frame(frame, roi, pad=self.template_pad)
                    self.templates.append(templ_gray)
                    self.template_sizes.append(tsize)
                    break
                elif key == 27:
                    cv2.destroyWindow("Select Objects")
                    self.status_msg.emit("ROI selection canceled")
                    return
                self.frame_idx += 1

        cv2.destroyWindow("Select Objects")
        # reset video to beginning for processing
        # self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        # self.frame_idx = 0
        self.status_msg.emit(f"Selected {len(self.rois)} ROIs")

    def reinitiate_roi_interactively(self, roi_idx):
        """Reads frames and allows user to choose ROIs using OpenCV windows.
           This runs in the worker thread to avoid blocking GUI.
        """
        if not self.cap:
            self.status_msg.emit("No video loaded")
            return

        if self.num_trackers < roi_idx:
            self.status_msg.emit("Number of trackers is less than tracker selected")

        roi = []
        while True:
            ret, frame = self.cap.read()
            if not ret:
                self.status_msg.emit("Failed to read frame during ROI selection")
                return
            frame = cv2.resize(frame, None, fx=self.scale_fx, fy=self.scale_fy)
            disp = frame.copy()
            cv2.putText(disp, f"Press N for next frame, R to select ROI {roi_idx+1}/{self.num_trackers}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.imshow("Select Objects", disp)
            key = cv2.waitKey(0) & 0xFF
            if key in [ord("n"), ord("N")]:
                continue
            elif key in [ord("r"), ord("R")]:
                roi = cv2.selectROI("Select Objects", frame, False, False)
                if roi[2] == 0 or roi[3] == 0:
                    self.status_msg.emit("Invalid ROI; try again")
                    continue
                break
            elif key == 27:
                cv2.destroyWindow("Select Objects")
                self.status_msg.emit("ROI selection canceled")
            self.frame_idx += 1

        cv2.destroyWindow("Select Objects")
        return roi

    def run(self):
        if not self.cap:
            self.status_msg.emit("Load a video first")
            return

        self.stop_requested = False
        self.paused = False
        # self.frame_idx = 0
        # self.frames_since_reinit = 0

        # Setup video writer (optional use - set path later if needed)
        frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) * self.scale_fx)
        frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * self.scale_fy)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # default writer (in-memory) - user may decide to save; here we will create writer on demand
        self.video_writer = cv2.VideoWriter("tracked_output_gui.mp4", fourcc, self.fps_video, (frame_w, frame_h))

        while self.cap.isOpened() and not self.stop_requested:
            if self.paused:
                time.sleep(0.05)
                continue

            t0 = time.time()
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, None, fx=self.scale_fx, fy=self.scale_fy)
            vis = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # reset per-frame reinit flags
            reinit_attempted = [False] * self.num_trackers
            reinit_failed = [False] * self.num_trackers

            # update trackers
            for i in range(self.num_trackers):
                if i >= len(self.trackers):
                    # not initialized
                    continue

                tracker = self.trackers[i]
                ok, new_roi = tracker.update(frame)
                reinit_success = False

                measured = None
                if ok:
                    self.rois[i] = new_roi
                    x, y, w, h = map(int, self.rois[i])
                    cx = x + w/2.0
                    cy = y + h/2.0
                    measured = np.array([[np.float32(cx)], [np.float32(cy)], [np.float32(w)], [np.float32(h)]], dtype=np.float32)

                    # update template
                    templ_gray, tsize = extract_template_from_frame(frame, (x,y,w,h), pad=self.template_pad)
                    if templ_gray is not None and templ_gray.size>0:
                        self.templates[i] = templ_gray
                        self.template_sizes[i] = tsize

                    # correct Kalman
                    if self.kalman_filters[i] is None:
                        self.kalman_filters[i] = create_kalman_from_roi((x,y,w,h))
                    else:
                        self.kalman_filters[i].correct(measured)
                else:
                    # Try automatic re-init using template matching
                    reinit_attempted[i] = True
                    templ = self.templates[i]
                    if templ is not None:
                        res = cv2.matchTemplate(gray, templ, self.match_method)
                        _, max_val, _, max_loc = cv2.minMaxLoc(res)
                        if max_val >= self.match_thresh:
                            tx, ty = max_loc
                            tw, th = self.template_sizes[i]
                            new_x, new_y, new_w, new_h = int(tx), int(ty), int(tw), int(th)

                            # sanity bounds
                            fh, fw = gray.shape
                            new_x = max(0, min(new_x, fw - 1))
                            new_y = max(0, min(new_y, fh - 1))
                            new_w = max(1, min(new_w, fw - new_x))
                            new_h = max(1, min(new_h, fh - new_y))

                            new_roi = (new_x, new_y, new_w, new_h)
                            try:
                                new_tracker = cv2.legacy.TrackerCSRT_create()
                                new_tracker.init(frame, new_roi)
                                self.trackers[i] = new_tracker
                                self.rois[i] = new_roi
                                # update template
                                templ_gray2, tsize2 = extract_template_from_frame(frame, new_roi, pad=self.template_pad)
                                if templ_gray2 is not None and templ_gray2.size>0:
                                    self.templates[i] = templ_gray2
                                    self.template_sizes[i] = tsize2
                                # kalman correct
                                cx = new_x + new_w/2.0
                                cy = new_y + new_h/2.0
                                measured = np.array([[np.float32(cx)], [np.float32(cy)], [np.float32(new_w)], [np.float32(new_h)]], dtype=np.float32)
                                if self.kalman_filters[i] is None:
                                    self.kalman_filters[i] = create_kalman_from_roi(new_roi)
                                else:
                                    self.kalman_filters[i].correct(measured)
                                reinit_success = True
                            except Exception as e:
                                reinit_failed[i] = True
                        else:
                            reinit_failed[i] = True
                    else:
                        reinit_failed[i] = True

                # Kalman prediction for visualization (use prediction if measurement missing)
                if self.kalman_filters[i] is None:
                    # fallback to roi if exist
                    try:
                        vis_x, vis_y, vis_w, vis_h = map(int, self.rois[i])
                    except:
                        continue
                else:
                    kf = self.kalman_filters[i]
                    pred = kf.predict()
                    pred_cx = float(pred[0])
                    pred_cy = float(pred[1])
                    pred_w = float(pred[4])
                    pred_h = float(pred[5])
                    vis_w = int(max(1, pred_w))
                    vis_h = int(max(1, pred_h))
                    vis_x = int(pred_cx - vis_w/2.0)
                    vis_y = int(pred_cy - vis_h/2.0)

                # Draw results: green if ok or reinit success, yellow predicted if lost, red message if reinit failed
                if ok or reinit_success:
                    if self.frames_since_reinit <= 15:  # avoid drawing smoothed box and trails in the first few frames
                        vis_x, vis_y, vis_w, vis_h = map(int, self.rois[i])

                    cv2.rectangle(vis, (vis_x, vis_y), (vis_x+vis_w, vis_y+vis_h), self.colors[i], 2)
                    cv2.putText(vis, f"T{i+1}", (vis_x, max(12, vis_y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors[i], 2)
                    # update trail
                    center = (vis_x + vis_w//2, vis_y + vis_h//2)
                    self.trails[i].append(center)
                    if len(self.trails[i])>40:
                        self.trails[i].pop(0)
                    for t in range(1, len(self.trails[i])):
                        cv2.line(vis, self.trails[i][t-1], self.trails[i][t], self.colors[i], 2)
                else:
                    # lost
                    cv2.rectangle(vis, (vis_x, vis_y), (vis_x+vis_w, vis_y+vis_h), (0,255,255), 1)
                    cv2.putText(vis, f"T{i+1} LOST", (vis_x, max(12, vis_y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)
                    if reinit_failed[i]:
                        cv2.putText(vis, f"T{i+1} RE-INIT FAILED", (20, 60 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

                # Logging: measured and smoothed states
                if measured is not None:
                    meas_cx = float(measured[0]); meas_cy = float(measured[1]); meas_w = float(measured[2]); meas_h = float(measured[3])
                    meas_x = meas_cx - meas_w/2.0; meas_y = meas_cy - meas_h/2.0
                else:
                    meas_x = meas_y = meas_w = meas_h = np.nan

                if self.kalman_filters[i] is not None:
                    state = self.kalman_filters[i].statePost.flatten()
                    smooth_cx = float(state[0]); smooth_cy = float(state[1])
                    smooth_w = float(state[4]); smooth_h = float(state[5])
                    smooth_x = smooth_cx - smooth_w/2.0; smooth_y = smooth_cy - smooth_h/2.0
                else:
                    smooth_x = vis_x; smooth_y = vis_y; smooth_w = vis_w; smooth_h = vis_h

                self.csv_rows.append({
                    "frame": self.frame_idx,
                    "tracker_id": i+1,
                    "meas_x": meas_x, "meas_y": meas_y, "meas_w": meas_w, "meas_h": meas_h,
                    "smooth_x": smooth_x, "smooth_y": smooth_y, "smooth_w": smooth_w, "smooth_h": smooth_h,
                    "ok": bool(ok),
                    "reinit_attempted": bool(reinit_attempted[i]),
                    "reinit_success": bool(reinit_success),
                    "reinit_failed": bool(reinit_failed[i])
                })

            # increment frame index
            self.frame_idx += 1
            self.frames_since_reinit += 1

            # overlay FPS
            elapsed = (time.time() - t0)
            fps = int(1/elapsed) if elapsed>0 else 0
            cv2.putText(vis, f"FPS: {fps}", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)

            # write output
            self.video_writer.write(vis)

            # convert to QImage and emit
            rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
            self.change_pixmap.emit(qt_img)

            # handle manual reinit request if any
            if self.manual_reinit_request is not None:
                req_idx = self.manual_reinit_request
                self.manual_reinit_request = None
                # pause briefly and ask user to select new ROI for that tracker
                self.paused = True
                self.status_msg.emit(f"Manual reinit: select ROI for tracker {req_idx+1} (OpenCV window)")
                # show current frame and ask for ROI
                new_roi = self.reinitiate_roi_interactively(req_idx)

                if new_roi[2] == 0 or new_roi[3] == 0:
                    self.status_msg.emit("Manual reinit canceled or invalid ROI")
                else:
                    try:
                        new_tracker = cv2.legacy.TrackerCSRT_create()
                        new_tracker.init(frame, new_roi)
                        self.trackers[req_idx] = new_tracker
                        self.rois[req_idx] = new_roi
                        templ_gray2, tsize2 = extract_template_from_frame(frame, new_roi, pad=self.template_pad)
                        if templ_gray2 is not None and templ_gray2.size>0:
                            self.templates[req_idx] = templ_gray2
                            self.template_sizes[req_idx] = tsize2
                        # reset kalman
                        self.kalman_filters[req_idx] = create_kalman_from_roi(new_roi)
                        self.status_msg.emit(f"Manual reinit for tracker {req_idx+1} done")
                    except Exception as e:
                        self.status_msg.emit(f"Manual reinit error: {e}")
                self.paused = False
                self.frames_since_reinit = 0

            # keep a sane frame rate - small sleep if processing very fast
            time.sleep(0.001)

        # cleanup after loop
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
        # queue manual reinit for tracker idx
        if 0 <= idx < self.num_trackers:
            self.manual_reinit_request = idx
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

# ----------------------------
# Main window GUI (Option B)
# ----------------------------
class MainWindow(QWidget):
    def __init__(self, num_of_tracker=2, use_kf=None):
        super().__init__()
        self.setWindowTitle("Multi-Tracker GUI")
        self.resize(1200, 700)

        # left image label
        self.image_label = QLabel("Load a video to begin")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(900, 700)

        # right side controls
        self.spin_num = QSpinBox()
        self.spin_num.setMinimum(1)
        self.spin_num.setValue(num_of_tracker)
        self.spin_num.setMaximum(20)

        self.btn_load = QPushButton("Start Video Rendering")
        self.btn_select_rois = QPushButton("Select ROIs")
        self.btn_start = QPushButton("Start Tracking")
        self.btn_pause = QPushButton("Pause/Resume")
        self.combo_reinit = QComboBox()
        self.combo_reinit.addItems([f"Tracker {i+1}" for i in range(4)])  # will refresh
        self.btn_reinit = QPushButton("Reinit Selected")
        self.btn_export = QPushButton("Export CSV")
        self.btn_exit = QPushButton("Exit")

        # right layout
        vbox = QVBoxLayout()
        vbox.addWidget(QLabel("Number of trackers:"))
        vbox.addWidget(self.spin_num)
        vbox.addSpacing(10)
        vbox.addWidget(self.btn_load)
        vbox.addWidget(self.btn_select_rois)
        vbox.addWidget(self.btn_start)
        vbox.addWidget(self.btn_pause)
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

        # thread
        self.worker = VideoThread()
        self.worker.use_kalman_filtering(use_kf)
        self.worker.change_pixmap.connect(self.update_image)
        self.worker.status_msg.connect(self.show_status)
        self.worker.finished_processing.connect(self.on_finished)

        # connect signals
        self.btn_load.clicked.connect(self.on_load)
        self.btn_select_rois.clicked.connect(self.on_select_rois)
        self.btn_start.clicked.connect(self.on_start_tracking)
        self.btn_pause.clicked.connect(self.on_pause)
        self.btn_reinit.clicked.connect(self.on_manual_reinit)
        self.btn_export.clicked.connect(self.on_export)
        self.btn_exit.clicked.connect(self.close)
        self.spin_num.valueChanged.connect(self.on_num_changed)

        # status bar-like label
        self.status_label = QLabel("")
        vbox.addWidget(self.status_label)

    @Slot()
    def on_load(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv)")
        if not fname:
            return
        try:
            self.worker.load_video(fname)
            self.image_label.setText("Video loaded. Now select ROIs.")
            self.show_status("Video loaded: " + fname)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cannot load video: {e}")

    @Slot()
    def on_select_rois(self):
        # set num trackers in worker
        n = self.spin_num.value()
        self.worker.set_num_trackers(n)
        # refresh reinit combo
        self.combo_reinit.clear()
        self.combo_reinit.addItems([f"Tracker {i+1}" for i in range(n)])
        # run selection (in worker thread)
        self.worker.select_rois_interactively()
        self.show_status("ROIs selected (if completed)")
        self.on_start_tracking()

    @Slot()
    def on_start_tracking(self):
        if not self.worker.cap:
            QMessageBox.warning(self, "Warning", "Load a video first")
            return
        if len(self.worker.trackers) < self.worker.num_trackers:
            QMessageBox.warning(self, "Warning", "Select ROIs first")
            return
        # set current parameters from GUI
        self.worker.scale_fx = 0.65
        self.worker.scale_fy = 0.65
        # start thread if not running
        if not self.worker.isRunning():
            self.worker.start()
            self.show_status("Processing started")
        else:
            # if running but paused, ensure running
            self.worker.paused = False
            self.show_status("Resumed processing")

    @Slot()
    def on_pause(self):
        if not self.worker.isRunning():
            return
        self.worker.pause_toggle()

    @Slot()
    def on_manual_reinit(self):
        # request manual reinit for selected tracker
        idx = self.combo_reinit.currentIndex()
        if not self.worker.isRunning():
            QMessageBox.warning(self, "Warning", "Start processing first")
            return
        self.worker.request_manual_reinit(idx)

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
    def update_image(self, qt_img):
        # scale pixmap to fit label preserving aspect
        pix = QPixmap.fromImage(qt_img)
        pix = pix.scaled(self.image_label.size(), Qt.KeepAspectRatio)
        self.image_label.setPixmap(pix)

    @Slot(str)
    def show_status(self, msg):
        self.status_label.setText(msg)

    @Slot()
    def on_finished(self):
        QMessageBox.information(self, "Finished", "Processing finished and files saved (if enabled).")

    @Slot(int)
    def on_num_changed(self, val):
        # update combo options
        self.combo_reinit.clear()
        self.combo_reinit.addItems([f"Tracker {i+1}" for i in range(val)])
        self.worker.set_num_trackers(val)

    def closeEvent(self, event):
        # ensure worker stops
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
