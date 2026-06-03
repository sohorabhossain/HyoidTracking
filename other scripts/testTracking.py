import cv2
import time
import numpy as np
import pandas as pd

# ----------------------------------------------------------
# USER CONFIGURATION
# ----------------------------------------------------------
videoFileName = r"C:\Users\vesna\Desktop\sohorab\Hyoid tracking\clippedData\39\Effortful 2 MS.mp4"
scaleFactor_fx = 0.65
scaleFactor_fy = 0.65
num_trackers = 2
#kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))

output_video_path = "tracked_output_with_kalman.mp4"
output_csv_path = "tracking_data_with_kalman.csv"
trail_length = 40              # Number of historical positions to store
initial_playback_speed = 1.0   # Normal speed

# Template matching parameters for re-init
match_method = cv2.TM_CCOEFF_NORMED
match_thresh = 0.60            # match quality threshold (0..1); adjust as needed
template_pad = 8               # pixels of padding around ROI for template
# ----------------------------------------------------------

cap = cv2.VideoCapture(videoFileName)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

fps_video = cap.get(cv2.CAP_PROP_FPS)
if fps_video <= 0 or np.isnan(fps_video):
    fps_video = 30.0

# Tracker storage
trackers = []
rois = []
colors = []

# Trails
trails = [[] for _ in range(num_trackers)]

# Kalman filters, templates, and status flags
kalman_filters = [None] * num_trackers
templates = [None] * num_trackers            # last good template (grayscale)
template_sizes = [None] * num_trackers       # (w,h) of template
reinit_flags = [False] * num_trackers        # whether a reinit was attempted this frame
reinit_fail = [False] * num_trackers         # whether reinit failed (used to display message)

# Random colors for trackers
rng = np.random.default_rng(42)
for _ in range(num_trackers):
    colors.append(tuple(int(c) for c in rng.integers(50, 255, 3)))

# Utility: init Kalman for a tracker given initial ROI (x, y, w, h)
def create_kalman_from_roi(roi):
    # state = [cx, cy, vx, vy, w, h] (6)
    # measurement = [cx, cy, w, h] (4)
    kf = cv2.KalmanFilter(6, 4)
    # Transition matrix (A)
    dt = 1.0
    kf.transitionMatrix = np.array([
        [1, 0, dt, 0, 0, 0],
        [0, 1, 0, dt, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ], np.float32)
    # Measurement matrix (H)
    kf.measurementMatrix = np.array([
        [1, 0, 0, 0, 0, 0],  # cx
        [0, 1, 0, 0, 0, 0],  # cy
        [0, 0, 0, 0, 1, 0],  # w
        [0, 0, 0, 0, 0, 1],  # h
    ], np.float32)
    # Process noise covariance
    kf.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-2
    # Measurement noise covariance
    kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
    # Error covariance
    kf.errorCovPost = np.eye(6, dtype=np.float32)

    x, y, w, h = roi
    cx = x + w / 2.0
    cy = y + h / 2.0

    kf.statePost = np.array([[cx], [cy], [0.], [0.], [w], [h]], dtype=np.float32)
    return kf

# Template extraction utility (grayscale, with padding and boundary checks)
def extract_template_from_frame(frame, roi, pad=template_pad):
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

# ----------------------------------------------------------
# SELECT ROIs
# ----------------------------------------------------------
for i in range(num_trackers):
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame during ROI selection.")
            exit()

        frame = cv2.resize(frame, None, fx=scaleFactor_fx, fy=scaleFactor_fy)

        display = frame.copy()
        cv2.putText(display,
                    f"Press N for next frame, R to select ROI {i+1}/{num_trackers}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)

        cv2.imshow("Select Objects", display)
        key = cv2.waitKey(0)

        if key in [ord("n"), ord("N")]:
            continue

        elif key in [ord("r"), ord("R")]:
            roi = cv2.selectROI("Select Objects", frame, False, False)
            if roi[2] == 0 or roi[3] == 0:
                print("Invalid ROI. Try again.")
                continue

            rois.append(roi)

            tracker = cv2.legacy.TrackerCSRT_create()
            tracker.init(frame, roi)
            trackers.append(tracker)

            # Init Kalman and template for this tracker
            kf = create_kalman_from_roi(roi)
            kalman_filters[i] = kf
            templ_gray, tsize = extract_template_from_frame(frame, roi)
            templates[i] = templ_gray
            template_sizes[i] = tsize
            reinit_flags[i] = False
            reinit_fail[i] = False
            break

        elif key == 27:
            exit()

cv2.destroyWindow("Select Objects")
print("Tracking started. Automatic re-init (template matching) enabled. Controls: Space=Pause, ← slower, → faster, ESC quit.")

# ----------------------------------------------------------
# SET UP VIDEO WRITER
# ----------------------------------------------------------
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scaleFactor_fx)
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scaleFactor_fy)
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps_video, (frame_w, frame_h))

# ----------------------------------------------------------
# CSV STORAGE
# ----------------------------------------------------------
csv_data = []

# ----------------------------------------------------------
# TRACKING LOOP
# ----------------------------------------------------------
paused = False
playback_speed = initial_playback_speed
frame_idx = 0

while True:
    if not paused:
        start_time = time.time()
        ret, frame = cap.read()

        if not ret:
            break

        # Resize
        frame = cv2.resize(frame, None, fx=scaleFactor_fx, fy=scaleFactor_fy)
        temp_frame = frame.copy()
        gray = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2GRAY)

        # Reset reinit flags for this frame
        for ri in range(num_trackers):
            reinit_flags[ri] = False
            reinit_fail[ri] = False

        # Update trackers
        for i, tracker in enumerate(trackers):
            ok, new_roi = tracker.update(temp_frame)

            measured = None
            reinit_success = False

            if ok:
                # Successful measurement from CSRT
                rois[i] = new_roi
                x, y, w, h = map(int, rois[i])
                cx = x + w / 2.0
                cy = y + h / 2.0
                measured = np.array([[np.float32(cx)], [np.float32(cy)], [np.float32(w)], [np.float32(h)]], dtype=np.float32)

                # Update template with the new good measurement
                templ_gray, tsize = extract_template_from_frame(temp_frame, (x, y, w, h))
                if templ_gray is not None and templ_gray.size > 0:
                    templates[i] = templ_gray
                    template_sizes[i] = tsize

                # Kalman correction with measurement
                if kalman_filters[i] is None:
                    kalman_filters[i] = create_kalman_from_roi((x, y, w, h))
                else:
                    kf = kalman_filters[i]
                    kf.correct(measured)

            else:
                # Tracker failed: attempt automatic re-initialization using template matching
                reinit_flags[i] = True
                templ = templates[i]
                if templ is not None:
                    # run template matching on grayscale frame
                    res = cv2.matchTemplate(gray, templ, match_method)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

                    # For TM_CCOEFF_NORMED, higher is better (max_val)
                    best_val = max_val
                    best_loc = max_loc

                    if best_val >= match_thresh:
                        # compute matched top-left corner in full frame coordinates (already scaled)
                        tx, ty = best_loc
                        tw, th = template_sizes[i]
                        new_x, new_y, new_w, new_h = int(tx), int(ty), int(tw), int(th)

                        # Bound check
                        fh, fw = gray.shape
                        new_x = max(0, min(new_x, fw - 1))
                        new_y = max(0, min(new_y, fh - 1))
                        new_w = max(1, min(new_w, fw - new_x))
                        new_h = max(1, min(new_h, fh - new_y))

                        new_roi = (new_x, new_y, new_w, new_h)

                        # Re-create tracker and init
                        try:
                            tracker = cv2.legacy.TrackerCSRT_create()
                            tracker.init(temp_frame, new_roi)
                            trackers[i] = tracker
                            rois[i] = new_roi

                            # Kalman: correct with re-detected measurement (cx,cy,w,h)
                            cx = new_x + new_w / 2.0
                            cy = new_y + new_h / 2.0
                            measured = np.array([[np.float32(cx)], [np.float32(cy)], [np.float32(new_w)], [np.float32(new_h)]], dtype=np.float32)
                            if kalman_filters[i] is None:
                                kalman_filters[i] = create_kalman_from_roi(new_roi)
                            else:
                                kalman_filters[i].correct(measured)

                            # Update template
                            templ_gray, tsize = extract_template_from_frame(temp_frame, new_roi)
                            if templ_gray is not None and templ_gray.size > 0:
                                templates[i] = templ_gray
                                template_sizes[i] = tsize

                            reinit_success = True
                        except Exception as e:
                            # Reinit failed due to some error
                            reinit_success = False
                            reinit_fail[i] = True
                    else:
                        # template matching didn't find a high-confidence match
                        reinit_success = False
                        reinit_fail[i] = True
                else:
                    # no template available
                    reinit_success = False
                    reinit_fail[i] = True

            # Kalman prediction for visualization (use prediction if measurement missing)
            if kalman_filters[i] is None:
                # No Kalman: fallback to raw roi (may be invalid)
                vis_x, vis_y, vis_w, vis_h = map(int, rois[i])
            else:
                kf = kalman_filters[i]
                prediction = kf.predict()  # shape (6,1)
                pred_cx = float(prediction[0])
                pred_cy = float(prediction[1])
                pred_w = float(prediction[4])
                pred_h = float(prediction[5])
                vis_w = int(max(1, pred_w))
                vis_h = int(max(1, pred_h))
                vis_x = int(pred_cx - vis_w / 2.0)
                vis_y = int(pred_cy - vis_h / 2.0)

                # If we had a fresh measurement (measured not None and reinit_success or ok),
                # the Kalman was corrected above. The prediction will reflect corrected state.

            # Draw on frame: if last measurement ok or reinit_success -> green box, else yellow (predicted) or red on failure
            if (ok) or reinit_success:
                if frame_idx <= 15:  # avoid drawing smoothed box and trails in the first few frames
                    vis_x, vis_y, vis_w, vis_h = map(int, rois[i])
                # draw smoothed bbox (from kalman)
                cv2.rectangle(frame, (vis_x, vis_y), (vis_x + vis_w, vis_y + vis_h), colors[i], 2)
                cv2.putText(frame, f"Tracker {i+1}", (vis_x, max(12, vis_y - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2)

                # Update trail using smoothed center
                center = (vis_x + vis_w // 2, vis_y + vis_h // 2)
                trails[i].append(center)
                if len(trails[i]) > trail_length:
                    trails[i].pop(0)

                for t in range(1, len(trails[i])):
                    cv2.line(frame, trails[i][t-1], trails[i][t], colors[i], 2)

                reinit_fail[i] = False
            else:
                # Display failure messages
                # Show predicted box in a different color (yellow) to indicate uncertain position
                cv2.rectangle(frame, (vis_x, vis_y), (vis_x + vis_w, vis_y + vis_h), (0, 255, 255), 1)
                cv2.putText(frame, f"Tracker {i+1} LOST", (vis_x, max(12, vis_y - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

                # If reinit attempted but failed, show explicit message
                if reinit_fail[i]:
                    cv2.putText(frame, f"Tracker {i+1} RE-INIT FAILED", (20, 60 + i * 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # LOG to CSV: measured (if any) and smoothed (Kalman)
            if measured is not None:
                meas_cx = float(measured[0])
                meas_cy = float(measured[1])
                meas_w = float(measured[2])
                meas_h = float(measured[3])
                meas_x = meas_cx - meas_w / 2.0
                meas_y = meas_cy - meas_h / 2.0
            else:
                meas_x = meas_y = meas_w = meas_h = np.nan

            # Smoothed state from Kalman (if available)
            if kalman_filters[i] is not None:
                state = kalman_filters[i].statePost.flatten()
                smooth_cx = float(state[0])
                smooth_cy = float(state[1])
                smooth_w = float(state[4])
                smooth_h = float(state[5])
                smooth_x = smooth_cx - smooth_w / 2.0
                smooth_y = smooth_cy - smooth_h / 2.0
            else:
                smooth_x = vis_x
                smooth_y = vis_y
                smooth_w = vis_w
                smooth_h = vis_h

            csv_data.append({
                "frame": frame_idx,
                "tracker_id": i + 1,
                "meas_x": meas_x,
                "meas_y": meas_y,
                "meas_w": meas_w,
                "meas_h": meas_h,
                "smooth_x": smooth_x,
                "smooth_y": smooth_y,
                "smooth_w": smooth_w,
                "smooth_h": smooth_h,
                "ok": bool(ok),
                "reinit_attempted": bool(reinit_flags[i]),
                "reinit_success": bool(reinit_success),
                "reinit_failed": bool(reinit_fail[i])
            })

        frame_idx += 1

        # FPS display (timing for whole processing)
        elapsed_ms = (time.time() - start_time) * 1000
        fps = int(1000 / elapsed_ms) if elapsed_ms > 0 else 0
        cv2.putText(frame, f"FPS: {fps}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        # Write video & show
        video_writer.write(frame)
        cv2.imshow("Advanced Multi-Tracker (Kalman + Auto Re-init)", frame)

    # Keyboard controls (playback speed & pause)
    key = cv2.waitKey(int(max(1, (1000.0 / fps_video) / playback_speed)))

    if key == 27:  # ESC
        break
    elif key == 32:  # Spacebar = Pause/Resume
        paused = not paused
    elif key == 81:  # Left arrow (slower)
        playback_speed = max(0.1, playback_speed - 0.1)
    elif key == 83:  # Right arrow (faster)
        playback_speed += 0.1
    elif key in [ord("r"), ord("R")]:  # Reset speed
        playback_speed = initial_playback_speed

# ----------------------------------------------------------
# SAVE CSV
# ----------------------------------------------------------
df = pd.DataFrame(csv_data)
df.to_csv(output_csv_path, index=False)

print("Processing completed.")
print("Saved video:", output_video_path)
print("Saved CSV:", output_csv_path)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
