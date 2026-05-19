# HyoidTracking

A real-time multi-target tracking tool built for ultrasound imaging research, designed to track hyoid bone (and other anatomical structures) motion from a live screen-mirrored ultrasound feed. The tool provides a dual-window GUI — an **Experimenter View** for setup and control, and a **Participant View** with configurable overlays.

---

## Features

- **Screen-region capture** — mirror any region of the screen (e.g., an ultrasound display) as the video source
- **Multi-target tracking** — track up to 20 simultaneous ROIs using OpenCV CSRT trackers
- **Kalman filter smoothing** — optional Kalman filter (6-state: position, velocity, size) for noise reduction
- **ORB-based re-initialization** — automatically recovers lost trackers using ORB feature matching and homography estimation
- **Manual re-initialization** — draw a new ROI at any time to reset any tracker
- **Trail visualization** — per-tracker motion trails (last 40 frames)
- **Dual-window output**
  - Experimenter View: full annotated feed with bounding boxes and FPS
  - Participant View: configurable overlay with four modes (raw frame / black + gradient box / frame + gradient box / swallow strength meter)
- **Draggable gradient box** — drag the shaded reference box horizontally in both the Participant View and the Experimenter View; the two views stay in sync; double-click to reset
- **Box overlay on experimenter view** — optional checkbox mirrors the gradient box onto the main display
- **Adjustable shading** — control box width, opacity range, and number of shading steps
- **Swallow marking** — one-click toggle to record tracker trajectories for each swallow event; completed trails are drawn on the Participant View in distinct colors, with the active trial shown in white
- **Configurable trail history** — choose how many past swallow trajectories to display (1–20)
- **Participant view zoom** — optionally zoom the Participant View into the region around the gradient box; supports auto-zoom (based on box position) or a custom zoom region drawn on the Experimenter View
- **CSV export** — per-frame, per-tracker data: measured position, Kalman-smoothed position, re-init flags
- **Video recording** — saves the annotated feed to `tracked_output_gui.mp4`

---

## Requirements

- Python 3.9+
- [OpenCV](https://opencv.org/) with legacy trackers (`opencv-contrib-python`)
- [PySide6](https://doc.qt.io/qtforpython/)
- [NumPy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)

Install dependencies:

```bash
pip install opencv-contrib-python PySide6 numpy pandas
```

---

## Usage

```bash
python Scripts/muti_tracker_with_overlay_2.py
```

### Workflow

1. **Screen Mirror Region** — click and drag to select the screen region to capture (e.g., an ultrasound window)
2. **Select ROIs** — draw bounding boxes around the targets to track; tracking starts automatically
3. Use **Pause/Resume** to pause the feed at any time
4. Use **Overlay mode** slider (1–4) to switch the Participant View display:
   - `1` — Copy: mirrored frame only
   - `2` — Black+Box: black background with stepped-gradient reference box + tracker dots
   - `3` — Frame+Box: frame background with reference box + tracker dots
   - `4` — Strength Meter: real-time swallow strength bar showing horizontal excursion of the tracked target; displays current and past swallow markers with an auto-scaling gauge
5. Check **Move Box** to drag the reference box horizontally in either view (double-click to reset); enable **Show box on main view** to mirror it on the Experimenter View
6. Click **Adjust Shaded Box Width** or **Box Shading** to fine-tune the overlay appearance
7. **Swallow Marking** — click **Mark Swallow Start** when a swallow begins, then **Mark Swallow End** when it finishes; the trajectory is saved and drawn on the Participant View; use the **Show last N** spinner to control how many past swallows are displayed
8. **Zoom Participant View** — check **Zoom participant view** to zoom into the area around the gradient box; click **Set Zoom Region** to draw a custom zoom area on the Experimenter View, or **Reset to Auto** to revert to automatic zoom
9. **Manual Reinit** — select a tracker from the dropdown and draw a new ROI to reset it
10. **Export CSV** — save all tracking data to a `.csv` file

---

## Output Files

| File | Description |
|---|---|
| `tracked_output_gui.mp4` | Annotated video recording of the session |
| `tracking_data.csv` | Per-frame tracking results (measured + Kalman-smoothed) |
| `tracking_data_with_kalman.csv` | Tracking results from a Kalman-filtered run |

### CSV columns

| Column | Description |
|---|---|
| `frame` | Frame index |
| `tracker_id` | Tracker number (1-based) |
| `meas_x/y/w/h` | Raw tracker measurement (top-left x, y, width, height) |
| `smooth_x/y/w/h` | Kalman-smoothed position and size |
| `ok` | Whether the tracker succeeded on this frame |
| `reinit_attempted` | Whether ORB re-initialization was attempted |
| `reinit_success` | Whether re-initialization succeeded |
| `reinit_failed` | Whether re-initialization failed |

---

## Project Structure

```
HyoidTracking/
├── Scripts/
│   ├── muti_tracker_with_overlay_2.py   # Main GUI application
│   ├── tracking_data.csv                # Example tracking output
│   └── tracking_data_with_kalman.csv    # Example Kalman-filtered output
└── README.md
```

---

## Notes

- Default target FPS is 60; adjust `fps_video` in `VideoThread.__init__` if needed.
- The tracker uses a **local search region** around each ROI to improve speed and reduce drift.
