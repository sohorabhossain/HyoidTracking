# HyoidTracking

A real-time multi-target tracking tool built for ultrasound imaging research, designed to track hyoid bone (and other anatomical structures) motion from a live screen-mirrored ultrasound feed. The tool provides a dual-window GUI — an **Experimenter View** for setup and control, and a **Participant View** with configurable overlays.

---

## Features

- **Screen-region capture** — mirror any region of the screen (e.g., an ultrasound display) as the video source
- **Multi-target tracking** — track up to 20 simultaneous ROIs using OpenCV CSRT trackers
- **Kalman filter smoothing** — optional Kalman filter (6-state: position, velocity, size) for noise reduction
- **ORB-based re-initialization** — automatically recovers lost trackers using ORB feature matching and homography estimation
- **Manual re-initialization** — draw a new ROI at any time to reset any tracker; the tracker selection dropdown and Reinit button are grouped together near the top of the control panel for quick access
- **Trail visualization** — per-tracker motion trails (last 40 frames)
- **Dual-window output**
  - Experimenter View: full annotated feed with bounding boxes and FPS; control panel is scrollable so all controls are always reachable regardless of window height
  - Participant View: configurable overlay with five modes (raw frame / black + gradient box / frame + gradient box / swallow strength meter / swallow speedometer)
- **Draggable gradient box** — drag the shaded reference box horizontally in both the Participant View and the Experimenter View; the two views stay in sync; double-click to reset
- **Box overlay on experimenter view** — optional checkbox mirrors the gradient box onto the main display
- **Adjustable shading** — control box width, opacity range, and number of shading steps
- **Swallow marking** — one-click toggle to record tracker trajectories for each swallow event; completed trails are drawn on the Participant View in distinct colors, with the active trial shown in white
- **Configurable trail history** — choose how many past swallow trajectories to display (1–20); clear all recorded trajectories at any time with the **Clear Trajectories** button
- **Strength metric selector** — choose between **Displacement** (straight-line distance from start to end; default) or **Arc Length** (cumulative frame-to-frame path length) for the Mode 4 strength meter; each metric maintains its own independently auto-scaling range (Displacement starts at 30 px, Arc Length at 500 px)
- **Scale settings** — per-metric sliders to manually set the maximum scale range for Mode 4 (Displacement: 1–500 px; Arc Length: 1–5000 px) and Mode 5 (Speed: 100–10000 px/s); independent auto-expand checkboxes for the strength scale and speed scale
- **Participant label toggle** — **Show participant labels** checkbox instantly hides or reveals all non-title overlays on the participant screen in Mode 4 and 5 (scale tick marks and values, live/peak readout, metric label, swallow count, LIVE badge); the mode title ("SWALLOW STRENGTH" / "SWALLOW SPEED") is always shown
- **Participant view zoom** — optionally zoom the Participant View into the region around the gradient box; supports auto-zoom (based on box position) or a custom zoom region drawn on the Experimenter View
- **Tracker dot color feedback (modes 2 & 3)** — tracker circles start red; once a circle enters or passes to the left of the gradient box and remains there for 3 continuous seconds (without exiting through the right side), it turns yellow; the circle reverts to red as soon as it moves back to the right of the box
- **CSV export** — per-frame, per-tracker data: measured position, Kalman-smoothed position, re-init flags
- **Video recording** — saves the annotated feed to `tracked_output_gui.mp4`
- **Keyboard shortcuts** — common actions are accessible without reaching for the mouse (see table below)

---

## Keyboard Shortcuts

| Shortcut | Action |
|---|---|
| `Ctrl+I` | Select ROIs |
| `Ctrl+T` | Start Tracking |
| `Ctrl+R` | Reinit Selected Tracker |
| `Ctrl+S` | Mark Swallow Start / End (toggle) |
| `Ctrl+P` | Pause / Resume |
| `Ctrl+C` | Clear Trajectories |

All shortcuts are also shown in the corresponding button labels in the Experimenter View.

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
python Scripts/multi_tracker_with_overlay_2.py
```

### Workflow

1. **Screen Mirror Region** — click and drag to select the screen region to capture (e.g., an ultrasound window)
2. **Select ROIs** (`Ctrl+I`) — draw bounding boxes around the targets to track; tracking starts automatically
3. **Manual Reinit** — select a tracker from the dropdown and click **Reinit Selected (draw)** (`Ctrl+R`) to draw a new ROI and reset that tracker; these controls are grouped directly below Start Tracking for quick access
4. Use **Pause/Resume** (`Ctrl+P`) to pause the feed at any time
5. Use the **Overlay mode** slider (1–5) to switch the Participant View display:
   - `1` — Copy: mirrored frame only
   - `2` — Black+Box: black background with stepped-gradient reference box + tracker dots (dots turn **yellow** after 3 s inside/left of box; revert to **red** on exit to the right)
   - `3` — Frame+Box: frame background with reference box + tracker dots (same color-feedback logic as mode 2)
   - `4` — Strength Meter: real-time swallow strength bar showing target excursion
     - Color gradient runs **red → green** (low → high strength)
     - Select metric: **Displacement** (default, start-to-end distance) or **Arc Length** (total path length)
     - Current metric is shown below the pixel readout on the bar (`Disp.` or `Arc Len.`)
     - Past swallow markers shown on the left of the bar
   - `5` — Speedometer: circular gauge showing peak swallow speed (px/s)
     - Color gradient runs **red → green** (slow → fast)
     - Updates live during an active swallow; holds the last peak value between swallows
6. Check **Move Box** to drag the reference box horizontally in either view (double-click to reset); enable **Show box on main view** to mirror it on the Experimenter View
7. Click **Adjust Shaded Box Width** or **Box Shading** to fine-tune the overlay appearance
8. **Swallow Marking** — click **Mark Swallow Start** (`Ctrl+S`) when a swallow begins, then **Mark Swallow End** (`Ctrl+S`) when it finishes; the trajectory is saved and drawn on the Participant View; use the **Show last N** spinner to control how many past swallows are displayed; click **Clear Trajectories** (`Ctrl+C`) to remove all saved trails
9. **Strength metric** — use the **Strength metric** dropdown to switch between Displacement and Arc Length; the bar and scale update immediately
10. **Scale Settings (Mode 4 & 5)**:
    - **Auto-expand strength scale** — when checked, the strength scale automatically expands (×1.2 headroom) if a swallow exceeds the current maximum; uncheck to lock the scale to the slider value
    - **Disp. max** slider — set the Displacement scale ceiling (1–500 px; default 30)
    - **Arc max** slider — set the Arc Length scale ceiling (1–5000 px; default 500)
    - **Auto-expand speed scale** — same auto-expand behaviour for the Mode 5 speedometer
    - **Speed max** slider — set the speed scale ceiling (100–10000 px/s; default 2500)
11. **Participant labels** — uncheck **Show participant labels** to hide all non-title overlays on the participant screen (scale marks, numeric values, swallow count, LIVE badge) while keeping the mode title visible; re-check to restore them
12. **Zoom Participant View** — check **Zoom participant view** to zoom into the area around the gradient box; click **Set Zoom Region** to draw a custom zoom area on the Experimenter View, or **Reset to Auto** to revert to automatic zoom
13. **Export CSV** — save all tracking data to a `.csv` file

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
- All scale values (strength and speed) are in **frame pixels** or **frame pixels per second** — no real-world calibration is applied.
- Strength metric and scale settings persist for the duration of the session but reset to defaults on restart.
