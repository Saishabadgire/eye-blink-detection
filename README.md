# Eye Blink Detection System

An end-to-end, real-time Computer Vision and Human-Computer Interaction application built in Python and OpenCV, based on the **Eye Blink Detection System Project Synopsis**.

---

## 📌 Project Overview & Synopsis Alignment

This implementation fulfills all objectives and architectural modules outlined in the academic synopsis:

| Synopsis Module / Section | Implementation in Project | Description |
|:---|:---|:---|
| **Module 1: Video Capture** | [`src/video_capture.py`](src/video_capture.py) | Reads live frames from webcam (device 0) or recorded video files with rolling FPS estimation. |
| **Module 2: Face Detection** | [`src/face_mesh.py`](src/face_mesh.py) | Locates face and 468 facial landmark mesh using MediaPipe, with an OpenCV Haar Cascade fallback engine. |
| **Module 3: Eye Landmark & State** | [`src/eye_detector.py`](src/eye_detector.py) | Calculates **Eye Aspect Ratio (EAR)** for canonical eye points; dynamic auto-calibration of open-eye baseline. |
| **Module 4: Blink Counting** | [`src/blink_counter.py`](src/blink_counter.py) | Finite state machine tracking open $\rightarrow$ closed $\rightarrow$ open transitions, counting blinks, and measuring closure duration. |
| **Module 5: Alert & Display** | [`src/display_overlay.py`](src/display_overlay.py) & [`src/audio_alert.py`](src/audio_alert.py) | Real-time heads-up display (HUD), eye contour rendering, EAR bar gauge, and asynchronous audio alert (`winsound`). |
| **Module 6: Testing & Evaluation** | [`src/evaluator.py`](src/evaluator.py) & [`evaluate.py`](evaluate.py) | Saves session logs to CSV, calculates precision, recall, F1-score, and blink-rate metrics against ground truth. |
| **Module 7: Interactive Interfaces** | [`main.py`](main.py) & [`app_streamlit.py`](app_streamlit.py) | Real-time OpenCV desktop app + modern Streamlit browser dashboard. |

---

## 📐 Mathematical Formulation (Eye Aspect Ratio - EAR)

The system estimates eye state using the **Eye Aspect Ratio (EAR)** proposed by Soukupová and Čech:

$$\text{EAR} = \frac{||p_2 - p_6|| + ||p_3 - p_5||}{2 \cdot ||p_1 - p_4||}$$

Where:
- $p_1, p_4$ are the horizontal eye corner points (outer and inner canthi).
- $p_2, p_6$ and $p_3, p_5$ are the pairs of vertical eyelid landmark points.

```
       p2        p3
        •--------•
       /          \
  p1 •              • p4    (EAR drops rapidly when eye closes)
       \          /
        •--------•
       p6        p5
```

- When eyes are **OPEN**, EAR is typically $\approx 0.28 - 0.35$.
- When eyes are **CLOSED** during a blink, EAR drops below $\approx 0.20 - 0.22$.
- If EAR remains below the threshold for more than the configurable duration (default: $1.5\text{s}$), a **PROLONGED CLOSURE (DROWSINESS) ALERT** is triggered.

---

## 📁 Project Structure

```
eye_blink_detection/
├── .vscode/
│   ├── launch.json              # 5 pre-configured VS Code Run/Debug profiles
│   └── settings.json            # VS Code Python environment & test runner settings
├── config/
│   ├── __init__.py
│   └── settings.py              # Thresholds, colors, landmark indices, sound settings
├── src/
│   ├── __init__.py
│   ├── video_capture.py         # Webcam / Video capture manager
│   ├── face_mesh.py             # MediaPipe FaceMesh & OpenCV Haar fallback
│   ├── eye_detector.py          # EAR math & dynamic calibration
│   ├── blink_counter.py         # Blink event state tracker & closure timer
│   ├── display_overlay.py       # OpenCV HUD overlay & alert banners
│   ├── audio_alert.py           # Non-blocking audio warning (winsound)
│   └── evaluator.py             # CSV logging & ground truth evaluation
├── tests/
│   ├── __init__.py
│   ├── test_ear.py              # Automated unit tests for EAR and state logic
│   └── generate_test_video.py   # Generates synthetic face video with scheduled blinks
├── data/
│   ├── logs/                    # Session CSV export files
│   └── sample_videos/           # Video files for offline testing & evaluation
├── main.py                      # Main desktop OpenCV real-time application
├── app_streamlit.py             # Web dashboard with interactive charts & upload
├── evaluate.py                  # Offline evaluation & benchmarking tool
├── requirements.txt             # Python package dependencies
└── README.md                    # Project documentation
```

---

## 🚀 Quick Start Guide

### 1. Installation

Ensure you have Python 3.10+ installed. In your terminal:

```bash
cd eye_blink_detection
python -m pip install -r requirements.txt
```

### 2. Running in VS Code

1. Open the project folder in **Visual Studio Code**.
2. Press `Ctrl + Shift + D` (or click the **Run and Debug** icon in the sidebar).
3. Select any of the pre-configured profiles from the dropdown:
   - **👁️ Run Live Webcam Detection**: Starts `main.py` using your webcam.
   - **🌐 Run Streamlit Dashboard**: Opens the web dashboard in your browser.
   - **🎬 Generate & Test Synthetic Video**: Creates a sample blinking video.
   - **📊 Run Benchmark Evaluation**: Runs benchmark metrics against recorded video.
   - **🧪 Run Unit Tests**: Executes unit tests with unittest.
4. Press `F5` to run!

---

## 💻 Running via Command Line

### Option A: Real-Time Live Webcam Detection

```bash
python main.py
```

#### Keyboard Shortcuts during live monitoring:
| Key | Action |
|:---|:---|
| `q` or `ESC` | Quit the application and export session summary |
| `r` | Reset blink counter and metrics |
| `c` | Trigger dynamic open-eye re-calibration |
| `+` / `-` | Dynamically adjust EAR threshold by $\pm 0.01$ |

#### CLI Options:
```bash
python main.py --source 0                    # Webcam device 0
python main.py --source path/to/video.mp4    # Analyze video file
python main.py --threshold 0.24              # Custom EAR threshold
python main.py --prolonged-sec 2.0           # 2-second closure limit
python main.py --no-audio                    # Disable audio beep
python main.py --no-calibration              # Disable auto-calibration
```

---

### Option B: Interactive Streamlit Web Dashboard

```bash
streamlit run app_streamlit.py
```

Features:
- Live video stream display with annotated eye contours and HUD.
- Real-time EAR signal time-series chart with threshold line.
- Live metric cards: Total Blinks, Blinks Per Minute (BPM), EAR value, Eye State.
- Configurable sliders for thresholds on the sidebar.
- Video file uploader to analyze recorded clips.

---

### Option C: Offline Video Evaluation & Benchmarking

```bash
# 1. Generate synthetic test video (contains 3 blinks + 1 prolonged closure)
python tests/generate_test_video.py

# 2. Run benchmark evaluation against ground truth
python evaluate.py --video data/sample_videos/synthetic_blinks.mp4 --save-plot
```

Sample Evaluation Output:
```text
=================================================================
                     EVALUATION RESULTS
=================================================================
 Total Frames Processed : 300
 Video Duration (s)     : 10.00s
 Processing Speed       : 38.4 FPS
 Detected Blinks        : 3
 Detected Blink Times(s): [2.03, 4.03, 6.03]
 Prolonged Closures     : 1
-----------------------------------------------------------------
 ACCURACY METRICS (VS GROUND TRUTH):
 Ground Truth Blinks    : 3
 True Positives (TP)    : 3
 False Positives (FP)   : 0
 False Negatives (FN)   : 0
 Precision              : 100.00%
 Recall                 : 100.00%
 F1-Score               : 100.00%
 Blink Count Error      : 0
=================================================================
```

---

## 🧪 Running Automated Unit Tests

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

All tests verify EAR calculations, threshold logic, blink state transitions, and prolonged closure alerting.

---

## 📊 Session Logs & CSV Export

Every session automatically records blink events in `data/logs/blink_session_<timestamp>.csv`:

```csv
event_id,timestamp_iso,relative_time_sec,duration_frames,duration_sec,min_ear
1,2026-11-07 17:28:10,2.033,6,0.200,0.062
2,2026-11-07 17:28:12,4.033,5,0.167,0.071
3,2026-11-07 17:28:14,6.033,7,0.233,0.058
```

---

## 📚 References

1. Soukupová, T., & Čech, J. (2016). *Real-Time Eye Blink Detection using Facial Landmarks*. 21st Computer Vision Winter Workshop.
2. OpenCV Documentation: https://docs.opencv.org/
3. Google MediaPipe Face Landmarker: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker
