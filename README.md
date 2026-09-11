# 🎭 Reaction AI — Advanced Emotion Detection & Face Recognition Suite

A real-time emotion detection and face recognition system powered by **DeepFace**, **OpenCV**, and **face_recognition**. Features a cinematic HUD overlay with live analytics, multi-face comparison, session recording, and exportable reports.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ Features

### 🔍 Core Detection
- **Real-time emotion analysis** — Detects 7 emotions (Happy, Sad, Angry, Surprise, Fear, Disgust, Neutral) using deep neural networks
- **Face recognition** — Identifies known individuals from a pre-enrolled face database
- **Multi-face tracking** — Smooth temporal tracking with exponential filtering to eliminate jitter

### 📊 HUD & Analytics
- **Full Dashboard HUD** — Glassmorphism overlay with emotion breakdown bars, vibe meter, and rolling sparkline
- **Minimalist HUD** — Compact status badge for distraction-free monitoring
- **Multi-Face Comparison Panel** — Side-by-side emotion breakdown for every detected face with individual vibe gauges and sparklines
- **Group Vibe Index** — Averaged emotional valence across all tracked faces
- **Vibe Meter** — 0–100% emotional valence gauge (Upbeat → Calm → Tense)

### 🎥 Recording & Export
- **Video recording** — Capture sessions as MP4 with HUD overlay baked in
- **Snapshot capture** — Save individual frames as JPEG
- **Session analytics export** — JSON reports with emotion distribution, FPS stats, and session duration

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Webcam / camera device
- Windows / macOS / Linux

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/Face-Detection.git
cd Face-Detection

# Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install opencv-python numpy deepface face_recognition
```

> **Note:** `face_recognition` requires [dlib](http://dlib.net/). On Windows, you may need to install Visual Studio Build Tools or use `conda install -c conda-forge dlib` first.

### Enroll Known Faces

Place photos of people you want to identify in the `known_faces/` directory. The filename (without extension) becomes the person's display name.

```
known_faces/
├── keya.jpeg
├── john.png
└── sarah.jpg
```

### Run

```bash
# Advanced Emotion Detection Suite (recommended)
python reaction.py

# Basic Face Recognition (lightweight)
python face_det.py
```

---

## ⌨️ Keyboard Controls

| Key | Action |
|-----|--------|
| `Q` | Quit & save session summary |
| `H` | Cycle HUD mode (Full → Minimal → Hidden) |
| `C` | Toggle multi-face comparison panel |
| `V` | Start / stop video recording |
| `S` | Capture snapshot |
| `M` | Toggle selfie mirror mode |
| `R` | Toggle face identification |
| `E` | Export session analytics to JSON |

---

## 📁 Project Structure

```
Face-Detection/
├── reaction.py          # Main app — emotion detection + HUD + analytics
├── face_det.py          # Lightweight face recognition script
├── known_faces/         # Enrolled face images for identification
│   └── keya.jpeg
├── snapshots/           # Captured screenshots (auto-created)
├── recordings/          # Recorded video sessions (auto-created)
├── reports/             # Exported JSON analytics (auto-created)
├── LICENSE              # MIT License
└── README.md
```

---

## 🧠 How It Works

### Architecture

```
┌─────────────┐     ┌──────────────────────┐     ┌─────────────────┐
│   Webcam     │────▶│  Background Thread    │────▶│  Smooth Tracker  │
│   (30 FPS)   │     │  (DeepFace Inference)  │     │  (EMA Filter)    │
└─────────────┘     └──────────────────────┘     └────────┬────────┘
                                                          │
                              ┌────────────────────────────┘
                              ▼
                    ┌──────────────────┐
                    │   HUD Renderer    │
                    │  • Full Dashboard  │
                    │  • Minimalist      │
                    │  • Compare Panel   │
                    └──────────────────┘
```

1. **Capture** — Webcam frames at 640×480 resolution
2. **Detect** — Background daemon thread runs DeepFace emotion analysis + face_recognition on downsampled frames
3. **Track** — `SmoothFaceTracker` applies exponential moving average (EMA) to bounding boxes and emotion scores, eliminating jitter
4. **Render** — HUD overlays drawn on the main thread at full FPS with glassmorphism styling
5. **Compare** — Multi-face panel shows per-face emotion cards with individual sparklines and a group vibe summary

### Emotion Palette

| Emotion | Color | Valence Weight |
|---------|-------|---------------|
| 😄 Happy | Emerald Green | +1.0 |
| 😲 Surprise | Radiant Gold | +0.4 |
| 😐 Neutral | Silver White | 0.0 |
| 😢 Sad | Ocean Blue | −0.5 |
| 😡 Angry | Crimson Red | −1.0 |
| 😨 Fear | Royal Violet | −0.7 |
| 🤢 Disgust | Deep Coral | −0.8 |

---

## 🛠️ Dependencies

| Package | Purpose |
|---------|---------|
| [OpenCV](https://opencv.org/) | Camera capture, rendering, video I/O |
| [NumPy](https://numpy.org/) | Numerical operations |
| [DeepFace](https://github.com/serengil/deepface) | Emotion analysis via deep neural networks |
| [face_recognition](https://github.com/ageitgey/face_recognition) | Face identification (optional) |

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

