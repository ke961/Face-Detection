import os
import sys
import time
import json
import collections
import threading
from datetime import datetime

# Windows console UTF-8 protection
if sys.platform == "win32":
    import io
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Suppress TensorFlow C++ informational logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import numpy as np
from deepface import DeepFace

# Optional face recognition support
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False


# Vibrant emotion styling & color palette (BGR format for OpenCV)
EMOTION_CONFIG = {
    "happy": {
        "color": (50, 205, 50),       # Emerald / Lime Green
        "label": "HAPPY",
        "tag": "JOYFUL",
        "valence_weight": 1.0
    },
    "surprise": {
        "color": (0, 215, 255),       # Radiant Gold / Yellow
        "label": "SURPRISE",
        "tag": "AMAZED",
        "valence_weight": 0.4
    },
    "neutral": {
        "color": (225, 225, 225),     # Silver / Soft White
        "label": "NEUTRAL",
        "tag": "CALM",
        "valence_weight": 0.0
    },
    "sad": {
        "color": (255, 144, 30),      # Ocean Blue
        "label": "SAD",
        "tag": "GLOOMY",
        "valence_weight": -0.5
    },
    "angry": {
        "color": (34, 34, 220),       # Vivid Crimson Red
        "label": "ANGRY",
        "tag": "INTENSE",
        "valence_weight": -1.0
    },
    "fear": {
        "color": (211, 0, 148),       # Royal Violet / Purple
        "label": "FEAR",
        "tag": "ANXIOUS",
        "valence_weight": -0.7
    },
    "disgust": {
        "color": (0, 140, 255),       # Deep Coral / Orange
        "label": "DISGUST",
        "tag": "AVERSION",
        "valence_weight": -0.8
    }
}


class BackgroundEmotionDetector:
    """
    Asynchronous emotion detection worker.
    Runs deep neural network inference on a daemon background thread so the
    webcam captures and renders smoothly at full 30+ FPS.
    """
    def __init__(self, known_faces_dir="known_faces"):
        self.lock = threading.Lock()
        self.latest_frame = None
        self.raw_results = []
        self.is_running = True
        self.is_processing = False
        self.enable_recognition = FACE_RECOGNITION_AVAILABLE

        # Load known face encodings if available
        self.known_face_encodings = []
        self.known_face_names = []
        if self.enable_recognition and os.path.isdir(known_faces_dir):
            self._load_known_faces(known_faces_dir)

        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def _load_known_faces(self, known_faces_dir):
        print(f"[INFO] Scanning '{known_faces_dir}' for known individuals...")
        for filename in os.listdir(known_faces_dir):
            if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                path = os.path.join(known_faces_dir, filename)
                try:
                    img = face_recognition.load_image_file(path)
                    encs = face_recognition.face_encodings(img)
                    if encs:
                        self.known_face_encodings.append(encs[0])
                        self.known_face_names.append(os.path.splitext(filename)[0])
                        print(f"       + Enrolled face: {filename}")
                except Exception as e:
                    print(f"       - Failed to load {filename}: {e}")
        print(f"[INFO] Enrolled {len(self.known_face_encodings)} known face(s).")

    def update_frame(self, frame):
        with self.lock:
            self.latest_frame = frame.copy()

    def get_latest_detections(self):
        with self.lock:
            return list(self.raw_results)

    def toggle_recognition(self):
        self.enable_recognition = not self.enable_recognition
        return self.enable_recognition

    def _worker_loop(self):
        while self.is_running:
            frame_to_process = None
            with self.lock:
                if self.latest_frame is not None:
                    frame_to_process = self.latest_frame
                    self.latest_frame = None

            if frame_to_process is None:
                time.sleep(0.01)
                continue

            try:
                self.is_processing = True
                h, w = frame_to_process.shape[:2]

                # Downsample frame for fast, responsive inference
                target_w = 480
                if w > target_w:
                    scale = target_w / float(w)
                    small_img = cv2.resize(frame_to_process, (int(w * scale), int(h * scale)))
                else:
                    scale = 1.0
                    small_img = frame_to_process

                inv_scale = 1.0 / scale

                # DeepFace emotion analysis
                raw_results = DeepFace.analyze(
                    small_img,
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend='opencv',
                    silent=True
                )

                if not isinstance(raw_results, list):
                    raw_results = [raw_results]

                rgb_small = None
                if self.enable_recognition and self.known_face_encodings:
                    rgb_small = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)

                parsed_faces = []
                for r in raw_results:
                    region = r.get("region", {})
                    rw = region.get("w", 0)
                    rh = region.get("h", 0)

                    # Reject tiny ghost artifacts
                    if rw < 15 or rh < 15:
                        continue

                    rx = int(region.get("x", 0) * inv_scale)
                    ry = int(region.get("y", 0) * inv_scale)
                    box_w = int(rw * inv_scale)
                    box_h = int(rh * inv_scale)

                    dominant = r.get("dominant_emotion", "neutral")
                    emotions = {k: float(v) for k, v in r.get("emotion", {}).items()}
                    confidence = float(r.get("face_confidence", 0.0))

                    # Identify face identity against known dataset
                    name = None
                    if rgb_small is not None:
                        sx = max(0, region.get("x", 0))
                        sy = max(0, region.get("y", 0))
                        sw = min(rgb_small.shape[1] - sx, rw)
                        sh = min(rgb_small.shape[0] - sy, rh)

                        if sw > 20 and sh > 20:
                            loc = (sy, sx + sw, sy + sh, sx)
                            try:
                                encs = face_recognition.face_encodings(rgb_small, [loc])
                                if encs:
                                    dists = face_recognition.face_distance(self.known_face_encodings, encs[0])
                                    if len(dists) > 0:
                                        best_match = np.argmin(dists)
                                        if dists[best_match] < 0.55:
                                            name = self.known_face_names[best_match]
                            except Exception:
                                pass

                    parsed_faces.append({
                        "box": (rx, ry, box_w, box_h),
                        "dominant_emotion": dominant,
                        "emotion_scores": emotions,
                        "confidence": confidence,
                        "name": name
                    })

                with self.lock:
                    self.raw_results = parsed_faces

            except Exception:
                pass
            finally:
                self.is_processing = False

    def stop(self):
        self.is_running = False


class SmoothFaceTracker:
    """
    Temporal tracking & exponential smoothing filter.
    Eliminates bounding box jitter and score flicker for a cinema-grade experience.
    """
    def __init__(self, alpha_box=0.42, alpha_score=0.28):
        self.alpha_box = alpha_box
        self.alpha_score = alpha_score
        self.tracked_faces = {}
        self.next_id = 1

    def update(self, detected_faces):
        now = time.time()
        updated_ids = set()

        for det in detected_faces:
            dbx, dby, dbw, dbh = det["box"]
            dcx, dcy = dbx + dbw / 2.0, dby + dbh / 2.0

            # Match with closest active tracked face
            best_id = None
            best_dist = 140.0

            for fid, tface in self.tracked_faces.items():
                if fid in updated_ids:
                    continue
                tx, ty, tw, th = tface["box"]
                tcx, tcy = tx + tw / 2.0, ty + th / 2.0
                dist = np.hypot(dcx - tcx, dcy - tcy)
                if dist < best_dist:
                    best_dist = dist
                    best_id = fid

            if best_id is not None:
                # Smooth position
                tface = self.tracked_faces[best_id]
                old_x, old_y, old_w, old_h = tface["box"]
                tface["box"] = (
                    int(self.alpha_box * dbx + (1.0 - self.alpha_box) * old_x),
                    int(self.alpha_box * dby + (1.0 - self.alpha_box) * old_y),
                    int(self.alpha_box * dbw + (1.0 - self.alpha_box) * old_w),
                    int(self.alpha_box * dbh + (1.0 - self.alpha_box) * old_h),
                )

                # Smooth emotion scores
                old_scores = tface["emotion_scores"]
                new_scores = {}
                for k, v in det["emotion_scores"].items():
                    old_v = old_scores.get(k, v)
                    new_scores[k] = float(self.alpha_score * v + (1.0 - self.alpha_score) * old_v)

                tface["emotion_scores"] = new_scores
                tface["dominant_emotion"] = max(new_scores.items(), key=lambda item: item[1])[0]
                tface["last_seen"] = now
                if det.get("name"):
                    tface["name"] = det["name"]
                updated_ids.add(best_id)
            else:
                # Register new face
                new_id = self.next_id
                self.next_id += 1
                self.tracked_faces[new_id] = {
                    "id": new_id,
                    "box": (dbx, dby, dbw, dbh),
                    "dominant_emotion": det["dominant_emotion"],
                    "emotion_scores": dict(det["emotion_scores"]),
                    "name": det.get("name"),
                    "confidence": det.get("confidence", 0.0),
                    "last_seen": now
                }
                updated_ids.add(new_id)

        # Prune faces not seen for > 1.2s
        stale = [fid for fid, t in self.tracked_faces.items() if now - t["last_seen"] > 1.2]
        for fid in stale:
            del self.tracked_faces[fid]

        return list(self.tracked_faces.values())


def compute_vibe_index(scores):
    """
    Compute a 0-100% emotional valence (Vibe Index).
    50% = Perfectly Neutral / Centered
    > 75% = High Joy / Upbeat
    < 35% = Negative / Agitated
    """
    valence = 0.0
    for emo, cfg in EMOTION_CONFIG.items():
        weight = cfg.get("valence_weight", 0.0)
        valence += (scores.get(emo, 0.0) / 100.0) * weight

    vibe = 50.0 + (valence * 50.0)
    return float(np.clip(vibe, 0.0, 100.0))


def draw_cyber_corners(img, x, y, w, h, color, thickness=2, corner_len=20):
    """Draw futuristic HUD corner reticles around a face."""
    # Outer thin bounding box
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)

    # Top-Left corner
    cv2.line(img, (x, y), (x + corner_len, y), color, thickness)
    cv2.line(img, (x, y), (x, y + corner_len), color, thickness)

    # Top-Right corner
    cv2.line(img, (x + w, y), (x + w - corner_len, y), color, thickness)
    cv2.line(img, (x + w, y), (x + w, y + corner_len), color, thickness)

    # Bottom-Left corner
    cv2.line(img, (x, y + h), (x + corner_len, y + h), color, thickness)
    cv2.line(img, (x, y + h), (x, y + h - corner_len), color, thickness)

    # Bottom-Right corner
    cv2.line(img, (x + w, y + h), (x + w - corner_len, y + h), color, thickness)
    cv2.line(img, (x + w, y + h), (x + w, y + h - corner_len), color, thickness)

    # Subtle target center cross
    cx, cy = x + w // 2, y + h // 2
    cv2.line(img, (cx - 6, cy), (cx + 6, cy), color, 1)
    cv2.line(img, (cx, cy - 6), (cx, cy + 6), color, 1)


def draw_sparkline(frame, x, y, w, h, history_deque, line_color=(0, 255, 200), bg_color=(25, 25, 30)):
    """Render a real-time rolling sparkline trend graph on the HUD."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), bg_color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (55, 55, 60), 1)

    # Midline representing 50% neutral
    mid_y = y + h // 2
    cv2.line(frame, (x, mid_y), (x + w, mid_y), (50, 50, 55), 1)

    if len(history_deque) < 2:
        return

    pts = []
    n = len(history_deque)
    for i, val in enumerate(history_deque):
        px = int(x + (i / max(1, n - 1)) * w)
        py = int(y + h - (val / 100.0) * h)
        py = int(np.clip(py, y + 2, y + h - 2))
        pts.append((px, py))

    pts_arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts_arr], False, line_color, 2, cv2.LINE_AA)
    cv2.circle(frame, pts[-1], 3, (255, 255, 255), -1)


def draw_full_hud(frame, dominant_emotion, scores, fps, face_count, primary_name, rec_enabled, vibe, sparkline_data):
    """Render the master Glassmorphism HUD overlay with analytics & breakdown bars."""
    panel_w = 280
    panel_h = 395
    x0, y0 = 16, 16

    # Glassmorphism dark background with alpha blend
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (16, 16, 20), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (60, 60, 70), 1)

    # Title & FPS Counter
    cv2.putText(frame, "REACTION AI", (x0 + 12, y0 + 26), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1)
    cv2.putText(frame, f"FPS: {fps:.0f}", (x0 + panel_w - 74, y0 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 180), 1)

    # Dominant Badge
    cfg = EMOTION_CONFIG.get(dominant_emotion.lower(), EMOTION_CONFIG["neutral"])
    badge_color = cfg["color"]
    cv2.rectangle(frame, (x0 + 12, y0 + 38), (x0 + panel_w - 12, y0 + 68), badge_color, -1)

    if face_count > 0:
        if primary_name:
            status_text = f"{primary_name.capitalize()}: {cfg['label']}"
        else:
            status_text = f"Mood: {cfg['label']} ({cfg['tag']})"
    else:
        status_text = "Scanning for faces..."

    text_color = (0, 0, 0) if dominant_emotion.lower() in ["happy", "surprise", "neutral"] else (255, 255, 255)
    cv2.putText(frame, status_text, (x0 + 18, y0 + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

    # Vibe Meter Gauge
    vibe_y = y0 + 88
    cv2.putText(frame, f"VIBE METER: {vibe:.0f}%", (x0 + 12, vibe_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

    # Vibe color interpolation
    if vibe > 70:
        vibe_color = (50, 205, 50)
        vibe_title = "Vibrant / Upbeat"
    elif vibe >= 45:
        vibe_color = (220, 220, 220)
        vibe_title = "Centered / Calm"
    else:
        vibe_color = (34, 34, 220)
        vibe_title = "Intense / Tense"

    vibe_bar_w = panel_w - 24
    cv2.rectangle(frame, (x0 + 12, vibe_y + 6), (x0 + 12 + vibe_bar_w, vibe_y + 16), (40, 40, 45), -1)
    vibe_fill = int(vibe_bar_w * (vibe / 100.0))
    if vibe_fill > 0:
        cv2.rectangle(frame, (x0 + 12, vibe_y + 6), (x0 + 12 + vibe_fill, vibe_y + 16), vibe_color, -1)

    # Emotion Breakdown Progress Bars
    order = ["happy", "neutral", "surprise", "sad", "angry", "fear", "disgust"]
    bar_y = vibe_y + 36
    for emo in order:
        score = scores.get(emo, 0.0)
        ecfg = EMOTION_CONFIG.get(emo, EMOTION_CONFIG["neutral"])
        color = ecfg["color"]

        cv2.putText(frame, f"{emo.capitalize():<8}", (x0 + 12, bar_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (190, 190, 190), 1)

        bar_x = x0 + 82
        bar_max_w = panel_w - 150
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_max_w, bar_y + 12), (40, 40, 45), -1)

        fill_w = int(bar_max_w * (score / 100.0))
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + 12), color, -1)

        cv2.putText(frame, f"{score:4.1f}%", (bar_x + bar_max_w + 8, bar_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 230, 230), 1)
        bar_y += 22

    # Emotion Trend Sparkline
    spark_y = bar_y + 8
    cv2.putText(frame, "EMOTIONAL TRAJECTORY (TIMELINE)", (x0 + 12, spark_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 170, 180), 1)
    draw_sparkline(frame, x0 + 12, spark_y + 6, panel_w - 24, 38, sparkline_data, line_color=vibe_color)

    # Bottom Status Bar
    footer_y = y0 + panel_h - 10
    rec_str = "ACTIVE" if rec_enabled else "OFF"
    cv2.putText(frame, f"Face ID: {rec_str}", (x0 + 12, footer_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 150), 1)
    cv2.putText(frame, f"Tracked: {face_count}", (x0 + panel_w - 85, footer_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 150), 1)


def draw_minimal_hud(frame, fps, face_count, dominant_emotion, vibe):
    """Compact minimalist HUD mode."""
    h, w = frame.shape[:2]
    badge_w = 280
    badge_h = 36
    x0, y0 = 16, 16

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + badge_w, y0 + badge_h), (16, 16, 20), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x0 + badge_w, y0 + badge_h), (60, 60, 70), 1)

    cfg = EMOTION_CONFIG.get(dominant_emotion.lower(), EMOTION_CONFIG["neutral"])
    cv2.circle(frame, (x0 + 16, y0 + 18), 6, cfg["color"], -1)
    cv2.putText(frame, f"{cfg['label']} ({vibe:.0f}%)", (x0 + 30, y0 + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
    cv2.putText(frame, f"FPS: {fps:.0f}", (x0 + badge_w - 65, y0 + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 180), 1)


def draw_bottom_ribbon(frame, is_recording, is_mirrored):
    """Render keyboard controls guide & system state ribbon."""
    h, w = frame.shape[:2]
    ribbon_text = "[Q] Quit | [H] HUD Mode | [V] Record | [S] Snap | [M] Mirror | [R] Face ID | [E] Export"
    cv2.putText(frame, ribbon_text, (20, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)


def save_session_analytics(session_data, export_dir="reports"):
    """Export detailed session analytics report to JSON."""
    os.makedirs(export_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(export_dir, f"reaction_session_{timestamp}.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=4)
        print(f"[INFO] Session analytics successfully saved to: {path}")
        return path
    except Exception as e:
        print(f"[ERROR] Failed to save session analytics: {e}")
        return None


def main():
    print("=" * 65)
    print("      ADVANCED EMOTION DETECTION & REACTION AI SUITE")
    print("=" * 65)
    print("[INFO] Connecting to camera feed...")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Please verify your camera device connection.")
        return

    # 640x480 resolution ensures fluid 30-60 FPS performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Initialize asynchronous background detector & smooth tracker
    detector = BackgroundEmotionDetector(known_faces_dir="known_faces")
    tracker = SmoothFaceTracker()

    # Session State
    hud_mode = 0  # 0: Full Dashboard, 1: Minimalist, 2: Hidden
    is_mirrored = True
    is_recording = False
    video_writer = None
    recording_start = 0.0

    toast_message = ""
    toast_expiry = 0.0

    sparkline_data = collections.deque(maxlen=80)
    emotion_counter = collections.Counter()

    prev_time = time.time()
    fps = 30.0
    start_time = time.time()
    frame_counter = 0

    print("[INFO] Engine started successfully.")
    print("       [Q] Quit & Save Summary    [H] Cycle HUD Mode")
    print("       [V] Start/Stop Recording   [S] Capture Snapshot")
    print("       [M] Toggle Selfie Mirror   [R] Toggle Face Identification")
    print("       [E] Export Session Report")
    print("-" * 65)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Camera frame capture failed.")
                break

            frame_counter += 1

            # Flip horizontal for natural mirror view
            if is_mirrored:
                frame = cv2.flip(frame, 1)

            # Measure real-time FPS
            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            # Update background inference thread
            detector.update_frame(frame)

            # Run smooth tracking filter on latest detections
            raw_detections = detector.get_latest_detections()
            tracked_faces = tracker.update(raw_detections)

            primary_emotion = "neutral"
            primary_scores = {}
            primary_name = None

            # Render tracked faces
            for face in tracked_faces:
                rx, ry, rw, rh = face["box"]
                dominant = face["dominant_emotion"]
                scores = face["emotion_scores"]
                name = face["name"]

                emotion_counter[dominant] += 1

                if not primary_scores:
                    primary_emotion = dominant
                    primary_scores = scores
                    primary_name = name

                cfg = EMOTION_CONFIG.get(dominant.lower(), EMOTION_CONFIG["neutral"])
                color = cfg["color"]

                # Draw high-tech corner box reticle
                draw_cyber_corners(frame, rx, ry, rw, rh, color)

                # Format dominant emotion label badge
                conf = scores.get(dominant, 0.0)
                if name and detector.enable_recognition:
                    badge_label = f"{name.capitalize()} | {cfg['label']} ({conf:.0f}%)"
                else:
                    badge_label = f"{cfg['label']} ({conf:.0f}%)"

                (tw, th), _ = cv2.getTextSize(badge_label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
                tag_y1 = max(0, ry - th - 12)
                tag_y2 = max(th + 12, ry)
                cv2.rectangle(frame, (rx, tag_y1), (rx + tw + 14, tag_y2), color, -1)

                text_c = (0, 0, 0) if dominant.lower() in ["happy", "surprise", "neutral"] else (255, 255, 255)
                cv2.putText(frame, badge_label, (rx + 7, tag_y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.52, text_c, 2)

            # Compute Vibe / Valence Index
            vibe_val = compute_vibe_index(primary_scores)
            sparkline_data.append(vibe_val)

            # HUD Display Rendering
            if hud_mode == 0:
                draw_full_hud(
                    frame,
                    primary_emotion,
                    primary_scores,
                    fps,
                    len(tracked_faces),
                    primary_name,
                    detector.enable_recognition,
                    vibe_val,
                    sparkline_data
                )
            elif hud_mode == 1:
                draw_minimal_hud(
                    frame,
                    fps,
                    len(tracked_faces),
                    primary_emotion,
                    vibe_val
                )

            # Bottom Ribbon Guide
            draw_bottom_ribbon(frame, is_recording, is_mirrored)

            # Active Recording Indicator
            if is_recording:
                rec_secs = int(time.time() - recording_start)
                rec_mins = rec_secs // 60
                rec_rem = rec_secs % 60
                rec_text = f"REC  {rec_mins:02d}:{rec_rem:02d}"

                # Pulsing red dot
                dot_blink = (int(time.time() * 2) % 2 == 0)
                if dot_blink:
                    cv2.circle(frame, (frame.shape[1] - 125, 28), 7, (0, 0, 255), -1)
                cv2.putText(frame, rec_text, (frame.shape[1] - 110, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 255), 2)

                # Write frame to video
                if video_writer is not None:
                    video_writer.write(frame)

            # Toast Notification
            if toast_message and time.time() < toast_expiry:
                (tw, th), _ = cv2.getTextSize(toast_message, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                toast_x = (frame.shape[1] - tw) // 2
                cv2.rectangle(frame, (toast_x - 12, 16), (toast_x + tw + 12, 16 + th + 18), (20, 20, 24), -1)
                cv2.rectangle(frame, (toast_x - 12, 16), (toast_x + tw + 12, 16 + th + 18), (0, 255, 120), 1)
                cv2.putText(frame, toast_message, (toast_x, 16 + th + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 120), 2)

            # Display Window
            cv2.imshow("Reaction AI - Advanced Emotion Suite", frame)

            # Keyboard Input Handling
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('h'):
                hud_mode = (hud_mode + 1) % 3
                mode_names = ["Full Dashboard", "Minimalist", "Hidden"]
                toast_message = f"HUD: {mode_names[hud_mode]}"
                toast_expiry = time.time() + 1.2

            elif key == ord('m'):
                is_mirrored = not is_mirrored
                toast_message = f"Mirror: {'ON' if is_mirrored else 'OFF'}"
                toast_expiry = time.time() + 1.2

            elif key == ord('r'):
                enabled = detector.toggle_recognition()
                toast_message = f"Face ID: {'Enabled' if enabled else 'Disabled'}"
                toast_expiry = time.time() + 1.5

            elif key == ord('s'):
                # Capture snapshot
                os.makedirs("snapshots", exist_ok=True)
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                snap_path = os.path.join("snapshots", f"reaction_{stamp}.jpg")
                cv2.imwrite(snap_path, frame)
                toast_message = f"Saved: {os.path.basename(snap_path)}"
                toast_expiry = time.time() + 2.0
                print(f"[INFO] Snapshot saved to {snap_path}")

            elif key == ord('v'):
                # Toggle Video Recording
                if not is_recording:
                    os.makedirs("recordings", exist_ok=True)
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    rec_path = os.path.join("recordings", f"reaction_{stamp}.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(rec_path, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
                    is_recording = True
                    recording_start = time.time()
                    toast_message = "Recording Started"
                    toast_expiry = time.time() + 1.5
                    print(f"[INFO] Started recording: {rec_path}")
                else:
                    is_recording = False
                    if video_writer is not None:
                        video_writer.release()
                        video_writer = None
                    toast_message = "Recording Saved"
                    toast_expiry = time.time() + 2.0
                    print("[INFO] Video recording stopped and finalized.")

            elif key == ord('e'):
                # Export Session Analytics
                elapsed = time.time() - start_time
                summary = {
                    "timestamp": datetime.now().isoformat(),
                    "duration_seconds": round(elapsed, 1),
                    "total_frames": frame_counter,
                    "avg_fps": round(frame_counter / max(1.0, elapsed), 1),
                    "emotion_distribution": dict(emotion_counter),
                    "current_vibe_score": round(vibe_val, 1)
                }
                rep_path = save_session_analytics(summary)
                if rep_path:
                    toast_message = "Analytics Exported"
                    toast_expiry = time.time() + 2.0

    finally:
        # Final cleanup & save report
        print("[INFO] Finalizing session and releasing hardware resources...")
        detector.stop()
        if video_writer is not None:
            video_writer.release()
        cap.release()
        cv2.destroyAllWindows()

        # Generate automatic exit session summary
        elapsed = time.time() - start_time
        if frame_counter > 10:
            summary = {
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": round(elapsed, 1),
                "total_frames": frame_counter,
                "avg_fps": round(frame_counter / max(1.0, elapsed), 1),
                "emotion_distribution": dict(emotion_counter)
            }
            save_session_analytics(summary)

        print("[INFO] Session terminated cleanly.")


if __name__ == "__main__":
    main()
