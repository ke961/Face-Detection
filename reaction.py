import os
import sys
import time
import threading
from datetime import datetime

# Configure Windows console to support UTF-8 and avoid charmap encoding crashes
if sys.platform == "win32":
    import io
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Suppress TensorFlow C++ informational & warning logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import numpy as np
from deepface import DeepFace

# Optional face recognition support if known_faces exists
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False


# Vibrant color palette for each emotion (BGR format for OpenCV)
EMOTION_COLORS = {
    "happy": (50, 205, 50),       # Emerald / Lime Green
    "surprise": (0, 215, 255),    # Vibrant Gold / Yellow
    "neutral": (220, 220, 220),   # Crisp Silver / Cyan
    "sad": (255, 144, 30),        # Ocean Blue
    "angry": (34, 34, 220),       # Vivid Crimson Red
    "fear": (211, 0, 148),        # Violet / Purple
    "disgust": (0, 140, 255),     # Coral / Orange
}


class BackgroundEmotionDetector:
    """
    Asynchronous emotion detection worker.
    Processes camera frames in a background daemon thread so the
    OpenCV video display runs at full camera FPS without freezing.
    """
    def __init__(self, known_faces_dir="known_faces"):
        self.lock = threading.Lock()
        self.latest_frame = None
        self.results = []
        self.is_running = True
        self.is_processing = False
        self.enable_recognition = FACE_RECOGNITION_AVAILABLE

        # Load known faces if face_recognition is available and directory exists
        self.known_face_encodings = []
        self.known_face_names = []
        if self.enable_recognition and os.path.isdir(known_faces_dir):
            self._load_known_faces(known_faces_dir)

        # Start worker thread
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def _load_known_faces(self, known_faces_dir):
        print(f"[INFO] Loading known faces from '{known_faces_dir}'...")
        for filename in os.listdir(known_faces_dir):
            if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                path = os.path.join(known_faces_dir, filename)
                try:
                    img = face_recognition.load_image_file(path)
                    encs = face_recognition.face_encodings(img)
                    if encs:
                        self.known_face_encodings.append(encs[0])
                        self.known_face_names.append(os.path.splitext(filename)[0])
                        print(f"       + Loaded: {filename}")
                except Exception as e:
                    print(f"       - Failed to load {filename}: {e}")
        print(f"[INFO] Total known faces loaded: {len(self.known_face_encodings)}")

    def update_frame(self, frame):
        """Pass the latest camera frame to the background worker."""
        with self.lock:
            self.latest_frame = frame.copy()

    def get_results(self):
        """Get the latest detected faces and emotion predictions."""
        with self.lock:
            return list(self.results)

    def toggle_recognition(self):
        """Toggle known face identification on/off."""
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

                # Downsample slightly for faster inference if frame is large
                target_w = 480
                if w > target_w:
                    scale = target_w / float(w)
                    small_img = cv2.resize(frame_to_process, (int(w * scale), int(h * scale)))
                else:
                    scale = 1.0
                    small_img = frame_to_process

                inv_scale = 1.0 / scale

                # Analyze emotion using DeepFace
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

                    # Filter out invalid or tiny ghost detections
                    if rw < 15 or rh < 15:
                        continue

                    rx = int(region.get("x", 0) * inv_scale)
                    ry = int(region.get("y", 0) * inv_scale)
                    box_w = int(rw * inv_scale)
                    box_h = int(rh * inv_scale)

                    dominant = r.get("dominant_emotion", "unknown")
                    emotions = {k: float(v) for k, v in r.get("emotion", {}).items()}
                    confidence = float(r.get("face_confidence", 0.0))

                    # Identify person if known_faces available
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
                    self.results = parsed_faces

            except Exception:
                pass
            finally:
                self.is_processing = False

    def stop(self):
        self.is_running = False


def draw_tech_corners(img, x, y, w, h, color, thickness=2, corner_len=18):
    """Draw stylish corner brackets around a face bounding box."""
    # Semi-transparent bounding box outline
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


def draw_hud(frame, dominant_emotion, emotion_scores, fps, face_count, primary_name=None, rec_enabled=True):
    """Draw a modern glassmorphism HUD card with emotion breakdown progress bars."""
    panel_w = 260
    panel_h = 295
    x0, y0 = 16, 16

    # Glassmorphism dark background with alpha blend
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (18, 18, 22), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    # Subtle border
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (60, 60, 70), 1)

    # Header & Live FPS
    cv2.putText(frame, "REACTION AI", (x0 + 12, y0 + 26), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1)
    cv2.putText(frame, f"FPS: {fps:.0f}", (x0 + panel_w - 72, y0 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 180), 1)

    # Status / Dominant Emotion Badge
    badge_color = EMOTION_COLORS.get(dominant_emotion.lower(), (140, 140, 140))
    cv2.rectangle(frame, (x0 + 12, y0 + 38), (x0 + panel_w - 12, y0 + 68), badge_color, -1)

    if face_count > 0:
        if primary_name:
            status_text = f"{primary_name.capitalize()}: {dominant_emotion.upper()}"
        else:
            status_text = f"Mood: {dominant_emotion.upper()}"
    else:
        status_text = "Scanning for faces..."

    text_color = (0, 0, 0) if dominant_emotion.lower() in ["happy", "surprise", "neutral"] else (255, 255, 255)
    cv2.putText(frame, status_text, (x0 + 18, y0 + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.52, text_color, 2)

    # Emotion Probability Progress Bars
    order = ["happy", "neutral", "surprise", "sad", "angry", "fear", "disgust"]
    bar_y = y0 + 92
    for emo in order:
        score = emotion_scores.get(emo, 0.0)
        color = EMOTION_COLORS.get(emo, (150, 150, 150))

        # Emotion Name
        cv2.putText(frame, f"{emo.capitalize():<8}", (x0 + 12, bar_y + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

        # Background track
        bar_x = x0 + 80
        bar_max_w = panel_w - 145
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_max_w, bar_y + 13), (45, 45, 50), -1)

        # Filled progress
        fill_w = int(bar_max_w * (score / 100.0))
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + 13), color, -1)

        # Percentage string
        cv2.putText(frame, f"{score:4.1f}%", (bar_x + bar_max_w + 8, bar_y + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 230, 230), 1)
        bar_y += 24

    # Recognition status indicator at panel bottom
    rec_str = "ON" if rec_enabled else "OFF"
    cv2.putText(frame, f"Face ID: {rec_str}", (x0 + 12, y0 + panel_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 150), 1)
    cv2.putText(frame, f"Faces: {face_count}", (x0 + panel_w - 75, y0 + panel_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 150), 1)


def draw_controls_banner(frame):
    """Draw bottom keyboard controls cheat-sheet."""
    h, w = frame.shape[:2]
    cv2.putText(
        frame,
        "[Q] Quit  |  [H] Toggle HUD  |  [S] Save Snapshot  |  [R] Toggle Face ID",
        (20, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1
    )


def main():
    print("=" * 60)
    print("      REAL-TIME EMOTION DETECTION & REACTION AI")
    print("=" * 60)
    print("[INFO] Initializing webcam...")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not access webcam. Please verify your camera connection.")
        return

    # Set camera resolution (640x480 for optimal responsiveness)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Initialize asynchronous emotion detector
    analyzer = BackgroundEmotionDetector(known_faces_dir="known_faces")

    print("[INFO] Engine started successfully.")
    print("       Press 'q' to exit.")
    print("       Press 'h' to toggle the HUD breakdown card.")
    print("       Press 's' to capture a snapshot.")
    print("       Press 'r' to toggle face recognition names.")
    print("-" * 60)

    show_hud = True
    toast_message = ""
    toast_expiry = 0.0

    prev_time = time.time()
    fps = 30.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to grab camera frame.")
                break

            # Calculate FPS
            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            # Send frame to background worker
            analyzer.update_frame(frame)

            # Retrieve latest detection results
            results = analyzer.get_results()

            primary_emotion = "neutral"
            primary_scores = {}
            primary_name = None

            # Render face bounding boxes
            for face in results:
                rx, ry, rw, rh = face["box"]
                dominant = face["dominant_emotion"]
                scores = face["emotion_scores"]
                name = face["name"]

                if not primary_scores:
                    primary_emotion = dominant
                    primary_scores = scores
                    primary_name = name

                color = EMOTION_COLORS.get(dominant.lower(), (0, 255, 0))

                # Draw high-tech corner box
                draw_tech_corners(frame, rx, ry, rw, rh, color)

                # Prepare label text
                conf = scores.get(dominant, 0.0)
                if name and analyzer.enable_recognition:
                    label = f"{name.capitalize()} | {dominant.upper()} ({conf:.0f}%)"
                else:
                    label = f"{dominant.upper()} ({conf:.0f}%)"

                # Label background banner
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
                tag_y1 = max(0, ry - th - 12)
                tag_y2 = max(th + 12, ry)
                cv2.rectangle(frame, (rx, tag_y1), (rx + tw + 14, tag_y2), color, -1)

                text_c = (0, 0, 0) if dominant.lower() in ["happy", "surprise", "neutral"] else (255, 255, 255)
                cv2.putText(frame, label, (rx + 7, tag_y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.52, text_c, 2)

            # Render HUD panel
            if show_hud:
                draw_hud(
                    frame,
                    primary_emotion,
                    primary_scores,
                    fps,
                    len(results),
                    primary_name,
                    analyzer.enable_recognition
                )

            # Bottom controls banner
            draw_controls_banner(frame)

            # Toast notification (e.g. for saved screenshots)
            if toast_message and time.time() < toast_expiry:
                cv2.putText(
                    frame,
                    toast_message,
                    (frame.shape[1] // 2 - 120, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 100),
                    2
                )

            # Display frame
            cv2.imshow("Emotion Detection & Reaction AI", frame)

            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('h'):
                show_hud = not show_hud
            elif key == ord('r'):
                enabled = analyzer.toggle_recognition()
                toast_message = f"Face ID: {'Enabled' if enabled else 'Disabled'}"
                toast_expiry = time.time() + 1.5
            elif key == ord('s'):
                # Save snapshot to snapshots/ directory
                os.makedirs("snapshots", exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                snap_path = os.path.join("snapshots", f"snapshot_{timestamp}.jpg")
                cv2.imwrite(snap_path, frame)
                toast_message = f"Saved: {os.path.basename(snap_path)}"
                toast_expiry = time.time() + 2.0
                print(f"[INFO] Snapshot saved to {snap_path}")

    finally:
        print("[INFO] Cleaning up and exiting...")
        analyzer.stop()
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Done.")


if __name__ == "__main__":
    main()
