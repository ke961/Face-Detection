from deepface import DeepFace
import cv2

cap = cv2.VideoCapture(0)

print("[INFO] Starting webcam... Press 'q' to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("[ERROR] Failed to grab frame")
        break

    try:
        result = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False
        )

        emotion = result[0]["dominant_emotion"]
        scores = result[0]["emotion"]

    except Exception as e:
        print(f"ERROR: {e}")
        emotion = "Unknown"
        scores = {}

    # Show dominant emotion
    cv2.putText(
        frame,
        f"Emotion: {emotion}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )

    y = 80

    # Show emotion scores
    for emo, score in scores.items():
        cv2.putText(
            frame,
            f"{emo}: {score:.1f}%",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        y += 30

    cv2.imshow("Emotion Detection AI", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


