import cv2
import numpy as np
import os
import face_recognition

import time


# LOAD KNOWN FACES

known_face_encodings = []
known_face_names = []

base_dir = os.path.dirname(os.path.abspath(__file__))
known_faces_dir = os.path.join(base_dir, "known_faces")

print("Loading known faces...")

for filename in os.listdir(known_faces_dir):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        try:
            path = os.path.join(known_faces_dir, filename)
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)

            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(os.path.splitext(filename)[0])
                print("Loaded:", filename)

        except Exception as e:
            print("Skipping:", filename, e)

print("Loaded faces:", len(known_face_encodings))


# CAMERA

video_capture = cv2.VideoCapture(0)

# Higher smoothness: reduce resolution more
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# SMOOTHING VARIABLES

process_interval = 2   # process every 2 frames (smooth + fast)
frame_count = 0

face_locations = []
face_names = []



# MAIN LOOP

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    frame_count += 1

    # Resize for speed (important for smoothness)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Only process every N frames
    if frame_count % process_interval == 0:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []


        for (top, right,bottom,left),encoding in zip(face_locations,
                                                     face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, encoding, tolerance=0.5)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, encoding)

            if len(face_distances) > 0:
                best_match = np.argmin(face_distances)

                if matches[best_match]:
                    name = known_face_names[best_match]

            face_names.append(name)





   
    # DRAW RESULTS (SMOOTH)
   
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Smooth rectangle
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Smooth label box
        cv2.rectangle(frame, (left, bottom - 30), (right, bottom), (0, 255, 0), cv2.FILLED)

        cv2.putText(frame, name, (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
    


    cv2.imshow("Smooth Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# CLEAN EXIT

video_capture.release()
cv2.destroyAllWindows()