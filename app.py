import cv2
import face_recognition
import numpy as np
import csv
from datetime import datetime
import os

# Load known faces
known_face_encodings = []
known_face_names = []
known_faces_dir = "known_faces"  # Folder with student photos

for filename in os.listdir(known_faces_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image = face_recognition.load_image_file(os.path.join(known_faces_dir, filename))
        encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(encoding)
        known_face_names.append(filename.split(".")[0])  # Name from filename

# Start webcam
video_capture = cv2.VideoCapture(0)  # 0 means default webcam

# Attendance file
today = datetime.now().strftime("%Y-%m-%d")
csv_file = f"attendance_{today}.csv"
with open(csv_file, "a", newline="") as file:
    writer = csv.writer(file)
    if os.stat(csv_file).st_size == 0:  # Write header if file is empty
        writer.writerow(["Name", "Time"])

print("Camera is on. Press 'q' to quit.")
while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Camera error!")
        break

    # Find faces in the frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Recognize faces
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Check if face matches a known person
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Log attendance
        if name != "Unknown":
            current_time = datetime.now().strftime("%H:%M:%S")
            with open(csv_file, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([name, current_time])
            print(f"Attendance marked for {name} at {current_time}")

    # Show the camera feed
    cv2.imshow("Attendance System", frame)

    # Quit with 'q' key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Clean up
video_capture.release()
cv2.destroyAllWindows()
print("Attendance complete. Check the CSV file!")