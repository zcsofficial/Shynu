import cv2
import face_recognition
import numpy as np
import csv
import os
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import pyttsx3
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash
import threading

# Flask app setup
app = Flask(__name__)
app.secret_key = "supersecretkey"  # For flash messages

# Initialize text-to-speech
engine = pyttsx3.init()

# Load known faces
known_face_encodings = []
known_face_names = []
known_faces_dir = "known_faces"

def load_faces():
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []
    for filename in os.listdir(known_faces_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image = face_recognition.load_image_file(os.path.join(known_faces_dir, filename))
            encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(encoding)
            known_face_names.append(filename.split(".")[0])

load_faces()
all_students = known_face_names.copy()  # Initial student list

# Email config (default values, editable via web)
email_config = {
    "smtp_server": "smtp.hostinger.com",
    "smtp_port": 587,
    "sender_email": "adnan@sealexoman.com",
    "password": "Adnan@66202",
    "receiver_email": "contact.adnanks@gmail.com"
}

# Attendance system
def run_attendance():
    video_capture = cv2.VideoCapture(0)
    today = datetime.now().strftime("%Y-%m-%d")
    csv_file = f"attendance_{today}.csv"
    cutoff_time = "09:00:00"

    if not os.path.exists("attendance_history"):
        os.makedirs("attendance_history")

    with open(csv_file, "a", newline="") as file:
        writer = csv.writer(file)
        if os.stat(csv_file).st_size == 0:
            writer.writerow(["Name", "Time", "Status"])

    print("Camera is on. Press 'q' to quit.")
    detected_names = set()

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Camera error!")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            if name != "Unknown" and name not in detected_names:
                current_time = datetime.now().strftime("%H:%M:%S")
                status = "On Time" if current_time <= cutoff_time else "Late"
                with open(csv_file, "a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([name, current_time, status])
                detected_names.add(name)
                print(f"{name} marked as {status} at {current_time}")
                engine.say(f"Welcome, {name}")
                engine.runAndWait()

        cv2.imshow("Attendance System", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Report
    present = detected_names
    absent = [name for name in all_students if name not in present]
    with open(f"report_{today}.txt", "w") as report:
        report.write(f"Date: {today}\nPresent: {len(present)}\nAbsent: {len(absent)}\n")
        report.write(f"Present Names: {', '.join(present)}\nAbsent Names: {', '.join(absent)}")

    # Move CSV to history
    os.rename(csv_file, os.path.join("attendance_history", csv_file))

    # Email
    msg = MIMEMultipart()
    msg["From"] = email_config["sender_email"]
    msg["To"] = email_config["receiver_email"]
    msg["Subject"] = f"Attendance for {today}"
    msg.attach(MIMEText("Here’s the attendance file and report.", "plain"))

    with open(os.path.join("attendance_history", csv_file), "rb") as f:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename={csv_file}")
        msg.attach(part)

    with open(f"report_{today}.txt", "rb") as f:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename=report_{today}.txt")
        msg.attach(part)

    with smtplib.SMTP(email_config["smtp_server"], email_config["smtp_port"]) as server:
        server.starttls()
        server.login(email_config["sender_email"], email_config["password"])
        server.sendmail(email_config["sender_email"], email_config["receiver_email"], msg.as_string())

    # History
    attendance_data = {}
    for file in os.listdir("attendance_history"):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join("attendance_history", file))
            for name in df["Name"]:
                attendance_data[name] = attendance_data.get(name, 0) + 1

    with open("history_summary.txt", "w") as f:
        f.write(f"Attendance History as of {today}:\n")
        for name, count in attendance_data.items():
            f.write(f"{name}: {count} days\n")

    video_capture.release()
    cv2.destroyAllWindows()
    print("Attendance system shut down.")

# Web routes
@app.route("/")
def index():
    today = datetime.now().strftime("%Y-%m-%d")
    csv_file = os.path.join("attendance_history", f"attendance_{today}.csv")
    history_data = {}
    if os.path.exists("history_summary.txt"):
        with open("history_summary.txt", "r") as f:
            history_data = dict(line.strip().split(": ") for line in f.readlines() if ": " in line)
    attendance_today = []
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        attendance_today = df.to_dict("records")
    return render_template("index.html", attendance=attendance_today, history=history_data, students=all_students)

@app.route("/add_student", methods=["GET", "POST"])
def add_student():
    if request.method == "POST":
        name = request.form["name"]
        photo = request.files["photo"]
        if name and photo:
            filename = f"{name}.jpg"
            photo.save(os.path.join(known_faces_dir, filename))
            load_faces()
            all_students.append(name)
            flash("Student added successfully!")
            return redirect(url_for("index"))
    return render_template("add_student.html")

@app.route("/remove_student/<name>")
def remove_student(name):
    if name in all_students:
        all_students.remove(name)
        file_path = os.path.join(known_faces_dir, f"{name}.jpg")
        if os.path.exists(file_path):
            os.remove(file_path)
        load_faces()
        flash("Student removed successfully!")
    return redirect(url_for("index"))

@app.route("/configure_email", methods=["GET", "POST"])
def configure_email():
    if request.method == "POST":
        email_config["smtp_server"] = request.form["smtp_server"]
        email_config["smtp_port"] = int(request.form["smtp_port"])
        email_config["sender_email"] = request.form["sender_email"]
        email_config["password"] = request.form["password"]
        email_config["receiver_email"] = request.form["receiver_email"]
        flash("Email settings updated!")
        return redirect(url_for("index"))
    return render_template("configure_email.html", config=email_config)

# Run Flask and attendance in parallel
if __name__ == "__main__":
    threading.Thread(target=run_attendance).start()
    app.run(debug=True, host="0.0.0.0", port=5000)