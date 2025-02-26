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
import logging
from werkzeug.utils import secure_filename

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)
app.secret_key = "supersecretkey123"

# Initialize text-to-speech
engine = pyttsx3.init()

# Global variables
known_face_encodings = []
known_face_names = []
known_faces_dir = "known_faces"
history_dir = "attendance_history"

# Email config (custom SMTP support)
email_config = {
    "smtp_server": "smtp.hostinger.com",
    "smtp_port": 587,
    "sender_email": "adnan@sealexoman.com",
    "password": "Adnan@66202",
    "receiver_email": "contact.adnanks@gmail.com"
}

def load_faces():
    """Load student faces from known_faces directory."""
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []
    try:
        for filename in os.listdir(known_faces_dir):
            if filename.lower().endswith((".jpg", ".png")):
                path = os.path.join(known_faces_dir, filename)
                image = face_recognition.load_image_file(path)
                encodings = face_recognition.face_encodings(image)
                if encodings:  # Ensure at least one face is detected
                    known_face_encodings.append(encodings[0])
                    known_face_names.append(filename.split(".")[0])
                    logger.info(f"Loaded face: {filename}")
                else:
                    logger.warning(f"No face detected in {filename}")
    except Exception as e:
        logger.error(f"Error loading faces: {e}")

load_faces()
all_students = known_face_names.copy()

def run_attendance():
    """Run the attendance system with camera."""
    try:
        # Try multiple camera indices as fallback
        for i in range(3):  # Try 0, 1, 2
            video_capture = cv2.VideoCapture(i)
            if video_capture.isOpened():
                break
        else:
            logger.error("No camera found!")
            return

        today = datetime.now().strftime("%Y-%m-%d")
        csv_file = f"attendance_{today}.csv"
        cutoff_time = "09:00:00"

        if not os.path.exists(history_dir):
            os.makedirs(history_dir)

        csv_path = os.path.join(history_dir, csv_file)
        with open(csv_path, "a", newline="") as file:
            writer = csv.writer(file)
            if os.stat(csv_path).st_size == 0:
                writer.writerow(["Name", "Time", "Status"])

        logger.info("Camera started. Press 'q' to quit.")
        detected_names = set()

        while True:
            ret, frame = video_capture.read()
            if not ret:
                logger.error("Failed to grab frame!")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                if matches:
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                if name != "Unknown" and name not in detected_names:
                    current_time = datetime.now().strftime("%H:%M:%S")
                    status = "On Time" if current_time <= cutoff_time else "Late"
                    with open(csv_path, "a", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow([name, current_time, status])
                    detected_names.add(name)
                    logger.info(f"{name} marked as {status} at {current_time}")
                    engine.say(f"Welcome, {name}")
                    engine.runAndWait()

            cv2.imshow("Attendance System", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Generate report
        present = detected_names
        absent = [name for name in all_students if name not in present]
        report_file = f"report_{today}.txt"
        with open(report_file, "w") as report:
            report.write(f"Date: {today}\nPresent: {len(present)}\nAbsent: {len(absent)}\n")
            report.write(f"Present: {', '.join(present)}\nAbsent: {', '.join(absent)}")
        logger.info("Report generated")

        # Email
        msg = MIMEMultipart()
        msg["From"] = email_config["sender_email"]
        msg["To"] = email_config["receiver_email"]
        msg["Subject"] = f"Attendance for {today}"
        msg.attach(MIMEText("Attendance file and report attached.", "plain"))

        for file_path in [csv_path, report_file]:
            with open(file_path, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(file_path)}")
                msg.attach(part)

        try:
            with smtplib.SMTP(email_config["smtp_server"], email_config["smtp_port"]) as server:
                server.starttls()
                server.login(email_config["sender_email"], email_config["password"])
                server.sendmail(email_config["sender_email"], email_config["receiver_email"], msg.as_string())
            logger.info("Attendance emailed")
        except Exception as e:
            logger.error(f"Email failed: {e}")

        # History
        attendance_data = {}
        for file in os.listdir(history_dir):
            if file.endswith(".csv"):
                try:
                    df = pd.read_csv(os.path.join(history_dir, file))
                    if "Name" in df.columns:
                        for name in df["Name"].dropna():
                            attendance_data[name] = attendance_data.get(name, 0) + 1
                except Exception as e:
                    logger.error(f"Error reading {file}: {e}")

        with open("history_summary.txt", "w") as f:
            f.write(f"Attendance History as of {today}:\n")
            for name, count in attendance_data.items():
                f.write(f"{name}: {count} days\n")
        logger.info("History updated")

        video_capture.release()
        cv2.destroyAllWindows()

    except Exception as e:
        logger.error(f"Attendance system failed: {e}")

# Web routes
@app.route("/")
def index():
    today = datetime.now().strftime("%Y-%m-%d")
    csv_file = os.path.join(history_dir, f"attendance_{today}.csv")
    history_data = {}
    attendance_today = []

    if os.path.exists("history_summary.txt"):
        with open("history_summary.txt", "r") as f:
            history_data = dict(line.strip().split(": ") for line in f.readlines() if ": " in line)

    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
            attendance_today = df.to_dict("records")
        except Exception as e:
            logger.error(f"Error reading today’s CSV: {e}")

    return render_template("index.html", attendance=attendance_today, history=history_data, students=all_students)

@app.route("/add_student", methods=["GET", "POST"])
def add_student():
    if request.method == "POST":
        name = request.form["name"]
        photo = request.files["photo"]
        if name and photo:
            filename = secure_filename(f"{name}.jpg")
            photo.save(os.path.join(known_faces_dir, filename))
            load_faces()
            all_students.append(name)
            flash("Student added successfully!", "success")
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
        flash("Student removed successfully!", "success")
    return redirect(url_for("index"))

@app.route("/configure_email", methods=["GET", "POST"])
def configure_email():
    if request.method == "POST":
        email_config["smtp_server"] = request.form["smtp_server"]
        email_config["smtp_port"] = int(request.form["smtp_port"])
        email_config["sender_email"] = request.form["sender_email"]
        email_config["password"] = request.form["password"]
        email_config["receiver_email"] = request.form["receiver_email"]
        flash("Email settings updated!", "success")
        return redirect(url_for("index"))
    return render_template("configure_email.html", config=email_config)

# Run everything
if __name__ == "__main__":
    # Ensure directories exist
    for dir_name in [known_faces_dir, history_dir]:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
    
    # Start attendance in a thread
    threading.Thread(target=run_attendance, daemon=True).start()
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)  # Disable reloader to avoid duplicate threads