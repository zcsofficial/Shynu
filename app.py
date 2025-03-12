import cv2
import face_recognition
import numpy as np
import sqlite3
import os
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import pyttsx3
import pandas as pd
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import logging
from starlette.requests import Request
import asyncio
import base64
from typing import List

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize text-to-speech
engine = pyttsx3.init()

# Global variables
known_face_encodings = []
known_face_names = []
known_faces_dir = "known_faces"
history_dir = "attendance_history"

# SQLite database setup
def init_db():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS students 
                 (id INTEGER PRIMARY KEY, name TEXT UNIQUE)''')
    c.execute('''CREATE TABLE IF NOT EXISTS attendance 
                 (id INTEGER PRIMARY KEY, student_name TEXT, date TEXT, time TEXT, status TEXT,
                 FOREIGN KEY(student_name) REFERENCES students(name))''')
    conn.commit()
    conn.close()

# Email config
email_config = {
    "smtp_server": "smtp.hostinger.com",
    "smtp_port": 587,
    "sender_email": "adnan@sealexoman.com",
    "password": "Adnan@66202",
    "receiver_email": "contact.adnanks@gmail.com"
}

def load_faces():
    """Load student faces from known_faces directory and database."""
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []
    
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("SELECT name FROM students")
    known_face_names = [row[0] for row in c.fetchall()]
    conn.close()

    for name in known_face_names:
        filename = f"{name}.jpg"
        path = os.path.join(known_faces_dir, filename)
        if os.path.exists(path):
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                logger.info(f"Loaded face: {name}")
            else:
                logger.warning(f"No face detected in {filename}")

async def process_camera(websocket: WebSocket):
    """Process camera feed and send to WebSocket."""
    try:
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            logger.error("No camera found!")
            await websocket.send_text("Error: No camera found")
            return

        cutoff_time = "09:00:00"
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
                    current_date = datetime.now().strftime("%Y-%m-%d")
                    status = "On Time" if current_time <= cutoff_time else "Late"
                    
                    conn = sqlite3.connect('attendance.db')
                    c = conn.cursor()
                    c.execute("INSERT INTO attendance (student_name, date, time, status) VALUES (?, ?, ?, ?)",
                            (name, current_date, current_time, status))
                    conn.commit()
                    conn.close()
                    
                    detected_names.add(name)
                    logger.info(f"{name} marked as {status} at {current_time}")
                    engine.say(f"Welcome, {name}")
                    engine.runAndWait()

            # Convert frame to base64 for WebSocket
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            await websocket.send_text(f"data:image/jpeg;base64,{frame_base64}")
            await asyncio.sleep(0.033)  # ~30 fps

        video_capture.release()

    except Exception as e:
        logger.error(f"Camera processing failed: {e}")
        await websocket.send_text(f"Error: {str(e)}")

# API endpoints
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    
    today = datetime.now().strftime("%Y-%m-%d")
    c.execute("SELECT student_name, time, status FROM attendance WHERE date = ?", (today,))
    attendance_today = [{"name": row[0], "time": row[1], "status": row[2]} for row in c.fetchall()]
    
    c.execute("SELECT student_name, COUNT(*) as count FROM attendance GROUP BY student_name")
    history_data = dict(c.fetchall())
    
    c.execute("SELECT name FROM students")
    all_students = [row[0] for row in c.fetchall()]
    
    conn.close()
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "attendance": attendance_today,
        "history": history_data,
        "students": all_students
    })

@app.post("/add_student")
async def add_student(name: str = Form(...), photo: UploadFile = File(...)):
    filename = f"{name}.jpg"
    file_path = os.path.join(known_faces_dir, filename)
    with open(file_path, "wb") as f:
        f.write(await photo.read())
    
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    try:
        c.execute("INSERT INTO students (name) VALUES (?)", (name,))
        conn.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Student already exists")
    conn.close()
    
    load_faces()
    return {"message": "Student added successfully"}

@app.get("/remove_student/{name}")
async def remove_student(name: str):
    file_path = os.path.join(known_faces_dir, f"{name}.jpg")
    if os.path.exists(file_path):
        os.remove(file_path)
    
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("DELETE FROM students WHERE name = ?", (name,))
    conn.commit()
    conn.close()
    
    load_faces()
    return {"message": "Student removed successfully"}

@app.post("/send_report")
async def send_report():
    today = datetime.now().strftime("%Y-%m-%d")
    conn = sqlite3.connect('attendance.db')
    df = pd.read_sql_query("SELECT * FROM attendance WHERE date = ?", conn, params=(today,))
    conn.close()

    csv_file = f"attendance_{today}.csv"
    df.to_csv(csv_file, index=False)
    
    present = df['student_name'].tolist()
    absent = [name for name in known_face_names if name not in present]
    
    report_file = f"report_{today}.txt"
    with open(report_file, "w") as report:
        report.write(f"Date: {today}\nPresent: {len(present)}\nAbsent: {len(absent)}\n")
        report.write(f"Present: {', '.join(present)}\nAbsent: {', '.join(absent)}")

    msg = MIMEMultipart()
    msg["From"] = email_config["sender_email"]
    msg["To"] = email_config["receiver_email"]
    msg["Subject"] = f"Attendance for {today}"
    msg.attach(MIMEText("Attendance file and report attached.", "plain"))

    for file_path in [csv_file, report_file]:
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
        return {"message": "Report sent successfully"}
    except Exception as e:
        logger.error(f"Email failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to send email")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await process_camera(websocket)

# Run everything
if __name__ == "__main__":
    # Ensure directories exist
    for dir_name in [known_faces_dir, history_dir, "templates", "static"]:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
    
    init_db()
    load_faces()
    
    uvicorn.run(app, host="0.0.0.0", port=5000)