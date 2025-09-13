import os
import cv2
import time
import psycopg2
import numpy as np
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request
from collections import defaultdict
from ultralytics import YOLO
from datetime import datetime
import threading
from flask_cors import CORS


load_dotenv()
app = Flask(__name__)
CORS(app)

emotion_colors = {
    "happy": (0, 255, 0),        # hijau
    "sad": (255, 0, 0),          # biru
    "angry": (0, 0, 255),        # merah
    "surprise": (255, 255, 0),   # kuning
    "fear": (128, 0, 128),       # ungu
    "disgust": (0, 128, 128),    # teal
    "neutral": (200, 200, 200)   # abu-abu
}

# ---------------- Database ----------------
def get_db_connection():
    return psycopg2.connect(
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME")
    )

def init_db():
    conn = get_db_connection()
    cur = conn.cursor()
    # sessions table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id SERIAL PRIMARY KEY,
            start_time TIMESTAMP DEFAULT NOW(),
            end_time TIMESTAMP
        );
    """)
    # face_emotions table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS face_emotions (
            id SERIAL PRIMARY KEY,
            session_id INT REFERENCES sessions(id) ON DELETE CASCADE,
            face_id INT,
            emotion TEXT,
            duration FLOAT
        );
    """)
    conn.commit()
    cur.close()
    conn.close()

# ---------------- Global State ----------------
current_session_id = None

# ---------------- Emotion Tracker ----------------
class EmotionTracker:
    def __init__(self, model_path="./weight/best5.pt", frame_skip=5, use_cuda=True):
        self.model = YOLO(model_path)
        if use_cuda:
            try:
                self.model.to("cuda")
                print("Using GPU (CUDA) ✅")
            except:
                print("CUDA not available, using CPU ❌")

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.frame_skip = frame_skip
        self.frame_count = 0
        self.emotion_labels = list(self.model.names.values()) if hasattr(self.model, "names") else []
        self.tracked_faces = {}
        self.face_id_counter = 0
        self.tracking_threshold = 50
        self.emotion_durations = defaultdict(lambda: defaultdict(float))

    def calculate_distance(self, c1, c2):
        return np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)

    def match_face(self, center):
        min_dist, matched_id = float("inf"), None
        for fid, info in self.tracked_faces.items():
            dist = self.calculate_distance(center, info["last_center"])
            if dist < self.tracking_threshold and dist < min_dist:
                min_dist, matched_id = dist, fid
        return matched_id

    def update_emotion_duration(self, face_id, emotion):
        current_time = time.time()
        last_update = self.tracked_faces[face_id].get("last_update", current_time)
        last_emotion = self.tracked_faces[face_id].get("last_emotion", None)

        if last_emotion:
            elapsed = current_time - last_update
            self.emotion_durations[face_id][last_emotion] += elapsed

        self.tracked_faces[face_id]["last_update"] = current_time
        self.tracked_faces[face_id]["last_emotion"] = emotion


    def process_frame(self, frame):
        self.frame_count += 1
        frame_display = frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

        detections = []
        for (x, y, w, h) in faces:
            cx, cy = x + w // 2, y + h // 2
            face_id = self.match_face((cx, cy))
            if face_id is None:
                face_id = self.face_id_counter
                self.face_id_counter += 1
                self.tracked_faces[face_id] = {
                    "last_center": (cx, cy),
                    "last_emotion": "neutral",
                    "last_seen": time.time(),
                    "last_update": time.time()
                }

            emotion = self.tracked_faces[face_id]["last_emotion"]
            conf = 0.0
            if self.frame_count % self.frame_skip == 0:
                face_roi = frame[y:y+h, x:x+w]
                if face_roi.size > 0:
                    # tetap pakai grayscale biar konsisten
                    face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                    face_input = cv2.cvtColor(face_gray, cv2.COLOR_GRAY2BGR)

                    yolo_res = self.model.predict(face_input, verbose=False)
                    if len(yolo_res) > 0 and len(yolo_res[0].boxes) > 0:
                        boxes = yolo_res[0].boxes
                        confs = boxes.conf.detach().cpu().numpy()
                        clss = boxes.cls.detach().cpu().numpy().astype(int)

                        best_idx = confs.argmax()
                        conf = float(confs[best_idx])
                        class_id = clss[best_idx]

                        if class_id < len(self.emotion_labels):
                            emotion = self.emotion_labels[class_id]

                        print(f"[YOLO] ID {face_id} → {emotion}, conf={conf:.2f}")



            self.tracked_faces[face_id].update({
                "last_center": (cx, cy),
                "last_seen": time.time(),
                "last_conf": conf if conf > 0 else self.tracked_faces[face_id].get("last_conf", 0.0)
            })

            self.update_emotion_duration(face_id, emotion)

            detections.append(((x, y, w, h), face_id, emotion, conf))

        now = time.time()
        to_remove = [fid for fid, info in self.tracked_faces.items() if now - info["last_seen"] > 3]
        for fid in to_remove:
            del self.tracked_faces[fid]

        # gambar hasil dengan warna sesuai emosi
        for (x, y, w, h), face_id, emotion, conf in detections:
            color = emotion_colors.get(emotion.lower(), (255, 255, 255))
            cv2.rectangle(frame_display, (x, y), (x+w, y+h), color, 1)  # tipis
            conf_display = self.tracked_faces[face_id].get("last_conf", 0.0)
            cv2.putText(frame_display,
                        f"ID {face_id}: {emotion} ({conf_display:.2f})",
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)


        return frame_display

    def get_summary(self, session_id=None):
        summary = {}
        if session_id is None:
            emos = self.emotion_durations
        else:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("""
                SELECT face_id, emotion, SUM(duration)
                FROM face_emotions
                WHERE session_id = %s
                GROUP BY face_id, emotion
            """, (session_id,))
            rows = cur.fetchall()
            cur.close()
            conn.close()

            emos = defaultdict(lambda: defaultdict(float))
            for face_id, emo, dur in rows:
                emos[face_id][emo] = dur

        for fid, emos_data in emos.items():
            total = sum(emos_data.values())
            summary[f"person_{fid}"] = {
                "total_time": total,
                "emotions": dict(emos_data),
                "percentages": {e:(d/total*100) if total>0 else 0 for e,d in emos_data.items()}
            }
        return summary

# ---------------- Camera Class ----------------
class VideoCamera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.frame = None
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while True:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame

    def get_frame(self):
        return self.frame

# ---------------- Flask Routes ----------------
tracker = EmotionTracker("./weight/best5.pt")
camera = VideoCamera()

def gen_frames():
    while True:
        frame = camera.get_frame()
        if frame is None:
            continue
        frame = tracker.process_frame(frame)
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/start_session", methods=["POST"])
def start_session():
    global current_session_id
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("INSERT INTO sessions DEFAULT VALUES RETURNING id;")
    session_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    current_session_id = session_id
    return jsonify({"session_id": session_id})

@app.route("/stop_session", methods=["POST"])
def stop_session():
    global current_session_id, tracker
    conn = get_db_connection()
    cur = conn.cursor()

    # update end_time session
    cur.execute("UPDATE sessions SET end_time = %s WHERE id = %s;",
                (datetime.now(), current_session_id))

    # simpan semua durasi emosi dari tracker
    for face_id, emos in tracker.emotion_durations.items():
        for emo, dur in emos.items():
            cur.execute("""
                INSERT INTO face_emotions (session_id, face_id, emotion, duration)
                VALUES (%s, %s, %s, %s)
            """, (current_session_id, face_id, emo, dur))

    conn.commit()
    cur.close()
    conn.close()

    sid = current_session_id
    current_session_id = None
    tracker = EmotionTracker("./weight/best5.pt")  # reset tracker

    return jsonify({"session_id": sid, "status": "stopped"})

@app.route("/sessions", methods=["GET"])
def get_sessions():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, start_time, end_time FROM sessions ORDER BY id DESC;")
    sessions = [
        {"id": row[0], "start_time": row[1], "end_time": row[2]}
        for row in cur.fetchall()
    ]
    cur.close()
    conn.close()
    return jsonify(sessions)

@app.route("/summary")
def summary():
    session_id = request.args.get("session_id", type=int)
    return jsonify(tracker.get_summary(session_id))

# ---------------- Main ----------------
if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5000, debug=True)
