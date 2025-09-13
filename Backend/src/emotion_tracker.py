import cv2
import numpy as np
import time
from ultralytics import YOLO
from collections import defaultdict


class EmotionTracker:
    def __init__(self, model_path="./weight/best5.pt", frame_skip=5, use_cuda=True):
        # Load YOLO model
        self.model = YOLO(model_path)
        if use_cuda:
            try:
                self.model.to("cuda")
                print("Using GPU (CUDA) ✅")
            except:
                print("CUDA not available, using CPU ❌")

        # Haar cascade for face detection (ringan)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # Frame skipping
        self.frame_skip = frame_skip
        self.frame_count = 0

        # Labels
        self.emotion_labels = list(self.model.names.values()) if hasattr(self.model, "names") else []

        # Tracking system
        self.tracked_faces = {}   # face_id -> {last_center, last_emotion, last_seen, last_update}
        self.face_id_counter = 0
        self.tracking_threshold = 50  # pixels

        # Durasi emosi
        self.emotion_durations = defaultdict(lambda: defaultdict(float))  # {face_id: {emotion: duration}}

        # Warna per emosi
        self.color_map = {
            'happy': (0, 255, 0),
            'sad': (255, 0, 0),
            'angry': (0, 0, 255),
            'surprise': (255, 255, 0),
            'fear': (128, 0, 128),
            'disgust': (0, 128, 128),
            'neutral': (128, 128, 128)
        }

    def calculate_distance(self, c1, c2):
        return np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)

    def match_face(self, center):
        """Cocokkan wajah dengan ID lama"""
        min_dist, matched_id = float("inf"), None
        for fid, info in self.tracked_faces.items():
            dist = self.calculate_distance(center, info["last_center"])
            if dist < self.tracking_threshold and dist < min_dist:
                min_dist, matched_id = dist, fid
        return matched_id

    def update_emotion_duration(self, face_id, emotion):
        """Update durasi emosi untuk face_id"""
        current_time = time.time()
        last_update = self.tracked_faces[face_id].get("last_update", current_time)
        last_emotion = self.tracked_faces[face_id].get("last_emotion", None)

        # Tambahkan durasi untuk emosi sebelumnya
        if last_emotion:
            elapsed = current_time - last_update
            self.emotion_durations[face_id][last_emotion] += elapsed

        # Update info wajah
        self.tracked_faces[face_id]["last_update"] = current_time
        self.tracked_faces[face_id]["last_emotion"] = emotion

    def process_frame(self, frame):
        self.frame_count += 1
        frame_display = frame.copy()

        # Resize untuk percepatan
        scale = 0.5
        frame_small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

        # Deteksi wajah
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

        detections = []
        for (x, y, w, h) in faces:
            # Skala kembali
            x_full, y_full = int(x/scale), int(y/scale)
            w_full, h_full = int(w/scale), int(h/scale)
            cx, cy = x_full + w_full//2, y_full + h_full//2

            # Match ID lama atau buat baru
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

            # Prediksi emosi tiap beberapa frame
            emotion = self.tracked_faces[face_id]["last_emotion"]
            conf = 0.0
            if self.frame_count % self.frame_skip == 0:
                face_roi = frame[y_full:y_full+h_full, x_full:x_full+w_full]
                if face_roi.size > 0:
                    face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                    face_input = cv2.cvtColor(face_gray, cv2.COLOR_GRAY2BGR)

                    yolo_res = self.model.predict(face_input, verbose=False)
                    if len(yolo_res) > 0 and len(yolo_res[0].boxes) > 0:
                        box = yolo_res[0].boxes[0]
                        class_id = int(box.cls.cpu().numpy()[0])
                        conf = float(box.conf.cpu().numpy()[0])
                        if class_id < len(self.emotion_labels):
                            emotion = self.emotion_labels[class_id]

            # Update tracking info + durasi emosi
            self.tracked_faces[face_id].update({
                "last_center": (cx, cy),
                "last_seen": time.time()
            })
            self.update_emotion_duration(face_id, emotion)

            detections.append(((x_full, y_full, w_full, h_full), face_id, emotion, conf))

        # Hapus wajah lama
        now = time.time()
        to_remove = [fid for fid, info in self.tracked_faces.items() if now - info["last_seen"] > 3]
        for fid in to_remove:
            del self.tracked_faces[fid]

        # Gambar hasil
        for (x, y, w, h), face_id, emotion, conf in detections:
            color = self.color_map.get(emotion.lower(), (255, 255, 255))
            cv2.rectangle(frame_display, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame_display, f"ID {face_id}: {emotion} ({conf:.2f})",
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Tampilkan durasi emosi untuk wajah ini
            y_offset = y + h + 20
            for emo, duration in self.emotion_durations[face_id].items():
                if duration > 0:
                    text = f"{emo}: {duration:.1f}s"
                    cv2.putText(frame_display, text, (x, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    y_offset += 15

        # Info total wajah
        cv2.putText(frame_display, f"Tracked Faces: {len(self.tracked_faces)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        return frame_display


def main():
    tracker = EmotionTracker("./weight/best5.pt", frame_skip=5, use_cuda=True)

    cap = cv2.VideoCapture(0)  # 0 untuk webcam default
    if not cap.isOpened():
        print("Error: Cannot access camera") 
        return

    print("Press 'q' to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated = tracker.process_frame(frame)
        cv2.imshow("Emotion Tracking + Duration", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
