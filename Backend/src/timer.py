import time

class EmotionTimer:
    def __init__(self):
        self.face_states = {}  # face_id -> {emotion, start_time, duration}

    def update(self, face_id, emotion):
        now = time.time()
        if face_id not in self.face_states:
            self.face_states[face_id] = {"emotion": emotion, "start_time": now, "duration": 0}
        else:
            state = self.face_states[face_id]
            if state["emotion"] == emotion:
                state["duration"] = now - state["start_time"]
            else:
                state["emotion"] = emotion
                state["start_time"] = now
                state["duration"] = 0
        return self.face_states[face_id]["duration"]
