import cv2

class FaceTracker:
    def __init__(self):
        self.trackers = {}  # face_id -> tracker
        self.next_id = 0

    def update(self, frame, detected_bboxes):
        new_trackers = {}
        face_ids = []

        # Update existing trackers
        for face_id, tracker in self.trackers.items():
            success, bbox = tracker.update(frame)
            if success:
                new_trackers[face_id] = tracker
                face_ids.append((face_id, bbox))

        # Add new trackers untuk wajah baru
        for bbox in detected_bboxes:
            x, y, w, h, _ = bbox
            tracker = cv2.legacy.TrackerCSRT_create()
            tracker.init(frame, (x, y, w, h))
            new_trackers[self.next_id] = tracker
            face_ids.append((self.next_id, (x, y, w, h)))
            self.next_id += 1

        self.trackers = new_trackers
        return face_ids
