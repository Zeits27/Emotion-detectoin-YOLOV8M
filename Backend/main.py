# import cv2
# import numpy as np
# from ultralytics import YOLO
# import time
# from collections import defaultdict
# import json
# from datetime import datetime, timedelta

# class EmotionTracker:
#     def __init__(self, model_path="./weight/best5.pt."):
#         """
#         Initialize emotion tracker with YOLOv8 model
        
#         Args:
#             model_path (str): Path to the trained YOLOv8 model
#         """
#         self.model = YOLO(model_path)
#         self.tracked_faces = {}  # Store tracked faces with their info
#         self.emotion_durations = defaultdict(lambda: defaultdict(float))  # person_id -> {emotion: duration}
#         self.face_id_counter = 0
#         self.tracking_threshold = 50  # pixels threshold for face matching
#         self.emotion_labels = []  # Will be populated from model
#         self.start_time = time.time()
        
#         # Get emotion labels from model
#         if hasattr(self.model, 'names'):
#             self.emotion_labels = list(self.model.names.values())
        
#     def calculate_distance(self, center1, center2):
#         """Calculate Euclidean distance between two points"""
#         return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
#     def get_face_center(self, box):
#         """Get center point of bounding box"""
#         x1, y1, x2, y2 = box
#         return ((x1 + x2) // 2, (y1 + y2) // 2)
    
#     def match_face_to_existing(self, current_center, current_emotion):
#         """
#         Match current face detection to existing tracked faces
        
#         Args:
#             current_center (tuple): Center point of current detection
#             current_emotion (str): Detected emotion
            
#         Returns:
#             int: Face ID if matched, None if new face
#         """
#         min_distance = float('inf')
#         matched_id = None
        
#         for face_id, face_info in self.tracked_faces.items():
#             last_center = face_info['last_center']
#             distance = self.calculate_distance(current_center, last_center)
            
#             if distance < self.tracking_threshold and distance < min_distance:
#                 min_distance = distance
#                 matched_id = face_id
        
#         return matched_id
    
#     def update_emotion_duration(self, face_id, emotion, frame_time):
#         """Update emotion duration for a specific face"""
#         current_time = time.time()
        
#         if face_id in self.tracked_faces:
#             last_time = self.tracked_faces[face_id]['last_update']
#             last_emotion = self.tracked_faces[face_id]['last_emotion']
            
#             # Add duration for the last emotion
#             if last_emotion:
#                 duration = current_time - last_time
#                 self.emotion_durations[face_id][last_emotion] += duration
        
#         # Update face info
#         self.tracked_faces[face_id].update({
#             'last_update': current_time,
#             'last_emotion': emotion,
#             'last_center': self.tracked_faces[face_id]['last_center']
#         })
    
#     def process_frame(self, frame):
#         """
#         Process single frame for emotion detection and tracking
        
#         Args:
#             frame: Input frame from video/camera
            
#         Returns:
#             frame: Annotated frame with detections and tracking info
#         """
#         current_time = time.time()
#         results = self.model(frame)
        
#         current_detections = []
        
#         # Process detections
#         for result in results:
#             boxes = result.boxes
#             if boxes is not None:
#                 for box in boxes:
#                     # Get bounding box coordinates
#                     x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
#                     confidence = box.conf[0].cpu().numpy()
#                     class_id = int(box.cls[0].cpu().numpy())
                    
#                     # Get emotion label
#                     emotion = self.emotion_labels[class_id] if class_id < len(self.emotion_labels) else f"Class_{class_id}"
                    
#                     # Get face center
#                     center = self.get_face_center([x1, y1, x2, y2])
                    
#                     current_detections.append({
#                         'box': [x1, y1, x2, y2],
#                         'center': center,
#                         'emotion': emotion,
#                         'confidence': confidence
#                     })
        
#         # Match detections to existing faces or create new ones
#         matched_faces = set()
        
#         for detection in current_detections:
#             center = detection['center']
#             emotion = detection['emotion']
#             box = detection['box']
#             confidence = detection['confidence']
            
#             # Try to match with existing face
#             matched_id = self.match_face_to_existing(center, emotion)
            
#             if matched_id is not None:
#                 # Update existing face
#                 self.update_emotion_duration(matched_id, emotion, current_time)
#                 self.tracked_faces[matched_id]['last_center'] = center
#                 matched_faces.add(matched_id)
#                 face_id = matched_id
#             else:
#                 # Create new face
#                 face_id = self.face_id_counter
#                 self.face_id_counter += 1
#                 self.tracked_faces[face_id] = {
#                     'last_center': center,
#                     'last_emotion': emotion,
#                     'last_update': current_time,
#                     'first_seen': current_time
#                 }
#                 matched_faces.add(face_id)
            
#             # Draw bounding box and info
#             x1, y1, x2, y2 = box
            
#             # Choose color based on emotion (you can customize this)
#             color_map = {
#                 'happy': (0, 255, 0),
#                 'sad': (255, 0, 0),
#                 'angry': (0, 0, 255),
#                 'surprise': (255, 255, 0),
#                 'fear': (128, 0, 128),
#                 'disgust': (0, 128, 128),
#                 'neutral': (128, 128, 128)
#                 # 'content': (0, 255, 255)
#             }
#             color = color_map.get(emotion.lower(), (255, 255, 255))
            
#             # Draw bounding box
#             cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
#             # Draw face ID and emotion
#             label = f"Person {face_id}: {emotion} ({confidence:.2f})"
#             cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
#             # Draw emotion durations for this face
#             y_offset = y2 + 20
#             total_time = sum(self.emotion_durations[face_id].values())
            
#             for emo, duration in self.emotion_durations[face_id].items():
#                 if duration > 0:
#                     percentage = (duration / total_time * 100) if total_time > 0 else 0
#                     duration_text = f"{emo}: {duration:.1f}s ({percentage:.1f}%)"
#                     cv2.putText(frame, duration_text, (x1, y_offset), 
#                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
#                     y_offset += 15
        
#         # Remove faces that haven't been seen for a while (cleanup)
#         faces_to_remove = []
#         for face_id, face_info in self.tracked_faces.items():
#             if face_id not in matched_faces:
#                 time_since_last_seen = current_time - face_info['last_update']
#                 if time_since_last_seen > 3.0:  # Remove after 3 seconds
#                     faces_to_remove.append(face_id)
        
#         for face_id in faces_to_remove:
#             del self.tracked_faces[face_id]
        
#         return frame
    
#     def get_emotion_summary(self):
#         """Get summary of emotion durations for all tracked faces"""
#         summary = {}
#         for face_id, emotions in self.emotion_durations.items():
#             total_time = sum(emotions.values())
#             summary[f"Person_{face_id}"] = {
#                 'total_time': total_time,
#                 'emotions': dict(emotions),
#                 'percentages': {emo: (duration/total_time*100) if total_time > 0 else 0 
#                               for emo, duration in emotions.items()}
#             }
#         return summary
    
#     def save_results(self, filename="emotion_tracking_results.json"):
#         """Save tracking results to JSON file"""
#         summary = self.get_emotion_summary()
#         summary['session_info'] = {
#             'total_duration': time.time() - self.start_time,
#             'timestamp': datetime.now().isoformat(),
#             'total_faces_tracked': len(self.emotion_durations)
#         }
        
#         with open(filename, 'w') as f:
#             json.dump(summary, f, indent=2)
        
#         print(f"Results saved to {filename}")


# def main():
#     """Main function to run emotion tracking"""
    
#     # Initialize tracker with improved parameters
#     print("Loading model...")
#     tracker = EmotionTracker("./weight/best5.pt")  # Higher confidence threshold
#     tracker.confidence_threshold = 0.6

#     # Initialize video capture (0 for webcam, or provide video file path)
#     cap = cv2.VideoCapture(0)  # Change to video file path if needed
    
#     if not cap.isOpened():
#         print("Error: Could not open video source")
#         return
    
#     print("Starting emotion tracking...")
#     print("Press 'q' to quit, 's' to save results")
#     print("Improved tracking algorithm - should reduce duplicate IDs")
    
#     try:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 print("End of video or failed to read frame")
#                 break
            
#             # Process frame
#             annotated_frame = tracker.process_frame(frame)
            
#             # Add general info to frame
#             info_text = f"Tracked Faces: {len(tracker.tracked_faces)}"
#             cv2.putText(annotated_frame, info_text, (10, 30), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
#             # Display frame
#             cv2.imshow("Emotion Tracking", annotated_frame)
            
#             # Handle key presses
#             key = cv2.waitKey(1) & 0xFF
#             if key == ord('q'):
#                 break
#             elif key == ord('s'):
#                 tracker.save_results()
#                 print("\nCurrent Summary:")
#                 summary = tracker.get_emotion_summary()
#                 for person, data in summary.items():
#                     print(f"{person}: {data['total_time']:.1f}s total")
#                     for emotion, percentage in data['percentages'].items():
#                         if percentage > 0:
#                             print(f"  {emotion}: {percentage:.1f}%")
    
#     except KeyboardInterrupt:
#         print("\nStopped by user")
    
#     finally:
#         # Cleanup
#         cap.release()
#         cv2.destroyAllWindows()
        
#         # Save final results
#         tracker.save_results()
        
#         # Print final summary
#         print("\n=== FINAL SUMMARY ===")
#         summary = tracker.get_emotion_summary()
#         for person, data in summary.items():
#             print(f"\n{person}:")
#             print(f"  Total time tracked: {data['total_time']:.1f}s")
#             for emotion, duration in data['emotions'].items():
#                 if duration > 0:
#                     percentage = data['percentages'][emotion]
#                     print(f"  {emotion}: {duration:.1f}s ({percentage:.1f}%)")


# if __name__ == "__main__":
#     main()