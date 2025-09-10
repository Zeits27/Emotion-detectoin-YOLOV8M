from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import io
import json
import base64
import asyncio
from typing import Optional, List
import tempfile
import os
from datetime import datetime
import uuid

# Import your existing modules
from main import EmotionTracker
from save_load_functions import get_emotion_summary

app = FastAPI(title="Emotion Tracking API", version="1.0.0")

# Add CORS middleware for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global tracker instance
tracker = None
active_sessions = {}

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.on_startup
async def startup_event():
    global tracker
    print("Initializing Emotion Tracker...")
    tracker = EmotionTracker("./weight/best5.pt")
    tracker.confidence_threshold = 0.6
    print("Emotion Tracker initialized successfully!")

@app.get("/")
async def root():
    return {"message": "Emotion Tracking API", "status": "running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": tracker is not None,
        "emotion_classes": tracker.emotion_labels if tracker else []
    }

@app.post("/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    """Analyze emotion from uploaded image"""
    if not tracker:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Process image
        results = tracker.model(image)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    if confidence < tracker.confidence_threshold:
                        continue
                    
                    emotion = tracker.emotion_labels[class_id] if class_id < len(tracker.emotion_labels) else f"Class_{class_id}"
                    
                    detections.append({
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "emotion": emotion,
                        "confidence": confidence
                    })
        
        return {
            "status": "success",
            "detections": detections,
            "total_faces": len(detections),
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/analyze/video")
async def analyze_video(file: UploadFile = File(...)):
    """Analyze emotion from uploaded video"""
    if not tracker:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            contents = await file.read()
            tmp_file.write(contents)
            tmp_file_path = tmp_file.name
        
        # Process video
        cap = cv2.VideoCapture(tmp_file_path)
        if not cap.isOpened():
            os.unlink(tmp_file_path)
            raise HTTPException(status_code=400, detail="Invalid video file")
        
        # Create new tracker instance for this video
        video_tracker = EmotionTracker("./weight/best5.pt")
        video_tracker.confidence_threshold = 0.6
        
        frame_results = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every 10th frame to speed up processing
            if frame_count % 10 == 0:
                processed_frame = video_tracker.process_frame(frame)
                
                # Get current detections
                current_detections = []
                if video_tracker.tracked_faces:
                    for face_id, face_info in video_tracker.tracked_faces.items():
                        current_detections.append({
                            "face_id": face_id,
                            "emotion": face_info.get('last_emotion', 'unknown'),
                            "timestamp": frame_count / cap.get(cv2.CAP_PROP_FPS)
                        })
                
                frame_results.append({
                    "frame": frame_count,
                    "timestamp": frame_count / cap.get(cv2.CAP_PROP_FPS),
                    "detections": current_detections
                })
            
            frame_count += 1
        
        cap.release()
        os.unlink(tmp_file_path)  # Clean up temp file
        
        # Get final summary
        summary = get_emotion_summary(video_tracker.emotion_durations)
        
        return {
            "status": "success",
            "summary": summary,
            "frame_results": frame_results[-10:],  # Return last 10 frames
            "total_frames": frame_count,
            "duration": frame_count / cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

@app.websocket("/ws/webcam/{session_id}")
async def websocket_webcam(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time webcam emotion tracking"""
    await manager.connect(websocket)
    
    # Create session-specific tracker
    session_tracker = EmotionTracker("./weight/best4.pt")
    session_tracker.confidence_threshold = 0.6
    active_sessions[session_id] = session_tracker
    
    try:
        while True:
            # Receive base64 encoded frame from client
            data = await websocket.receive_text()
            frame_data = json.loads(data)
            
            # Decode base64 image
            image_data = base64.b64decode(frame_data['image'].split(',')[1])
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is not None:
                # Process frame
                processed_frame = session_tracker.process_frame(frame)
                
                # Get current results
                current_results = {
                    "session_id": session_id,
                    "tracked_faces": len(session_tracker.tracked_faces),
                    "timestamp": datetime.now().isoformat()
                }
                
                # Add face details
                faces = []
                for face_id, face_info in session_tracker.tracked_faces.items():
                    emotion_data = dict(session_tracker.emotion_durations[face_id])
                    total_time = sum(emotion_data.values())
                    
                    faces.append({
                        "face_id": face_id,
                        "current_emotion": face_info.get('last_emotion', 'unknown'),
                        "total_time": total_time,
                        "emotions": emotion_data
                    })
                
                current_results["faces"] = faces
                
                # Encode processed frame back to base64
                _, buffer = cv2.imencode('.jpg', processed_frame)
                processed_image = base64.b64encode(buffer).decode('utf-8')
                current_results["processed_image"] = f"data:image/jpeg;base64,{processed_image}"
                
                await manager.send_personal_message(json.dumps(current_results), websocket)
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        if session_id in active_sessions:
            del active_sessions[session_id]
        print(f"Session {session_id} disconnected")
    except Exception as e:
        print(f"Error in webcam session {session_id}: {e}")
        if session_id in active_sessions:
            del active_sessions[session_id]

@app.get("/session/{session_id}/summary")
async def get_session_summary(session_id: str):
    """Get summary for a specific session"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_tracker = active_sessions[session_id]
    summary = get_emotion_summary(session_tracker.emotion_durations)
    
    return {
        "session_id": session_id,
        "summary": summary,
        "session_duration": session_tracker.start_time,
        "timestamp": datetime.now().isoformat()
    }

@app.delete("/session/{session_id}")
async def close_session(session_id: str):
    """Close and cleanup a session"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Get final summary before closing
    session_tracker = active_sessions[session_id]
    summary = get_emotion_summary(session_tracker.emotion_durations)
    
    # Cleanup
    del active_sessions[session_id]
    
    return {
        "status": "session_closed",
        "session_id": session_id,
        "final_summary": summary
    }

@app.get("/sessions")
async def list_active_sessions():
    """List all active sessions"""
    sessions = []
    for session_id, tracker in active_sessions.items():
        sessions.append({
            "session_id": session_id,
            "tracked_faces": len(tracker.tracked_faces),
            "start_time": tracker.start_time,
            "duration": datetime.now().timestamp() - tracker.start_time
        })
    
    return {
        "active_sessions": sessions,
        "total_sessions": len(sessions)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)