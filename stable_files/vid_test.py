import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket
from fastapi.responses import FileResponse, JSONResponse
from ultralytics import YOLO
import os
from pathlib import Path
import shutil
from fastapi.middleware.cors import CORSMiddleware
import platform
import json
from typing import Dict, Optional

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Configure paths
HOME = Path.cwd()
UPLOADS_DIR = HOME / "uploads"
OUTPUTS_DIR = HOME / "outputs"

# Model paths for different detection types
MODEL_PATHS = {
    'action': HOME / "runs/detect/train3/weights/best.pt",
    'proximity': HOME / "runs/detect/train3/weights/best.pt"
}

# Create directories if they don't exist
UPLOADS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# Initialize models dictionary
models: Dict[str, Optional[YOLO]] = {}

# Load models
for detection_type, model_path in MODEL_PATHS.items():
    try:
        models[detection_type] = YOLO(model_path)
        print(f"Loaded {detection_type} model successfully")
    except Exception as e:
        print(f"Error loading {detection_type} model: {e}")
        models[detection_type] = None

def get_video_writer(filename, fps, frame_size):
    system = platform.system().lower()
    if system == "darwin":
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
    elif system == "windows":
        fourcc = cv2.VideoWriter_fourcc(*'H264')
    else:
        fourcc = cv2.VideoWriter_fourcc(*'X264')
    
    return cv2.VideoWriter(filename, fourcc, fps, frame_size)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            await websocket.receive_text()
        except:
            break

async def send_progress(websocket: WebSocket, progress: float, stage: str):
    try:
        await websocket.send_json({
            "progress": progress,
            "stage": stage
        })
    except:
        pass

@app.post("/upload_video/")
async def upload_video(
    file: UploadFile = File(...),
    detection_type: str = "action"
):
    if detection_type not in models:
        raise HTTPException(status_code=400, detail="Invalid detection type")
    
    if not models[detection_type]:
        raise HTTPException(status_code=500, detail=f"{detection_type} model not initialized")
    
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        raise HTTPException(status_code=400, detail="Unsupported file format")

    video_path = None
    output_path = None
    
    try:
        # Save uploaded video
        video_path = UPLOADS_DIR / file.filename
        with video_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process video with YOLO
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        output_path = OUTPUTS_DIR / f"processed_{detection_type}_{file.filename}"
        out = get_video_writer(str(output_path), fps, (frame_width, frame_height))
        
        if not out.isOpened():
            raise HTTPException(status_code=500, detail="Failed to create video writer")

        # Process frames
        processed_frames = 0
        detections_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            results = models[detection_type](frame)
            result_frame = results[0].plot()
            out.write(result_frame)
            
            # Count detections
            if len(results[0].boxes) > 0:
                detections_count += len(results[0].boxes)
            
            processed_frames += 1

        cap.release()
        out.release()

        # Generate analysis report
        report = {
            "total_frames": total_frames,
            "processed_frames": processed_frames,
            "detection_type": detection_type,
            "total_detections": detections_count,
            "average_detections_per_frame": round(detections_count / processed_frames, 2),
            "video_duration": f"{round(total_frames/fps, 2)} seconds",
            "resolution": f"{frame_width}x{frame_height}",
            "fps": fps
        }

        # Clean up uploaded file
        video_path.unlink()

        return {
            "filename": f"processed_{detection_type}_{file.filename}",
            "report": report
        }

    except Exception as e:
        # Clean up any partial files
        if video_path and video_path.exists():
            video_path.unlink()
        if output_path and output_path.exists():
            output_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_status": {
            model_type: model is not None 
            for model_type, model in models.items()
        }
    }

@app.get("/outputs/{filename}")
async def get_processed_video(filename: str):
    video_path = OUTPUTS_DIR / filename
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    
    return FileResponse(
        path=video_path,
        media_type="video/mp4",
        headers={
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-cache",
        }
    )

@app.get("/download/{filename}")
async def download_video(filename: str):
    video_path = OUTPUTS_DIR / filename
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    
    return FileResponse(
        path=video_path,
        media_type="video/mp4",
        filename=filename,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
    )