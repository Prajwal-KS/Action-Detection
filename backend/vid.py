import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from pathlib import Path
import shutil
import platform
from typing import Dict, Optional, List, Tuple
import psutil
import time
from collections import Counter
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
from pydantic import BaseModel

from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Email configuration
conf = ConnectionConfig(
    MAIL_USERNAME = os.getenv('MAIL_USERNAME'),
    MAIL_PASSWORD = os.getenv('MAIL_PASSWORD'),
    MAIL_FROM = os.getenv('MAIL_FROM'),
    MAIL_PORT = int(os.getenv('MAIL_PORT', 587)),
    MAIL_SERVER = os.getenv('MAIL_SERVER'),
    MAIL_STARTTLS = os.getenv('MAIL_STARTTLS', 'True').lower() == 'true',
    MAIL_SSL_TLS = os.getenv('MAIL_SSL_TLS', 'False').lower() == 'true',
    USE_CREDENTIALS = os.getenv('USE_CREDENTIALS', 'True').lower() == 'true'
)

app = FastAPI()
# Initialize FastMail
fastmail = FastMail(conf)


# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Configure paths
HOME = Path.cwd()
UPLOADS_DIR = HOME / "uploads"
OUTPUTS_DIR = HOME / "outputs"

# Create directories
UPLOADS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# Activity definitions
ACTIVITIES = ['sitting', 'standing', 'bench', 'walking', 'sleeping', 'using phone','raising hand']
UNUSUAL_ACTIVITIES = ['sleeping', 'using phone']
NORMAL_ACTIVITIES = ['sitting', 'standing', 'bench', 'walking', 'raising hand', 'door', 'board', 'podium']

# Model paths
MODEL_PATHS = {
    'action': HOME / "action.pt",
    'proximity': HOME / "proximity.pt",
    'unusual': HOME / "action.pt"
}

# Initialize models and SVM
models: Dict[str, Optional[YOLO]] = {}
svm_model = None
scaler = None

def initialize_models():
    global svm_model, scaler
    
    # Load YOLO models
    for detection_type, model_path in MODEL_PATHS.items():
        try:
            models[detection_type] = YOLO(model_path)
            print(f"Loaded {detection_type} model successfully")
        except Exception as e:
            print(f"Error loading {detection_type} model: {e}")
            models[detection_type] = None
    
    

initialize_models()

def create_activity_vector(activity: str) -> np.ndarray:
    """Convert activity label to one-hot encoded vector"""
    return np.array([1 if act == activity else 0 for act in ACTIVITIES])

def calculate_unusualness_score(activity: str, scaler, svm) -> float:
    """Calculate unusualness score using OneClassSVM"""
    if not scaler or not svm:
        return 0.0
    activity_vector = create_activity_vector(activity)
    scaled_vector = scaler.transform(activity_vector.reshape(1, -1))
    decision_score = -svm.decision_function(scaled_vector)[0]
    return 1 / (1 + np.exp(-decision_score))

def detect_unusual_action(bounding_boxes: List[dict], scaler, svm) -> Tuple[str, float]:
    """Detect unusual actions and calculate their unusualness scores using SVM"""
    max_unusualness = 0
    unusual_message = ""
    
    for box_info in bounding_boxes:
        if box_info["label"] in UNUSUAL_ACTIVITIES:
            unusualness_score = calculate_unusualness_score(box_info["label"], scaler, svm)
            if unusualness_score > max_unusualness:
                max_unusualness = unusualness_score
                confidence_level = "High" if unusualness_score > 0.8 else "Medium" if unusualness_score > 0.5 else "Low"
                unusual_message = f"Unusual Activity Detected - {box_info['label']} ({confidence_level})"
    
    return unusual_message, max_unusualness



ACTIVITY_COLORS = {
    "sitting": (0, 255, 255), 
    "standing": (230, 230, 250),  
    "bench": (255, 255, 0),  
    "walking": (127, 255, 0),
    "sleeping":(0, 0, 220),
    "using phone":(0, 0, 220),
    "raising hand":(100,101,102),
    "default":(255,255,255),
}

def draw_detection_visuals(frame: np.ndarray, box_info: dict, score: float) -> np.ndarray:
    x1, y1, x2, y2 = map(int, box_info["box"])
    label = box_info["label"]

    # Text content
    text = f'{label} {score:.2f}'

    # Font selection and scaling
    font = cv2.FONT_HERSHEY_COMPLEX
    font_scale = 1.0

    # Get text size for background rectangle
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, 2)

    # Determine rectangle color
    rect_color = ACTIVITY_COLORS.get(label, ACTIVITY_COLORS["default"])

    # Draw background rectangle
    cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), rect_color, -1)

    # Main text in white (without shadow)
    cv2.putText(frame, text, (x1, y1 - 10), font, font_scale, (0, 0, 0), 2)

    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), rect_color, 2)

    return frame

def check_overlap(box1, box2) -> bool:
    """Check for overlap between two bounding boxes"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    
    return x_overlap > 0 and y_overlap > 0

def analyze_proximity(bounding_boxes: List[dict]) -> str:
    """Analyze proximity between detected objects with expanded detection"""
    proximity_message = ""
    
    proximity_rules = [
        ("standing", "bench"),
        ("standing", "door"),
        ("standing", "board"),
        ("standing", "podium")
    ]
    
    for i in range(len(bounding_boxes)):
        box1 = bounding_boxes[i]["box"]
        label1 = bounding_boxes[i]["label"]
        
        for j in range(i + 1, len(bounding_boxes)):
            box2 = bounding_boxes[j]["box"]
            label2 = bounding_boxes[j]["label"]
            
            if check_overlap(box1, box2):
                # Check all proximity rules in both directions
                if ((label1 == "standing" and label2 == "door") or 
                    (label1 == "door" and label2 == "standing")):
                    proximity_message = "Standing near door"
                    break
                
                if ((label1 == "standing" and label2 == "board") or 
                    (label1 == "board" and label2 == "standing")):
                    proximity_message = "Standing near board"
                    break
                
                if ((label1 == "standing" and label2 == "podium") or 
                    (label1 == "podium" and label2 == "standing")):
                    proximity_message = "Standing near podium"
                    break
                
                # Keep the existing bench rule
                if ((label1 == "standing" and label2 == "bench") or 
                    (label1 == "bench" and label2 == "standing")):
                    proximity_message = "Standing near bench"
                    break
    
    return proximity_message

def draw_messages(frame: np.ndarray, messages: List[str]) -> np.ndarray:
    """Draw messages on the frame"""
    if not messages:  # Skip if no messages
        return frame
        
    height, width, _ = frame.shape
    y_offset = 60  # Start from bottom with some padding
    
    for message in messages:
        if message:
            # Use larger font scale for better visibility
            font_scale = 1.5
            thickness = 3
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(
                message, font, font_scale, thickness)
            
            # Calculate position (centered horizontally)
            x = (width - text_width) // 2
            y = height - y_offset
            
            # Draw text with color based on message type
            # Red for unusual activities, green for proximity messages
            color = (0, 0, 255) if "Unusual" in message else (0, 255, 0)
            
            # Draw text directly without background
            cv2.putText(frame, message, (x, y), font, font_scale, color, thickness)
            
            y_offset += text_height + 40  # Increase offset for next message
    
    return frame


async def process_video_frames(cap, detection_type: str, out, total_frames: int, video_name: str, user_email: str, websocket: Optional[WebSocket] = None):
    processed_frames = 0
    detections_count = 0
    unusual_actions_count = 0
    start_time = time.time()
    detected_activities = Counter()
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    email_sent = False
    first_unusual_activity = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Regular object detection
        results = models[detection_type](frame)
        
        if len(results[0].boxes) > 0:
            bounding_boxes = []
            unusual_activity_detected = False
            
            for box, score, cls in zip(
                results[0].boxes.xyxy.cpu().numpy(),
                results[0].boxes.conf.cpu().numpy(),
                results[0].boxes.cls.cpu().numpy()
            ):
                label = models[detection_type].names[int(cls)]
                detected_activities[label] += 1

                if label in UNUSUAL_ACTIVITIES:
                    unusual_activity_detected = True
                
                if label in UNUSUAL_ACTIVITIES and not email_sent:  # Check if email hasn't been sent yet
                    unusual_actions_count += 1
                    
                    # Only store the first unusual activity details
                    if first_unusual_activity is None:
                        # Calculate timestamp
                        current_time = processed_frames / fps
                        timestamp = f"{int(current_time // 60):02d}:{int(current_time % 60):02d}"
                        
                        # Get frame location info
                        x1, y1, x2, y2 = map(int, box)
                        frame_height, frame_width = frame.shape[:2]
                        location = "center" if (frame_width//3 < x1 < 2*frame_width//3) else "left" if x1 < frame_width//3 else "right"
                        
                        # Store first unusual activity details
                        first_unusual_activity = {
                            "label": label,
                            "score": score,
                            "timestamp": timestamp,
                            "frame_number": processed_frames,
                            "location": location,
                            "normal_activities": [act for act in detected_activities.keys() if act in NORMAL_ACTIVITIES]
                        }
                        
                        # Send email only for the first detection
                        await send_unusual_activity_email(
                            email=user_email,
                            activity_type=label,
                            confidence=score * 100,
                            frame_number=processed_frames,
                            video_name=video_name,
                            timestamp=timestamp,
                            total_duration=total_frames/fps,
                            detection_count=1,  # Since this is first detection
                            fps=fps,
                            additional_info={
                                "location": location,
                                "duration": f"{score:.1f} seconds",
                                "normal_activities": first_unusual_activity["normal_activities"]
                            }
                        )
                        email_sent = True  # Set flag to prevent further emails
                elif label in UNUSUAL_ACTIVITIES:
                    unusual_actions_count += 1
                
                box_info = {
                    "label": label,
                    "box": box,
                    "score": score
                }
                bounding_boxes.append(box_info)
                
                frame = draw_detection_visuals(frame, box_info, score)
            
            # Rest of the processing remains the same...
            unusual_message, unusualness_score = detect_unusual_action(bounding_boxes, scaler, svm_model)
            proximity_message = analyze_proximity(bounding_boxes)
            messages = [msg for msg in [unusual_message, proximity_message] if msg]

            if unusual_activity_detected:
                messages.append("Unusual Activity Detected")

            frame = draw_messages(frame, messages)
            
            detections_count += len(bounding_boxes)
        
        out.write(frame)
        processed_frames += 1
        
        if websocket:
            try:
                progress = round((processed_frames / total_frames) * 100, 1)
                await websocket.send_json({
                    "progress": progress,
                    "stage": "processing",
                    "current_frame": processed_frames,
                    "total_frames": total_frames,
                    "elapsed_time": format_elapsed_time(time.time() - start_time)
                })
            except:
                pass
    
    processing_time = time.time() - start_time
    return {
        "processed_frames": processed_frames,
        "detections_count": detections_count,
        "unusual_actions_count": unusual_actions_count,
        "processing_time": processing_time,
        "detected_activities": dict(detected_activities),
        "results": results,
    }

class EmailSchema(BaseModel):
    email: str
    activity_type: str
    confidence: float
    timestamp: str
    frame_number: int
    total_duration: float
    video_name: str
    detection_count: int
    fps: int
    additional_info: dict = {}

# Add a new function to format the email content
async def send_unusual_activity_email(
    email: str,
    activity_type: str,
    confidence: float,
    frame_number: int,
    video_name: str,
    timestamp: str,
    total_duration: float,
    detection_count: int,
    fps: int,
    additional_info: dict = {}
):
    try:
        # Format duration in minutes and seconds
        minutes = int(total_duration // 60)
        seconds = int(total_duration % 60)
        duration_formatted = f"{minutes}m {seconds}s"

        # Calculate timestamp percentage through video
        timestamp_percentage = (frame_number / (total_duration * fps)) * 100

        message = MessageSchema(
            subject=f"⚠️ Unusual Activity Alert: {activity_type}",
            recipients=[email],
            body=f"""
            🚨 UNUSUAL ACTIVITY DETECTED IN VIDEO ANALYSIS 🚨
            
            Video Details:
            --------------
            File Name: {video_name}
            Total Duration: {duration_formatted}
            
            Activity Information:
            -------------------
            Type: {activity_type}
            Confidence Level: {confidence:.2f}%
            Timestamp: {timestamp}
            Location in Video: {timestamp_percentage:.1f}% through video
            Frame Number: {frame_number}
            
            Additional Information:
            ---------------------
            • Location in Frame: {additional_info.get('location', 'N/A')}
            • Associated Normal Activities: {', '.join(additional_info.get('normal_activities', []))}
            
            Please review this activity in your dashboard for more details and visual confirmation.
            
            This is an automated message. Please do not reply.
            """,
            subtype="plain"
        )
        
        await fastmail.send_message(message)
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False


# Rest of the FastAPI endpoints remain the same
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            await websocket.receive_text()
        except:
            break

@app.post("/upload_video/")
async def upload_video(
    user_email: str,
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

        # Process video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file")

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        output_path = OUTPUTS_DIR / f"processed_{detection_type}_{file.filename}"
        out = get_video_writer(str(output_path), fps, (frame_width, frame_height))
        
        if not out.isOpened():
            raise HTTPException(status_code=500, detail="Failed to create video writer")

        # Process frames with email notification
        processing_results = await process_video_frames(
            cap=cap,
            detection_type=detection_type,
            out=out,
            total_frames=total_frames,
            video_name=file.filename,
            user_email=user_email,
            websocket=None
        )
        
        cap.release()
        out.release()

        # Generate analysis report
        report = {
            "total_frames": total_frames,
            "processed_frames": processing_results["processed_frames"],
            "detection_type": detection_type,
            "total_detections": processing_results["detections_count"],
            "average_detections_per_frame": round(processing_results["detections_count"]/ processing_results["processed_frames"], 2),
            "unusual_actions_detected": processing_results["unusual_actions_count"],
            "detected_activities": processing_results["detected_activities"],
            "video_duration": f"{round(total_frames/fps, 2)} seconds",
            "video_duration_seconds": round(total_frames/fps, 2),
            "resolution": f"{frame_width}x{frame_height}",
            "fps": fps,
            "processing_time": f"{round(processing_results['processing_time'], 2)} seconds",
            "elapsed_time": format_elapsed_time(processing_results['processing_time']),
            "detection_confidence": float(processing_results['results'][0].boxes.conf.mean()) * 100 if len(processing_results['results'][0].boxes) > 0 else 0,
            "performance_metrics": {
                "cpu_usage": psutil.cpu_percent(),
                "memory_usage": round(psutil.Process().memory_info().rss / 1024 / 1024, 2),
                "processing_speed": round(processing_results['processed_frames'] / processing_results['processing_time'], 2)
            }
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
        },
        "svm_model_status": svm_model is not None,
        "scaler_status": scaler is not None
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

def get_video_writer(filename: str, fps: int, frame_size: tuple):
    """Get appropriate video writer based on operating system"""
    system = platform.system().lower()
    if system == "darwin":
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
    elif system == "windows":
        fourcc = cv2.VideoWriter_fourcc(*'H264')
    else:
        fourcc = cv2.VideoWriter_fourcc(*'X264')
    
    return cv2.VideoWriter(filename, fourcc, fps, frame_size)

def format_elapsed_time(seconds: float) -> str:
    """Convert seconds to MM:SS format"""
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    return f"{minutes}:{remaining_seconds:02d}"

@app.post("/train_svm")
async def train_svm():
    """Endpoint to train the SVM model with new data"""
    try:
        # Create training data from normal activities
        training_vectors = []
        for activity in NORMAL_ACTIVITIES:
            training_vectors.append(create_activity_vector(activity))
        training_data = np.array(training_vectors)
        
        # Initialize and fit StandardScaler
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(training_data)
        
        # Initialize and train OneClassSVM
        svm = OneClassSVM(kernel='rbf', nu=0.1)
        svm.fit(scaled_data)
        
       
        
        # Reinitialize the models
        initialize_models()
        
        return {"message": "SVM model trained and saved successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training SVM model: {str(e)}")

@app.get("/model_info")
async def get_model_info():
    """Get information about loaded models and their configurations"""
    return {
        "models": {
            model_type: {
                "loaded": model is not None,
                "path": str(MODEL_PATHS[model_type])
            }
            for model_type, model in models.items()
        },
        "activities": {
            "all": ACTIVITIES,
            "normal": NORMAL_ACTIVITIES,
            "unusual": UNUSUAL_ACTIVITIES
        },
        "svm_status": {
            "model_loaded": svm_model is not None,
            "scaler_loaded": scaler is not None,
        }
    }

@app.post("/analyze_frame")
async def analyze_single_frame(
    file: UploadFile = File(...),
    detection_type: str = "action"
):
    """Analyze a single frame/image for activities and unusual behaviors"""
    if detection_type not in models:
        raise HTTPException(status_code=400, detail="Invalid detection type")
    
    if not models[detection_type]:
        raise HTTPException(status_code=500, detail=f"{detection_type} model not initialized")
    
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        raise HTTPException(status_code=400, detail="Unsupported file format")
    
    try:
        # Read and process the image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        
        # Run detection
        results = models[detection_type](frame)
        
        # Process detections
        detections = []
        unusual_actions = []
        
        if len(results[0].boxes) > 0:
            bounding_boxes = []
            
            for box, score, cls in zip(
                results[0].boxes.xyxy.cpu().numpy(),
                results[0].boxes.conf.cpu().numpy(),
                results[0].boxes.cls.cpu().numpy()
            ):
                label = models[detection_type].names[int(cls)]
                detection = {
                    "label": label,
                    "confidence": float(score),
                    "box": box.tolist()
                }
                detections.append(detection)
                
                if label in UNUSUAL_ACTIVITIES:
                    unusualness_score = calculate_unusualness_score(label, scaler, svm_model)
                    unusual_actions.append({
                        "activity": label,
                        "unusualness_score": float(unusualness_score)
                    })
        
        # Analyze proximity
        proximity_alerts = []
        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                if check_overlap(detections[i]["box"], detections[j]["box"]):
                    proximity_alerts.append({
                        "object1": detections[i]["label"],
                        "object2": detections[j]["label"]
                    })
        
        return {
            "detections": detections,
            "unusual_actions": unusual_actions,
            "proximity_alerts": proximity_alerts,
            "total_detections": len(detections),
            "unusual_count": len(unusual_actions)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

