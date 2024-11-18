from ultralytics import YOLO
import cv2
import numpy as np
import os
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import joblib  # Import joblib to save/load the model

# Define paths
HOME = os.getcwd()
TRAINED_MODEL_PATH = f"{HOME}/backend/action.pt"
INPUT_VIDEO_PATH = f"{HOME}/testVideo.mp4"
OUTPUT_WITH_MESSAGES_PATH = f"{HOME}/backend/unusual.avi"
SVM_MODEL_PATH = f"{HOME}/backend/svm_model.pkl"  # Path to save/load the SVM model

# Create activity feature vectors
ACTIVITIES = ['sitting', 'standing', 'bench', 'walking', 'sleeping', 'using phone','raising hand']
NORMAL_ACTIVITIES = ['sitting', 'standing', 'bench', 'walking','raising hand']

def create_activity_vector(activity):
    """Convert activity label to one-hot encoded vector"""
    return np.array([1 if act == activity else 0 for act in ACTIVITIES])


def load_svm():
    """Load the scaler and SVM model from files."""
    scaler = joblib.load(f"{HOME}/backend/scaler.pkl")
    svm = joblib.load(SVM_MODEL_PATH)
    return scaler, svm

def calculate_unusualness_score(activity, scaler, svm):
    """Calculate unusualness score using OneClassSVM."""
    activity_vector = create_activity_vector(activity)
    scaled_vector = scaler.transform(activity_vector.reshape(1, -1))
    decision_score = -svm.decision_function(scaled_vector)[0]
    score = 1 / (1 + np.exp(-decision_score))
    return score

def detect_unusual_action(bounding_boxes, scaler, svm):
    """Detect unusual actions and calculate their unusualness scores using SVM."""
    unusual_actions = ["using phone", "sleeping"]
    max_unusualness = 0
    unusual_message = ""
    
    for box_info in bounding_boxes:
        if box_info["label"] in unusual_actions:
            unusualness_score = calculate_unusualness_score(box_info["label"], scaler, svm)
            if unusualness_score > max_unusualness:
                max_unusualness = unusualness_score
                confidence_level = "High" if unusualness_score > 0.8 else "Medium" if unusualness_score > 0.5 else "Low"
                unusual_message = f"Unusual action - ({confidence_level})"
    
    return unusual_message



# Load the trained SVM model and scaler
scaler, svm = load_svm()

# Load the trained YOLO model
model_inf = YOLO(TRAINED_MODEL_PATH)

# Create a VideoCapture object to read the video
cap = cv2.VideoCapture(INPUT_VIDEO_PATH)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(OUTPUT_WITH_MESSAGES_PATH, fourcc, fps, (frame_width, frame_height))

def check_overlap(box1, box2):
    """Check for overlap between two bounding boxes."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    
    return x_overlap > 0 and y_overlap > 0

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference on the current frame
    results = model_inf.predict(frame, stream=True)

    # Store bounding boxes for the contextual check
    bounding_boxes = []
    combined_message = ""

    # Draw bounding boxes on the frame and prepare messages
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()

        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = box
            label = model_inf.names[int(cls)]

            # Get the text size for background rectangle
            text = f'{label} {score:.2f}'
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)

            # Determine rectangle color based on activity type
            if label in ["using phone", "sleeping"]:
                rect_color = (0, 0, 255)  # Red for unusual activities
            else:
                rect_color = (0, 255, 0)  # Blue for normal activities

            # Draw a filled rectangle (background for the text)
            cv2.rectangle(frame, (int(x1), int(y1) - text_height - 10), 
                            (int(x1) + text_width, int(y1)), rect_color, -1)

            # Put the text above the rectangle
            cv2.putText(frame, text, (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)  # White text



                        # Draw the bounding box with color based on unusualness
            if label in ["using phone", "sleeping"]:
                color = (0, 0, 255)  # Red color for unusual actions
            else:
                color = (0, 255, 0)  # Default blue for normal activities

                
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            # Store bounding boxes for overlap checking
            bounding_boxes.append({"label": label, "box": [x1, y1, x2, y2]})

    # Check for overlaps and prepare message
    for i in range(len(bounding_boxes)):
        box1 = bounding_boxes[i]["box"]
        label1 = bounding_boxes[i]["label"]

        for j in range(i + 1, len(bounding_boxes)):
            box2 = bounding_boxes[j]["box"]
            label2 = bounding_boxes[j]["label"]

            if check_overlap(box1, box2):
                if (label1 == "standing" and label2 == "bench") or (
                    label1 == "bench" and label2 == "standing"):
                    combined_message = "Standing near bench"
                    break

    # Detect unusual actions with SVM-based scores
    unusual_message = detect_unusual_action(bounding_boxes, scaler, svm)

    # Display the combined message at the bottom center of the image
    if combined_message:
        height, width, _ = frame.shape
        text_size = cv2.getTextSize(combined_message, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 5)[0]
        text_x = (width - text_size[0]) // 2
        text_y = height - 30
        cv2.putText(frame, combined_message, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 5)

    # Display the unusual action message with score at the bottom center of the image
    if unusual_message:
        height, width, _ = frame.shape
        # Calculate text size for centering
        text_size = cv2.getTextSize(unusual_message, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 5)[0]
        # Calculate coordinates for centered text
        text_x = (width - text_size[0]) // 2  # Centering horizontally
        text_y = height - 10  # Adjust this value to move text higher or lower from the bottom
        cv2.putText(frame, unusual_message, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 5)


    # Write the processed frame to the output video
    out.write(frame)

    # Show the frame in a window (optional)
    cv2.imshow('Action Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()