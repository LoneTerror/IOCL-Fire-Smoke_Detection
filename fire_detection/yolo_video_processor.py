# fire_detection/yolo_video_processor.py (YOLOv8 Version)
import cv2
import os
import time
import numpy as np
from datetime import datetime
from django.conf import settings
from ultralytics import YOLO

MODEL_PATH = os.path.join(settings.BASE_DIR, 'fire_detection', 'ml_model', 'best.pt')
LOG_FILE_PATH = os.path.join(settings.BASE_DIR, 'detection_log.txt')

try:
    model = YOLO(MODEL_PATH)
    print("✅ YOLOv8 model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading YOLOv8 model: {e}")
    model = None

def log_detection_event(event_label, confidence, video_name):
    # ... (logging logic is unchanged)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] - Event: {event_label} | Confidence: {confidence:.2f}% | Video: {video_name}"
    print(f"LOGGING EVENT: {log_message}")
    try:
        with open(LOG_FILE_PATH, 'a') as log_file:
            log_file.write(log_message + "\n")
    except IOError as e:
        print(f"❌ Could not write to log file: {e}")

def process_frame(frame, source_name="Live Feed"):
    # ... (frame processing logic is unchanged)
    if model is None:
        cv2.putText(frame, "Error: Model not loaded", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return "Error", frame
    results = model(frame, verbose=False)
    result = results[0]
    for box in result.boxes:
        x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
        confidence = box.conf[0].item() * 100
        class_id = int(box.cls[0].item())
        label = model.names[class_id]
        if confidence > 50:
            color = (0, 0, 255) if 'fire' in label.lower() else (150, 150, 150)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            display_text = f"{label.capitalize()}: {confidence:.2f}%"
            (w, h), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
            cv2.putText(frame, display_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            log_detection_event(label.capitalize(), confidence, source_name)
    return "Processed", frame

def generate_video_frames(video_path, playback_speed=1.0, start_at_percent=0.0):
    # ... (uploaded video logic is unchanged)
    cap = cv2.VideoCapture(video_path)
    video_name = os.path.basename(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0
    target_frame_duration = 1.0 / (fps * playback_speed)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames > 0 and 0.0 < start_at_percent < 1.0:
        start_frame = int(total_frames * start_at_percent)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    while True:
        loop_start_time = time.time()
        success, frame = cap.read()
        if not success: break
        _, processed_frame = process_frame(frame, video_name)
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        if not ret: continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        processing_duration = time.time() - loop_start_time
        delay = target_frame_duration - processing_duration
        if delay > 0: time.sleep(delay)
    cap.release()

# --- THIS IS THE FIX: This function now accepts a camera index ---
def generate_realtime_frames(cam_index=0):
    cap = cv2.VideoCapture(cam_index) 
    if not cap.isOpened():
        print(f"❌ Error: Could not open webcam at index {cam_index}.")
        error_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_img, f"Webcam {cam_index} Not Found", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        ret, buffer = cv2.imencode('.jpg', error_img)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        return

    print(f"✅ Started real-time camera feed from index {cam_index}.")
    while True:
        success, frame = cap.read()
        if not success:
            print(f"❌ Error: Failed to capture frame from webcam {cam_index}.")
            break
        _, processed_frame = process_frame(frame, source_name=f"Real-time CAM-{cam_index}")
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        if not ret: continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()
    print(f"✅ Stopped real-time camera feed from index {cam_index}.")
