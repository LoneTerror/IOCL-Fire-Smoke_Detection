# fire_detection/video_processor.py
import cv2
import numpy as np
import tensorflow as tf
from django.conf import settings
import os
from datetime import datetime
from tensorflow.keras.applications.resnet50 import preprocess_input

# --- Model Loading ---
MODEL_PATH = os.path.join(settings.BASE_DIR, 'fire_detection', 'ml_model', 'resnet50_fire_detection_model.h5')
LOG_FILE_PATH = os.path.join(settings.BASE_DIR, 'detection_log.txt')

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except (IOError, ImportError) as e:
    print(f"❌ Error loading model: {e}")
    model = None

# --- Logging Function ---
def log_detection_event(event_label, confidence, video_name):
    """Appends a detection event to the log file and prints it to the console."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] - Event: {event_label} | Confidence: {confidence:.2f}% | Video: {video_name}"
    print(f"LOGGING EVENT: {log_message}")
    try:
        with open(LOG_FILE_PATH, 'a') as log_file:
            log_file.write(log_message + "\n")
    except IOError as e:
        print(f"❌ Could not write to log file: {e}")


# --- Video Processing Logic ---
def process_frame(frame, video_name):
    """Processes a single video frame to detect fire or smoke."""
    if model is None:
        cv2.putText(frame, "Error: Model not loaded", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return "Error", frame

    # 1. Preprocess the frame
    img_size = (224, 224)
    # We resize the raw BGR frame directly to match the training script.
    img = cv2.resize(frame, img_size)
    
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Use the dedicated preprocessing function for ResNet50
    img_array = preprocess_input(img_array)

    # 2. Make a prediction
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    class_names = ['Fire', 'Neutral', 'Smoke']
    label = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)
    
    # --- THIS IS THE FIX: Restore diagnostic labels on every frame ---
    # Display the top prediction
    text = f"Prediction: {label} ({confidence:.2f}%)"
    color = (0, 255, 0) if label == "Neutral" else ( (0, 0, 255) if label == "Fire" else (150, 150, 150) )

    # Create a black rectangle as a background for the text
    cv2.rectangle(frame, (10, 20), (450, 110), (0,0,0), -1)
    
    # Draw the top prediction text
    cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display the raw confidence scores for all classes
    scores_text = f"Scores: F={score[0]:.2f}, N={score[1]:.2f}, S={score[2]:.2f}"
    cv2.putText(frame, scores_text, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # The logging logic remains conditional
    if label in ["Fire", "Smoke"] and confidence > 60:
        log_detection_event(label, confidence, video_name)

    return label, frame


def generate_video_frames(video_path):
    """
    A generator function that reads a video file, processes each frame,
    and yields it as a byte string in JPEG format.
    """
    cap = cv2.VideoCapture(video_path)
    video_name = os.path.basename(video_path)

    if not cap.isOpened():
        print(f"❌ Error: Could not open video file at {video_path}")
        return

    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    start_log = f"\n--- ANALYSIS STARTED for {video_name} at {start_time} ---\n"
    try:
        with open(LOG_FILE_PATH, 'a') as log_file:
            log_file.write(start_log)
    except IOError as e:
        print(f"❌ Could not write to log file: {e}")

    print(f"✅ Started processing video: {video_name}")
    while True:
        success, frame = cap.read()
        if not success:
            break

        label, processed_frame = process_frame(frame, video_name)

        ret, buffer = cv2.imencode('.jpg', processed_frame)
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    print("✅ Finished processing video.")

    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    end_log = f"--- ANALYSIS FINISHED for {video_name} at {end_time} ---\n"
    try:
        with open(LOG_FILE_PATH, 'a') as log_file:
            log_file.write(end_log)
    except IOError as e:
        print(f"❌ Could not write to log file: {e}")
