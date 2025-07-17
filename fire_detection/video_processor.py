# fire_detection/video_processor.py
import cv2
import numpy as np
import tensorflow as tf
from django.conf import settings
import os

# --- Model Loading ---
# Load the pre-trained model once when the module is loaded.
# This is more efficient than loading it for every request.
# --- THIS IS THE FIX ---
# Updated to use the correct model filename.
MODEL_PATH = os.path.join(settings.BASE_DIR, 'fire_detection', 'ml_model', 'resnet50_fire_detection_model.h5')

try:
    # It's good practice to wrap this in a try-except block
    # in case the model file is missing or corrupted.
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except (IOError, ImportError) as e:
    print(f"❌ Error loading model: {e}")
    model = None

# --- Video Processing Logic ---

def process_frame(frame):
    """
    Processes a single video frame to detect fire or smoke.

    Args:
        frame: A numpy array representing a single video frame (from OpenCV).

    Returns:
        A tuple (label, frame) where:
        - label (str): The prediction ("Fire", "Smoke", "Neutral").
        - frame (numpy.ndarray): The processed frame with the label drawn on it.
    """
    if model is None:
        # If the model failed to load, return the original frame with an error message.
        cv2.putText(frame, "Error: Model not loaded", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return "Error", frame

    # 1. Preprocess the frame for the model
    # The error message told us the model expects 224x224 images.
    img_size = (224, 224)
    img = cv2.resize(frame, img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Create a batch

    # 2. Make a prediction
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    # Assuming your model has class names in this order.
    # **IMPORTANT**: Adjust these class names to match your model's output.
    class_names = ['Fire', 'Neutral', 'Smoke']
    label = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    # 3. Draw the result on the original frame
    # Only show labels for fire or smoke with a certain confidence
    if label in ["Fire", "Smoke"] and confidence > 60:
        text = f"{label}: {confidence:.2f}%"
        # Set color based on label
        color = (0, 0, 255) if label == "Fire" else (150, 150, 150) # Red for fire, Gray for smoke
        
        # Add a semi-transparent background rectangle for the text
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.rectangle(frame, (20, 30 - h - 5), (20 + w, 30 + 10), (0,0,0), -1)
        # Put the text on the frame
        cv2.putText(frame, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    return label, frame


def generate_video_frames(video_path):
    """
    A generator function that reads a video file, processes each frame,
    and yields it as a byte string in JPEG format.

    Args:
        video_path (str): The full path to the video file.
    """
    # Create a VideoCapture object
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Error: Could not open video file at {video_path}")
        return

    print(f"✅ Started processing video: {video_path}")
    while True:
        # Read one frame from the video
        success, frame = cap.read()
        if not success:
            # End of video
            break

        # Process the frame for fire/smoke detection
        label, processed_frame = process_frame(frame)

        # Encode the processed frame into JPEG format
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        if not ret:
            # Skip frame if encoding fails
            continue
        
        frame_bytes = buffer.tobytes()

        # Yield the frame in the format required for multipart streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    # Release the video capture object when done
    cap.release()
    print("✅ Finished processing video.")
