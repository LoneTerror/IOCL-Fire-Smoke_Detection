# fire_detection/video_processor.py (PyTorch Version)
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from django.conf import settings
import os
from datetime import datetime

# --- PyTorch Model Setup ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ PyTorch is using device: {DEVICE}")

CLASS_NAMES = ['Fire', 'Neutral', 'Smoke']
NUM_CLASSES = len(CLASS_NAMES)

# --- THIS IS THE FIX for the RuntimeError ---
# 1. Define the EXACT same custom model architecture you used for training.
#    The error message suggests your model has a 'backbone' and a 'classifier'.
#    This is a common pattern. The following is a LIKELY example using MobileNetV2,
#    as its layer names match the error keys like "backbone.features...".
#    You MUST adjust this to match the model definition in your actual training script.

class CustomFireModel(nn.Module):
    def __init__(self, num_classes):
        super(CustomFireModel, self).__init__()
        # Use a pre-trained model as the feature-extracting backbone
        self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        
        # Freeze the backbone layers so their weights don't change during training
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Replace the model's final classifier layer with a new one for our task
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

# 2. Instantiate your custom model structure
model = CustomFireModel(num_classes=NUM_CLASSES)


# 3. Load your trained weights into this custom structure.
MODEL_PATH = os.path.join(settings.BASE_DIR, 'fire_detection', 'ml_model', 'best_fire_detection_cnn.pth')
LOG_FILE_PATH = os.path.join(settings.BASE_DIR, 'detection_log.txt')

try:
    # Load the state dictionary (the saved weights) into your model structure
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval() # Set the model to evaluation mode
    print("✅ PyTorch model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading PyTorch model: {e}")
    model = None

# 4. Define the image transformations.
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- Logging Function ---
def log_detection_event(event_label, confidence, video_name):
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
    if model is None:
        cv2.putText(frame, "Error: Model not loaded", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return "Error", frame

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    
    input_tensor = preprocess(pil_image)
    input_batch = input_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_batch)
    
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    confidence, predicted_idx = torch.max(probabilities, 0)
    label = CLASS_NAMES[predicted_idx.item()]
    confidence = confidence.item() * 100

    text = f"Prediction: {label} ({confidence:.2f}%)"
    color = (0, 255, 0) if label == "Neutral" else ((0, 0, 255) if label == "Fire" else (150, 150, 150))

    cv2.rectangle(frame, (10, 20), (450, 110), (0,0,0), -1)
    cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    scores_text = f"Scores: F={probabilities[0]:.2f}, N={probabilities[1]:.2f}, S={probabilities[2]:.2f}"
    cv2.putText(frame, scores_text, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    if label in ["Fire", "Smoke"] and confidence > 60:
        log_detection_event(label, confidence, video_name)

    return label, frame


def generate_video_frames(video_path):
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
