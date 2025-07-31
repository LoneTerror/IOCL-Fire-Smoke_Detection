# IOCL Fire & Smoke Detection System
An advanced, AI-powered monitoring system developed for Indian Oil Corporation by Team Burnt Out Engineers. This Django web application utilizes a state-of-the-art YOLOv8 object detection model to identify fire and smoke hazards in real-time from both pre-recorded videos and live camera feeds.

The application provides a flexible multi-camera dashboard, allowing users to monitor several video streams simultaneously with interactive playback controls.

Key Features
🚀 Single Video Analysis: Upload a video for a focused, immediate analysis on a dedicated page with full playback controls.

🖥️ Multi-Camera Dashboard: A dynamic grid interface that starts with one camera and allows you to add up to 8 feeds for comprehensive monitoring.

📹 Real-time Webcam Feed: Allocate any camera slot to a live feed from any webcam connected to the host machine, with a dropdown to switch between devices.

🤖 State-of-the-Art AI Model: Powered by a custom-trained YOLOv8 model (best.pt) for high-accuracy object detection, drawing bounding boxes on identified hazards.

⏯️ Per-Camera Controls: Each camera feed has its own independent controls for play/pause, playback speed, and resizing (Small, Medium, Large) within the dashboard.

📋 Event Logging: Automatically logs all detected fire or smoke events with timestamps and confidence scores to a local detection_log.txt file for review.

How It Works
The application is built on a robust client-server architecture:

Frontend (Django Templates & JavaScript): The user interacts with the web interface to upload videos or manage the dashboard. All controls (play, pause, speed change) send requests to the backend.

Backend (Django & Python):

Views (views.py): Handle HTTP requests, manage file uploads, and serve the appropriate video streams.

URL Router (urls.py): Directs browser requests to the correct view function (e.g., /video_feed/, /realtime_feed/).

Video Processor (yolo_video_processor.py): This is the core AI engine. It uses OpenCV to read video frames and the ultralytics library to run the YOLOv8 model on each frame. It then draws bounding boxes on the processed frames.

Streaming: The processed frames are encoded as JPEGs and sent back to the browser in a continuous multipart/x-mixed-replace stream, creating the live video effect.


Prerequisites
Python 3.8+, Python 3.10.11 (recommended)

An NVIDIA GPU with CUDA installed (for GPU acceleration)

The project dependencies listed in requirements.txt

How to Run the Application
Follow these steps carefully to set up and run the project locally.


# Steps on how to run the Model for video analysis and output

Important! Follow the steps carefully

Place the AI Model
Place your trained YOLOv8 model file, named best.pt, inside the following directory:
fire_detection/ml_model/

1. You will need to create an virtual environment before running the below commands, to create virtual environment run: `python -m venv venv`
2. Start the venv: `.\venv\Scripts\activate`
3. Install Dependencies: `pip install -r requirements.txt`
4. Start the server locally: `python manage.py runserver`
5. Access your local webserver here: [`http://127.0.0.1:8000`](http://127.0.0.1:8000)




# How to Use the App
Single Video Analysis: From the main page, you can upload a video. You will be redirected to a dedicated analysis page where the video will start playing immediately.

Dashboard: Click the "Go to Dashboard" button.

Add/Remove Cameras: Use the "Add Camera" button to add new feeds. Use the "×" button on any camera (except the first) to remove it.

Allocate a Source: Click the "Allocate Video" button on any camera container. A modal will appear.

To play an uploaded video, simply click its name.

To start a live feed, select a connected webcam from the dropdown and click "Start Selected Camera".

Controls: Hover over any active video feed to access its individual controls for play/pause, speed, size and renaming the camera feeds (as per user requirements)
