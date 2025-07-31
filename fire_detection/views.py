# fire_detection/views.py
from django.shortcuts import render, redirect
from django.http import StreamingHttpResponse, HttpResponse, JsonResponse
from .yolo_video_processor import generate_video_frames, generate_realtime_frames
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import os
import cv2

def upload_video_view(request):
    """
    Handles video upload and redirects to the new analysis page.
    """
    if request.method == 'POST' and request.FILES.get('video'):
        video_file = request.FILES['video']
        videos_dir = os.path.join(settings.MEDIA_ROOT, 'videos')
        os.makedirs(videos_dir, exist_ok=True)
        fs = FileSystemStorage(location=videos_dir)
        filename = fs.save(video_file.name, video_file)
        
        # --- THIS IS THE CHANGE ---
        # Redirect to the analysis page, passing the filename
        return redirect('analysis', video_name=filename)

    return render(request, 'fire_detection/upload.html')

# --- NEW VIEW for the single video analysis page ---
def analysis_view(request, video_name):
    """
    Renders the page for analyzing a single, specific video.
    """
    video_path = os.path.join(settings.MEDIA_ROOT, 'videos', video_name)
    if not os.path.exists(video_path):
        # If video doesn't exist, redirect to the upload page
        return redirect('upload_video')
        
    context = {'video_name': video_name}
    return render(request, 'fire_detection/analysis.html', context)

def dashboard_view(request):
    videos_dir = os.path.join(settings.MEDIA_ROOT, 'videos')
    available_videos = []
    try:
        os.makedirs(videos_dir, exist_ok=True)
        available_videos = [f for f in os.listdir(videos_dir) if os.path.isfile(os.path.join(videos_dir, f))]
    except Exception as e:
        print(f"Error reading video directory: {e}")
    context = {'available_videos': available_videos}
    return render(request, 'fire_detection/dashboard.html', context)

def video_feed_view(request):
    video_name = request.GET.get('video_name')
    if not video_name:
        return HttpResponse("No video specified.", status=404)
    video_path = os.path.join(settings.MEDIA_ROOT, 'videos', video_name)
    if not os.path.exists(video_path):
        return HttpResponse(f"Video '{video_name}' not found.", status=404)
    
    playback_speed = float(request.GET.get('speed', 1.0))
    start_at_percent = float(request.GET.get('start_at', 0.0))
    
    frame_generator = generate_video_frames(video_path, playback_speed=playback_speed, start_at_percent=start_at_percent)
    return StreamingHttpResponse(frame_generator, content_type='multipart/x-mixed-replace; boundary=frame')

def realtime_feed_view(request):
    frame_generator = generate_realtime_frames()
    return StreamingHttpResponse(frame_generator, content_type='multipart/x-mixed-replace; boundary=frame')

def get_video_info(request):
    video_name = request.GET.get('video_name')
    if not video_name:
        return JsonResponse({'error': 'No video specified'}, status=400)
    video_path = os.path.join(settings.MEDIA_ROOT, 'videos', video_name)
    if not os.path.exists(video_path):
        return JsonResponse({'error': 'Video not found'}, status=404)
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        return JsonResponse({'duration': duration})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def view_log_file(request):
    # ... (no changes here)
    log_file_path = os.path.join(settings.BASE_DIR, 'detection_log.txt')
    log_content = "Log file is empty or does not exist yet."
    try:
        with open(log_file_path, 'r') as f:
            log_lines = f.readlines()
            if log_lines:
                log_content = "".join(reversed(log_lines))
    except FileNotFoundError:
        pass
    context = {'log_content': log_content}
    return render(request, 'fire_detection/view_log.html', context)
