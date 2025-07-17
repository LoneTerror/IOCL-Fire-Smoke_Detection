# fire_detection/views.py
from django.shortcuts import render, redirect
from django.http import StreamingHttpResponse
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from .video_processor import generate_video_frames
import os

def upload_video_view(request):
    """
    Handles the video upload form. When a video is POSTed, it saves the video
    and redirects the user to the monitoring page.
    """
    if request.method == 'POST' and request.FILES.get('video'):
        video_file = request.FILES['video']
        
        # Use FileSystemStorage to save the file
        fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, 'videos'))
        filename = fs.save(video_file.name, video_file)
        
        # Store the path to the video file in the session
        request.session['video_path'] = fs.path(filename)
        
        # Redirect to the page that will display the video stream
        return redirect('monitor')

    # --- THIS IS THE CRITICAL LINE ---
    # It ensures Django processes the HTML file as a template.
    return render(request, 'fire_detection/upload.html')


def monitor_view(request):
    """
    Renders the page that will display the live-processed video stream.
    """
    if 'video_path' not in request.session:
        return redirect('upload_video')
        
    return render(request, 'fire_detection/monitor.html')


def video_feed_view(request):
    """
    This is the main view for video streaming.
    """
    video_path = request.session.get('video_path')
    
    if not video_path or not os.path.exists(video_path):
        return redirect('upload_video')

    frame_generator = generate_video_frames(video_path)
    
    response = StreamingHttpResponse(
        frame_generator,
        content_type='multipart/x-mixed-replace; boundary=frame'
    )
    
    return response
