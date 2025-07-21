# fire_detection/views.py
from django.shortcuts import render, redirect
from django.http import StreamingHttpResponse, HttpResponse
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
        
        fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, 'videos'))
        filename = fs.save(video_file.name, video_file)
        
        request.session['video_path'] = fs.path(filename)
        
        return redirect('monitor')

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

# --- NEW VIEW FOR LOGS ---
def view_log_file(request):
    """
    Reads the detection_log.txt file and displays it on a new page.
    """
    log_file_path = os.path.join(settings.BASE_DIR, 'detection_log.txt')
    log_content = "Log file is empty or does not exist yet."
    try:
        with open(log_file_path, 'r') as f:
            # Read lines and reverse them to show the most recent logs first
            log_lines = f.readlines()
            if log_lines:
                log_content = "".join(reversed(log_lines))
    except FileNotFoundError:
        # The file hasn't been created yet, the default message will be shown.
        pass
    except Exception as e:
        log_content = f"Error reading log file: {e}"
        
    context = {'log_content': log_content}
    return render(request, 'fire_detection/view_log.html', context)
