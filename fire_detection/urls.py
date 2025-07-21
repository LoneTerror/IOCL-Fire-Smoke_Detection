# fire_detection/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_video_view, name='upload_video'),
    path('monitor/', views.monitor_view, name='monitor'),
    path('video_feed/', views.video_feed_view, name='video_feed'),
    
    # --- NEW URL FOR LOGS ---
    path('logs/', views.view_log_file, name='view_log'),
]
