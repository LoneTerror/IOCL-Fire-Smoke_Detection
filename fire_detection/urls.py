# fire_detection/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_video_view, name='upload_video'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
    
    # --- NEW URL for the single analysis page ---
    # The <str:video_name> part captures the filename from the URL
    path('analysis/<str:video_name>/', views.analysis_view, name='analysis'),
    
    path('video_feed/', views.video_feed_view, name='video_feed'),
    path('realtime_feed/', views.realtime_feed_view, name='realtime_feed'),
    path('logs/', views.view_log_file, name='view_log'),
    path('video_info/', views.get_video_info, name='video_info'),
]
