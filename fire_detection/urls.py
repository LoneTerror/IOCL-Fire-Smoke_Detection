# fire_detection/urls.py
from django.urls import path
from . import views

# This defines the URL patterns for the app.
# The name parameter is used to reference these URLs in templates.
urlpatterns = [
    # / -> Renders the upload page
    path('', views.upload_video_view, name='upload_video'),
    # /video_feed/ -> The endpoint that streams the processed video
    path('video_feed/', views.video_feed_view, name='video_feed'),
    # /monitor/ -> The page that displays the video stream
    path('monitor/', views.monitor_view, name='monitor'),
]
