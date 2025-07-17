# fire_detection_project/urls.py
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    # This line tells Django to look at the fire_detection app's urls.py
    # for any URL that isn't '/admin/'.
    path('', include('fire_detection.urls')),
]

# This is needed to serve uploaded media files during development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)