# fire_detection_project/urls.py (Corrected Version)

from django.contrib import admin
# You must import 'include'
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    
    # --- THIS IS THE FIX ---
    # This line tells Django to look inside the 'fire_detection' app's 
    # urls.py file for how to handle all incoming requests.
    path('', include('fire_detection.urls')),
]

# This part is also important for handling uploaded files later
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
