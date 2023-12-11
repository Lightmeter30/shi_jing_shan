# media_app/urls.py
from django.urls import path
from .views import upload_video, upload_image

urlpatterns = [
    path('upload_video/', upload_video, name='upload_video'),
    path('upload_image/', upload_image, name='upload_image'),
]
