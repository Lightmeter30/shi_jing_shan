# media_app/urls.py
from django.urls import path
from .views import upload_video, upload_image, upload_multiple_images, upload_multiple_videos, request_colmap

urlpatterns = [
    path('upload_video/', upload_video, name='upload_video'),
    path('upload_image/', upload_image, name='upload_image'),
    path('upload_multiple_images/',
         upload_multiple_images,
         name='upload_multiple_images'),
    path('upload_multiple_videos/',
         upload_multiple_videos,
         name='upload_multiple_videos'),
    path('request_colmap/', request_colmap, name='request_colmap'),
]
