# media_app/urls.py
from django.urls import path

from .views import upload_video, upload_image, upload_multiple_images, upload_multiple_videos, request_colmap_auto, \
    request_colmap, request_NVLAD, request_NVLAD_redir

urlpatterns = [
    path('upload_video/', upload_video, name='upload_video'),
    path('upload_image/', upload_image, name='upload_image'),
    path('upload_multiple_images/',
         upload_multiple_images,
         name='upload_multiple_images'),
    path('upload_multiple_videos/',
         upload_multiple_videos,
         name='upload_multiple_videos'),
    path('request_colmap_auto/', request_colmap_auto, name='request_colmap_auto'),
    path('request_colmap/', request_colmap, name='request_colmap'),
    path('request_NVLAD/', request_NVLAD, name='request_NVLAD'),
    path('request_NVLAD_redir/', request_NVLAD_redir, name='request_NVLAD_redir'),

]
