# media_app/urls.py
from django.urls import path

from .views import upload_datasets,upload_video, upload_image, upload_multiple_images, upload_multiple_videos, request_colmap_auto, \
    request_colmap, request_NVLAD, request_NVLAD_redir, test_read_image, get_scence_list, get_config_by_key, delete_scence, \
        get_single_file_by_id, get_multi_file_by_id, update_config

urlpatterns = [
    path('upload_datasets/', upload_datasets, name='upload_datasets'),
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
    path('test_read_image/', test_read_image, name='test_read_image'),
    path('get_scence_list/', get_scence_list, name='get_scence_list'),
    path('get_config/', get_config_by_key, name='get_config_by_key'),
    path('delete_scence/', delete_scence, name='delete_scence'),
    path('get_single_file/', get_single_file_by_id, name='get_single_file'),
    path('get_multi_file/', get_multi_file_by_id, name='get_multi_file'),
    path('update_config/', update_config, name='update_config'),
]
