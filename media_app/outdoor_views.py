from django.shortcuts import render
from django.http import JsonResponse, FileResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
from django.utils.text import get_valid_filename
from django.db import transaction
from django_project import settings

from utils.nvlad_utils import save_query_images, new_dir_name, setup_directories
from utils.outdoor_plt.relocation_test import relocate

from datetime import datetime
import os, re
import shutil
import subprocess
import numpy as np
import cv2, numpy
import time
from lightglue.utils import read_image, resize_image
import torch
import zipfile
import tarfile
import json, os


from accelerated_features.modules.xfeat import XFeat

@csrf_exempt
def vggt_camera_locate(request):
    '''request_NVLAD_redir DOC'''
    if request.method != 'POST':
        return JsonResponse({'error': 'POST request required'}, status=400)

    # Setup paths
    img_loc = request.GET.get('source_location', 'temps/')
    img_loc = os.path.join(img_loc, 'color')
    src_loc = os.path.join(settings.MEDIA_ROOT, 'images/', img_loc) # 数据集中图片的位置
    dataset_loc = img_loc.split('/')[0] # 数据集的根目录
    req_loc = os.path.join(settings.MEDIA_ROOT, 'nvlabs/', request.GET.get('request_location', img_loc))
    
    # Process input images
    images = request.FILES.getlist('images')
    if images is None or len(images) == 0:
        images = [request.FILES.get('images')]
    if not any(images):
        return JsonResponse({'error': 'post image required'}, status=404)
    
    # Setup processing directories
    tempfolder = os.path.join(req_loc, new_dir_name('query'))
    tempfeature = os.path.join(tempfolder, 'query_features')
    tempimages = os.path.join(tempfolder, 'query_folder')
    tempquery = os.path.join(tempfolder, 'query.txt')
    
    setup_directories(req_loc, tempfolder)
    
    # Save query images
    saved_images, _ = save_query_images(images, tempimages, tempquery)
   
    pose_unity = relocate(saved_images[0], src_loc, tempfolder)
   
    return JsonResponse({
        'message': 'Folder Found',
        'saved_path': saved_images,
        'positions': pose_unity[:3,:].tolist()
    }, status=200)