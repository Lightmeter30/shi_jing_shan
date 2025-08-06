# media_app/views.py
import json
import io

from django.shortcuts import render
from django.http import JsonResponse, FileResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
from django.utils.text import get_valid_filename
from django.db import transaction
# from django.conf import settings
from django_project import settings
from .models import *
from .forms import *
from utils.upload import new_name, new_dir_name
from utils.calib3d import *
from utils.draw import *
from utils.nvlad_utils import *
from utils.times import timer
from utils.logger_config import logger
from utils.vggt_utils import *
from utils.sql import insert_one_dataset, get_config_field
from utils.exif_utils import exifori_to_unity_rotation_matrix
# from utils.image_preprocess import process_single_image

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


from accelerated_features.modules.xfeat import XFeat
# from accelerated_features.modules.xfeat import XFeat

@csrf_exempt
def upload_datasets(request):
    # if request.method == 'OPTIONS':
    #      # 响应预检请求
    #     response = JsonResponse({'message': 'OK'})
    #     response['Access-Control-Allow-Origin'] = '*'
    #     response['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
    #     response['Access-Control-Allow-Headers'] = 'Content-Type'
    #     return response

    if request.method == 'POST':
        datasets = request.FILES.getlist('datasets')
        base_dir = os.path.abspath(settings.MEDIA_ROOT)
        target_path = os.path.abspath(os.path.join(base_dir, 'images/'))
        if not target_path.startswith(base_dir):
            return JsonResponse({'error': '不允许的路径'}, status=403)
        if not datasets:
            return JsonResponse({'error': '缺少 datasets 参数'}, status=400)
        results = []
        for dataset in datasets:
            file_name = get_valid_filename(dataset.name)
            base_name, ext = os.path.splitext(file_name)
            if ext in ['.gz', '.tgz'] and file_name.endswith('.tar.gz'):
                base_name = file_name.replace('.tar.gz', '')
            elif ext == '.gz' and file_name.endswith('.tgz'):
                base_name = file_name.replace('.tgz', '')
            
            # 解压目标子目录（与文件同名）
            file_extract_dir = os.path.join(target_path, base_name)
            if Dataset.objects.filter(name=base_name).exists():
                results.append({
                    'filename': file_name,
                    'status': f'数据集 {base_name} 已存在, 请重命名后上传',
                })
                continue

            os.makedirs(file_extract_dir, exist_ok=True)
            # 保存上传的压缩包为临时文件
            temp_path = os.path.join(target_path, file_name)
            # 保存上传的压缩包
            with open(temp_path, 'wb+') as f:
                for chunk in dataset.chunks():
                    f.write(chunk)
            # 解压缩
            try:
                config_path = os.path.join(file_extract_dir, 'config.json')
                info_path = os.path.join(file_extract_dir, 'info.json')
                if file_name.endswith('.zip'):
                    with zipfile.ZipFile(temp_path, 'r') as zip_ref:
                        zip_ref.extractall(file_extract_dir)
                    if not os.path.exists(config_path) or not os.path.exists(info_path):
                        raise ValueError("缺少 config.json 或 info.json 文件")
                    insert_one_dataset(base_name, file_extract_dir, base_dir, info_path, config_path)
                    status = '解压成功'
                elif file_name.endswith(('.tar', '.tar.gz', '.tgz')):
                    with tarfile.open(temp_path, 'r:*') as tar_ref:
                        tar_ref.extractall(file_extract_dir)
                    if not os.path.exists(config_path) or not os.path.exists(info_path):
                        raise ValueError("缺少 config.json 或 info.json 文件")
                    insert_one_dataset(base_name, file_extract_dir, base_dir, info_path, config_path)
                    status = '数据集处理成功'
                else:
                    status = '不支持的格式'
            except Exception as e:
                if os.path.exists(file_extract_dir):
                    shutil.rmtree(file_extract_dir)
                status = f'数据集处理失败: {str(e)}'
            finally:
                os.remove(temp_path)  # 清理临时文件
                results.append({
                    'filename': file_name,
                    'status': status,
                })
        return JsonResponse({'message': '批量处理完成', 'results': results})
    return JsonResponse({'error': '仅支持 POST 请求'}, status=405)

@csrf_exempt
def get_scence_list(request):
    """
    获取 SCENCE 表中的所有记录
    """
    if request.method != 'GET':
        return JsonResponse({'error': 'GET request required'}, status=400)
    try:
        # 获取所有记录
        if not Dataset.objects.exists():
            return JsonResponse({'items': []}, status=200)
        records = Dataset.objects.all().values('id', 'name')
        # 字段重命名
        records = [{'scenceName': r['name'], 'scenceKey': str(r['id'])} for r in records]
        return JsonResponse({'items': records}, status=200)
    except Exception as e:
        logger.error(f"Error fetching scence list: {e}")
        return JsonResponse({'error': 'Failed to fetch scence list'}, status=500)

@csrf_exempt
def get_config_by_key(request):
    """
    根据ID获取 SCENCE 表中的 CONFIG 字段
    """
    if request.method != 'GET':
        return JsonResponse({'error': 'GET request required'}, status=400)
    
    key = int(request.GET.get('sceneKey', '-1'))
    if key == -1:
        return JsonResponse({'error': 'key parameter is required'}, status=400)
    
    try:
        # config = get_config_field(key)
        config = Dataset.objects.get(id=key).config
        if not config:
            return JsonResponse({'error': 'No records found for the given key'}, status=404)
        return JsonResponse(config, status=200)
    except Exception as e:
        logger.error(f"Error fetching config: {e}")
        return JsonResponse({'error': 'Failed to fetch config'}, status=500)

@csrf_exempt
def delete_scence(request):
    """
    根据名称删除 SCENCE 表中的记录
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'POST request required'}, status=400)
    
    name = request.POST.get('name')
    if not name:
        return JsonResponse({'error': 'name parameter is required'}, status=400)
    
    if not Dataset.objects.exists(name=name):
        logger.warning(f"No records found with NAME {name}")
        return JsonResponse({'error': f'No records found with NAME {name}'}, status=404)

    try:
        with transaction.atomic():
            Dataset.objects.delete(name=name)
        
            rm_scence_dir = os.path.join(settings.MEDIA_ROOT, 'images', name)
            rm_scence_log_dir = os.path.join(settings.MEDIA_ROOT, 'nvlabs', name)
            if os.path.exists(rm_scence_dir):
                shutil.rmtree(rm_scence_dir)
            if os.path.exists(rm_scence_log_dir):
                shutil.rmtree(rm_scence_log_dir)
            logger.info(f"Successfully deleted records with NAME {name}")
            return JsonResponse({'message': f'Successfully deleted records with NAME {name}'}, status=200)
    except Exception as e:
        logger.error(f"Error deleting records: {e}")
        return JsonResponse({'error': 'Failed to delete records'}, status=500)

@csrf_exempt
def update_config(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=400)
    key = int(request.GET.get('key', '-1'))
    if key == -1:
        return JsonResponse({'error': 'key required'}, status=400)
    new_config = json.loads(request.POST.get('config', ''))
    if new_config == '':
        return JsonResponse({'error': 'the new config is null!'}, status=400)
    try:
        target = Dataset.objects.get(id=key)
        target.old_config = target.config
        target.config = new_config
        target.save()
    except Exception as e:
        logger.error(f"the error is {e}")
        return JsonResponse({'error': f'{e}'}, status=400)
    return JsonResponse({'message': f'配置更新成功!'}, status=200)

@csrf_exempt
def get_single_file_by_id(request):
    if request.method != 'GET':
        return JsonResponse({'error': 'GET request required'}, status=400)
    key = int(request.GET.get('key', '-1'))
    if key == -1:
        return JsonResponse({'error': 'key required'}, status=400)
    try:
        dataset_file = DatasetFile.objects.get(id=key)
    except Exception as e:
        logger.error(f'error info: {e}')
        return JsonResponse({'error': 'there is no file matching the given key!'}, status=404)
    base_dir = os.path.abspath(settings.MEDIA_ROOT)
    file_path = os.path.join(base_dir, dataset_file.file_path)
    file_name = os.path.basename(file_path)
    if not os.path.exists(file_path):
        return JsonResponse({'error': 'there is no file matching the file path'}, status=404)
    return FileResponse(open(file_path, 'rb'), as_attachment=True, filename=file_name)

def get_multi_file_by_id(request):
    if request.method != 'GET':
        return JsonResponse({'error': 'GET request required'}, status=400)
    key = int(request.GET.get('key', '-1'))
    if key == -1:
        return JsonResponse({'error': 'key required'}, status=400)
    files = DatasetFile.objects.filter(dataset_id=key)
    if len(files) == 0:
        return JsonResponse({'error': 'there is no files under this scence!'}, status=404)
    base_dir = os.path.abspath(settings.MEDIA_ROOT)
    file_list = [(f.name, os.path.join(base_dir, f.file_path)) for f in files]
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        for name, file_path in file_list:
            zip_file.write(file_path, arcname=name)
    zip_buffer.seek(0)
    return FileResponse(zip_buffer, as_attachment=True, filename='archive.zip')

def upload_video(request):
    if request.method == 'POST':
        custom_location = request.GET.get('custom_location', 'videos/')
        isRename = request.GET.get('isRename', 'false').lower()
        isRename = isRename == 'true'
        form = VideoForm(request.POST, request.FILES)
        if form.is_valid():
            instance = form.save(commit=False)
            instance.video.name = new_name(custom_location,
                                           instance.video.name, isRename)
            # Assign the custom location to the image instance
            res = instance.save()
            return JsonResponse({'message': 'Video uploaded successfully', 'saved_path': res},
                                status=200)
    else:
        form = VideoForm()
    return render(request, 'upload_video.html', {'form': form})


@csrf_exempt
def upload_image(request):
    if request.method == 'POST':
        custom_location = request.GET.get('custom_location', 'images/')
        custom_location = os.path.join(settings.MEDIA_ROOT, custom_location)
        print(f"{custom_location}")
        print(custom_location)
        isRename = request.GET.get('isRename', 'false').lower()
        isRename = isRename == 'true'
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            image_instance = form.save(commit=False)
            image_instance.image.name = new_name(custom_location,
                                                 image_instance.image.name, isRename)
            # Assign the custom location to the image instance
            res = image_instance.save()
            return JsonResponse({'message': 'Image uploaded successfully', 'saved_path': res},
                                status=200)
        else:
            image = request.FILES['image']
            image_instance = Image(image=image)
            image_instance.image.name = new_name(custom_location, image.name)
            res = image_instance.save()
            return JsonResponse({'err': 'form invalid', 'saved_path': res}, status=404)
    else:
        form = ImageForm()
    return render(request, 'upload_image.html', {'form': form})


@csrf_exempt
def upload_multiple_images(request):
    if request.method == 'POST' and request.FILES.getlist('images'):
        custom_location = request.GET.get('custom_location', 'images/')
        isRename = request.GET.get('isRename', 'false').lower()
        isRename = isRename == 'true'
        res = []
        for image_file in request.FILES.getlist('images'):
            image_instance = Image(image=image_file)
            image_instance.image.name = new_name(custom_location,
                                                 image_instance.image.name, isRename)
            res.append(image_instance.save())
        return JsonResponse(
            {'message': 'Images uploaded successfully to custom location', 'saved_path': res},
            status=200)
    return JsonResponse({'error': 'POST request and images required'},
                        status=400)


def upload_multiple_videos(request):
    if request.method == 'POST' and request.FILES.getlist('videos'):
        custom_location = request.GET.get('custom_location', 'videos/')
        isRename = request.GET.get('isRename', 'false').lower()
        isRename = isRename == 'true'
        res = []
        for video_file in request.FILES.getlist('videos'):
            instance = Video(video=video_file)
            instance.video.name = new_name(custom_location,
                                           instance.video.name, isRename)
            res.append(instance.save())
        return JsonResponse({'message': 'Videos uploaded successfully', 'saved_path': res},
                            status=200)
    return JsonResponse({'error': 'POST request and videos required'},
                        status=400)


@csrf_exempt
def request_colmap_auto(request):
    if request.method == 'GET':
        request_location = request.GET.get('request_location', 'temps/')
        save_location = request.GET.get('save_location', 'temps/')
        sav_loc = os.path.join(settings.MEDIA_ROOT, 'colmaps/', save_location)
        if not os.path.exists(sav_loc):
            os.makedirs(sav_loc)
        colmap_params = request.GET.get('colmap_params', 'automatic_reconstructor')
        colmap_params = colmap_params + ' ' + request.POST.get('colmap_params', '')
        folder = os.path.join(settings.MEDIA_ROOT, 'images/', request_location)
        if os.path.exists(folder):
            command = settings.COLMAP_PATH + ' ' + colmap_params + ' --image_path ' + \
                      os.path.join(settings.MEDIA_ROOT, 'images/', request_location) + \
                      ' --workspace_path ' + \
                      os.path.join(settings.MEDIA_ROOT, 'colmaps/', save_location)
            subprocess.run(command, shell=True)
            return JsonResponse({'messag e': 'Folder found', 'saved_path': sav_loc}, status=200)
        else:
            return JsonResponse(
                {'error': 'No images found in the specified folder'},
                status=404)

    return JsonResponse({'error': 'Get request required'}, status=400)


@csrf_exempt
def request_colmap(request):
    if request.method == 'GET':
        project_location = request.GET.get('project_location', 'temps/')
        sav_loc = os.path.join(settings.MEDIA_ROOT, 'colmaps/', project_location)
        if not os.path.exists(sav_loc):
            os.makedirs(sav_loc)
        colmap_params = request.GET.get('colmap_params', '')
        colmap_params = colmap_params + ' ' + request.POST.get('colmap_params', '')
        command = 'cd ' + sav_loc + ' && ' + settings.COLMAP_PATH + ' ' + colmap_params
        subprocess.run(command, shell=True)
        return JsonResponse({'messag e': 'Folder found', 'saved_path': sav_loc}, status=200)

    return JsonResponse({'error': 'Get request required'}, status=400)


@csrf_exempt
def request_NVLAD(request):
    req_loc = request.GET.get('request_location', 'temps/')
    req_loc = os.path.join(req_loc, 'color')
    if req_loc[-1] != '/':
        req_loc = req_loc + '/'
    save_loc = os.path.join(settings.MEDIA_ROOT, 'nvlabs/', request.GET.get('save_location', req_loc))
    if save_loc[-1] != '/':
        save_loc = save_loc + '/'
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)
        os.makedirs(os.path.join(save_loc, 'index_features/'))
    netpath = settings.NetVLAD_PATH
    nv_params = request.GET.get('nv_params', '')
    # 获取处理后的原始数据集的图片 /media/images/request_location/color
    folder = os.path.join(settings.MEDIA_ROOT, 'images/', req_loc)
    if os.path.exists(folder):
        # feature extraction
        # command_conda = "source /home/vr717/anaconda3/etc/profile.d/conda.sh && conda activate patchnetvlad "
        # command_conda = command_conda + f'&& bash make_dataset_and_extract.sh {os.path.join(settings.MEDIA_ROOT, "images/", req_loc)} {save_loc} '
        # command = f'cd {netpath} && bash -c "{command_conda}"'
        command = f'cd {netpath} && bash make_dataset_and_extract.sh {os.path.join(settings.MEDIA_ROOT, "images/", req_loc)} {save_loc} '
        # subprocess.run(command, shell=True)
        os.system(command)
        return JsonResponse({'message': 'Folder Found', 'saved_path': save_loc}, status=200)
    else:
        return JsonResponse({'error': 'NO such folder'}, status=404)

@csrf_exempt
def test_read_image(request):
  # Django Image based on PIL
  depth_image = Image.open('/home/takune/relocation/shi_jing_shan/media/images/gxl_03/depth/frame-000000.depth.jpg')
  if depth_image is None:
    raise ValueError('Could not read the depth image.')
  d_width, d_height = depth_image.size
  print(f'before width x height: {d_width} x {d_height}')
  # depth_image = depth_image.rotate(90, expand=True)
  # depth_image = depth_image.transpose(Image.FLIP_TOP_BOTTOM)
  depth_image = depth_image.resize((480, 640))
  d_width, d_height = depth_image.size
  print(f'after width x height: {d_width} x {d_height}')
  
  depth_image.save('/home/takune/relocation/shi_jing_shan/media/test/test.jpg')
  # 输出深度图的基本信息
  print(f"Depth image shape: {depth_image.size}")
  print(f"Depth image shape: {depth_image.getpixel((470,630))}")
  
  return JsonResponse({'success': 'test_image_ok'}, status=200)

@timer
@csrf_exempt
def request_NVLAD_redir(request):
    '''request_NVLAD_redir DOC'''
    if request.method != 'POST':
        return JsonResponse({'error': 'POST request required'}, status=400)

    # 是否将P1的旋转矩阵和位移矩阵分开
    is_divide = False
    # 是否将K1和K3设置为相等
    is_K_equal = True
    # 测试时使用, 计算PnPRANSAC结果与GT pose的误差
    error_metrics = None
    # 是否打印debug信息
    is_debug = True
    # 是否打印长txt文本的中间信息(best_result_2d.txt best_result_3d.txt best_result_camera_3D.txt valid_key_point.txt)
    is_write_long_txt = True
    # 是否提升效率，提前退出循环
    is_accelerate = False
    # Setup paths
    img_loc = request.GET.get('source_location', 'temps/')
    img_loc = os.path.join(img_loc, 'color')
    src_loc = os.path.join(settings.MEDIA_ROOT, 'images/', img_loc)
    dataset_loc = img_loc.split('/')[0] # 数据集的根目录
    
    intri_loc = os.path.join(settings.MEDIA_ROOT, 'images/',
                            request.GET.get('camera_intri_location', os.path.join(dataset_loc, 'intrinsic')))
    exter_loc = os.path.join(settings.MEDIA_ROOT, 'images/',
                            request.GET.get('camera_exter_location', os.path.join(dataset_loc, 'pose')))
    depth_loc = os.path.join(settings.MEDIA_ROOT, 'images/',
                            request.GET.get('camera_depth_location', os.path.join(dataset_loc, 'depth')))
    req_loc = os.path.join(settings.MEDIA_ROOT, 'nvlabs/', request.GET.get('request_location', img_loc))
    dataset_info = os.path.join(settings.MEDIA_ROOT, 'images/', dataset_loc, 'info.json')
    with open(dataset_info, 'r') as f:
        dataset_info = json.load(f)
    dataset_K = np.array([[dataset_info['intrinsic']['fx'], 0, dataset_info['intrinsic']['cx']],
                          [0, dataset_info['intrinsic']['fy'], dataset_info['intrinsic']['cy']],
                          [0, 0, 1]])
    dataset_H = dataset_info['image_size']['height']
    dataset_W = dataset_info['image_size']['width']
    dataset_EXIF = dataset_info['exif']
    # TODO: 后续TARGET最好作为参数从前端传过来
    TARGET = {'X': 'right', 'Y': 'up', 'Z': 'forward'}
    M_DATASET_CV2 = compute_M_A2B(dataset_info['coordinate'])
    M_CV2_TARGET = compute_M_A2B({'X': 'right', 'Y': 'down', 'Z': 'forward'}, TARGET)
    M_DATASET_TARGET = compute_M_A2B(dataset_info['coordinate'], TARGET)
    '''
    target_pose_path = os.path.join(exter_loc, 'frame-000055.pose.txt')
    # target_image_path = os.path.join(img_loc, 'frame-000000.color.jpg')
    target_image_path = '/home/takune/relocation/shi_jing_shan/media/images/gxl_03/color/frame-000000.color.jpg'
    UNITY_ROTATION_MATRIX = exifori_to_unity_rotation_matrix(dataset_EXIF)
    target_DATASET = read_pose_3dscanner(target_pose_path)
    target_DATASET = np.vstack((target_DATASET, np.array([0,0,0,1])))
    logger.info(f"target_DATASET: {target_DATASET}")
    target_CV2 = transfer_Pose_from_A2B(target_DATASET, M_DATASET_CV2)
    target = transfer_Pose_from_A2B(target_CV2, M_CV2_TARGET)
    target = UNITY_ROTATION_MATRIX @ target # Apply the unity rotation matrix
    return JsonResponse({
      'message': 'Folder Found',
      'saved_path': ["/home/takune/relocation/shi_jing_shan/media/nvlabs/gxl_02/color/query_20250521030653_a90b882970/query_folder/image.jpg"],
      'positions': {
        'image.jpg': target[:3,:].tolist()
      }
    }, status=200)
    # '''
    if req_loc[-1] != '/':
        req_loc = req_loc + '/'
    if src_loc[-1] != '/':
        src_loc = src_loc + '/'
        
    if not os.path.exists(req_loc) or not os.path.exists(src_loc):
        return JsonResponse({'error': 'No such folder'}, status=404)
    
    # Setup processing directories
    tempfolder = os.path.join(req_loc, new_dir_name('query'))
    tempfeature = os.path.join(tempfolder, 'query_features')
    tempimages = os.path.join(tempfolder, 'query_folder')
    tempquery = os.path.join(tempfolder, 'query.txt')
    
    setup_directories(req_loc, tempfolder)
    
    # Process input images
    images = request.FILES.getlist('images')
    if images is None or len(images) == 0:
        images = [request.FILES.get('images')]
    if not any(images):
        return JsonResponse({'error': 'post image required'}, status=404)
    
    # Save query images
    camera_matrix = request.POST.get('camera_matrix')
    if camera_matrix is not None:
        camera_matrix = json.loads(camera_matrix)
    saved_images, qintrinsic = save_query_images(images, tempimages, tempquery, camera_matrix)
    
    # Run NetVLAD matching
    command = f'cd {settings.NetVLAD_PATH} && bash match_and_cal_pose.sh {req_loc} {src_loc} {tempfolder}'
    os.system(command)
    
    # Process results
    resfolder = os.path.join(tempfolder, 'result/')
    positions = {}
    
    predictions_file = os.path.join(resfolder, 'PatchNetVLAD_predictions.txt')
    if not os.path.exists(predictions_file):
        return JsonResponse({'error': 'PatchNetVLAD failed to match'}, status=200)
    
    pred_imgs = process_predictions(predictions_file, intri_loc, exter_loc, depth_loc)
    
    # TODO: 根据前端采集的图片分辨率，修改W, H
    distCoeffs = None
    
    # Process each query image
    for qimname, v in pred_imgs.items():
        # Initialize result containers
        best_results = {
            'inliners': np.array([]),
            'inliners_rate': 0,
            'points2d': None,
            'points3d': np.array([]),
            'K': None,
            'depth': None,
            'depth_image': None,
            'P': None,
            'origin_shift': None,
            'image_name': None,
            'keypoints': None,
            'image_RGB': None,
            'data_image_name': None,
            'qimname': qimname,
            'camera_3dpoints_DATASET': None,
            'point_valid_list': None,
        }
        
        second_best_results = {
            'inliners': np.array([]),
            'inliners_rate': 0,
            'points2d': None,
            'points3d': None,
            'K': None,
            'depth': None,
            'depth_image': None,
            'P': None,
            'origin_shift': None,
            'image_name': None,
            'keypoints': None,
            'image_RGB': None,
            'data_image_name': None,
            'qimname': qimname,
            'camera_3dpoints_DATASET': None,
            'point_valid_list': None
        }
        
        is_break = False
        inliners_lambda = 1.0 # 调和内点率和内点数的占比
        # Process query image
        qim = os.path.join(tempimages, qimname)

        image3 = process_single_image(qim, dataset_EXIF ,is_resize=False)
        H, W = image3.shape[0], image3.shape[1]
        print(f"image3 H: {H}, W: {W}")
        if is_debug:
            img = Image.fromarray(image3)
            img.save(os.path.join(tempimages, 'image3.jpg'), "JPEG")

            
        # TODO: 根据前端采集的图片分辨率，修改K3
        K3 = np.array([[485, 0, 237], [0., 485, 320], [0, 0, 1]])
        # Get ground truth pose if available
        if is_K_equal:
            ground_truth = os.path.join(exter_loc, qimname.split('.')[0] + '.pose.txt')
            ground_P3 = read_pose_3dscanner(ground_truth) if os.path.exists(ground_truth) else None
        
        # Process each potential match
        xfeat = XFeat()
        for i in range(min(len(v), 20)):
            # Process source image
            sim1 = v[i][0]
            image1 = process_single_image(sim1, dataset_EXIF, H, W, is_resize = dataset_info["type"] != "VGGT")
            UNITY_ROTATION_MATRIX = exifori_to_unity_rotation_matrix(dataset_EXIF)
            print(f"image1 shape 0: {image1.shape[0]}, shape 1: {image1.shape[1]}")
            K1 = read_pose_3dscanner(v[i][1])[:, :-1] if os.path.exists(v[i][1]) else dataset_K
            scale = W / dataset_W
            print(f"scale: {scale}")
            K1[0, 0] *= scale # fx
            K1[1, 1] *= scale # fy
            K1[0, 2] *= scale # cx
            K1[1, 2] *= scale # cy
            if is_K_equal:
                K3 = K1
            # P1 读出来是一个3 x 4的矩阵
            P1_c2w_DATASET = read_pose_3dscanner(v[i][2]) if os.path.exists(v[i][2]) else np.eye(3, 4)

            p1_shift = None
            if is_divide:
                P1_c2w_DATASET, p1_shift = pose_divide(P1_c2w_DATASET)
                P1_c2w_DATASET = np.hstack((P1_c2w_DATASET, np.zeros((P1_c2w_DATASET.shape[0], 1))))
            
            P1_c2w_DATASET = np.vstack((P1_c2w_DATASET, np.array([0,0,0,1])))
            
            # Match features
            # kpoints1, kpoints3 = match_images_xfeat(image1, image3, xfeat)
            kpoints1, kpoints3 = match_images_lightglue(image1, image3)
            kpoints1 = np.floor(kpoints1)
            kpoints3 = np.floor(kpoints3)
            if kpoints1.shape[0] <= 100:
                logger.info(f"the data image is {v[i][0].split('/')[-1]}; the key points match number is {kpoints1.shape[0]} <= 200, which means the kp match is too low!")
                continue

            # 求3d点
            if dataset_info["type"] == "VGGT":
                # TODO:
                pointmap_loc = os.path.join(settings.MEDIA_ROOT, 'images/', os.path.join(dataset_loc, 'pointmap'))
                image_index = os.path.basename(sim1).split('.')[0]
                points_int = kpoints1.astype(np.int32)
                model_3dpoints_DATASET = read_point_map_from_vggt(points_int, image_index, pointmap_loc)
                # 这一行只是为了不报错写的
                camera_3dpoints_DATASET = model_3dpoints_DATASET
                point_valid_list = points_int
                depth_image = image1
            elif dataset_info["type"] == "3DS":
                # Convert to 3D points
                depth_image = read_image_and_remove_exif(v[i][3], dataset_EXIF)  # 读取深度图
                print(f"depth_image shape 0: {depth_image.shape[0]}, shape 1: {depth_image.shape[1]}")
                depth_image = cv2.resize(depth_image, (W, H), interpolation=cv2.INTER_NEAREST)  # 调整深度图size
                model_3dpoints_DATASET, remove_list, camera_3dpoints_DATASET, point_valid_list = pixel_to_model(
                kpoints1, depth_image, K1, P1_c2w_DATASET, dataset_info['Z_Far'], dataset_info['coordinate']
            )
                kpoints1 = np.delete(kpoints1, remove_list, axis=0)
                kpoints3 = np.delete(kpoints3, remove_list, axis=0)
            
            if model_3dpoints_DATASET.shape[0] >= 200:
                # Estimate pose
                if is_debug:
                    logger.info(f'the data base image name: {v[i][0].split("/")[-1]}')
                success, pose_c2w_target, inliners = estimate_pose_PNPRANSAC(model_3dpoints_DATASET, kpoints3, K3, P1_c2w_DATASET, M_DATASET_CV2, M_CV2_TARGET, UNITY_ROTATION_MATRIX,distCoeffs, is_debug, is_K_equal)
                if success:
                    current_results = {
                        'inliners': inliners,
                        'inliners_rate': float(len(inliners)) / float(len(model_3dpoints_DATASET)),
                        'points2d': [kpoints1, kpoints3],
                        'points3d': model_3dpoints_DATASET,
                        'K': [K1, K3],
                        'depth': v[i],
                        'depth_image': depth_image,  # 保存深度图
                        'P': [P1_c2w_DATASET, pose_c2w_target],
                        'origin_shift': p1_shift,
                        'image_name': [sim1],
                        'keypoints': [kpoints1, kpoints3],
                        'image_RGB': [image1, image3],
                        'data_image_name': v[i][0].split("/")[-1],
                        'qimname': qimname,
                        'camera_3dpoints_DATASET': camera_3dpoints_DATASET,
                        'point_valid_list': point_valid_list
                    }
                    
                    is_break = update_best_results(current_results, best_results, second_best_results, inliners_lambda=inliners_lambda)
            
            torch.cuda.empty_cache()
            if is_accelerate and is_break:
                break
        
        # Save results
        if best_results['P'] is not None:
            # Recalculate pose using inliers from RANSAC
            inliners_3D = best_results['points3d'][best_results['inliners']]
            inliners_2D = best_results['points2d'][1][best_results['inliners']].squeeze()
            best_results['camera_3dpoints_DATASET'] = best_results['camera_3dpoints_DATASET'][best_results['inliners']]
            if is_debug:
                # print inliners world 3D points and pixel 2D points
                if is_write_long_txt:
                    txt_3d = os.path.join(resfolder, 'best_result_3d.txt')
                    txt_2d = os.path.join(resfolder, 'best_result_2d.txt')
                    txt_camera = os.path.join(resfolder, 'best_result_camera_3D.txt')
                    txt_valid = os.path.join(resfolder, 'valid_key_point.txt')

                    debug_save_points_to_file(txt_2d, inliners_2D)
                    debug_save_points_to_file(txt_3d, inliners_3D)
                    debug_save_points_to_file(txt_camera, best_results['camera_3dpoints_DATASET'])
                    debug_save_points_to_file(txt_valid, best_results['point_valid_list'])
                # debug_evaluate_pnp_pose(inliners_3D, inliners_2D, best_results['K'][1], best_results['P'][1], M_DATASET_CV2, M_CV2_TARGET)
            # Use PNP to get more accurate pose
            if best_results['origin_shift'] is not None:
                best_results['P'][1][:3, 3] += best_results['origin_shift']
            positions[qimname] = best_results['P'][1].tolist()
            
            save_match_visualization(best_results, second_best_results, resfolder, tempimages, is_debug)
            
            if is_debug and is_K_equal and ground_P3 is not None:
                error_metrics = calculate_pose_error(ground_P3, best_results['P'], M_DATASET_TARGET)
        else:
            default_P = read_pose_3dscanner(v[0][2]) if os.path.exists(v[0][2]) else np.eye(3, 4)
            positions[qimname] = default_P.tolist()

        if is_debug and second_best_results['P'] is not None:
            inliners_3D = second_best_results['points3d'][second_best_results['inliners']]
            inliners_2D = second_best_results['points2d'][1][second_best_results['inliners']].squeeze()
            txt_3d = os.path.join(resfolder, 'second_best_result_3d.txt')
            txt_2d = os.path.join(resfolder, 'second_best_result_2d.txt')
            debug_save_points_to_file(txt_2d, inliners_2D)
            debug_save_points_to_file(txt_3d, inliners_3D)
            if second_best_results['origin_shift'] is not None:
              second_best_results['P'][1][:3, 3] += second_best_results['origin_shift']
            
            if is_debug and is_K_equal and ground_P3 is not None:
                error_metrics = calculate_pose_error(ground_P3, second_best_results['P'], M_DATASET_TARGET)
    
    # Write results to file
    result_txt = os.path.join(resfolder, 'result.txt')
    write_results_to_file(result_txt, best_results, second_best_results, error_metrics)
    
    return JsonResponse({
        'message': 'Folder Found',
        'saved_path': saved_images,
        'positions': positions
    }, status=200)
