# media_app/views.py
import json

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
# from django.conf import settings
from django_project import settings
from .models import Image, Video
from .forms import VideoForm, ImageForm
from utils.upload import new_name, new_dir_name
from utils.calib3d import *
from utils.draw import *
from utils.nvlad_utils import *
from datetime import datetime
import os, re
import subprocess
import numpy as np
import cv2, numpy
import time
from lightglue.utils import read_image, resize_image
import torch
import copy

from accelerated_features.modules.xfeat import XFeat
# from accelerated_features.modules.xfeat import XFeat

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
  depth_image = Image.open('/home/takune/relocation/shi_jing_shan/media/images/sjs1009/depth/frame-000000.depth.jpg')
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
  print(f"Depth image shape: {depth_image.getpixel((1800,1439))}")
  
  return JsonResponse({'success': 'test_image_ok'}, status=200)
'''
重定位涉及三套坐标系：
3DS: 3D Scanner坐标系, 
Unity: Unity坐标系, 
CV2: OpenCV坐标系, 

相机外参Pose: 
c2w: Camera to World, 
w2c: World to Camera, 
'''
@csrf_exempt
def request_NVLAD_redir(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST request required'}, status=400)

    start_init = time.time()
    # 是否将P1的旋转矩阵和位移矩阵分开
    is_divide = False
    # 是否将K1和K3设置为相等
    is_K_equal = False
    # Setup paths
    img_loc = request.GET.get('source_location', 'temps/')
    img_loc = os.path.join(img_loc, 'color')
    src_loc = os.path.join(settings.MEDIA_ROOT, 'images/', img_loc)
    dataset_loc = img_loc.split('/')[0]
    
    intri_loc = os.path.join(settings.MEDIA_ROOT, 'images/',
                            request.GET.get('camera_intri_location', os.path.join(dataset_loc, 'intrinsic')))
    exter_loc = os.path.join(settings.MEDIA_ROOT, 'images/',
                            request.GET.get('camera_exter_location', os.path.join(dataset_loc, 'pose')))
    depth_loc = os.path.join(settings.MEDIA_ROOT, 'images/',
                            request.GET.get('camera_depth_location', os.path.join(dataset_loc, 'depth')))
    req_loc = os.path.join(settings.MEDIA_ROOT, 'nvlabs/', request.GET.get('request_location', img_loc))
    '''
    target_pose = os.path.join(exter_loc, 'frame-000184.pose.txt')
    target = read_pose_3dscanner(target_pose)
    return JsonResponse({
      'message': 'Folder Found',
      'saved_path': ["/home/takune/relocation/shi_jing_shan/media/nvlabs/gxl_02/color/query_20250521030653_a90b882970/query_folder/image.jpg"],
      'positions': {
        'image.jpg': target.tolist()
      }
    }, status=200)
    '''
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
    tempdebug = os.path.join(tempfolder, 'debug')
    tempquery = os.path.join(tempfolder, 'query.txt')
    
    setup_directories(req_loc, tempfolder)
    
    # Read depth settings
    depth_info = os.path.join(depth_loc, 'depth.txt')
    with open(depth_info, 'r') as f:
        Z_Far = float(f.read())
    
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
    start = time.time()
    resfolder = os.path.join(tempfolder, 'result/')
    positions = {}
    
    predictions_file = os.path.join(resfolder, 'PatchNetVLAD_predictions.txt')
    if not os.path.exists(predictions_file):
        return JsonResponse({'error': 'PatchNetVLAD failed to match'}, status=200)
    
    pred_imgs = process_predictions(predictions_file, intri_loc, exter_loc, depth_loc)
    
    # Constants for processing
    W, H = 480, 640
    # W, H = 1440, 1920
    est_K = np.array([[485, 0, 237], [0., 485, 320], [0, 0, 1]])
    # est_K = np.array([[485, 0, 320], [0., 485, 237], [0, 0, 1]])
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
            'qimname': qimname,
            'camera_3dpoints_3DS': None,
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
            'qimname': qimname,
            'camera_3dpoints_3DS': None,
            'point_valid_list': None
        }
        
        # Process query image
        qim = os.path.join(tempimages, qimname)
        image3 = process_single_image(qim, H, W)
        img = Image.fromarray(image3)
        img.save(os.path.join(tempimages, 'image3.jpg'), "JPEG")
        K3 = est_K
        # Get ground truth pose if available
        ground_truth = os.path.join(exter_loc, qimname.split('.')[0] + '.pose.txt')
        ground_P3 = read_pose_3dscanner(ground_truth) if os.path.exists(ground_truth) else None
        
        # Process each potential match
        xfeat = XFeat()
        for i in range(min(len(v), 20)):
            # Process source image
            sim1 = v[i][0]
            image1 = process_single_image(sim1, H, W)
            K1 = read_pose_3dscanner(v[i][1])[:, :-1] if os.path.exists(v[i][1]) else est_K
            K1[0, 0] *= 1/3
            K1[1, 1] *= 1/3
            K1[0, 2] *= 1/3
            K1[1, 2] *= 1/3
            if is_K_equal:
                K3 = K1
            # P1 读出来是一个3 x 4的矩阵
            P1_c2w_3DS = read_pose_3dscanner(v[i][2]) if os.path.exists(v[i][2]) else np.eye(3, 4)

            p1_shift = None
            if is_divide:
                P1_c2w_3DS, p1_shift = pose_divide(P1_c2w_3DS)
                P1_c2w_3DS = np.hstack((P1_c2w_3DS, np.zeros((P1_c2w_3DS.shape[0], 1))))
            
            P1_c2w_3DS = np.vstack((P1_c2w_3DS, np.array([0,0,0,1])))
            
            # Match features
            # kpoints1, kpoints3 = match_images_xfeat(image1, image3, xfeat)
            kpoints1, kpoints3 = match_images_lightglue(image1, image3)
            kpoints1 = np.floor(kpoints1)
            kpoints3 = np.floor(kpoints3)
            # kpoints1 = np.array([[110, 372]])
            # kpoints3 = np.array([[110, 372]])
            if kpoints1.shape[0] <= 400:
                continue
            print(f'kpoints1: {kpoints1.shape}, kpoints3: {kpoints3.shape}')
            # Convert to 3D points
            depth_image = read_image(v[i][3])  # 读取深度图
            depth_image = cv2.resize(depth_image, (depth_image.shape[1] // 3, depth_image.shape[0] // 3), interpolation=cv2.INTER_NEAREST)  # 调整深度图大小
            
            model_3dpoints, remove_list, camera_3dpoints_3DS, point_valid_list, depth_list = pixel_to_world(
                kpoints1, depth_image, K1, P1_c2w_3DS, Z_Far
            )
            kpoints1 = np.delete(kpoints1, remove_list, axis=0)
            kpoints3 = np.delete(kpoints3, remove_list, axis=0)
            print(f'kpoints1: {kpoints1.shape}, kpoints3: {kpoints3.shape}')
            if model_3dpoints.shape[0] >= 200:
                # Estimate pose
                success, pose_c2w_3DS, inliners = estimate_pose_PNPRANSAC(model_3dpoints, kpoints3, K3, P1_c2w_3DS, distCoeffs)
                if success:
                    current_results = {
                        'inliners': inliners,
                        'inliners_rate': float(len(inliners)) / float(len(kpoints3)),
                        'points2d': [kpoints1, kpoints3],
                        'points3d': model_3dpoints,
                        'K': [K1, K3],
                        'depth': v[i],
                        'depth_image': depth_image,  # 保存深度图
                        'P': [P1_c2w_3DS, pose_c2w_3DS],
                        'origin_shift': p1_shift,
                        'image_name': [sim1],
                        'keypoints': [kpoints1, kpoints3],
                        'image_RGB': [image1, image3],
                        'qimname': qimname,
                        'camera_3dpoints_3DS': camera_3dpoints_3DS,
                        'point_valid_list': point_valid_list
                    }
                    
                    update_best_results(current_results, best_results, second_best_results)
            
            torch.cuda.empty_cache()
        
        # Save results
        if best_results['P'] is not None:
            # Recalculate pose using inliers from RANSAC
            inliners_3D = best_results['points3d'][best_results['inliners']]
            inliners_2D = best_results['points2d'][1][best_results['inliners']].squeeze()
            best_results['camera_3dpoints_3DS'] = best_results['camera_3dpoints_3DS'][best_results['inliners']]
            # print inliners world 3D points and pixel 2D points
            txt_3d = os.path.join(resfolder, 'best_result_3d.txt')
            txt_2d = os.path.join(resfolder, 'best_result_2d.txt')
            txt_camera = os.path.join(resfolder, 'best_result_camera_3D.txt')
            txt_valid = os.path.join(resfolder, 'valid_key_point.txt')
            save_points_to_file(txt_2d, inliners_2D)
            save_points_to_file(txt_3d, inliners_3D)
            save_points_to_file(txt_camera, best_results['camera_3dpoints_3DS'])
            save_points_to_file(txt_valid, best_results['point_valid_list'])
            debug_evaluate_pnp_pose(inliners_3D, inliners_2D, best_results['K'][1], best_results['P'][1], tempdebug)
            # Use PNP to get more accurate pose
            success, refined_pose = estimate_pose_PNP(
                inliners_3D, 
                inliners_2D, 
                best_results['K'][1], 
                best_results['P'][0], 
                distCoeffs
            )
            
            if success:
                best_results['P'][1] = refined_pose
                if best_results['origin_shift'] is not None:
                    best_results['P'][1][:3, 3] += best_results['origin_shift']
                positions[qimname] = refined_pose.tolist()
            else:
                positions[qimname] = best_results['P'][1].tolist()
            
            save_match_visualization(best_results, second_best_results, resfolder, tempimages)
            
            if ground_P3 is not None:
                error_metrics = calculate_pose_error(ground_P3, best_results['P'])
                print(f"Pose error metrics for {qimname}:", error_metrics)
        else:
            default_P = read_pose_3dscanner(v[0][2]) if os.path.exists(v[0][2]) else np.eye(3, 4)
            positions[qimname] = default_P.tolist()
        if second_best_results['P'] is not None:
            inliners_3D = second_best_results['points3d'][second_best_results['inliners']]
            inliners_2D = second_best_results['points2d'][1][second_best_results['inliners']].squeeze()
            txt_3d = os.path.join(resfolder, 'second_best_result_3d.txt')
            txt_2d = os.path.join(resfolder, 'second_best_result_2d.txt')
            save_points_to_file(txt_2d, inliners_2D)
            save_points_to_file(txt_3d, inliners_3D)
            success, refined_pose = estimate_pose_PNP(
                inliners_3D, 
                inliners_2D, 
                second_best_results['K'][1], 
                second_best_results['P'][0], 
                distCoeffs
            )
            if success:
              second_best_results['P'][1] = refined_pose
              if second_best_results['origin_shift'] is not None:
                  second_best_results['P'][1][:3, 3] += second_best_results['origin_shift']
    end = time.time()
    total_time = end - start_init
    
    # Write results to file
    result_txt = os.path.join(resfolder, 'result.txt')
    write_results_to_file(result_txt, best_results, second_best_results, total_time)
    
    return JsonResponse({
        'message': 'Folder Found',
        'saved_path': saved_images,
        'positions': positions
    }, status=200)
