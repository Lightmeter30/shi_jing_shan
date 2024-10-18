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
from datetime import datetime
import os, re
import subprocess
import numpy as np
import cv2, numpy
import time
from PIL import Image
from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd, read_image, numpy_image_to_torch, resize_image
from lightglue import viz2d
import torch


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


def image_transform(image: numpy):
    (height, width) = image.shape[:2]
    if height >= width:
        return image
    # rotate the src image 90 degrees clockwise
    rotated_image = cv2.transpose(image)
    # rotated_image = cv2.flip(rotated_image, 1)
    return rotated_image

'''
points: image1所有的特征点集合, 是一个shape为N1 x 2的数组
matchs: image1和image3的特征点匹配对, 是一个N2 x 2的数组, 其中match[0]为image1的index, match[1]时image3的index
depth: 深度图路径字符串
K: 相机内参 3x3
P: 相机位姿4x4
return point3D Nx4(齐次坐标)
'''
def pixel_to_world(points: numpy, matchs: numpy,depth, K: numpy, P: numpy):
  # TODO: may change
  Z_Near, Z_Far = 0.5, 8
  # K变换单位
  points3D = []
  camera_coords_list = []
  matchs_new = []
  depth_list = []
  # TODO: may change
  depth_image = Image.open(depth)
  if depth_image is None:
    raise ValueError('Could not read the depth image.')
  # depth_image = depth_image.rotate(90, expand=True)
  # depth_image = depth_image.transpose(Image.FLIP_TOP_BOTTOM)
  depth_image = depth_image.resize((480, 640))
  # print(f"Depth image shape: {depth_image.size}")
  
  # 构造齐次像素坐标Nx3
  points = np.hstack((points, np.ones((points.shape[0], 1))))
  # 计算相机位姿矩阵的逆矩阵
  P_inv = np.linalg.inv(P)
  for match in matchs:
    # TODO: may change将像素坐标转化为相机坐标 
    point = points[match[0]]
    # print(f'(x, y): ({point[1]}, {point[0]})')
    depth_temp = depth_image.getpixel((point[0], point[1]))[0]
    # depth_temp = Z_Near + (depth_image.getpixel((point[1], point[0]))[0] / 255) * ()
    if depth_temp != 255 and depth_temp != 0:
      matchs_new.append(match)
      depth_final = Z_Near + (depth_temp / 255) * (Z_Far - Z_Near)
      depth_list.append(depth_final)
      # print(f'depth output: {depth_final}')
      x = (point[0] - K[0,2]) * depth_final / K[0, 0]
      y = (point[1] - K[1,2]) * depth_final / K[1, 1]
      z = depth_final
      camera_coords = np.array([x, y, z, 1])
      camera_coords_list.append(camera_coords)
    else:
      # print(f'illegal depth: {depth_temp}')
      continue
    world_coords = P_inv @ camera_coords.T
    # print(f"world_coords: {world_coords}")
    # print(f"u, v: {point[1]} {point[0]}")
    # break
    points3D.append(world_coords)
  print(f'3d+match{np.array(points3D).shape}, {np.array(matchs_new).shape}')
  return [np.array(points3D), np.array(matchs_new), np.array(camera_coords_list), np.array(depth_list)]

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

@csrf_exempt
def request_NVLAD_redir(request):
    # saved_images = ['/media/vr717/image.jpg']
    # positions = {
    #     'image.jpg': [
    #         [0.022384, 0.993749, 0.109367, 2.965258],
    #         [-0.998496, 0.016746, 0.052199, -0.169641],
    #         [0.050041, -0.110371, 0.992630, 2.639569],
    #         ]
    # }
    # [0.000000, 0.000000, 0.000000, 1.000000]
    # return JsonResponse({'message': 'Folder Found', 'saved_path': saved_images, 'positions': positions}, status=200)
    start_init = time.time()
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
    if req_loc[-1] != '/':
        req_loc = req_loc + '/'
    if src_loc[-1] != '/':
        src_loc = src_loc + '/'
    if not os.path.exists(req_loc) or not os.path.exists(src_loc):
        return JsonResponse({'error': 'No such folder'}, status=404)
    nv_params = request.GET.get('nv_params', '')
    if request.method == 'POST':
        tempfolder = os.path.join(req_loc, new_dir_name('query'))
        tempfeature = os.path.join(tempfolder, 'query_features')
        tempimages = os.path.join(tempfolder, 'query_folder')
        tempquery = os.path.join(tempfolder, 'query.txt')
        if not os.path.exists(tempfolder):
            os.makedirs(tempfolder)
            os.makedirs(tempfeature)
            os.makedirs(tempimages)
        # get images from body.images
        images = request.FILES.getlist('images')

        if images is None or len(images) == 0:
            print('single image')
            images = [request.FILES.get('images')]
        if not any(images):
            return JsonResponse({'error': 'post image required'}, status=404)
        storage_path = tempimages

        saved_images = []
        K = np.array([[1428.643433, 0.000000, 1428.643433],
                      [0.000000, 970.724121, 716.204285],
                      [0.000000, 0.000000, 1.000000]])
        camera_matrix = request.POST.get('camera_matrix')
        if camera_matrix is not None:
            camera_matrix = json.loads(camera_matrix)
        # q相机内参
        qintrinsic = {}

        with open(tempquery, 'w') as qtxt:
            for i, image in enumerate(images):
                image_name, _ = os.path.splitext(image.name)
                image_name = image_name + '.jpg'
                if camera_matrix is not None and i < len(camera_matrix):
                    qintrinsic[image_name] = np.array(camera_matrix[i])
                # qintrinsic[image.name] = np.array(camera_matrix[i]) if camera_matrix and i < len(camera_matrix) else K
                fs = FileSystemStorage(location=storage_path)
                saved_image = fs.save(image_name, image)
                saved_images.append(
                    save_to_jpg(os.path.join(storage_path, saved_image), os.path.join(storage_path, image_name)))
                qtxt.write(image_name + '\n')


        # 此时图片已经被逆时针翻转90°
        # TODO: magic code return 000000
        # default_im = os.path.join(exter_loc, 'frame-000035.pose.txt')
        # default_P = read_pose_3dscanner(default_im)
        # print(default_im)
        # return JsonResponse({'message': 'Folder Found', 'saved_path': saved_images, 'positions': {'image.jpg': default_P.tolist()}}, status=200)

        # command_conda = "source /home/vr717/anaconda3/etc/profile.d/conda.sh && conda activate patchnetvlad "
        # command_conda = command_conda + f'&& bash match_and_cal_pose.sh {req_loc} {src_loc} {tempfolder} '
        # command_conda = f'bash match_and_cal_pose.sh {req_loc} {src_loc} {tempfolder} '
        # command = f'cd {settings.NetVLAD_PATH} && bash -c "{command_conda}"'
        command = f'cd {settings.NetVLAD_PATH} && bash match_and_cal_pose.sh {req_loc} {src_loc} {tempfolder}'

        # subprocess.run(command, shell=True)
        os.system(command)

        start = time.time()
        resfolder = os.path.join(tempfolder, 'result/')
        positions = {}

        pred_pattern = re.compile(r',\s*')
        pred_imgs = {}

        if not os.path.exists(os.path.join(resfolder, 'PatchNetVLAD_predictions.txt')):
            return JsonResponse({'error': 'PatchNetVLAD failed to match'}, status=200)
        with open(os.path.join(resfolder, 'PatchNetVLAD_predictions.txt'), 'r') as qtxt:
            for i, line in enumerate(qtxt, start=1):
                if not line.startswith('#'):
                    ims = re.split(pred_pattern, line.strip())
                    ims[0] = ims[0].strip()
                    ims[1] = ims[1].strip()
                    _, qimname = os.path.split(ims[0])
                    _, simname = os.path.split(ims[1])
                    if qimname not in pred_imgs.keys():
                        pred_imgs[qimname] = [
                            (ims[1], os.path.join(intri_loc, simname.split('.')[0] + '.intrinsic_color.txt'),
                             os.path.join(exter_loc, simname.split('.')[0] + '.pose.txt'),
                             os.path.join(depth_loc, simname.split('.')[0] + '.depth.jpg'))]
                    else:
                        pred_imgs[qimname].append(
                            (ims[1], os.path.join(intri_loc, simname.split('.')[0] + '.intrinsic_color.txt'),
                             os.path.join(exter_loc, simname.split('.')[0] + '.pose.txt'),
                             os.path.join(depth_loc, simname.split('.')[0] + '.depth.jpg')))

        feature_extractor = SuperPoint(max_num_keypoints=2048).eval().to(settings.DEVICE)  # load the extractor
        feature_match = LightGlue(features="superpoint").eval().to(settings.DEVICE)
        distCoeffs = None
        useFilter = False
        filter_num = 100
        filter_params = {'distCoeffs1': None, 'distCoeffs2': None, 'threshold': 8., 'prob': 0.99, 'no_intrinsic': True}
        drawMatch = True
        timeout = 40
        W = 480
        H = 640
        est_focal = np.sqrt(W ** 2 + H ** 2) * 1428.643433 / 1440
        est_K = np.array([[est_focal, 0, W / 2.], [0, est_focal, H / 2.], [0, 0, 1]])
        for qimname, v in pred_imgs.items():
            qim = os.path.join(tempimages, qimname)
            image3 = read_image(qim)
            print(qim)
            print(f"before image3 shape is : {image3.shape}")
            image3 = image_transform(image3)
            image3, _ = resize_image(image3, (H, W))
            print(f'after images3-shape: {image3.shape}')
            image3 = np.uint8(image3)
            img = Image.fromarray(image3)
            img.save(tempimages + '/image3.jpg', "JPEG")
            # TODO: may change
            # fs = FileSystemStorage('/home/takune/relocation/shi_jing_shan/media/images/sjs1009/depth/')
            # saved_image = fs.save('test.jpg', image3)
            
            # return JsonResponse({'success': 'test_image_ok'}, status=200)
            # image3 = cv2.resize(image3, (W, H))
            K3 = est_K
            # K3 = np.array([[968.857117, 0, 240.0], [0, 1337.749634, 320], [0, 0, 1]])
            # print(f'image3 intrinsic:{K3}')
            # print(f'image3 shape:{image3.shape}')

            best_inliners = np.array([])
            best_inliners_rate = 0
            min_residuals_norm = np.inf
            stop_residuals_norm = 0.1
            best_inliners_rate_window = 0.1
            best_image_name = None
            best_image_RGB = None
            best_keypoints = None
            best_description = None
            best_tracks = None
            best_depth = None
            best_match = None
            best_points2d = None
            best_points3d = np.array([])
            best_P = None
            best_K = None
            default_P = read_pose_3dscanner(v[0][2]) if os.path.exists(v[0][2]) else np.eye(3, 4)
            ground_truth = os.path.join(exter_loc, qimname.split('.')[0] + '.pose.txt')
            ground_P3 = read_pose_3dscanner(ground_truth) if os.path.exists(ground_truth) else default_P
            ground_truth = ''
            is_stop = False
            stop_inliner_rate = 0.96
            use_DST_inliner_rate = 0.5
            min_traverse_windows = 10
            init_traverse_windows = 30
            add_traverse_windows = 1.2

            for i in range(0, len(v) - 1):
                success = False
                pose = None
                traverse_windows = init_traverse_windows
                meet_best = False

                sim1 = v[i][0]
                image1 = read_image(sim1)
                image1 = image_transform(image1)
                image1, _ = resize_image(image1, (H, W))
                # save image1
                # image1 = np.uint8(image1)
                # img = Image.fromarray(image1)
                # img.save(tempimages + '/image1.jpg', "JPEG")
                # return JsonResponse({'message': 'SHIT'}, status=200)
                # image1 = cv2.resize(image1, (W, H))
                # TODO: may change print(f'K1: {read_pose_3dscanner(v[i][1])[:, :-1]}')
                # K1 = read_pose_3dscanner(v[i][1])[:, :-1] if os.path.exists(v[i][1]) else est_K
                # K1 = np.array([[968.857117, 0, 240.0], [0, 1337.749634, 320], [0, 0, 1]])
                K1 = K3
                # K1 = K1 / 1000
                # K1 = est_K
                # TODO: print(f'image1 intrinsic:{K1}')
                # print(f'image1 shape:{image1.shape}')
                P1 = read_pose_3dscanner(v[i][2]) if os.path.exists(v[i][2]) else np.eye(3, 4)
                # print(f'image1 pose:{P1}')
                

                feats1 = feature_extractor.extract(numpy_image_to_torch(image1).to(settings.DEVICE))
                feats1out = rbd(feats1)
                kp1 = feats1out['keypoints'].cpu().numpy()
                feats3 = feature_extractor.extract(numpy_image_to_torch(image3).to(settings.DEVICE))
                feats3out = rbd(feats3)
                kp3 = feats3out['keypoints'].cpu().numpy()
                matches13 = feature_match({"image0": feats1, "image1": feats3})
                matches13out = rbd(matches13)
                good_matches13 = matches13out['matches'].cpu().numpy()
                if useFilter:
                    m13, num13 = getInliners(kp1, kp3, good_matches13, K1, K3, **filter_params)
                    if num13 > filter_num:
                        good_matches13 = np.array(m13)
                
                # 根据13匹配点集计算
                if good_matches13.size < 10:
                  print("common points too low, pose est failed!")
                  end = time.time()
                  if end - start_init > timeout:
                    print("time out 3")
                    break
                  continue
                points3d, good_matches13, camera_coords_list, depth_list = pixel_to_world(kp1, good_matches13, v[i][3], K1, np.vstack((P1, np.array([0,0,0,1]))))
                # print(f'image1 pose:{P1}')
                # return JsonResponse({'message': 'test read P1'}, status=200)
                if good_matches13.shape[0] == 0:
                  continue
                points1 = kp1[good_matches13[:, 0]].reshape(-1, 1, 2)
                points3 = kp3[good_matches13[:, 1]].reshape(-1, 1, 2)
                # N x 3
                points3d = cv2.convertPointsFromHomogeneous(points3d).squeeze()
                print(f'after cv2 point3 shape:{points3.shape}')
                print(f'after cv2 point3d shape:{points3d.shape}')
                
                if points3d.shape[0] >= 100:
                  rot_vec1, _ = cv2.Rodrigues(P1[:3, :3])
                  shift1 = P1[:3, 3:]
                  # print(f'before solvePnPRansac R:{rot_vec1}')
                  # print(f'before solvePnPRansac T: {shift1}')
                  success, R, T, inliners = cv2.solvePnPRansac(points3d, points3, K3, distCoeffs,
                                                               useExtrinsicGuess=True, rvec=rot_vec1,
                                                               tvec=shift1)
                  # print(f'after solvePnPRansac R:{R}')
                  # print(f'after solvePnPRansac T: {T}')
                  # print(f'after solvePnPRansac image1 pose:{P1}')
                  # print(f'K3: {K3}')
                  # return JsonResponse({'message': 'test read P1'}, status=200)
                  if success and inliners is not None:
                    inliners = inliners.squeeze()
                    # print(f'inliner num:{inliners.shape}')
                    Rtmp, _ = cv2.Rodrigues(R)
                    pose = np.hstack((Rtmp, T))
                    print(f'after solvePnPRANSAC pose: {pose}')
                    residuals = ground_P3 - pose
                    if len(inliners) >= 100 and (
                            len(inliners) > (best_inliners_rate + best_inliners_rate_window) * len(points3) \
                            or (best_inliners_rate - best_inliners_rate_window) * len(points3) < len(inliners) \
                            and len(best_inliners) < len(inliners)) \
                            or len(best_inliners) < len(inliners) < 100 \
                            or os.path.exists(ground_truth) and np.linalg.norm(residuals) < min_residuals_norm:
                      print('found best')
                      best_inliners = inliners
                      best_inliners_rate = float(len(inliners)) / float(len(points3))
                      best_points2d = [points1, points3]
                      best_points3d = points3d
                      best_K = [K1, K3]
                      best_depth = v[i]
                      # Rtmp, _ = cv2.Rodrigues(R)
                      # pose = np.hstack((Rtmp, T))
                      best_P = [P1, pose]
                      best_image_name = [sim1]
                      best_keypoints = [kp1, kp3]
                      best_image_RGB = [image1, image3]
                      # best_match = matches123
                      best_match = good_matches13
                      # TODO: may change
                      # is_stop = best_inliners_rate > stop_inliner_rate
                      meet_best = True
                      traverse_windows = init_traverse_windows if best_inliners_rate >= 0.2 else 0.8 * traverse_windows
                      print('get best and break')
                      break
                      if os.path.exists(ground_truth) and np.linalg.norm(residuals) < min_residuals_norm:
                        min_residuals_norm = np.linalg.norm(residuals)
                        # TODO: may change
                        # is_stop = min_residuals_norm < stop_residuals_norm
                    elif len(inliners) < 0.2 * len(points3):
                            # print('too less inliners')
                            traverse_windows *= 0.95 if meet_best else 0.5
                    else:
                      traverse_windows *= 0.97 if meet_best else 0.8
                  else:
                    traverse_windows *= 0.95 if meet_best else 0.5
                else:
                  traverse_windows *= 0.95 if meet_best else 0.5
                  success = False

                torch.cuda.empty_cache()
                # if i == len(v) - 1 or end - start_init > timeout:
                    # if end - start_init > timeout:
                    #     print('best_ratio failed')
                    # if best_P is not None:
                    #     print(f'best inliner num:{len(best_inliners)}')
                    #     print(f'points3d num:{len(best_points3d)}')
                    #     useRANSAC = False
                    #     tmp_inliners = best_inliners if len(best_inliners) > 20 else np.arange(len(best_points3d))
                    #     # RANSAC pnp
                    #     rot_vec1, _ = cv2.Rodrigues(best_P[0][:3, :3])
                    #     shift1 = best_P[0][:3, 3:]
                    #     success0, R0, T0 = cv2.solvePnP(best_points3d, best_points2d[1].squeeze(), K3, distCoeffs,
                    #                                         useExtrinsicGuess=True, rvec=rot_vec1, tvec=shift1)
                    #     if success0:
                    #         Rtmp, _ = cv2.Rodrigues(R0)
                    #         pose = np.hstack((Rtmp, T0))
                    #         best_P[1] = pose
                    #     positions[qimname] = best_P[1].tolist()
                    #     if drawMatch:
                    #         dmatch13 = [cv2.DMatch(m[0], m[2], 0) for m in best_match[best_inliners]]
                    #         bkp1 = [cv2.KeyPoint(kp[0], kp[1], 1, -1, 0, 0, -1) for kp in best_keypoints[0]]
                    #         bkp3 = [cv2.KeyPoint(kp[0], kp[1], 1, -1, 0, 0, -1) for kp in best_keypoints[1]]
                    #         img_with_key13 = cv2.drawMatches(best_image_RGB[0], bkp1, best_image_RGB[1],
                    #                                          bkp3, dmatch13, None)
                    #         compression_params = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
                    #         cv2.imwrite(os.path.join(resfolder,
                    #                                  'match_' + os.path.basename(best_image_name[0]).split('.')[
                    #                                      0] + qimname), img_with_key13,
                    #                     compression_params)
                    # else:
                    #     positions[qimname] = default_P.tolist()
                    #     print("all pose est failed")
                    # break

            end = time.time()
            if end - start_init > timeout:
              print('best_ratio failed')
            if best_P is not None:
                # print(f'best inliner num:{len(best_inliners)}')
                # print(f'points3d num:{len(best_points3d)}')
                # useRANSAC = False
                # tmp_inliners = best_inliners if len(best_inliners) > 20 else np.arange(len(best_points3d))
                # # RANSAC pnp
                # rot_vec1, _ = cv2.Rodrigues(best_P[0][:3, :3])
                # shift1 = best_P[0][:3, 3:]
                # print(f'before solvePnP image1 pose: {P1}')
                # print(f'before solvePnP R0: {rot_vec1}')
                # print(f'before solvePnP T0: {shift1}')
                # success0, R0, T0 = cv2.solvePnP(best_points3d, best_points2d[1].squeeze(), K3, distCoeffs,
                #                                             useExtrinsicGuess=True, rvec=rot_vec1, tvec=shift1)
                # print(f'after solvePnP R0:{R0}')
                # print(f'after solvePnP T0: {T0}')
                # print(f'after solvePnP image1 pose: {P1}')

                # if success0:
                # best_P[1] = pose
                # print(f'after solvePnP pose: {pose}')
                positions[qimname] = best_P[1].tolist()
                if drawMatch:
                    print(f'best_depth: {best_depth}')
                    depth_image = Image.open(best_depth[3])
                    depth_image = depth_image.resize((480, 640))
                    depth_image.save(storage_path + os.sep + "best_depth.jpg")
                    d_width, d_height = depth_image.size
                    # print(f'width x height: {d_width} x {d_height}')
                    # print(f'best_match.shape: {best_match.shape}')
                    # print(f'best_inliners.shape: {best_inliners}')
                    # m = 0
                    # for index in best_match:
                    #   point = kp1[index[0]]
                    #   print(f'the index in best_match: {index}')
                    #   print(f'the point: {point}')
                    #   print(f'(u, v): ({point[1]}, {point[0]})')
                    #   print(f'camera (x,y,z,1): {camera_coords_list[m]}')
                    #   print(f'world (x,y,z): {best_points3d[m]}')
                    #   m = m + 1
                    
                    # for m in best_inliners:
                    #   index = best_match[m]
                    #   point = kp1[index[0]]
                    #   print(f'the index in best_match: {m}')
                    #   print(f'the point: {point}')
                    #   print(f'(u, v): ({point[1]}, {point[0]})')
                    #   print(f"depth: {depth_list[m]}")
                    #   print(f'camera (x,y,z,1): {camera_coords_list[m]}')
                    #   print(f'world (x,y,z): {best_points3d[m]}')
                        
                        
                    dmatch13 = [cv2.DMatch(m[0], m[1], 0) for m in best_match[best_inliners]]
                    # print(f'dmatch13: {dmatch13}')
                    bkp1 = [cv2.KeyPoint(kp[0], kp[1], 1, -1, 0, 0, -1) for kp in best_keypoints[0]]
                    bkp3 = [cv2.KeyPoint(kp[0], kp[1], 1, -1, 0, 0, -1) for kp in best_keypoints[1]]
                    img_with_key13 = cv2.drawMatches(best_image_RGB[0], bkp1, best_image_RGB[1],
                                                     bkp3, dmatch13, None)
                    compression_params = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
                    cv2.imwrite(os.path.join(resfolder,
                                               'match_' + os.path.basename(best_image_name[0]).split('.')[0] + qimname), img_with_key13, compression_params)
                    # else:
                    #     positions[qimname] = default_P.tolist()
                    #     print("all pose est failed")
            print(f'pnp time cost:{end - start}')
            print(f'total time cost:{end - start_init}')

            if ground_P3 is not None and best_P is not None:
                print(best_image_name)
                R3 = ground_P3[:3, :3]
                R3_qim = best_P[1][:3, :3]
                residuals = ground_P3 - best_P[1]
                rot_vec_p3, _ = cv2.Rodrigues(R3)
                rot_vec_qim, _ = cv2.Rodrigues(R3_qim)
                print(f'Loss shift:{np.linalg.norm(residuals[:, 3])}')
                print(f'Loss rot:{np.linalg.norm(residuals[:, :3])}')
                print(
                    f'Loss rot radius:{(np.linalg.norm(rot_vec_p3) - np.linalg.norm(rot_vec_qim)) * 180. / np.pi}')
                print(
                    f'Loss rot vec dir:{np.linalg.norm(rot_vec_p3 / np.linalg.norm(rot_vec_p3) - rot_vec_qim / np.linalg.norm(rot_vec_qim))}')
          
        print(f'saved_path:{saved_images}')
        print(f'positions:{positions}')
        print(f'final pose 1: {best_P[0]}')
        print(f'final pose 3: {best_P[1]}')
        # print(f'pose: {P1}')
        # print(f'intrinsic: {best_K[0]}')
        # print(f"database path: {best_depth}")
        return JsonResponse({'message': 'Folder Found', 'saved_path': saved_images, 'positions': positions}, status=200)

    else:
        return JsonResponse({'error': 'POST request required'}, status=400)
