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
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)
        os.makedirs(os.path.join(save_loc, 'index_features/'))
    netpath = settings.NetVLAD_PATH
    nv_params = request.GET.get('nv_params', '')
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


@csrf_exempt
def request_NVLAD_redir(request):
    start_init = time.time()
    img_loc = request.GET.get('source_location', 'temps/')
    img_loc = os.path.join(img_loc, 'color')
    src_loc = os.path.join(settings.MEDIA_ROOT, 'images/', img_loc)
    dataset_loc = img_loc.split('/')[0]
    intri_loc = os.path.join(settings.MEDIA_ROOT, 'images/',
                             request.GET.get('camera_intri_location', os.path.join(dataset_loc, 'intrinsic')))
    exter_loc = os.path.join(settings.MEDIA_ROOT, 'images/',
                             request.GET.get('camera_exter_location', os.path.join(dataset_loc, 'pose')))
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
        print(req_loc)
        print(tempfolder)
        print(tempquery)
        if not os.path.exists(tempfolder):
            os.makedirs(tempfolder)
            os.makedirs(tempfeature)
            os.makedirs(tempimages)

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
                             os.path.join(exter_loc, simname.split('.')[0] + '.pose.txt'))]
                    else:
                        pred_imgs[qimname].append(
                            (ims[1], os.path.join(intri_loc, simname.split('.')[0] + '.intrinsic_color.txt'),
                             os.path.join(exter_loc, simname.split('.')[0] + '.pose.txt')))

        feature_extractor = cv2.BRISK_create()
        # feature_extractor = cv2.ORB_create()
        # feature_extractor = cv2.xfeatures2d.SIFT_create()
        feature_match = cv2.BFMatcher(crossCheck=True)
        distCoeffs = None
        useFilter = True
        filter_num = 200
        filter_params = {'distCoeffs1': None, 'distCoeffs2': None, 'threshold': 8., 'prob': 0.99, 'no_intrinsic': True}
        drawMatch = True
        timeout = 30
        W = 480
        H = 640
        est_focal = np.sqrt(W ** 2 + H ** 2) * 1428.643433 / 1440
        est_K = np.array([[est_focal, 0, W / 2.], [0, est_focal, H / 2.], [0, 0, 1]])
        for qimname, v in pred_imgs.items():
            qim = os.path.join(tempimages, qimname)
            image3 = cv2.imread(qim)
            print("image3 src shape is :", image3.shape)
            image3 = image_transform(image3)
            image3 = cv2.resize(image3, (W, H))
            image3_gray = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)
            K3 = est_K
            print(f'image3 intrinsic:{K3}')
            print(f'image3 shape:{image3.shape}')

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
            stop_inliner_rate = 0.97
            use_DST_inliner_rate = 0.5
            min_traverse_windows = 3
            init_traverse_windows = 30
            add_traverse_windows = 1.2

            for i in range(0, len(v) - 1):
                success = False
                pose = None
                traverse_windows = init_traverse_windows
                meet_best = False

                sim1 = v[i][0]
                image1 = cv2.imread(sim1)
                image1 = image_transform(image1)
                image1 = cv2.resize(image1, (W, H))
                image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
                K1 = est_K
                print(f'image1 intrinsic:{K1}')
                print(f'image1 shape:{image1.shape}')
                P1 = read_pose_3dscanner(v[i][2]) if os.path.exists(v[i][2]) else np.eye(3, 4)
                print(f'image1 pose:{P1}')

                keypoints1, descriptions1 = feature_extractor.detectAndCompute(image1_gray, None)
                keypoints3, descriptions3 = feature_extractor.detectAndCompute(image3_gray, None)
                matches13 = feature_match.match(descriptions1, descriptions3)
                if useFilter:
                    m13, num13 = getInliners(keypoints1, keypoints3, matches13, K1, K3, **filter_params)
                    if num13 > filter_num:
                        matches13 = m13
                good_matches13 = sorted(matches13, key=lambda x: x.distance)[:5000]
                tracks13 = {}
                for m in good_matches13:
                    if m.queryIdx not in tracks13:
                        tracks13[m.queryIdx] = m.trainIdx

                print(f"matches13 num:{len(tracks13)}")

                for j in range(i + 1, len(v)):
                    success = False
                    pose = None
                    if j - i > max(traverse_windows, min_traverse_windows):
                        print('excced windows')
                        break

                    sim2 = v[j][0]
                    image2 = cv2.imread(sim2)
                    image2 = image_transform(image2)
                    image2 = cv2.resize(image2, (W, H))
                    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
                    K2 = est_K
                    P2 = read_pose_3dscanner(v[j][2]) if os.path.exists(v[j][2]) else np.eye(3, 4)
                    keypoints2, descriptions2 = feature_extractor.detectAndCompute(image2_gray, None)
                    matches12 = feature_match.match(descriptions1, descriptions2)
                    print(f'matches 12:{len(matches12)}')
                    if useFilter:
                        m12, num12 = getInliners(keypoints1, keypoints2, matches12, K1, K2, **filter_params)
                        if num12 > filter_num:
                            matches12 = m12
                    good_matches12 = sorted(matches12, key=lambda x: x.distance)[:5000]
                    tracks12 = {}
                    for m in good_matches12:
                        if m.queryIdx not in tracks12:
                            tracks12[m.queryIdx] = m.trainIdx
                    tracks1 = list(set(tracks12.keys()) & set(tracks13.keys()))
                    if tracks1 is None or len(tracks1) < 10:
                        print("common points too low, pose est failed!")
                        end = time.time()
                        if end - start_init > timeout:
                            print("time out 3")
                            break
                        continue
                    points1 = np.float32([keypoints1[i].pt for i in tracks1]).reshape(-1, 1, 2)
                    points2 = np.float32([keypoints2[tracks12[i]].pt for i in tracks1]).reshape(-1, 1, 2)
                    points3 = np.float32([keypoints3[tracks13[i]].pt for i in tracks1]).reshape(-1, 1, 2)
                    points3d = cv2.triangulatePoints(K1 @ P1, K2 @ P2, points1, points2)
                    points3d = cv2.convertPointsFromHomogeneous(points3d.T).squeeze()

                    print(qimname)
                    print(v[i][0])
                    print(v[j][0])
                    print(f"matches12 num:{len(tracks12)}")
                    print(f"matches123 num:{len(tracks1)}")
                    print(f"3d points num:{points3d.shape[0]}")

                    if points3d.shape[0] >= 100:
                        rot_vec1, _ = cv2.Rodrigues(P1[:3, :3])
                        shift1 = P1[:3, 3:]
                        success, R, T, inliners = cv2.solvePnPRansac(points3d, points3, K3, distCoeffs,
                                                                     useExtrinsicGuess=False, rvec=rot_vec1,
                                                                     tvec=shift1)
                        if success and inliners is not None:
                            inliners = inliners.squeeze()
                            print(f'inliner num:{inliners.shape}')
                            Rtmp, _ = cv2.Rodrigues(R)
                            pose = np.hstack((Rtmp, T))
                            residuals = ground_P3 - pose
                            if len(inliners) >= 200 and (
                                    len(inliners) > (best_inliners_rate + best_inliners_rate_window) * len(points3) \
                                    or (best_inliners_rate - best_inliners_rate_window) * len(points3) < len(inliners) \
                                    and len(best_inliners) < len(inliners)) \
                                    or len(best_inliners) < len(inliners) < 200 \
                                    or os.path.exists(ground_truth) and np.linalg.norm(residuals) < min_residuals_norm:
                                print('found best')
                                best_inliners = inliners
                                best_inliners_rate = float(len(inliners)) / float(len(points3))
                                best_points2d = [points1, points2, points3]
                                best_points3d = points3d
                                best_K = [K1, K2, K3]
                                Rtmp, _ = cv2.Rodrigues(R)
                                pose = np.hstack((Rtmp, T))
                                best_P = [P1, P2, pose]
                                best_image_name = [sim1, sim2]
                                best_description = [descriptions1, descriptions2, descriptions3]
                                best_keypoints = [keypoints1, keypoints2, keypoints3]
                                best_tracks = [tracks1, tracks12, tracks13]
                                best_image_RGB = [image1, image2, image3]
                                best_match = [good_matches12, good_matches13]

                                is_stop = best_inliners_rate > stop_inliner_rate
                                meet_best = True
                                traverse_windows = init_traverse_windows if best_inliners_rate >= 0.2 else 0.8 * traverse_windows
                                if os.path.exists(ground_truth) and np.linalg.norm(residuals) < min_residuals_norm:
                                    min_residuals_norm = np.linalg.norm(residuals)
                                    is_stop = min_residuals_norm < stop_residuals_norm
                            elif len(inliners) < 0.2 * len(points3):
                                print('too less inliners')
                                traverse_windows *= 0.95 if meet_best or j - i < min_traverse_windows else 0.5
                            else:
                                traverse_windows *= 0.97 if meet_best or j - i < min_traverse_windows else 0.8
                        else:
                            traverse_windows *= 0.95 if meet_best or j - i < min_traverse_windows else 0.5
                    else:
                        traverse_windows *= 0.95 if meet_best or j - i < min_traverse_windows else 0.5
                        success = False

                    if is_stop:
                        print('stop reached')
                        break
                    else:
                        if not success:
                            print("pose est failed!")
                        end = time.time()
                        if end - start_init > timeout:
                            print("time out 1")
                            break

                end = time.time()
                if is_stop or i == len(v) - 2 or end - start_init > timeout:
                    if end - start_init > timeout:
                        print("time out 2")
                        print('best_ratio failed')
                    if best_P is not None:
                        print(f'best inliner num:{len(best_inliners)}')
                        print(f'points3d num:{len(best_points3d)}')
                        useRANSAC = False
                        tmp_inliners = best_inliners if len(best_inliners) > 20 else np.arange(len(best_points3d))
                        if 20 <= len(best_inliners) < max(50, len(best_points3d) * use_DST_inliner_rate):
                            init_params = DLS_pose_est_init_params(best_P[0], best_K[2])
                            print(f'init_params:{init_params}')
                            success0, R0, T0, K3new, _ = DLS_pose_est(best_points3d[tmp_inliners],
                                                                      best_points2d[2][tmp_inliners].squeeze(),
                                                                      init_params,
                                                                      useRANSAC=useRANSAC)
                            print(f'new K3:{K3new}')
                            print(f'DLS pose est success:{success0}')
                            if success0:
                                Rtmp, _ = cv2.Rodrigues(R0)
                                pose = np.hstack((Rtmp, T0))
                                best_P[2] = pose
                        elif len(best_inliners) <= 12:
                            rot_vec1, _ = cv2.Rodrigues(best_P[0][:3, :3])
                            shift1 = best_P[0][:3, 3:]
                            success0, R0, T0 = cv2.solvePnP(best_points3d, best_points2d[2].squeeze(), K3, distCoeffs,
                                                            useExtrinsicGuess=True, rvec=rot_vec1, tvec=shift1)
                            if success0:
                                Rtmp, _ = cv2.Rodrigues(R0)
                                pose = np.hstack((Rtmp, T0))
                                best_P[2] = pose
                        positions[qimname] = best_P[2].tolist()
                        if drawMatch:
                            dmatch13 = [cv2.DMatch(i, best_tracks[2][i], 0) for i in best_tracks[0]]
                            dmatch23 = [cv2.DMatch(best_tracks[1][i], best_tracks[2][i], 0) for i in best_tracks[0]]
                            img_with_key13 = cv2.drawMatches(best_image_RGB[0], best_keypoints[0], best_image_RGB[2],
                                                             best_keypoints[2], dmatch13, None)
                            img_with_key23 = cv2.drawMatches(best_image_RGB[1], best_keypoints[1], best_image_RGB[2],
                                                             best_keypoints[2], dmatch23, None)
                            img_with_key13_all = cv2.drawMatches(best_image_RGB[0], best_keypoints[0],
                                                                 best_image_RGB[2], best_keypoints[2], best_match[1],
                                                                 None)
                            img_with_key12_all = cv2.drawMatches(best_image_RGB[0], best_keypoints[0],
                                                                 best_image_RGB[1], best_keypoints[1], best_match[0],
                                                                 None)
                            compression_params = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
                            cv2.imwrite(os.path.join(resfolder,
                                                     'match_' + os.path.basename(best_image_name[0]).split('.')[
                                                         0] + qimname), img_with_key13,
                                        compression_params)
                            cv2.imwrite(os.path.join(resfolder,
                                                     'match_' + os.path.basename(best_image_name[1]).split('.')[
                                                         0] + qimname), img_with_key23,
                                        compression_params)
                            cv2.imwrite(os.path.join(resfolder,
                                                     'match_all_' + os.path.basename(best_image_name[0]).split('.')[
                                                         0] + qimname), img_with_key13_all,
                                        compression_params)
                            cv2.imwrite(os.path.join(resfolder,
                                                     'match_all_' + os.path.basename(best_image_name[0]).split('.')[
                                                         0] + os.path.basename(best_image_name[1])), img_with_key12_all,
                                        compression_params)
                    else:
                        positions[qimname] = default_P.tolist()
                        print("all pose est failed")
                    break

            end = time.time()
            print(f'pnp time cost:{end - start}')
            print(f'total time cost:{end - start_init}')

            if ground_P3 is not None and best_P is not None:
                print(best_image_name)
                R3 = ground_P3[:3, :3]
                R3_qim = best_P[2][:3, :3]
                residuals = ground_P3 - best_P[2]
                rot_vec_p3, _ = cv2.Rodrigues(R3)
                rot_vec_qim, _ = cv2.Rodrigues(R3_qim)
                print(f'Loss shift:{np.linalg.norm(residuals[:, 3])}')
                print(f'Loss rot:{np.linalg.norm(residuals[:, :3])}')
                print(
                    f'Loss rot radius:{(np.linalg.norm(rot_vec_p3) - np.linalg.norm(rot_vec_qim)) * 180. / np.pi}')
                print(
                    f'Loss rot vec dir:{np.linalg.norm(rot_vec_p3 / np.linalg.norm(rot_vec_p3) - rot_vec_qim / np.linalg.norm(rot_vec_qim))}')

        return JsonResponse({'message': 'Folder Found', 'saved_path': saved_images, 'positions': positions}, status=200)

    else:
        return JsonResponse({'error': 'POST request required'}, status=400)
