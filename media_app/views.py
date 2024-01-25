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
    rotated_image = cv2.flip(rotated_image, 1)
    return rotated_image

@csrf_exempt
def request_NVLAD_redir(request):
    start_init = time.time()
    img_loc = request.GET.get('source_location', 'temps/')
    img_loc = os.path.join(img_loc, 'color')
    src_loc = os.path.join(settings.MEDIA_ROOT, 'images/', img_loc)
    dataset_loc = img_loc.split('/')[0]
    intri_loc = os.path.join(settings.MEDIA_ROOT, 'images/', request.GET.get('camera_intri_location', os.path.join(dataset_loc, 'intrinsic')))
    exter_loc = os.path.join(settings.MEDIA_ROOT, 'images/', request.GET.get('camera_exter_location', os.path.join(dataset_loc, 'pose')))
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
                image_name = image_name+'.jpg'
                if camera_matrix is not None and i < len(camera_matrix):
                    qintrinsic[image_name] = np.array(camera_matrix[i])
                # qintrinsic[image.name] = np.array(camera_matrix[i]) if camera_matrix and i < len(camera_matrix) else K
                fs = FileSystemStorage(location=storage_path)
                saved_image = fs.save(image_name, image)
                saved_images.append(save_to_jpg(os.path.join(storage_path, saved_image), os.path.join(storage_path, image_name)))
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
                        pred_imgs[qimname] = [(ims[1], os.path.join(intri_loc, simname.split('.')[0]+'.intrinsic_color.txt'),
                                               os.path.join(exter_loc, simname.split('.')[0]+'.pose.txt'))]
                    else:
                        pred_imgs[qimname].append((ims[1], os.path.join(intri_loc, simname.split('.')[0]+'.intrinsic_color.txt'),
                                               os.path.join(exter_loc, simname.split('.')[0]+'.pose.txt')))

        feature_extractor = cv2.BRISK_create()
        # feature_extractor = cv2.ORB_create()
        # feature_extractor = cv2.xfeatures2d.SIFT_create()
        feature_match = cv2.BFMatcher(crossCheck=True)
        distCoeffs = None
        useFilter = True
        filter_num = 200
        filter_params = {'distCoeffs1': None, 'distCoeffs2': None, 'threshold': 8., 'prob':0.99, 'no_intrinsic':True}
        drawMatch = True
        W = 480
        H = 640
        est_focal = np.sqrt(W**2+H**2) * 1428.643433 / 1440
        # est_focal = max(W,H) * 1428.643433 / 1440
        est_K = np.array([[est_focal, 0, W/2.], [0, est_focal, H/2.], [0, 0, 1]])
        for qimname, v in pred_imgs.items():
            qim = os.path.join(tempimages, qimname)
            sim1 = v[0][0]

            image1 = cv2.imread(sim1)
            image1 = image_transform(image1)
            factor1 = [float(W) / image1.shape[1], float(H) / image1.shape[0]]
            image1 = cv2.resize(image1, (W, H))
            image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            image3 = cv2.imread(qim)
            print("image3 src shape is :", image3.shape)
            image3 = image_transform(image3)
            est_focal3 = np.linalg.norm(image3.shape[:2]) * 1428.643433 / 1440
            est_K3 = np.array([[est_focal3, 0, image3.shape[1]/2], [0, est_focal3, image3.shape[0] / 2.], [0, 0, 1]])
            factor3 = [float(W) / image3.shape[1], float(H) / image3.shape[0]]
            image3 = cv2.resize(image3, (W, H))
            image3_gray = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)
            K1 = read_camera_matrix_3dscanner(v[0][1]) if os.path.exists(v[0][1]) else K
            K1[0,:] *= factor1[0]
            K1[1,:] *= factor1[1]
            K1 = est_K
            if qimname in qintrinsic:
                K3 = qintrinsic[qimname]
                K3[0, :] *= factor3[0]
                K3[1, :] *= factor3[1]
            else:
                # assume camera focal = diag_len/diag(24,36) * diag(H, W), original point in centre H/2, W/2
                tmp_focal = image3.shape[1] * 1428.643433/1440
                # tmp_focal = 500
                K3 = np.array([[est_K3[0,0]*factor3[0], 0.000000, est_K3[0,2]*factor3[0]],
                      [0.000000, est_K3[1,1]*factor3[1], est_K3[1,2]*factor3[1]],
                      [0.000000, 0.000000, 1.000000]])
            K3 = K1
            print(f'image1 intrinsic:{K1}')
            print(f'image3 intrinsic:{K3}')
            print(f'image1 shape:{image1.shape}')
            print(f'image3 shape:{image3.shape}')
            P1 = read_pose_3dscanner(v[0][2]) if os.path.exists(v[0][2]) else np.eye(3, 4)
            print(f'image1 pose:{P1}')
            ground_truth = os.path.join(exter_loc, qimname.split('.')[0] + '.pose.txt')
            P3 = read_pose_3dscanner(ground_truth) if os.path.exists(ground_truth) else P1

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

            print(qim)
            print(v[0])
            print(f"matches13 num:{len(tracks13)}")

            for i in range(1, len(v)):
                sim2 = v[i][0]

                image2 = cv2.imread(sim2)
                image2 = image_transform(image2)
                factor2 = [float(W) / image2.shape[1], float(H) / image2.shape[0]]
                image2 = cv2.resize(image2, (W, H))
                image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
                K2 = read_camera_matrix_3dscanner(v[i][1]) if os.path.exists(v[i][1]) else K
                K2[0, :] *= factor2[0]
                K2[1, :] *= factor2[1]
                K2 = K1
                P2 = read_pose_3dscanner(v[i][2]) if os.path.exists(v[i][2]) else np.eye(3, 4)
                keypoints2, descriptions2 = feature_extractor.detectAndCompute(image2_gray, None)
                matches12 = feature_match.match(descriptions1, descriptions2)
                # matches21 = feature_match.match(descriptions2, descriptions1)
                print(f'matches 12:{len(matches12)}')
                # print(f'matches 21:{len(matches21)}')
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
                if tracks1 is None or len(tracks1) < 3:
                    print("pose est failed!")
                    pose = None
                    continue
                points1 = np.float32([keypoints1[i].pt for i in tracks1]).reshape(-1, 1, 2)
                points2 = np.float32([keypoints2[tracks12[i]].pt for i in tracks1]).reshape(-1, 1, 2)
                points3 = np.float32([keypoints3[tracks13[i]].pt for i in tracks1]).reshape(-1, 1, 2)
                points3d = cv2.triangulatePoints(K1 @ P1, K2 @ P2, points1, points2)
                points3d = cv2.convertPointsFromHomogeneous(points3d.T).squeeze()

                print(v[i])
                print(f"matches12 num:{len(tracks12)}")
                print(f"matches123 num:{len(tracks1)}")
                print(f"3d points num:{points3d.shape[0]}")

                if points3d.shape[0] >= 12:
                    success, R, T, inliners = cv2.solvePnPRansac(points3d, points3, K3, distCoeffs)
                    if inliners is not None:
                        inliners = inliners.squeeze()
                        print(f'inliner num:{inliners.shape}')
                        # if len(inliners) < 12:
                        #     success = False

                        useRANSAC = False
                        useDLS = False
                        if len(inliners) < 12 and success:
                            inliners = np.arange(len(points3))
                            if len(inliners) < 30:
                                success, R, T = cv2.solvePnP(points3d, points3, K3, distCoeffs)
                            elif len(inliners) < 50:
                                useDLS = True
                                useRANSAC = False
                            else:
                                useDLS = True
                                useRANSAC = True
                        elif len(inliners) < max(50, len(points3)/2):
                            useDLS = True


                        if useDLS:
                            rot_vec1, _ = cv2.Rodrigues(P1[:3, :3])
                            rot_vec1 = rot_vec1.flatten()
                            shift1 = P1[:3, 3]
                            intrinsic3 = np.array([K3[0, 0], K3[1, 1], K3[0, 2], K3[1, 2]])
                            init_params = np.hstack((intrinsic3, rot_vec1, shift1))
                            print(f'init_params:{init_params}')
                            success0, R0, T0, K3new, _ = DLS_pose_est(points3d[inliners], points3[inliners].squeeze(), init_params, useRANSAC=useRANSAC)
                            print(f'new K3:{K3new}')
                            if success0:
                                success, R, T = success0, R0, T0
                # elif points3.shape[0] >= 6:
                #     success, R, T = cv2.solvePnP(points3d, points3, K3, distCoeffs)
                else:
                    success = False

                if drawMatch and success:
                    dmatch13 = [cv2.DMatch(i, tracks13[i], 0) for i in tracks1]
                    dmatch23 = [cv2.DMatch(tracks12[i], tracks13[i], 0) for i in tracks1]
                    img_with_key13 = cv2.drawMatches(image1, keypoints1, image3, keypoints3, dmatch13, None)
                    img_with_key23 = cv2.drawMatches(image2, keypoints2, image3, keypoints3, dmatch23, None)
                    img_with_key13_all = cv2.drawMatches(image1, keypoints1, image3, keypoints3, good_matches13, None)
                    img_with_key12_all = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches12, None)
                    compression_params = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
                    cv2.imwrite(os.path.join(resfolder,
                                             'match_' + os.path.basename(sim1).split('.')[0] + qimname), img_with_key13,
                                compression_params)
                    cv2.imwrite(os.path.join(resfolder,
                                             'match_' + os.path.basename(sim2).split('.')[0] + qimname), img_with_key23,
                                compression_params)
                    cv2.imwrite(os.path.join(resfolder,
                                             'match_all_' + os.path.basename(sim1).split('.')[0] + qimname), img_with_key13_all,
                                compression_params)
                    cv2.imwrite(os.path.join(resfolder,
                                             'match_all_' + os.path.basename(sim1).split('.')[0] + os.path.basename(sim2)), img_with_key12_all,
                                compression_params)

                if success:
                    R, _ = cv2.Rodrigues(R)
                    pose = np.hstack((R, T))
                else:
                    print("pose est failed!")
                    pose = None

                end = time.time()
                print(f'pnp time cost:{end-start}')
                print(f'total time cost:{end-start_init}')
                if pose is not None:
                    positions[qimname] = pose.tolist()
                    break
                elif end-start_init > 30 or i == len(v)-1:
                    positions[qimname] = P1.tolist()
                    pose = P1
                    break
            if P3 is not None and pose is not None:
                R3 = P3[:3, :3]
                R3_qim = pose[:3, :3]
                residuals = P3 - pose
                rot_vec_p3, _ = cv2.Rodrigues(R3)
                rot_vec_qim, _ = cv2.Rodrigues(R3_qim)
                print(f'Loss shift:{np.linalg.norm(residuals[:,3])}')
                print(f'Loss rot:{np.linalg.norm(residuals[:,:3])}')
                print(f'Loss rot radius:{(np.linalg.norm(rot_vec_p3) - np.linalg.norm(rot_vec_qim)) * 180. / np.pi}')
                print(f'Loss rot vec dir:{np.linalg.norm(rot_vec_p3/np.linalg.norm(rot_vec_p3) - rot_vec_qim/np.linalg.norm(rot_vec_qim))}')

        return JsonResponse({'message': 'Folder Found', 'saved_path': saved_images, 'positions':positions} , status=200)

    else:
        return JsonResponse({'error': 'POST request required'}, status=400)
