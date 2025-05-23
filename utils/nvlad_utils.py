import os
import re
import time
import json
import numpy as np
import cv2

from PIL import Image
from django.core.files.storage import FileSystemStorage
from django_project import settings
from lightglue.utils import read_image, resize_image, rbd, numpy_image_to_torch
from accelerated_features.modules.xfeat import XFeat
from lightglue import LightGlue, SuperPoint, DISK
from utils.upload import new_name, new_dir_name
from utils.calib3d import *
from utils.draw import *
import copy

feature_extractor = DISK(max_num_keypoints=2048).eval().to(settings.DEVICE)  # load the extractor
feature_match = LightGlue(features="disk", depth_confidence=-1, width_confidence=-1).eval().to(settings.DEVICE)

def setup_directories(req_loc, tempfolder):
    """Setup necessary directories for processing."""
    if not os.path.exists(req_loc):
        os.makedirs(req_loc)
    if not os.path.exists(tempfolder):
        os.makedirs(tempfolder)
        os.makedirs(os.path.join(tempfolder, 'query_features'))
        os.makedirs(os.path.join(tempfolder, 'query_folder'))

def save_query_images(images, storage_path, tempquery, camera_matrix=None):
    """Save query images and create query.txt file."""
    saved_images = []
    K = np.array([[485, 0, 240], [0., 485, 320], [0, 0, 1]])
    qintrinsic = {}
    
    with open(tempquery, 'w') as qtxt:
        for i, image in enumerate(images):
            image_name, _ = os.path.splitext(image.name)
            image_name = image_name + '.jpg'
            if camera_matrix is not None and i < len(camera_matrix):
                qintrinsic[image_name] = np.array(camera_matrix[i])
            else:
                qintrinsic[image_name] = K
            
            fs = FileSystemStorage(location=storage_path)
            saved_image = fs.save(image_name, image)
            saved_images.append(
                save_to_jpg(os.path.join(storage_path, saved_image), 
                           os.path.join(storage_path, image_name)))
            qtxt.write(image_name + '\n')
    
    return saved_images, qintrinsic

def process_predictions(predictions_file, intri_loc, exter_loc, depth_loc):
    """Process PatchNetVLAD predictions and organize image pairs."""
    pred_pattern = re.compile(r',\s*')
    pred_imgs = {}
    
    with open(predictions_file, 'r') as qtxt:
        for line in qtxt:
            if not line.startswith('#'):
                ims = re.split(pred_pattern, line.strip())
                ims[0] = ims[0].strip()
                ims[1] = ims[1].strip()
                _, qimname = os.path.split(ims[0])
                _, simname = os.path.split(ims[1])
                
                if qimname not in pred_imgs:
                    pred_imgs[qimname] = [
                        (ims[1], 
                         os.path.join(intri_loc, simname.split('.')[0] + '.intrinsic_color.txt'),
                         os.path.join(exter_loc, simname.split('.')[0] + '.pose.txt'),
                         os.path.join(depth_loc, simname.split('.')[0] + '.depth.jpg'),
                         simname.split('.')[0])]
                else:
                    pred_imgs[qimname].append(
                        (ims[1],
                         os.path.join(intri_loc, simname.split('.')[0] + '.intrinsic_color.txt'),
                         os.path.join(exter_loc, simname.split('.')[0] + '.pose.txt'),
                         os.path.join(depth_loc, simname.split('.')[0] + '.depth.jpg'),
                         simname.split('.')[0]))
    
    return pred_imgs

def process_single_image(image_path, H=640, W=480):
    """Process a single image for matching."""
    image = read_image(image_path)
    image = image_transform(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.equalizeHist(image)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image, _ = resize_image(image, (H, W))
    image = np.uint8(image)
    return image

def match_images_xfeat(image1, image3, xfeat):
    """Match features between two images using XFeat."""
    kpoints1, kpoints3 = xfeat.match_xfeat_star(image1, image3, top_k=2000)
    return kpoints1, kpoints3

def match_images_lightglue(image1, image3):
    feats1 = feature_extractor.extract(numpy_image_to_torch(image1).to(settings.DEVICE))
    feats1out = rbd(feats1)
    kp1 = feats1out['keypoints'].cpu().numpy()
    feats3 = feature_extractor.extract(numpy_image_to_torch(image3).to(settings.DEVICE))
    feats3out = rbd(feats3)
    kp3 = feats3out['keypoints'].cpu().numpy()
    matches13 = feature_match({"image0": feats1, "image1": feats3})
    matches13out = rbd(matches13)
    good_matches13 = matches13out['matches'].cpu().numpy()
    kp1 = kp1[good_matches13[:, 0]]
    kp3 = kp3[good_matches13[:, 1]]
    return kp1, kp3

def estimate_pose_PNPRANSAC(points3d, kpoints3, K3, P1, distCoeffs=None):
    """Estimate camera pose using PnP."""
    rot_vec1, _ = cv2.Rodrigues(P1[:3, :3])
    shift1 = copy.deepcopy(P1[:3, 3:])
    
    success, R, T, inliners = cv2.solvePnPRansac(
        points3d, kpoints3, K3, distCoeffs,
        useExtrinsicGuess=True,
        rvec=rot_vec1,
        tvec=shift1,
        reprojectionError=8,
        confidence=0.95,
        iterationsCount=200
    )
    
    if success and inliners is not None:
        inliners = inliners.squeeze()
        Rtmp, _ = cv2.Rodrigues(R)
        pose = np.hstack((Rtmp, T))
        return True, pose, inliners
    
    return False, None, None

def estimate_pose_PNP(points3d, kpoints3, K3, P1, distCoeffs=None):
    """Estimate camera pose using PnP."""
    rot_vec1, _ = cv2.Rodrigues(P1[:3, :3])
    shift1 = copy.deepcopy(P1[:3, 3:])
    
    success, R, T = cv2.solvePnP(
        points3d, kpoints3, K3, distCoeffs,
        useExtrinsicGuess=True, rvec=rot_vec1, tvec=shift1
    )
    if success:
        Rtmp, _ = cv2.Rodrigues(R)
        pose = np.hstack((Rtmp, T))
        return True, pose
    
    return False, None

def update_best_results(current_results, best_results, second_best_results):
    """Update best and second best results."""
    if current_results['points3d'].shape[0] > best_results['points3d'].shape[0]:
        # Move current best to second best
        second_best_results.update(best_results)
        # Update best with current
        best_results.update(current_results)
        return True
    elif (second_best_results['points3d'] is None or 
          current_results['points3d'].shape[0] > second_best_results['points3d'].shape[0]):
        second_best_results.update(current_results)
        return False
    return False

def save_match_visualization(best_results, second_best_results, resfolder, storage_path):
    """Save visualization of matches."""
    if best_results['P'] is not None:
        depth_image = Image.open(best_results['depth'][3])
        depth_image = depth_image.resize((480, 640))
        depth_image.save(os.path.join(storage_path, "best_depth.jpg"))
        inliners_1 = best_results['keypoints'][0][best_results['inliners']]
        inliners_2 = best_results['keypoints'][1][best_results['inliners']]
        
        canvas = warp_corners_and_draw_matches(
            inliners_1, 
            inliners_2,
            best_results['image_RGB'][0], 
            best_results['image_RGB'][1]
        )
        kp_img1 = draw_keypoints(best_results['image_RGB'][0], inliners_1)
        kp_img2 = draw_keypoints(best_results['image_RGB'][1], inliners_2)
        compression_params = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
        cv2.imwrite(
            os.path.join(resfolder, 'match_' + os.path.basename(best_results['image_name'][0]).split('.')[0] + best_results['qimname'] + '.jpg'),
            canvas, compression_params
        )
        cv2.imwrite(
            os.path.join(resfolder, os.path.basename(best_results['image_name'][0]).split('.')[0] + '_keypoints.jpg'),
            kp_img1, compression_params
        )
        cv2.imwrite(
            os.path.join(resfolder, best_results['qimname'].split('.')[0] + '_keypoints.jpg'),
            kp_img2, compression_params
        )

    if second_best_results['P'] is not None:
        depth_image = Image.open(second_best_results['depth'][3])
        depth_image = depth_image.resize((480, 640))
        depth_image.save(os.path.join(storage_path, "second_best_depth.jpg"))
        inliners_1 = second_best_results['keypoints'][0][second_best_results['inliners']]
        inliners_2 = second_best_results['keypoints'][1][second_best_results['inliners']]
        
        canvas = warp_corners_and_draw_matches(
            inliners_1,
            inliners_2,
            second_best_results['image_RGB'][0],
            second_best_results['image_RGB'][1]
        )
        kp_img1 = draw_keypoints(second_best_results['image_RGB'][0], inliners_1)
        kp_img2 = draw_keypoints(second_best_results['image_RGB'][1], inliners_2)
        compression_params = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
        cv2.imwrite(
            os.path.join(resfolder, 'second_best_match_' + os.path.basename(second_best_results['image_name'][0]).split('.')[0] + second_best_results['qimname'] + '.jpg'),
            canvas, compression_params
        )
        cv2.imwrite(
            os.path.join(resfolder, os.path.basename(second_best_results['image_name'][0]).split('.')[0] + '_keypoints.jpg'),
            kp_img1, compression_params
        )
            

def calculate_pose_error(ground_P3, best_P):
    """Calculate pose estimation error metrics."""
    R3 = ground_P3[:3, :3]
    R3_qim = best_P[1][:3, :3]
    residuals = ground_P3 - best_P[1]
    rot_vec_p3, _ = cv2.Rodrigues(R3)
    rot_vec_qim, _ = cv2.Rodrigues(R3_qim)
    
    return {
        'shift_error': np.linalg.norm(residuals[:, 3]),
        'rot_error': np.linalg.norm(residuals[:, :3]),
        'rot_radius_error': (np.linalg.norm(rot_vec_p3) - np.linalg.norm(rot_vec_qim)) * 180. / np.pi,
        'rot_vec_dir_error': np.linalg.norm(rot_vec_p3 / np.linalg.norm(rot_vec_p3) - rot_vec_qim / np.linalg.norm(rot_vec_qim))
    } 
    
def write_results_to_file(result_txt, best_results, second_best_results, total_time):
    """Write processing results to result.txt file."""
    with open(result_txt, 'w') as f:
        f.write(f"Total processing time: {total_time:.3f}s\n")
        
        # Write best results
        f.write('----------------best result-------------------\n')
        if best_results['P'] is not None:
            f.write(f'find the best image match key points number: {best_results["points3d"].shape[0]}\n')
            f.write(f'find the best image match inlier number: {best_results["inliners"].shape[0]}\n')
            f.write('find the best match image pose:\n')
            for row in best_results['P'][0]:
                f.write(' '.join(map(str, row)) + '\n')
            f.write('find the redirect image pose:\n')
            for row in best_results['P'][1]:
                f.write(' '.join(map(str, row)) + '\n')
        else:
            f.write('No best match found\n')
        
        # Write second best results
        f.write('----------------second best result-------------------\n')
        if second_best_results['P'] is not None:
            f.write(f'find the second best image match key points number: {second_best_results["points3d"].shape[0]}\n')
            f.write(f'find the second best image match inlier number: {second_best_results["inliners"].shape[0]}\n')
            f.write('find the second best match image pose:\n')
            for row in second_best_results['P'][0]:
                f.write(' '.join(map(str, row)) + '\n')
            f.write('find the second redirect image pose:\n')
            for row in second_best_results['P'][1]:
                f.write(' '.join(map(str, row)) + '\n')
        else:
            f.write('No second best match found\n') 

def save_points_to_file(txt, points):
    """Save points to file. Each row in points is written as a line with K numbers separated by spaces."""
    with open(txt, 'w') as f:
        for i in range(points.shape[0]):
            f.write(' '.join(map(str, points[i])) + '\n')