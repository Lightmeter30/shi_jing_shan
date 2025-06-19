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
import matplotlib.pyplot as plt
from .times import timer
from .logger import logger

feature_extractor = DISK(max_num_keypoints=2048).eval().to(settings.DEVICE)  # load the extractor
feature_match = LightGlue(features="disk", depth_confidence=-1, width_confidence=-1).eval().to(settings.DEVICE)
M_3DS_CV2 = np.array([[0, 1, 0],
                      [1, 0, 0],
                      [0, 0, -1]])

def transfer_Pose_from_3DS_to_CV2(P):
    '''
    3DS坐标系 -> OpenCV坐标系
    P: 3DS坐标系下的位姿 4x4
    '''
    Pose = np.eye(4)
    Pose[:3, :3] = M_3DS_CV2 @ P[:3, :3] @ M_3DS_CV2.T
    Pose[:3, 3] = M_3DS_CV2 @ P[:3, 3]
    return Pose

def transfer_Pose_from_CV2_to_3DS(P):
    '''
    OpenCV坐标系 -> 3DS坐标系
    P: OpenCV坐标系下的位姿 4x4
    '''
    Pose = np.eye(4)
    Pose[:3, :3] = M_3DS_CV2.T @ P[:3, :3] @ M_3DS_CV2
    Pose[:3, 3] = M_3DS_CV2.T @ P[:3, 3]
    return Pose

def transfer_Point_from_3DS_to_CV2(Points):
    '''
    OpenCV坐标系 -> 3DS坐标系
    Points: OpenCV坐标系下的点 Nx3
    '''
    return (M_3DS_CV2 @ Points.T).T

def invert_Pose_Matrix(P):
    '''
    求P的逆矩阵
    P: 4x4
    '''
    Pose = np.eye(4)
    R = P[:3,:3]
    t = P[:3,3]
    Pose[:3,:3] = R.T
    Pose[:3,3] = -R.T @ t
    return Pose

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
    # Convert to grayscale but maintain 3 channels
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.equalizeHist(image)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    # Convert back to 3 channels by duplicating grayscale values
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image, _ = resize_image(image, (H, W))
    image = np.uint8(image)
    return image

@timer
def match_images_xfeat(image1, image3, xfeat):
    """Match features between two images using XFeat."""
    kpoints1, kpoints3 = xfeat.match_xfeat_star(image1, image3, top_k=2000)
    return kpoints1, kpoints3

@timer
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

def pose_divide(pose):
    rotate = pose[:3, :3]
    translate = pose[:3, 3]

    return [rotate, translate]

'''
points: image1所有的特征点集合(pixel坐标), 是一个shape为N1 x 2的数组
depth_image: 深度图
K: 相机内参 3x3
P_c2w_3DS: 3DS坐标系下的相机位姿4x4 应该是Camera to World
返回值:
world_3dpoints_cv2 Nx3(opencv坐标系下的世界坐标)
remove_index_list: 被移除点的索引
camera_3dpoints_unity Nx4(unity坐标系下的相机坐标)
points_valid: 有效点 Nx2
depths: 有效点深度 N
'''
@timer
def pixel_to_model(kpoints1, depth_image, K1, P_c2w_3DS, Z_Far):
    """Process a single image for matching."""
    # 1. 检查深度图
    if depth_image is None:
        raise ValueError('Could not read the depth image.')
    
    # 2. 设置相机内参 coord2
    f_x, f_y, c_x, c_y = K1[0, 0], K1[1, 1], K1[0, 2], K1[1, 2]
    
    
    # 3. 批量读取深度值 - 使用numpy的高级索引
    points_int = kpoints1.astype(np.int32)  # 转换为整数坐标

    depth_values = depth_image[points_int[:, 1], points_int[:, 0], 2]  # 只取B通道的值
    
    # 4. 计算有效掩码
    valid_mask = (depth_values != 255) & (depth_values != 0)
    # valid_mask = valid_mask.reshape(-1)  # 确保valid_mask是一维的
    
    # 5. 计算深度值
    depths = (depth_values[valid_mask] / 255) * (Z_Far)
    
    # 6. 获取有效点
    points_cv2 = points_int[valid_mask]
    
    # 7. 计算相机坐标, 转化为unity相机坐标系

    points_3DS = points_cv2[:, [1, 0]]
    camera_3dpoints_3DS = np.zeros((len(points_3DS), 4))
    camera_3dpoints_3DS[:, 0] = (points_3DS[:, 0] - c_x) * depths / f_x
    camera_3dpoints_3DS[:, 1] = (points_3DS[:, 1] - c_y) * depths / f_y
    camera_3dpoints_3DS[:, 2] = - depths
    camera_3dpoints_3DS[:, 3] = 1

    
    model_3dpoints_3DS = (P_c2w_3DS @ camera_3dpoints_3DS.T).T
    model_3dpoints_3DS = cv2.convertPointsFromHomogeneous(model_3dpoints_3DS).squeeze()
    

    # 11. 记录被移除点的索引
    remove_index_list = np.where(~valid_mask)[0].tolist()
    
    return [model_3dpoints_3DS, remove_index_list, camera_3dpoints_3DS, points_cv2]

'''
PNPRANSAC算法本身接收一个W2C的位姿作为初始估计, 计算的结果也是W2C的位姿, 最后需要将结果转换为C2W, 即取逆
points3d: 3d点云 Nx3
kpoints3: image3的特征点 Nx2
K3: 相机内参 3x3
P1: 相机位姿4x4 应该是3DS的C2W
返回值：
success: 
pose_c2w_3DS: 相机位姿3x4 应该是对应Unity坐标系下的C2W
inliners: 有效点索引
'''
@timer
def estimate_pose_PNPRANSAC(points3d_3DS, kpoints3, K3, P1_c2w_3DS, distCoeffs=None, is_debug=False, is_K_equal=False):
    """Estimate camera pose using PnP."""
    # 将3DS的C2W转换为opencv的W2C
    points3d_cv2 = transfer_Point_from_3DS_to_CV2(points3d_3DS)
    P1_c2w_cv2 = transfer_Pose_from_3DS_to_CV2(P1_c2w_3DS)
    P1_w2c_cv2 = invert_Pose_Matrix(P1_c2w_cv2)

    rot_vec1, _ = cv2.Rodrigues(P1_w2c_cv2[:3, :3])
    shift1 = copy.deepcopy(P1_w2c_cv2[:3, 3:])

    if is_K_equal:
        K3[0, 2], K3[1, 2] = K3[1, 2], K3[0, 2]
    
    kpoints3_new = kpoints3
    # kpoints3_new = kpoints3[:, [1, 0]]
    # RANSAC 中使用的是P1(W2C)
    success, R, T, inliners = cv2.solvePnPRansac(
        points3d_cv2, kpoints3_new, K3, distCoeffs,
        useExtrinsicGuess=True,
        rvec=rot_vec1,
        tvec=shift1,
        # useExtrinsicGuess=False,
        reprojectionError=4,
        confidence=0.99,
        iterationsCount=100
    )
    
    if success and inliners is not None:
        inliners = inliners.squeeze()
        Rtmp, _ = cv2.Rodrigues(R)
        pose_w2c_cv2 = np.hstack((Rtmp, T))
        pose_w2c_cv2 = np.vstack((pose_w2c_cv2, np.array([0,0,0,1])))
        pose_c2w_cv2 = invert_Pose_Matrix(pose_w2c_cv2)
        pose_c2w_3DS = transfer_Pose_from_CV2_to_3DS(pose_c2w_cv2)
        if is_debug:
            logger.info(f'pose_c2w_3DS: {pose_c2w_3DS}')
        return True, pose_c2w_3DS[:3, :], inliners
    
    return False, None, None

'''
PNP算法本身接收一个W2C的位姿作为初始估计, 计算的结果也是W2C的位姿, 最后需要将结果转换为W2C, 即取逆
points3d: 3d点云 Nx3, 基于opencv坐标系
kpoints3: image3的特征点 Nx2
K3: 相机内参 3x3
P1: 相机位姿4x4 应该是3DS的C2W
返回值：
success: 
pose_c2w_3DS: 相机位姿3x4 应该是对应3DS坐标系下的C2W
'''
def estimate_pose_PNP(points3d_3DS, kpoints3, K3, P1, distCoeffs=None):
    """Estimate camera pose using PnP."""
    # 将3DS的C2W转换为opencv的W2C
    points3d_cv2 = transfer_Point_from_3DS_to_CV2(points3d_3DS)
    P1_c2w_cv2 = transfer_Pose_from_3DS_to_CV2(P1)
    P1_w2c_cv2 = invert_Pose_Matrix(P1_c2w_cv2)

    rot_vec1, _ = cv2.Rodrigues(P1_w2c_cv2[:3, :3])
    shift1 = copy.deepcopy(P1_w2c_cv2[:3, 3:])

    kpoints3_new = kpoints3
    # kpoints3_new = kpoints3[:, [1, 0]]
    
    success, R, T = cv2.solvePnP(
        points3d_cv2, kpoints3_new, K3, distCoeffs,
        useExtrinsicGuess=True, rvec=rot_vec1, tvec=shift1
    )
    if success:
        Rtmp, _ = cv2.Rodrigues(R)
        pose = np.hstack((Rtmp, T))
        pose_w2c_cv2 = np.vstack((pose, np.array([0,0,0,1])))
        pose_c2w_cv2 = invert_Pose_Matrix(pose_w2c_cv2)
        pose_c2w_3DS = transfer_Pose_from_CV2_to_3DS(pose_c2w_cv2)
        return True, pose_c2w_3DS[:3, :]
    return False, None

def update_best_results(current_results, best_results, second_best_results, inliners_lambda, best_inliners_rate_window=0.1):
    """
    Update best and second best results.
    当前pose会被选为最优pose, 满足以下任一条件: 
    1. 当前内点数 >= 100 and
        (a) 当前内点数 > (历史最佳内点率 + 容差窗口) x 当前点数
        or (b) 当前内点数 > (历史最佳内点率 - 容差窗口) x 当前点数
    2. 历史最佳内点数 < 当前内点数 < 100
    """
    best_inliners_num = len(best_results['inliners'])
    second_inliners_num = len(second_best_results['inliners'])
    cur_inliners_num = len(current_results['inliners'])
    cur_points_num = len(current_results['points3d'])
    if (cur_inliners_num >= 100 and \
        (cur_inliners_num > (best_results['inliners_rate'] + best_inliners_rate_window) * cur_points_num or (cur_inliners_num > (best_results['inliners_rate'] - best_inliners_rate_window) * cur_points_num and cur_inliners_num > best_inliners_num)))\
        or best_inliners_num < cur_inliners_num < 100:
        # Move current best to second best
        second_best_results.update(best_results)
        # Update best with current
        best_results.update(current_results)
    elif second_best_results['points3d'] is None or\
        (cur_inliners_num >= 100 and \
        (cur_inliners_num > (second_best_results['inliners_rate'] + best_inliners_rate_window) * cur_points_num or (cur_inliners_num > (second_best_results['inliners_rate'] - best_inliners_rate_window) * cur_points_num and cur_inliners_num > second_inliners_num)))\
        or second_inliners_num < cur_inliners_num < 100:
            second_best_results.update(current_results)
    if inliners_lambda * best_results['inliners_rate'] > 0.95 and len(best_results['inliners']) >= 500:
        return True
    return False

@timer
def save_match_visualization(best_results, second_best_results, resfolder, storage_path, is_debug):
    """Save visualization of matches."""
    if best_results['P'] is not None:
        if is_debug:
            depth_image = best_results['depth_image']  # 直接使用已经读取的深度图
            cv2.imwrite(os.path.join(storage_path, "best_depth.jpg"), depth_image)
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

    if is_debug and second_best_results['P'] is not None:
        depth_image = second_best_results['depth_image']  # 直接使用已经读取的深度图
        cv2.imwrite(os.path.join(storage_path, "second_best_depth.jpg"), depth_image)
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
            
@timer
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

@timer
def write_results_to_file(result_txt, best_results, second_best_results, error_metrics):
    """Write processing results to result.txt file."""
    with open(result_txt, 'w') as f:
        
        # Write best results
        f.write('----------------best result-------------------\n')
        if best_results['P'] is not None:
            f.write(f'find the best image match key points number: {best_results["points3d"].shape[0]}\n')
            f.write(f'find the best image match inlier number: {best_results["inliners"].shape[0]}\n')
            f.write(f"find the best match image {best_results['data_image_name']} pose:\n")
            origin_P = best_results['P'][0]
            if best_results['origin_shift'] is not None:
                origin_P[:3, 3] += best_results['origin_shift']
            for row in origin_P:
                f.write(' '.join(map(str, row)) + '\n')
            f.write('find the redirect image pose:\n')
            for row in best_results['P'][1]:
                f.write(' '.join(map(str, row)) + '\n')
        else:
            f.write('No best match found\n')
        if error_metrics is not None:
            f.write('--------------error metrics-------------------------\n')
            f.write(f"GT image {second_best_results['qimname']} & second best match: {error_metrics}\n")
        # Write second best results
        f.write('----------------second best result-------------------\n')
        if second_best_results['P'] is not None:
            f.write(f'find the second best image match key points number: {second_best_results["points3d"].shape[0]}\n')
            f.write(f'find the second best image match inlier number: {second_best_results["inliners"].shape[0]}\n')
            f.write(f"find the second best match image {second_best_results['data_image_name']} pose:\n")
            origin_P = second_best_results['P'][0]  
            if second_best_results['origin_shift'] is not None:
                origin_P[:3, 3] += second_best_results['origin_shift']
            for row in origin_P:
                f.write(' '.join(map(str, row)) + '\n')
            f.write('find the second redirect image pose:\n')
            for row in second_best_results['P'][1]:
                f.write(' '.join(map(str, row)) + '\n')
        else:
            f.write('No second best match found\n') 

def debug_save_points_to_file(txt, points):
    """Save points to file. Each row in points is written as a line with K numbers separated by spaces."""
    with open(txt, 'w') as f:
        for i in range(points.shape[0]):
            f.write(' '.join(map(str, points[i])) + '\n')

def debug_evaluate_pnp_pose(model_3dpoints_3DS, points_2d, K, pose_matrix):
    """
    直接评估PnP求解的位姿
    
    参数:
        model_3dpoints_3DS: 已知的3D点 (N, 3)
        points_2d: 对应的2D图像点 (N, 2) 
        K: 相机内参
        pose_matrix: PnP求解的位姿矩阵 c2w 3DS
    """
    points3d_cv2 = transfer_Point_from_3DS_to_CV2(model_3dpoints_3DS)
    pose_w2c_3ds = invert_Pose_Matrix(pose_matrix)
    pose_w2c_cv2 = transfer_Pose_from_3DS_to_CV2(pose_w2c_3ds)
    # 构建投影矩阵
    T = pose_w2c_cv2
    P = K @ T[:3, :]
    
    points_2d_new = points_2d
    # points_2d_new = points_2d[:, [1, 0]]

    # 直接重投影已知3D点
    points3d_cv2_h = np.hstack((points3d_cv2, np.ones((points3d_cv2.shape[0], 1))))
    proj_2d_h = (P @ points3d_cv2_h.T).T
    proj_2d = (proj_2d_h[:, :2].T / proj_2d_h[:, 2]).T

    # 计算重投影误差
    reprojection_errors = np.linalg.norm(proj_2d - points_2d_new, axis=1)
    logger.info(f"重投影误差: mean: {np.mean(reprojection_errors)}; median: {np.median(reprojection_errors)}; max error: {np.max(reprojection_errors)}; min error: {np.min(reprojection_errors)}")

        
