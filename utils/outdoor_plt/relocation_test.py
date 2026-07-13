import numpy as np
from pathlib import Path
import shutil
import os
import subprocess
from .read_write_model import read_model
import time
from .exifori_to_unitymatrix import exifori_to_unity_rotation_matrix

SCRIPT_DIR=Path(os.path.dirname(os.path.abspath(__file__)))

def create_tmp_workspace(image_path, database_path, tmp_folder_path):
    tmp_image_folder_path = Path(tmp_folder_path) / 'images'
    database_image_folder_path = Path(database_path)
    tmp_image_folder_path.mkdir(parents=True, exist_ok=True)

    shutil.copy(image_path, tmp_image_folder_path)
    for filename in os.listdir(database_image_folder_path):
        database_image_path = database_image_folder_path / filename
        if os.path.isfile(database_image_path):
            shutil.copy(database_image_path, tmp_image_folder_path)

def run_vggt(tmp_folder_path):
    vggt_script_path = str(SCRIPT_DIR / 'run_vggt.sh')
    subprocess.run([vggt_script_path, tmp_folder_path])

def qvec2rotmat(q):
    """
    将四元数转换为旋转矩阵
    :param q: 四元数 [w, x, y, z]
    :return: 旋转矩阵 (3x3)
    """
    w, x, y, z = q
    # 计算旋转矩阵的每个元素
    R = np.array([
        [1 - 2 * (y**2 + z**2),   2 * (x * y - w * z),   2 * (x * z + w * y)],
        [2 * (x * y + w * z),   1 - 2 * (x**2 + z**2),   2 * (y * z - w * x)],
        [2 * (x * z - w * y),   2 * (y * z + w * x),   1 - 2 * (x**2 + y**2)]
    ])
    return R

def get_image_pose_by_name(images: dict, image_name: str):
    """
    在 COLMAP images dict 中按文件名查找 Image 对象，并返回其 4x4 变换矩阵。

    :param images: 从 read_model 得到的 dict[int, Image]
    :param image_name: 要匹配的文件名（区分大小写）
    :return: 4x4 变换矩阵；未找到返回 None
    """
    for img in images.values():  # img 是 Image 对象
        if img.name == image_name:
            qvec = img.qvec
            tvec = img.tvec
            R = qvec2rotmat(qvec)  # 旋转矩阵
            t = tvec.reshape((3, 1))  # 平移向量
            # 构造 4x4 变换矩阵
            pose_matrix = np.vstack((np.hstack((R, t)), [0, 0, 0, 1]))
            return pose_matrix
    return None

def compute_M_A2B(A, B='RDF'):
    '''
    计算数据集坐标系到目标坐标系的转换矩阵M_A2B
    A: 初始坐标系，格式如 RDF
    B: 目标坐标系，格式如 ULB
    '''
    # 定义坐标系方向的映射（标准世界坐标系中的单位向量）
    direction_map = {
        'R': [1, 0, 0],
        'L': [-1, 0, 0],
        'U': [0, 1, 0],
        'D': [0, -1, 0],
        'F': [0, 0, 1],
        'B': [0, 0, -1]
    }

    # 构建坐标系矩阵
    # 每一列代表该坐标系的一个轴在世界坐标系中的方向
    dataset_matrix = np.array([
        direction_map[A[0]],  # X轴方向
        direction_map[A[1]],  # Y轴方向
        direction_map[A[2]]   # Z轴方向
    ]).T

    target_matrix = np.array([
        direction_map[B[0]],
        direction_map[B[1]],
        direction_map[B[2]]
    ]).T

    # 验证坐标系是否正交
    def is_orthogonal(matrix):
        return np.allclose(matrix @ matrix.T, np.eye(3))

    if not is_orthogonal(dataset_matrix):
        raise ValueError("Dataset A system is not orthogonal")
    if not is_orthogonal(target_matrix):
        raise ValueError("Target A system is not orthogonal")

    M_A2B = target_matrix.T @ dataset_matrix
    M_A2B_homo = np.eye(4)
    M_A2B_homo[:3, :3] = M_A2B

    return M_A2B_homo

def calculate_pose_unity(query_image_path, database_path, tmp_folder_path):
    database_model_path = str(Path(database_path) / 'sparse')
    tmp_model_path = str(Path(tmp_folder_path) / 'sparse')
    _, images_db, _ = read_model(database_model_path, ext='.bin')
    _, images_tmp, _ = read_model(tmp_model_path, ext='.bin')

    query_image_name = os.path.basename(query_image_path)
    pose_image1_db = get_image_pose_by_name(images_db, 'frame-000001.color.jpg')
    pose_image1_tmp = get_image_pose_by_name(images_tmp, 'frame-000001.color.jpg')
    pose_image_query_tmp = get_image_pose_by_name(images_tmp, query_image_name)
    pose_cv_w2c = pose_image_query_tmp @ np.linalg.inv(pose_image1_tmp) @ pose_image1_db
    
    pose_cv_c2w = np.linalg.inv(pose_cv_w2c)
    M_opencv2unity = compute_M_A2B('RDF', 'RUF')
    pose_unity_c2w = M_opencv2unity @ pose_cv_c2w @ np.linalg.inv(M_opencv2unity)
    print_pose(pose_unity_c2w)
    pose_unity_c2w =  pose_unity_c2w @ np.linalg.inv(exifori_to_unity_rotation_matrix(query_image_path))
    
    return pose_unity_c2w 

def relocate(image_path, database_path, tmp_folder_path):
    create_tmp_workspace(image_path, database_path, tmp_folder_path)
    run_vggt(tmp_folder_path)
    pose_unity = calculate_pose_unity(image_path, database_path, tmp_folder_path)
    return pose_unity

def print_pose(pose):
    for i in range(pose.shape[0]):
        print('[', end='')
        for j in range(pose.shape[1]):
            print(pose[i][j], end='')
            if j < pose.shape[1] - 1:
                print(',', end='')
        print(']', end='')
        if i < pose.shape[0] - 1:
            print(',')
        else:
            print()

def main():
    image_path = '/home/tcluan/0-Desktop/gxl_test.jpg'
    database_path = '/home/tcluan/data/Outdoor_PLT/nmb_platform'
    pose = relocate(image_path, database_path)
    print_pose(pose)

def test():
    image_path = '/home/tcluan/0-Desktop/gxl_test.jpg'
    database_path = '/home/tcluan/data/Outdoor_PLT/nmb_platform'
    #create_tmp_workspace(image_path, database_path)
    #run_vggt()
    pose = calculate_pose_unity(image_path, database_path)
    print_pose(pose)

if __name__ == '__main__':
    main()
