import json
import os
import sys
import fnmatch
import shutil
import numpy as np
from PIL import Image

def get_image_data(image_path):
    """
    获取图片的EXIF数据。

    参数:
        image_path (str): 图片文件的路径。

    返回:
        list: 依次返回图片width、height以及旋转方向（1-8），默认值为1（正常方向）。
    """
    image = Image.open(image_path)
    orientation = image.getexif().get(274, 1)
    return [image.width, image.height, orientation]

def set_coordinate(type: str):
    """
    根据类型设置坐标轴转换矩阵。

    参数:
        type (str): 坐标系类型，支持 '3DS' 和 'opencv'。

    返回:
        dict: 
    """
    result = {}
    match type:
        case '3DS':
            result['X'] = "right"
            result['Y'] = "up"
            result['Z'] = "backward"
        case 'opencv':
            result['X'] = "right"
            result['Y'] = "down"
            result['Z'] = "forward"
        case _:
            raise ValueError("Unsupported coordinate type. Use '3DS' or 'opencv'.")
    return result

def transfer_json_2_pose(json_file, output_folder, new_name):
    with open(json_file, 'r', encoding='utf8') as fp:
        json_dict = json.load(fp)
        intrinsics_output_file = output_folder + os.sep + 'intrinsic' + os.sep + new_name + '.intrinsic_color.txt'
        pose_output_file = output_folder + os.sep + 'pose' + os.sep + new_name + '.pose.txt'

        intrinsics = json_dict['intrinsics']
        # print(intrinsics)
        fx = intrinsics[0]
        fy = intrinsics[4]
        cx = intrinsics[2]
        cy = intrinsics[5]
        A = np.zeros((4, 4))
        A[0, 0] = fx
        A[1, 1] = fy
        A[0, 2] = cx
        A[1, 2] = cy
        A[2, 2] = 1
        A[3, 3] = 1
        np.savetxt(intrinsics_output_file, A, fmt='%6f')

        cameraPoseARFrame = json_dict['cameraPoseARFrame']
        P = np.array(cameraPoseARFrame)
        P = P.reshape((4, 4))
        np.savetxt(pose_output_file, P, fmt='%6f')


def make_3ds_dataset(input_folder, output_folder, info):
    if len(input_folder) != 0:
        print("使用预定义input数据集位置: " + input_folder)
    elif sys.argv and len(sys.argv[0]) != 0:
        input_folder = sys.argv[0]
        print("使用传入input数据集位置: " + input_folder)
    else:
        return "input数据集参数错误"

    if len(output_folder) != 0:
        print("使用预定义output数据集位置: " + output_folder)
    elif sys.argv and len(sys.argv[1]) != 0:
        output_folder = sys.argv[1]
        print("使用传入output数据集位置: " + output_folder)
    else:
        return "output数据集参数错误"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    os.makedirs(output_folder + os.sep + 'color')
    os.makedirs(output_folder + os.sep + 'depth')
    os.makedirs(output_folder + os.sep + 'intrinsic')
    os.makedirs(output_folder + os.sep + 'pose')

    img_list = os.listdir(input_folder)
    count = 0

    # collect per-frame intrinsics to compute averages
    fx_vals = []
    fy_vals = []
    cx_vals = []
    cy_vals = []
    first_image_info = None

    for frame in img_list:
        if fnmatch.fnmatch(frame, 'frame_*.jpg'):
            if first_image_info is None:
                try:
                    first_image_info = get_image_data(os.path.join(input_folder, frame))
                except Exception:
                    first_image_info = None

            new_name = 'frame-' + str(count).zfill(6)
            shutil.copyfile(input_folder + os.sep + frame,
                            output_folder + os.sep + 'color' + os.sep + new_name + '.color.jpg')

            pose = frame.replace('jpg', 'json')
            pose_path = input_folder + os.sep + pose
            if os.path.exists(pose_path):
                try:
                    with open(pose_path, 'r', encoding='utf8') as jf:
                        jdict = json.load(jf)
                        intr = jdict.get('intrinsics', None)
                        if intr and len(intr) >= 6:
                            fx_vals.append(float(intr[0]))
                            fy_vals.append(float(intr[4]))
                            cx_vals.append(float(intr[2]))
                            cy_vals.append(float(intr[5]))
                except Exception:
                    pass

            transfer_json_2_pose(input_folder + os.sep + pose,
                                 output_folder,
                                 new_name)
            count += 1
        else:
            continue

    # compute averaged intrinsics (mean across frames) with one decimal
    def mean_or_fallback(vals, key):
        if vals:
            return round(float(np.mean(vals)), 1)
        return round(float(info.get('intrinsic', {}).get(key, 0.0)), 1)

    fx = mean_or_fallback(fx_vals, 'fx')
    fy = mean_or_fallback(fy_vals, 'fy')
    cx = mean_or_fallback(cx_vals, 'cx')
    cy = mean_or_fallback(cy_vals, 'cy')

    # image size and exif from first image or fallback to info
    if first_image_info:
        width, height, orientation = first_image_info
    else:
        img_size = info.get('image_size', {})
        width = img_size.get('width', 0)
        height = img_size.get('height', 0)
        orientation = info.get('exif', 1)

    # coordinate from set_coordinate using info['type']
    coord = set_coordinate(info.get('type', '3DS'))

    info_output = {
        "name": info.get('name', ''),
        "type": info.get('type', ''),
        "Z_Near": info.get('Z_Near', 0.0),
        "Z_Far": info.get('Z_Far', 0.0),
        "coordinate": coord,
        "intrinsic": {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy
        },
        "image_size": {
            "width": int(width),
            "height": int(height)
        },
        "exif": int(orientation)
    }

    # save info.json to output_folder
    info_out_path = output_folder + os.sep + 'info.json'
    with open(info_out_path, 'w', encoding='utf8') as outf:
        json.dump(info_output, outf, ensure_ascii=False, indent=4)

    config_out = output_folder + os.sep + 'config.json'
    with open(config_out, 'w', encoding='utf8') as cf:
        json.dump({}, cf, ensure_ascii=False, indent=4)

    obj_path = input_folder + os.sep + 'textured_output.obj'
    mtl_path = input_folder + os.sep + 'textured_output.mtl'
    texture_path = input_folder + os.sep + 'textured_output.jpg'
    if os.path.exists(obj_path):
        shutil.copyfile(obj_path, output_folder + os.sep + 'textured_output.obj')
    if os.path.exists(mtl_path):
        shutil.copyfile(mtl_path, output_folder + os.sep + 'textured_output.mtl')
    if os.path.exists(texture_path):
        shutil.copyfile(texture_path, output_folder + os.sep + 'textured_output.jpg')

    return "数据集转换完毕"
