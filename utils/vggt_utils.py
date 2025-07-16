from PIL import Image
import cv2
import numpy as np
import os

def resize_image_to_square(image_cv, target_size=518):
    H, W, C = image_cv.shape[:2]
    assert C >= 3

    # If there's an alpha channel, blend onto white background
    if C == 4:
        background = Image.new("RGBA", image_cv.size, (255, 255, 255, 255))
        image_pil = Image.fromarray(image_cv)
        image_pil = Image.alpha_composite(background, image_pil)
        image_pil = image_cv.convert("RGB")
    else:
        image_pil = Image.fromarray(image_cv)

    max_dim = max(H, W)

    # Calculate padding
    left = (max_dim - W) // 2
    top = (max_dim - H) // 2
    
    scale = target_size / max_dim

    # Calculate final coordinates of original image in target space
    x1 = left * scale
    y1 = top * scale
    x2 = (left + W) * scale
    y2 = (top + H) * scale

    image_pil_square = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
    image_pil_square.paste(image_pil)
    image_pil_square = image_pil_square.resize((target_size, target_size), Image.Resampling.BICUBICA)

    return np.array(image_pil_square)
    
def read_point_map_from_vggt(point2d_list, image_idx, pointmap_loc):
    pointmap_name = f'{image_idx}.pointmap.npy' # !!!修改成文件保存的位置!!!
    pointmap_path = os.path.join(pointmap_loc, pointmap_name)
    point_map = np.load(pointmap_path)
    x, y = point2d_list[:, 0], point2d_list[:, 1]
    return point_map[y, x] # !!!需要根据point2d坐标系调整x,y的顺序!!!
