import piexif # pip install piexif
import numpy as np

'''
The Exif orientation value gives the orientation of the camera
relative to the scene when the image was captured.  The relation
of the '0th row' and '0th column' to visual position is shown as
below.

0th Row     | 0th Column
------------+-----------
top         | left side
top         | right side
bottom      | right side
bottom      | left side
left side   | top
right side  | top
right side  | bottom
left side   | bottom

For convenience, here is what the letter F would look like if it were
tagged correctly and displayed by a program that ignores the orientation
tag:
tag对应的图像是原始图像的方向，经过exif变换后得到tag1

  1        2       3      4         5            6           7          8

888888  888888      88  88      8888888888  88                  88  8888888888
88          88      88  88      88  88      88  88          88  88      88  88
8888      8888    8888  8888    88          8888888888  8888888888          88
88          88      88  88
88          88  888888  888888
'''

def get_rotation_deg_z(exif_orientation):
    return {
        0: 0,
        3: 180,
        6: -90,
        8: 90
    }.get(exif_orientation, 0)

def get_unity_rotation_matrix(rotation_deg_z, EPSILON=1e-5):
    theta = np.radians(rotation_deg_z)
    c, s = np.cos(theta), np.sin(theta)
    unity_rotation_matrix = np.array([
        [c, -s, 0, 0],
        [s, c, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=float)
    unity_rotation_matrix[np.abs(unity_rotation_matrix)<EPSILON] = 0
    return unity_rotation_matrix

def exifori_to_unity_rotation_matrix(image_path):
    exif_orientation = piexif.load(image_path)["0th"][274]  # 或者直接通过参数传递exif_orientation
    valid_orientation = (0, 3, 6, 8)
    if exif_orientation not in valid_orientation:
        exif_orientation = 0
    rotation_deg_z = get_rotation_deg_z(exif_orientation)
    return get_unity_rotation_matrix(rotation_deg_z)

if __name__ == '__main__':
    image_path = '/home/takune/relocation/shi_jing_shan/media/images/gxl_03/color/frame-000000.color.jpg'
    m = exifori_to_unity_rotation_matrix(image_path)
    print(m)
