import cv2
import numpy as np
from typing import Dict, Optional, Tuple
from lightglue.utils import read_image, resize_image, rbd, numpy_image_to_torch

def process_single_image(image_path: str, 
                                 H: int = 640, W: int = 480,
                                 is_resize: bool = True, 
                                 augment_params: Optional[Dict] = None,
                                 feature_enhance: bool = True) -> np.ndarray:
    """
    对单张图像进行增强处理，为后续特征点提取与匹配做准备
    
    Args:
        image_path: Path to input image
        from_front: Whether image is from front view
        H, W: Target height and width
        is_resize: Whether to resize image
        augment_params: Augmentation parameters dict
        feature_enhance: Whether to apply feature-friendly enhancements
    
    Returns:
        Processed image as numpy array
    """
    # Read and transform image
    image = read_image(image_path)
    
    # Resize if needed
    if is_resize:
        image, _ = resize_image(image, (H, W))

    # Convert to grayscale with detail preservation
    if len(image.shape) == 3:
        # Weighted grayscale conversion (preserves more detail than cv2.cvtColor)
        gray = 0.299 * image[:,:,2] + 0.587 * image[:,:,1] + 0.114 * image[:,:,0]
        gray = gray.astype(np.uint8)
    else:
        gray = image.copy()
    
    # Feature-friendly preprocessing
    if feature_enhance:
        gray = feature_friendly_processing(gray)
    
    # Adaptive histogram enhancement
    gray = adaptive_histogram_enhancement(gray)
    
    # Conditional filtering
    gray = conditional_filtering(gray)
    
    # Apply data augmentation if specified
    if augment_params:
        gray = apply_augmentation(gray, augment_params)
    
    # Convert back to 3 channels
    image_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    return np.uint8(image_3ch)

def feature_friendly_processing(image: np.ndarray) -> np.ndarray:
    """
    增强灰度图中的角点、边缘和纹理，利于特征提取
    
    Args:
        image: Input grayscale image
        
    Returns:
        Enhanced image
    """
    # Edge-preserving denoising
    denoised = cv2.bilateralFilter(image, 9, 75, 75)
    
    # Enhance texture and details using adaptive thresholding
    adaptive_thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Combine original and thresholded image to enhance features
    enhanced = cv2.addWeighted(denoised, 0.8, 
                              cv2.bitwise_not(adaptive_thresh), 0.2, 0)
    
    return enhanced

def adaptive_histogram_enhancement(image: np.ndarray) -> np.ndarray:
    """
    根据图像的对比度和亮度，自适应地增强图像的对比度
    
    Args:
        image: Input grayscale image
        
    Returns:
        Enhanced image
    """
    # Calculate image contrast
    contrast = np.std(image)
    mean_brightness = np.mean(image)
    
    if contrast < 25:  # Low contrast
        # Use CLAHE with higher clip limit
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        return clahe.apply(image)
    elif contrast > 80:  # High contrast
        if mean_brightness < 100:  # Dark image with high contrast
            # Light enhancement for dark regions
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
            return clahe.apply(image)
        else:
            return image  # Keep as is
    else:  # Medium contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)

def conditional_filtering(image: np.ndarray) -> np.ndarray:
    """
    根据图像清晰度（通过 Laplacian 方差）选择性滤波
    
    Args:
        image: Input grayscale image
        
    Returns:
        Filtered image
    """
    # Calculate image sharpness using Laplacian variance
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
    
    if laplacian_var > 800:  # Very sharp image
        # Light Gaussian blur to reduce noise while preserving features
        return cv2.GaussianBlur(image, (3, 3), 0.5)
    elif laplacian_var < 50:  # Blurry image
        # Apply unsharp masking
        gaussian = cv2.GaussianBlur(image, (5, 5), 2.0)
        unsharp_mask = cv2.addWeighted(image, 2.0, gaussian, -1.0, 0)
        return np.clip(unsharp_mask, 0, 255).astype(np.uint8)
    elif laplacian_var < 200:  # Moderately blurry
        # Mild sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) / 9.0
        kernel[1,1] = 2.0
        sharpened = cv2.filter2D(image, -1, kernel)
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    else:
        return image  # Good sharpness, keep as is

def apply_augmentation(image: np.ndarray, params: Dict) -> np.ndarray:
    """
    实现轻量数据增强，保持特征匹配稳定性的同时提高鲁棒性
    
    Args:
        image: Input grayscale image
        params: Dictionary of augmentation parameters
        
    Returns:
        Augmented image
    """
    augmented = image.copy()
    
    # Brightness and contrast adjustment
    if params.get('brightness', False):
        alpha = np.random.uniform(0.85, 1.15)  # Contrast
        beta = np.random.randint(-15, 15)      # Brightness
        augmented = cv2.convertScaleAbs(augmented, alpha=alpha, beta=beta)
    
    # Add slight noise to improve robustness
    if params.get('noise', False):
        noise_level = np.random.uniform(1, 8)
        noise = np.random.normal(0, noise_level, augmented.shape)
        augmented = np.clip(augmented + noise, 0, 255).astype(np.uint8)
    
    # Slight geometric transformation (preserving feature relationships)
    if params.get('affine', False):
        rows, cols = augmented.shape
        # Small random translation and rotation
        dx, dy = np.random.uniform(-3, 3, 2)
        angle = np.random.uniform(-2, 2)  # degrees
        
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        M[0, 2] += dx
        M[1, 2] += dy
        augmented = cv2.warpAffine(augmented, M, (cols, rows), 
                                 borderMode=cv2.BORDER_REFLECT)
    
    # Perspective transformation (very subtle)
    if params.get('perspective', False):
        rows, cols = augmented.shape
        # Define slight perspective change
        offset = np.random.uniform(-2, 2, 8).reshape(4, 2)
        pts1 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
        pts2 = np.float32(pts1 + offset)
        
        M = cv2.getPerspectiveTransform(pts1, pts2)
        augmented = cv2.warpPerspective(augmented, M, (cols, rows),
                                      borderMode=cv2.BORDER_REFLECT)
    
    return augmented

def get_default_augment_params(training: bool = True) -> Dict:
    """
    Get default augmentation parameters.
    
    Args:
        training: Whether in training mode (more aggressive augmentation)
        
    Returns:
        Dictionary of augmentation parameters
    """
    if training:
        return {
            'brightness': True,
            'noise': True,
            'affine': True,
            'perspective': False  # Use sparingly
        }
    else:
        return {
            'brightness': False,
            'noise': False,
            'affine': False,
            'perspective': False
        }

