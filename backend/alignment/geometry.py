import numpy as np
import cv2

def scale_down(img_in: np.ndarray, s: int) -> np.ndarray:
    """
    Downscales img_in by integer factor s.
    Sums s x s blocks of pixels and normalizes to [0, 255] to preserve peaks.
    """
    if s <= 1:
        return img_in
    h, w = img_in.shape[:2]
    h_out = h // s
    w_out = w // s
    # Crop img_in to a multiple of s
    img_cropped = img_in[:h_out * s, :w_out * s]
    
    if len(img_cropped.shape) == 3:
        reshaped = img_cropped.reshape(h_out, s, w_out, s, 3)
        block_sum = reshaped.sum(axis=(1, 3))
    else:
        reshaped = img_cropped.reshape(h_out, s, w_out, s)
        block_sum = reshaped.sum(axis=(1, 3))
        
    # Max normalize to avoid flat plateaus of clipped 255s
    max_val = block_sum.max()
    if max_val > 0:
        return (block_sum / max_val * 255).astype(np.uint8)
    return block_sum.astype(np.uint8)

def transform_image(img: np.ndarray, M: np.ndarray, target_shape=None) -> np.ndarray:
    """Applies affine transform M to img."""
    if target_shape is None:
        h, w = img.shape[:2]
    else:
        h, w = target_shape[:2]
        
    return cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

def transform(img: np.ndarray, M: np.ndarray, target_shape=None) -> np.ndarray:
    """Applies affine transform M to img (alias for transform_image)."""
    return transform_image(img, M, target_shape)

def compose_transforms(T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
    """Composes two 2x3 affine transformation matrices T1 and T2 (T1 * T2)."""
    T1_3x3 = np.vstack([T1, [0, 0, 1]])
    T2_3x3 = np.vstack([T2, [0, 0, 1]])
    out_3x3 = np.dot(T1_3x3, T2_3x3)
    return out_3x3[:2, :].astype(np.float32)
