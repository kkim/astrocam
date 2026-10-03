"""
alignment_utils.py - Backwards-compatible facade for AstroCam image alignment.

This module re-exports core functionality from the modular `alignment` package
(StarAligner, StarDetector, StarMatcher, TransformEstimator, etc.) to ensure
seamless backwards compatibility across the codebase.
"""

import numpy as np
import cv2

from alignment import (
    AlignmentConfig,
    AlignmentResult,
    AlignmentStatus,
    ReferenceFeatures,
    scale_down,
    transform,
    transform_image,
    compose_transforms,
    StarDetector,
    detect_stars,
    StarMatcher,
    compute_star_descriptors,
    match_star_descriptors,
    TransformEstimator,
    StarAligner,
    align_images,
)

def accumulate_panorama_frame(
    sum_img: np.ndarray,
    sum_wgt: np.ndarray,
    offset_img0: tuple,
    T_prev_to_0: np.ndarray,
    img_prev: np.ndarray,
    img: np.ndarray,
    translation_only: bool = True
):
    """
    Aligns the current frame `img` to `img_prev` (if provided), updates the cumulative 
    transform matrix mapping back to frame 0, and warps/accumulates the frame 
    into the panorama buffers.
    """
    if img_prev is None:
        # For the first frame (img0), T_curr_to_0 is just the initial cumulative transform (Identity)
        T_curr_to_0 = T_prev_to_0
    else:
        # Align current frame to previous frame
        T_step = align_images(img_prev, img, translation_only=translation_only)
        # Compose with the previous cumulative transform
        T_curr_to_0 = compose_transforms(T_prev_to_0, T_step)
    
    # Compute absolute transformation to the panorama buffer coordinates
    base_x, base_y = offset_img0
    T_buffer = T_curr_to_0.copy()
    T_buffer[0, 2] += base_x
    T_buffer[1, 2] += base_y
    
    # Warp the current frame into the buffer shape
    buf_h, buf_w = sum_img.shape[:2]
    warped_frame = transform(img, T_buffer, target_shape=(buf_h, buf_w))
    
    # Warp a binary coverage mask to know exactly which pixels were populated
    h, w = img.shape[:2]
    frame_mask = np.ones((h, w), dtype=np.float32)
    warped_mask = cv2.warpAffine(
        frame_mask, T_buffer, (buf_w, buf_h), 
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    mask = warped_mask > 0.5
    
    # Accumulate into the buffers
    sum_img[mask] += warped_frame[mask].astype(np.float32)
    sum_wgt[mask] += 1.0
    
    return sum_img, sum_wgt, T_curr_to_0

__all__ = [
    "AlignmentConfig",
    "AlignmentResult",
    "AlignmentStatus",
    "ReferenceFeatures",
    "scale_down",
    "transform",
    "transform_image",
    "compose_transforms",
    "StarDetector",
    "detect_stars",
    "StarMatcher",
    "compute_star_descriptors",
    "match_star_descriptors",
    "TransformEstimator",
    "StarAligner",
    "align_images",
    "accumulate_panorama_frame",
]
