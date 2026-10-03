from typing import List, Tuple, Optional
import numpy as np
import cv2
from .models import AlignmentConfig

class StarDetector:
    """
    Detects star centroids in grayscale images using morphological dilation
    for local peak finding, Non-Maximum Suppression (NMS), and sub-pixel centroiding.
    """
    def __init__(self, config: Optional[AlignmentConfig] = None):
        self.config = config or AlignmentConfig()

    def detect(
        self,
        img_gray: np.ndarray,
        threshold: Optional[float] = None,
        min_dist: Optional[int] = None,
        max_candidates: Optional[int] = None
    ) -> List[Tuple[float, float]]:
        """
        Finds local maxima in img_gray that are above threshold,
        enforces a minimum distance (NMS) to prevent duplicate detections,
        and refines their positions to sub-pixel centroids.
        """
        th = threshold if threshold is not None else self.config.detection_threshold
        dist = min_dist if min_dist is not None else self.config.detection_min_dist
        max_c = max_candidates if max_candidates is not None else self.config.max_candidates
        win = self.config.centroid_window
        half_win = win // 2

        # Use dilation to find local maxima candidates
        dilated = cv2.dilate(img_gray, None)
        local_max = (img_gray == dilated) & (img_gray > th)
        y_coords, x_coords = np.where(local_max)

        if len(y_coords) == 0:
            return []

        # Sort local maxima by intensity to prioritize strongest stars in NMS
        intensities = img_gray[y_coords, x_coords]
        sort_idx = np.argsort(intensities)[::-1]

        # Limit candidates to keep the NMS loop fast
        sort_idx = sort_idx[:max_c]

        stars: List[Tuple[float, float]] = []
        h, w = img_gray.shape[:2]
        dist_sq = dist ** 2

        for idx in sort_idx:
            x = int(x_coords[idx])
            y = int(y_coords[idx])
            if x < half_win or x >= w - half_win or y < half_win or y >= h - half_win:
                continue

            # Non-Maximum Suppression check
            too_close = False
            for sx, sy in stars:
                if (x - sx) ** 2 + (y - sy) ** 2 < dist_sq:
                    too_close = True
                    break
            if too_close:
                continue

            # Compute subpixel centroid in a window (center of gravity)
            patch = img_gray[y - half_win : y + half_win + 1, x - half_win : x + half_win + 1].astype(np.float32)
            patch_sum = patch.sum()
            if patch_sum > 0:
                xx, yy = np.meshgrid(
                    np.arange(x - half_win, x + half_win + 1),
                    np.arange(y - half_win, y + half_win + 1)
                )
                cx = float((xx * patch).sum() / patch_sum)
                cy = float((yy * patch).sum() / patch_sum)
                stars.append((cx, cy))

        return stars

def detect_stars(img_gray: np.ndarray, threshold: float = 15, min_dist: int = 5) -> List[Tuple[float, float]]:
    """Legacy helper function for standalone star detection."""
    return StarDetector().detect(img_gray, threshold=threshold, min_dist=min_dist)
