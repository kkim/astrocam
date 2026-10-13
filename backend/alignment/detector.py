from typing import List, Tuple, Optional
import numpy as np
import cv2
from .models import AlignmentConfig


def find_bright_star_centroid(
    img: np.ndarray,
    min_snr: float = 3.0,
    min_peak_above_bg: float = 6.0,
    patch_radius: int = 15,
    search_center: Optional[Tuple[float, float]] = None,
    search_radius: Optional[int] = None,
) -> Optional[Tuple[float, float, float, float]]:
    """
    Locates the target star using blurred peak detection and sub-pixel center-of-mass centroiding.
    
    If search_center=(x, y) is specified, restricts the search to a local search window (lock box)
    and selects the candidate closest to the expected position, preventing confusion if another
    star of similar brightness is in the field of view.
    
    Args:
        img: Input image (BGR or grayscale, uint8/uint16/float).
        min_snr: Minimum peak-to-noise ratio to accept as a valid star.
        min_peak_above_bg: Minimum intensity increment above background.
        patch_radius: Radius around peak pixel for center-of-mass calculation.
        search_center: Optional (x, y) coordinates of expected star location.
        search_radius: Radius in pixels around search_center to search (default 80).
        
    Returns:
        (cx, cy, peak_val, bg_val) if a distinct star peak is detected, else None.
    """
    if img is None:
        return None

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    gray_f = gray.astype(np.float32)
    bg_val = float(np.median(gray_f))
    std_val = float(np.std(gray_f))
    
    if std_val < 1e-4:
        return None

    # Gaussian blur (7x7, sigma=1.5) filters single-pixel sensor noise and hot pixels
    blurred = cv2.GaussianBlur(gray_f, (7, 7), 1.5)
    h, w = gray.shape[:2]

    if search_center is not None:
        cx_exp, cy_exp = search_center
        sr = search_radius if search_radius is not None else 80
        x1 = max(0, int(cx_exp - sr))
        x2 = min(w, int(cx_exp + sr + 1))
        y1 = max(0, int(cy_exp - sr))
        y2 = min(h, int(cy_exp + sr + 1))

        roi = blurred[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        # Detect local peaks inside search ROI
        dilated = cv2.dilate(roi, None)
        local_max = (roi == dilated) & (roi > bg_val + min_peak_above_bg)
        py, px = np.where(local_max)

        if len(py) > 0:
            # If multiple peaks exist in ROI, select the peak closest to expected search_center
            best_dist = float("inf")
            best_x0, best_y0 = None, None
            for r_y, r_x in zip(py, px):
                abs_x = x1 + r_x
                abs_y = y1 + r_y
                dist = (abs_x - cx_exp) ** 2 + (abs_y - cy_exp) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_x0, best_y0 = abs_x, abs_y
            x0, y0 = best_x0, best_y0
            max_val = float(blurred[y0, x0])
        else:
            _, roi_max, _, roi_loc = cv2.minMaxLoc(roi)
            x0 = x1 + roi_loc[0]
            y0 = y1 + roi_loc[1]
            max_val = float(roi_max)
    else:
        _, max_val, _, max_loc = cv2.minMaxLoc(blurred)
        x0, y0 = max_loc

    peak_delta = max_val - bg_val
    peak_snr = peak_delta / max(std_val, 1e-4)

    if peak_snr < min_snr or peak_delta < min_peak_above_bg:
        return None

    h, w = gray.shape[:2]
    x_min = max(0, x0 - patch_radius)
    x_max = min(w, x0 + patch_radius + 1)
    y_min = max(0, y0 - patch_radius)
    y_max = min(h, y0 + patch_radius + 1)

    # Sub-pixel center-of-mass on background-subtracted patch
    patch = np.maximum(gray_f[y_min:y_max, x_min:x_max] - bg_val, 0.0)
    p_max = float(np.max(patch))
    if p_max <= 0:
        return float(x0), float(y0), float(max_val), bg_val

    # Threshold patch at 15% of local peak to ignore distant background fluctuations
    patch_th = np.where(patch > 0.15 * p_max, patch, 0.0)
    total_mass = float(np.sum(patch_th))
    if total_mass <= 0:
        return float(x0), float(y0), float(max_val), bg_val

    yy, xx = np.indices(patch.shape)
    cx = float(np.sum(xx * patch_th) / total_mass + x_min)
    cy = float(np.sum(yy * patch_th) / total_mass + y_min)

    return cx, cy, float(max_val), bg_val


class StarDetector:
    """
    Detects star centroids in images, with the brightest guide star first,
    followed by secondary stars sorted by intensity.
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
        Finds star centroids in img_gray. Prioritizes the brightest star as the
        first element, followed by secondary stars above the background noise floor.
        """
        if img_gray is None:
            return []

        th = threshold if threshold is not None else self.config.detection_threshold
        dist = min_dist if min_dist is not None else self.config.detection_min_dist
        max_c = max_candidates if max_candidates is not None else self.config.max_candidates
        win = self.config.centroid_window
        half_win = max(win // 2, 4)

        stars: List[Tuple[float, float]] = []

        # 1. First find the primary brightest guide star
        bright_res = find_bright_star_centroid(
            img_gray,
            min_snr=self.config.min_snr,
            min_peak_above_bg=self.config.min_peak_above_bg,
            patch_radius=self.config.patch_radius
        )
        if bright_res is not None:
            stars.append((bright_res[0], bright_res[1]))

        # 2. Search for additional stars using blurred dilation to filter sensor noise
        gray_f = img_gray.astype(np.float32)
        bg_val = float(np.median(gray_f))
        blurred = cv2.GaussianBlur(gray_f, (7, 7), 1.5)

        # Baseline-relative threshold to prevent thousands of false detections on gray background
        eff_th = bg_val + max(float(th), 4.0)
        dilated = cv2.dilate(blurred, None)
        local_max = (blurred == dilated) & (blurred > eff_th)
        y_coords, x_coords = np.where(local_max)

        if len(y_coords) == 0:
            return stars

        intensities = blurred[y_coords, x_coords]
        sort_idx = np.argsort(intensities)[::-1][:max_c]

        h, w = img_gray.shape[:2]
        dist_sq = float(dist ** 2)

        for idx in sort_idx:
            x = int(x_coords[idx])
            y = int(y_coords[idx])
            if x < half_win or x >= w - half_win or y < half_win or y >= h - half_win:
                continue

            # Non-Maximum Suppression against already found stars
            too_close = False
            for sx, sy in stars:
                if (x - sx) ** 2 + (y - sy) ** 2 < dist_sq:
                    too_close = True
                    break
            if too_close:
                continue

            # Compute sub-pixel centroid
            patch = np.maximum(gray_f[y - half_win : y + half_win + 1, x - half_win : x + half_win + 1] - bg_val, 0.0)
            p_max = float(np.max(patch))
            if p_max <= 0:
                stars.append((float(x), float(y)))
                continue

            patch_th = np.where(patch > 0.15 * p_max, patch, 0.0)
            patch_sum = float(np.sum(patch_th))
            if patch_sum > 0:
                yy, xx = np.indices(patch.shape)
                cx = float(np.sum(xx * patch_th) / patch_sum + (x - half_win))
                cy = float(np.sum(yy * patch_th) / patch_sum + (y - half_win))
                stars.append((cx, cy))
            else:
                stars.append((float(x), float(y)))

        return stars


def detect_stars(img_gray: np.ndarray, threshold: float = 15, min_dist: int = 5) -> List[Tuple[float, float]]:
    """Helper function for standalone star detection."""
    return StarDetector().detect(img_gray, threshold=threshold, min_dist=min_dist)
