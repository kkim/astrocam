#!/usr/bin/env python3
"""
AstroCam Star Alignment & Mosaic Stitcher
==========================================
Aligns a sequence of astrophotography frames containing a bright guide star,
builds a full-FOV weighted average mosaic, and exports high dynamic range (HDR)
renders to reveal fine faint details.
"""

import os
import sys
import glob
import json
import time
import argparse
from typing import Optional, Tuple, List, Dict, Any

# Ensure venv Python is used if running with a bare system interpreter lacking dependencies
try:
    import cv2
    import numpy as np
except ImportError:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    venv_py = os.path.abspath(os.path.join(script_dir, "../backend/venv/bin/python"))
    if os.path.exists(venv_py) and sys.executable != venv_py:
        os.execv(venv_py, [venv_py] + sys.argv)
    raise


def write_progress(
    output_dir: Optional[str],
    stage: str,
    step: int,
    total_steps: int,
    current: int,
    total: int,
    percent: float,
    message: str
) -> None:
    """Atomically writes live progress state to .progress.json inside output_dir."""
    if not output_dir:
        return
    try:
        os.makedirs(output_dir, exist_ok=True)
        p_path = os.path.join(output_dir, ".progress.json")
        p_tmp = os.path.join(output_dir, ".progress.json.tmp")
        payload = {
            "stage": stage,
            "step": step,
            "total_steps": total_steps,
            "current": int(current),
            "total": int(total),
            "percent": round(float(percent), 1),
            "message": message,
            "timestamp": time.time(),
        }
        with open(p_tmp, "w") as f:
            json.dump(payload, f)
        os.replace(p_tmp, p_path)
    except Exception:
        pass


def detect_corruption_stripes(
    img: np.ndarray,
    min_stripe_width: int = 60,
    min_aspect_ratio: float = 3.5,
    vertical_margin: int = 2,
) -> List[Tuple[int, int, int, int]]:
    """
    Detects horizontal "white stripe" artifacts caused by dropped MJPEG packets.

    When the camera drops USB packets in MJPEG mode, one or more MCU rows
    (typically 8-16 px tall) decode as a flat/noisy bright band spanning a
    large part of the frame width. Stars are compact (~10-30 px wide), so
    stripes are separated from stars by their width and horizontal aspect ratio.

    Args:
        img: Input frame (grayscale or BGR, uint8 or uint16).
        min_stripe_width: Minimum horizontal extent (px) of a stripe component.
        min_aspect_ratio: Minimum width/height ratio of a stripe component.
        vertical_margin: Extra rows masked above/below each stripe to cover bleed.

    Returns:
        List of (x0, y0, x1, y1) rectangles (exclusive end) covering each stripe.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape
    scale = 257.0 if gray.dtype == np.uint16 else 1.0

    gray_f = gray.astype(np.float32)
    bg_median = float(np.median(gray_f))
    bg_std = float(np.std(gray_f))

    # A stripe pixel must be well above the sky background
    thresh = max(bg_median + 3.5 * bg_std, bg_median * 1.8, 90.0 * scale)
    bright = (gray_f >= thresh).astype(np.uint8)
    if not np.any(bright):
        return []

    # Bridge small horizontal gaps inside noisy stripes
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, close_kernel)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    rects = []
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        aspect = cw / max(ch, 1)
        if (cw >= min_stripe_width and aspect >= min_aspect_ratio) or cw >= w * 0.25:
            # Mask the full bounding box: the whole MCU band is unreliable,
            # including darker gaps inside a noisy stripe.
            rects.append((
                int(x),
                max(0, int(y) - vertical_margin),
                int(x + cw),
                min(h, int(y + ch) + vertical_margin),
            ))
    return rects


def stripe_rects_to_mask(rects: List[Tuple[int, int, int, int]], h: int, w: int) -> np.ndarray:
    """Rasterizes stripe rectangles into a boolean (h, w) mask (True = corrupted)."""
    mask = np.zeros((h, w), dtype=bool)
    for x0, y0, x1, y1 in rects:
        mask[y0:y1, x0:x1] = True
    return mask


def detect_star_centroid(
    img: np.ndarray,
    min_snr: float = 4.0,
    min_peak_above_bg: float = 8.0,
    patch_radius: int = 15,
    exclude_mask: Optional[np.ndarray] = None,
) -> Optional[Tuple[float, float, float, float]]:
    """
    Locates the bright star with sub-pixel centroid accuracy.
    
    Args:
        img: Input image (BGR uint8/uint16 or grayscale float/int).
        min_snr: Minimum peak-to-background SNR to qualify as a valid star.
        min_peak_above_bg: Minimum intensity increment above background.
        patch_radius: Radius around the peak pixel used for centroid calculation.
        exclude_mask: Optional boolean (h, w) mask of pixels (e.g. corruption
            stripes) that must never be considered part of a star.
        
    Returns:
        (cx, cy, peak_val, bg_val) if star detected, else None.
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    gray_f = gray.astype(np.float64)
    if exclude_mask is not None and np.any(exclude_mask):
        valid = ~exclude_mask
        if not np.any(valid):
            return None
        bg_val = float(np.median(gray_f[valid]))
        std_val = float(np.std(gray_f[valid]))
        # Flatten excluded regions to background so they can't win the peak search
        gray_f = gray_f.copy()
        gray_f[exclude_mask] = bg_val
    else:
        bg_val = float(np.median(gray_f))
        std_val = float(np.std(gray_f))
    
    if std_val < 1e-4:
        # Uniform/blank flat frame
        return None

    # Gaussian blur to filter out single-pixel sensor hot noise
    blurred = cv2.GaussianBlur(gray_f, (7, 7), 1.5)
    _, max_val, _, max_loc = cv2.minMaxLoc(blurred)

    peak_delta = max_val - bg_val
    peak_snr = peak_delta / max(std_val, 1e-4)

    if peak_snr < min_snr or peak_delta < min_peak_above_bg:
        return None

    x0, y0 = max_loc
    h, w = gray.shape
    x_min = max(0, x0 - patch_radius)
    x_max = min(w, x0 + patch_radius + 1)
    y_min = max(0, y0 - patch_radius)
    y_max = min(h, y0 + patch_radius + 1)

    # Sub-pixel center of mass on background-subtracted patch
    patch = np.maximum(gray_f[y_min:y_max, x_min:x_max] - bg_val, 0.0)
    p_max = np.max(patch)
    if p_max <= 0:
        return float(x0), float(y0), max_val, bg_val

    # Threshold patch at 20% of local peak to ignore distant background fluctuations
    patch_th = np.where(patch > 0.20 * p_max, patch, 0.0)
    total_mass = np.sum(patch_th)
    if total_mass <= 0:
        return float(x0), float(y0), max_val, bg_val

    yy, xx = np.indices(patch.shape)
    cx = float(np.sum(xx * patch_th) / total_mass + x_min)
    cy = float(np.sum(yy * patch_th) / total_mass + y_min)

    return cx, cy, float(max_val), bg_val


def create_feather_mask(h: int, w: int, feather_px: int = 15) -> np.ndarray:
    """
    Creates a smooth 2D feathering weight mask with linear falloff at the borders.
    This eliminates sharp seams at overlapping tile boundaries.
    """
    if feather_px <= 0:
        return np.ones((h, w), dtype=np.float32)

    # Linear ramps for X and Y borders
    ramp_x = np.ones(w, dtype=np.float32)
    ramp_y = np.ones(h, dtype=np.float32)

    f_x = min(feather_px, w // 2)
    f_y = min(feather_px, h // 2)

    if f_x > 0:
        ramp_x[:f_x] = np.linspace(0.0, 1.0, f_x, dtype=np.float32)
        ramp_x[-f_x:] = np.linspace(1.0, 0.0, f_x, dtype=np.float32)

    if f_y > 0:
        ramp_y[:f_y] = np.linspace(0.0, 1.0, f_y, dtype=np.float32)
        ramp_y[-f_y:] = np.linspace(1.0, 0.0, f_y, dtype=np.float32)

    return np.outer(ramp_y, ramp_x)


def extract_defect_mask(arr: np.ndarray) -> np.ndarray:
    """
    Converts either a 2D binary defect mask or a 2D/3D master dark baseline frame
    into a 2D uint8 defect mask (1 = defect, 0 = healthy) using the standard
    max(med + 6, med + 5 * sigma) threshold.
    """
    if arr.ndim == 2 and arr.dtype in (np.uint8, np.bool_) and np.max(arr) <= 1:
        return (arr > 0).astype(np.uint8)
    if arr.ndim == 3:
        dark_gray = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        dark_gray = arr.astype(np.float32)
    med = float(np.median(dark_gray))
    mad = float(np.median(np.abs(dark_gray - med)))
    sigma = max(1.0, 1.4826 * mad)
    threshold = max(med + 6.0, med + 5.0 * sigma)
    return (dark_gray > threshold).astype(np.uint8)


def build_defect_weights(d_mask: np.ndarray, radius: int = 1, soft_px: int = 0) -> np.ndarray:
    """
    1.0 = healthy, 0.0 = defect or within `radius` px; optional linear
    ramp back to 1.0 over the next `soft_px` px.
    """
    bad = extract_defect_mask(d_mask)
    if radius > 0:
        # For radius=1, use MORPH_RECT (full 3x3 neighborhood) so diagonal Bayer/DCT
        # spill at (+/-1, +/-1) is masked; for larger radii use MORPH_ELLIPSE.
        shape = cv2.MORPH_RECT if radius == 1 else cv2.MORPH_ELLIPSE
        k = cv2.getStructuringElement(shape, (2 * radius + 1, 2 * radius + 1))
        bad = cv2.dilate(bad, k)
    if soft_px <= 0:
        return (1.0 - bad).astype(np.float64)
    # distance (px) from nearest bad pixel -> ramp 0..1
    dist = cv2.distanceTransform(1 - bad, cv2.DIST_L2, 3)
    return np.clip(dist / float(soft_px), 0.0, 1.0).astype(np.float64)


def stretch_hdr(
    img: np.ndarray,
    valid_mask: np.ndarray,
    method: str = "asinh",
    stretch_factor: float = 60.0,
    black_percentile: float = 0.5,
    white_percentile: float = 99.98,
) -> np.ndarray:
    """
    Applies astronomical non-linear stretch to reveal faint nebulosity/stars
    while keeping bright star cores under control.

    Args:
        img: Float array of the mosaic.
        valid_mask: 2D boolean mask of pixels that received data.
        method: 'asinh', 'mtf', 'log', or 'linear'.
        stretch_factor: Strength of stretch (beta for asinh, multiplier for log).
        black_percentile: Percentile of valid pixels used for black point.
        white_percentile: Percentile of valid pixels used for white point.

    Returns:
        Stretched float array normalized to [0.0, 1.0].
    """
    out = np.zeros_like(img, dtype=np.float32)
    if not np.any(valid_mask):
        return out

    # Determine black point and white point
    valid_pixels = img[valid_mask]
    med = float(np.median(valid_pixels))
    std = float(np.std(valid_pixels))
    max_v = float(np.max(valid_pixels))

    # Anchor black point slightly below the median sky background noise floor
    # This keeps deep space background at a pleasing dark charcoal (~5-8% intensity)
    # while allowing faint nebulosity and faint stars to emerge with high contrast.
    if black_percentile is not None and black_percentile > 0.0:
        p_bp = float(np.percentile(valid_pixels, black_percentile))
        bp = max(min(p_bp, med - 1.0 * std), 0.0)
    else:
        bp = max(med - 1.5 * std, 0.0)

    # White point anchored to peak to prevent star core saturation
    wp = max_v
    span = max(wp - bp, 1e-5)

    # Shift background to 0 and normalize to [0, 1]
    norm = np.clip((img - bp) / span, 0.0, 1.0).astype(np.float32)

    if method == "asinh":
        beta = max(float(stretch_factor), 1.0)
        stretched = np.arcsinh(beta * norm) / np.arcsinh(beta)
    elif method == "log":
        c = max(float(stretch_factor), 1.0)
        stretched = np.log1p(c * norm) / np.log1p(c)
    elif method == "mtf":
        # Midtone Transfer Function: mtf(m, x) with midtone parameter m in (0, 1)
        m = 0.08  # Target midtone elevation
        stretched = np.where(
            norm > 0,
            ((m - 1.0) * norm) / ((2.0 * m - 1.0) * norm - m),
            0.0
        )
    else:  # linear
        stretched = norm

    stretched[~valid_mask] = 0.0
    return np.clip(stretched, 0.0, 1.0).astype(np.float32)


def combine_images(
    input_dir: str,
    output_dir: Optional[str] = None,
    fov_mode: str = "full",
    feather_px: int = 12,
    stretch_method: str = "asinh",
    stretch_factor: float = 60.0,
    black_percentile: float = 0.5,
    max_frames: Optional[int] = None,
    defect_mask_path: Optional[str] = None,
    defect_radius: int = 1,
    defect_soft_px: int = 0,
    use_defect_mask: bool = True,
    save_individual_renders: bool = True,
    verbose: bool = True,
    filter_stripes: bool = True,
    max_stripe_fraction: float = 0.5,
) -> Dict[str, Any]:
    """
    Main function to align and mosaic astrophotography images by their bright star.

    Args:
        input_dir: Path to directory containing image files (.tif, .tiff, .png, etc.).
        output_dir: Directory to save outputs (defaults to <input_dir>/mosaic_v1).
        fov_mode: 'full' (canvas expands to fit entire drift FOV) or 'first' (fixed to 1st frame).
        feather_px: Pixels along image boundary for smooth seam feathering.
        stretch_method: 'asinh', 'mtf', 'log', or 'linear'.
        stretch_factor: Stretch strength parameter.
        black_percentile: Lower percentile for background black-point calibration.
        max_frames: Limit number of frames to process (useful for rapid testing).
        defect_mask_path: Path to defect_mask.npy (defaults to captures/calibration/defect_mask.npy).
        defect_radius: Dilation radius around bad pixels to zero-weight neighbor spill.
        defect_soft_px: Soft linear ramp distance in pixels outside the dilated defect zone.
        use_defect_mask: Whether to apply sensor defect masking.
        save_individual_renders: Save linear 16-bit, HDR 16-bit, HDR 8-bit, and weight map.
        verbose: Print progress to stdout.
        filter_stripes: Detect MJPEG packet-loss stripes, exclude them from star
            detection, and give them zero weight in the mosaic.
        max_stripe_fraction: Drop a frame entirely if stripes cover more than this
            fraction of its area.

    Returns:
        Dictionary with processing metadata, output paths, and resulting arrays.
    """
    input_dir = os.path.abspath(input_dir)
    if not os.path.isdir(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")

    if output_dir is None:
        output_dir = os.path.join(input_dir, "mosaic_v1")
    os.makedirs(output_dir, exist_ok=True)

    # Gather image files
    all_files = sorted([
        os.path.join(input_dir, f) for f in os.listdir(input_dir)
        if not f.startswith(".") and f.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg", ".fits"))
    ])

    # Filter out empty (0-byte) files
    valid_files = [f for f in all_files if os.path.getsize(f) > 0]
    if not valid_files:
        raise ValueError(f"No non-empty image files found in {input_dir}")

    if max_frames is not None and max_frames > 0:
        valid_files = valid_files[:max_frames]

    if verbose:
        print(f"============================================================")
        print(f" AstroCam Star-Aligned Mosaic Stitcher")
        print(f"============================================================")
        print(f"Input Directory  : {input_dir}")
        print(f"Total Files Found: {len(all_files)} ({len(valid_files)} non-empty)")
        print(f"FOV Mode         : {fov_mode.upper()}")
        print(f"Stretch Method   : {stretch_method.upper()} (factor={stretch_factor})")
        print(f"Output Directory : {output_dir}")
        print(f"------------------------------------------------------------")

    t_start = time.time()

    # Pre-load sensor defect mask if enabled and available
    raw_defect_mask = None
    if use_defect_mask:
        if defect_mask_path is None:
            calib_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../captures/calibration"))
            # 1. Check session nav.json for exact exposure/gain dark baseline
            nav_path = os.path.join(input_dir, "nav.json")
            if os.path.exists(nav_path):
                try:
                    with open(nav_path, "r") as f:
                        nav_data = json.load(f)
                    b_key = (nav_data.get("camera_settings") or {}).get("baseline_key")
                    if b_key:
                        candidate = os.path.join(calib_root, "baselines", f"{b_key}.npy")
                        if os.path.exists(candidate):
                            defect_mask_path = candidate
                except Exception:
                    pass
            # 2. Fallback to global defect_mask.npy
            if defect_mask_path is None:
                default_mask_path = os.path.join(calib_root, "defect_mask.npy")
                if os.path.exists(default_mask_path):
                    defect_mask_path = default_mask_path

        if defect_mask_path and os.path.exists(defect_mask_path):
            try:
                raw_defect_mask = np.load(defect_mask_path)
            except Exception as e:
                print(f"[!] Warning reading defect mask ({defect_mask_path}): {e}")

    pixel_weights = None
    defect_bool = None

    # Pass 1: Star centroid detection
    if verbose:
        print(f"[*] Step 1/3: Detecting star centroids across {len(valid_files)} frames...")

    write_progress(output_dir, "detecting", 1, 3, 0, len(valid_files), 0.0, f"Starting star detection (0/{len(valid_files)})")

    detections = []
    skipped_no_star = 0
    skipped_corrupt = 0
    frames_with_stripes = 0
    ref_shape = None
    channel_counts = {1: 0, 3: 0}

    for idx, filepath in enumerate(valid_files):
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if img is None:
            skipped_no_star += 1
            continue

        c = img.shape[2] if len(img.shape) == 3 else 1
        channel_counts[c] = channel_counts.get(c, 0) + 1

        if ref_shape is None:
            ref_shape = img.shape
            h0, w0 = ref_shape[:2]
            if raw_defect_mask is not None:
                if raw_defect_mask.shape[:2] == (h0, w0):
                    binary_defects = extract_defect_mask(raw_defect_mask)
                    pixel_weights = build_defect_weights(binary_defects, radius=defect_radius, soft_px=defect_soft_px)
                    defect_bool = (pixel_weights == 0.0)
                    raw_defects = int(np.sum(binary_defects > 0))
                    masked_px = int(np.sum(defect_bool))
                    pct_sensor = (masked_px / float(h0 * w0)) * 100.0
                    if verbose:
                        print(f"[*] Defect mask ({os.path.basename(defect_mask_path)}): {raw_defects} defects -> {masked_px} px masked after dilation (r={defect_radius}), {pct_sensor:.2f}% of sensor")
                    if pct_sensor > 5.0:
                        print(f"[!] Warning: Defect mask covers {pct_sensor:.2f}% of sensor (>5%). Consider a smaller --defect-radius or recalibration.")
                else:
                    print(f"[!] Warning: Defect mask shape {raw_defect_mask.shape} does not match frame shape {(h0, w0)}; ignoring defect mask.")

        # Detect MJPEG packet-loss stripes; exclude them from star search
        stripe_rects = detect_corruption_stripes(img) if filter_stripes else []
        stripe_mask = None
        if stripe_rects:
            stripe_mask = stripe_rects_to_mask(stripe_rects, img.shape[0], img.shape[1])
            frames_with_stripes += 1
            if float(np.mean(stripe_mask)) > max_stripe_fraction:
                # Frame is mostly garbage: drop it entirely
                skipped_corrupt += 1
                continue

        if stripe_mask is not None and defect_bool is not None and img.shape[:2] == defect_bool.shape:
            exclude_mask = stripe_mask | defect_bool
        elif stripe_mask is not None:
            exclude_mask = stripe_mask
        elif defect_bool is not None and img.shape[:2] == defect_bool.shape:
            exclude_mask = defect_bool
        else:
            exclude_mask = None

        res = detect_star_centroid(img, exclude_mask=exclude_mask)
        if res is not None:
            cx, cy, peak, bg = res
            detections.append({
                "path": filepath,
                "filename": os.path.basename(filepath),
                "cx": cx,
                "cy": cy,
                "peak": peak,
                "bg": bg,
                "stripes": stripe_rects,
            })
        else:
            skipped_no_star += 1

        if (idx + 1) % 10 == 0 or (idx + 1) == len(valid_files):
            step1_pct = ((idx + 1) / len(valid_files)) * 45.0
            write_progress(output_dir, "detecting", 1, 3, idx + 1, len(valid_files), step1_pct, f"Step 1/3: Star detection ({idx + 1}/{len(valid_files)})")

        if verbose and ((idx + 1) % 50 == 0 or (idx + 1) == len(valid_files)):
            pct = (idx + 1) / len(valid_files) * 100.0
            print(f"    Progress: {idx + 1}/{len(valid_files)} ({pct:.1f}%) | Detected: {len(detections)} | Skipped: {skipped_no_star} | Striped: {frames_with_stripes}")

    if not detections:
        raise RuntimeError("No stars detected in any of the provided frames.")

    if verbose:
        print(f"[*] Detection complete: {len(detections)} valid frames with guide star, {skipped_no_star} skipped (no star).")
        if filter_stripes:
            print(f"    Corruption stripes: {frames_with_stripes} frames affected (stripe pixels zero-weighted), {skipped_corrupt} frames dropped as mostly corrupt.")

    write_progress(output_dir, "geometry", 2, 3, 1, 1, 48.0, "Step 2/3: Computing canvas geometry")

    # Pass 2: Compute canvas geometry and shifts
    h, w = ref_shape[:2]
    # Set mosaic depth based on predominant frame format across detections
    num_channels = 3 if channel_counts.get(3, 0) > channel_counts.get(1, 0) else 1
    ref_cx = detections[0]["cx"]
    ref_cy = detections[0]["cy"]

    shifts = []
    for d in detections:
        tx = ref_cx - d["cx"]
        ty = ref_cy - d["cy"]
        shifts.append((tx, ty))

    if fov_mode == "full":
        txs = [s[0] for s in shifts]
        tys = [s[1] for s in shifts]
        min_tx, max_tx = min(txs), max(txs)
        min_ty, max_ty = min(tys), max(tys)

        canvas_w = int(np.ceil(w + max_tx - min_tx))
        canvas_h = int(np.ceil(h + max_ty - min_ty))
        offset_x = -min_tx
        offset_y = -min_ty
    else:  # 'first'
        canvas_w = w
        canvas_h = h
        offset_x = 0.0
        offset_y = 0.0

    if verbose:
        print(f"[*] Step 2/3: Canvas geometry initialized:")
        print(f"    Single Frame Resolution : {w} x {h}")
        print(f"    Mosaic Canvas Resolution: {canvas_w} x {canvas_h}")
        if fov_mode == "full":
            span_x = max(txs) - min(txs)
            span_y = max(tys) - min(tys)
            print(f"    Drift Span Covered      : X={span_x:.1f} px, Y={span_y:.1f} px")

    # Pass 3: Warping and accumulation with weighted feathering
    if verbose:
        print(f"[*] Step 3/3: Warping and accumulating frames...")

    acc_shape = (canvas_h, canvas_w, num_channels) if num_channels > 1 else (canvas_h, canvas_w)
    accumulator = np.zeros(acc_shape, dtype=np.float64)
    weight_map = np.zeros((canvas_h, canvas_w), dtype=np.float64)

    feather_mask = create_feather_mask(h, w, feather_px).astype(np.float64)
    sensor_weight = feather_mask * pixel_weights if pixel_weights is not None else feather_mask

    for idx, (d, (tx, ty)) in enumerate(zip(detections, shifts)):
        frame = cv2.imread(d["path"], cv2.IMREAD_UNCHANGED)
        if frame is None:
            continue

        # Harmonize channel format to match accumulator canvas
        if num_channels == 1:
            if frame.ndim == 3:
                if frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.ndim == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        frame_f = frame.astype(np.float64)

        # Zero-weight corruption stripes so they never contribute to the mosaic
        if d.get("stripes"):
            frame_sensor_weight = sensor_weight.copy()
            for x0, y0, x1, y1 in d["stripes"]:
                frame_sensor_weight[y0:y1, x0:x1] = 0.0
                # Also zero the pixel data so bilinear warping can't bleed
                # stripe values into neighbouring (valid) canvas pixels
                frame_f[y0:y1, x0:x1] = 0.0
        else:
            frame_sensor_weight = sensor_weight

        total_tx = tx + offset_x
        total_ty = ty + offset_y

        M = np.array([[1.0, 0.0, total_tx], [0.0, 1.0, total_ty]], dtype=np.float32)

        # Premultiply source frame by weights prior to warping so zero-weight defects/stripes
        # never bleed into neighboring canvas pixels during sub-pixel bilinear interpolation
        w_arr = frame_sensor_weight
        frame_w = frame_f * w_arr[:, :, None] if num_channels > 1 else frame_f * w_arr

        warped_frame_w = cv2.warpAffine(
            frame_w,
            M,
            (canvas_w, canvas_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )

        warped_weights = cv2.warpAffine(
            w_arr,
            M,
            (canvas_w, canvas_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )

        accumulator += warped_frame_w
        weight_map += warped_weights

        if (idx + 1) % 10 == 0 or (idx + 1) == len(detections):
            pct_step3 = 50.0 + ((idx + 1) / len(detections)) * 45.0
            write_progress(output_dir, "accumulating", 3, 3, idx + 1, len(detections), pct_step3, f"Step 3/3: Accumulating ({idx + 1}/{len(detections)})")

        if verbose and ((idx + 1) % 50 == 0 or (idx + 1) == len(detections)):
            pct = (idx + 1) / len(detections) * 100.0
            print(f"    Accumulated: {idx + 1}/{len(detections)} frames ({pct:.1f}%)")

    write_progress(output_dir, "rendering", 3, 3, len(detections), len(detections), 96.0, "Rendering 16-bit HDR outputs...")

    # Compute weighted average
    valid_mask = weight_map > 1e-4
    mosaic_linear = np.zeros_like(accumulator, dtype=np.float32)

    if num_channels > 1:
        mosaic_linear[valid_mask] = accumulator[valid_mask] / weight_map[valid_mask, None]
    else:
        mosaic_linear[valid_mask] = accumulator[valid_mask] / weight_map[valid_mask]

    # Save output renders
    output_files = {}

    if save_individual_renders:
        if verbose:
            print(f"[*] Rendering output files to {output_dir}...")

        # 1. 16-bit Linear TIFF (scaled to 16-bit uint16 dynamic range [0, 65535])
        # Preserves high precision float averages
        max_val = np.max(mosaic_linear[valid_mask]) if np.any(valid_mask) else 1.0
        # Determine source bit depth scale: if input was uint8, 255.0 is full scale; if uint16, 65535.0
        is_source_16bit = (max_val > 255.0)
        scale_to_16bit = 1.0 if is_source_16bit else 257.0  # 255 * 257 = 65535

        linear_16bit = np.clip(mosaic_linear * scale_to_16bit, 0, 65535).astype(np.uint16)
        p_linear_16 = os.path.join(output_dir, "mosaic_linear_16bit.tif")
        cv2.imwrite(p_linear_16, linear_16bit)
        output_files["linear_16bit_tif"] = p_linear_16

        # 2. Linear 8-bit standard render
        linear_8bit = np.clip(mosaic_linear, 0, 255).astype(np.uint8) if not is_source_16bit else np.clip(mosaic_linear / 257.0, 0, 255).astype(np.uint8)
        p_linear_8 = os.path.join(output_dir, "mosaic_linear_8bit.png")
        cv2.imwrite(p_linear_8, linear_8bit)
        output_files["linear_8bit_png"] = p_linear_8

        # 3. High Dynamic Range (HDR) Stretched Render (16-bit TIFF)
        hdr_stretched = stretch_hdr(
            mosaic_linear,
            valid_mask,
            method=stretch_method,
            stretch_factor=stretch_factor,
            black_percentile=black_percentile,
        )
        hdr_16bit = np.clip(hdr_stretched * 65535.0, 0, 65535).astype(np.uint16)
        p_hdr_16 = os.path.join(output_dir, f"mosaic_hdr_{stretch_method}_16bit.tif")
        cv2.imwrite(p_hdr_16, hdr_16bit)
        output_files["hdr_16bit_tif"] = p_hdr_16

        # 4. High Dynamic Range (HDR) Stretched Render (8-bit PNG for display)
        hdr_8bit = np.clip(hdr_stretched * 255.0, 0, 255).astype(np.uint8)
        p_hdr_8 = os.path.join(output_dir, f"mosaic_hdr_{stretch_method}_8bit.png")
        cv2.imwrite(p_hdr_8, hdr_8bit)
        output_files["hdr_8bit_png"] = p_hdr_8

        # 5. Coverage / Weight Map (heat map showing overlap count)
        w_norm = np.clip((weight_map / max(np.max(weight_map), 1.0)) * 255.0, 0, 255).astype(np.uint8)
        w_heatmap = cv2.applyColorMap(w_norm, cv2.COLORMAP_VIRIDIS)
        w_heatmap[~valid_mask] = 0
        p_weights = os.path.join(output_dir, "mosaic_coverage_weights.png")
        cv2.imwrite(p_weights, w_heatmap)
        output_files["coverage_weights_png"] = p_weights

    elapsed = time.time() - t_start

    write_progress(output_dir, "ready", 3, 3, len(detections), len(detections), 100.0, "Mosaic complete")

    if verbose:
        print(f"============================================================")
        print(f" SUCCESS: Mosaic generation completed in {elapsed:.1f} seconds!")
        print(f" Output files:")
        for k, v in output_files.items():
            print(f"   [{k}]: {v}")
        print(f"============================================================")

    return {
        "success": True,
        "elapsed_sec": elapsed,
        "canvas_size": (canvas_w, canvas_h),
        "total_files": len(all_files),
        "valid_files": len(valid_files),
        "aligned_frames": len(detections),
        "skipped_frames": skipped_no_star,
        "output_files": output_files,
        "mosaic_linear": mosaic_linear,
        "mosaic_hdr": hdr_stretched if save_individual_renders else None,
        "weight_map": weight_map,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="AstroCam Star Alignment & Full-FOV Mosaic Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Path to directory containing capture TIFF/image files",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Output directory (defaults to <input_dir>/mosaic_v1)",
    )
    parser.add_argument(
        "--fov",
        choices=["full", "first"],
        default="full",
        help="FOV mode: 'full' expands canvas to cover total drift; 'first' retains first frame FOV",
    )
    parser.add_argument(
        "--feather",
        type=int,
        default=12,
        help="Feathering margin in pixels for seamless edge blending",
    )
    parser.add_argument(
        "--stretch",
        choices=["asinh", "mtf", "log", "linear"],
        default="asinh",
        help="HDR tone curve method for rendering fine details",
    )
    parser.add_argument(
        "--stretch-factor",
        type=float,
        default=60.0,
        help="Strength factor for asinh / log stretch",
    )
    parser.add_argument(
        "--black-percentile",
        type=float,
        default=0.5,
        help="Lower percentile used as background black-point calibration",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process (useful for quick test runs)",
    )
    parser.add_argument(
        "--defect-mask",
        type=str,
        default=None,
        help="Path to defect_mask.npy containing sensor defect map to zero-weight hot pixels",
    )
    parser.add_argument(
        "--defect-radius",
        type=int,
        default=1,
        help="Dilation radius in pixels around each bad pixel to zero-weight neighbor spill",
    )
    parser.add_argument(
        "--defect-soft-px",
        type=int,
        default=0,
        help="Soft linear ramp distance in pixels outside the dilated defect zone",
    )
    parser.add_argument(
        "--no-defect-mask",
        action="store_true",
        help="Disable sensor defect mask weighting (useful for A/B comparison)",
    )
    parser.add_argument(
        "--no-stripe-filter",
        action="store_true",
        help="Disable detection/zero-weighting of MJPEG packet-loss stripes",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    combine_images(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        fov_mode=args.fov,
        feather_px=args.feather,
        stretch_method=args.stretch,
        stretch_factor=args.stretch_factor,
        black_percentile=args.black_percentile,
        max_frames=args.max_frames,
        defect_mask_path=args.defect_mask,
        defect_radius=args.defect_radius,
        defect_soft_px=args.defect_soft_px,
        use_defect_mask=not args.no_defect_mask,
        save_individual_renders=True,
        verbose=True,
        filter_stripes=not args.no_stripe_filter,
    )
