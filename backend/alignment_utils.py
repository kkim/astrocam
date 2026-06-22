import numpy as np
import cv2

def scale_down(img_in, s):
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

def detect_stars(img_gray, threshold=15, min_dist=5):
    """
    Finds local maxima in img_gray that are above threshold,
    enforces a minimum distance (NMS) to prevent duplicate detections,
    and refines their positions to sub-pixel centroids.
    """
    # Use dilation to find local maxima candidates
    dilated = cv2.dilate(img_gray, None)
    local_max = (img_gray == dilated) & (img_gray > threshold)
    y_coords, x_coords = np.where(local_max)
    
    # Sort local maxima by intensity to prioritize strongest stars in NMS
    intensities = img_gray[y_coords, x_coords]
    sort_idx = np.argsort(intensities)[::-1]
    
    # Limit candidates to the top 300 brightest to keep the NMS loop extremely fast
    sort_idx = sort_idx[:300]
    
    stars = []
    h, w = img_gray.shape
    for idx in sort_idx:
        x = x_coords[idx]
        y = y_coords[idx]
        if x < 5 or x >= w - 5 or y < 5 or y >= h - 5:
            continue
            
        # Non-Maximum Suppression check
        too_close = False
        for sx, sy in stars:
            if (x - sx)**2 + (y - sy)**2 < min_dist**2:
                too_close = True
                break
        if too_close:
            continue
            
        # Compute subpixel centroid in a 5x5 window (center of gravity)
        patch = img_gray[y-2:y+3, x-2:x+3].astype(np.float32)
        patch_sum = patch.sum()
        if patch_sum > 0:
            xx, yy = np.meshgrid(np.arange(x-2, x+3), np.arange(y-2, y+3))
            cx = float((xx * patch).sum() / patch_sum)
            cy = float((yy * patch).sum() / patch_sum)
            stars.append((cx, cy))
    return stars

def compute_star_descriptors(stars, N=3):
    """
    For each star, find its N nearest neighbors, compute their relative
    offsets, and normalize by rotation to make the descriptor rotation-invariant.
    """
    descriptors = []
    valid_stars = []
    num_stars = len(stars)
    if num_stars <= N:
        return np.empty((0, 2 * N), dtype=np.float32), []
        
    stars_arr = np.array(stars) # Shape: (K, 2)
    for i, p in enumerate(stars):
        diffs = stars_arr - p
        dists_sq = (diffs**2).sum(axis=1)
        sorted_indices = np.argsort(dists_sq)
        
        # Take the top N nearest neighbors (excluding the star itself at index 0)
        neighbor_idx = sorted_indices[1:N+1]
        if len(neighbor_idx) < N:
            continue
            
        neighbor_diffs = diffs[neighbor_idx]
        
        # Rotation normalization based on the nearest neighbor
        dx1, dy1 = neighbor_diffs[0]
        theta = np.arctan2(dy1, dx1)
        c, s = np.cos(-theta), np.sin(-theta)
        
        rotated_diffs = []
        for dx, dy in neighbor_diffs:
            rx = dx * c - dy * s
            ry = dx * s + dy * c
            rotated_diffs.append((rx, ry))
            
        feat = np.array(rotated_diffs).flatten()
        descriptors.append(feat)
        valid_stars.append(p)
        
    return np.array(descriptors, dtype=np.float32), valid_stars

def match_star_descriptors(des_ref, des_src, stars_ref, stars_src, N=3, max_dist=5.0):
    """
    Matches source descriptors with reference descriptors using L2 norm
    and filters ambiguous matches via Lowe's ratio test.
    """
    matches = []
    if len(des_ref) == 0 or len(des_src) == 0:
        return matches
        
    for i, d_src in enumerate(des_src):
        # Compute L2 distances to all reference descriptors
        dists = np.linalg.norm(des_ref - d_src, axis=1)
        best_idx = np.argmin(dists)
        if dists[best_idx] < max_dist:
            # Lowe's ratio check
            sorted_dists = np.sort(dists)
            if len(sorted_dists) < 2 or sorted_dists[0] < 0.75 * sorted_dists[1]:
                matches.append((stars_src[i], stars_ref[best_idx]))
    return matches

def align_images(img_ref, img_src, nfeatures=100, translation_only=False, s=4, return_metrics=False):
    """
    Finds the affine transform M that maps img_src to img_ref using
    our custom Star Neighborhood Descriptors.
    Returns a 2x3 matrix M.
    """
    # Scale down images using box-sum binning
    ref_scaled = scale_down(img_ref, s)
    src_scaled = scale_down(img_src, s)

    if len(ref_scaled.shape) == 3:
        gray_ref = cv2.cvtColor(ref_scaled, cv2.COLOR_BGR2GRAY)
    else:
        gray_ref = ref_scaled
        
    if len(src_scaled.shape) == 3:
        gray_src = cv2.cvtColor(src_scaled, cv2.COLOR_BGR2GRAY)
    else:
        gray_src = src_scaled

    # Detect stars
    stars_ref = detect_stars(gray_ref)
    stars_src = detect_stars(gray_src)
    
    # Compute descriptors (using 3 nearest neighbors)
    N_neighbors = 3
    des_ref, valid_ref = compute_star_descriptors(stars_ref, N=N_neighbors)
    des_src, valid_src = compute_star_descriptors(stars_src, N=N_neighbors)
    
    # Match descriptors
    matches = match_star_descriptors(des_ref, des_src, valid_ref, valid_src, N=N_neighbors)
    
    if len(matches) < 3:
        # Fall back to identity if not enough matches
        M = np.eye(2, 3, dtype=np.float32)
        if return_metrics:
            return M, {"inlier_ratio": 0.0}
        return M
        
    src_pts = np.float32([m[0] for m in matches]).reshape(-1, 1, 2)
    ref_pts = np.float32([m[1] for m in matches]).reshape(-1, 1, 2)

    if translation_only:
        # Run RANSAC to find consensus translation
        M, mask = cv2.estimateAffinePartial2D(src_pts, ref_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        inliers = int(np.sum(mask)) if mask is not None else 0
        inlier_ratio = float(inliers / len(matches)) if len(matches) > 0 else 0.0

        if M is not None and mask is not None and np.sum(mask) >= 3:
            inliers_src = src_pts[mask.ravel() == 1]
            inliers_ref = ref_pts[mask.ravel() == 1]
            diffs = inliers_ref - inliers_src
        else:
            diffs = ref_pts - src_pts

        dxs = diffs[:, 0, 0]
        dys = diffs[:, 0, 1]
        dx = float(np.median(dxs))
        dy = float(np.median(dys))
        
        # Scale up the translation components back to the original image coordinate space
        dx *= s
        dy *= s
        
        # Sanity check on translation
        if abs(dx) > 100.0 or abs(dy) > 100.0:
            M_final = np.eye(2, 3, dtype=np.float32)
            if return_metrics:
                return M_final, {"inlier_ratio": 0.0}
            return M_final
            
        M_final = np.float32([[1, 0, dx], [0, 1, dy]])
        if return_metrics:
            return M_final, {"inlier_ratio": inlier_ratio}
        return M_final

    # estimateAffinePartial2D finds Translation, Rotation, and Scale.
    M, mask = cv2.estimateAffinePartial2D(src_pts, ref_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    inliers = int(np.sum(mask)) if mask is not None else 0
    inlier_ratio = float(inliers / len(matches)) if len(matches) > 0 else 0.0
    
    if M is None:
        M_final = np.eye(2, 3, dtype=np.float32)
        if return_metrics:
            return M_final, {"inlier_ratio": 0.0}
        return M_final
        
    # Scale up the translation components back to the original image coordinate space
    M[0, 2] *= s
    M[1, 2] *= s
    
    # Sanity checks on the estimated affine matrix.
    scale = np.sqrt(M[0, 0]**2 + M[1, 0]**2)
    if abs(scale - 1.0) > 0.05:
        M_final = np.eye(2, 3, dtype=np.float32)
        if return_metrics:
            return M_final, {"inlier_ratio": 0.0}
        return M_final
        
    angle = np.arctan2(M[1, 0], M[0, 0]) * 180.0 / np.pi
    if abs(angle) > 5.0:
        M_final = np.eye(2, 3, dtype=np.float32)
        if return_metrics:
            return M_final, {"inlier_ratio": 0.0}
        return M_final
        
    dx = M[0, 2]
    dy = M[1, 2]
    if abs(dx) > 100.0 or abs(dy) > 100.0:
        M_final = np.eye(2, 3, dtype=np.float32)
        if return_metrics:
            return M_final, {"inlier_ratio": 0.0}
        return M_final
        
    M_final = M.astype(np.float32)
    if return_metrics:
        return M_final, {"inlier_ratio": inlier_ratio}
    return M_final

def transform_image(img, M, target_shape=None):
    """Applies affine transform M to img."""
    if target_shape is None:
        h, w = img.shape[:2]
    else:
        h, w = target_shape[:2]
        
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

def transform(img, M, target_shape=None):
    """Applies affine transform M to img (alias for transform_image)."""
    return transform_image(img, M, target_shape)

def compose_transforms(T1, T2):
    """Composes two 2x3 affine transformation matrices T1 and T2."""
    T1_3x3 = np.vstack([T1, [0, 0, 1]])
    T2_3x3 = np.vstack([T2, [0, 0, 1]])
    out_3x3 = np.dot(T1_3x3, T2_3x3)
    return out_3x3[:2, :].astype(np.float32)

def accumulate_panorama_frame(sum_img, sum_wgt, offset_img0, T_prev_to_0, img_prev, img, translation_only=True):
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

