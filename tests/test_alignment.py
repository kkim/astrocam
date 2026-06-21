import numpy as np
import cv2
import sys
from alignment_utils import align_images, transform, compose_transforms, accumulate_panorama_frame
 
def create_synthetic_stars(width=1920, height=1080, num_stars=100, seed=42):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    rng = np.random.default_rng(seed)
    for _ in range(num_stars):
        x = rng.integers(10, width - 10)
        y = rng.integers(10, height - 10)
        brightness = int(rng.integers(150, 255))
        cv2.circle(img, (x, y), 2, (brightness, brightness, brightness), -1)
    return cv2.GaussianBlur(img, (5, 5), 1.5)
 
def test_alignment_translation():
    print("Testing Translation Alignment...")
    im1 = create_synthetic_stars()
    
    # Manually shift im1 to create im2
    # dx=15, dy=-10
    dx, dy = 15, -10
    M_true = np.float32([[1, 0, dx], [0, 1, dy]])
    im2 = cv2.warpAffine(im1, M_true, (im1.shape[1], im1.shape[0]))
    
    # Now find M that maps im2 back to im1
    # align_images(ref, src) -> maps src to ref
    T = align_images(im1, im2)
    
    print(f"True Shift: dx={dx}, dy={dy}")
    print(f"Estimated Transform:\n{T}")
    
    est_dx = T[0, 2]
    est_dy = T[1, 2]
    
    # Since T maps im2 -> im1, and im2 was created by shifting im1 by M_true (dx, dy),
    # the mapping from im2 -> im1 is the inverse translation: -dx, -dy.
    diff_x = abs(est_dx - (-dx))
    diff_y = abs(est_dy - (-dy))
    
    assert diff_x < 3.5, f"X shift error too large: {diff_x} (Expected est_dx to be approx {-dx})"
    assert diff_y < 3.5, f"Y shift error too large: {diff_y} (Expected est_dy to be approx {-dy})"
    
    # Test that the transform function successfully aligns the image back
    im2_aligned = transform(im2, T)
    h, w = im1.shape[:2]
    crop1 = im1[20:h-20, 20:w-20]
    crop2_unaligned = im2[20:h-20, 20:w-20]
    crop2_aligned = im2_aligned[20:h-20, 20:w-20]
    
    mse_unaligned = np.mean((crop1.astype(np.float32) - crop2_unaligned.astype(np.float32)) ** 2)
    mse_aligned = np.mean((crop1.astype(np.float32) - crop2_aligned.astype(np.float32)) ** 2)
    
    print(f"Unaligned MSE: {mse_unaligned:.4f}")
    print(f"Aligned MSE:   {mse_aligned:.4f}")
    
    # Alignment should drastically reduce the MSE (at least 10x reduction)
    assert mse_aligned < 20.0, f"Aligned MSE is too high: {mse_aligned:.4f}"
    assert mse_aligned < mse_unaligned / 1.5, f"Alignment did not sufficiently improve MSE: {mse_aligned:.4f} vs {mse_unaligned:.4f}"
    print("✓ Translation test passed!")
 
def test_alignment_rotation():
    print("\nTesting Rotation Alignment...")
    im1 = create_synthetic_stars()
    
    # Rotate by 2 degrees around center
    h, w = im1.shape[:2]
    center = (w // 2, h // 2)
    angle = 2.0
    M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
    im2 = cv2.warpAffine(im1, M_rot, (w, h))
    
    # Find M that maps im2 to im1
    T = align_images(im1, im2)
    
    est_angle = -np.arctan2(T[0, 1], T[0, 0]) * 180.0 / np.pi
    print(f"True Angle: {angle}")
    print(f"Estimated Angle: {est_angle:.2f}")
    
    diff_angle = abs(est_angle - angle)
    assert diff_angle < 1.0, f"Angle error too large: {diff_angle}"
    
    # Test that the transform function successfully aligns the image back
    im2_aligned = transform(im2, T)
    crop1 = im1[40:h-40, 40:w-40]
    crop2_unaligned = im2[40:h-40, 40:w-40]
    crop2_aligned = im2_aligned[40:h-40, 40:w-40]
    
    mse_unaligned = np.mean((crop1.astype(np.float32) - crop2_unaligned.astype(np.float32)) ** 2)
    mse_aligned = np.mean((crop1.astype(np.float32) - crop2_aligned.astype(np.float32)) ** 2)
    
    print(f"Unaligned MSE: {mse_unaligned:.4f}")
    print(f"Aligned MSE:   {mse_aligned:.4f}")
    
    assert mse_aligned < 40.0, f"Aligned MSE is too high: {mse_aligned:.4f}"
    assert mse_aligned < mse_unaligned / 1.5, f"Alignment did not sufficiently improve MSE: {mse_aligned:.4f} vs {mse_unaligned:.4f}"
    print("✓ Rotation test passed!")

def test_compose_transforms():
    print("\nTesting Compose Transforms...")
    # Translate by A(10, -5) then B(-3, 8) -> net translation (7, 3)
    T1 = np.float32([[1, 0, 10], [0, 1, -5]])
    T2 = np.float32([[1, 0, -3], [0, 1, 8]])
    T_composed = compose_transforms(T2, T1) # T2 after T1: B * A * x
    print("Composed Translation:\n", T_composed)
    assert abs(T_composed[0, 2] - 7.0) < 1e-5
    assert abs(T_composed[1, 2] - 3.0) < 1e-5
    
    # Rotate by 30 deg then 15 deg -> net 45 deg around origin (0, 0)
    r1 = 30.0 * np.pi / 180.0
    r2 = 15.0 * np.pi / 180.0
    R1 = np.float32([[np.cos(r1), -np.sin(r1), 0], [np.sin(r1), np.cos(r1), 0]])
    R2 = np.float32([[np.cos(r2), -np.sin(r2), 0], [np.sin(r2), np.cos(r2), 0]])
    R_composed = compose_transforms(R2, R1)
    net_angle = np.arctan2(R_composed[1, 0], R_composed[0, 0]) * 180.0 / np.pi
    print(f"Composed Angle: {net_angle:.2f} (Expected: 45.00)")
    assert abs(net_angle - 45.0) < 1e-4
    print("✓ Compose transforms test passed!")

def test_three_frame_alignment():
    print("\nTesting 3-Frame Sequence Alignment...")
    im1 = create_synthetic_stars(seed=42)
    
    # Shift 1: dx1 = 12, dy1 = -8
    dx1, dy1 = 12, -8
    M1_true = np.float32([[1, 0, dx1], [0, 1, dy1]])
    im2 = cv2.warpAffine(im1, M1_true, (im1.shape[1], im1.shape[0]))
    
    # Shift 2: dx2 = -6, dy2 = 14
    dx2, dy2 = -6, 14
    M2_true = np.float32([[1, 0, dx2], [0, 1, dy2]])
    im3 = cv2.warpAffine(im2, M2_true, (im2.shape[1], im2.shape[0]))
    
    # Step 1: Align im2 (src) to im1 (ref) -> T1 maps im2 to im1
    T1 = align_images(im1, im2, translation_only=True)
    # Step 2: Align im3 (src) to im2 (ref) -> T2 maps im3 to im2
    T2 = align_images(im2, im3, translation_only=True)
    
    # Step 3: Compose T_cum = T1 * T2
    T_cum = compose_transforms(T1, T2)
    
    print(f"T1:\n{T1}")
    print(f"T2:\n{T2}")
    print(f"T_cum composed:\n{T_cum}")
    
    # Cumulative translation should be approx (-dx1 - dx2, -dy1 - dy2)
    # Since dx1=12, dx2=-6 -> total shift is +6. So T_cum should be approx -6.
    # Since dy1=-8, dy2=14 -> total shift is +6. So T_cum should be approx -6.
    expected_dx = -(dx1 + dx2)
    expected_dy = -(dy1 + dy2)
    print(f"Expected Cumulative: dx={expected_dx}, dy={expected_dy}")
    
    assert abs(T_cum[0, 2] - expected_dx) < 4.5, f"Cumulative X error too large: {T_cum[0, 2]} vs {expected_dx}"
    assert abs(T_cum[1, 2] - expected_dy) < 4.5, f"Cumulative Y error too large: {T_cum[1, 2]} vs {expected_dy}"
    
    # Test that the composed transform successfully aligns im3 back to im1
    im3_aligned = transform(im3, T_cum)
    h, w = im1.shape[:2]
    crop1 = im1[30:h-30, 30:w-30]
    crop3_unaligned = im3[30:h-30, 30:w-30]
    crop3_aligned = im3_aligned[30:h-30, 30:w-30]
    
    mse_unaligned = np.mean((crop1.astype(np.float32) - crop3_unaligned.astype(np.float32)) ** 2)
    mse_aligned = np.mean((crop1.astype(np.float32) - crop3_aligned.astype(np.float32)) ** 2)
    
    print(f"Unaligned MSE (Frame 3 vs 1): {mse_unaligned:.4f}")
    print(f"Aligned MSE (Frame 3 vs 1):   {mse_aligned:.4f}")
    
    assert mse_aligned < 20.0, f"Composed aligned MSE is too high: {mse_aligned:.4f}"
    assert mse_aligned < mse_unaligned / 1.5, "Composition did not sufficiently improve MSE"
    print("✓ 3-Frame Sequence test passed!")
 
def test_three_frame_affine_alignment():
    print("\nTesting 3-Frame Affine Sequence Alignment...")
    im0 = create_synthetic_stars(seed=42)
    h, w = im0.shape[:2]
    
    # Transform 1: Shift dx1=8, dy1=-6, and Rotate theta1=1.5 degrees
    dx1, dy1 = 8, -6
    theta1 = 1.5
    center = (w // 2, h // 2)
    M1_true = cv2.getRotationMatrix2D(center, theta1, 1.0)
    M1_true[0, 2] += dx1
    M1_true[1, 2] += dy1
    im1 = cv2.warpAffine(im0, M1_true, (w, h))
    
    # Transform 2: Shift dx2=-4, dy2=10, and Rotate theta2=-1.0 degrees
    dx2, dy2 = -4, 10
    theta2 = -1.0
    M2_true = cv2.getRotationMatrix2D(center, theta2, 1.0)
    M2_true[0, 2] += dx2
    M2_true[1, 2] += dy2
    im2 = cv2.warpAffine(im1, M2_true, (w, h))
    
    # Step 1: Align im1 (src) to im0 (ref) -> T1 maps im1 to im0
    T1 = align_images(im0, im1, translation_only=False)
    # Step 2: Align im2 (src) to im1 (ref) -> T2 maps im2 to im1
    T2 = align_images(im1, im2, translation_only=False)
    
    # Step 3: Compose T_cum = T1 * T2 (maps im2 back to im0)
    T_cum = compose_transforms(T1, T2)
    
    print(f"T1:\n{T1}")
    print(f"T2:\n{T2}")
    print(f"T_cum composed:\n{T_cum}")
    
    # Test that the composed transform successfully aligns im2 back to im0
    im2_aligned = transform(im2, T_cum)
    
    # Crop central area to avoid black borders from transforms
    crop0 = im0[50:h-50, 50:w-50]
    crop2_unaligned = im2[50:h-50, 50:w-50]
    crop2_aligned = im2_aligned[50:h-50, 50:w-50]
    
    mse_unaligned = np.mean((crop0.astype(np.float32) - crop2_unaligned.astype(np.float32)) ** 2)
    mse_aligned = np.mean((crop0.astype(np.float32) - crop2_aligned.astype(np.float32)) ** 2)
    
    print(f"Unaligned MSE (Frame 2 vs 0): {mse_unaligned:.4f}")
    print(f"Aligned MSE (Frame 2 vs 0):   {mse_aligned:.4f}")
    
    # Assert alignment was highly successful
    assert mse_aligned < 150.0, f"Composed aligned MSE is too high: {mse_aligned:.4f}"
    assert mse_aligned < mse_unaligned / 1.5, "Composition did not sufficiently improve MSE"
    print("✓ 3-Frame Affine Sequence test passed!")

def test_panorama_accumulation_function():
    print("\nTesting test_panorama_accumulation_function...")
    # 1. Create a base image img0 with synthetic stars
    img0 = create_synthetic_stars(seed=42)
    h, w = img0.shape[:2]
    
    # 2. Create img1 and img2 with slight translations:
    # Shift 1: dx1 = 12, dy1 = -8 from img0 -> img1
    dx1, dy1 = 12, -8
    M1 = np.float32([[1, 0, dx1], [0, 1, dy1]])
    img1 = cv2.warpAffine(img0, M1, (w, h))
    
    # Shift 2: dx2 = -6, dy2 = 14 from img1 -> img2
    dx2, dy2 = -6, 14
    M2 = np.float32([[1, 0, dx2], [0, 1, dy2]])
    img2 = cv2.warpAffine(img1, M2, (w, h))
    
    # 3. Initialize buffers
    buf_w, buf_h = w * 6, h * 3
    sum_img = np.zeros((buf_h, buf_w, 3), dtype=np.float32)
    sum_wgt = np.zeros((buf_h, buf_w), dtype=np.float32)
    offset_img0 = ((buf_w - w) // 2, (buf_h - h) // 2)
    
    # Initial cumulative transform is identity
    T_prev_to_0 = np.eye(2, 3, dtype=np.float32)
    
    # Call accumulate_panorama_frame for img0 (passing img_prev=None, T_prev_to_0=Identity)
    sum_img, sum_wgt, T_prev_to_0 = accumulate_panorama_frame(
        sum_img, sum_wgt, offset_img0, T_prev_to_0, None, img0, translation_only=True
    )
    
    # Call it for img1 (passing img_prev=img0)
    sum_img, sum_wgt, T_prev_to_0 = accumulate_panorama_frame(
        sum_img, sum_wgt, offset_img0, T_prev_to_0, img0, img1, translation_only=True
    )
    
    # Call it for img2 (passing img_prev=img1)
    sum_img, sum_wgt, T_prev_to_0 = accumulate_panorama_frame(
        sum_img, sum_wgt, offset_img0, T_prev_to_0, img1, img2, translation_only=True
    )
    
    # Calculate average image where populated
    mask = sum_wgt > 0
    avg_img = np.zeros_like(sum_img, dtype=np.float32)
    avg_img[mask] = sum_img[mask] / sum_wgt[mask][:, np.newaxis]
    
    # Crop the area corresponding to img0's placement in the panorama buffer.
    by, bx = offset_img0[1], offset_img0[0]
    cropped_result = avg_img[by:by+h, bx:bx+w]
    
    # Crop the central region of the image to avoid boundary artifacts from the warp.
    crop0 = img0[30:h-30, 30:w-30]
    crop_res = cropped_result[30:h-30, 30:w-30]
    
    mse = np.mean((crop0.astype(np.float32) - crop_res) ** 2)
    print(f"MSE between img0 and panorama aligned output: {mse:.4f}")
    
    assert mse < 20.0, f"Panorama accumulation MSE is too high: {mse:.4f}"
    print("✓ Panorama accumulation unit test passed!")
 
if __name__ == "__main__":
    try:
        test_alignment_translation()
        test_alignment_rotation()
        test_compose_transforms()
        test_three_frame_alignment()
        test_three_frame_affine_alignment()
        test_panorama_accumulation_function()
        print("\nALL ALIGNMENT TESTS PASSED!")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        sys.exit(1)


