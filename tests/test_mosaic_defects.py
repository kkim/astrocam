import os
import sys
import cv2
import numpy as np
import pytest

# Add bin/ to sys.path so mosaic_stars can be imported directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../bin")))

from mosaic_stars import build_defect_weights, combine_images


def _add_gaussian_star(img: np.ndarray, cx: float, cy: float, peak: float = 15000.0, sigma: float = 2.2):
    """Adds a synthetic sub-pixel 2D Gaussian star to a float image."""
    h, w = img.shape[:2]
    yy, xx = np.indices((h, w), dtype=np.float64)
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    star = peak * np.exp(-r2 / (2.0 * sigma ** 2))
    if img.ndim == 3:
        img += star[:, :, None]
    else:
        img += star


def _write_synthetic_session(
    frames_dir,
    shifts,
    h: int = 120,
    w: int = 120,
    bg: float = 100.0,
    star_base=(60.0, 60.0),
    hot_pixel=(25, 30),
    hot_val: float = 4000.0,
    halo_val: float = 0.0,
):
    """
    Creates synthetic 16-bit grayscale TIFF frames where the star moves by `shifts`
    while the sensor defect at `hot_pixel` (hy, hx) remains fixed in sensor coordinates.
    """
    os.makedirs(frames_dir, exist_ok=True)
    hy, hx = hot_pixel
    for idx, (dx, dy) in enumerate(shifts):
        frame = np.full((h, w), bg, dtype=np.float64)
        _add_gaussian_star(frame, star_base[0] + dx, star_base[1] + dy)

        if halo_val > 0.0:
            for dy_h in (-1, 0, 1):
                for dx_h in (-1, 0, 1):
                    if dy_h == 0 and dx_h == 0:
                        continue
                    ny, nx = hy + dy_h, hx + dx_h
                    if 0 <= ny < h and 0 <= nx < w:
                        frame[ny, nx] = halo_val

        if hot_val > 0.0:
            frame[hy, hx] = hot_val

        frame_u16 = np.clip(np.round(frame), 0, 65535).astype(np.uint16)
        cv2.imwrite(os.path.join(frames_dir, f"frame_{idx:03d}.tif"), frame_u16)


def test_build_defect_weights_dilation_and_soft_ramp():
    mask = np.zeros((31, 31), dtype=np.uint8)
    mask[15, 15] = 1

    # Hard dilation with radius=1
    w_hard = build_defect_weights(mask, radius=1, soft_px=0)
    assert w_hard.shape == (31, 31)
    assert w_hard.dtype == np.float64
    assert w_hard[15, 15] == 0.0
    assert w_hard[14, 15] == 0.0
    assert w_hard[15, 16] == 0.0
    assert w_hard[10, 10] == 1.0

    # Soft ramp over 4 pixels outside radius=1
    w_soft = build_defect_weights(mask, radius=1, soft_px=4)
    assert w_soft[15, 15] == 0.0
    assert w_soft[15, 16] == 0.0  # within radius 1
    assert 0.0 < w_soft[15, 17] < 0.5  # 1 px outside dilated zone -> ~0.25
    assert 0.4 < w_soft[15, 18] < 0.6  # 2 px outside -> ~0.50
    assert w_soft[15, 22] == 1.0  # well outside soft zone


def test_subpixel_interpolation_leak(tmp_path):
    """
    1. Leak test (P1): synthetic flat frames (100 ADU) with one hot pixel (4000 ADU)
    marked in the mask, shifted by sub-pixel amounts. Assert max deviation from 100
    in the background region around the hot pixel is < 1 ADU.
    """
    h, w = 120, 120
    hot_y, hot_x = 25, 30
    shifts = [(0.0, 0.0), (0.3, 0.7), (1.4, 2.2), (2.7, 3.6), (4.1, 5.3)]

    frames_dir = tmp_path / "frames"
    out_dir = tmp_path / "out"
    mask_path = tmp_path / "mask.npy"

    _write_synthetic_session(
        str(frames_dir),
        shifts=shifts,
        h=h,
        w=w,
        bg=100.0,
        star_base=(75.0, 75.0),
        hot_pixel=(hot_y, hot_x),
        hot_val=4000.0,
        halo_val=0.0,
    )

    d_mask = np.zeros((h, w), dtype=np.uint8)
    d_mask[hot_y, hot_x] = 1
    np.save(str(mask_path), d_mask)

    res = combine_images(
        str(frames_dir),
        output_dir=str(out_dir),
        fov_mode="first",
        defect_mask_path=str(mask_path),
        defect_radius=0,  # Even with radius=0, premultiplied warp must prevent sub-pixel leak!
        save_individual_renders=False,
        verbose=False,
        filter_stripes=False,
        feather_px=0,
    )

    mosaic = res["mosaic_linear"]
    # Inspect a 20x20 patch around the hot pixel trajectory (far from the star at 75,75)
    patch = mosaic[15:35, 20:40]
    max_dev = float(np.max(np.abs(patch - 100.0)))
    assert max_dev < 1.0, f"Sub-pixel interpolation leaked hot pixel into mosaic (max_dev={max_dev:.3f})"


def test_vicinity_halo_suppression(tmp_path):
    """
    2. Vicinity test (P2): add a +/-1 px halo (400 ADU) around the hot pixel (4000 ADU).
    Don't include the halo in the mask.
    With defect_radius=1, assert the halo does not appear (< 1 ADU deviation).
    With defect_radius=0, assert the halo DOES appear (> 10 ADU deviation).
    """
    h, w = 120, 120
    hot_y, hot_x = 25, 30
    shifts = [(0.0, 0.0), (3.3, 1.7), (6.6, 3.4), (9.9, 5.1)]

    frames_dir = tmp_path / "frames"
    mask_path = tmp_path / "mask.npy"

    _write_synthetic_session(
        str(frames_dir),
        shifts=shifts,
        h=h,
        w=w,
        bg=100.0,
        star_base=(75.0, 75.0),
        hot_pixel=(hot_y, hot_x),
        hot_val=4000.0,
        halo_val=400.0,
    )

    d_mask = np.zeros((h, w), dtype=np.uint8)
    d_mask[hot_y, hot_x] = 1
    np.save(str(mask_path), d_mask)

    # Case A: radius=1 (default) suppresses the +/-1 px halo
    res_r1 = combine_images(
        str(frames_dir),
        output_dir=str(tmp_path / "out_r1"),
        fov_mode="first",
        defect_mask_path=str(mask_path),
        defect_radius=1,
        save_individual_renders=False,
        verbose=False,
        filter_stripes=False,
        feather_px=0,
    )
    patch_r1 = res_r1["mosaic_linear"][15:35, 15:40]
    dev_r1 = float(np.max(np.abs(patch_r1 - 100.0)))
    assert dev_r1 < 1.0, f"Expected halo to be suppressed with defect_radius=1, got max_dev={dev_r1:.2f}"

    # Case B: radius=0 leaves the unmasked halo visible
    res_r0 = combine_images(
        str(frames_dir),
        output_dir=str(tmp_path / "out_r0"),
        fov_mode="first",
        defect_mask_path=str(mask_path),
        defect_radius=0,
        save_individual_renders=False,
        verbose=False,
        filter_stripes=False,
        feather_px=0,
    )
    patch_r0 = res_r0["mosaic_linear"][15:35, 15:40]
    dev_r0 = float(np.max(np.abs(patch_r0 - 100.0)))
    assert dev_r0 > 20.0, f"Expected halo to remain visible with defect_radius=0, got max_dev={dev_r0:.2f}"


def test_coverage_no_holes_across_shifted_frames(tmp_path):
    """
    3. Coverage test: with shifts of several pixels between frames, the canvas has
    no holes (weight_map > 0) over the overlap region, even with the dilated mask.
    """
    h, w = 120, 120
    hot_y, hot_x = 30, 30
    shifts = [(0.0, 0.0), (5.0, 2.0), (10.0, 4.0), (15.0, 6.0)]

    frames_dir = tmp_path / "frames"
    mask_path = tmp_path / "mask.npy"

    _write_synthetic_session(
        str(frames_dir),
        shifts=shifts,
        h=h,
        w=w,
        bg=100.0,
        star_base=(70.0, 70.0),
        hot_pixel=(hot_y, hot_x),
        hot_val=4000.0,
        halo_val=400.0,
    )

    d_mask = np.zeros((h, w), dtype=np.uint8)
    d_mask[hot_y, hot_x] = 1
    np.save(str(mask_path), d_mask)

    res = combine_images(
        str(frames_dir),
        output_dir=str(tmp_path / "out"),
        fov_mode="first",
        defect_mask_path=str(mask_path),
        defect_radius=1,
        save_individual_renders=False,
        verbose=False,
        filter_stripes=False,
        feather_px=0,
    )

    weight_map = res["weight_map"]
    # Central overlap region covered by all frames must have strictly positive weight everywhere
    overlap = weight_map[20:90, 20:90]
    assert np.all(overlap > 0.5), f"Found unexpected zero-weight hole in overlap region (min={np.min(overlap)})"


def test_shape_mismatch_warning(tmp_path, capsys):
    """
    4. Shape-mismatch test (P3): a mask of the wrong shape is ignored and a warning is printed.
    """
    h, w = 100, 100
    frames_dir = tmp_path / "frames"
    mask_path = tmp_path / "wrong_mask.npy"

    _write_synthetic_session(
        str(frames_dir),
        shifts=[(0.0, 0.0), (2.0, 2.0)],
        h=h,
        w=w,
        bg=100.0,
        star_base=(50.0, 50.0),
        hot_pixel=(20, 20),
        hot_val=0.0,
    )

    # Save a mask with mismatched shape (50, 60) instead of (100, 100)
    wrong_mask = np.zeros((50, 60), dtype=np.uint8)
    wrong_mask[10, 10] = 1
    np.save(str(mask_path), wrong_mask)

    res = combine_images(
        str(frames_dir),
        output_dir=str(tmp_path / "out"),
        fov_mode="first",
        defect_mask_path=str(mask_path),
        save_individual_renders=False,
        verbose=False,
        filter_stripes=False,
    )

    assert res["success"] is True
    captured = capsys.readouterr()
    assert "does not match frame shape" in captured.out
    assert "(50, 60)" in captured.out
    assert "(100, 100)" in captured.out


def test_star_detection_excludes_defects(tmp_path):
    """
    5. Star detection exclusion test (P4): if a stationary hot defect cluster is brighter
    than the true moving star, excluding the dilated defect mask allows star detection
    to lock onto the true moving star.
    """
    h, w = 120, 120
    hot_y, hot_x = 25, 25
    # True star moves by 6 px per frame; hot pixel cluster stays fixed at (25, 25)
    shifts = [(0.0, 0.0), (6.0, 0.0), (12.0, 0.0)]

    frames_dir = tmp_path / "frames"
    mask_path = tmp_path / "mask.npy"

    # Give the hot pixel + halo a higher peak (30000) than the star (10000)
    _write_synthetic_session(
        str(frames_dir),
        shifts=shifts,
        h=h,
        w=w,
        bg=100.0,
        star_base=(60.0, 60.0),
        hot_pixel=(hot_y, hot_x),
        hot_val=30000.0,
        halo_val=25000.0,
    )

    d_mask = np.zeros((h, w), dtype=np.uint8)
    d_mask[hot_y, hot_x] = 1
    np.save(str(mask_path), d_mask)

    res = combine_images(
        str(frames_dir),
        output_dir=str(tmp_path / "out"),
        fov_mode="full",
        defect_mask_path=str(mask_path),
        defect_radius=1,
        save_individual_renders=False,
        verbose=False,
        filter_stripes=False,
        feather_px=0,
    )

    # Because the true star moved by 12 px in X, the full-FOV canvas width should expand by ~12 px
    canvas_w, canvas_h = res["canvas_size"]
    assert canvas_w >= w + 10, f"Expected canvas to expand with moving star (~132 px), got {canvas_w}"


def test_constellation_rank_swap_and_static_warm_pixel(tmp_path):
    """
    6. Multi-star constellation + temporal static-peak rejection test:
    Two stars separated by a constant offset (+32, +24) swap #1 brightness rank across
    frames, while an unmasked stationary warm cluster sits at (140, 35) and outshines
    both stars during a dim frame. Verifies that constellation mode rejects the static
    warm cluster, resolves all rank swaps without 40px jumps, and aligns the sequence.
    """
    h, w = 180, 180
    frames_dir = tmp_path / "constellation_frames"
    os.makedirs(frames_dir, exist_ok=True)
    yy, xx = np.indices((h, w), dtype=np.float64)

    n_frames = 14
    vec_ab = (32.0, 24.0)  # 40.0 px separation
    static_warm = (140.0, 35.0)  # unmasked stationary warm cluster

    for idx in range(n_frames):
        dx = idx * 1.5
        dy = idx * 1.0
        # Star A and Star B alternate which one is #1 brightest
        if idx % 2 == 0:
            amp_a, amp_b = 9000.0, 7500.0
        else:
            amp_a, amp_b = 7200.0, 9500.0

        # Frame 7 simulates a cloud dimming both stars below the static warm cluster
        if idx == 7:
            amp_a, amp_b = 2200.0, 2000.0

        ax, ay = 40.0 + dx, 50.0 + dy
        bx, by = ax + vec_ab[0], ay + vec_ab[1]

        img = np.full((h, w), 100.0, dtype=np.float64)
        img += amp_a * np.exp(-((xx - ax) ** 2 + (yy - ay) ** 2) / (2.0 * 1.8 ** 2))
        img += amp_b * np.exp(-((xx - bx) ** 2 + (yy - by) ** 2) / (2.0 * 1.8 ** 2))
        # Unmasked stationary warm cluster present in every frame
        img += 4500.0 * np.exp(-((xx - static_warm[0]) ** 2 + (yy - static_warm[1]) ** 2) / (2.0 * 1.5 ** 2))

        cv2.imwrite(str(frames_dir / f"frame_{idx:03d}.tif"), np.clip(img, 0, 65535).astype(np.uint16))

    res = combine_images(
        str(frames_dir),
        output_dir=str(tmp_path / "out_const"),
        fov_mode="full",
        use_defect_mask=False,
        save_individual_renders=False,
        verbose=False,
        filter_stripes=False,
        align_mode="constellation",
        min_snr=2.5,
        static_filter=True,
    )

    meta = res["mosaic_meta"]
    assert res["aligned_frames"] == n_frames
    assert meta["constellation_stars"] >= 2
    assert meta["static_peaks_rejected"] >= 1
    assert meta["identity_swaps_prevented"] >= 5
    # Total true drift is 13 * 1.5 = 19.5 px in X and 13 * 1.0 = 13.0 px in Y.
    # If a 40px rank-swap or 100px hot-pixel jump had occurred, span would be > 50 px.
    span_x, span_y = meta["drift_span_px"]
    assert 17.5 <= span_x <= 21.5, f"Expected X drift span ~19.5 px without rank-swap jumps, got {span_x}"
    assert 11.5 <= span_y <= 14.5, f"Expected Y drift span ~13.0 px without rank-swap jumps, got {span_y}"

