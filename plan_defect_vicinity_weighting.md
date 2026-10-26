# Plan: Zero-Weight Bad Pixels *and Their Vicinity* in Mosaics

Status: **Implemented**

## 1. Where we are today

- **Building the defect map:** [`_finalize_black_calibration`](file:///home/kio/projects/astrocam/backend/pipeline.py#L481-L541) averages the dark frames, then marks every pixel above `max(med + 6, med + 5σ)`. It saves the result to `captures/calibration/defect_mask.npy` as a `uint8` array, `(h, w)`.
- **Using it in the mosaic:** [`combine_images`](file:///home/kio/projects/astrocam/bin/mosaic_stars.py#L498-L518) loads the mask and multiplies the feather mask by `pixel_weights = (mask == 0)`. So defective pixels already get **weight 0**, but only the exact pixels in the mask.
- **Caller:** [`mosaic_service.py`](file:///home/kio/projects/astrocam/backend/mosaic_service.py#L203) runs the script without a mask argument, so the default calibration path is used.

So yes, this is feasible. Most of the plumbing already exists.

## 2. Problems with the current approach

| # | Problem | Effect |
|---|---------|--------|
| P1 | **Leak through interpolation.** The frame and the weights are warped separately with `INTER_LINEAR`, then multiplied: `acc += warp(frame) * warp(w)`. On a sub-pixel shift, the hot value is spread into neighbouring canvas pixels by `warp(frame)`, and `warp(w)` is still nonzero there. Stripes avoid this because their pixels are zeroed (`frame_f[...] = 0`). Defects are not. | Faint hot-pixel trails/halos can still show up in the mosaic, even with weight 0. |
| P2 | **No vicinity.** Hot pixels on this sensor spill into neighbours. MJPEG compression smears them across 8×8 DCT blocks, and debayer/blooming adds a ±1–2 px halo. Those neighbours sit just under the 5σ threshold, so they are not in the mask. | A dimmer "ring" stays around each masked pixel. |
| P3 | **Mismatched shapes are skipped silently.** If `d_mask.shape != (h, w)` (for example, after a resolution change), the mask is ignored with no warning. | The user thinks defects are masked when they aren't. |
| P4 | **Star detection can lock onto a hot pixel.** `detect_star_centroid` doesn't receive the defect mask (only `stripe_mask`). | Rare, but a strong hot pixel could become the "guide star". |

## 3. Proposed design

### 3.1 Dilate the defect mask (the vicinity), plus an optional soft falloff

New helper in `bin/mosaic_stars.py`:

```python
def build_defect_weights(d_mask, radius=1, soft_px=0) -> np.ndarray:
    """1.0 = healthy, 0.0 = defect or within `radius` px; optional linear
    ramp back to 1.0 over the next `soft_px` px."""
    bad = (d_mask > 0).astype(np.uint8)
    if radius > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
        bad = cv2.dilate(bad, k)
    if soft_px <= 0:
        return (1.0 - bad).astype(np.float64)
    # distance (px) from nearest bad pixel -> ramp 0..1
    dist = cv2.distanceTransform(1 - bad, cv2.DIST_L2, 3)
    return np.clip(dist / soft_px, 0.0, 1.0).astype(np.float64)
```

- Default `radius=1` (3×3 ellipse/cross). This handles debayer/bloom spill.
- Optional `soft_px` gives a smooth ramp instead of a hard hole, which helps avoid tiny seams in places with low overlap.
- **Coverage guard:** with radius 1 and a typical few hundred defects, we lose well under 1% of sensor area. Log the masked-area fraction. If it's above a threshold (for example 5%), warn and suggest a smaller radius or recalibration.

### 3.2 Fix the interpolation leak (P1): premultiplied accumulation

Change the accumulation in [Pass 3](file:///home/kio/projects/astrocam/bin/mosaic_stars.py#L540-L579) to:

```python
w = frame_sensor_weight                       # (h, w) float64: feather * defect * stripes
frame_w = frame_f * w[:, :, None] if num_channels > 1 else frame_f * w
warped_frame_w = cv2.warpAffine(frame_w, M, (canvas_w, canvas_h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
warped_weights = cv2.warpAffine(w, M, (canvas_w, canvas_h), ...same flags...)
accumulator += warped_frame_w                 # NO extra "* warped_weights" here
weight_map  += warped_weights
```

> [!WARNING]
> Delete the existing `accumulator += warped_frame * warped_weights[...]` lines. Keeping them alongside the new code would apply the weight twice. The final normalization (`accumulator / weight_map`, under the `# Compute weighted average` comment) stays the same.

This is the correct weighted-average formulation: a zero-weight source pixel contributes **nothing**, whatever the interpolation. It also covers stripes, so the special `frame_f[y0:y1, x0:x1] = 0.0` hack becomes unnecessary (it can stay; it's harmless). The cost is one extra multiply per frame.

### 3.3 Optional: also exclude defects from star detection (P4)

Combine the dilated defect mask with `stripe_mask` before calling `detect_star_centroid(img, exclude_mask=...)`. It's cheap and removes a failure mode.

### 3.4 Visibility (P3)

- Print a warning (stdout goes to the mosaic log) when the mask shape doesn't match, with both shapes.
- Print: `N defects -> M px masked after dilation (r=1), X.XX% of sensor`.

### 3.5 Configuration

| Knob | Where | Default |
|------|-------|---------|
| `--defect-radius` | `mosaic_stars.py` CLI + `combine_images(defect_radius=1)` | `1` |
| `--defect-soft-px` | same | `0` |
| `--no-defect-mask` | same (to make A/B testing easy) | off |

`mosaic_service.py` doesn't need changes at first (defaults apply). Later it could read these from `config.json`.

### 3.6 Optional: do it at calibration time instead?

Option: store a dilated mask next to the raw one in `_finalize_black_calibration`. **Rejected for now.** Keeping the raw mask and dilating at mosaic time lets us tune the radius without recalibrating. It also keeps the live-view softening in [`pipeline.py`](file:///home/kio/projects/astrocam/backend/pipeline.py#L313-L319) unchanged. (Live view could reuse the same helper later.)

## 4. Implementation steps

> [!NOTE]
> Line-number links in this doc reflect the code on 2026-10-25 and may drift. Find code by the symbols and comments named below instead.

1. **`bin/mosaic_stars.py`: helper.** Add `build_defect_weights()` (3.1) next to `create_feather_mask()`.
2. **`combine_images()`: mask loading.** Find the block under `# Load sensor defect mask if provided or available in captures/calibration`. Replace `pixel_weights = np.where(d_mask > 0, 0.0, 1.0)` with `pixel_weights = build_defect_weights(d_mask, defect_radius, defect_soft_px)`. Add an `else:` branch to `if d_mask.shape == (h, w):` that prints the mismatch warning (3.4).
   - Ordering problem: the mask is currently loaded in Pass 3, **after** star detection (Pass 1). For 3.3, move the mask loading before Pass 1. `h, w` aren't known yet at that point, so check the shape against the first frame's `img.shape[:2]` lazily inside the loop.
3. **`combine_images()`: Pass 3.** Find the loop `for idx, (d, (tx, ty)) in enumerate(zip(detections, shifts)):` and apply 3.2.
4. **Pass 1 (optional, 3.3):** in `detect_star_centroid(img, exclude_mask=stripe_mask)`, pass `exclude_mask = stripe_mask | defect_bool` (both `bool (h, w)`). `stripe_mask` can be `None`, so handle that case. `stripe_rects_to_mask()` returns `bool`, and `detect_star_centroid` expects `bool`.
5. **Signature / CLI.** Add `defect_radius: int = 1, defect_soft_px: int = 0, use_defect_mask: bool = True` to `combine_images()`. Add matching flags in `parse_args()` (existing pattern: `--defect-mask`, `--no-stripe-filter`). Pass them through in the `combine_images(...)` call at the bottom of the file.
   - `bin/batch_mosaics.py` and `backend/mosaic_service.py` (subprocess `cmd = [py_exec, script_path, s_dir, "-o", output_dir]`) need **no changes**, because the defaults apply.
6. **Docs.** Add a short note on defect handling to `GEMINI.md` / `ARCHITECTURE.md`.

### Gotchas

- `combine_images()` **auto-loads** `captures/calibration/defect_mask.npy` when `defect_mask_path=None`. Tests must always pass an explicit `defect_mask_path` (a tmp file), or the real calibration mask may be picked up.
- The defect mask is computed from **grayscale** luminance (`pipeline.py`, `_finalize_black_calibration`). It is `(h, w)` even for BGR captures, so broadcast it across channels as in 3.2.
- `cv2.distanceTransform` requires `uint8` input. `cv2.warpAffine` works with `float64`.
- Frame shifts are float centroid differences, so sub-pixel shifts are the normal case (that's why P1 matters).

## 5. Tests (`tests/test_mosaic_defects.py`)

**Harness:** `combine_images()` works on a directory, so each test should:
- write synthetic 16-bit grayscale TIFFs into `tmp_path` with `cv2.imwrite`;
- save the mask with `np.save(tmp_path / "mask.npy", ...)`;
- call `combine_images(str(tmp_path / "frames"), output_dir=..., defect_mask_path=..., save_individual_renders=False, verbose=False, filter_stripes=False, feather_px=0)`;
- check `result["mosaic_linear"]` and `result["weight_map"]`.

Each frame needs a single bright Gaussian "star" (σ≈2 px, peak ≫ background) placed at known sub-pixel offsets, because alignment uses its centroid. `tests/conftest.py` only adds `backend/` to `sys.path`, so add `bin/` too (e.g. `sys.path.insert` in the test file).

1. **Leak test:** synthetic flat frames (value 100) with one hot pixel (value 4000) marked in the mask, shifted by sub-pixel amounts (0.3, 0.7). Assert that the max deviation from 100 in the mosaic is < 1 ADU. This fails on current code (P1) and should pass after 3.2.
2. **Vicinity test:** add a ±1 px halo (value 400) around the hot pixel. Don't include the halo in the mask. With `radius=1`, assert the halo doesn't appear. With `radius=0`, assert it does (proves the knob works).
3. **Coverage test:** with shifts of several pixels between frames, the canvas has no holes (`weight_map > 0`) over the overlap region, even with the dilated mask.
4. **Shape-mismatch test:** a mask of the wrong shape is ignored and a warning is printed.
5. **Regression:** run on an existing capture session. Visually compare `mosaic_hdr_*` before/after and check that `mosaic_coverage_weights.png` shows small pinholes only where expected.

## 6. Risks and notes

- **Holes with too few frames:** if a canvas pixel is only covered by one frame and that pixel is masked, it becomes `valid_mask=False` (black). That's acceptable, and better than a hot pixel. The soft ramp and the coverage guard keep it rare. Optional follow-up: fill remaining holes with `cv2.inpaint` on the final mosaic.
- **Drift direction:** with almost purely 1-D drift (sidereal), a column of defects could line up with the motion. Dithering (a future roadmap item) is the real fix for that.
- **Performance:** dilation/distance transform runs once per mosaic, so it's negligible. The premultiply adds one array multiply per frame.
