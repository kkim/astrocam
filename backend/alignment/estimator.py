from typing import List, Tuple, Optional
import numpy as np
import cv2
from .models import AlignmentConfig, AlignmentResult, AlignmentStatus

class TransformEstimator:
    """
    Estimates 2D partial affine or translation-only transformations from matched star pairs
    using RANSAC outlier rejection and astrophysical sanity checks.
    """
    def __init__(self, config: Optional[AlignmentConfig] = None):
        self.config = config or AlignmentConfig()

    def estimate(
        self,
        matches: List[Tuple[Tuple[float, float], Tuple[float, float]]],
        translation_only: bool = False,
        scale_factor: int = 1
    ) -> AlignmentResult:
        """
        Estimates the 2x3 affine matrix M mapping source coordinates to reference coordinates.
        Applies coordinate upscaling by scale_factor for translation components.
        """
        s = float(scale_factor)
        num_matches = len(matches)

        if num_matches < self.config.min_matches:
            return AlignmentResult(
                transform=np.eye(2, 3, dtype=np.float32),
                success=False,
                status=AlignmentStatus.INSUFFICIENT_MATCHES,
                match_count=num_matches,
                error_message=f"Need at least {self.config.min_matches} matches, found {num_matches}"
            )

        src_pts = np.float32([m[0] for m in matches]).reshape(-1, 1, 2)
        ref_pts = np.float32([m[1] for m in matches]).reshape(-1, 1, 2)

        M, mask = cv2.estimateAffinePartial2D(
            src_pts, ref_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.config.ransac_reproj_threshold
        )
        inliers = int(np.sum(mask)) if mask is not None else 0
        inlier_ratio = float(inliers / num_matches) if num_matches > 0 else 0.0

        if translation_only:
            if M is not None and mask is not None and inliers >= self.config.min_matches:
                inliers_src = src_pts[mask.ravel() == 1]
                inliers_ref = ref_pts[mask.ravel() == 1]
                diffs = inliers_ref - inliers_src
            else:
                diffs = ref_pts - src_pts

            dxs = diffs[:, 0, 0]
            dys = diffs[:, 0, 1]
            dx = float(np.median(dxs)) * s
            dy = float(np.median(dys)) * s

            if abs(dx) > self.config.max_translation or abs(dy) > self.config.max_translation:
                return AlignmentResult(
                    transform=np.eye(2, 3, dtype=np.float32),
                    success=False,
                    status=AlignmentStatus.TRANSLATION_OUT_OF_BOUNDS,
                    dx=dx,
                    dy=dy,
                    inlier_count=inliers,
                    inlier_ratio=0.0,
                    match_count=num_matches,
                    error_message=f"Translation (|dx|={abs(dx):.1f}, |dy|={abs(dy):.1f}) exceeded limit {self.config.max_translation}"
                )

            M_final = np.float32([[1, 0, dx], [0, 1, dy]])
            return AlignmentResult(
                transform=M_final,
                success=True,
                status=AlignmentStatus.SUCCESS,
                dx=dx,
                dy=dy,
                rotation_deg=0.0,
                scale=1.0,
                inlier_count=inliers,
                inlier_ratio=inlier_ratio,
                match_count=num_matches
            )

        if M is None:
            return AlignmentResult(
                transform=np.eye(2, 3, dtype=np.float32),
                success=False,
                status=AlignmentStatus.RANSAC_FAILED,
                match_count=num_matches,
                error_message="RANSAC failed to estimate affine matrix"
            )

        # Scale translation components up to original image resolution
        M[0, 2] *= s
        M[1, 2] *= s

        scale = float(np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2))
        if abs(scale - 1.0) > self.config.max_scale_diff:
            return AlignmentResult(
                transform=np.eye(2, 3, dtype=np.float32),
                success=False,
                status=AlignmentStatus.SCALE_OUT_OF_BOUNDS,
                scale=scale,
                inlier_count=inliers,
                inlier_ratio=0.0,
                match_count=num_matches,
                error_message=f"Scale {scale:.3f} deviates from 1.0 by more than {self.config.max_scale_diff}"
            )

        angle = float(np.arctan2(M[1, 0], M[0, 0]) * 180.0 / np.pi)
        if abs(angle) > self.config.max_angle_deg:
            return AlignmentResult(
                transform=np.eye(2, 3, dtype=np.float32),
                success=False,
                status=AlignmentStatus.ANGLE_OUT_OF_BOUNDS,
                rotation_deg=angle,
                inlier_count=inliers,
                inlier_ratio=0.0,
                match_count=num_matches,
                error_message=f"Rotation {angle:.2f}° exceeded limit {self.config.max_angle_deg}°"
            )

        dx = float(M[0, 2])
        dy = float(M[1, 2])
        if abs(dx) > self.config.max_translation or abs(dy) > self.config.max_translation:
            return AlignmentResult(
                transform=np.eye(2, 3, dtype=np.float32),
                success=False,
                status=AlignmentStatus.TRANSLATION_OUT_OF_BOUNDS,
                dx=dx,
                dy=dy,
                inlier_count=inliers,
                inlier_ratio=0.0,
                match_count=num_matches,
                error_message=f"Translation (|dx|={abs(dx):.1f}, |dy|={abs(dy):.1f}) exceeded limit {self.config.max_translation}"
            )

        return AlignmentResult(
            transform=M.astype(np.float32),
            success=True,
            status=AlignmentStatus.SUCCESS,
            dx=dx,
            dy=dy,
            rotation_deg=angle,
            scale=scale,
            inlier_count=inliers,
            inlier_ratio=inlier_ratio,
            match_count=num_matches
        )
