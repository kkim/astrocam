from typing import Optional, Union, Tuple, Dict, Any
import numpy as np

from .models import AlignmentConfig, AlignmentResult, AlignmentStatus
from .detector import StarDetector, find_bright_star_centroid


class StarAligner:
    """
    High-level star alignment engine. Performs deterministic x-y translation alignment
    based on the maximum brightness guide star with sub-pixel center-of-mass accuracy.
    No RANSAC or multi-star asterism descriptors required.
    """
    def __init__(self, config: Optional[AlignmentConfig] = None):
        self.config = config or AlignmentConfig()
        self.detector = StarDetector(self.config)
        self._ref_centroid: Optional[Tuple[float, float]] = None
        self._last_centroid: Optional[Tuple[float, float]] = None

    def set_reference(
        self,
        img_ref: np.ndarray,
        search_center: Optional[Tuple[float, float]] = None,
        search_radius: Optional[int] = None
    ) -> None:
        """
        Precomputes and caches the target star centroid for the reference frame.
        Use this when locking onto a target for closed-loop autoguiding.
        If search_center is provided, locks onto the star near that position.
        """
        radius = search_radius if search_radius is not None else self.config.search_radius
        res = find_bright_star_centroid(
            img_ref,
            min_snr=self.config.min_snr,
            min_peak_above_bg=self.config.min_peak_above_bg,
            patch_radius=self.config.patch_radius,
            search_center=search_center,
            search_radius=radius
        )
        if res is not None:
            self._ref_centroid = (res[0], res[1])
            self._last_centroid = (res[0], res[1])
        else:
            self._ref_centroid = None
            self._last_centroid = None

    def clear_reference(self) -> None:
        """Clears cached reference star centroid."""
        self._ref_centroid = None
        self._last_centroid = None

    @property
    def has_reference(self) -> bool:
        """Returns True if a reference star is currently cached."""
        return self._ref_centroid is not None

    def align_to_reference(
        self,
        img_src: np.ndarray,
        translation_only: bool = True,
        search_radius: Optional[int] = None
    ) -> AlignmentResult:
        """
        Aligns incoming source frame against the cached reference frame star centroid.
        Searches in a lock box around the last observed centroid to prevent jumping
        between stars of similar brightness.
        """
        if self._ref_centroid is None:
            return AlignmentResult(
                transform=np.eye(2, 3, dtype=np.float32),
                success=False,
                status=AlignmentStatus.NO_FEATURES,
                error_message="No reference star cached. Call set_reference() first."
            )

        target_center = self._last_centroid if self._last_centroid is not None else self._ref_centroid
        radius = search_radius if search_radius is not None else self.config.search_radius

        src_res = find_bright_star_centroid(
            img_src,
            min_snr=self.config.min_snr,
            min_peak_above_bg=self.config.min_peak_above_bg,
            patch_radius=self.config.patch_radius,
            search_center=target_center,
            search_radius=radius
        )
        if src_res is None:
            return AlignmentResult(
                transform=np.eye(2, 3, dtype=np.float32),
                success=False,
                status=AlignmentStatus.NO_FEATURES,
                error_message="No star detected in source frame within lock box."
            )

        self._last_centroid = (src_res[0], src_res[1])

        dx = float(self._ref_centroid[0] - src_res[0])
        dy = float(self._ref_centroid[1] - src_res[1])

        if abs(dx) > self.config.max_translation or abs(dy) > self.config.max_translation:
            return AlignmentResult(
                transform=np.eye(2, 3, dtype=np.float32),
                success=False,
                status=AlignmentStatus.TRANSLATION_OUT_OF_BOUNDS,
                dx=dx,
                dy=dy,
                error_message=f"Translation (|dx|={abs(dx):.1f}, |dy|={abs(dy):.1f}) exceeded limit {self.config.max_translation}"
            )

        M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        return AlignmentResult(
            transform=M,
            success=True,
            status=AlignmentStatus.SUCCESS,
            dx=dx,
            dy=dy,
            rotation_deg=0.0,
            scale=1.0,
            inlier_count=1,
            inlier_ratio=1.0,
            match_count=1,
            metrics={"inlier_ratio": 1.0, "match_count": 1, "dx": dx, "dy": dy, "status": "success"}
        )

    def align(
        self,
        img_ref: np.ndarray,
        img_src: np.ndarray,
        translation_only: bool = True,
        search_center: Optional[Tuple[float, float]] = None,
        search_radius: Optional[int] = None
    ) -> AlignmentResult:
        """
        Stateless alignment: finds the target star in img_ref and searches for it in
        img_src within a localized search radius (lock box).
        """
        radius = search_radius if search_radius is not None else self.config.search_radius
        ref_res = find_bright_star_centroid(
            img_ref,
            min_snr=self.config.min_snr,
            min_peak_above_bg=self.config.min_peak_above_bg,
            patch_radius=self.config.patch_radius,
            search_center=search_center,
            search_radius=radius
        )
        if ref_res is None:
            return AlignmentResult(
                transform=np.eye(2, 3, dtype=np.float32),
                success=False,
                status=AlignmentStatus.NO_FEATURES,
                error_message="No star detected in reference frame."
            )

        src_search_center = (ref_res[0], ref_res[1])
        src_res = find_bright_star_centroid(
            img_src,
            min_snr=self.config.min_snr,
            min_peak_above_bg=self.config.min_peak_above_bg,
            patch_radius=self.config.patch_radius,
            search_center=src_search_center,
            search_radius=radius
        )
        if src_res is None:
            return AlignmentResult(
                transform=np.eye(2, 3, dtype=np.float32),
                success=False,
                status=AlignmentStatus.NO_FEATURES,
                error_message="No star detected in source frame within search radius."
            )

        dx = float(ref_res[0] - src_res[0])
        dy = float(ref_res[1] - src_res[1])

        if abs(dx) > self.config.max_translation or abs(dy) > self.config.max_translation:
            return AlignmentResult(
                transform=np.eye(2, 3, dtype=np.float32),
                success=False,
                status=AlignmentStatus.TRANSLATION_OUT_OF_BOUNDS,
                dx=dx,
                dy=dy,
                error_message=f"Translation (|dx|={abs(dx):.1f}, |dy|={abs(dy):.1f}) exceeded limit {self.config.max_translation}"
            )

        M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        return AlignmentResult(
            transform=M,
            success=True,
            status=AlignmentStatus.SUCCESS,
            dx=dx,
            dy=dy,
            rotation_deg=0.0,
            scale=1.0,
            inlier_count=1,
            inlier_ratio=1.0,
            match_count=1,
            metrics={"inlier_ratio": 1.0, "match_count": 1, "dx": dx, "dy": dy, "status": "success"}
        )


def align_images(
    img_ref: np.ndarray,
    img_src: np.ndarray,
    nfeatures: int = 100,
    translation_only: bool = True,
    s: int = 1,
    return_metrics: bool = False,
    search_center: Optional[Tuple[float, float]] = None,
    search_radius: Optional[int] = None
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
    """
    Finds the 2x3 affine translation matrix M that maps img_src to img_ref
    based on the target star centroid within the specified search window.
    """
    aligner = StarAligner()
    result = aligner.align(
        img_ref,
        img_src,
        translation_only=translation_only,
        search_center=search_center,
        search_radius=search_radius
    )

    if return_metrics:
        return result.transform, result.metrics
    return result.transform
