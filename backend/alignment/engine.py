from typing import Optional, Union, Tuple, Dict, Any
import numpy as np
import cv2

from .models import AlignmentConfig, AlignmentResult, AlignmentStatus, ReferenceFeatures
from .geometry import scale_down
from .detector import StarDetector
from .matcher import StarMatcher
from .estimator import TransformEstimator

class StarAligner:
    """
    High-level star alignment engine. Coordinates preprocessing, star detection,
    descriptor matching, and transform estimation.
    
    Supports stateful reference frame caching (extracting stars/descriptors once for
    long-running autoguiding sessions) as well as stateless pair alignment.
    """
    def __init__(self, config: Optional[AlignmentConfig] = None):
        self.config = config or AlignmentConfig()
        self.detector = StarDetector(self.config)
        self.matcher = StarMatcher(self.config)
        self.estimator = TransformEstimator(self.config)
        self._ref_features: Optional[ReferenceFeatures] = None

    def _extract_frame_features(self, img: np.ndarray, s: int) -> Tuple[np.ndarray, list, np.ndarray, list]:
        """Downscales, converts to grayscale, detects stars, and computes descriptors."""
        img_scaled = scale_down(img, s)
        if len(img_scaled.shape) == 3:
            gray = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_scaled

        stars = self.detector.detect(gray)
        des, valid_stars = self.matcher.compute_descriptors(stars)
        return gray, stars, des, valid_stars

    def set_reference(self, img_ref: np.ndarray) -> None:
        """
        Precomputes and caches stars and descriptors for the reference frame.
        Use this when locking onto a target for continuous guiding or tracking.
        """
        gray, stars, des, valid_stars = self._extract_frame_features(img_ref, self.config.scale_factor)
        self._ref_features = ReferenceFeatures(
            stars=stars,
            descriptors=des,
            valid_stars=valid_stars,
            scaled_shape=gray.shape[:2]
        )

    def clear_reference(self) -> None:
        """Clears the cached reference frame features."""
        self._ref_features = None

    @property
    def has_reference(self) -> bool:
        """Returns True if a reference frame is currently cached."""
        return self._ref_features is not None

    def align_to_reference(self, img_src: np.ndarray, translation_only: bool = False) -> AlignmentResult:
        """
        Aligns incoming source frame against the cached reference frame.
        Skips feature extraction on the reference frame to save CPU cycles.
        """
        if self._ref_features is None:
            return AlignmentResult(
                transform=np.eye(2, 3, dtype=np.float32),
                success=False,
                status=AlignmentStatus.NO_FEATURES,
                error_message="No reference frame cached. Call set_reference() first."
            )

        ref = self._ref_features
        _, src_stars, src_des, src_valid = self._extract_frame_features(img_src, self.config.scale_factor)

        if len(ref.descriptors) == 0 or len(src_des) == 0:
            return AlignmentResult(
                transform=np.eye(2, 3, dtype=np.float32),
                success=False,
                status=AlignmentStatus.NO_FEATURES,
                match_count=0,
                error_message="Insufficient features detected in one or both frames"
            )

        matches = self.matcher.match(ref.descriptors, src_des, ref.valid_stars, src_valid)
        return self.estimator.estimate(
            matches,
            translation_only=translation_only,
            scale_factor=self.config.scale_factor
        )

    def align(self, img_ref: np.ndarray, img_src: np.ndarray, translation_only: bool = False) -> AlignmentResult:
        """
        Stateless alignment: computes features for both images, matches them,
        and estimates the transform mapping img_src to img_ref.
        """
        _, _, ref_des, ref_valid = self._extract_frame_features(img_ref, self.config.scale_factor)
        _, _, src_des, src_valid = self._extract_frame_features(img_src, self.config.scale_factor)

        if len(ref_des) == 0 or len(src_des) == 0:
            return AlignmentResult(
                transform=np.eye(2, 3, dtype=np.float32),
                success=False,
                status=AlignmentStatus.NO_FEATURES,
                match_count=0,
                error_message="Insufficient features detected in one or both frames"
            )

        matches = self.matcher.match(ref_des, src_des, ref_valid, src_valid)
        return self.estimator.estimate(
            matches,
            translation_only=translation_only,
            scale_factor=self.config.scale_factor
        )

def align_images(
    img_ref: np.ndarray,
    img_src: np.ndarray,
    nfeatures: int = 100,
    translation_only: bool = False,
    s: int = 4,
    return_metrics: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
    """
    Finds the affine transform M that maps img_src to img_ref using
    Star Neighborhood Descriptors.
    
    Provides 100% backwards compatibility with the legacy align_images function.
    """
    config = AlignmentConfig(scale_factor=s)
    aligner = StarAligner(config)
    result = aligner.align(img_ref, img_src, translation_only=translation_only)

    if return_metrics:
        return result.transform, result.metrics
    return result.transform
