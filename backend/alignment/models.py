from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

class AlignmentStatus(Enum):
    SUCCESS = "success"
    NO_FEATURES = "no_features"
    INSUFFICIENT_MATCHES = "insufficient_matches"
    RANSAC_FAILED = "ransac_failed"
    SCALE_OUT_OF_BOUNDS = "scale_out_of_bounds"
    ANGLE_OUT_OF_BOUNDS = "angle_out_of_bounds"
    TRANSLATION_OUT_OF_BOUNDS = "translation_out_of_bounds"

@dataclass
class AlignmentConfig:
    """Configuration parameters for star detection, matching, and transformation estimation."""
    scale_factor: int = 4
    detection_threshold: float = 15.0
    detection_min_dist: int = 5
    max_candidates: int = 300
    centroid_window: int = 5
    n_neighbors: int = 3
    match_max_dist: float = 5.0
    match_lowe_ratio: float = 0.75
    ransac_reproj_threshold: float = 3.0
    max_translation: float = 100.0
    max_scale_diff: float = 0.05
    max_angle_deg: float = 5.0
    min_matches: int = 3

@dataclass
class AlignmentResult:
    """Structured result returned by the alignment engine."""
    transform: np.ndarray
    success: bool
    status: AlignmentStatus
    dx: float = 0.0
    dy: float = 0.0
    rotation_deg: float = 0.0
    scale: float = 1.0
    inlier_count: int = 0
    inlier_ratio: float = 0.0
    match_count: int = 0
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.metrics:
            self.metrics = {"inlier_ratio": self.inlier_ratio}

@dataclass
class ReferenceFeatures:
    """Precomputed star locations and descriptors for a locked reference frame."""
    stars: List[Tuple[float, float]]
    descriptors: np.ndarray
    valid_stars: List[Tuple[float, float]]
    scaled_shape: Tuple[int, int]
