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
    """Configuration parameters for star detection and x-y translation alignment."""
    scale_factor: int = 1
    detection_threshold: float = 15.0
    detection_min_dist: int = 5
    max_candidates: int = 300
    centroid_window: int = 15
    patch_radius: int = 15
    search_radius: int = 80
    min_snr: float = 3.0
    min_peak_above_bg: float = 6.0
    n_neighbors: int = 3
    match_max_dist: float = 5.0
    match_lowe_ratio: float = 0.75
    ransac_reproj_threshold: float = 3.0
    max_translation: float = 500.0
    max_scale_diff: float = 0.05
    max_angle_deg: float = 5.0
    min_matches: int = 1

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
            self.metrics = {
                "inlier_ratio": self.inlier_ratio,
                "match_count": self.match_count,
                "dx": self.dx,
                "dy": self.dy,
                "status": self.status.value if hasattr(self.status, "value") else str(self.status)
            }

@dataclass
class ReferenceFeatures:
    """Precomputed star locations and properties for a locked reference frame."""
    stars: List[Tuple[float, float]] = field(default_factory=list)
    descriptors: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    valid_stars: List[Tuple[float, float]] = field(default_factory=list)
    scaled_shape: Tuple[int, int] = (0, 0)
    centroid: Optional[Tuple[float, float]] = None
