from typing import List, Tuple, Optional
import numpy as np
from .models import AlignmentConfig

class StarMatcher:
    """
    Builds rotation-invariant local neighborhood descriptors for stars and matches them
    using Euclidean distance and Lowe's ratio test.
    """
    def __init__(self, config: Optional[AlignmentConfig] = None):
        self.config = config or AlignmentConfig()

    def compute_descriptors(
        self,
        stars: List[Tuple[float, float]],
        N: Optional[int] = None
    ) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        """
        For each star, find its N nearest neighbors, compute their relative
        offsets, and normalize by rotation to make the descriptor rotation-invariant.
        """
        n_neighbors = N if N is not None else self.config.n_neighbors
        num_stars = len(stars)
        if num_stars <= n_neighbors:
            return np.empty((0, 2 * n_neighbors), dtype=np.float32), []

        stars_arr = np.array(stars, dtype=np.float32)
        descriptors = []
        valid_stars = []

        for i, p in enumerate(stars):
            diffs = stars_arr - p
            dists_sq = (diffs ** 2).sum(axis=1)
            sorted_indices = np.argsort(dists_sq)

            # Take the top N nearest neighbors (excluding the star itself at index 0)
            neighbor_idx = sorted_indices[1 : n_neighbors + 1]
            if len(neighbor_idx) < n_neighbors:
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

            feat = np.array(rotated_diffs, dtype=np.float32).flatten()
            descriptors.append(feat)
            valid_stars.append(p)

        if len(descriptors) == 0:
            return np.empty((0, 2 * n_neighbors), dtype=np.float32), []

        return np.array(descriptors, dtype=np.float32), valid_stars

    def match(
        self,
        des_ref: np.ndarray,
        des_src: np.ndarray,
        stars_ref: List[Tuple[float, float]],
        stars_src: List[Tuple[float, float]],
        max_dist: Optional[float] = None,
        lowe_ratio: Optional[float] = None
    ) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        Matches source descriptors with reference descriptors using L2 norm
        and filters ambiguous matches via Lowe's ratio test.
        Returns list of ((src_x, src_y), (ref_x, ref_y)) tuples.
        """
        max_d = max_dist if max_dist is not None else self.config.match_max_dist
        ratio = lowe_ratio if lowe_ratio is not None else self.config.match_lowe_ratio

        matches: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        if len(des_ref) == 0 or len(des_src) == 0:
            return matches

        for i, d_src in enumerate(des_src):
            dists = np.linalg.norm(des_ref - d_src, axis=1)
            best_idx = int(np.argmin(dists))
            if dists[best_idx] < max_d:
                sorted_dists = np.sort(dists)
                if len(sorted_dists) < 2 or sorted_dists[0] < ratio * sorted_dists[1]:
                    matches.append((stars_src[i], stars_ref[best_idx]))

        return matches

def compute_star_descriptors(stars: List[Tuple[float, float]], N: int = 3) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """Legacy helper function for star descriptor computation."""
    return StarMatcher().compute_descriptors(stars, N=N)

def match_star_descriptors(
    des_ref: np.ndarray,
    des_src: np.ndarray,
    stars_ref: List[Tuple[float, float]],
    stars_src: List[Tuple[float, float]],
    N: int = 3,
    max_dist: float = 5.0
) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Legacy helper function for star descriptor matching."""
    return StarMatcher().match(des_ref, des_src, stars_ref, stars_src, max_dist=max_dist)
