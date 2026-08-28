"""From-scratch Structure-from-Motion and Multi-View Stereo.

Implemented without any library that performs the SfM task itself. OpenCV is
used for image I/O, SIFT features and descriptor matching; NumPy for linear
algebra; SciPy for a generic non-linear least-squares solver. All geometry -
epipolar estimation, pose recovery, triangulation, PnP, bundle adjustment,
dense stereo and surface extraction - is implemented in this package.
"""

from .geometry import (
    ransac_essential, decompose_essential, select_pose_by_cheirality,
    triangulate_dlt, triangulate_multiview, ransac_pnp, pnp_linear,
    reprojection_error, rotation_angle_deg,
)
from .features import FeatureSet, extract_features, match_pair, build_tracks, Track
from .bundle import bundle_adjust, BundleResult
from .mapper import IncrementalMapper, MapperConfig, Reconstruction

__all__ = [
    "ransac_essential", "decompose_essential", "select_pose_by_cheirality",
    "triangulate_dlt", "triangulate_multiview", "ransac_pnp", "pnp_linear",
    "reprojection_error", "rotation_angle_deg",
    "FeatureSet", "extract_features", "match_pair", "build_tracks", "Track",
    "bundle_adjust", "BundleResult",
    "IncrementalMapper", "MapperConfig", "Reconstruction",
]
