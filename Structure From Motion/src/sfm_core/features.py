"""Feature extraction, matching and track construction.

OpenCV is used here for SIFT detection/description and brute-force descriptor
matching. Those are image-processing primitives: they find and describe
interest points, and compare descriptor vectors. They do not estimate any
geometry, so they sit outside the "SfM task" the project constraint refers to.

Everything geometric - verification, pose, triangulation - is implemented in
:mod:`src.sfm_core.geometry`.

Tracks
------
A *track* is one physical 3D point observed in several images. Pairwise matches
only relate images two at a time; chaining them with union-find turns
``(img A, feat i) <-> (img B, feat j)`` links into connected components, each of
which becomes a candidate 3D point. Longer tracks are better constrained, and
track length is the main lever on reconstruction accuracy.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..utils import ensure_dir, get_logger
from .geometry import ransac_essential, ransac_fundamental

__all__ = ["FeatureSet", "extract_features", "match_pair", "PairMatch",
           "build_tracks", "Track", "make_matcher"]

LOGGER = get_logger()


@dataclass
class FeatureSet:
    """Keypoints and descriptors for one image.

    ``method`` is carried alongside the data because the descriptor's *type*
    decides how it must be compared: SIFT is a float histogram matched under
    L2, while AKAZE and ORB are binary strings matched under Hamming distance.
    Comparing a binary descriptor with L2 silently produces nonsense matches
    rather than an error, so the metric is derived from this field instead of
    being assumed.
    """

    name: str
    keypoints: np.ndarray          # (N, 2) pixel coordinates
    descriptors: np.ndarray        # (N, D) float32 (SIFT) or uint8 (AKAZE/ORB)
    scales: np.ndarray             # (N,) keypoint size
    colors: np.ndarray             # (N, 3) RGB sampled at the keypoint
    width: int = 0
    height: int = 0
    method: str = "sift"

    @property
    def is_binary(self) -> bool:
        return self.descriptors.dtype == np.uint8

    def __len__(self) -> int:
        return len(self.keypoints)


#: Detectors this project can use. All are local feature detector/descriptors:
#: they find and describe interest points and estimate no geometry, which is
#: why they sit outside the assignment's prohibition.
FEATURE_METHODS = ("sift", "akaze", "orb")


def make_detector(method: str, max_features: int, contrast_threshold: float = 0.04):
    """Build one of the supported detectors.

    The three differ in ways that matter for reconstruction, not just speed:

    ``sift``
        Float histogram over gradient orientations. The strongest baseline on
        high-frequency texture and scale change, and the only one the RootSIFT
        (Hellinger) transform applies to.
    ``akaze``
        Nonlinear scale space, so edges are not blurred across as they are in
        a Gaussian pyramid. Binary descriptor; often better on smooth or mixed
        surfaces, sometimes yielding fewer long tracks.
    ``orb``
        Oriented FAST plus rotated BRIEF. Much the fastest and the smallest
        descriptor, but the most sensitive to large viewpoint change.
    """
    method = method.lower()
    if method not in FEATURE_METHODS:
        raise ValueError(f"unknown feature method {method!r}; expected one of {FEATURE_METHODS}")

    factory = {"sift": "SIFT_create", "akaze": "AKAZE_create", "orb": "ORB_create"}[method]
    if not hasattr(cv2, factory):
        # Which detectors ship with OpenCV is a build-time decision, and it
        # changed: OpenCV 5 dropped AKAZE/KAZE, which 4.x provided. Say so
        # plainly rather than failing with a bare AttributeError.
        available = [m for m in FEATURE_METHODS
                     if hasattr(cv2, {"sift": "SIFT_create", "akaze": "AKAZE_create",
                                      "orb": "ORB_create"}[m])]
        raise RuntimeError(
            f"OpenCV {cv2.__version__} does not provide cv2.{factory}, so the "
            f"{method!r} detector is unavailable. Available here: {available}. "
            f"This project's requirements.txt pins opencv-contrib-python==4.14.0.94, "
            f"which provides all of {list(FEATURE_METHODS)}; OpenCV 5 removed AKAZE.")

    if method == "sift":
        return cv2.SIFT_create(nfeatures=max_features,
                               contrastThreshold=contrast_threshold)
    if method == "akaze":
        # Threshold traded down from the 1e-3 default: this project's images are
        # 1.9 MP and recompressed, and the default yields too few keypoints.
        return cv2.AKAZE_create(threshold=6e-4)
    return cv2.ORB_create(nfeatures=max_features, scaleFactor=1.2, nlevels=8)


def extract_features(
    image_paths: Sequence[Path],
    max_features: int = 8000,
    max_image_size: int = 0,
    cache_path: Optional[Path] = None,
    contrast_threshold: float = 0.04,
    root_sift: bool = True,
    method: str = "sift",
) -> Tuple[List[FeatureSet], float]:
    """Detect SIFT features in every image. Returns (features, scale_applied).

    ``max_image_size`` downscales the long side before detection; the returned
    keypoint coordinates are in the *downscaled* frame, and the caller must use
    an intrinsics matrix consistent with that frame.

    RootSIFT
    --------
    ``root_sift`` L1-normalises each descriptor and takes its square root, so
    that Euclidean distance between the transformed vectors equals the
    Hellinger distance between the originals (Arandjelovic & Zisserman, 2012).
    SIFT descriptors are histograms, and Euclidean distance over-weights the
    few large bins; the Hellinger metric compares them on more equal terms and
    reliably recovers matches the ratio test would otherwise discard. It is
    three lines of arithmetic on a descriptor - no part of the reconstruction
    task - and costs nothing at match time.
    """
    if cache_path and Path(cache_path).is_file():
        try:
            with open(cache_path, "rb") as fh:
                payload = pickle.load(fh)
            if payload.get("names") == [p.name for p in image_paths] and \
               payload.get("max_features") == max_features and \
               payload.get("max_image_size") == max_image_size and \
               payload.get("root_sift") == root_sift and \
               payload.get("method", "sift") == method:
                LOGGER.info("Loaded cached features for %d images", len(payload["features"]))
                return payload["features"], payload["scale"]
        except Exception as exc:
            LOGGER.debug("Feature cache unusable (%s); recomputing", exc)

    detector = make_detector(method, max_features, contrast_threshold)
    features: List[FeatureSet] = []
    scale_applied = 1.0

    for path in image_paths:
        data = np.fromfile(str(path), dtype=np.uint8)   # unicode-safe read
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Could not decode image: {path}")
        if max_image_size:
            long_side = max(bgr.shape[:2])
            if long_side > max_image_size:
                s = max_image_size / long_side
                bgr = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                scale_applied = s
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        kps, desc = detector.detectAndCompute(gray, None)
        if desc is None or len(kps) == 0:
            LOGGER.warning("No features found in %s", path.name)
            kps, desc = [], np.zeros((0, 128), np.float32)

        # AKAZE and ORB return uint8 binary strings; SIFT returns float
        # histograms. Preserve the dtype - casting a binary descriptor to
        # float would let it be compared under the wrong metric.
        desc = np.asarray(desc)
        if desc.dtype != np.uint8:
            desc = desc.astype(np.float32)

        # RootSIFT is a transform of a *histogram*; it is meaningless on a
        # binary descriptor, so it applies to SIFT alone.
        if root_sift and method == "sift" and len(desc) and desc.dtype != np.uint8:
            # L1-normalise, then square-root: Euclidean distance over the
            # result is the Hellinger distance over the original histograms.
            desc = desc / np.maximum(desc.sum(axis=1, keepdims=True), 1e-7)
            desc = np.sqrt(desc, dtype=np.float32)

        pts = np.array([k.pt for k in kps], dtype=np.float64).reshape(-1, 2)
        sizes = np.array([k.size for k in kps], dtype=np.float64)
        h, w = bgr.shape[:2]
        ix = np.clip(pts[:, 0].astype(int), 0, w - 1)
        iy = np.clip(pts[:, 1].astype(int), 0, h - 1)
        colors = bgr[iy, ix][:, ::-1].astype(np.uint8) if len(pts) else np.zeros((0, 3), np.uint8)

        features.append(FeatureSet(path.name, pts, desc,
                                   sizes, colors, w, h, method=method))
        LOGGER.debug("%s: %d features", path.name, len(pts))

    if cache_path:
        ensure_dir(Path(cache_path).parent)
        with open(cache_path, "wb") as fh:
            pickle.dump({"names": [p.name for p in image_paths], "features": features,
                         "scale": scale_applied, "max_features": max_features,
                         "max_image_size": max_image_size,
                         "root_sift": root_sift, "method": method}, fh)
    return features, scale_applied


@dataclass
class PairMatch:
    """Geometrically verified correspondences between two images."""

    i: int
    j: int
    idx_i: np.ndarray
    idx_j: np.ndarray
    num_raw: int = 0
    num_ratio: int = 0
    num_inliers: int = 0
    #: Fundamental matrix from verification (intrinsics-free).
    F: Optional[np.ndarray] = None
    median_angle: float = 0.0

    @property
    def inlier_ratio(self) -> float:
        return self.num_inliers / max(1, self.num_ratio)


def _mutual_ratio_match(desc_a: np.ndarray, desc_b: np.ndarray, ratio: float,
                        matcher: cv2.BFMatcher) -> Tuple[np.ndarray, np.ndarray, int]:
    """Lowe ratio test in both directions, keeping only mutual agreements."""
    if len(desc_a) < 2 or len(desc_b) < 2:
        return np.empty(0, np.int64), np.empty(0, np.int64), 0

    def one_way(d1, d2):
        out = {}
        for pair in matcher.knnMatch(d1, d2, k=2):
            if len(pair) < 2:
                continue
            best, second = pair
            if second.distance > 0 and best.distance < ratio * second.distance:
                out[best.queryIdx] = best.trainIdx
        return out

    fwd = one_way(desc_a, desc_b)
    n_ratio = len(fwd)
    bwd = one_way(desc_b, desc_a)
    mutual = {a: b for a, b in fwd.items() if bwd.get(b) == a}
    if not mutual:
        return np.empty(0, np.int64), np.empty(0, np.int64), n_ratio
    items = sorted(mutual.items())
    return (np.array([a for a, _ in items], np.int64),
            np.array([b for _, b in items], np.int64), n_ratio)


def make_matcher(kind: str = "flann", binary: bool = False):
    """Descriptor matcher. ``flann`` is approximate but ~20x faster than ``bf``.

    Approximate search costs a small number of correspondences, which the
    subsequent RANSAC verification absorbs; exhaustive matching of 100+ images
    is otherwise dominated by brute-force descriptor comparison.

    ``binary`` selects the metric, and it is not optional. AKAZE and ORB emit
    bit strings whose distance is Hamming; scoring those under L2 returns
    numbers rather than an error, so the failure would appear as an
    inexplicably poor reconstruction rather than a crash. FLANN's KD-tree
    cannot index bit strings either, so binary descriptors use LSH.
    """
    if binary:
        if kind == "bf":
            return cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=False)
        # FLANN_INDEX_LSH
        return cv2.FlannBasedMatcher(
            dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1),
            dict(checks=32))
    if kind == "bf":
        return cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=False)
    index_params = dict(algorithm=1, trees=4)      # FLANN_INDEX_KDTREE
    return cv2.FlannBasedMatcher(index_params, dict(checks=32))


def match_pair(
    fa: FeatureSet, fb: FeatureSet, i: int, j: int, K: np.ndarray,
    K2: Optional[np.ndarray] = None,
    ratio: float = 0.8, threshold_px: float = 1.5, min_inliers: int = 25,
    matcher: Optional[object] = None, seed: int = 0,
    min_inlier_ratio: float = 0.0, floor_inliers: int = 0,
) -> Optional[PairMatch]:
    """Match two images and verify the correspondences geometrically.

    Verification uses the project's own RANSAC essential-matrix estimator, so
    the pair is kept only if a single rigid two-view geometry explains enough
    of the matches. This removes the bulk of repeated-structure mismatches
    before they can poison the reconstruction.

    Acceptance
    ----------
    A flat inlier count is a poor test on its own, because it conflates two
    different things: how much the images overlap, and how well a single rigid
    geometry explains that overlap. Measured on this project's capture, pair
    71-74 had 27 inliers from 70 ratio-matches (39% - an excellent geometric
    fit on a small overlap) and was rejected at a floor of 30, while pair
    84-85 had 20 inliers from 229 (9% - consistent with noise) and differed
    only in being further below the same floor.

    So a pair is accepted when it clears ``min_inliers`` outright, *or* when
    it clears the lower ``floor_inliers`` while also explaining at least
    ``min_inlier_ratio`` of its ratio-matches. Leaving ``min_inlier_ratio``
    at 0 disables the second route and restores a plain count test.
    """
    matcher = matcher or make_matcher("flann", binary=fa.is_binary)
    accept_floor = min(min_inliers, floor_inliers) if min_inlier_ratio > 0 else min_inliers
    accept_floor = max(8, accept_floor)  # below 8 the 8-point solver has nothing to fit

    idx_a, idx_b, n_ratio = _mutual_ratio_match(fa.descriptors, fb.descriptors, ratio, matcher)
    if len(idx_a) < accept_floor:
        return None

    pts_a = fa.keypoints[idx_a]
    pts_b = fb.keypoints[idx_b]
    # Verification uses F, not E: it is independent of the intrinsics, so the
    # match graph does not inherit errors from the focal-length prior.
    result = ransac_fundamental(pts_a, pts_b, threshold_px=threshold_px, seed=seed)
    if result.model is None:
        return None
    inlier_ratio = result.num_inliers / max(1, n_ratio)
    strong = result.num_inliers >= min_inliers
    well_explained = (min_inlier_ratio > 0
                      and result.num_inliers >= accept_floor
                      and inlier_ratio >= min_inlier_ratio)
    if not (strong or well_explained):
        return None

    keep = result.inliers
    return PairMatch(
        i=i, j=j, idx_i=idx_a[keep], idx_j=idx_b[keep],
        num_raw=min(len(fa), len(fb)), num_ratio=n_ratio,
        num_inliers=int(keep.sum()), F=result.model,
    )


# ===========================================================================
# Tracks
# ===========================================================================


class _UnionFind:
    """Disjoint-set over (image, feature) observations, with path compression."""

    def __init__(self) -> None:
        self.parent: Dict[Tuple[int, int], Tuple[int, int]] = {}

    def find(self, x: Tuple[int, int]) -> Tuple[int, int]:
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:      # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: Tuple[int, int], b: Tuple[int, int]) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


@dataclass
class Track:
    """One candidate 3D point: a set of observations across images."""

    observations: Dict[int, int] = field(default_factory=dict)   # image index -> feature index
    point3d: Optional[np.ndarray] = None
    color: Optional[np.ndarray] = None
    error: float = 0.0

    def __len__(self) -> int:
        return len(self.observations)


def build_tracks(matches: Sequence[PairMatch], min_length: int = 2,
                 max_length: Optional[int] = None) -> List[Track]:
    """Chain pairwise matches into multi-image tracks via union-find.

    Tracks containing two features from the *same* image are discarded: a
    single 3D point cannot project to two different places in one view, so such
    a component is proof that some match in the chain was wrong. Dropping them
    is a cheap and effective consistency filter, especially on scenes with
    repeated structure.
    """
    uf = _UnionFind()
    for m in matches:
        for a, b in zip(m.idx_i, m.idx_j):
            uf.union((m.i, int(a)), (m.j, int(b)))

    groups: Dict[Tuple[int, int], Dict[int, int]] = {}
    conflicted: set = set()
    for node in list(uf.parent):
        root = uf.find(node)
        image, feature = node
        bucket = groups.setdefault(root, {})
        if image in bucket and bucket[image] != feature:
            conflicted.add(root)
        else:
            bucket[image] = feature

    tracks: List[Track] = []
    for root, observations in groups.items():
        if root in conflicted:
            continue
        if len(observations) < min_length:
            continue
        if max_length is not None and len(observations) > max_length:
            continue
        tracks.append(Track(observations=observations))

    LOGGER.info("Built %d tracks (%d conflicted components discarded)",
                len(tracks), len(conflicted))
    return tracks
