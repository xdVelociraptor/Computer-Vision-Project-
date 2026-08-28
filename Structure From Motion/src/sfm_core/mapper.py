"""Incremental Structure from Motion.

The reconstruction loop, implemented from scratch:

1. **Match graph** - every considered image pair is matched and geometrically
   verified with the project's own RANSAC essential-matrix estimator.
2. **Tracks** - pairwise matches are chained into multi-image tracks.
3. **Seed pair** - a pair is chosen for many inliers *and* wide parallax. This
   choice largely determines whether the whole reconstruction succeeds: a
   low-parallax seed produces a badly conditioned initial structure that later
   views inherit.
4. **Initialise** - relative pose from E, resolved by cheirality, then
   triangulate the tracks the pair shares.
5. **Grow** - repeatedly register the image with the most 2D-3D
   correspondences using RANSAC PnP, triangulate its newly-connected tracks,
   and run a local bundle adjustment.
6. **Refine** - periodic global bundle adjustment, followed by filtering on
   reprojection error and triangulation angle.

Why incremental rather than global: with an unordered set of phone photos there
is no reliable initial estimate for a global method, and incremental SfM can
reject an image that fails to register instead of corrupting everything.
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from ..utils import get_logger
from .bundle import bundle_adjust
from .features import (FeatureSet, PairMatch, Track, build_tracks, make_matcher,
                       match_pair)
from .geometry import (
    decompose_essential,
    undistort_points,
    ransac_essential,
    ransac_pnp,
    reprojection_error,
    rotation_angle_deg,
    select_pose_by_cheirality,
    triangulate_multiview,
    triangulation_angle_deg,
)

__all__ = ["Reconstruction", "MapperConfig", "IncrementalMapper"]

LOGGER = get_logger()


@dataclass
class MapperConfig:
    """Tunable behaviour of the incremental mapper."""

    match_ratio: float = 0.8
    match_threshold_px: float = 1.5
    min_pair_inliers: int = 30
    #: Second route to accepting a pair, for genuine but small overlaps. A
    #: pair below ``min_pair_inliers`` is still kept when it has at least
    #: ``min_pair_floor_inliers`` inliers *and* they account for this fraction
    #: of its ratio-matches. Measured on this capture: frames 54 and 74 sit
    #: outside the match graph entirely despite bridging pairs that explain
    #: 28-39% of their matches, while spurious pairs of similar inlier count
    #: explain under 10%. 0 disables the second route.
    min_pair_inlier_ratio: float = 0.25
    min_pair_floor_inliers: int = 15
    #: 0 = exhaustive. Otherwise only pairs within this index distance are
    #: matched, which is correct for a sequential capture and O(n) instead of
    #: O(n^2).
    sequential_window: int = 0
    #: flann (fast, approximate) or bf (exact, much slower)
    matcher: str = "flann"

    min_track_length: int = 2
    seed_min_parallax_deg: float = 3.0
    #: Pairs examined when choosing the seed, taken in order of inlier count.
    #:
    #: Must be large enough to reach past the near-duplicates. A burst capture
    #: fills the top of the inlier ranking with frames shot from the same spot -
    #: they match superbly and have no baseline - so a small pool can contain
    #: no usable seed at all. Measured on a 146-image burst at ~2 frames a
    #: second: every one of the top 40 pairs was under 3 degrees of parallax
    #: and the best was 0.051, which triangulates nothing.
    #:
    #: Each extra candidate costs one RANSAC essential-matrix estimate, so this
    #: is a few seconds of one-time work, paid once per reconstruction.
    seed_candidates: int = 200

    pnp_threshold_px: float = 8.0
    #: P3P plus non-linear refinement produces tighter, cleaner inlier sets
    #: than the old 6-point DLT, so the acceptance bar can be lower without
    #: admitting bad poses.
    #:
    #: Re-measured 2026-08-21 after the BA fixes, and 8 survives - but far
    #: more sharply than before: 8 -> 85 cameras, while 12, 16 and 20 all
    #: give *the same* 27. Identical results mean growth stalls at the same
    #: place, so somewhere past 27 cameras this capture's only way forward is
    #: an image sharing 8-11 points with the model; cross that bottleneck and
    #: images with 300-1400 inliers become reachable again.
    #:
    #: Treat this as a known fragility rather than a good property. A pose
    #: accepted on 8 correspondences is thin, and the value is low because the
    #: capture demands it, not because it is safe. Registering bottlenecks
    #: with a stricter pose check would be the principled fix.
    pnp_min_inliers: int = 8

    #: Re-derived 2026-08-21 after the bundle-adjustment fixes. The previous
    #: value of 0.5 came from a sweep run while adjustment was not converging
    #: ("2.0 -> 19 images registered, 0.5 -> 65"), which no longer holds: with
    #: BA fixed, a stricter floor gives *more* cameras and much better
    #: geometry, because the near-parallel-ray points it excludes were
    #: actively misleading the adjustment rather than supporting it.
    #:
    #: Measured on the 110-image capture (`run_scratch_ablation.py`):
    #:
    #:     0.50 -> 85 cameras, 2.189 px mean, 1.662 px median, bbox 15.9x
    #:     1.25 -> 88 cameras, 1.721 px
    #:     1.40 -> 89 cameras, 1.785 px
    #:     1.50 -> 87 cameras, 1.578 px mean, 1.135 px median, bbox 7.8x
    #:     2.00 -> 77 cameras, 1.759 px
    #:
    #: 1.25-1.5 is a plateau, not a lucky single value - incremental SfM is
    #: path-dependent, so the neighbours were checked deliberately. 1.5 is
    #: chosen for the best reprojection error within it. It costs roughly
    #: twice the mapping time, because better structure means larger and
    #: better-conditioned adjustment problems.
    min_triangulation_angle_deg: float = 1.5
    #: A stricter parallax requirement applied once, at the end. Growth needs a
    #: permissive threshold to bootstrap - a walking capture offers only 1-3
    #: degrees between neighbouring frames - but a point kept at 0.5 degrees
    #: has a depth uncertainty of roughly sigma/(f*theta), which at one pixel
    #: of feature noise and f=1150 is about 10% of its depth. Requiring more
    #: parallax of the *final* cloud costs nothing during construction.
    #: 0 disables it.
    final_min_triangulation_angle_deg: float = 0.0
    #: Drop final points lying further from the nearest camera than this
    #: multiple of the cloud's *median* viewing distance. Such points are
    #: triangulation failures along near-degenerate rays: they still reproject
    #: correctly in the two or three views that see them, because sliding a
    #: point along its viewing ray is exactly the direction those observations
    #: cannot constrain, so no reprojection test can catch them. 0 disables it.
    #:
    #: The median viewing distance is the right thing to normalise by, and the
    #: camera-path extent is not. Measured:
    #:
    #:     scene_01     median viewing distance 0.13x the path extent
    #:     _selftest3d  median viewing distance 1.18x the path extent
    #:
    #: A walking capture covers far more ground than it sees; a room capture
    #: barely moves while viewing everything. An earlier version of this knob
    #: scaled by path extent, and on _selftest3d dropped 36% of the points in
    #: one calibration sub-model while costing the final model only 3.4%. That
    #: instability - one threshold meaning different things across sub-models
    #: of a single scene - is what makes path extent the wrong reference.
    #:
    #: Chosen to sit just above the 99th percentile of viewing distance, so it
    #: removes the extreme tail without truncating the scene. Measured on the
    #: final scene_01 cloud: p99/median 6.53, max/median 45.3 - a handful of
    #: points sit an order of magnitude beyond everything else, which is the
    #: signature of a failed triangulation rather than of distant structure.
    #:
    #:      3x -> 98.3% kept, bbox 1.79x path extent
    #:      5x -> 98.8% kept, bbox 2.32x
    #:      8x -> 99.1% kept, bbox 2.91x
    #:     15x -> 99.5% kept, bbox 3.96x
    #:
    #: _selftest3d is untouched at every one of these (max/median 1.8), which
    #: is the point of normalising this way.
    #:
    #: There is no knee in the distribution, so this remains a judgement about
    #: how much far field to trust rather than something the data determines.
    #: The sweeps are in docs/HANDOFF.md.
    max_point_distance_median_factor: float = 8.0
    max_reprojection_px: float = 8.0
    #: Solve for radial distortion during global bundle adjustment. Without it
    #: the focal length has to absorb barrel distortion and will not converge.
    #: OFF by default. Radial distortion trades off against focal length and
    #: against structure, so with a weakly observable calibration it absorbs
    #: error instead of correcting it. Measured with the focal fixed at
    #: 1150 px: distortion refinement on gave 39 registered images, off gave
    #: 85. Enable it only with a genuinely calibrated camera.
    refine_distortion: bool = False
    #: Scan for the focal length before reconstructing. Bundle adjustment cannot
    #: recover it on its own (see :meth:`IncrementalMapper.calibrate_focal`), so
    #: without this an uncalibrated capture inherits whatever the prior guessed.
    auto_focal: bool = True
    #: Two registered cameras are treated as connected when they share at
    #: least this many triangulated points. Used to detect a reconstruction
    #: that has silently split into disconnected sub-models.
    min_covisible_points: int = 15
    #: Keep only the largest covisibility component. Disabling this preserves
    #: every camera but leaves the model geometrically inconsistent.
    enforce_single_component: bool = True
    #: Re-solve every camera's pose against the completed point cloud before
    #: the final adjustment. Cameras registered early, or squeezed through a
    #: bottleneck on the minimum number of correspondences, were fitted to a
    #: fraction of the structure that eventually exists, and bundle adjustment
    #: cannot move a pose RANSAC placed in the wrong basin.
    final_pose_refinement: bool = True
    #: Rounds of "adjust, filter a little harder, re-triangulate" applied once
    #: the model is complete. Filtering hard *during* growth starves the
    #: frontier and collapses registration (measured: a 2 px limit gives 22
    #: cameras against 85 at 8 px). At the end there is no frontier left to
    #: starve, so the limit can be walked down without costing cameras: only
    #: points are discarded, and each round gives the adjustment a cleaner
    #: problem than the last. 0 disables.
    final_polish_rounds: int = 3
    #: Reprojection limit the last polish round filters at, in pixels. The
    #: rounds interpolate from ``max_reprojection_px`` down to this.
    #:
    #: Kept well above the model's own error rather than close to it. At 3.0 px
    #: against a model with 1.56 px mean and 1.12 px median, the rounds
    #: discarded 4,012 of 21,444 points - that is legitimate structure, not
    #: outliers, and a lower reported error bought by deleting the evidence is
    #: the same survivorship trap that made the old 1.85 px figure look good.
    final_polish_reprojection_px: float = 5.0
    #: Multiplier applied to the reprojection limit while the model is still
    #: growing; the strict limit is always applied once at the end.
    #:
    #: Measured on a 110-image walking capture. Filtering strictly after every
    #: global adjustment starves the frontier when adjustments are frequent
    #: (global_ba_every=3 gave 20 registered images at slack 1.0, but 64 at
    #: slack 3.0). At the default cadence of 10 the adjustment is rare enough
    #: that slack is unnecessary and slightly hurts accuracy (65 images at
    #: 1.94 px with slack 1.0, versus 64 at 2.55 px with slack 3.0), so the
    #: default is 1.0 and the knob exists for tighter BA schedules.
    growth_filter_slack: float = 1.0
    #: Candidate focals as multiples of the image's long side.
    focal_candidates: Tuple[float, ...] = (0.55, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 1.75, 2.0)
    #: Images used per candidate trial. Must exceed 2 - a two-view
    #: reconstruction fits any focal, so the ambiguity only breaks with three
    #: or more views sharing one calibration.
    #:
    #: This is how many frames the *probe* samples, not how many the capture
    #: contains, and 8 was too few. The focal is strongly observable on the
    #: full set - measured on scene_02, 1.56 px at f=1150 against 2.78 px at
    #: f=1800, with registration halving away from the optimum - yet the
    #: 8-image probe still chose 923. Too small a sample cannot see a signal
    #: that is plainly there in the whole capture.
    #:
    #: Raising it costs one subset reconstruction per candidate, paid once.
    focal_trial_images: int = 24

    local_ba_every: int = 1
    global_ba_every: int = 10
    #: Cameras a *local* adjustment is allowed to move: the most recently
    #: registered ``local_ba_window``. Every other camera observing the same
    #: points is still included, held fixed, so its observations continue to
    #: constrain the structure without being free to drift.
    #:
    #: 0 restores the previous behaviour, in which every "local" adjustment
    #: silently optimised the entire model. That made each of them cost as
    #: much as a global adjustment while sharing the same evaluation budget,
    #: so neither converged. Measured on the 110-image capture: 67% of
    #: adjustments improved the RMSE by less than 1%, and 96% finished above
    #: 5 px.
    local_ba_window: int = 12
    #: Function-evaluation budget handed to scipy's least_squares.
    #:
    #: This is ``max_nfev``, a cap on *residual evaluations*, not on
    #: iterations - a distinction worth keeping in mind when changing it.
    ba_max_iterations: int = 40
    #: A global adjustment solves a far larger problem than a local one and
    #: needs a correspondingly larger budget. Sharing the local budget is
    #: what left the full model unconverged.
    ba_max_iterations_global: int = 200
    refine_focal: bool = True

    seed: int = 0



#: Golden-section iterations used to refine the focal inside the coarse
#: bracket, and the resulting worst-case trial count (two to open the bracket,
#: one per iteration). Only used for progress reporting.
_REFINE_ITERATIONS = 4
_REFINE_TRIALS = _REFINE_ITERATIONS + 2


def _rank_focal_trials(results):
    """Order focal trials best-first: ``(error, focal, views, points)``.

    Two failure modes have to be avoided at once, and they pull in opposite
    directions.

    Ranking by error alone picks a *failure*: a trial that registers only part
    of the subset fits fewer constraints and so reports a lower error. Measured
    on an 8-image probe, plain argmin chose f=1600 (7 of 8 views, 1.67 px) over
    the correct region near f=1120 (8 of 8 views, 1.00 px).

    Ranking by view count first then over-corrects, because a single missing
    view outranks any amount of error. Measured on a 24-image probe, that chose
    f=1360 (24 views, 1.289 px) over f=1120 (23 views, **0.959 px**) - and the
    true focal is about 1150. One view was allowed to outweigh a 26% better fit.

    So candidates within a small tolerance of the best view count are treated
    as equally complete, and the winner among them is the one that fits best.
    The tolerance scales with the probe size: one view out of eight is
    meaningful, one out of twenty-four is not.
    """
    if not results:
        return []
    best_views = max(r[2] for r in results)
    tolerance = max(1, int(round(0.1 * best_views)))
    floor = max(3, best_views - tolerance)
    admissible = [r for r in results if r[2] >= floor]
    return sorted(admissible or results, key=lambda r: r[0])


@dataclass
class Reconstruction:
    """The reconstructed model."""

    poses: Dict[int, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    points3d: Dict[int, np.ndarray] = field(default_factory=dict)
    colors: Dict[int, np.ndarray] = field(default_factory=dict)
    errors: Dict[int, float] = field(default_factory=dict)
    track_lengths: Dict[int, int] = field(default_factory=dict)
    Ks: Dict[int, np.ndarray] = field(default_factory=dict)
    groups: Dict[int, int] = field(default_factory=dict)
    #: Radial distortion (k1, k2) per camera group, solved by bundle adjustment.
    distortion: Dict[int, np.ndarray] = field(default_factory=dict)
    image_names: List[str] = field(default_factory=list)
    registered_order: List[int] = field(default_factory=list)

    @property
    def K(self) -> np.ndarray:
        """Intrinsics of the first registered camera (single-source shorthand)."""
        return self.Ks[sorted(self.Ks)[0]] if self.Ks else np.eye(3)

    @property
    def num_registered(self) -> int:
        return len(self.poses)

    def camera_centers(self) -> Dict[int, np.ndarray]:
        """``C = -R^T t`` for every registered camera."""
        return {i: -R.T @ t for i, (R, t) in self.poses.items()}

    def points_array(self) -> Tuple[np.ndarray, np.ndarray]:
        ids = sorted(self.points3d)
        if not ids:
            return np.zeros((0, 3)), np.zeros((0, 3), np.uint8)
        pts = np.array([self.points3d[i] for i in ids])
        cols = np.array([self.colors.get(i, np.array([200, 200, 200])) for i in ids], np.uint8)
        return pts, cols

    def observation_list(self, tracks: Sequence[Track],
                         features: Sequence[FeatureSet]) -> List[Tuple[int, int, np.ndarray]]:
        out = []
        for tid in self.points3d:
            for img, feat in tracks[tid].observations.items():
                if img in self.poses:
                    out.append((img, tid, features[img].keypoints[feat]))
        return out


class IncrementalMapper:
    """Builds a :class:`Reconstruction` from features and a match graph."""

    def __init__(self, features: Sequence[FeatureSet], K, 
                 config: Optional[MapperConfig] = None,
                 groups: Optional[Sequence[int]] = None,
                 match_cache: Optional[Path] = None) -> None:
        self.features = list(features)
        n = len(self.features)
        # K may be one matrix shared by every image, or one per image when the
        # inputs come from different sources (photographs plus video frames).
        if isinstance(K, (list, tuple)) or (isinstance(K, np.ndarray) and K.ndim == 3):
            self.Ks = [np.asarray(k, dtype=np.float64) for k in K]
        else:
            self.Ks = [np.asarray(K, dtype=np.float64).copy() for _ in range(n)]
        self.groups = list(groups) if groups is not None else [0] * n
        self.cfg = config or MapperConfig()
        self.matches: List[PairMatch] = []
        self.tracks: List[Track] = []
        #: Verified matches are expensive (minutes for 100+ images) and depend
        #: only on the features and matching settings, so they are cached.
        self.match_cache = Path(match_cache) if match_cache else None
        self.rec = Reconstruction(image_names=[f.name for f in self.features])
        # track id -> set of images whose observation is already triangulated
        self._track_of_feature: Dict[Tuple[int, int], int] = {}
        self.stats: Dict[str, float] = {}

    # -- stage 1: matching --------------------------------------------------

    def _cache_key(self) -> tuple:
        return (len(self.features), self.cfg.sequential_window, self.cfg.matcher,
                round(self.cfg.match_ratio, 4), round(self.cfg.match_threshold_px, 4),
                self.cfg.min_pair_inliers,
                # Part of the key: changing the acceptance rule changes which
                # pairs exist, and silently reusing a cache built under the
                # old rule would hide the change entirely.
                round(self.cfg.min_pair_inlier_ratio, 4), self.cfg.min_pair_floor_inliers,
                tuple(f.name for f in self.features[:5]),
                tuple(int(len(f)) for f in self.features[:5]),
                # A fingerprint of the descriptors themselves. Names and
                # keypoint counts do not change when the *descriptor* changes -
                # RootSIFT rewrites every vector while leaving both identical -
                # so without this a cache built from the old descriptors is
                # reused against new ones and the change silently does nothing.
                self._descriptor_fingerprint())

    def _descriptor_fingerprint(self) -> tuple:
        """Cheap, order-stable summary of the descriptors in play."""
        out = []
        for f in self.features[:5]:
            d = f.descriptors
            if d is None or not len(d):
                out.append(0.0)
                continue
            out.append(round(float(np.asarray(d[0], np.float64).sum()), 6))
        return tuple(out)

    def match_all(self) -> List[PairMatch]:
        if self.match_cache and self.match_cache.is_file():
            try:
                with open(self.match_cache, "rb") as fh:
                    payload = pickle.load(fh)
                if payload.get("key") == self._cache_key():
                    self.matches = payload["matches"]
                    self.stats["matching_time"] = 0.0
                    self.stats["verified_pairs"] = len(self.matches)
                    LOGGER.info("Loaded %d cached verified pairs from %s",
                                len(self.matches), self.match_cache.name)
                    return self.matches
            except Exception as exc:
                LOGGER.debug("Match cache unusable (%s); rematching", exc)

        n = len(self.features)
        if self.cfg.sequential_window > 0:
            pairs = [(i, j) for i in range(n)
                     for j in range(i + 1, min(n, i + 1 + self.cfg.sequential_window))]
        else:
            pairs = list(combinations(range(n), 2))

        LOGGER.info("Matching %d image pairs (%s)", len(pairs),
                    "exhaustive" if self.cfg.sequential_window == 0
                    else f"sequential window {self.cfg.sequential_window}")
        # Metric follows the descriptor type, not a setting: AKAZE and ORB
        # are binary and must be compared under Hamming distance.
        binary = bool(self.features) and self.features[0].is_binary
        matcher = make_matcher(self.cfg.matcher, binary=binary)
        start = time.perf_counter()
        out: List[PairMatch] = []
        for i, j in tqdm(pairs, desc="matching", unit="pair"):
            m = match_pair(self.features[i], self.features[j], i, j,
                           self.Ks[i], self.Ks[j],
                           ratio=self.cfg.match_ratio,
                           threshold_px=self.cfg.match_threshold_px,
                           min_inliers=self.cfg.min_pair_inliers,
                           min_inlier_ratio=self.cfg.min_pair_inlier_ratio,
                           floor_inliers=self.cfg.min_pair_floor_inliers,
                           matcher=matcher, seed=self.cfg.seed + i * 977 + j)
            if m is not None:
                out.append(m)
        self.matches = out
        self.stats["matching_time"] = time.perf_counter() - start
        self.stats["verified_pairs"] = len(out)
        LOGGER.info("  %d/%d pairs survived geometric verification in %.1f s",
                    len(out), len(pairs), self.stats["matching_time"])
        if self.match_cache:
            try:
                self.match_cache.parent.mkdir(parents=True, exist_ok=True)
                with open(self.match_cache, "wb") as fh:
                    pickle.dump({"key": self._cache_key(), "matches": out}, fh)
            except Exception as exc:
                LOGGER.debug("Could not write match cache: %s", exc)
        return out

    # -- stage 2: tracks ----------------------------------------------------

    def build_tracks(self) -> List[Track]:
        self.tracks = build_tracks(self.matches, min_length=self.cfg.min_track_length)
        self._track_of_feature = {}
        for tid, track in enumerate(self.tracks):
            for img, feat in track.observations.items():
                self._track_of_feature[(img, feat)] = tid
        lengths = np.array([len(t) for t in self.tracks]) if self.tracks else np.zeros(1)
        self.stats["num_tracks"] = len(self.tracks)
        self.stats["mean_track_length"] = float(lengths.mean())
        LOGGER.info("  %d tracks, mean length %.2f, max %d",
                    len(self.tracks), lengths.mean(), int(lengths.max()))
        return self.tracks

    # -- stage 2b: focal calibration ---------------------------------------

    def calibrate_focal(self) -> Optional[float]:
        """Choose the focal length by scanning candidates on a subset of images.

        Why a scan rather than optimisation: 3D structure is triangulated using
        the current focal, so it is always self-consistent with it. A model
        built around a wrong focal therefore reprojects *its own points*
        acceptably and the optimiser sees no gradient - the error is absorbed by
        a projective deformation of the scene. Measured on the project's
        synthetic scene, a focal 60% too large stayed 60% too large through
        every bundle adjustment.

        The error is nevertheless plainly visible once **three or more** views
        share one calibration, because no single deformation satisfies them all.
        On that same scene the mean reprojection error was 0.28 px at the true
        focal and 0.7-1.4 px at every other candidate - a clean minimum. This
        method rebuilds a small reconstruction per candidate and returns the
        argmin.
        """
        if not self.cfg.auto_focal or not self.matches:
            return None

        long_side = max(self.features[0].width, self.features[0].height)
        # Rank images by how much verified structure they share, and keep a
        # well-connected cluster: a scattered subset would fail to register and
        # score every candidate equally badly.
        degree: Dict[int, int] = {}
        for m in self.matches:
            degree[m.i] = degree.get(m.i, 0) + m.num_inliers
            degree[m.j] = degree.get(m.j, 0) + m.num_inliers
        if not degree:
            return None
        hub = max(degree, key=lambda k: degree[k])
        connected = sorted({(m.j if m.i == hub else m.i)
                            for m in self.matches if hub in (m.i, m.j)} | {hub})
        # Spread the trial subset across the connected set instead of taking the
        # strongest links. The strongest links are adjacent, near-duplicate
        # frames whose baselines are tiny - and focal length is unobservable
        # without parallax, so that subset scores every candidate about equally
        # and the argmin becomes noise. Even spacing buys real baseline while
        # staying inside one connected component.
        if len(connected) > self.cfg.focal_trial_images:
            picks = np.linspace(0, len(connected) - 1,
                                self.cfg.focal_trial_images).astype(int)
            subset = [connected[k] for k in sorted(set(picks.tolist()))]
        else:
            subset = list(connected)
        if hub not in subset:
            subset.append(hub)
        subset = sorted(set(subset))
        if len(subset) < 3:
            return None
        subset_set = set(subset)
        remap = {old: new for new, old in enumerate(subset)}
        sub_features = [self.features[i] for i in subset]
        sub_matches = [
            PairMatch(i=remap[m.i], j=remap[m.j], idx_i=m.idx_i, idx_j=m.idx_j,
                      num_raw=m.num_raw, num_ratio=m.num_ratio,
                      num_inliers=m.num_inliers, F=m.F)
            for m in self.matches if m.i in subset_set and m.j in subset_set
        ]
        if len(sub_matches) < 3:
            return None

        LOGGER.info("Calibrating focal on %d well-connected images (%d pairs), "
                    "%d candidates", len(subset), len(sub_matches),
                    len(self.cfg.focal_candidates))

        def trial_error(focal: float):
            """Reconstruct the subset at one focal; return (error, views, points)."""
            # The probe deliberately uses STRICT geometric thresholds, not the
            # permissive ones the main reconstruction runs with. The two have
            # opposite goals: the reconstruction wants to tolerate marginal
            # geometry so it can register as much of the capture as possible,
            # whereas the probe needs a wrong focal to *fail visibly*. With the
            # permissive thresholds every candidate registered all 8 trial
            # views and the error curve became noise (2.96 px at f=880 vs
            # 3.19 px at f=1120); with strict ones the curve has a clean
            # minimum and the selected focal agrees with an independent
            # implementation to within 3%.
            trial_cfg = MapperConfig(
                **{**self.cfg.__dict__,
                   "auto_focal": False, "refine_focal": False,
                   "refine_distortion": False, "global_ba_every": 3,
                   "ba_max_iterations": 25,
                   "min_triangulation_angle_deg": 2.0,
                   "pnp_min_inliers": 20,
                   "pnp_threshold_px": 4.0,
                   "max_reprojection_px": 4.0,
                   "growth_filter_slack": 1.0})
            K = np.array([[focal, 0.0, sub_features[0].width / 2.0],
                          [0.0, focal, sub_features[0].height / 2.0],
                          [0.0, 0.0, 1.0]])
            trial = IncrementalMapper(sub_features, K, trial_cfg)
            trial.matches = sub_matches
            try:
                rec = trial.run()
            except Exception as exc:
                LOGGER.debug("  f=%.0f failed: %s", focal, exc)
                return None
            if rec.num_registered < 3 or len(rec.points3d) < 50:
                return None
            return (float(np.mean(list(rec.errors.values()))),
                    rec.num_registered, len(rec.points3d))

        results = []
        for factor in self.cfg.focal_candidates:
            focal = factor * long_side
            out = trial_error(focal)
            if out is None:
                continue
            results.append((out[0], focal, out[1], out[2]))
            LOGGER.info("    f=%7.1f  reproj %.4f px  %d views  %d points",
                        focal, out[0], out[1], out[2])

        # Refine around the coarse winner. The grid spacing is ~15% of the
        # focal, which is far coarser than the minimum is sharp, so the coarse
        # argmin is only a bracket - a golden-section search inside it recovers
        # the rest.
        if results:
            best_focal = _rank_focal_trials(results)[0][1]
            factors = sorted(self.cfg.focal_candidates)
            centre = best_focal / long_side
            pos = factors.index(min(factors, key=lambda f: abs(f - centre)))
            lo = factors[max(0, pos - 1)] * long_side
            hi = factors[min(len(factors) - 1, pos + 1)] * long_side
            if hi > lo:
                phi = (np.sqrt(5.0) - 1.0) / 2.0
                a, b = lo, hi
                c, d = b - phi * (b - a), a + phi * (b - a)

                # Each refinement trial rebuilds the subset and takes as long
                # as a coarse candidate, so report them as they finish. Logging
                # only at the end left this stage silent for 15-30 minutes,
                # which is indistinguishable from a hang.
                refine_step = 0

                def probe(focal: float):
                    nonlocal refine_step
                    refine_step += 1
                    out = trial_error(focal)
                    if out is None:
                        LOGGER.info("    refine %d/%d  f=%7.1f  no usable reconstruction",
                                    refine_step, _REFINE_TRIALS, focal)
                    else:
                        LOGGER.info("    refine %d/%d  f=%7.1f  reproj %.4f px  "
                                    "%d views  %d points",
                                    refine_step, _REFINE_TRIALS, focal,
                                    out[0], out[1], out[2])
                    return out

                LOGGER.info("  refining within [%.1f, %.1f] px, %d trials",
                            lo, hi, _REFINE_TRIALS)
                fc = probe(c)
                fd = probe(d)
                # Same rule inside the golden-section search: a meaningfully
                # less complete reconstruction loses, otherwise the better fit
                # wins. ``out`` is (error, views, points).
                def better(a, b):
                    if a is None:
                        return False
                    if b is None:
                        return True
                    tol = max(1, int(round(0.1 * max(a[1], b[1]))))
                    if a[1] < b[1] - tol:
                        return False
                    if b[1] < a[1] - tol:
                        return True
                    return a[0] < b[0]

                for _ in range(_REFINE_ITERATIONS):
                    if fc is None or fd is None:
                        break
                    if better(fc, fd):
                        b, d, fd = d, c, fc
                        c = b - phi * (b - a)
                        fc = probe(c)
                    else:
                        a, c, fc = c, d, fd
                        d = a + phi * (b - a)
                        fd = probe(d)
                for focal, out in ((c, fc), (d, fd)):
                    if out is not None:
                        results.append((out[0], focal, out[1], out[2]))

        if not results:
            LOGGER.warning("Focal calibration produced no usable trial; keeping the prior")
            return None
        best_error, best_focal, views, points = _rank_focal_trials(results)[0]
        LOGGER.info("  selected f=%.1f px (%.4f px reprojection over %d views, %d points)",
                    best_focal, best_error, views, points)
        for i in range(len(self.Ks)):
            self.Ks[i] = np.array([[best_focal, 0.0, self.features[i].width / 2.0],
                                   [0.0, best_focal, self.features[i].height / 2.0],
                                   [0.0, 0.0, 1.0]])
        self.stats["calibrated_focal"] = best_focal
        self.stats["focal_trial_error"] = best_error
        return best_focal

    # -- stage 3: seed ------------------------------------------------------

    def _pair_parallax(self, m: PairMatch) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
        """Median triangulation angle for a verified pair, plus its pose."""
        pts_i = self.features[m.i].keypoints[m.idx_i]
        pts_j = self.features[m.j].keypoints[m.idx_j]
        if len(pts_i) < 8:
            return 0.0, None, None
        # Matches were verified with F; the pose needs E, which is estimated
        # here from the already-clean inliers using the current intrinsics.
        res = ransac_essential(pts_i, pts_j, self.Ks[m.i], self.Ks[m.j],
                               threshold_px=self.cfg.match_threshold_px,
                               seed=self.cfg.seed + m.i)
        if res.model is None:
            return 0.0, None, None
        R, t, mask, X = select_pose_by_cheirality(decompose_essential(res.model), pts_i, pts_j,
                                                  self.Ks[m.i], K2=self.Ks[m.j])
        if R is None or X is None or mask.sum() < self.cfg.min_pair_inliers:
            return 0.0, None, None
        C2 = -R.T @ t
        angles = triangulation_angle_deg(X[mask], np.zeros(3), C2)
        return float(np.median(angles)) if len(angles) else 0.0, R, t

    def select_seed(self) -> Optional[Tuple[PairMatch, np.ndarray, np.ndarray]]:
        """Choose the initial pair: many inliers AND wide parallax.

        Scoring on inlier count alone reliably picks a near-duplicate pair,
        whose tiny baseline yields depths that are numerically meaningless.
        """
        if not self.matches:
            return None
        candidates = sorted(self.matches, key=lambda m: -m.num_inliers)[: self.cfg.seed_candidates]
        best = None
        best_score = -1.0
        evaluated = []
        for m in candidates:
            angle, R, t = self._pair_parallax(m)
            if R is None:
                continue
            evaluated.append((angle, m, R, t))
            if angle < self.cfg.seed_min_parallax_deg:
                continue
            # Reward inliers, but saturate parallax: beyond ~15 degrees extra
            # angle costs overlap without improving conditioning much.
            score = m.num_inliers * min(angle, 15.0)
            if score > best_score:
                best_score, best = score, (m, R, t)
        if best is None:
            # Fall back to the WIDEST BASELINE seen, never to the most inliers.
            # Ranking candidates by inlier count is what created this situation:
            # in a burst capture the highest-inlier pairs are near-duplicates
            # shot from the same spot, so they top the list precisely because
            # they have no baseline. Measured on a 146-image burst (~2 frames a
            # second), the old fallback chose a pair with 2130 inliers and
            # 0.051 degrees of parallax, which triangulated zero points and
            # aborted the reconstruction.
            if not evaluated:
                return None
            angle, m, R, t = max(evaluated, key=lambda e: e[0])
            LOGGER.warning("No pair met the seed parallax requirement (%.1f deg); "
                           "using the widest baseline available instead: "
                           "%s <-> %s at %.2f deg",
                           self.cfg.seed_min_parallax_deg,
                           self.features[m.i].name, self.features[m.j].name, angle)
            if angle < self.cfg.min_triangulation_angle_deg:
                LOGGER.error(
                    "The widest seed baseline in the top %d pairs is only %.3f deg, "
                    "below the %.2f deg needed to triangulate. Every strong pair in "
                    "this capture is a near-duplicate; raise seed_candidates so "
                    "wider-baseline pairs are considered, or shoot with more "
                    "movement between frames.",
                    self.cfg.seed_candidates, angle,
                    self.cfg.min_triangulation_angle_deg)
            best = (m, R, t)
        LOGGER.info("Seed pair: %s <-> %s (%d inliers)",
                    self.features[best[0].i].name, self.features[best[0].j].name,
                    best[0].num_inliers)
        return best

    # -- stage 4: triangulation --------------------------------------------

    def _dist(self, image: int) -> Optional[np.ndarray]:
        """Distortion coefficients for an image's camera group, if solved."""
        return self.rec.distortion.get(self.groups[image])

    def _projection(self, image: int) -> np.ndarray:
        R, t = self.rec.poses[image]
        return self.rec.Ks[image] @ np.hstack([R, t.reshape(3, 1)])

    def _triangulate_track(self, tid: int) -> bool:
        """Triangulate one track from every registered view that observes it."""
        track = self.tracks[tid]
        views = [(img, feat) for img, feat in track.observations.items() if img in self.rec.poses]
        if len(views) < 2:
            return False

        projections = [self._projection(img) for img, _ in views]
        obs = []
        for img, feat in views:
            xy = self.features[img].keypoints[feat]
            dist = self._dist(img)
            if dist is not None and np.any(dist):
                xy = undistort_points(xy[None, :], self.rec.Ks[img], dist)[0]
            obs.append(xy)
        X = triangulate_multiview(projections, obs)
        if X is None or not np.all(np.isfinite(X)):
            return False

        # Cheirality in every observing view.
        for img, _ in views:
            R, t = self.rec.poses[img]
            if (R @ X + t)[2] <= 1e-9:
                return False

        # Parallax: the widest angle any pair of observing cameras subtends.
        centers = [(-self.rec.poses[img][0].T @ self.rec.poses[img][1]) for img, _ in views]
        best_angle = 0.0
        for a in range(len(centers)):
            for b in range(a + 1, len(centers)):
                ang = triangulation_angle_deg(X[None, :], centers[a], centers[b])[0]
                best_angle = max(best_angle, ang)
                if best_angle >= self.cfg.min_triangulation_angle_deg:
                    break
            if best_angle >= self.cfg.min_triangulation_angle_deg:
                break
        if best_angle < self.cfg.min_triangulation_angle_deg:
            return False

        errors = []
        for (img, feat) in views:
            R, t = self.rec.poses[img]
            errors.append(reprojection_error(X[None, :], self.features[img].keypoints[feat][None, :],
                                             self.rec.Ks[img], R, t, self._dist(img))[0])
        mean_error = float(np.mean(errors))
        if not np.isfinite(mean_error) or mean_error > self.cfg.max_reprojection_px:
            return False

        self.rec.points3d[tid] = X
        self.rec.errors[tid] = mean_error
        self.rec.track_lengths[tid] = len(views)
        img0, feat0 = views[0]
        self.rec.colors[tid] = self.features[img0].colors[feat0]
        return True

    def _triangulate_all(self) -> int:
        added = 0
        for tid in range(len(self.tracks)):
            registered = sum(1 for img in self.tracks[tid].observations if img in self.rec.poses)
            if registered < 2:
                continue
            if tid in self.rec.points3d and self.rec.track_lengths.get(tid, 0) >= registered:
                continue
            if self._triangulate_track(tid):
                added += 1
        return added

    # -- stage 5: growing ---------------------------------------------------

    def _correspondences_for(self, image: int) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """2D-3D correspondences between an unregistered image and the model."""
        pts3d, pts2d, tids = [], [], []
        for feat_idx in range(len(self.features[image])):
            tid = self._track_of_feature.get((image, feat_idx))
            if tid is None or tid not in self.rec.points3d:
                continue
            pts3d.append(self.rec.points3d[tid])
            pts2d.append(self.features[image].keypoints[feat_idx])
            tids.append(tid)
        if not pts3d:
            return np.zeros((0, 3)), np.zeros((0, 2)), []
        return np.array(pts3d), np.array(pts2d), tids

    def _next_image(self, remaining: Set[int]) -> Optional[Tuple[int, int]]:
        best, best_count = None, 0
        for image in remaining:
            _, _, tids = self._correspondences_for(image)
            if len(tids) > best_count:
                best, best_count = image, len(tids)
        if best is None or best_count < self.cfg.pnp_min_inliers:
            return None
        return best, best_count

    def _local_ba_sets(self) -> Optional[Tuple[Set[int], Set[int], Set[int]]]:
        """Cameras and points for a local adjustment: ``(free, fixed, points)``.

        The free cameras are the most recently registered ones. The points are
        those they observe, and the fixed cameras are every *other* registered
        camera observing those same points. Including the fixed neighbours is
        what anchors the window to the part of the model already solved: drop
        them and the window is free to slide, which is worse than not adjusting
        at all.
        """
        window = self.cfg.local_ba_window
        if window <= 0 or len(self.rec.poses) <= window:
            return None

        free = set(self.rec.registered_order[-window:]) & set(self.rec.poses)
        if len(free) < 2:
            return None

        points = {tid for tid in self.rec.points3d
                  if any(img in free for img in self.tracks[tid].observations)}
        if not points:
            return None

        involved: Set[int] = set()
        for tid in points:
            involved |= {img for img in self.tracks[tid].observations
                         if img in self.rec.poses}
        fixed = involved - free
        if not fixed:
            # Nothing anchors the window, so fall back to fixing the gauge on
            # the first registered camera as a global adjustment would.
            fixed = {self.rec.registered_order[0]}
            free -= fixed
            if len(free) < 2:
                return None
        return free, fixed, points

    def _run_ba(self, global_ba: bool) -> None:
        sets = None if global_ba else self._local_ba_sets()

        if sets is None:
            poses_in = self.rec.poses
            points_in = self.rec.points3d
            observations = self.rec.observation_list(self.tracks, self.features)
            fixed = {self.rec.registered_order[0]} if self.rec.registered_order else set()
            budget = (self.cfg.ba_max_iterations_global if global_ba
                      else self.cfg.ba_max_iterations)
        else:
            free, fixed, point_ids = sets
            cameras = free | fixed
            poses_in = {i: self.rec.poses[i] for i in cameras}
            points_in = {p: self.rec.points3d[p] for p in point_ids}
            observations = [
                (img, tid, self.features[img].keypoints[feat])
                for tid in point_ids
                for img, feat in self.tracks[tid].observations.items()
                if img in cameras
            ]
            budget = self.cfg.ba_max_iterations

        if len(observations) < 20:
            return
        poses, points, Ks, report = bundle_adjust(
            poses_in, points_in, observations,
            {i: self.rec.Ks[i] for i in poses_in},
            groups={i: self.groups[i] for i in poses_in},
            fixed_cameras=fixed,
            # Refine the focal in EVERY adjustment, not only global ones.
            # Structure is triangulated using the current focal, so it is always
            # self-consistent with it; once a large model has been built around
            # a wrong focal there is no gradient left to correct it. The focal
            # must therefore be free from the seed pair onward, while the
            # structure is still small enough to follow it.
            # Focal and distortion are shared across a whole camera group, so a
            # local window must not touch them: it would fit them to its own
            # few cameras and then write the result back over every camera in
            # the group. Calibration is only ever refined globally.
            refine_focal=self.cfg.refine_focal and sets is None,
            refine_distortion=self.cfg.refine_distortion and global_ba,
            distortion=self.rec.distortion,
            max_iterations=budget,
        )
        # A local adjustment returns only the cameras and points it was given,
        # so the results are merged in. Assigning them wholesale would delete
        # the rest of the model.
        self.rec.poses.update(poses)
        self.rec.points3d.update(points)
        self.rec.Ks.update(Ks)
        if report.distortion:
            self.rec.distortion.update(report.distortion)

    def _filter_points(self, threshold: Optional[float] = None) -> int:
        """Drop points that no longer reproject well after adjustment.

        ``threshold`` defaults to the configured limit. During growth a looser
        value is used deliberately: filtering hard on a half-built model
        removes points the frontier still needs, and the reconstruction starves
        itself. Measured on this project's scene, filtering strictly after every
        global adjustment cut the final registration from 65 images to 20. The
        strict threshold is applied once at the end instead.
        """
        limit = self.cfg.max_reprojection_px if threshold is None else threshold
        removed = []
        for tid, X in self.rec.points3d.items():
            views = [(img, feat) for img, feat in self.tracks[tid].observations.items()
                     if img in self.rec.poses]
            if len(views) < 2:
                removed.append(tid)
                continue
            errs = []
            behind = False
            for img, feat in views:
                R, t = self.rec.poses[img]
                if (R @ X + t)[2] <= 1e-9:
                    behind = True
                    break
                errs.append(reprojection_error(X[None, :],
                                               self.features[img].keypoints[feat][None, :],
                                               self.rec.Ks[img], R, t, self._dist(img))[0])
            if behind or not errs:
                removed.append(tid)
                continue
            mean_error = float(np.mean(errs))
            if not np.isfinite(mean_error) or mean_error > limit:
                removed.append(tid)
            else:
                self.rec.errors[tid] = mean_error
                self.rec.track_lengths[tid] = len(views)
        for tid in removed:
            self.rec.points3d.pop(tid, None)
            self.rec.errors.pop(tid, None)
            self.rec.track_lengths.pop(tid, None)
            self.rec.colors.pop(tid, None)
        return len(removed)

    def _is_pose_consistent(self, cameras: Set[int], sample: int = 30,
                            tolerance_deg: float = 12.0,
                            min_agreement: float = 0.6) -> bool:
        """Do the images themselves agree with where the model put these cameras?

        Replaces a spacing heuristic that could not tell a corrupted model from
        an ordinary capture gap. For a sample of the strongest verified pairs
        inside the set, the relative rotation is recovered from the pair's own
        correspondences and compared with the relative rotation the model
        implies. Rotation only: two-view translation is known up to scale and
        its direction is poorly conditioned on short baselines.

        The distinction this draws is the one that matters. Two unrelated parts
        of a capture wrongly fused by repeated structure are bridged by pairs
        whose own geometry *contradicts* the model, and that shows up here. A
        photographer who stopped shooting and walked twenty metres leaves no
        bridging pairs to contradict anything - the cameras are simply far
        apart, which is not evidence of anything being wrong.
        """
        inside = [m for m in self.matches
                  if m.i in cameras and m.j in cameras
                  and m.i in self.rec.poses and m.j in self.rec.poses]
        if len(inside) < 5:
            return True                      # nothing to contradict the model
        inside.sort(key=lambda m: -m.num_inliers)
        checked = agreed = 0
        for m in inside[:sample]:
            pts1 = self.features[m.i].keypoints[m.idx_i]
            pts2 = self.features[m.j].keypoints[m.idx_j]
            if len(pts1) < 8:
                continue
            res = ransac_essential(pts1, pts2, self.rec.Ks[m.i], self.rec.Ks[m.j],
                                   threshold_px=self.cfg.match_threshold_px,
                                   seed=self.cfg.seed + m.i * 977 + m.j)
            if res.model is None or res.num_inliers < 8:
                continue
            inl = res.inliers
            R_pair, _, mask, _ = select_pose_by_cheirality(
                decompose_essential(res.model), pts1[inl], pts2[inl],
                self.rec.Ks[m.i], K2=self.rec.Ks[m.j])
            if R_pair is None or int(mask.sum()) < 8:
                continue
            R_i, _ = self.rec.poses[m.i]
            R_j, _ = self.rec.poses[m.j]
            checked += 1
            if rotation_angle_deg((R_j @ R_i.T).T @ R_pair) <= tolerance_deg:
                agreed += 1
        if checked < 5:
            return True
        fraction = agreed / checked
        LOGGER.info("  pose consistency over %d verified pairs: %d agree (%.0f%%)",
                    checked, agreed, 100 * fraction)
        return fraction >= min_agreement

    def _covisibility_components(self, min_shared: Optional[int] = None) -> List[Set[int]]:
        """Connected components of registered cameras, joined by shared points.

        Two cameras are adjacent when they both observe at least
        ``min_covisible_points`` of the same triangulated points. Components are
        returned largest first. More than one component means the reconstruction
        is not a single rigid scene, however good its reprojection error looks.
        """
        threshold = self.cfg.min_covisible_points if min_shared is None else min_shared
        shared: Dict[Tuple[int, int], int] = {}
        for tid in self.rec.points3d:
            observers = sorted(i for i in self.tracks[tid].observations
                               if i in self.rec.poses)
            for a in range(len(observers)):
                for b in range(a + 1, len(observers)):
                    key = (observers[a], observers[b])
                    shared[key] = shared.get(key, 0) + 1

        adjacency: Dict[int, Set[int]] = {i: set() for i in self.rec.poses}
        for (a, b), count in shared.items():
            if count >= threshold:
                adjacency[a].add(b)
                adjacency[b].add(a)

        components: List[Set[int]] = []
        unvisited = set(self.rec.poses)
        while unvisited:
            start = unvisited.pop()
            component = {start}
            stack = [start]
            while stack:
                node = stack.pop()
                for neighbour in adjacency[node]:
                    if neighbour not in component:
                        component.add(neighbour)
                        unvisited.discard(neighbour)
                        stack.append(neighbour)
            components.append(component)
        components.sort(key=len, reverse=True)
        return components

    def _keep_largest_component(self) -> int:
        """Discard cameras outside the largest covisibility component.

        Merging the components instead would need a similarity transform
        estimated from the handful of points they share - and those points are
        precisely the suspect ones, since a genuine shared observation would
        have kept the components joined in the first place. Keeping the largest
        component yields a smaller but geometrically valid model, which is worth
        more than a larger inconsistent one.
        """
        # Escalate the connectivity requirement only when the images themselves
        # contradict the model. A few false matches from repeated structure can
        # bridge two unrelated parts of a capture, and at a low threshold the
        # graph calls that one component even though it is two scenes.
        #
        # The test used to be spacing-based - single-linkage on camera centres -
        # which cannot distinguish that from a photographer who stopped
        # shooting and walked. Measured twice on this capture, it declared
        # sound models split and escalated until they shattered: 85 cameras
        # culled to 41 before the bundle-adjustment fix, and 87 culled to 38
        # after RootSIFT changed the camera spacing. Both models were fine;
        # this capture simply has six consecutive frame pairs with no verified
        # match at all, which is a capture gap, not a defect.
        threshold = self.cfg.min_covisible_points
        components = self._covisibility_components(threshold)
        escalations = 0
        while (components and not self._is_pose_consistent(components[0])
               and escalations < 6):
            threshold *= 2
            escalations += 1
            components = self._covisibility_components(threshold)
            LOGGER.info("  largest component contradicts its own image pairs; "
                        "requiring %d shared points -> components %s",
                        threshold, [len(c) for c in components[:4]])

        self.stats["covisibility_components"] = len(components)
        self.stats["component_sizes"] = [len(c) for c in components]
        self.stats["covisibility_threshold"] = threshold
        if len(components) <= 1:
            return 0

        keep = components[0]
        drop = [i for i in list(self.rec.poses) if i not in keep]
        LOGGER.warning(
            "Reconstruction split into %d covisibility components %s - the capture did "
            "not form one connected scene. Keeping the largest (%d cameras) and dropping "
            "%d; reprojection error alone cannot detect this.",
            len(components), [len(c) for c in components], len(keep), len(drop))

        for image in drop:
            self.rec.poses.pop(image, None)
            if image in self.rec.registered_order:
                self.rec.registered_order.remove(image)
        # Points supported only by discarded cameras go with them.
        self._filter_points()
        return len(drop)

    def point_parallax(self) -> Dict[int, float]:
        """Widest angle any pair of observing cameras subtends at each point.

        This is the quantity that decides how well a point's *depth* is
        determined. Reprojection error does not measure it: a point pushed far
        along a near-degenerate ray still reprojects correctly in every view
        that sees it, because moving along the ray is exactly the direction
        the observations cannot constrain.
        """
        centres = self.rec.camera_centers()
        out: Dict[int, float] = {}
        for tid, X in self.rec.points3d.items():
            obs = [i for i in self.tracks[tid].observations if i in self.rec.poses]
            best = 0.0
            for a in range(len(obs)):
                for b in range(a + 1, len(obs)):
                    ang = triangulation_angle_deg(
                        X[None, :], centres[obs[a]], centres[obs[b]])[0]
                    if ang > best:
                        best = ang
            out[tid] = best
        return out

    def _drop_low_parallax(self, min_angle_deg: float) -> int:
        """Remove final points whose widest observing angle is too small."""
        if min_angle_deg <= 0 or not self.rec.points3d:
            return 0
        parallax = self.point_parallax()
        doomed = [tid for tid, ang in parallax.items() if ang < min_angle_deg]
        for tid in doomed:
            self.rec.points3d.pop(tid, None)
            self.rec.errors.pop(tid, None)
            self.rec.track_lengths.pop(tid, None)
            self.rec.colors.pop(tid, None)
        if doomed:
            LOGGER.info("  parallax filter (>= %.2f deg): dropped %d of %d points",
                        min_angle_deg, len(doomed), len(parallax))
        return len(doomed)

    def _drop_distant_points(self, factor: float) -> int:
        """Remove final points implausibly far from every camera.

        The scale reference is the cloud's own median viewing distance, so the
        threshold means the same thing whether the cameras walked a long way
        past nearby structure or barely moved while viewing a whole room.
        """
        if factor <= 0 or len(self.rec.poses) < 2 or not self.rec.points3d:
            return 0
        centres = np.array(list(self.rec.camera_centers().values()))
        tids = list(self.rec.points3d)
        pts = np.array([self.rec.points3d[t] for t in tids])
        # Distance to the nearest camera, not to the centroid: a scene captured
        # along a path is not a ball around one point.
        d = np.linalg.norm(pts[:, None, :] - centres[None, :, :], axis=2).min(axis=1)
        median = float(np.median(d))
        if not np.isfinite(median) or median <= 1e-9:
            return 0
        limit = factor * median
        doomed = [t for t, dist in zip(tids, d) if dist > limit]
        for tid in doomed:
            self.rec.points3d.pop(tid, None)
            self.rec.errors.pop(tid, None)
            self.rec.track_lengths.pop(tid, None)
            self.rec.colors.pop(tid, None)
        if doomed:
            LOGGER.info("  distance filter (<= %.1fx median viewing distance %.3f): "
                        "dropped %d of %d points",
                        factor, median, len(doomed), len(tids))
        return len(doomed)

    def _polish(self) -> None:
        """Adjust and filter repeatedly, tightening the limit each round.

        A single filter pass at a strict threshold throws away points that
        would have survived had the adjustment been given a chance to fix them
        first. Walking the limit down interleaves the two, so each round only
        discards what the previous adjustment could not explain.
        """
        rounds = self.cfg.final_polish_rounds
        if rounds <= 0:
            return
        start = self.cfg.max_reprojection_px
        end = min(self.cfg.final_polish_reprojection_px, start)
        for k in range(rounds):
            frac = (k + 1) / rounds
            limit = start + (end - start) * frac
            self._run_ba(global_ba=True)
            removed = self._filter_points(limit)
            self._triangulate_all()
            LOGGER.info("  polish round %d/%d at %.2f px: dropped %d points, "
                        "%d remain", k + 1, rounds, limit, removed,
                        len(self.rec.points3d))
        self._run_ba(global_ba=True)
        self._filter_points(end)

    def _refine_all_poses(self) -> int:
        """Re-solve every registered camera against the *final* structure.

        Incremental SfM registers each image against whatever model existed at
        the time, so the earliest and the hardest images get the worst
        structure to work with. An image squeezed in at a bottleneck on eight
        correspondences is fitted to a fraction of the points that will
        eventually exist, and nothing later revisits that decision: bundle
        adjustment refines a pose but cannot rescue one that RANSAC placed in
        the wrong basin.

        This pass gives every camera a second attempt with the completed point
        cloud, and keeps the new pose only when it is supported by strictly
        more inliers than the one it replaces. Measured on this capture, three
        cameras (frames 53, 54 and 62 - all adjacent to image pairs with zero
        verified inliers) sat 4.5-15.7 median-steps outside the camera path
        while every other camera sat within 1.3.
        """
        if not self.rec.points3d:
            return 0
        improved = 0
        for image in sorted(self.rec.poses):
            pts3d, pts2d, _ = self._correspondences_for(image)
            if len(pts3d) < self.cfg.pnp_min_inliers:
                continue
            K = self.rec.Ks[image]
            dist = self._dist(image)
            obs = pts2d
            if dist is not None and np.any(dist):
                obs = undistort_points(pts2d, K, dist)

            R_old, t_old = self.rec.poses[image]
            before = int((reprojection_error(pts3d, pts2d, K, R_old, t_old, dist)
                          < self.cfg.pnp_threshold_px).sum())

            R_new, t_new, inliers = ransac_pnp(
                pts3d, obs, K, threshold_px=self.cfg.pnp_threshold_px,
                seed=self.cfg.seed + image)
            if R_new is None:
                continue
            after = int(inliers.sum())
            # Strictly better only: equal support means the existing pose is
            # already as good, and swapping it would just add RANSAC noise.
            if after > before and after >= self.cfg.pnp_min_inliers:
                self.rec.poses[image] = (R_new, t_new)
                improved += 1
        if improved:
            LOGGER.info("  pose refinement: re-solved %d of %d cameras against the "
                        "final structure", improved, len(self.rec.poses))
        return improved

    def _drop_bad_cameras(self, error_factor: float = 3.0) -> int:
        """Remove cameras whose observations reproject far worse than typical.

        Incremental SfM accumulates drift, and a handful of poorly constrained
        views can end up badly misplaced. They are individually plausible - each
        passed the PnP inlier test when it was registered - but they distort the
        model around them and stretch its bounding box. On this project's
        capture the median inter-camera step was 1.3 units while the worst was
        34, i.e. one view sat far outside the actual camera path.

        A camera is dropped when its mean reprojection error exceeds
        ``error_factor`` times the median across all cameras. Its points are
        left in place and re-filtered afterwards, so structure supported by
        other views survives.
        """
        if len(self.rec.poses) < 6:
            return 0
        per_camera: Dict[int, List[float]] = {}
        for tid, X in self.rec.points3d.items():
            for img, feat in self.tracks[tid].observations.items():
                if img not in self.rec.poses:
                    continue
                R, t = self.rec.poses[img]
                err = reprojection_error(X[None, :],
                                         self.features[img].keypoints[feat][None, :],
                                         self.rec.Ks[img], R, t, self._dist(img))[0]
                if np.isfinite(err):
                    per_camera.setdefault(img, []).append(float(err))

        means = {img: float(np.mean(v)) for img, v in per_camera.items() if v}
        if len(means) < 6:
            return 0
        median = float(np.median(list(means.values())))
        limit = median * error_factor
        # Never drop the reference camera: it defines the gauge.
        reference = self.rec.registered_order[0] if self.rec.registered_order else None
        bad = [img for img, e in means.items() if e > limit and img != reference]
        # Views that contribute no surviving structure are also useless.
        bad += [img for img in self.rec.poses
                if img not in means and img != reference]

        for img in set(bad):
            self.rec.poses.pop(img, None)
            if img in self.rec.registered_order:
                self.rec.registered_order.remove(img)
        if bad:
            LOGGER.info("Dropped %d camera(s) with mean reprojection error above "
                        "%.2f px (median %.2f px)", len(set(bad)), limit, median)
        return len(set(bad))

    # -- driver -------------------------------------------------------------

    def run(self) -> Reconstruction:
        overall = time.perf_counter()
        if not self.matches:
            self.match_all()
        if not self.tracks:
            self.build_tracks()

        if self.cfg.auto_focal:
            if self.calibrate_focal() is not None:
                # Freeze the focal once the scan has found it. Bundle
                # adjustment cannot determine focal length reliably - structure
                # is triangulated with the current value and is always
                # self-consistent with it - so leaving it free lets it drift
                # away from the scanned optimum and drags registration down
                # with it. Measured: scan found 1176.7 px and 65 registered
                # images with the focal frozen; leaving BA free to move it gave
                # 1543.3 px and only 36 images.
                self.cfg.refine_focal = False
                self.cfg.refine_distortion = False

        seed = self.select_seed()
        if seed is None:
            raise RuntimeError(
                "No usable seed pair. Every verified pair either had too few inliers or "
                "too little parallax - the capture likely lacks translation between views."
            )
        m, R, t = seed
        # Register every image's intrinsics up front so the reconstruction can
        # look them up by index regardless of registration order.
        self.rec.Ks = {i: self.Ks[i].copy() for i in range(len(self.features))}
        self.rec.groups = {i: self.groups[i] for i in range(len(self.features))}
        self.rec.poses[m.i] = (np.eye(3), np.zeros(3))
        self.rec.poses[m.j] = (R, t)
        self.rec.registered_order = [m.i, m.j]

        added = self._triangulate_all()
        LOGGER.info("Seed reconstruction: %d points from %d tracks", added, len(self.tracks))
        if added < 20:
            raise RuntimeError(f"Seed pair triangulated only {added} points; aborting.")
        self._run_ba(global_ba=False)

        remaining = set(range(len(self.features))) - set(self.rec.poses)
        failed: Set[int] = set()
        step = 0
        retries = 0
        progress_since_retry = True
        start_grow = time.perf_counter()

        with tqdm(total=len(remaining), desc="registering", unit="img") as bar:
            while remaining:
                choice = self._next_image(remaining - failed)
                if choice is None:
                    # An image that could not be registered against a small
                    # model may register easily once the model has grown and
                    # covers more of the scene. Blacklisting permanently is a
                    # classic way to end up with a fraction of the capture
                    # reconstructed, so failures are retried whenever the model
                    # has advanced since they were recorded.
                    if failed and progress_since_retry:
                        retries += 1
                        LOGGER.info("Retry pass %d: re-examining %d image(s) against a model "
                                    "of %d cameras and %d points",
                                    retries, len(failed), self.rec.num_registered,
                                    len(self.rec.points3d))
                        failed.clear()
                        progress_since_retry = False
                        continue
                    break
                image, n_corr = choice
                pts3d, pts2d, tids = self._correspondences_for(image)
                # PnP assumes a pinhole projection, so observations are
                # undistorted first once the coefficients are known.
                dist = self._dist(image)
                if dist is not None and np.any(dist):
                    pts2d = undistort_points(pts2d, self.rec.Ks[image], dist)
                R_new, t_new, inliers = ransac_pnp(
                    pts3d, pts2d, self.rec.Ks[image],
                    threshold_px=self.cfg.pnp_threshold_px,
                    seed=self.cfg.seed + image,
                )
                if R_new is None or int(inliers.sum()) < self.cfg.pnp_min_inliers:
                    failed.add(image)
                    self.stats["pnp_failures"] = self.stats.get("pnp_failures", 0) + 1
                    LOGGER.debug("  %s failed PnP (%d/%d inliers)",
                                 self.features[image].name,
                                 0 if R_new is None else int(inliers.sum()), n_corr)
                    continue

                self.rec.poses[image] = (R_new, t_new)
                self.rec.registered_order.append(image)
                remaining.discard(image)
                progress_since_retry = True
                bar.update(1)
                bar.set_postfix_str(f"{self.features[image].name} "
                                    f"{int(inliers.sum())}/{n_corr} inl, "
                                    f"{len(self.rec.points3d)} pts")

                self._triangulate_all()
                step += 1
                if self.cfg.local_ba_every and step % self.cfg.local_ba_every == 0:
                    self._run_ba(global_ba=False)
                if self.cfg.global_ba_every and step % self.cfg.global_ba_every == 0:
                    self._run_ba(global_ba=True)
                    # Loose during growth; the strict pass happens at the end.
                    self._filter_points(self.cfg.max_reprojection_px
                                        * self.cfg.growth_filter_slack)

        self._run_ba(global_ba=True)
        dropped = self._filter_points()
        self._triangulate_all()
        self._run_ba(global_ba=True)
        self._filter_points()

        # Second attempt at every camera now that the structure is complete,
        # then re-solve everything around the corrected poses. Runs before
        # _drop_bad_cameras so that a camera which merely landed badly gets a
        # chance to be fixed rather than discarded.
        if self.cfg.final_pose_refinement:
            if self._refine_all_poses():
                self._run_ba(global_ba=True)
                self._triangulate_all()
                self._run_ba(global_ba=True)
                self._filter_points()

        removed_cams = self._drop_bad_cameras()
        if removed_cams:
            self._run_ba(global_ba=True)
            self._filter_points()

        if self.cfg.enforce_single_component:
            if self._keep_largest_component():
                # Re-solve on the surviving component: the discarded cameras
                # were pulling on the shared parameters.
                self._triangulate_all()
                self._run_ba(global_ba=True)
                self._filter_points()

        # Polish last, and specifically *after* component culling. The
        # covisibility test asks how many triangulated points two cameras
        # share, so thinning the cloud in front of it makes well-connected
        # cameras look disconnected. Running these three rounds before the
        # cull once shattered a sound 87-camera model into 24 components and
        # threw away 52 cameras - the geometry was fine, the connectivity
        # test was starved.
        self._polish()

        # Structure quality filters. These must come after the very last
        # _triangulate_all: that call rebuilds every track it can, so running
        # them earlier gets silently undone. Both target points no reprojection
        # test can reach, and both are end-of-pipeline by design - during
        # growth they would starve the frontier.
        self.stats["dropped_low_parallax"] = self._drop_low_parallax(
            self.cfg.final_min_triangulation_angle_deg)
        self.stats["dropped_distant"] = self._drop_distant_points(
            self.cfg.max_point_distance_median_factor)

        if remaining:
            LOGGER.warning("%d image(s) never registered after %d retry pass(es): %s",
                           len(remaining), retries,
                           ", ".join(self.features[i].name for i in sorted(remaining)[:8])
                           + (" ..." if len(remaining) > 8 else ""))
        components = self._covisibility_components(
            self.stats.get("covisibility_threshold", self.cfg.min_covisible_points))
        self.stats["final_components"] = len(components)
        self.stats["retry_passes"] = retries
        self.stats["unregistered"] = len(remaining)
        self.stats["mapping_time"] = time.perf_counter() - start_grow
        self.stats["total_time"] = time.perf_counter() - overall
        self.stats["registered"] = self.rec.num_registered
        self.stats["points"] = len(self.rec.points3d)
        errs = np.array(list(self.rec.errors.values())) if self.rec.errors else np.zeros(1)
        self.stats["mean_reprojection_error"] = float(errs.mean())
        lengths = np.array(list(self.rec.track_lengths.values())) if self.rec.track_lengths else np.zeros(1)
        self.stats["mean_observed_track_length"] = float(lengths.mean())

        LOGGER.info("Reconstruction complete: %d/%d images, %d points, "
                    "mean reprojection error %.4f px, mean track length %.2f",
                    self.rec.num_registered, len(self.features), len(self.rec.points3d),
                    errs.mean(), lengths.mean())
        return self.rec
