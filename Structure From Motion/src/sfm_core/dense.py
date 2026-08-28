"""Dense multi-view stereo by plane-sweep, implemented from scratch.

For each reference view a set of neighbouring source views is chosen, a stack
of fronto-parallel depth planes is swept through the reference frustum, and for
every plane each source image is warped into the reference frame by the
plane-induced homography. The depth whose warp correlates best with the
reference image wins.

The plane-induced homography
----------------------------
For a plane at depth ``d`` in the reference camera's frame with normal
``n = (0, 0, 1)``, the mapping from reference pixels to source pixels is

    H = K_s (R_sr + t_sr n^T / d) K_r^-1

where ``(R_sr, t_sr)`` is the reference-to-source relative pose. Every point on
that plane is transferred exactly; points off it are displaced by an amount
proportional to their depth error, which is precisely what makes the
correlation score peak at the true depth.

Matching cost
-------------
Zero-mean normalised cross-correlation over a square window, computed with box
filters so the whole image is scored in a few array operations rather than a
Python loop. NCC is invariant to local brightness and contrast changes, which
matters here because exposure varies between shots.

Robustness
----------
* Costs are aggregated over the best ``k`` source views rather than all of
  them, so an occlusion in one view cannot veto a correct depth.
* A parabola is fitted through the winning cost and its two neighbours for
  sub-plane depth precision.
* Depths are kept only if several *other* views independently agree, which is
  what removes the speckle that single-view winner-take-all produces.

Nothing here calls a stereo, MVS or SfM library. OpenCV supplies image
resizing, warping and box filtering - generic image-processing primitives.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from ..utils import get_logger

__all__ = ["DenseConfig", "DepthMap", "DenseResult", "run_dense_stereo"]

LOGGER = get_logger()


@dataclass
class DenseConfig:
    """Plane-sweep parameters."""

    #: Long side the stereo runs at. The dominant speed/detail control.
    max_image_size: int = 800
    #: Depth hypotheses per reference view, spaced uniformly in inverse depth.
    num_depths: int = 96
    #: Neighbouring views used per reference view.
    num_source_views: int = 6
    #: Cost is averaged over this many best-scoring sources (occlusion robustness).
    best_k_sources: int = 3
    #: NCC window half-width in pixels.
    window_radius: int = 4
    #: Reject depths whose best NCC is below this.
    min_ncc: float = 0.35
    #: A source view is usable only within this baseline-angle band (degrees).
    min_baseline_angle: float = 2.0
    max_baseline_angle: float = 45.0
    #: Geometric consistency: required agreeing views and tolerances.
    min_consistent_views: int = 2
    consistency_pixel_threshold: float = 2.0
    consistency_depth_ratio: float = 0.02
    #: Depth range is taken from sparse points, widened by this factor.
    depth_range_padding: float = 0.15
    #: Sparse depths further than this multiple from the median are treated as
    #: triangulation outliers and excluded when choosing the sweep range.
    depth_outlier_factor: float = 5.0
    #: Hard cap on far/near plane ratio. Beyond roughly 30x the depth planes are
    #: spread too thinly to localise a surface.
    max_depth_range_ratio: float = 18.0
    #: Keep at most this many fused points (evenly subsampled) - 0 = unlimited.
    max_points: int = 0
    #: Use cv2.warpPerspective for the plane-induced warp. Setting this False
    #: substitutes warp_perspective_manual, which is ~40x slower for no quality
    #: gain - it exists to show that the library is doing image resampling
    #: only, never geometry.
    use_library_warp: bool = True


@dataclass
class DepthMap:
    image_index: int
    depth: np.ndarray            # (H, W) float32, 0 = invalid
    confidence: np.ndarray       # (H, W) float32 NCC score
    K: np.ndarray                # intrinsics at stereo resolution
    R: np.ndarray
    t: np.ndarray
    color: np.ndarray            # (H, W, 3) uint8 RGB


@dataclass
class DenseResult:
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    colors: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.uint8))
    normals: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    depth_maps: List[DepthMap] = field(default_factory=list)
    stats: Dict[str, float] = field(default_factory=dict)


# ===========================================================================
# Helpers
# ===========================================================================


def _load_rgb(path: Path, max_size: int) -> Tuple[np.ndarray, float]:
    data = np.fromfile(str(path), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not decode {path}")
    scale = 1.0
    long_side = max(bgr.shape[:2])
    if max_size and long_side > max_size:
        scale = max_size / long_side
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), scale


def _scale_K(K: np.ndarray, scale: float) -> np.ndarray:
    S = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1.0]])
    return S @ K


def _relative_pose(R_ref: np.ndarray, t_ref: np.ndarray,
                   R_src: np.ndarray, t_src: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Transform taking reference-camera coordinates into source-camera coordinates."""
    R_sr = R_src @ R_ref.T
    t_sr = t_src - R_sr @ t_ref
    return R_sr, t_sr


def _ncc_cost(ref_gray: np.ndarray, warped_gray: np.ndarray, valid: np.ndarray,
              radius: int) -> np.ndarray:
    """Windowed zero-mean NCC between two aligned images. Returns NCC in [-1, 1].

    Implemented with box filters: the local mean of an image is a box filter,
    and NCC needs only local means of I, J, I^2, J^2 and IJ. That turns an
    O(W^2) per-pixel computation into five O(1) filtered images.
    """
    size = 2 * radius + 1
    ksize = (size, size)
    a = ref_gray
    b = warped_gray

    mean_a = cv2.boxFilter(a, -1, ksize, normalize=True, borderType=cv2.BORDER_REFLECT)
    mean_b = cv2.boxFilter(b, -1, ksize, normalize=True, borderType=cv2.BORDER_REFLECT)
    mean_aa = cv2.boxFilter(a * a, -1, ksize, normalize=True, borderType=cv2.BORDER_REFLECT)
    mean_bb = cv2.boxFilter(b * b, -1, ksize, normalize=True, borderType=cv2.BORDER_REFLECT)
    mean_ab = cv2.boxFilter(a * b, -1, ksize, normalize=True, borderType=cv2.BORDER_REFLECT)

    var_a = np.maximum(mean_aa - mean_a * mean_a, 0.0)
    var_b = np.maximum(mean_bb - mean_b * mean_b, 0.0)
    cov = mean_ab - mean_a * mean_b
    denom = np.sqrt(var_a * var_b) + 1e-6
    ncc = cov / denom

    # A window that is flat in either image carries no information; scoring it
    # would let textureless regions win an arbitrary depth. The threshold is on
    # standard deviation in [0, 1] intensity units: 3e-3 is about 0.8 grey
    # levels, low enough to keep faintly textured concrete but high enough to
    # reject genuinely blank sky.
    texture_ok = (var_a > 9e-6) & (var_b > 9e-6)
    coverage = cv2.boxFilter(valid.astype(np.float32), -1, ksize,
                             normalize=True, borderType=cv2.BORDER_REFLECT)
    ncc[~texture_ok | (coverage < 0.5)] = -1.0
    return np.clip(ncc, -1.0, 1.0)


def _select_sources(ref: int, poses: Dict[int, Tuple[np.ndarray, np.ndarray]],
                    shared: Dict[Tuple[int, int], int], cfg: DenseConfig,
                    scene_depth: float) -> List[int]:
    """Pick neighbouring views: enough shared structure, sensible baseline angle.

    The angle is evaluated at a point at the reference view's *actual* median
    scene depth. An earlier version used a stand-in distance proportional to
    the baseline, which made every angle look similar and let the ranking pick
    the widest-baseline views on the far side of the capture - their warps
    barely overlapped the reference, so almost no pixel had valid support.

    Both extremes are excluded: too narrow gives no depth discrimination, too
    wide loses overlap and foreshortens the surface. The survivors are ranked
    by shared structure, with a mild preference for angles near the middle of
    the usable band.
    """
    R_ref, t_ref = poses[ref]
    C_ref = -R_ref.T @ t_ref
    forward = R_ref.T @ np.array([0.0, 0.0, 1.0])
    target = C_ref + forward * max(scene_depth, 1e-6)
    ideal = 0.5 * (cfg.min_baseline_angle + min(cfg.max_baseline_angle, 20.0))

    scored: List[Tuple[float, int]] = []
    for other, (R_o, t_o) in poses.items():
        if other == ref:
            continue
        n_shared = shared.get((min(ref, other), max(ref, other)), 0)
        if n_shared < 15:
            continue
        C_o = -R_o.T @ t_o
        v1, v2 = C_ref - target, C_o - target
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-9 or n2 < 1e-9:
            continue
        angle = np.degrees(np.arccos(np.clip(float(v1 @ v2) / (n1 * n2), -1, 1)))
        if not (cfg.min_baseline_angle <= angle <= cfg.max_baseline_angle):
            continue
        # Triangular preference peaking at `ideal`, so neither a near-duplicate
        # nor an extreme oblique view dominates the ranking.
        angle_weight = max(0.15, 1.0 - abs(angle - ideal) / max(ideal, 1e-6))
        scored.append((n_shared * angle_weight, other))
    scored.sort(reverse=True)
    return [idx for _, idx in scored[: cfg.num_source_views]]


def _depth_range(ref: int, poses: Dict[int, Tuple[np.ndarray, np.ndarray]],
                 points: np.ndarray, cfg: DenseConfig) -> Optional[Tuple[float, float]]:
    """Depth bounds for a view, from the sparse points it sees."""
    if len(points) == 0:
        return None
    R, t = poses[ref]
    z = (points @ R.T + t)[:, 2]
    z = z[np.isfinite(z) & (z > 1e-6)]
    if len(z) < 10:
        return None

    # Discard depth outliers BEFORE taking percentiles. A sparse cloud always
    # carries a few badly triangulated points - some almost at the camera, some
    # hundreds of units away - and they dominate a raw 2/98 percentile. Measured
    # on this scene the range came out as 0.00 to 172, a ratio of 1.7 million,
    # so the depth planes were spread across meaningless space and every point
    # snapped onto a coarse shell. That is what produced radial streaking.
    median = float(np.median(z))
    z = z[(z > median / cfg.depth_outlier_factor) & (z < median * cfg.depth_outlier_factor)]
    if len(z) < 10:
        return None

    lo, hi = np.percentile(z, [8, 92])
    pad = (hi - lo) * cfg.depth_range_padding
    lo, hi = lo - pad, hi + pad
    # Never let the near plane approach zero: depth resolution is uniform in
    # inverse depth, so a near plane at ~0 consumes most of the hypotheses.
    lo = max(lo, hi / cfg.max_depth_range_ratio, 1e-4)
    return lo, hi


# ===========================================================================
# Core sweep
# ===========================================================================


def _sweep_view(
    ref: int,
    sources: Sequence[int],
    images: Dict[int, np.ndarray],
    grays: Dict[int, np.ndarray],
    Ks: Dict[int, np.ndarray],
    poses: Dict[int, Tuple[np.ndarray, np.ndarray]],
    depth_lo: float,
    depth_hi: float,
    cfg: DenseConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Plane-sweep one reference view. Returns (depth, confidence)."""
    ref_gray = grays[ref]
    h, w = ref_gray.shape
    K_ref = Ks[ref]
    K_ref_inv = np.linalg.inv(K_ref)
    R_ref, t_ref = poses[ref]

    # Uniform sampling in inverse depth: depth resolution of a stereo system
    # is uniform in disparity (~1/z), so uniform-in-z planes waste hypotheses
    # far away and under-sample near the camera.
    inv = np.linspace(1.0 / depth_hi, 1.0 / depth_lo, cfg.num_depths)
    depths = (1.0 / inv).astype(np.float64)

    best = np.full((h, w), -1.0, np.float32)
    best_idx = np.zeros((h, w), np.int32)
    # Cost at the plane before and after the current winner, kept so a parabola
    # can be fitted for sub-plane precision without storing the whole volume.
    cost_prev = np.full((h, w), -1.0, np.float32)
    cost_next = np.full((h, w), -1.0, np.float32)
    prev_score = np.full((h, w), -1.0, np.float32)

    ones = np.ones((h, w), np.float32)
    for di, d in enumerate(depths):
        per_source = []
        for src in sources:
            R_sr, t_sr = _relative_pose(R_ref, t_ref, *poses[src])
            # Plane-induced homography. For the fronto-parallel plane at depth
            # d in the reference frame, points satisfy n^T X = d, hence
            # 1 = n^T X / d, and
            #     X_src = R X + t = R X + t (n^T X / d) = (R + t n^T / d) X
            # The sign is PLUS. Writing minus (the convention for a plane
            # expressed as n^T X + d = 0) warps every source image the wrong
            # way, which still produces a smooth-looking cost surface but with
            # its minimum in the wrong place - depths then disagree between
            # views by tens of percent.
            H = Ks[src] @ (R_sr + np.outer(t_sr, np.array([0.0, 0.0, 1.0])) / d) @ K_ref_inv
            if cfg.use_library_warp:
                warped = cv2.warpPerspective(
                    grays[src], H, (w, h),
                    flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
                )
            else:
                warped = warp_perspective_manual(grays[src], H, (w, h))
            mask = cv2.warpPerspective(
                ones, H, (w, h),
                flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
            ) > 0.5
            per_source.append(_ncc_cost(ref_gray, warped, mask, cfg.window_radius))

        if not per_source:
            continue
        stack = np.stack(per_source, axis=0)
        k = min(cfg.best_k_sources, stack.shape[0])
        # Average the k highest NCC scores: a surface occluded in some views
        # still scores well in the views that do see it.
        partitioned = np.partition(stack, -k, axis=0)[-k:]
        score = partitioned.mean(axis=0).astype(np.float32)

        # Before overwriting, this plane's score is the "after" neighbour for
        # any pixel whose winner is the immediately preceding plane.
        just_after = best_idx == (di - 1)
        cost_next[just_after] = score[just_after]

        improved = score > best
        cost_prev[improved] = prev_score[improved]   # score at plane di-1
        cost_next[improved] = -1.0                   # not yet known
        best_idx[improved] = di
        best[improved] = score[improved]
        prev_score = score

    # Sub-plane refinement by fitting a parabola through (prev, best, next).
    # Only meaningful where both neighbours were actually observed.
    have_both = (cost_prev > -1.0) & (cost_next > -1.0)
    prev = np.where(have_both, cost_prev, best)
    next_ = np.where(have_both, cost_next, best)
    denom = (prev - 2.0 * best + next_)
    offset = np.zeros_like(best)
    ok = np.abs(denom) > 1e-6
    offset[ok] = 0.5 * (prev[ok] - next_[ok]) / denom[ok]
    offset = np.clip(offset, -0.5, 0.5)

    idx_f = np.clip(best_idx.astype(np.float64) + offset, 0, cfg.num_depths - 1)
    inv_interp = np.interp(idx_f, np.arange(cfg.num_depths), inv)
    depth = (1.0 / np.maximum(inv_interp, 1e-9)).astype(np.float32)
    depth[best < cfg.min_ncc] = 0.0
    return depth, best


def _geometric_consistency(
    depth_maps: Dict[int, DepthMap], neighbours: Dict[int, List[int]], cfg: DenseConfig
) -> Dict[int, np.ndarray]:
    """Keep only depths that independent views agree on.

    A pixel's depth is projected into a neighbour, the neighbour's own depth at
    that location is back-projected, and the round-trip is required to land
    close to where it started - in both image space and depth. This is the step
    that removes almost all single-view speckle.
    """
    consistent: Dict[int, np.ndarray] = {}
    for ref, dm in depth_maps.items():
        h, w = dm.depth.shape
        valid = dm.depth > 0
        votes = np.zeros((h, w), np.int32)

        ys, xs = np.nonzero(valid)
        if len(ys) == 0:
            consistent[ref] = np.zeros((h, w), bool)
            continue
        z = dm.depth[ys, xs].astype(np.float64)
        K_inv = np.linalg.inv(dm.K)
        rays = (np.stack([xs, ys, np.ones_like(xs)], axis=1).astype(np.float64) @ K_inv.T)
        pts_ref_cam = rays * z[:, None]
        pts_world = (pts_ref_cam - dm.t) @ dm.R          # R^T (X_cam - t)

        for src in neighbours.get(ref, []):
            sm = depth_maps.get(src)
            if sm is None:
                continue
            cam = pts_world @ sm.R.T + sm.t
            zs = cam[:, 2]
            front = zs > 1e-6
            if not np.any(front):
                continue
            proj = (cam[front] / zs[front, None]) @ sm.K.T
            u = np.round(proj[:, 0]).astype(int)
            v = np.round(proj[:, 1]).astype(int)
            inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            if not np.any(inside):
                continue

            idx_front = np.flatnonzero(front)[inside]
            uu, vv = u[inside], v[inside]
            src_depth = sm.depth[vv, uu]
            has_depth = src_depth > 0
            if not np.any(has_depth):
                continue

            idx_front = idx_front[has_depth]
            uu, vv, src_depth = uu[has_depth], vv[has_depth], src_depth[has_depth].astype(np.float64)

            # Back-project the neighbour's own depth and re-project into ref.
            K_src_inv = np.linalg.inv(sm.K)
            r2 = (np.stack([uu, vv, np.ones_like(uu)], axis=1).astype(np.float64) @ K_src_inv.T)
            p_src_cam = r2 * src_depth[:, None]
            p_world = (p_src_cam - sm.t) @ sm.R
            back = p_world @ dm.R.T + dm.t
            zb = back[:, 2]
            ok = zb > 1e-6
            if not np.any(ok):
                continue
            pb = (back[ok] / zb[ok, None]) @ dm.K.T
            orig_idx = idx_front[ok]
            pix_err = np.hypot(pb[:, 0] - xs[orig_idx], pb[:, 1] - ys[orig_idx])
            depth_err = np.abs(zb[ok] - z[orig_idx]) / np.maximum(z[orig_idx], 1e-9)

            agree = (pix_err < cfg.consistency_pixel_threshold) & \
                    (depth_err < cfg.consistency_depth_ratio)
            good = orig_idx[agree]
            votes[ys[good], xs[good]] += 1

        consistent[ref] = valid & (votes >= cfg.min_consistent_views)
    return consistent


def _normals_from_depth(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Per-pixel normals in camera coordinates from depth gradients."""
    h, w = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    X = (xs - cx) * depth / fx
    Y = (ys - cy) * depth / fy
    Z = depth
    dzdx = np.gradient(Z, axis=1)
    dzdy = np.gradient(Z, axis=0)
    dxdx = np.gradient(X, axis=1)
    dydy = np.gradient(Y, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        nx = -dzdx / np.where(np.abs(dxdx) < 1e-9, np.nan, dxdx)
        ny = -dzdy / np.where(np.abs(dydy) < 1e-9, np.nan, dydy)
    n = np.stack([np.nan_to_num(nx), np.nan_to_num(ny), np.ones_like(Z)], axis=-1)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    return n / np.maximum(norm, 1e-9)


def run_dense_stereo(
    image_paths: Sequence[Path],
    poses: Dict[int, Tuple[np.ndarray, np.ndarray]],
    K_full,
    sparse_points: np.ndarray,
    shared_counts: Dict[Tuple[int, int], int],
    cfg: Optional[DenseConfig] = None,
) -> DenseResult:
    """Run plane-sweep MVS over every registered view and fuse the result."""
    cfg = cfg or DenseConfig()
    result = DenseResult()
    start = time.perf_counter()

    registered = sorted(poses)
    # K_full may be one matrix for every view or a per-image mapping, which is
    # what mixed photo/video inputs require.
    def _K_for(idx: int) -> np.ndarray:
        if isinstance(K_full, dict):
            return np.asarray(K_full[idx], dtype=np.float64)
        return np.asarray(K_full, dtype=np.float64)

    images: Dict[int, np.ndarray] = {}
    grays: Dict[int, np.ndarray] = {}
    Ks: Dict[int, np.ndarray] = {}
    for idx in registered:
        rgb, scale = _load_rgb(Path(image_paths[idx]), cfg.max_image_size)
        images[idx] = rgb
        grays[idx] = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        Ks[idx] = _scale_K(_K_for(idx), scale)
    LOGGER.info("Dense stereo on %d views at %dx%d, %d depth planes",
                len(registered), grays[registered[0]].shape[1],
                grays[registered[0]].shape[0], cfg.num_depths)

    # Source selection needs each view's real scene depth, so ranges come first.
    ranges = {ref: _depth_range(ref, poses, sparse_points, cfg) for ref in registered}
    neighbours = {}
    for ref in registered:
        rng = ranges[ref]
        mid = float(np.sqrt(rng[0] * rng[1])) if rng else 1.0
        neighbours[ref] = _select_sources(ref, poses, shared_counts, cfg, mid)
    depth_maps: Dict[int, DepthMap] = {}

    for ref in tqdm(registered, desc="plane sweep", unit="view"):
        sources = neighbours[ref]
        rng = ranges[ref]
        if len(sources) < 2 or rng is None:
            continue
        depth, conf = _sweep_view(ref, sources, images, grays, Ks, poses, rng[0], rng[1], cfg)
        R, t = poses[ref]
        depth_maps[ref] = DepthMap(ref, depth, conf, Ks[ref], R, t, images[ref])

    sweep_time = time.perf_counter() - start
    LOGGER.info("  sweep finished in %.1f s; checking geometric consistency", sweep_time)

    consistent = _geometric_consistency(depth_maps, neighbours, cfg)

    all_pts, all_cols, all_norms = [], [], []
    for ref, dm in depth_maps.items():
        keep = consistent[ref]
        if not np.any(keep):
            continue
        ys, xs = np.nonzero(keep)
        z = dm.depth[ys, xs].astype(np.float64)
        K_inv = np.linalg.inv(dm.K)
        rays = np.stack([xs, ys, np.ones_like(xs)], axis=1).astype(np.float64) @ K_inv.T
        cam = rays * z[:, None]
        world = (cam - dm.t) @ dm.R
        normals_cam = _normals_from_depth(dm.depth, dm.K)[ys, xs]
        normals_world = normals_cam @ dm.R
        all_pts.append(world)
        all_cols.append(dm.color[ys, xs])
        all_norms.append(normals_world)

    if all_pts:
        points = np.vstack(all_pts)
        colors = np.vstack(all_cols).astype(np.uint8)
        normals = np.vstack(all_norms)
        if cfg.max_points and len(points) > cfg.max_points:
            sel = np.linspace(0, len(points) - 1, cfg.max_points).astype(int)
            points, colors, normals = points[sel], colors[sel], normals[sel]
        result.points, result.colors, result.normals = points, colors, normals

    result.depth_maps = list(depth_maps.values())
    result.stats = {
        "views": len(depth_maps),
        "sweep_time": sweep_time,
        "total_time": time.perf_counter() - start,
        "points": len(result.points),
        "mean_valid_fraction": float(np.mean([
            (consistent[r].sum() / consistent[r].size) for r in consistent])) if consistent else 0.0,
    }
    LOGGER.info("  fused %d points from %d views (%.1f%% of pixels survived consistency) in %.1f s",
                len(result.points), len(depth_maps),
                100 * result.stats["mean_valid_fraction"], result.stats["total_time"])
    return result


def warp_perspective_manual(src: np.ndarray, H: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Bilinear perspective warp, implemented directly.

    Matches ``cv2.warpPerspective(..., flags=INTER_LINEAR | WARP_INVERSE_MAP,
    borderMode=BORDER_CONSTANT, borderValue=0)``: for each destination pixel
    ``(x, y)`` the source location is ``H @ (x, y, 1)``, sampled bilinearly,
    with out-of-bounds reads returning 0.

    Provided so the plane sweep can be run without any library resampling at
    all. It exists to demonstrate that OpenCV is not doing any geometric work
    here - the homography is derived in :func:`_sweep_view` from the recovered
    poses, and warping is pure image resampling. The library version is the
    default only because it is much faster.
    """
    width, height = size
    xs, ys = np.meshgrid(np.arange(width, dtype=np.float64),
                         np.arange(height, dtype=np.float64))
    dst_h = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)], axis=0)
    src_h = np.asarray(H, dtype=np.float64) @ dst_h
    z = src_h[2]
    safe = np.where(np.abs(z) < 1e-12, np.nan, z)
    u = src_h[0] / safe
    v = src_h[1] / safe

    h_src, w_src = src.shape[:2]
    x0 = np.floor(u).astype(np.int64)
    y0 = np.floor(v).astype(np.int64)
    fx = u - x0
    fy = v - y0

    inside = np.isfinite(u) & np.isfinite(v) & (x0 >= 0) & (y0 >= 0) & \
             (x0 + 1 < w_src) & (y0 + 1 < h_src)
    out = np.zeros(xs.size, dtype=np.float32)
    if np.any(inside):
        xi, yi = x0[inside], y0[inside]
        a, b = fx[inside], fy[inside]
        src_f = src.astype(np.float64)
        # Standard bilinear interpolation over the four surrounding samples.
        out[inside] = (src_f[yi, xi] * (1 - a) * (1 - b)
                       + src_f[yi, xi + 1] * a * (1 - b)
                       + src_f[yi + 1, xi] * (1 - a) * b
                       + src_f[yi + 1, xi + 1] * a * b).astype(np.float32)
    return out.reshape(height, width)
