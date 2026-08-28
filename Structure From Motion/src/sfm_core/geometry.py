"""Multi-view geometry, implemented from first principles.

WHY THIS EXISTS
---------------
The project constraint forbids calling library functions that *directly
implement the Structure-from-Motion task*. Everything in this module is
therefore written here rather than delegated:

============================  ==================================================
Implemented here              Deliberately NOT used
============================  ==================================================
normalized 8-point algorithm  ``cv2.findFundamentalMat``
essential matrix estimation   ``cv2.findEssentialMat``
E decomposition + cheirality  ``cv2.recoverPose``
DLT triangulation             ``cv2.triangulatePoints``
linear PnP + RANSAC           ``cv2.solvePnP`` / ``cv2.solvePnPRansac``
RANSAC driver                 ``cv2.USAC_*``
============================  ==================================================

OpenCV is used only for image I/O, SIFT feature detection and descriptor
matching - image-processing primitives, not the geometric reconstruction.
NumPy supplies linear algebra (SVD, least squares); SciPy supplies a generic
optimiser for bundle adjustment. Neither knows anything about SfM.

CONVENTIONS
-----------
* A pose is world-to-camera: ``x_cam = R @ x_world + t``.
* The camera centre is therefore ``C = -R.T @ t``, never ``-t``.
* Image points are (N, 2) float64 arrays of pixel coordinates.
* "Normalised" coordinates mean ``K^-1 x``, i.e. calibrated rays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "similarity_from_points", "ransac_similarity",
    "hartley_normalize",
    "fundamental_from_points",
    "essential_from_points",
    "sampson_distance",
    "symmetric_epipolar_distance",
    "ransac",
    "ransac_essential",
    "ransac_fundamental",
    "decompose_essential",
    "select_pose_by_cheirality",
    "triangulate_dlt",
    "triangulate_multiview",
    "reprojection_error",
    "project_points",
    "apply_radial_distortion",
    "undistort_points",
    "pnp_linear",
    "ransac_pnp",
    "rotation_angle_deg",
    "look_at_pose",
]


# ===========================================================================
# Point normalisation
# ===========================================================================


def hartley_normalize(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Isotropic normalisation (Hartley 1997). Returns (normalised, T).

    Translates the centroid to the origin and scales so the mean distance from
    it is sqrt(2). Without this the 8-point algorithm is numerically hopeless:
    the design matrix mixes terms of order 10^6 (x*x') with terms of order 1,
    so the smallest singular vector is dominated by round-off.

    ``T`` is the 3x3 similarity applied, so that ``x_norm = T @ x_hom``.
    """
    points = np.asarray(points, dtype=np.float64)
    centroid = points.mean(axis=0)
    centred = points - centroid
    mean_dist = float(np.sqrt((centred ** 2).sum(axis=1)).mean())
    scale = np.sqrt(2.0) / mean_dist if mean_dist > 1e-12 else 1.0
    T = np.array([
        [scale, 0.0, -scale * centroid[0]],
        [0.0, scale, -scale * centroid[1]],
        [0.0, 0.0, 1.0],
    ])
    normalised = centred * scale
    return normalised, T


def _to_homogeneous(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return np.hstack([points, np.ones((len(points), 1))])


# ===========================================================================
# Fundamental / essential matrix
# ===========================================================================


def fundamental_from_points(pts1: np.ndarray, pts2: np.ndarray) -> Optional[np.ndarray]:
    """Normalized 8-point algorithm for the fundamental matrix.

    Each correspondence contributes one row of the epipolar constraint
    ``x2^T F x1 = 0``, written as ``A f = 0`` with ``f`` the 9 entries of F.
    The solution is the right singular vector of A with the smallest singular
    value. F is then forced to rank 2 by zeroing its smallest singular value,
    which is what makes all epipolar lines meet at a single epipole.

    Returns ``None`` when the configuration is degenerate.
    """
    if len(pts1) < 8 or len(pts2) != len(pts1):
        return None

    x1, T1 = hartley_normalize(pts1)
    x2, T2 = hartley_normalize(pts2)

    u1, v1 = x1[:, 0], x1[:, 1]
    u2, v2 = x2[:, 0], x2[:, 1]
    A = np.column_stack([
        u2 * u1, u2 * v1, u2,
        v2 * u1, v2 * v1, v2,
        u1, v1, np.ones(len(u1)),
    ])

    try:
        _, _, Vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return None
    F = Vt[-1].reshape(3, 3)

    # Enforce det(F) = 0 by nulling the smallest singular value.
    U, S, Vt2 = np.linalg.svd(F)
    S[2] = 0.0
    F = U @ np.diag(S) @ Vt2

    # Undo the normalisation: F_pixels = T2^T F_norm T1
    F = T2.T @ F @ T1
    norm = np.linalg.norm(F)
    return F / norm if norm > 1e-12 else None


def essential_from_points(rays1: np.ndarray, rays2: np.ndarray) -> Optional[np.ndarray]:
    """Essential matrix from calibrated rays, via the 8-point algorithm.

    Identical machinery to :func:`fundamental_from_points`, except the inputs
    are already calibrated (``K^-1 x``) and the singular values are forced to
    ``(1, 1, 0)`` rather than merely ``(s1, s2, 0)``. That extra constraint is
    what distinguishes an essential matrix from a fundamental one: E has only
    5 degrees of freedom because its two non-zero singular values must be
    equal.
    """
    E = fundamental_from_points(rays1, rays2)
    if E is None:
        return None
    U, _, Vt = np.linalg.svd(E)
    E = U @ np.diag([1.0, 1.0, 0.0]) @ Vt
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        # Keep the decomposition in SO(3); a reflection would produce mirrored
        # poses that fail cheirality for every candidate.
        E = -E
    return E


def sampson_distance(F: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """First-order geometric distance to the epipolar line, in input units.

    ``d = (x2^T F x1)^2 / ((Fx1)_1^2 + (Fx1)_2^2 + (F^T x2)_1^2 + (F^T x2)_2^2)``

    Returned as the square root so the values are distances, directly
    comparable against a pixel threshold. The raw algebraic residual
    ``x2^T F x1`` is not a distance and is biased by where the point sits in
    the image.
    """
    x1 = _to_homogeneous(pts1)
    x2 = _to_homogeneous(pts2)
    Fx1 = x1 @ F.T
    Ftx2 = x2 @ F
    numerator = np.sum(x2 * Fx1, axis=1) ** 2
    denominator = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        squared = np.where(denominator > 1e-15, numerator / denominator, np.inf)
    return np.sqrt(squared)


def symmetric_epipolar_distance(F: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """Mean point-to-epipolar-line distance in both images."""
    x1 = _to_homogeneous(pts1)
    x2 = _to_homogeneous(pts2)
    l2 = x1 @ F.T
    l1 = x2 @ F
    d2 = np.abs(np.sum(x2 * l2, axis=1)) / np.maximum(np.hypot(l2[:, 0], l2[:, 1]), 1e-12)
    d1 = np.abs(np.sum(x1 * l1, axis=1)) / np.maximum(np.hypot(l1[:, 0], l1[:, 1]), 1e-12)
    return 0.5 * (d1 + d2)


# ===========================================================================
# Generic RANSAC
# ===========================================================================


@dataclass
class RansacResult:
    model: Optional[np.ndarray]
    inliers: np.ndarray
    num_inliers: int
    iterations: int
    support: float = 0.0


def ransac(
    n_data: int,
    min_samples: int,
    fit_fn: Callable[[np.ndarray], Optional[np.ndarray]],
    residual_fn: Callable[[np.ndarray], np.ndarray],
    threshold: float,
    max_iterations: int = 2000,
    confidence: float = 0.9999,
    rng: Optional[np.random.Generator] = None,
    refit: bool = True,
) -> RansacResult:
    """Generic RANSAC with adaptive iteration count.

    The iteration budget is re-derived from the best inlier ratio seen so far::

        N = log(1 - confidence) / log(1 - w^m)

    so a clean problem stops early while a contaminated one keeps sampling up
    to ``max_iterations``.

    ``refit`` re-estimates the model on the full inlier set at the end, which
    matters: the model from a minimal sample is only as accurate as those few
    points, whereas the refit uses every consistent measurement.
    """
    rng = rng or np.random.default_rng(0)
    if n_data < min_samples:
        return RansacResult(None, np.zeros(n_data, dtype=bool), 0, 0)

    best_model: Optional[np.ndarray] = None
    best_inliers = np.zeros(n_data, dtype=bool)
    best_count = 0
    best_support = 0.0

    iterations = 0
    budget = max_iterations
    log_conf = np.log(max(1e-12, 1.0 - confidence))

    while iterations < min(budget, max_iterations):
        iterations += 1
        sample = rng.choice(n_data, size=min_samples, replace=False)
        model = fit_fn(sample)
        if model is None:
            continue
        residuals = residual_fn(model)
        inliers = np.isfinite(residuals) & (residuals < threshold)
        count = int(inliers.sum())
        if count < min_samples:
            continue
        # MSAC-style score: among models with equal inlier counts, prefer the
        # one whose inliers fit more tightly.
        support = float(np.sum(np.clip(threshold - residuals[inliers], 0, None)))
        if count > best_count or (count == best_count and support > best_support):
            best_model, best_inliers, best_count, best_support = model, inliers, count, support
            w = count / n_data
            if 0.0 < w < 1.0:
                denom = np.log(max(1e-12, 1.0 - w ** min_samples))
                budget = int(np.ceil(log_conf / denom)) if denom < 0 else max_iterations

    if refit and best_model is not None and best_count >= min_samples:
        refined = fit_fn(np.flatnonzero(best_inliers))
        if refined is not None:
            residuals = residual_fn(refined)
            inliers = np.isfinite(residuals) & (residuals < threshold)
            if int(inliers.sum()) >= best_count:
                best_model, best_inliers, best_count = refined, inliers, int(inliers.sum())

    return RansacResult(best_model, best_inliers, best_count, iterations, best_support)


def ransac_fundamental(
    pts1: np.ndarray,
    pts2: np.ndarray,
    threshold_px: float = 2.0,
    max_iterations: int = 2000,
    confidence: float = 0.9999,
    seed: int = 0,
) -> RansacResult:
    """Robustly estimate the fundamental matrix directly in pixel coordinates.

    Used for match verification because it needs **no intrinsics**: a pair is
    accepted when one epipolar geometry explains enough of its correspondences,
    regardless of what the focal length turns out to be. Verifying with the
    essential matrix instead couples the match graph to the calibration guess,
    so a poor focal prior silently discards good pairs.
    """
    pts1 = np.asarray(pts1, dtype=np.float64)
    pts2 = np.asarray(pts2, dtype=np.float64)

    def fit(idx: np.ndarray) -> Optional[np.ndarray]:
        return fundamental_from_points(pts1[idx], pts2[idx])

    def residual(F: np.ndarray) -> np.ndarray:
        return sampson_distance(F, pts1, pts2)

    return ransac(
        len(pts1), 8, fit, residual, threshold_px,
        max_iterations=max_iterations, confidence=confidence,
        rng=np.random.default_rng(seed),
    )


def ransac_essential(
    pts1: np.ndarray,
    pts2: np.ndarray,
    K: np.ndarray,
    K2: Optional[np.ndarray] = None,
    threshold_px: float = 1.5,
    max_iterations: int = 2000,
    confidence: float = 0.9999,
    seed: int = 0,
) -> RansacResult:
    """Robustly estimate the essential matrix from pixel correspondences.

    ``K`` calibrates the first view and ``K2`` the second; when ``K2`` is
    omitted both views share ``K``. Distinct intrinsics matter when the images
    come from different sources - photographs and video frames of the same
    scene have different fields of view, and calibrating both with one matrix
    would bias every recovered pose.

    The threshold is specified in pixels for interpretability but applied to
    calibrated rays, so it is converted using the mean focal length.
    """
    K = np.asarray(K, dtype=np.float64)
    K2 = K if K2 is None else np.asarray(K2, dtype=np.float64)
    rays1 = (_to_homogeneous(pts1) @ np.linalg.inv(K).T)[:, :2]
    rays2 = (_to_homogeneous(pts2) @ np.linalg.inv(K2).T)[:, :2]
    focal = 0.25 * (K[0, 0] + K[1, 1] + K2[0, 0] + K2[1, 1])
    threshold = threshold_px / max(focal, 1e-9)

    def fit(idx: np.ndarray) -> Optional[np.ndarray]:
        return essential_from_points(rays1[idx], rays2[idx])

    def residual(E: np.ndarray) -> np.ndarray:
        return sampson_distance(E, rays1, rays2)

    return ransac(
        len(pts1), 8, fit, residual, threshold,
        max_iterations=max_iterations, confidence=confidence,
        rng=np.random.default_rng(seed),
    )


# ===========================================================================
# Pose from the essential matrix
# ===========================================================================


def decompose_essential(E: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return the four (R, t) candidates consistent with E.

    ``E = [t]_x R`` determines the pose only up to a four-fold ambiguity:
    two rotations (differing by a 180 degree twist about the baseline) times
    two translation signs. Only one places points in front of both cameras -
    :func:`select_pose_by_cheirality` resolves it.

    The translation is returned with unit norm; its magnitude is
    unrecoverable, which is the origin of monocular SfM's arbitrary scale.
    """
    U, _, Vt = np.linalg.svd(E)
    # Keep both factors in SO(3): an odd number of reflections would yield
    # rotation matrices with det = -1, which are not physical poses.
    if np.linalg.det(U) < 0:
        U = -U
    if np.linalg.det(Vt) < 0:
        Vt = -Vt

    W = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:, 2]
    t = t / max(np.linalg.norm(t), 1e-12)
    return [(R1, t), (R1, -t), (R2, t), (R2, -t)]


def select_pose_by_cheirality(
    candidates: Sequence[Tuple[np.ndarray, np.ndarray]],
    pts1: np.ndarray,
    pts2: np.ndarray,
    K: np.ndarray,
    max_reprojection_px: float = 8.0,
    K2: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray, Optional[np.ndarray]]:
    """Pick the (R, t) that puts the most points in front of both cameras.

    Returns ``(R, t, mask, points3d)`` where ``mask`` marks correspondences
    that are in front of both cameras *and* reproject accurately. Requiring
    both is important: a wrong candidate can occasionally place a few noisy
    points in front of both cameras, but it cannot reproject them well.
    """
    K = np.asarray(K, dtype=np.float64)
    K2 = K if K2 is None else np.asarray(K2, dtype=np.float64)
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])

    best = (None, None, np.zeros(len(pts1), dtype=bool), None)
    best_count = -1
    for R, t in candidates:
        P2 = K2 @ np.hstack([R, t.reshape(3, 1)])
        X = triangulate_dlt(P1, P2, pts1, pts2)
        depth1 = X[:, 2]
        depth2 = (X @ R.T + t)[:, 2]
        in_front = (depth1 > 1e-9) & (depth2 > 1e-9) & np.isfinite(X).all(axis=1)

        err1 = reprojection_error(X, pts1, K, np.eye(3), np.zeros(3))
        err2 = reprojection_error(X, pts2, K2, R, t)
        good = in_front & (err1 < max_reprojection_px) & (err2 < max_reprojection_px)
        count = int(good.sum())
        if count > best_count:
            best_count = count
            best = (R, t, good, X)
    return best


# ===========================================================================
# Triangulation
# ===========================================================================


def triangulate_dlt(P1: np.ndarray, P2: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """Two-view triangulation by the Direct Linear Transform.

    From ``x ~ P X`` the cross product ``x x (P X) = 0`` gives two independent
    equations per view::

        u (p3^T X) - (p1^T X) = 0
        v (p3^T X) - (p2^T X) = 0

    Stacking both views gives ``A X = 0`` (4x4); the solution is the right
    singular vector for the smallest singular value. Vectorised over all
    correspondences via a batched SVD.

    Returns Euclidean (N, 3) points; rows whose homogeneous coordinate is
    effectively zero (points at infinity, produced by near-zero parallax) are
    returned as NaN so they can be filtered rather than silently becoming
    enormous finite values.
    """
    pts1 = np.asarray(pts1, dtype=np.float64)
    pts2 = np.asarray(pts2, dtype=np.float64)
    n = len(pts1)

    A = np.empty((n, 4, 4), dtype=np.float64)
    A[:, 0] = pts1[:, 0:1] * P1[2] - P1[0]
    A[:, 1] = pts1[:, 1:2] * P1[2] - P1[1]
    A[:, 2] = pts2[:, 0:1] * P2[2] - P2[0]
    A[:, 3] = pts2[:, 1:2] * P2[2] - P2[1]

    _, _, Vt = np.linalg.svd(A)
    Xh = Vt[:, -1, :]
    w = Xh[:, 3]
    out = np.full((n, 3), np.nan)
    ok = np.abs(w) > 1e-12
    out[ok] = Xh[ok, :3] / w[ok, None]
    return out


def triangulate_multiview(
    projections: Sequence[np.ndarray], observations: Sequence[np.ndarray]
) -> Optional[np.ndarray]:
    """N-view DLT triangulation of a single point from >= 2 observations.

    Same construction as the two-view case with two rows per view. Using every
    observation rather than just two is what makes long tracks more accurate.
    """
    if len(projections) < 2:
        return None
    rows = []
    for P, xy in zip(projections, observations):
        rows.append(xy[0] * P[2] - P[0])
        rows.append(xy[1] * P[2] - P[1])
    A = np.asarray(rows, dtype=np.float64)
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    try:
        _, _, Vt = np.linalg.svd(A / norms)
    except np.linalg.LinAlgError:
        return None
    Xh = Vt[-1]
    if abs(Xh[3]) < 1e-12:
        return None
    return Xh[:3] / Xh[3]


def apply_radial_distortion(x: np.ndarray, y: np.ndarray,
                            dist: Optional[Sequence[float]]) -> Tuple[np.ndarray, np.ndarray]:
    """Apply the radial model to normalised image coordinates.

        r^2 = x^2 + y^2
        factor = 1 + k1 r^2 + k2 r^4
        (x', y') = factor * (x, y)

    Only radial terms are modelled. Tangential distortion is negligible on
    phone lenses compared with the radial component, and adding parameters the
    data cannot constrain makes the bundle adjustment less stable, not more.
    """
    if dist is None or len(dist) == 0:
        return x, y
    k1 = float(dist[0])
    k2 = float(dist[1]) if len(dist) > 1 else 0.0
    r2 = x * x + y * y
    factor = 1.0 + k1 * r2 + k2 * r2 * r2
    return x * factor, y * factor


def project_points(points3d: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray,
                   dist: Optional[Sequence[float]] = None) -> np.ndarray:
    """Project world points to pixels through a camera, with optional distortion."""
    cam = np.asarray(points3d, dtype=np.float64) @ np.asarray(R).T + np.asarray(t).ravel()
    z = cam[:, 2]
    safe = np.where(np.abs(z) < 1e-12, np.nan, z)
    x, y = cam[:, 0] / safe, cam[:, 1] / safe
    x, y = apply_radial_distortion(x, y, dist)
    K = np.asarray(K, dtype=np.float64)
    return np.column_stack([K[0, 0] * x + K[0, 2], K[1, 1] * y + K[1, 2]])


def reprojection_error(
    points3d: np.ndarray, points2d: np.ndarray, K: np.ndarray,
    R: np.ndarray, t: np.ndarray, dist: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Euclidean reprojection error in pixels, NaN-safe."""
    proj = project_points(points3d, K, R, t, dist)
    return np.linalg.norm(proj - np.asarray(points2d, dtype=np.float64), axis=1)


def triangulation_angle_deg(X: np.ndarray, C1: np.ndarray, C2: np.ndarray) -> np.ndarray:
    """Angle subtended at each 3D point by two camera centres, in degrees.

    The conditioning number for triangulation: as it approaches zero the two
    rays become parallel and depth becomes arbitrarily sensitive to noise.
    """
    r1 = X - np.asarray(C1, dtype=np.float64)
    r2 = X - np.asarray(C2, dtype=np.float64)
    n1 = np.linalg.norm(r1, axis=1)
    n2 = np.linalg.norm(r2, axis=1)
    ok = (n1 > 1e-12) & (n2 > 1e-12)
    cosang = np.ones(len(X))
    cosang[ok] = np.sum(r1[ok] * r2[ok], axis=1) / (n1[ok] * n2[ok])
    return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))


# ===========================================================================
# Perspective-n-Point (registering a new view against known 3D points)
# ===========================================================================


def pnp_linear(points3d: np.ndarray, points2d: np.ndarray, K: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Linear PnP: recover (R, t) from >= 6 3D-2D correspondences.

    Solves for the full 3x4 projection matrix by DLT on calibrated rays, then
    extracts a rotation. The raw 3x3 block is not orthonormal because the
    linear solution ignores that constraint, so it is projected onto SO(3) by
    SVD (the nearest rotation in the Frobenius sense), and the translation is
    rescaled by the same factor that normalising the rotation implied.

    The sign is fixed by requiring positive depth for the correspondences -
    without it the solution can come back mirrored through the camera.
    """
    points3d = np.asarray(points3d, dtype=np.float64)
    points2d = np.asarray(points2d, dtype=np.float64)
    if len(points3d) < 6:
        return None

    K_inv = np.linalg.inv(np.asarray(K, dtype=np.float64))
    rays = (_to_homogeneous(points2d) @ K_inv.T)[:, :2]

    # Condition the 3D points too: the same scaling argument as Hartley's 2D
    # normalisation applies to the world coordinates.
    centroid = points3d.mean(axis=0)
    spread = float(np.sqrt(((points3d - centroid) ** 2).sum(axis=1)).mean())
    scale = np.sqrt(3.0) / spread if spread > 1e-12 else 1.0
    Xn = (points3d - centroid) * scale

    n = len(Xn)
    A = np.zeros((2 * n, 12), dtype=np.float64)
    Xh = np.hstack([Xn, np.ones((n, 1))])
    u, v = rays[:, 0], rays[:, 1]
    A[0::2, 0:4] = Xh
    A[0::2, 8:12] = -u[:, None] * Xh
    A[1::2, 4:8] = Xh
    A[1::2, 8:12] = -v[:, None] * Xh

    try:
        _, _, Vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return None
    P = Vt[-1].reshape(3, 4)

    M = P[:, :3]
    U, S, Vt2 = np.linalg.svd(M)
    if np.any(S < 1e-12):
        return None
    R = U @ Vt2
    if np.linalg.det(R) < 0:
        R = -R
        P = -P
    # The DLT solution is defined up to scale; recover it from the singular
    # values of the rotation block.
    beta = 1.0 / float(np.mean(S))
    t_normalised = P[:, 3] * beta

    # Undo the world-space conditioning. The DLT solved
    #     x_cam  ~  R @ Xn + t_normalised,      Xn = scale * (X - centroid)
    # so substituting and dividing through by the positive constant `scale`
    # (which leaves the projective ray unchanged):
    #     x_cam  ~  R @ X + (t_normalised / scale - R @ centroid)
    t = t_normalised / scale - R @ centroid

    depths = points3d @ R.T + t
    if np.mean(depths[:, 2] > 0) < 0.5:
        R, t = -R, -t
        if np.linalg.det(R) < 0:
            return None
    return R, t


def refine_pose(points3d: np.ndarray, points2d: np.ndarray, K: np.ndarray,
                R: np.ndarray, t: np.ndarray,
                dist: Optional[Sequence[float]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Non-linear refinement of a pose over its 6 degrees of freedom.

    A minimal solver fits exactly three points, so its accuracy is limited by
    those three measurements. Re-optimising over every inlier costs a few
    milliseconds and typically halves the reprojection error, which in turn
    lets more points pass the inlier test on the next iteration.
    """
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    x0 = np.hstack([Rotation.from_matrix(R).as_rotvec(), np.asarray(t).ravel()])

    def residuals(x: np.ndarray) -> np.ndarray:
        Rx = Rotation.from_rotvec(x[:3]).as_matrix()
        proj = project_points(points3d, K, Rx, x[3:6], dist)
        return np.nan_to_num(proj - points2d, nan=1e3).ravel()

    try:
        sol = least_squares(residuals, x0, method="lm", max_nfev=100)
    except Exception:
        return R, t
    return Rotation.from_rotvec(sol.x[:3]).as_matrix(), sol.x[3:6]


def ransac_pnp(
    points3d: np.ndarray,
    points2d: np.ndarray,
    K: np.ndarray,
    threshold_px: float = 4.0,
    max_iterations: int = 2000,
    confidence: float = 0.9999,
    seed: int = 0,
    minimal: str = "p3p",
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
    """Robust PnP. Returns ``(R, t, inlier_mask)``.

    Uses a three-point minimal solver by default. Sample size dominates RANSAC's
    behaviour: the chance a random sample is outlier-free is ``w^m``, so at a
    94% outlier rate - which frontier views in a real capture routinely hit -
    a 6-point solver succeeds on 5e-8 of samples against P3P's 2e-4. Measured on
    synthetic data at 70% outliers: 6-point 24/40 successes, P3P 40/40.
    """
    from .p3p import solve_pnp_p3p

    points3d = np.asarray(points3d, dtype=np.float64)
    points2d = np.asarray(points2d, dtype=np.float64)
    use_p3p = minimal == "p3p" and len(points3d) >= 3
    min_samples = 3 if use_p3p else 6

    def fit(idx: np.ndarray) -> Optional[np.ndarray]:
        if use_p3p and len(idx) == 3:
            candidates = solve_pnp_p3p(points3d[idx], points2d[idx], K)
            if not candidates:
                return None
            # P3P returns up to four poses; keep the one that best explains the
            # sample it was built from.
            best, best_error = None, np.inf
            for R, t in candidates:
                err = reprojection_error(points3d[idx], points2d[idx], K, R, t)
                score = float(np.nanmean(err))
                if np.isfinite(score) and score < best_error:
                    best_error, best = score, np.hstack([R, t.reshape(3, 1)])
            return best
        solution = pnp_linear(points3d[idx], points2d[idx], K)
        if solution is None:
            return None
        R, t = solution
        return np.hstack([R, t.reshape(3, 1)])

    def residual(Rt: np.ndarray) -> np.ndarray:
        return reprojection_error(points3d, points2d, K, Rt[:, :3], Rt[:, 3])

    result = ransac(
        len(points3d), min_samples, fit, residual, threshold_px,
        max_iterations=max_iterations, confidence=confidence,
        rng=np.random.default_rng(seed),
        refit=False,      # a 3-point solver cannot consume the full inlier set
    )
    if result.model is None:
        return None, None, np.zeros(len(points3d), dtype=bool)

    R, t = result.model[:, :3], result.model[:, 3]
    inliers = result.inliers
    # Refine on the inliers, then re-score: the improved pose usually admits
    # more inliers, and one extra round is enough to converge.
    for _ in range(2):
        if int(inliers.sum()) < 4:
            break
        R_new, t_new = refine_pose(points3d[inliers], points2d[inliers], K, R, t)
        errs = reprojection_error(points3d, points2d, K, R_new, t_new)
        new_inliers = np.isfinite(errs) & (errs < threshold_px)
        if int(new_inliers.sum()) >= int(inliers.sum()):
            R, t, inliers = R_new, t_new, new_inliers
        else:
            break
    return R, t, inliers


# ===========================================================================
# Small helpers
# ===========================================================================


def undistort_points(points: np.ndarray, K: np.ndarray,
                     dist: Optional[Sequence[float]], iterations: int = 8) -> np.ndarray:
    """Remove radial distortion from pixel coordinates.

    The forward model is not analytically invertible, so it is inverted by
    fixed-point iteration: start from the distorted normalised coordinate and
    repeatedly divide by the distortion factor evaluated at the current
    estimate. It converges in a handful of steps for realistic coefficients.
    """
    points = np.asarray(points, dtype=np.float64)
    if dist is None or len(dist) == 0 or not np.any(np.asarray(dist)):
        return points
    K = np.asarray(K, dtype=np.float64)
    x = (points[:, 0] - K[0, 2]) / K[0, 0]
    y = (points[:, 1] - K[1, 2]) / K[1, 1]
    xu, yu = x.copy(), y.copy()
    k1 = float(dist[0])
    k2 = float(dist[1]) if len(dist) > 1 else 0.0
    for _ in range(iterations):
        r2 = xu * xu + yu * yu
        factor = 1.0 + k1 * r2 + k2 * r2 * r2
        factor = np.where(np.abs(factor) < 1e-9, 1.0, factor)
        xu, yu = x / factor, y / factor
    return np.column_stack([K[0, 0] * xu + K[0, 2], K[1, 1] * yu + K[1, 2]])


def rotation_angle_deg(R: np.ndarray) -> float:
    """Axis-angle magnitude of a rotation matrix, in degrees."""
    cos_theta = (float(np.trace(R)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))


def look_at_pose(eye: np.ndarray, target: np.ndarray, up: np.ndarray = np.array([0.0, -1.0, 0.0])):
    """World-to-camera (R, t) for a camera at ``eye`` looking at ``target``."""
    eye = np.asarray(eye, dtype=np.float64)
    forward = np.asarray(target, dtype=np.float64) - eye
    forward /= max(np.linalg.norm(forward), 1e-12)
    right = np.cross(forward, up)
    right /= max(np.linalg.norm(right), 1e-12)
    down = np.cross(forward, right)
    R = np.vstack([right, down, forward])
    return R, -R @ eye


# ===========================================================================
# Similarity alignment between two reconstructions
# ===========================================================================


def similarity_from_points(src: np.ndarray, dst: np.ndarray
                           ) -> Optional[Tuple[float, np.ndarray, np.ndarray]]:
    """Least-squares similarity ``(s, R, t)`` with ``s R x + t ~ y``.

    The closed-form solution of Umeyama (1991), implemented here: centre both
    clouds, take the SVD of their cross-covariance for the rotation, and read
    the scale off the ratio of variances. NumPy contributes the SVD, which is a
    matrix factorisation and not a reconstruction step.

    Two reconstructions of the same scene agree only up to a similarity - SfM
    fixes the world frame and the overall scale arbitrarily - so this is what
    has to be estimated before independent models can be merged. The reflection
    guard matters: an unconstrained SVD will happily return a determinant of
    -1, which mirrors the scene and looks plausible in reprojection while being
    physically impossible.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or len(src) < 3:
        return None

    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    a, b = src - mu_s, dst - mu_d
    var_s = float((a ** 2).sum() / len(a))
    if var_s < 1e-12:
        return None

    U, S, Vt = np.linalg.svd((b.T @ a) / len(a))
    d = np.ones(3)
    if np.linalg.det(U @ Vt) < 0:
        d[2] = -1.0                      # forbid a mirrored solution
    R = U @ np.diag(d) @ Vt
    scale = float((S * d).sum() / var_s)
    if not np.isfinite(scale) or scale <= 1e-12:
        return None
    t = mu_d - scale * (R @ mu_s)
    return scale, R, t


def ransac_similarity(src: np.ndarray, dst: np.ndarray,
                      threshold: float, max_iterations: int = 2000,
                      confidence: float = 0.999, seed: int = 0
                      ) -> Tuple[Optional[Tuple[float, np.ndarray, np.ndarray]], np.ndarray]:
    """Robust similarity fit. Returns ``((s, R, t), inlier_mask)``.

    Correspondences between two independently-built models come from matches
    that were never good enough to reconstruct with, so a large outlier
    fraction is expected and a plain least-squares fit would be dragged apart
    by it. ``threshold`` is a distance in the destination model's units; scale
    it to that model rather than passing an absolute number.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = len(src)
    if n < 3:
        return None, np.zeros(n, dtype=bool)

    rng = np.random.default_rng(seed)
    best_model, best_inliers, best_count = None, np.zeros(n, dtype=bool), 0
    iterations = max_iterations
    i = 0
    while i < iterations:
        i += 1
        idx = rng.choice(n, size=3, replace=False)
        model = similarity_from_points(src[idx], dst[idx])
        if model is None:
            continue
        s, R, t = model
        residual = np.linalg.norm(src @ R.T * s + t - dst, axis=1)
        inliers = residual < threshold
        count = int(inliers.sum())
        if count > best_count:
            best_count, best_inliers, best_model = count, inliers, model
            ratio = count / n
            if 0.0 < ratio < 1.0:
                denom = np.log(max(1.0 - ratio ** 3, 1e-12))
                iterations = min(max_iterations,
                                 int(np.log(max(1.0 - confidence, 1e-12)) / denom) + 1)
    if best_model is None or best_count < 3:
        return None, np.zeros(n, dtype=bool)

    # Refit on the consensus set: the minimal sample fixed the support, but the
    # final estimate should use all of it.
    refined = similarity_from_points(src[best_inliers], dst[best_inliers])
    return (refined or best_model), best_inliers
