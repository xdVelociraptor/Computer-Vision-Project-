"""Multi-view bundle adjustment.

Minimises total reprojection error over camera poses and 3D points
simultaneously::

    min  sum_ij  rho( || x_ij - pi(K [R_i|t_i] X_j) ||^2 )

``scipy.optimize.least_squares`` supplies the Levenberg-Marquardt / trust-region
machinery. That is a *generic non-linear least-squares solver* - it knows
nothing about cameras or reconstruction. The parameterisation, the residual
function, the Jacobian sparsity pattern and the gauge handling are all
implemented here.

Three details that decide whether this works at all
--------------------------------------------------
**Rotation parameterisation.** Rotations are optimised as axis-angle
(Rodrigues) vectors: 3 numbers, minimal and unconstrained. Optimising the 9
entries of a matrix would need orthonormality constraints the solver cannot
express, and quaternions would need a unit-norm constraint.

**Sparsity.** Observation ``(i, j)`` depends only on camera ``i`` and point
``j``, so the Jacobian is overwhelmingly zero. A 100-camera, 30k-point problem
has a dense Jacobian of ~5x10^10 entries; declaring the sparsity pattern makes
it tractable. This is the single reason bundle adjustment scales.

**Gauge freedom.** The cost is invariant to a global similarity transform
(7 DOF: 3 rotation, 3 translation, 1 scale). Left free, the solver can wander
along those directions. Fixing the first camera at the origin removes 6; the
remaining scale is handled by holding a second camera's translation norm, or
simply left to drift because the cost is exactly invariant to it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from ..utils import get_logger

__all__ = ["BundleResult", "bundle_adjust"]

LOGGER = get_logger()

#: Floor for the robust-loss scale, in pixels. Below roughly this the loss
#: begins rejecting honest feature-localisation noise as outliers, so a
#: well-converged model gains nothing from tightening further.
_MIN_F_SCALE = 2.0


@dataclass
class BundleResult:
    ran: bool = False
    num_cameras: int = 0
    num_points: int = 0
    num_observations: int = 0
    num_parameters: int = 0
    iterations: int = 0
    duration_s: float = 0.0
    rmse_before: float = 0.0
    rmse_after: float = 0.0
    mean_before: float = 0.0
    mean_after: float = 0.0
    median_before: float = 0.0
    median_after: float = 0.0
    improvement_percent: float = 0.0
    focal_before: float = 0.0
    focal_after: float = 0.0
    termination: str = ""
    #: Robust-loss scale actually used, in pixels.
    f_scale: float = 0.0
    note: str = ""
    distortion: Dict[int, np.ndarray] = field(default_factory=dict)


def _project(rotvecs: np.ndarray, translations: np.ndarray, points: np.ndarray,
             focals: np.ndarray, group_of_obs: np.ndarray,
             cx: np.ndarray, cy: np.ndarray,
             cam_idx: np.ndarray, pt_idx: np.ndarray,
             dist: Optional[np.ndarray] = None) -> np.ndarray:
    """Project points into their observing cameras. Vectorised over observations.

    ``focals`` holds one focal length per camera *group*; ``group_of_obs`` maps
    each observation to its group. Principal points stay fixed per observation:
    with an uncalibrated phone they are very weakly constrained and readily
    absorb errors that belong elsewhere.
    """
    rot = Rotation.from_rotvec(rotvecs[cam_idx])
    cam_points = rot.apply(points[pt_idx]) + translations[cam_idx]
    z = cam_points[:, 2]
    safe_z = np.where(np.abs(z) < 1e-9, 1e-9, z)
    x = cam_points[:, 0] / safe_z
    y = cam_points[:, 1] / safe_z
    if dist is not None:
        # Radial distortion, evaluated per observation from its group's
        # coefficients. Modelling it is what lets the focal length converge:
        # a pinhole-only model must fold barrel distortion into the focal, so
        # the two are indistinguishable and the optimiser cannot move either.
        r2 = x * x + y * y
        k1 = dist[group_of_obs, 0]
        k2 = dist[group_of_obs, 1]
        factor = 1.0 + k1 * r2 + k2 * r2 * r2
        x, y = x * factor, y * factor
    f = focals[group_of_obs]
    return np.column_stack([f * x + cx, f * y + cy])


def bundle_adjust(
    poses: Dict[int, Tuple[np.ndarray, np.ndarray]],
    points3d: Dict[int, np.ndarray],
    observations: Sequence[Tuple[int, int, np.ndarray]],
    Ks: Dict[int, np.ndarray],
    *,
    groups: Optional[Dict[int, int]] = None,
    fixed_cameras: Optional[Set[int]] = None,
    refine_focal: bool = False,
    refine_distortion: bool = False,
    distortion: Optional[Dict[int, np.ndarray]] = None,
    loss: str = "huber",
    f_scale: float = 0.0,
    max_iterations: int = 60,
    verbose: bool = False,
) -> Tuple[Dict[int, Tuple[np.ndarray, np.ndarray]], Dict[int, np.ndarray], np.ndarray, BundleResult]:
    """Refine poses, points and optionally the focal length.

    Parameters
    ----------
    poses:
        ``{image_index: (R, t)}`` world-to-camera.
    points3d:
        ``{track_index: xyz}``.
    observations:
        ``[(image_index, track_index, (u, v)), ...]``.
    fixed_cameras:
        Camera indices held constant. Fixing at least one removes the
        rotation/translation gauge freedom.
    refine_focal:
        Also optimise a single shared focal length (fx = fy). Principal point
        is left fixed at the image centre - with an uncalibrated phone it is
        very weakly constrained and tends to absorb other errors.

    Returns
    -------
    (poses, points3d, K, report)
    """
    report = BundleResult()
    fixed_cameras = fixed_cameras or set()

    cam_ids = sorted(poses)
    pt_ids = sorted(points3d)
    cam_pos = {c: i for i, c in enumerate(cam_ids)}
    pt_pos = {p: i for i, p in enumerate(pt_ids)}

    valid = [(c, p, xy) for c, p, xy in observations if c in cam_pos and p in pt_pos]
    if len(valid) < 12 or len(cam_ids) < 2:
        report.note = (f"too small to adjust ({len(cam_ids)} cameras, "
                       f"{len(pt_ids)} points, {len(valid)} observations)")
        return poses, points3d, Ks, report

    cam_idx = np.array([cam_pos[c] for c, _, _ in valid], dtype=np.int64)
    pt_idx = np.array([pt_pos[p] for _, p, _ in valid], dtype=np.int64)
    uv = np.array([xy for _, _, xy in valid], dtype=np.float64)

    rotvecs = np.array([Rotation.from_matrix(poses[c][0]).as_rotvec() for c in cam_ids])
    translations = np.array([np.asarray(poses[c][1], dtype=np.float64).ravel() for c in cam_ids])
    points = np.array([np.asarray(points3d[p], dtype=np.float64) for p in pt_ids])

    free_cams = [i for i, c in enumerate(cam_ids) if c not in fixed_cameras]
    free_pos = {i: k for k, i in enumerate(free_cams)}
    n_free = len(free_cams)
    n_pts = len(pt_ids)

    groups = groups or {c: 0 for c in cam_ids}
    group_ids = sorted({groups.get(c, 0) for c in cam_ids})
    group_pos = {g: i for i, g in enumerate(group_ids)}
    n_groups = len(group_ids)

    Ks = {c: np.array(Ks[c], dtype=np.float64, copy=True) for c in cam_ids}
    focal0 = np.array([
        float(np.mean([0.5 * (Ks[c][0, 0] + Ks[c][1, 1])
                       for c in cam_ids if groups.get(c, 0) == g]))
        for g in group_ids
    ])
    group_of_obs = np.array([group_pos[groups.get(cam_ids[c], 0)] for c in cam_idx],
                            dtype=np.int64)
    cx_obs = np.array([Ks[cam_ids[c]][0, 2] for c in cam_idx])
    cy_obs = np.array([Ks[cam_ids[c]][1, 2] for c in cam_idx])

    dist0 = np.zeros((n_groups, 2), dtype=np.float64)
    if distortion:
        for g, i in group_pos.items():
            if g in distortion:
                d = np.asarray(distortion[g], dtype=np.float64).ravel()
                dist0[i, :min(2, len(d))] = d[:2]

    x0 = np.hstack([
        rotvecs[free_cams].ravel(),
        translations[free_cams].ravel(),
        points.ravel(),
        focal0 if refine_focal else [],
        dist0.ravel() if refine_distortion else [],
    ])

    def unpack(x: np.ndarray):
        rv = rotvecs.copy()
        tr = translations.copy()
        offset = 0
        if n_free:
            rv[free_cams] = x[offset:offset + 3 * n_free].reshape(n_free, 3)
            offset += 3 * n_free
            tr[free_cams] = x[offset:offset + 3 * n_free].reshape(n_free, 3)
            offset += 3 * n_free
        pts = x[offset:offset + 3 * n_pts].reshape(n_pts, 3)
        offset += 3 * n_pts
        if refine_focal:
            f = x[offset:offset + n_groups]
            offset += n_groups
        else:
            f = focal0
        d = x[offset:offset + 2 * n_groups].reshape(n_groups, 2) if refine_distortion else dist0
        return rv, tr, pts, f, d

    def residuals(x: np.ndarray) -> np.ndarray:
        rv, tr, pts, f, d = unpack(x)
        return (_project(rv, tr, pts, f, group_of_obs, cx_obs, cy_obs,
                         cam_idx, pt_idx, d) - uv).ravel()

    def sparsity() -> lil_matrix:
        m = 2 * len(valid)
        n = (6 * n_free + 3 * n_pts + (n_groups if refine_focal else 0)
             + (2 * n_groups if refine_distortion else 0))
        S = lil_matrix((m, n), dtype=np.int8)
        rows = np.arange(len(valid))
        # Each observation touches its camera's 6 pose parameters (only if that
        # camera is free) and its point's 3 coordinates - nothing else.
        is_free = np.array([cam_ids[c] not in fixed_cameras for c in cam_idx])
        fpos = np.array([free_pos.get(c, -1) for c in cam_idx])
        for k in range(3):
            sel = is_free
            S[2 * rows[sel], 3 * fpos[sel] + k] = 1
            S[2 * rows[sel] + 1, 3 * fpos[sel] + k] = 1
            S[2 * rows[sel], 3 * n_free + 3 * fpos[sel] + k] = 1
            S[2 * rows[sel] + 1, 3 * n_free + 3 * fpos[sel] + k] = 1
        base = 6 * n_free
        for k in range(3):
            S[2 * rows, base + 3 * pt_idx + k] = 1
            S[2 * rows + 1, base + 3 * pt_idx + k] = 1
        tail = (n_groups if refine_focal else 0) + (2 * n_groups if refine_distortion else 0)
        base_cam = 6 * n_free + 3 * n_pts
        for g in range(n_groups):
            rows_g = np.flatnonzero(group_of_obs == g)
            if rows_g.size == 0:
                continue
            cols = []
            if refine_focal:
                cols.append(base_cam + g)
            if refine_distortion:
                start = base_cam + (n_groups if refine_focal else 0) + 2 * g
                cols += [start, start + 1]
            for col in cols:
                S[2 * rows_g, col] = 1
                S[2 * rows_g + 1, col] = 1
        return S

    before = np.linalg.norm(residuals(x0).reshape(-1, 2), axis=1)
    report.rmse_before = float(np.sqrt((before ** 2).mean()))
    report.mean_before = float(before.mean())
    report.median_before = float(np.median(before))
    report.focal_before = float(np.mean(focal0))
    report.num_cameras = len(cam_ids)
    report.num_points = n_pts
    report.num_observations = len(valid)
    report.num_parameters = int(x0.size)

    # A robust loss needs a scale that says what "large" means *for this
    # problem right now*. A fixed 2 px is right for polishing a converged
    # model and disastrous for a model that starts tens of pixels out: Huber
    # damps residuals above f_scale by roughly f_scale/r, so when every
    # observation is far out, every gradient is damped, the trust region
    # collapses, and least_squares reports ftol/xtol convergence at a
    # solution that is nowhere near one. Measured on this project's capture:
    # adjustments terminating "successfully" at 51 px RMSE having improved
    # it by 0.00%.
    #
    # Tying the scale to the current median residual keeps roughly half the
    # observations inside the quadratic region, so there is always a usable
    # gradient, and anneals automatically - as the model improves the median
    # falls and the loss tightens towards the fixed floor on its own.
    if f_scale <= 0:
        f_scale = float(max(_MIN_F_SCALE, np.median(before)))
    report.f_scale = f_scale

    LOGGER.info("Bundle adjustment: %d cameras (%d free), %d points, %d observations, "
                "%d parameters, f_scale %.2f px",
                len(cam_ids), n_free, n_pts, len(valid), x0.size, f_scale)

    start = time.perf_counter()
    solution = least_squares(
        residuals, x0,
        jac_sparsity=sparsity(),
        method="trf",           # tolerates the rank deficiency left by the scale gauge
        loss=loss, f_scale=f_scale,
        max_nfev=max_iterations,
        x_scale="jac",
        verbose=2 if verbose else 0,
    )
    report.duration_s = time.perf_counter() - start
    report.iterations = int(solution.nfev)
    report.termination = str(solution.message)
    report.ran = True

    rv, tr, pts, f, d = unpack(solution.x)
    after = np.linalg.norm(residuals(solution.x).reshape(-1, 2), axis=1)
    report.rmse_after = float(np.sqrt((after ** 2).mean()))
    report.mean_after = float(after.mean())
    report.median_after = float(np.median(after))
    report.focal_after = float(np.mean(f))
    report.improvement_percent = round(
        100.0 * (report.rmse_before - report.rmse_after) / max(1e-12, report.rmse_before), 3)

    new_poses = {c: (Rotation.from_rotvec(rv[i]).as_matrix(), tr[i])
                 for i, c in enumerate(cam_ids)}
    new_points = {p: pts[i] for i, p in enumerate(pt_ids)}
    if refine_focal:
        for c in cam_ids:
            Ks[c][0, 0] = Ks[c][1, 1] = float(f[group_pos[groups.get(c, 0)]])
    report.distortion = {g: d[group_pos[g]].copy() for g in group_ids}

    # The termination reason and evaluation count are the difference between
    # "converged" and "gave up", and an adjustment that stalls looks identical
    # to one that succeeded if only the RMSE is reported. Both are logged.
    # RMSE is reported alongside the median because with a robust loss they
    # answer different questions. The optimiser minimises the robust cost, so
    # a large RMSE that refuses to move usually means a handful of gross
    # outliers the loss is correctly ignoring - not a failed adjustment. The
    # median is what shows whether the bulk of the model is improving.
    LOGGER.info("  RMSE %.4f -> %.4f px (%.2f%%), median %.4f -> %.4f px, "
                "in %.1f s [nfev=%d, %s]%s",
                report.rmse_before, report.rmse_after, report.improvement_percent,
                report.median_before, report.median_after,
                report.duration_s, report.iterations, report.termination,
                ((", focals " + " ".join(f"{a:.1f}->{b:.1f}"
                                         for a, b in zip(focal0, np.atleast_1d(f))))
                 if refine_focal else "")
                + ((", k1 " + " ".join(f"{v:.5f}" for v in np.atleast_2d(d)[:, 0]))
                   if refine_distortion else ""))
    return new_poses, new_points, Ks, report
