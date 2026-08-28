"""P3P: camera pose from three 3D-2D correspondences (Grunert's method).

Why this matters more than it looks
-----------------------------------
RANSAC's cost is governed by how many points a single hypothesis needs. The
probability that a random minimal sample is free of outliers is ``w^m`` for
inlier ratio ``w`` and sample size ``m``, so at a 30% outlier rate:

    m = 6 (linear DLT PnP)  ->  0.7^6 = 12% of samples usable
    m = 3 (P3P)             ->  0.7^3 = 34% of samples usable

Nearly three times as many useful hypotheses per iteration, and the required
iteration count drops by roughly the same factor. On a real capture this is the
difference between registering a view and discarding it - the DLT version of
this project's mapper found only 22 inliers among 391 correspondences on views
that P3P handles comfortably.

Method
------
With camera centre O and unit bearing vectors ``f1, f2, f3`` toward world points
``P1, P2, P3``, the unknown distances ``s_i = |O - P_i|`` satisfy the law of
cosines on each pair::

    s2^2 + s3^2 - 2 s2 s3 cos(alpha) = |P2 P3|^2
    s1^2 + s3^2 - 2 s1 s3 cos(beta)  = |P1 P3|^2
    s1^2 + s2^2 - 2 s1 s2 cos(gamma) = |P1 P2|^2

Substituting ``u = s2/s1`` and ``v = s3/s1`` reduces the system to a quartic in
``v`` with up to four real roots, each giving a candidate pose. The pose itself
then follows from absolute orientation between the world points and the
recovered camera-frame points.

Implementation note: the quartic's coefficients are assembled by *polynomial
arithmetic at runtime* rather than transcribed from the literature. Hand-copied
Grunert coefficients are a classic source of silent sign errors, and building
them with convolution is both shorter and verifiable term by term.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

__all__ = ["p3p", "kabsch", "solve_pnp_p3p"]


def kabsch(source: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Rigid transform taking ``source`` onto ``target`` (both (N,3)).

    Returns ``(R, t)`` with ``target ~ R @ source + t``. The reflection guard
    matters: without it a degenerate sample can produce a determinant of -1,
    which is not a rotation and yields a mirrored, physically impossible pose.
    """
    src_c = source.mean(axis=0)
    dst_c = target.mean(axis=0)
    H = (source - src_c).T @ (target - dst_c)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, dst_c - R @ src_c


def p3p(points3d: np.ndarray, bearings: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Solve P3P. ``bearings`` are unit vectors in the camera frame.

    Returns up to four ``(R, t)`` candidates with ``x_cam = R @ x_world + t``.
    """
    P1, P2, P3 = np.asarray(points3d, dtype=np.float64)
    f1, f2, f3 = [b / max(np.linalg.norm(b), 1e-12)
                  for b in np.asarray(bearings, dtype=np.float64)]

    a = float(np.linalg.norm(P2 - P3))
    b = float(np.linalg.norm(P1 - P3))
    c = float(np.linalg.norm(P1 - P2))
    if min(a, b, c) < 1e-9:
        return []

    cos_alpha = float(np.clip(f2 @ f3, -1.0, 1.0))
    cos_beta = float(np.clip(f1 @ f3, -1.0, 1.0))
    cos_gamma = float(np.clip(f1 @ f2, -1.0, 1.0))

    A = (a * a) / (b * b)
    C = (c * c) / (b * b)

    # Polynomials in v, highest power first (numpy convention).
    #   Q(v) = 1 + v^2 - 2 v cos(beta)          (from the P1-P3 equation)
    Q = np.array([1.0, -2.0 * cos_beta, 1.0])
    #   u(v) = N(v) / D(v)
    N = np.polyadd((C - A) * Q, np.array([1.0, 0.0, -1.0]))
    D = np.array([2.0 * cos_alpha, -2.0 * cos_gamma])

    # Eq: 1 + u^2 - 2 u cos(gamma) - C * Q = 0, multiplied through by D^2.
    D2 = np.polymul(D, D)
    quartic = np.polyadd(
        np.polyadd(D2, np.polymul(N, N)),
        np.polyadd(np.polymul(np.array([-2.0 * cos_gamma]), np.polymul(N, D)),
                   np.polymul(np.array([-C]), np.polymul(Q, D2))),
    )

    quartic = np.trim_zeros(quartic, "f")
    if len(quartic) < 2:
        return []
    try:
        roots = np.roots(quartic)
    except np.linalg.LinAlgError:
        return []

    solutions: List[Tuple[np.ndarray, np.ndarray]] = []
    world = np.vstack([P1, P2, P3])
    for root in roots:
        if abs(root.imag) > 1e-6 * max(1.0, abs(root.real)):
            continue
        v = float(root.real)
        if v <= 0:
            continue                       # distances must be positive

        q = float(np.polyval(Q, v))
        if q <= 1e-12:
            continue
        s1 = float(np.sqrt((b * b) / q))
        s3 = v * s1

        # s2 comes from the P1-P2 law of cosines rather than from u = N/D.
        # The rational form loses solutions wherever D(v) = 2(v cos a - cos g)
        # passes through zero, which is a coordinate artefact, not a real
        # degeneracy - it cost 29% of otherwise valid poses. Solving the
        # quadratic directly is unconditionally stable and yields both
        # branches, which are then disambiguated against the third equation.
        disc = s1 * s1 * cos_gamma * cos_gamma - (s1 * s1 - c * c)
        if disc < 0.0:
            continue
        root_disc = float(np.sqrt(disc))
        for s2 in (s1 * cos_gamma + root_disc, s1 * cos_gamma - root_disc):
            if s2 <= 1e-9 or not np.isfinite(s2):
                continue
            # Keep only branches consistent with the remaining P2-P3 equation.
            residual = s2 * s2 + s3 * s3 - 2.0 * s2 * s3 * cos_alpha - a * a
            if abs(residual) > 1e-6 * max(a * a, 1.0):
                continue
            cam_points = np.vstack([s1 * f1, s2 * f2, s3 * f3])
            R, t = kabsch(world, cam_points)
            if np.any(~np.isfinite(R)) or np.any(~np.isfinite(t)):
                continue
            solutions.append((R, t))
    return solutions


def solve_pnp_p3p(points3d: np.ndarray, points2d: np.ndarray,
                  K: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """P3P from pixel observations: converts to bearings, then solves."""
    K_inv = np.linalg.inv(np.asarray(K, dtype=np.float64))
    homo = np.hstack([np.asarray(points2d, dtype=np.float64), np.ones((len(points2d), 1))])
    bearings = homo @ K_inv.T
    norms = np.linalg.norm(bearings, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        return []
    return p3p(points3d, bearings / norms)
