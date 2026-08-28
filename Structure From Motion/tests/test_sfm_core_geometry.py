"""Unit tests for the from-scratch reconstruction geometry.

Every test builds a synthetic scene with known cameras and known 3D points,
runs one component, and checks the recovered answer against the truth it was
generated from. That is the only way to tell a subtly wrong implementation
from a correct one: on real photographs a broken estimator still returns
plausible-looking numbers.

These cover the components that the incremental mapper depends on. If one of
them regresses, everything downstream silently degrades - which is exactly
what happened to this project once, when bundle adjustment stopped converging
and the failure showed up only as "fewer images registered".
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.sfm_core.geometry import (  # noqa: E402
    decompose_essential,
    ransac_essential,
    ransac_pnp,
    ransac_similarity,
    reprojection_error,
    rotation_angle_deg,
    select_pose_by_cheirality,
    similarity_from_points,
    triangulate_dlt,
    triangulate_multiview,
)
from src.sfm_core.p3p import p3p  # noqa: E402


def rotation(axis: int, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    if axis == 0:
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 1:
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def project(K: np.ndarray, R: np.ndarray, t: np.ndarray, X: np.ndarray) -> np.ndarray:
    cam = X @ R.T + t
    return (cam[:, :2] / cam[:, 2:3]) * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])


class SyntheticScene:
    """A two-view scene with a known relative pose and known structure."""

    def __init__(self, seed: int = 7, n: int = 220) -> None:
        rng = np.random.default_rng(seed)
        self.K = np.array([[1150.0, 0, 600.0], [0, 1150.0, 800.0], [0, 0, 1.0]])
        self.R1, self.t1 = np.eye(3), np.zeros(3)
        self.R2 = rotation(1, 0.22) @ rotation(0, -0.06)
        self.t2 = np.array([-1.4, 0.05, 0.12])
        self.points = rng.uniform([-2.0, -1.4, 5.0], [2.0, 1.4, 11.0], size=(n, 3))
        self.uv1 = project(self.K, self.R1, self.t1, self.points)
        self.uv2 = project(self.K, self.R2, self.t2, self.points)


class EssentialMatrixTests(unittest.TestCase):
    def test_recovers_relative_rotation(self) -> None:
        s = SyntheticScene()
        result = ransac_essential(s.uv1, s.uv2, s.K, s.K, threshold_px=1.0, seed=0)
        self.assertIsNotNone(result.model)
        R, t, mask, _ = select_pose_by_cheirality(
            decompose_essential(result.model), s.uv1, s.uv2, s.K, K2=s.K)
        self.assertIsNotNone(R)
        # Relative rotation must match; translation only up to scale.
        self.assertLess(rotation_angle_deg(R.T @ s.R2), 0.5)
        direction = t.ravel() / np.linalg.norm(t)
        truth = s.t2 / np.linalg.norm(s.t2)
        # Clip before arccos: an exact match can land marginally above 1.0 in
        # floating point, which returns nan rather than 0 degrees.
        cosine = float(np.clip(abs(direction @ truth), -1.0, 1.0))
        self.assertLess(np.degrees(np.arccos(cosine)), 1.0)

    def test_rejects_outliers(self) -> None:
        s = SyntheticScene()
        rng = np.random.default_rng(3)
        uv2 = s.uv2.copy()
        corrupt = rng.choice(len(uv2), size=len(uv2) // 4, replace=False)
        uv2[corrupt] += rng.normal(scale=90.0, size=(len(corrupt), 2))

        result = ransac_essential(s.uv1, uv2, s.K, s.K, threshold_px=1.5, seed=0)
        self.assertIsNotNone(result.model)
        # The consensus set must exclude the corrupted correspondences.
        flagged = np.zeros(len(uv2), dtype=bool)
        flagged[corrupt] = True
        self.assertLess(result.inliers[flagged].mean(), 0.15)
        self.assertGreater(result.inliers[~flagged].mean(), 0.85)


class TriangulationTests(unittest.TestCase):
    def test_two_view_dlt_is_exact_on_noiseless_input(self) -> None:
        s = SyntheticScene()
        P1 = s.K @ np.hstack([s.R1, s.t1.reshape(3, 1)])
        P2 = s.K @ np.hstack([s.R2, s.t2.reshape(3, 1)])
        X = triangulate_dlt(P1, P2, s.uv1, s.uv2)
        self.assertLess(np.abs(X - s.points).max(), 1e-8)

    def test_multiview_beats_two_view_under_noise(self) -> None:
        """Long tracks are the main lever on accuracy - verify that directly.

        Averaged over many noise draws, not one. Using more observations lowers
        the *expected* error; on any single draw two views can happen to win,
        so asserting on one sample would make this test flaky rather than
        informative.
        """
        rng = np.random.default_rng(5)
        poses = None
        two_errors, many_errors = [], []

        for trial in range(120):
            s = SyntheticScene(seed=1000 + trial, n=1)
            if poses is None:
                poses = [(s.R1, s.t1), (s.R2, s.t2)] + [
                    (rotation(1, 0.22 + 0.08 * k), np.array([-1.4 - 0.5 * k, 0.05, 0.1]))
                    for k in range(1, 5)
                ]
            obs, projections = [], []
            for R, t in poses:
                uv = project(s.K, R, t, s.points)[0]
                obs.append(uv + rng.normal(scale=0.6, size=2))
                projections.append(s.K @ np.hstack([R, t.reshape(3, 1)]))

            truth = s.points[0]
            two = triangulate_dlt(projections[0], projections[1],
                                  np.array([obs[0]]), np.array([obs[1]]))[0]
            many = triangulate_multiview(projections, obs)
            two_errors.append(np.linalg.norm(two - truth))
            many_errors.append(np.linalg.norm(many - truth))

        self.assertLess(float(np.median(many_errors)), float(np.median(two_errors)))


class ResectionTests(unittest.TestCase):
    def test_p3p_recovers_pose_from_three_points(self) -> None:
        s = SyntheticScene()
        pts = s.points[:3]
        cam = pts @ s.R2.T + s.t2
        bearings = cam / np.linalg.norm(cam, axis=1, keepdims=True)

        best = min(
            (rotation_angle_deg(R.T @ s.R2) for R, _ in p3p(pts, bearings)),
            default=None)
        self.assertIsNotNone(best)
        self.assertLess(best, 1e-3)

    def test_pnp_ransac_survives_heavy_contamination(self) -> None:
        s = SyntheticScene()
        rng = np.random.default_rng(4)
        uv = s.uv2.copy()
        corrupt = rng.choice(len(uv), size=int(len(uv) * 0.4), replace=False)
        uv[corrupt] += rng.normal(scale=140.0, size=(len(corrupt), 2))

        R, t, inliers = ransac_pnp(s.points, uv, s.K, threshold_px=3.0, seed=1)
        self.assertIsNotNone(R)
        self.assertLess(rotation_angle_deg(R.T @ s.R2), 1.0)
        self.assertLess(np.linalg.norm(t - s.t2), 0.1)
        flagged = np.zeros(len(uv), dtype=bool)
        flagged[corrupt] = True
        self.assertLess(inliers[flagged].mean(), 0.1)


class ReprojectionTests(unittest.TestCase):
    def test_error_is_zero_at_the_true_pose(self) -> None:
        s = SyntheticScene()
        err = reprojection_error(s.points, s.uv2, s.K, s.R2, s.t2)
        self.assertLess(err.max(), 1e-6)

    def test_error_grows_when_the_pose_is_wrong(self) -> None:
        s = SyntheticScene()
        wrong = rotation(2, 0.02) @ s.R2
        err = reprojection_error(s.points, s.uv2, s.K, wrong, s.t2)
        self.assertGreater(np.median(err), 5.0)


class SimilarityTests(unittest.TestCase):
    """Used to align independently reconstructed models (multimodel.py)."""

    def test_recovers_scale_rotation_translation(self) -> None:
        rng = np.random.default_rng(2)
        src = rng.normal(size=(180, 3)) * 2.0
        R = rotation(0, 0.7) @ rotation(1, -0.4)
        scale, t = 2.35, np.array([4.0, -1.5, 0.9])
        dst = scale * (src @ R.T) + t

        s_hat, R_hat, t_hat = similarity_from_points(src, dst)
        self.assertAlmostEqual(s_hat, scale, places=9)
        self.assertLess(rotation_angle_deg(R_hat.T @ R), 1e-6)
        self.assertLess(np.linalg.norm(t_hat - t), 1e-8)

    def test_refuses_to_mirror(self) -> None:
        """An unguarded SVD returns det(R) = -1, which is not a rotation."""
        rng = np.random.default_rng(9)
        src = rng.normal(size=(60, 3))
        dst = src.copy()
        dst[:, 0] *= -1.0
        _, R_hat, _ = similarity_from_points(src, dst)
        self.assertGreater(np.linalg.det(R_hat), 0.0)

    def test_ransac_survives_eighty_percent_outliers(self) -> None:
        rng = np.random.default_rng(1)
        src = rng.normal(size=(300, 3)) * 2.0
        R = rotation(0, 0.7) @ rotation(1, -0.4)
        scale, t = 2.3, np.array([5.0, -2.0, 1.0])
        dst = scale * (src @ R.T) + t
        bad = rng.choice(300, 240, replace=False)
        dst[bad] += rng.normal(scale=20.0, size=(240, 3))

        model, inliers = ransac_similarity(src, dst, threshold=0.05, seed=1)
        self.assertIsNotNone(model)
        s_hat, R_hat, _ = model
        self.assertAlmostEqual(s_hat, scale, places=6)
        self.assertLess(rotation_angle_deg(R_hat.T @ R), 1e-6)
        self.assertEqual(int(inliers.sum()), 60)


if __name__ == "__main__":
    unittest.main(verbosity=2)
