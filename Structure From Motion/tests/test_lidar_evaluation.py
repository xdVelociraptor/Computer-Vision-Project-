from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.sfm_core.lidar_evaluation import apply_similarity, estimate_similarity, nearest_neighbors


class LidarEvaluationTests(unittest.TestCase):
    def test_similarity_recovery(self) -> None:
        rng = np.random.default_rng(4)
        source = rng.normal(size=(80, 3))
        angle = 0.37
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
        )
        target = apply_similarity(source, 2.3, rotation, np.array([1.2, -0.8, 0.4]))
        scale, recovered_rotation, translation = estimate_similarity(source, target)
        recovered = apply_similarity(source, scale, recovered_rotation, translation)
        self.assertLess(float(np.max(np.abs(recovered - target))), 1e-9)

    def test_nearest_neighbors(self) -> None:
        target = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        source = np.array([[0.2, 0.0, 0.0], [2.7, 0.0, 0.0]])
        distances, indices = nearest_neighbors(source, target, chunk_size=1)
        np.testing.assert_allclose(distances, [0.2, 0.3])
        np.testing.assert_array_equal(indices, [0, 1])


if __name__ == "__main__":
    unittest.main()
