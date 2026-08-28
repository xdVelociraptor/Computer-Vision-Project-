"""The assignment constraint, enforced as a test.

    You may call some existing function in your development but cannot call
    existing functions that directly implement the SfM task.

Checking this by hand does not scale and does not survive edits. It was checked
manually twice during development and both passes missed something: the first
left `pycolmap` in requirements.txt and an Open3D Poisson meshing option in the
config, and neither was caught until a third look. A test runs every time.
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT = PROJECT_ROOT / "scripts" / "audit_sfm_compliance.py"


class ComplianceTests(unittest.TestCase):
    def test_no_sfm_implementation_is_called(self) -> None:
        result = subprocess.run(
            [sys.executable, str(AUDIT)],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        )
        self.assertEqual(result.returncode, 0, msg=f"\n{result.stdout}\n{result.stderr}")

    def test_no_sfm_package_is_declared_as_a_dependency(self) -> None:
        """A declared dependency is a claim about what the project needs."""
        text = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
        for package in ("pycolmap", "opensfm", "pyceres", "gtsam"):
            self.assertNotIn(package, text,
                             msg=f"{package} must not be a dependency of this project")

    def test_the_audit_actually_detects_a_violation(self) -> None:
        """A check that cannot fail is worse than no check.

        Plant a real violation, confirm the audit rejects it, and remove it
        again — so a future refactor that quietly defangs the audit is caught
        here rather than by a marker.
        """
        probe = PROJECT_ROOT / "scripts" / "_compliance_probe_tmp.py"
        probe.write_text(
            "import cv2, numpy as np\n"
            "E, mask = cv2.findEssentialMat(np.zeros((8, 2)), np.zeros((8, 2)), np.eye(3))\n",
            encoding="utf-8")
        try:
            result = subprocess.run(
                [sys.executable, str(AUDIT)],
                capture_output=True, text=True, cwd=str(PROJECT_ROOT))
            self.assertEqual(result.returncode, 1,
                             msg="the audit failed to detect a planted violation")
            self.assertIn("findEssentialMat", result.stdout)
        finally:
            probe.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
