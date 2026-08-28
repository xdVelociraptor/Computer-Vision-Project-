"""Check that no code calls a function which directly implements the SfM task.

    python scripts/audit_sfm_compliance.py

Exit code 0 means clean; 1 means a violation was found.

Why this parses rather than greps
---------------------------------
A plain text search cannot tell these three things apart:

    E, mask = cv2.findEssentialMat(pts1, pts2, K)     <- a violation
    # we implement findEssentialMat ourselves         <- a comment about one
    def bundle_adjust(poses, points, ...):            <- our own implementation

The first must fail the audit, the second is documentation, and the third is
precisely the work the assignment asks for. An earlier grep-based version of
this script reported 68 findings on a compliant tree, of which every single one
was a comment, a docstring, or one of our own function definitions - a result
that is worse than no check at all, because it trains you to ignore it.

So Python files are parsed into a syntax tree and only *calls*, *imports* and
*subprocess invocations* are inspected. Comments and docstrings are invisible
to the parser and therefore cannot raise a false alarm. Shell scripts, which
cannot be parsed this way, fall back to a text scan with comment lines removed.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IGNORED_PARTS = {".venv", "data", "outputs", "work", "__pycache__", ".git"}

#: Library functions that carry out a reconstruction step themselves.
FORBIDDEN_FUNCTIONS = {
    "findEssentialMat": "essential-matrix estimation",
    "findFundamentalMat": "fundamental-matrix estimation",
    "recoverPose": "relative pose from E",
    "decomposeEssentialMat": "essential-matrix decomposition",
    "decomposeHomographyMat": "homography decomposition",
    "findHomography": "homography estimation",
    "triangulatePoints": "triangulation",
    "solvePnP": "camera resection",
    "solvePnPRansac": "camera resection",
    "solvePnPGeneric": "camera resection",
    "correctMatches": "optimal correspondence correction",
    "stereoRectify": "stereo rectification",
    "reprojectImageTo3D": "disparity to depth",
    "calibrateCamera": "camera calibration",
    "estimateAffine3D": "3D alignment",
    "registration_icp": "point-cloud registration",
    "registration_ransac_based_on_feature_matching": "point-cloud registration",
    "create_from_point_cloud_poisson": "surface reconstruction",
    "ScalableTSDFVolume": "TSDF fusion",
    "UniformTSDFVolume": "TSDF fusion",
}

#: Packages that are, in whole, an SfM implementation.
FORBIDDEN_MODULES = {"pycolmap", "opensfm", "colmap", "pyceres", "gtsam", "g2o"}

#: Executables that are an SfM implementation.
FORBIDDEN_BINARIES = re.compile(r"\bcolmap(?:\.exe|\.bat)?\b", re.IGNORECASE)


class CallAudit(ast.NodeVisitor):
    """Collects forbidden imports and calls from one parsed module."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.findings: list[str] = []

    def _report(self, line: int, detail: str) -> None:
        rel = self.path.relative_to(PROJECT_ROOT)
        self.findings.append(f"{rel}:{line}: {detail}")

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root in FORBIDDEN_MODULES:
                self._report(node.lineno, f"imports {alias.name!r}, an SfM implementation")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        root = (node.module or "").split(".")[0]
        if root in FORBIDDEN_MODULES:
            self._report(node.lineno, f"imports from {node.module!r}, an SfM implementation")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = None
        if isinstance(node.func, ast.Attribute):
            name = node.func.attr
        elif isinstance(node.func, ast.Name):
            name = node.func.id

        if name in FORBIDDEN_FUNCTIONS:
            self._report(node.lineno, f"calls {name}() - {FORBIDDEN_FUNCTIONS[name]}")

        # A reconstruction engine invoked as a subprocess is still a call.
        if name in {"run", "check_call", "check_output", "Popen", "call", "run_command"}:
            for arg in ast.walk(node):
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    if FORBIDDEN_BINARIES.search(arg.value):
                        self._report(node.lineno, "invokes the COLMAP executable")
                        break
        self.generic_visit(node)


def audit_python(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
    except SyntaxError as exc:
        rel = path.relative_to(PROJECT_ROOT)
        return [f"{rel}: could not be parsed ({exc.msg}); audit it by hand"]
    visitor = CallAudit(path)
    visitor.visit(tree)
    return visitor.findings


def audit_shell(path: Path) -> list[str]:
    """Text scan for shell scripts, with comment lines stripped."""
    findings = []
    rel = path.relative_to(PROJECT_ROOT)
    for number, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        line = raw.split("#", 1)[0]
        if FORBIDDEN_BINARIES.search(line):
            findings.append(f"{rel}:{number}: invokes the COLMAP executable")
    return findings


def main() -> int:
    findings: list[str] = []
    scanned = 0
    for path in sorted(PROJECT_ROOT.rglob("*")):
        if not path.is_file() or any(part in IGNORED_PARTS for part in path.parts):
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        if path.suffix == ".py":
            scanned += 1
            findings.extend(audit_python(path))
        elif path.suffix.lower() in {".ps1", ".bat", ".cmd", ".sh"}:
            scanned += 1
            findings.extend(audit_shell(path))

    if findings:
        print(f"COMPLIANCE FAILED - {len(findings)} violation(s) in {scanned} files:\n")
        for line in findings:
            print(f"  {line}")
        return 1

    print(f"COMPLIANCE OK - {scanned} files scanned, no call to an SfM implementation.")
    print("\nChecked for:")
    print(f"  - calls to {len(FORBIDDEN_FUNCTIONS)} library functions that perform a")
    print("    reconstruction step (essential/fundamental estimation, pose recovery,")
    print("    triangulation, resection, registration, TSDF, Poisson meshing)")
    print(f"  - imports of {len(FORBIDDEN_MODULES)} SfM packages")
    print("  - subprocess invocation of the COLMAP executable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
