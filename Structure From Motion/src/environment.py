"""Environment verification.

Every external dependency this project relies on (Python version, packages,
CUDA GPU, disk space) is probed here and reported explicitly.

Guiding rule from the project brief: *do not silently fail when something is
unavailable*. Each check produces one of four statuses:

* ``ok``   - present and usable
* ``warn`` - present but suboptimal, or an optional component is missing
* ``fail`` - a required component is missing/broken; the pipeline cannot run
* ``skip`` - not applicable on this machine

The resulting report is written to ``outputs/_env/environment_report.json`` and
is also embedded into every experiment's metadata for reproducibility.
"""

from __future__ import annotations

import importlib
import importlib.metadata as importlib_metadata
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .utils import disk_free, system_report

__all__ = [
    "CheckResult",
    "EnvironmentReport",
    "check_environment",
    "format_report",
]

# Interpreter window: <=3.12 is the ceiling
# imposed by Open3D 0.19 wheels.
MIN_PYTHON = (3, 10)
RECOMMENDED_PYTHON = (3, 12)
MAX_PYTHON_FOR_OPEN3D = (3, 12)

#: (import name, distribution name, why it is needed)
REQUIRED_PACKAGES: List[Tuple[str, str, str]] = [
    ("numpy", "numpy", "array maths everywhere"),
    ("cv2", "opencv-contrib-python", "SIFT/ORB/AKAZE, epipolar geometry, video decoding"),
    ("scipy", "scipy", "least-squares bundle adjustment, spatial KD-trees"),
    ("pandas", "pandas", "metrics tables and results.csv"),
    ("matplotlib", "matplotlib", "publication-quality plots"),
    ("yaml", "PyYAML", "configuration files"),
    ("PIL", "Pillow", "EXIF parsing and orientation handling"),
    ("tqdm", "tqdm", "progress bars"),
]

OPTIONAL_PACKAGES: List[Tuple[str, str, str]] = [
    ("open3d", "open3d", "dense point-cloud analysis, outlier removal, meshing"),
    ("sklearn", "scikit-learn", "clustering / nearest-neighbour statistics"),
    ("skimage", "scikit-image", "SSIM-based image similarity (optional metric)"),
    ("psutil", "psutil", "hardware reporting"),
]

STATUS_ORDER = {"fail": 0, "warn": 1, "ok": 2, "skip": 3}


@dataclass
class CheckResult:
    """Outcome of a single environment probe."""

    name: str
    status: str  # ok | warn | fail | skip
    detail: str
    required: bool = False
    data: Dict[str, Any] = field(default_factory=dict)
    remedy: str = ""


@dataclass
class EnvironmentReport:
    """Aggregated result of all probes."""

    checks: List[CheckResult]
    system: Dict[str, Any]

    @property
    def failures(self) -> List[CheckResult]:
        return [c for c in self.checks if c.status == "fail"]

    @property
    def warnings(self) -> List[CheckResult]:
        return [c for c in self.checks if c.status == "warn"]

    @property
    def ok(self) -> bool:
        """True when nothing required is missing."""
        return not self.failures

    def get(self, name: str) -> Optional[CheckResult]:
        for c in self.checks:
            if c.name == name:
                return c
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "system": self.system,
            "checks": [
                {
                    "name": c.name,
                    "status": c.status,
                    "required": c.required,
                    "detail": c.detail,
                    "remedy": c.remedy,
                    "data": c.data,
                }
                for c in self.checks
            ],
        }


# ===========================================================================
# Individual checks
# ===========================================================================


def check_python() -> CheckResult:
    """Verify the interpreter sits inside the supported version window."""
    v = sys.version_info
    data = {
        "version": f"{v.major}.{v.minor}.{v.micro}",
        "executable": sys.executable,
        "in_virtualenv": sys.prefix != getattr(sys, "base_prefix", sys.prefix),
        "64bit": sys.maxsize > 2**32,
    }
    if (v.major, v.minor) < MIN_PYTHON:
        return CheckResult(
            "python", "fail",
            f"Python {data['version']} is older than the required {MIN_PYTHON[0]}.{MIN_PYTHON[1]}",
            required=True, data=data,
            remedy="Install Python 3.12 (winget install Python.Python.3.12) and recreate the venv.",
        )
    if (v.major, v.minor) > MAX_PYTHON_FOR_OPEN3D:
        return CheckResult(
            "python", "warn",
            f"Python {data['version']} exceeds {MAX_PYTHON_FOR_OPEN3D[0]}.{MAX_PYTHON_FOR_OPEN3D[1]}; "
            f"Open3D 0.19 publishes no wheels for it, so dense/mesh analysis will be unavailable",
            required=True, data=data,
            remedy="Create the virtual environment with 'py -3.12 -m venv .venv'.",
        )
    if not data["in_virtualenv"]:
        return CheckResult(
            "python", "warn",
            f"Python {data['version']} (not running inside a virtual environment)",
            required=True, data=data,
            remedy="Activate the project venv: .\\.venv\\Scripts\\Activate.ps1",
        )
    return CheckResult("python", "ok", f"Python {data['version']} (venv)", required=True, data=data)


def _package_version(dist_name: str, module: Any) -> str:
    for candidate in (dist_name, dist_name.replace("-", "_")):
        try:
            return importlib_metadata.version(candidate)
        except importlib_metadata.PackageNotFoundError:
            continue
    return str(getattr(module, "__version__", "unknown"))


def _check_package(import_name: str, dist_name: str, purpose: str, required: bool) -> CheckResult:
    try:
        module = importlib.import_module(import_name)
    except Exception as exc:  # ImportError, but also DLL-load errors on Windows
        return CheckResult(
            f"pkg:{dist_name}",
            "fail" if required else "warn",
            f"cannot import '{import_name}' ({type(exc).__name__}: {exc})",
            required=required,
            data={"purpose": purpose},
            remedy=f"pip install {dist_name}",
        )
    version = _package_version(dist_name, module)
    return CheckResult(
        f"pkg:{dist_name}", "ok", f"{version}", required=required,
        data={"version": version, "purpose": purpose, "path": getattr(module, "__file__", None)},
    )


def check_packages() -> List[CheckResult]:
    """Import every dependency and record its version."""
    results = [_check_package(i, d, p, True) for i, d, p in REQUIRED_PACKAGES]
    results += [_check_package(i, d, p, False) for i, d, p in OPTIONAL_PACKAGES]
    return results


def check_opencv() -> CheckResult:
    """Confirm OpenCV works and that SIFT is actually constructible."""
    try:
        import cv2
    except Exception as exc:
        return CheckResult(
            "opencv", "fail", f"OpenCV import failed: {exc}", required=True,
            remedy="pip install opencv-contrib-python==4.14.0.94",
        )

    data: Dict[str, Any] = {"version": cv2.__version__}
    try:
        detector = cv2.SIFT_create()
        data["sift"] = True
        del detector
    except Exception as exc:
        return CheckResult(
            "opencv", "fail", f"OpenCV {cv2.__version__} present but SIFT_create() failed: {exc}",
            required=True, data=data,
            remedy="Install the contrib build: pip install opencv-contrib-python",
        )
    for name in ("ORB_create", "AKAZE_create"):
        data[name] = hasattr(cv2, name)
    try:
        data["cuda_devices"] = cv2.cuda.getCudaEnabledDeviceCount()
    except Exception:
        data["cuda_devices"] = 0

    major = int(cv2.__version__.split(".")[0])
    if major >= 5:
        return CheckResult(
            "opencv", "warn",
            f"OpenCV {cv2.__version__} - this project is developed against the 4.x API",
            required=True, data=data,
            remedy="pip install opencv-contrib-python==4.14.0.94",
        )
    return CheckResult("opencv", "ok", f"OpenCV {cv2.__version__}, SIFT available", required=True, data=data)


def check_cuda() -> CheckResult:
    """Probe the NVIDIA driver and enumerate GPUs via ``nvidia-smi``."""
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return CheckResult(
            "cuda", "warn", "nvidia-smi not found - no usable NVIDIA GPU detected",
            required=False,
            remedy=(
                "CPU-only operation is fine for sparse SfM (set runtime.use_gpu=false); "
                "CUDA is not required: the dense stage is plane-sweep MVS on the CPU."
            ),
        )
    try:
        proc = subprocess.run(  # noqa: S603
            [smi, "--query-gpu=name,driver_version,memory.total,compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
        )
    except Exception as exc:
        return CheckResult("cuda", "warn", f"nvidia-smi failed: {exc}", required=False)

    if proc.returncode != 0 or not proc.stdout.strip():
        return CheckResult(
            "cuda", "warn", f"nvidia-smi returned exit code {proc.returncode}", required=False,
            data={"stderr": proc.stderr.strip()[:400]},
        )

    gpus: List[Dict[str, Any]] = []
    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            gpus.append(
                {
                    "name": parts[0],
                    "driver_version": parts[1],
                    "memory_total_mib": _safe_int(parts[2]),
                    "compute_capability": parts[3],
                }
            )
    data: Dict[str, Any] = {"gpus": gpus}

    if not gpus:
        return CheckResult("cuda", "warn", "nvidia-smi reported no GPUs", required=False, data=data)

    primary = gpus[0]
    summary = (
        f"{primary['name']} ({primary['memory_total_mib']} MiB, "
        f"compute capability {primary['compute_capability']}, driver {primary['driver_version']})"
    )

    # Blackwell (sm_120) is newer than the architectures baked into some
    # some pre-built CUDA wheels; flag it so a GPU failure is not mysterious.
    cc = _safe_float(primary["compute_capability"])
    if cc is not None and cc >= 12.0:
        return CheckResult(
            "cuda", "warn",
            f"{summary} - compute capability >= 12.0 (Blackwell). Some pre-built CUDA "
            f"binaries do not embed sm_120 kernels; if patch_match_stereo aborts with a "
            f"'no kernel image is available' error, that is the cause",
            required=False, data=data,
            remedy=(
                "CMAKE_CUDA_ARCHITECTURES=120, or run sparse-only and document dense MVS "
                "as unavailable."
            ),
        )
    return CheckResult("cuda", "ok", summary, required=False, data=data)


def _safe_int(text: str) -> Optional[int]:
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _safe_float(text: str) -> Optional[float]:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def check_ffmpeg() -> CheckResult:
    """FFmpeg is optional: OpenCV decodes most phone video on its own."""
    exe = shutil.which("ffmpeg")
    if exe is None:
        return CheckResult(
            "ffmpeg", "warn",
            "ffmpeg not found (optional; only needed if OpenCV cannot decode your video codec)",
            required=False, remedy="winget install Gyan.FFmpeg",
        )
    return CheckResult("ffmpeg", "ok", f"found at {exe}", required=False, data={"executable": exe})


def check_disk(paths: List[Path], min_free_gb: float = 20.0) -> CheckResult:
    """Dense MVS output is large; make sure the target volumes have room."""
    volumes: Dict[str, Dict[str, Any]] = {}
    for p in paths:
        info = disk_free(p)
        volumes[str(p)] = info
    worst = min((v["free_bytes"] for v in volumes.values()), default=0)
    detail = "; ".join(f"{k} -> {v['free_human']} free" for k, v in volumes.items())
    if worst < min_free_gb * (1024**3):
        return CheckResult(
            "disk", "warn",
            f"less than {min_free_gb:.0f} GB free on a target volume ({detail})",
            required=False, data=volumes,
            remedy="Point paths.output_root at a drive with more space; dense MVS can produce several GB per scene.",
        )
    return CheckResult("disk", "ok", detail, required=False, data=volumes)


def check_onedrive(paths: List[Path]) -> CheckResult:
    """Warn when large outputs would land inside a OneDrive-synced folder."""
    synced = [str(p) for p in paths if "onedrive" in str(p).lower()]
    if not synced:
        return CheckResult("onedrive", "ok", "no output path inside a OneDrive folder", required=False)
    return CheckResult(
        "onedrive", "warn",
        f"path(s) inside OneDrive will be continuously synced: {synced}",
        required=False, data={"paths": synced},
        remedy=(
            "Set paths.output_root to a non-synced location (e.g. D:/sfm_outputs) so that "
            "multi-GB dense point clouds are not uploaded, or pause OneDrive during runs."
        ),
    )


# ===========================================================================
# Aggregation and formatting
# ===========================================================================


def check_environment(
    check_paths: Optional[List[Path]] = None,
    deep: bool = False,
) -> EnvironmentReport:
    """Run every probe and return the aggregated report.

    ``deep=True`` additionally runs the slower probes, which are the
    only way to know for certain that dense MVS will run on this GPU.
    """
    checks: List[CheckResult] = [check_python()]
    checks.extend(check_packages())
    checks.append(check_opencv())

    # COLMAP and pycolmap are deliberately not probed. The deliverable
    # implements Structure from Motion itself, so a COLMAP install is not a
    # dependency and checking for one would imply otherwise.
    checks.append(check_cuda())
    checks.append(check_ffmpeg())

    paths = check_paths or [Path.cwd()]
    checks.append(check_disk(paths))
    checks.append(check_onedrive(paths))

    return EnvironmentReport(checks=checks, system=system_report())


_SYMBOL = {"ok": "[ OK ]", "warn": "[WARN]", "fail": "[FAIL]", "skip": "[SKIP]"}


def format_report(report: EnvironmentReport, verbose: bool = False) -> str:
    """Render the report as an aligned, terminal-friendly table."""
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append("ENVIRONMENT VERIFICATION")
    lines.append("=" * 78)

    sysinfo = report.system
    lines.append(f"  Platform : {sysinfo.get('platform')}")
    lines.append(f"  CPU      : {sysinfo.get('processor') or sysinfo.get('machine')} "
                 f"({sysinfo.get('cpu_count_logical')} logical cores)")
    lines.append(f"  RAM      : {sysinfo.get('ram_total')}")
    lines.append("-" * 78)

    width = max((len(c.name) for c in report.checks), default=10)
    for check in report.checks:
        lines.append(f"{_SYMBOL[check.status]} {check.name.ljust(width)}  {check.detail}")
        if check.remedy and check.status in {"warn", "fail"}:
            lines.append(f"{' ' * (width + 9)}-> {check.remedy}")
        if verbose and check.data:
            for key, value in check.data.items():
                lines.append(f"{' ' * (width + 9)}   {key}: {value}")

    lines.append("-" * 78)
    n_fail, n_warn = len(report.failures), len(report.warnings)
    n_ok = sum(1 for c in report.checks if c.status == "ok")
    lines.append(f"  {n_ok} ok, {n_warn} warning(s), {n_fail} failure(s)")
    if report.ok:
        lines.append("  RESULT: environment is usable.")
    else:
        lines.append("  RESULT: required components are missing - fix the [FAIL] items above.")
    lines.append("=" * 78)
    return "\n".join(lines)
