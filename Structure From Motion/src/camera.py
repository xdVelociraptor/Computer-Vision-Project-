"""Camera intrinsics: estimation from EXIF, K matrices, COLMAP model mapping.

Why this module exists separately
---------------------------------
The Fundamental matrix needs no calibration, but the **Essential** matrix,
``recoverPose`` and metric triangulation all require the intrinsic matrix K.
Monocular SfM from an uncalibrated phone therefore rests on an explicit chain of
assumptions, and those assumptions must be stated, not buried:

1. Square pixels and zero skew  (true for essentially every phone sensor)
2. Principal point at the image centre  (approximately true; COLMAP can refine)
3. Focal length from EXIF where available, else a heuristic multiple of the
   image's long side

Assumption 3 is the weak one. This module records *which* branch produced the
focal length (``provenance``) so that every downstream number can be traced back
to whether it rested on real EXIF data or on a guess.

Scale note
----------
Nothing here recovers metric scale. A monocular reconstruction is determined
only up to an arbitrary similarity transform; absolute scale requires external
information (a measured distance, GPS, or a known object size). That is handled
separately in the measurement experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "CameraIntrinsics",
    "estimate_intrinsics",
    "scale_intrinsics",
    "colmap_params_from_intrinsics",
]

#: Standard 35 mm film frame width in millimetres, used to convert an EXIF
#: "35 mm equivalent" focal length into pixels.
FILM_35MM_WIDTH_MM = 36.0


@dataclass
class CameraIntrinsics:
    """Pinhole intrinsics for one image, plus how they were obtained."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    #: Where the focal length came from: exif_35mm | exif_focal_plane | heuristic | explicit
    provenance: str = "heuristic"
    model: str = "SIMPLE_RADIAL"
    #: Distortion coefficients; empty means "assumed undistorted at this stage".
    distortion: List[float] = field(default_factory=list)

    @property
    def K(self) -> np.ndarray:
        """3x3 intrinsic matrix (float64), zero skew."""
        return np.array(
            [[self.fx, 0.0, self.cx],
             [0.0, self.fy, self.cy],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @property
    def is_calibrated(self) -> bool:
        """True when the focal length came from real metadata, not a guess."""
        return self.provenance.startswith("exif") or self.provenance == "explicit"

    def horizontal_fov_deg(self) -> float:
        """Horizontal field of view in degrees - a sanity check on the focal length."""
        return float(np.degrees(2.0 * np.arctan(self.width / (2.0 * self.fx))))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "fx": round(self.fx, 4),
            "fy": round(self.fy, 4),
            "cx": round(self.cx, 4),
            "cy": round(self.cy, 4),
            "provenance": self.provenance,
            "is_calibrated": self.is_calibrated,
            "model": self.model,
            "distortion": self.distortion,
            "horizontal_fov_deg": round(self.horizontal_fov_deg(), 3),
        }


def estimate_intrinsics(
    width: int,
    height: int,
    *,
    focal_length_35mm: Optional[float] = None,
    focal_length_mm: Optional[float] = None,
    sensor_width_mm: Optional[float] = None,
    default_focal_factor: float = 1.2,
    model: str = "SIMPLE_RADIAL",
) -> CameraIntrinsics:
    """Estimate K for an image, preferring real EXIF data over the heuristic.

    Resolution order
    ----------------
    1. **EXIF 35 mm equivalent focal length** - the most reliable single EXIF
       field, because it already folds in the sensor's crop factor::

           f_px = (focal_35mm / 36 mm) * max(width, height)

    2. **EXIF focal length + known sensor width**::

           f_px = (focal_mm / sensor_width_mm) * max(width, height)

    3. **Heuristic fallback** (COLMAP uses the same default of 1.2)::

           f_px = default_focal_factor * max(width, height)

    The long side is used rather than the width because the formulas are defined
    for the sensor's long axis, which stays correct for portrait images.

    Parameters
    ----------
    width, height:
        Pixel dimensions of the image the intrinsics apply to.
    default_focal_factor:
        Multiplier for branch 3. 1.2 corresponds to roughly a 45 degree
        horizontal field of view.

    Raises
    ------
    ValueError
        If the image dimensions are not positive.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions: {width}x{height}")

    long_side = float(max(width, height))

    if focal_length_35mm and focal_length_35mm > 0:
        focal_px = (float(focal_length_35mm) / FILM_35MM_WIDTH_MM) * long_side
        provenance = "exif_35mm"
    elif focal_length_mm and focal_length_mm > 0 and sensor_width_mm and sensor_width_mm > 0:
        focal_px = (float(focal_length_mm) / float(sensor_width_mm)) * long_side
        provenance = "exif_focal_plane"
    else:
        focal_px = float(default_focal_factor) * long_side
        provenance = "heuristic"

    return CameraIntrinsics(
        width=int(width),
        height=int(height),
        # Square pixels: fx == fy. Phone sensors have no meaningful pixel
        # aspect distortion, and splitting them would add a parameter that the
        # data cannot constrain.
        fx=focal_px,
        fy=focal_px,
        cx=width / 2.0,
        cy=height / 2.0,
        provenance=provenance,
        model=model,
    )


def scale_intrinsics(intrinsics: CameraIntrinsics, scale: float) -> CameraIntrinsics:
    """Rescale K for a resized image.

    Required whenever analysis runs on a downscaled copy: the Essential matrix
    is only correct if K matches the pixel coordinates actually fed to it.
    Every intrinsic parameter scales linearly with image size.
    """
    if scale <= 0:
        raise ValueError(f"Scale must be positive, got {scale}")
    return CameraIntrinsics(
        width=int(round(intrinsics.width * scale)),
        height=int(round(intrinsics.height * scale)),
        fx=intrinsics.fx * scale,
        fy=intrinsics.fy * scale,
        cx=intrinsics.cx * scale,
        cy=intrinsics.cy * scale,
        provenance=intrinsics.provenance,
        model=intrinsics.model,
        distortion=list(intrinsics.distortion),
    )


def colmap_params_from_intrinsics(intrinsics: CameraIntrinsics, model: Optional[str] = None) -> Tuple[str, List[float]]:
    """Pack intrinsics into COLMAP's ``--ImageReader.camera_params`` ordering.

    COLMAP parameter orders (see ``src/colmap/sensor/models.h``):

    ==================  ================================
    Model               Parameters
    ==================  ================================
    SIMPLE_PINHOLE      f, cx, cy
    PINHOLE             fx, fy, cx, cy
    SIMPLE_RADIAL       f, cx, cy, k
    RADIAL              f, cx, cy, k1, k2
    OPENCV              fx, fy, cx, cy, k1, k2, p1, p2
    ==================  ================================

    Distortion coefficients are initialised to zero: they are what bundle
    adjustment is meant to solve for, and seeding them with guesses biases the
    optimisation.

    Raises
    ------
    ValueError
        For a camera model this function does not know how to pack.
    """
    name = (model or intrinsics.model).upper()
    f = (intrinsics.fx + intrinsics.fy) / 2.0
    cx, cy = intrinsics.cx, intrinsics.cy

    if name == "SIMPLE_PINHOLE":
        return name, [f, cx, cy]
    if name == "PINHOLE":
        return name, [intrinsics.fx, intrinsics.fy, cx, cy]
    if name == "SIMPLE_RADIAL":
        return name, [f, cx, cy, 0.0]
    if name == "RADIAL":
        return name, [f, cx, cy, 0.0, 0.0]
    if name == "OPENCV":
        return name, [intrinsics.fx, intrinsics.fy, cx, cy, 0.0, 0.0, 0.0, 0.0]
    if name == "FULL_OPENCV":
        return name, [intrinsics.fx, intrinsics.fy, cx, cy] + [0.0] * 8
    raise ValueError(
        f"Do not know the COLMAP parameter ordering for camera model '{name}'. "
        f"Supported: SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV, FULL_OPENCV."
    )
