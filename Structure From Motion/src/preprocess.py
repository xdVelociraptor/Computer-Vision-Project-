"""Stage 1 - input pipeline: video frame extraction and image quality analysis.

Responsibilities
----------------
1. **Video -> frames** with FPS/every-N/max-frames sampling, optional in-loop
   blur rejection and consecutive-similarity rejection, preserving frame order.
2. **Per-image diagnostics**: resolution, sharpness (two independent metrics),
   brightness, contrast, exposure clipping, colourfulness, texture/edge density,
   entropy, perceptual hash, and EXIF (including focal length, which COLMAP can
   use as an intrinsic prior).
3. **Near-duplicate detection** via perceptual dHash + Hamming distance.
4. **Accept / reject decisions** written to a CSV, with rejected images *copied*
   (never moved, never deleted) into a review folder.
5. **Materialisation** of the accepted set into ``processed_images/``.

Design decisions worth stating in the report
--------------------------------------------
* Raw input is treated as read-only. Nothing in this module writes to
  ``raw_images/``; the processed set is always a separate copy.
* Sharpness is measured **after** rescaling the long side to a fixed length
  (``quality.blur_eval_long_side``). The variance of the Laplacian scales with
  image resolution, so comparing native-resolution values across images of
  different sizes is meaningless.
* The default policy is ``report_only``: every readable image is accepted.
  Whether discarding blurry/duplicate images helps or hurts reconstruction is
  measured experimentally later, not assumed here.
"""

from __future__ import annotations

import math
import shutil
from dataclasses import asdict, dataclass, field
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm

from .config import ProjectConfig
from .utils import (
    IMAGE_EXTENSIONS,
    ensure_dir,
    get_logger,
    human_size,
    iter_images,
    params_digest,
    prepare_output_dir,
    read_json,
    write_json,
)

__all__ = [
    "ImageMetrics",
    "PreprocessResult",
    "analyze_image",
    "analyze_directory",
    "extract_frames",
    "run_preprocess",
]

LOGGER = get_logger()

#: Columns the project brief explicitly requires, emitted with these exact names.
REQUIRED_CSV_COLUMNS = [
    "image_name",
    "width",
    "height",
    "blur_score",
    "brightness",
    "contrast",
    "accepted",
    "rejection_reason",
]


# ===========================================================================
# Low-level image IO (Windows-safe)
# ===========================================================================


def imread_unicode(path: Path, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """Read an image via ``np.fromfile`` + ``cv2.imdecode``.

    ``cv2.imread`` goes through a narrow-char API on Windows and silently
    returns ``None`` for paths containing non-ASCII characters. Decoding from a
    byte buffer avoids that entire class of "mysteriously missing image" bugs.
    """
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buffer.size == 0:
        return None
    image = cv2.imdecode(buffer, flags)
    return image


def imwrite_unicode(path: Path, image: np.ndarray, jpeg_quality: int = 95) -> bool:
    """Encode in memory and write with ``ndarray.tofile`` (unicode-safe)."""
    ext = path.suffix.lower() or ".jpg"
    params: List[int] = []
    if ext in {".jpg", ".jpeg"}:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    elif ext == ".png":
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
    ok, encoded = cv2.imencode(ext, image, params)
    if not ok:
        return False
    ensure_dir(path.parent)
    encoded.tofile(str(path))
    return True


def resize_long_side(image: np.ndarray, target: int, interpolation: Optional[int] = None) -> np.ndarray:
    """Scale so the long side equals ``target``, preserving aspect ratio.

    Returns the input unchanged when ``target <= 0`` or no scaling is required.
    """
    if target <= 0:
        return image
    h, w = image.shape[:2]
    long_side = max(h, w)
    if long_side == target:
        return image
    scale = target / float(long_side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    if interpolation is None:
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(image, (new_w, new_h), interpolation=interpolation)


# ===========================================================================
# Quality metrics
# ===========================================================================


def laplacian_variance(gray: np.ndarray) -> float:
    """Variance of the Laplacian - the standard no-reference blur indicator.

    A sharp image has strong second-order intensity derivatives, so the
    Laplacian response has high variance; blur suppresses high frequencies and
    the variance collapses. The value is resolution- and content-dependent,
    which is why callers must pass a size-normalised image.
    """
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def tenengrad(gray: np.ndarray) -> float:
    """Mean squared Sobel gradient magnitude (second, independent sharpness cue).

    Reported alongside the Laplacian variance so that a single metric's failure
    mode (e.g. sensitivity to noise) is visible rather than hidden.
    """
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(gx * gx + gy * gy))


def edge_density(gray: np.ndarray) -> float:
    """Fraction of pixels on a Canny edge - a cheap proxy for scene texture.

    Low-texture surfaces (blank walls, sky, water) produce few features and are
    the dominant cause of failed reconstructions, so this is a useful covariate
    in the cross-scene comparison.
    """
    median = float(np.median(gray))
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, 1.33 * median))
    edges = cv2.Canny(gray, lower, upper)
    return float(np.count_nonzero(edges)) / float(edges.size)


def shannon_entropy(gray: np.ndarray) -> float:
    """Entropy of the 256-bin intensity histogram, in bits."""
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    total = hist.sum()
    if total <= 0:
        return 0.0
    p = hist / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def colorfulness(bgr: np.ndarray) -> float:
    """Hasler & Suesstrunk (2003) colourfulness metric.

    Included because heavily desaturated captures (fog, overcast, night) often
    correlate with weak feature matching.
    """
    b, g, r = cv2.split(bgr.astype(np.float32))
    rg = r - g
    yb = 0.5 * (r + g) - b
    std = math.sqrt(float(np.std(rg)) ** 2 + float(np.std(yb)) ** 2)
    mean = math.sqrt(float(np.mean(rg)) ** 2 + float(np.mean(yb)) ** 2)
    return float(std + 0.3 * mean)


def dhash(gray: np.ndarray, hash_size: int = 8) -> int:
    """Difference hash: compare horizontally adjacent pixels of a tiny thumbnail.

    Produces ``hash_size**2`` bits packed into a Python int. Robust to scale,
    JPEG artefacts and mild brightness shifts, which is exactly what we need for
    "is this frame effectively the same shot as the previous one?".
    """
    small = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    bits = 0
    for bit in diff.flatten():
        bits = (bits << 1) | int(bit)
    return bits


_POPCOUNT_TABLE = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _popcount(values: np.ndarray) -> np.ndarray:
    """Population count for a uint64 array (uses NumPy 2's intrinsic if present)."""
    bitwise_count = getattr(np, "bitwise_count", None)
    if bitwise_count is not None:
        return bitwise_count(values).astype(np.int32)
    view = values.view(np.uint8).reshape(*values.shape, 8)
    return _POPCOUNT_TABLE[view].sum(axis=-1).astype(np.int32)


def hamming_distance(a: int, b: int) -> int:
    """Bit difference between two integer hashes."""
    return int(bin(a ^ b).count("1"))


# ===========================================================================
# EXIF
# ===========================================================================

_EXIF_IFD_POINTER = 0x8769
_EXIF_TAGS = {
    "make": 0x010F,
    "model": 0x0110,
    "orientation": 0x0112,
    "software": 0x0131,
}
_EXIF_SUB_TAGS = {
    "exposure_time_s": 0x829A,
    "f_number": 0x829D,
    "iso": 0x8827,
    "datetime_original": 0x9003,
    "focal_length_mm": 0x920A,
    "focal_length_35mm": 0xA405,
    "exif_pixel_width": 0xA002,
    "exif_pixel_height": 0xA003,
    "lens_model": 0xA434,
}


def _rational_to_float(value: Any) -> Any:
    """Convert PIL ``IFDRational`` / tuples to plain floats."""
    if value is None:
        return None
    if isinstance(value, tuple) and len(value) == 2:
        num, den = value
        return float(num) / float(den) if den else None
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def read_exif(path: Path) -> Dict[str, Any]:
    """Extract the EXIF fields relevant to SfM.

    The focal length matters: COLMAP derives an initial focal-length prior from
    EXIF (35 mm equivalent or sensor-database lookup). Images without EXIF fall
    back to ``camera.default_focal_factor * max(width, height)``, which is a much
    weaker starting point for bundle adjustment.
    """
    info: Dict[str, Any] = {key: None for key in (*_EXIF_TAGS, *_EXIF_SUB_TAGS)}
    info["has_exif"] = False
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            if not exif:
                return info
            info["has_exif"] = True
            for name, tag in _EXIF_TAGS.items():
                info[name] = exif.get(tag)
            try:
                sub = exif.get_ifd(_EXIF_IFD_POINTER)
            except Exception:
                sub = {}
            for name, tag in _EXIF_SUB_TAGS.items():
                info[name] = sub.get(tag)
    except Exception as exc:  # corrupt EXIF must never abort the pipeline
        info["exif_error"] = f"{type(exc).__name__}: {exc}"
        return info

    for key in ("exposure_time_s", "f_number", "focal_length_mm", "focal_length_35mm"):
        info[key] = _rational_to_float(info[key])
    if isinstance(info.get("iso"), (tuple, list)) and info["iso"]:
        info["iso"] = info["iso"][0]
    for key in ("make", "model", "software", "lens_model", "datetime_original"):
        if isinstance(info.get(key), bytes):
            info[key] = info[key].decode("utf-8", errors="replace")
        if isinstance(info.get(key), str):
            info[key] = info[key].strip().strip("\x00")
    if info.get("orientation") is not None:
        try:
            info["orientation"] = int(info["orientation"])
        except (TypeError, ValueError):
            info["orientation"] = None
    return info


# ===========================================================================
# Per-image analysis
# ===========================================================================


@dataclass
class ImageMetrics:
    """All diagnostics for a single input image. One row of ``image_quality.csv``."""

    image_name: str
    relative_path: str
    index: int = -1
    readable: bool = True
    error: str = ""

    width: int = 0
    height: int = 0
    channels: int = 0
    megapixels: float = 0.0
    aspect_ratio: float = 0.0
    file_size_bytes: int = 0

    #: Laplacian variance at normalised scale - the headline sharpness number.
    blur_score: float = 0.0
    #: Laplacian variance at native resolution (not comparable across sizes).
    blur_score_native: float = 0.0
    tenengrad: float = 0.0
    edge_density: float = 0.0
    entropy: float = 0.0

    brightness: float = 0.0
    brightness_median: float = 0.0
    contrast: float = 0.0
    contrast_p95_p05: float = 0.0
    shadow_clipped_frac: float = 0.0
    highlight_clipped_frac: float = 0.0
    colorfulness: float = 0.0

    dhash_hex: str = ""
    hamming_to_prev: int = -1
    duplicate_group: int = -1
    duplicate_of: str = ""

    has_exif: bool = False
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    lens_model: Optional[str] = None
    focal_length_mm: Optional[float] = None
    focal_length_35mm: Optional[float] = None
    f_number: Optional[float] = None
    exposure_time_s: Optional[float] = None
    iso: Optional[int] = None
    orientation: Optional[int] = None
    datetime_original: Optional[str] = None

    accepted: bool = True
    rejection_reason: str = ""


def analyze_image(
    path: Path,
    *,
    blur_eval_long_side: int = 1024,
    hash_size: int = 8,
    root: Optional[Path] = None,
) -> ImageMetrics:
    """Compute every quality metric for one image.

    Never raises: an unreadable or corrupt file is returned with
    ``readable=False`` and an explanatory ``error`` so it appears in the report
    instead of crashing the batch.
    """
    path = Path(path)
    rel = str(path.relative_to(root)) if root and path.is_relative_to(root) else path.name
    metrics = ImageMetrics(image_name=path.name, relative_path=rel)

    try:
        metrics.file_size_bytes = path.stat().st_size
    except OSError as exc:
        metrics.readable = False
        metrics.error = f"stat failed: {exc}"
        metrics.accepted = False
        metrics.rejection_reason = "unreadable"
        return metrics

    if metrics.file_size_bytes == 0:
        metrics.readable = False
        metrics.error = "file is empty (0 bytes)"
        metrics.accepted = False
        metrics.rejection_reason = "unreadable"
        return metrics

    bgr = imread_unicode(path, cv2.IMREAD_COLOR)
    if bgr is None or bgr.size == 0:
        metrics.readable = False
        metrics.error = "cv2 could not decode the file (corrupt or unsupported codec, e.g. HEIC)"
        metrics.accepted = False
        metrics.rejection_reason = "unreadable"
        return metrics

    h, w = bgr.shape[:2]
    metrics.width, metrics.height = int(w), int(h)
    metrics.channels = int(bgr.shape[2]) if bgr.ndim == 3 else 1
    metrics.megapixels = round(w * h / 1e6, 3)
    metrics.aspect_ratio = round(w / h, 4) if h else 0.0

    gray_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Exposure statistics on the native-resolution image: downscaling would
    # average away genuinely clipped pixels.
    metrics.brightness = float(np.mean(gray_full))
    metrics.brightness_median = float(np.median(gray_full))
    metrics.contrast = float(np.std(gray_full))
    p05, p95 = np.percentile(gray_full, [5, 95])
    metrics.contrast_p95_p05 = float(p95 - p05)
    total_px = float(gray_full.size)
    metrics.shadow_clipped_frac = float(np.count_nonzero(gray_full <= 5)) / total_px
    metrics.highlight_clipped_frac = float(np.count_nonzero(gray_full >= 250)) / total_px
    metrics.entropy = shannon_entropy(gray_full)
    metrics.blur_score_native = laplacian_variance(gray_full)

    # Structure metrics at a fixed scale so images of different resolutions
    # are directly comparable.
    gray_norm = resize_long_side(gray_full, blur_eval_long_side)
    metrics.blur_score = laplacian_variance(gray_norm)
    metrics.tenengrad = tenengrad(gray_norm)
    metrics.edge_density = edge_density(gray_norm)
    metrics.dhash_hex = format(dhash(gray_norm, hash_size), "x")

    metrics.colorfulness = colorfulness(resize_long_side(bgr, 512))

    exif = read_exif(path)
    metrics.has_exif = bool(exif.get("has_exif"))
    metrics.camera_make = exif.get("make")
    metrics.camera_model = exif.get("model")
    metrics.lens_model = exif.get("lens_model")
    metrics.focal_length_mm = exif.get("focal_length_mm")
    metrics.focal_length_35mm = exif.get("focal_length_35mm")
    metrics.f_number = exif.get("f_number")
    metrics.exposure_time_s = exif.get("exposure_time_s")
    metrics.iso = exif.get("iso")
    metrics.orientation = exif.get("orientation")
    metrics.datetime_original = exif.get("datetime_original")

    return metrics


# ===========================================================================
# Batch analysis with caching
# ===========================================================================


def _cache_key(path: Path, params: Dict[str, Any]) -> str:
    stat = path.stat()
    return params_digest(
        {"name": path.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, **params}
    )


def analyze_directory(
    directory: Path,
    *,
    blur_eval_long_side: int = 1024,
    hash_size: int = 8,
    workers: int = 8,
    cache_path: Optional[Path] = None,
    use_cache: bool = True,
    show_progress: bool = True,
) -> List[ImageMetrics]:
    """Analyse every image in ``directory`` (sorted, parallel, cached).

    Images are decoded one at a time inside worker threads; the full set is
    never held in memory simultaneously. Results are returned in filename order
    regardless of completion order, so the pipeline stays deterministic.
    """
    paths = iter_images(directory, IMAGE_EXTENSIONS)
    if not paths:
        return []

    params = {"blur_eval_long_side": blur_eval_long_side, "hash_size": hash_size, "v": 1}
    cache: Dict[str, Dict[str, Any]] = {}
    if use_cache and cache_path and Path(cache_path).is_file():
        try:
            cache = read_json(cache_path)
            LOGGER.debug("Loaded %d cached image metrics from %s", len(cache), cache_path)
        except Exception as exc:
            LOGGER.warning("Ignoring unreadable metrics cache %s (%s)", cache_path, exc)
            cache = {}

    results: List[Optional[ImageMetrics]] = [None] * len(paths)
    pending: List[Tuple[int, Path, str]] = []

    for i, path in enumerate(paths):
        try:
            key = _cache_key(path, params)
        except OSError:
            key = ""
        if use_cache and key and key in cache:
            try:
                results[i] = ImageMetrics(**cache[key])
                continue
            except TypeError:
                # Cache written by an older schema version; recompute.
                pass
        pending.append((i, path, key))

    if pending:
        LOGGER.info("Analysing %d image(s) (%d served from cache)", len(pending), len(paths) - len(pending))

        def _work(item: Tuple[int, Path, str]) -> Tuple[int, str, ImageMetrics]:
            idx, p, k = item
            return idx, k, analyze_image(
                p, blur_eval_long_side=blur_eval_long_side, hash_size=hash_size, root=directory
            )

        iterator: Any
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            iterator = pool.map(_work, pending)
            if show_progress:
                iterator = tqdm(iterator, total=len(pending), desc="quality analysis", unit="img")
            for idx, key, metrics in iterator:
                results[idx] = metrics
                if key:
                    cache[key] = asdict(metrics)
    else:
        LOGGER.info("All %d image metrics served from cache", len(paths))

    if use_cache and cache_path:
        write_json(Path(cache_path), cache)

    final: List[ImageMetrics] = []
    for i, metrics in enumerate(results):
        assert metrics is not None  # every slot is filled above
        metrics.index = i
        final.append(metrics)
    return final


# ===========================================================================
# Near-duplicate detection
# ===========================================================================


class _UnionFind:
    """Minimal disjoint-set structure for grouping near-duplicate images."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def detect_near_duplicates(
    metrics: Sequence[ImageMetrics],
    max_hamming_distance: int,
    keep: str = "sharpest",
    block: int = 512,
) -> Dict[str, Any]:
    """Group images whose perceptual hashes are within ``max_hamming_distance``.

    Also fills ``hamming_to_prev`` for every image, which is a useful capture
    diagnostic on its own: a long run of near-zero distances means the camera
    was static and those frames add cost without adding baseline.

    Returns a summary dict; ``metrics`` entries are updated in place.
    """
    readable = [m for m in metrics if m.readable and m.dhash_hex]
    for i in range(1, len(readable)):
        readable[i].hamming_to_prev = hamming_distance(
            int(readable[i - 1].dhash_hex, 16), int(readable[i].dhash_hex, 16)
        )
    if readable:
        readable[0].hamming_to_prev = -1

    summary: Dict[str, Any] = {"num_groups": 0, "num_duplicates": 0, "groups": []}
    if max_hamming_distance <= 0 or len(readable) < 2:
        return summary

    hashes = np.array([int(m.dhash_hex, 16) for m in readable], dtype=np.uint64)
    n = len(hashes)
    uf = _UnionFind(n)

    # Blocked pairwise XOR + popcount: O(n^2) bit operations but fully
    # vectorised, and memory is bounded by `block` rows at a time.
    for start in range(0, n, block):
        stop = min(start + block, n)
        chunk = hashes[start:stop, None] ^ hashes[None, :]
        distances = _popcount(chunk)
        rows, cols = np.nonzero(distances <= max_hamming_distance)
        for r, c in zip(rows, cols):
            i, j = start + int(r), int(c)
            if i < j:
                uf.union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(uf.find(i), []).append(i)

    group_id = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        if keep == "sharpest":
            keeper = max(members, key=lambda i: readable[i].blur_score)
        else:
            keeper = min(members)
        for i in members:
            readable[i].duplicate_group = group_id
            if i != keeper:
                readable[i].duplicate_of = readable[keeper].image_name
        summary["groups"].append(
            {
                "group": group_id,
                "keep": readable[keeper].image_name,
                "members": [readable[i].image_name for i in members],
            }
        )
        summary["num_duplicates"] += len(members) - 1
        group_id += 1

    summary["num_groups"] = group_id
    return summary


# ===========================================================================
# Accept / reject policy
# ===========================================================================


def _blur_threshold(values: np.ndarray, cfg: ProjectConfig) -> float:
    """Resolve the sharpness cut-off for the configured threshold mode."""
    blur_cfg = cfg.quality.blur
    if blur_cfg.threshold_mode == "absolute":
        return float(blur_cfg.min_score)
    if values.size == 0:
        return float("-inf")
    if blur_cfg.threshold_mode == "percentile":
        return float(np.percentile(values, blur_cfg.percentile))
    # Median absolute deviation: robust to a few extremely sharp/blurred outliers.
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return median - blur_cfg.mad_k * 1.4826 * mad


def apply_quality_policy(metrics: List[ImageMetrics], cfg: ProjectConfig) -> Dict[str, Any]:
    """Set ``accepted`` / ``rejection_reason`` on each image.

    Unreadable files are always rejected. Every other rule fires only when
    ``quality.mode == "filter"`` and the rule is listed in ``quality.filters``.
    Multiple reasons are joined with ``;`` so nothing is lost.
    """
    q = cfg.quality
    active = set(q.filters) if q.mode == "filter" else set()
    readable = [m for m in metrics if m.readable]

    blur_values = np.array([m.blur_score for m in readable], dtype=np.float64)
    blur_cut = _blur_threshold(blur_values, cfg) if "blur" in active else float("-inf")

    modal_size: Optional[Tuple[int, int]] = None
    if readable:
        sizes = pd.Series([(m.width, m.height) for m in readable])
        modal_size = sizes.mode().iloc[0] if not sizes.mode().empty else None

    policy = {
        "mode": q.mode,
        "active_filters": sorted(active),
        "blur_threshold": None if blur_cut == float("-inf") else round(blur_cut, 3),
        "blur_threshold_mode": q.blur.threshold_mode,
        "modal_resolution": list(modal_size) if modal_size else None,
    }

    for m in metrics:
        reasons: List[str] = []
        if not m.readable:
            m.accepted = False
            m.rejection_reason = m.rejection_reason or "unreadable"
            continue

        if "blur" in active and m.blur_score < blur_cut:
            reasons.append(f"blur({m.blur_score:.1f}<{blur_cut:.1f})")
        if "brightness" in active and not (q.brightness.min <= m.brightness <= q.brightness.max):
            reasons.append(f"brightness({m.brightness:.1f})")
        if "contrast" in active and m.contrast < q.contrast.min:
            reasons.append(f"contrast({m.contrast:.1f}<{q.contrast.min:.1f})")
        if "resolution" in active:
            if m.width < q.resolution.min_width or m.height < q.resolution.min_height:
                reasons.append(f"resolution({m.width}x{m.height})")
            elif q.resolution.require_consistent and modal_size and (m.width, m.height) != tuple(modal_size):
                reasons.append(f"inconsistent_resolution({m.width}x{m.height})")
        if "clipping" in active:
            if m.shadow_clipped_frac > q.clipping.shadow_frac_max:
                reasons.append(f"shadow_clipping({m.shadow_clipped_frac:.2f})")
            if m.highlight_clipped_frac > q.clipping.highlight_frac_max:
                reasons.append(f"highlight_clipping({m.highlight_clipped_frac:.2f})")
        if "duplicate" in active and m.duplicate_of:
            reasons.append(f"near_duplicate_of({m.duplicate_of})")

        m.accepted = not reasons
        m.rejection_reason = ";".join(reasons)

    return policy


# ===========================================================================
# Materialisation
# ===========================================================================


def _needs_reencode(m: ImageMetrics, cfg: ProjectConfig) -> bool:
    if cfg.processing.copy_mode == "reencode":
        return True
    if cfg.processing.copy_mode == "copy":
        return False
    if cfg.processing.resize_max_dim and max(m.width, m.height) > cfg.processing.resize_max_dim:
        return True
    if cfg.processing.apply_exif_transpose and (m.orientation or 1) != 1:
        return True
    return False


def materialise_images(
    metrics: List[ImageMetrics],
    source_dir: Path,
    cfg: ProjectConfig,
    *,
    force: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Copy accepted images into ``processed_images/`` and rejects into ``review/``.

    Byte-for-byte copying is preferred whenever no resize or rotation is needed,
    because it preserves the EXIF block that COLMAP reads for its focal-length
    prior. Re-encoding happens only when the pixels actually change, and the
    EXIF block is carried across.
    """
    processed_dir = cfg.processed_images_dir
    review_dir = cfg.review_dir

    stats = {
        "processed_dir": str(processed_dir),
        "review_dir": str(review_dir),
        "copied": 0,
        "reencoded": 0,
        "rejected_copied": 0,
        "failed": [],
        "output_bytes": 0,
    }
    if dry_run:
        stats["dry_run"] = True
        return stats

    prepare_output_dir(processed_dir, force=force, logger=LOGGER)
    prepare_output_dir(review_dir, force=force, logger=LOGGER)

    accepted = [m for m in metrics if m.accepted]
    rejected = [m for m in metrics if not m.accepted]

    for m in tqdm(accepted, desc="writing processed set", unit="img"):
        src = source_dir / m.relative_path
        dst = processed_dir / m.image_name
        try:
            if _needs_reencode(m, cfg):
                with Image.open(src) as img:
                    if cfg.processing.apply_exif_transpose:
                        img = ImageOps.exif_transpose(img)
                    if cfg.processing.resize_max_dim:
                        long_side = max(img.size)
                        if long_side > cfg.processing.resize_max_dim:
                            scale = cfg.processing.resize_max_dim / float(long_side)
                            new_size = (
                                max(1, int(round(img.size[0] * scale))),
                                max(1, int(round(img.size[1] * scale))),
                            )
                            img = img.resize(new_size, Image.Resampling.LANCZOS)
                    if img.mode not in ("RGB", "L"):
                        img = img.convert("RGB")
                    save_kwargs: Dict[str, Any] = {}
                    if dst.suffix.lower() in {".jpg", ".jpeg"}:
                        save_kwargs.update(quality=cfg.processing.jpeg_quality, subsampling=0)
                        exif_bytes = img.info.get("exif")
                        if exif_bytes:
                            save_kwargs["exif"] = exif_bytes
                    img.save(dst, **save_kwargs)
                stats["reencoded"] += 1
            else:
                shutil.copy2(src, dst)
                stats["copied"] += 1
            stats["output_bytes"] += dst.stat().st_size
        except Exception as exc:
            stats["failed"].append({"image": m.image_name, "error": f"{type(exc).__name__}: {exc}"})
            LOGGER.error("Failed to materialise %s: %s", m.image_name, exc)

    for m in rejected:
        src = source_dir / m.relative_path
        if not src.is_file():
            continue
        try:
            shutil.copy2(src, review_dir / m.image_name)
            stats["rejected_copied"] += 1
        except Exception as exc:
            stats["failed"].append({"image": m.image_name, "error": f"{type(exc).__name__}: {exc}"})

    stats["output_human"] = human_size(stats["output_bytes"])
    return stats


# ===========================================================================
# Video frame extraction
# ===========================================================================


def extract_frames(
    video_path: Path,
    output_dir: Path,
    cfg: ProjectConfig,
    *,
    force: bool = False,
) -> Dict[str, Any]:
    """Decode a video into ordered still frames according to ``cfg.video``.

    Sampling modes
    --------------
    ``fps``        keep roughly ``target_fps`` frames per second of footage
    ``every_n``    keep every N-th decoded frame
    ``max_frames`` spread ``max_frames`` samples evenly across the interval

    Frames are named ``frame_<source_index:06d>.jpg`` so the filename ordering
    equals the temporal ordering - which is what makes COLMAP's sequential
    matcher meaningful.
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(
            f"OpenCV could not open '{video_path}'. The codec may be unsupported - "
            f"try transcoding first: ffmpeg -i input.mov -c:v libx264 -crf 18 output.mp4"
        )

    src_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s = (total_frames / src_fps) if src_fps > 0 else 0.0

    LOGGER.info(
        "Video: %s | %dx%d | %.3f fps | %d frames | %.1f s",
        video_path.name, src_width, src_height, src_fps, total_frames, duration_s,
    )
    if src_fps <= 0:
        LOGGER.warning("Container reports no frame rate; falling back to every_n sampling")

    start_frame = int(round(cfg.video.start_time_s * src_fps)) if src_fps > 0 else 0
    end_frame = (
        int(round(cfg.video.end_time_s * src_fps))
        if (cfg.video.end_time_s > 0 and src_fps > 0)
        else (total_frames if total_frames > 0 else 10**9)
    )
    start_frame = max(0, start_frame)
    end_frame = max(start_frame + 1, end_frame)
    span = end_frame - start_frame

    mode = cfg.video.mode
    if mode == "fps" and src_fps > 0:
        step = max(1, int(round(src_fps / cfg.video.target_fps)))
    elif mode == "max_frames" and cfg.video.max_frames > 0 and total_frames > 0:
        step = max(1, int(math.ceil(span / cfg.video.max_frames)))
    else:
        if mode == "fps":
            LOGGER.warning("mode='fps' requires a known source fps; using every_n=%d", cfg.video.every_n)
        step = max(1, cfg.video.every_n)

    LOGGER.info(
        "Sampling mode=%s step=%d frames -> ~%.2f fps, frame range [%d, %d)",
        mode, step, (src_fps / step) if src_fps > 0 else float("nan"), start_frame, end_frame,
    )

    output_dir = prepare_output_dir(Path(output_dir), force=force, logger=LOGGER)

    records: List[Dict[str, Any]] = []
    kept = 0
    skipped_blur = 0
    skipped_similar = 0
    last_kept_hash: Optional[int] = None
    frame_index = -1
    ext = cfg.video.output_format if cfg.video.output_format.startswith(".") else f".{cfg.video.output_format}"

    progress = tqdm(total=(span if total_frames > 0 else None), desc="extracting frames", unit="frame")
    try:
        while True:
            grabbed = cap.grab()  # cheap: demux without full decode
            if not grabbed:
                break
            frame_index += 1
            if frame_index < start_frame:
                continue
            if frame_index >= end_frame:
                break
            progress.update(1)
            if (frame_index - start_frame) % step != 0:
                continue
            if cfg.video.max_frames > 0 and kept >= cfg.video.max_frames and mode != "max_frames":
                break

            ok, frame = cap.retrieve()  # decode only the frames we actually want
            if not ok or frame is None:
                records.append({"source_frame": frame_index, "kept": False, "reason": "decode_failed"})
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_norm = resize_long_side(gray, cfg.quality.blur_eval_long_side)
            blur = laplacian_variance(gray_norm)
            frame_hash = dhash(gray_norm, cfg.quality.duplicate.hash_size)
            timestamp_s = frame_index / src_fps if src_fps > 0 else float("nan")

            record: Dict[str, Any] = {
                "source_frame": frame_index,
                "timestamp_s": round(timestamp_s, 4) if src_fps > 0 else None,
                "blur_score": round(blur, 3),
                "kept": False,
                "reason": "",
                "filename": "",
            }

            if cfg.video.min_blur_score > 0 and blur < cfg.video.min_blur_score:
                record["reason"] = f"blur({blur:.1f}<{cfg.video.min_blur_score:.1f})"
                skipped_blur += 1
                records.append(record)
                continue

            if cfg.video.min_hamming_distance > 0 and last_kept_hash is not None:
                dist = hamming_distance(last_kept_hash, frame_hash)
                record["hamming_to_last_kept"] = dist
                if dist < cfg.video.min_hamming_distance:
                    record["reason"] = f"too_similar(d={dist}<{cfg.video.min_hamming_distance})"
                    skipped_similar += 1
                    records.append(record)
                    continue

            out_frame = resize_long_side(frame, cfg.video.resize_max_dim)
            filename = f"frame_{frame_index:06d}{ext}"
            if not imwrite_unicode(output_dir / filename, out_frame, cfg.video.jpeg_quality):
                record["reason"] = "write_failed"
                records.append(record)
                continue

            record["kept"] = True
            record["filename"] = filename
            records.append(record)
            last_kept_hash = frame_hash
            kept += 1

            if cfg.video.max_frames > 0 and kept >= cfg.video.max_frames:
                LOGGER.info("Reached video.max_frames=%d; stopping extraction", cfg.video.max_frames)
                break
    finally:
        progress.close()
        cap.release()

    report = {
        "video": str(video_path),
        "video_bytes": video_path.stat().st_size,
        "source_fps": round(src_fps, 4),
        "source_frames": total_frames,
        "source_resolution": [src_width, src_height],
        "duration_s": round(duration_s, 3),
        "sampling_mode": mode,
        "step": step,
        "start_frame": start_frame,
        "end_frame": min(end_frame, frame_index + 1),
        "frames_examined": len(records),
        "frames_written": kept,
        "skipped_blur": skipped_blur,
        "skipped_similar": skipped_similar,
        "effective_fps": round(src_fps / step, 4) if src_fps > 0 else None,
        "output_dir": str(output_dir),
        "frames": records,
    }
    LOGGER.info(
        "Extracted %d frame(s) from %d candidate(s) (blur-skipped %d, similarity-skipped %d)",
        kept, len(records), skipped_blur, skipped_similar,
    )
    if kept == 0:
        raise RuntimeError(
            "No frames were written. Loosen video.min_blur_score / video.min_hamming_distance, "
            "or check the start/end time window."
        )
    return report


# ===========================================================================
# Stage driver
# ===========================================================================


@dataclass
class PreprocessResult:
    """Everything stage 1 produced, returned to the CLI and written to JSON."""

    scene: str
    source_type: str
    total_images: int = 0
    accepted_images: int = 0
    rejected_images: int = 0
    unreadable_images: int = 0
    csv_path: str = ""
    summary_path: str = ""
    plot_path: str = ""
    extraction_report_path: str = ""
    policy: Dict[str, Any] = field(default_factory=dict)
    duplicates: Dict[str, Any] = field(default_factory=dict)
    materialisation: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


def _summarise(df: pd.DataFrame) -> Dict[str, Any]:
    """Descriptive statistics for the numeric quality columns."""
    numeric = [
        "width", "height", "megapixels", "blur_score", "blur_score_native", "tenengrad",
        "edge_density", "entropy", "brightness", "contrast", "contrast_p95_p05",
        "shadow_clipped_frac", "highlight_clipped_frac", "colorfulness",
    ]
    stats: Dict[str, Any] = {}
    for col in numeric:
        if col not in df.columns or df[col].dropna().empty:
            continue
        series = df[col].astype(float)
        stats[col] = {
            "min": float(series.min()),
            "p05": float(series.quantile(0.05)),
            "median": float(series.median()),
            "mean": float(series.mean()),
            "p95": float(series.quantile(0.95)),
            "max": float(series.max()),
            "std": float(series.std(ddof=0)) if len(series) > 1 else 0.0,
        }
    return stats


def _camera_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Which camera(s) produced this dataset, and is a focal prior available?"""
    out: Dict[str, Any] = {}
    for col in ("camera_make", "camera_model", "lens_model"):
        if col in df.columns:
            values = df[col].dropna().astype(str)
            out[col] = sorted(values.unique().tolist())
    for col in ("focal_length_mm", "focal_length_35mm", "f_number", "iso"):
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            out[col] = {
                "count": int(series.size),
                "unique": sorted({float(v) for v in series.unique()})[:20],
            } if not series.empty else {"count": 0, "unique": []}
    if "has_exif" in df.columns:
        out["images_with_exif"] = int(df["has_exif"].sum())
    if "orientation" in df.columns:
        counts = df["orientation"].dropna().astype(int).value_counts().to_dict()
        out["orientation_counts"] = {int(k): int(v) for k, v in counts.items()}
    resolutions = (
        df.groupby(["width", "height"]).size().sort_values(ascending=False)
        if {"width", "height"} <= set(df.columns) else pd.Series(dtype=int)
    )
    out["resolutions"] = [
        {"width": int(w), "height": int(h), "count": int(c)} for (w, h), c in resolutions.items()
    ]
    return out


def _warnings_from_summary(df: pd.DataFrame, result: PreprocessResult) -> List[str]:
    """Actionable capture-quality warnings surfaced to the user."""
    warnings: List[str] = []
    if result.total_images < 20:
        warnings.append(
            f"Only {result.total_images} image(s): incremental SfM typically needs 30+ "
            f"well-overlapping views for a stable reconstruction."
        )
    if result.unreadable_images:
        warnings.append(f"{result.unreadable_images} file(s) could not be decoded - see rejection_reason.")
    if "has_exif" in df.columns and int(df["has_exif"].sum()) == 0:
        warnings.append(
            "No EXIF found in any image. COLMAP will fall back to a heuristic focal-length "
            "prior (camera.default_focal_factor), which weakens initial bundle adjustment."
        )
    if {"width", "height"} <= set(df.columns) and df.groupby(["width", "height"]).ngroups > 1:
        warnings.append(
            "Mixed image resolutions detected. Keep camera.single_camera=true only if all "
            "images really come from the same physical camera and lens setting."
        )
    if "hamming_to_prev" in df.columns:
        static = df[(df["hamming_to_prev"] >= 0) & (df["hamming_to_prev"] <= 2)]
        if len(static) > 0.3 * max(1, len(df)):
            warnings.append(
                f"{len(static)} image(s) are nearly identical to their predecessor: the camera "
                f"may have been static. Insufficient baseline causes degenerate triangulation."
            )
    if "blur_score" in df.columns and not df["blur_score"].empty:
        low = int((df["blur_score"] < df["blur_score"].median() * 0.25).sum())
        if low:
            warnings.append(f"{low} image(s) are far blurrier than the dataset median - inspect before use.")
    return warnings


def run_preprocess(
    cfg: ProjectConfig,
    *,
    force: bool = False,
    dry_run: bool = False,
    skip_extraction: bool = False,
    make_plots: bool = True,
) -> PreprocessResult:
    """Run the full stage-1 pipeline for one scene and write every artefact."""
    ensure_dir(cfg.metrics_dir)
    ensure_dir(cfg.metadata_dir)
    ensure_dir(cfg.visualizations_dir)

    result = PreprocessResult(scene=cfg.scene.name, source_type=cfg.scene.source_type)

    # -- 1. obtain raw images ------------------------------------------------
    if cfg.scene.source_type == "video" and not skip_extraction:
        video_path = cfg.video_path
        if video_path is None:
            raise ValueError("scene.source_type is 'video' but scene.video_file is not configured")
        LOGGER.info("Extracting frames from %s", video_path)
        report = extract_frames(video_path, cfg.raw_images_dir, cfg, force=force)
        path = cfg.metadata_dir / "extraction_report.json"
        write_json(path, report)
        result.extraction_report_path = str(path)
        frames_df = pd.DataFrame(report["frames"])
        frames_df.to_csv(cfg.metadata_dir / "extraction_report.csv", index=False)
        LOGGER.info("Extraction report -> %s", path)

    raw_dir = cfg.raw_images_dir
    if not raw_dir.is_dir():
        raise FileNotFoundError(
            f"Raw image directory does not exist: {raw_dir}\n"
            f"Put your photographs there, or set scene.source_type=video and scene.video_file."
        )
    paths = iter_images(raw_dir)
    if not paths:
        raise FileNotFoundError(
            f"No images found in {raw_dir} (looked for {sorted(IMAGE_EXTENSIONS)}).\n"
            f"HEIC files from iPhone must be converted to JPEG first."
        )
    LOGGER.info("Found %d image file(s) in %s", len(paths), raw_dir)

    # -- 2. per-image metrics -----------------------------------------------
    metrics = analyze_directory(
        raw_dir,
        blur_eval_long_side=cfg.quality.blur_eval_long_side,
        hash_size=cfg.quality.duplicate.hash_size,
        workers=cfg.runtime.workers,
        cache_path=cfg.cache_dir / "image_metrics.json",
        use_cache=cfg.processing.cache,
    )

    # -- 3. duplicates and policy -------------------------------------------
    result.duplicates = detect_near_duplicates(
        metrics,
        max_hamming_distance=cfg.quality.duplicate.max_hamming_distance,
        keep=cfg.quality.duplicate.keep,
    )
    result.policy = apply_quality_policy(metrics, cfg)

    df = pd.DataFrame([asdict(m) for m in metrics])
    ordered = REQUIRED_CSV_COLUMNS + [c for c in df.columns if c not in REQUIRED_CSV_COLUMNS]
    df = df[ordered]

    result.total_images = len(df)
    result.accepted_images = int(df["accepted"].sum())
    result.rejected_images = int((~df["accepted"]).sum())
    result.unreadable_images = int((~df["readable"]).sum())

    csv_path = cfg.metrics_dir / "image_quality.csv"
    df.to_csv(csv_path, index=False)
    result.csv_path = str(csv_path)
    LOGGER.info("Quality table -> %s", csv_path)

    exif_path = cfg.metadata_dir / "exif_summary.json"
    write_json(exif_path, _camera_summary(df))

    # -- 4. materialise ------------------------------------------------------
    if result.accepted_images == 0:
        raise RuntimeError(
            "Every image was rejected. Loosen quality.filters / thresholds "
            "(quality.mode=report_only accepts everything readable)."
        )
    result.materialisation = materialise_images(metrics, raw_dir, cfg, force=force, dry_run=dry_run)

    # -- 5. plots and summary ------------------------------------------------
    if make_plots:
        from .visualization import plot_quality_overview  # local import: keeps CLI startup fast

        plot_path = cfg.visualizations_dir / "quality_overview.png"
        try:
            plot_quality_overview(df, plot_path, scene=cfg.scene.name, policy=result.policy)
            result.plot_path = str(plot_path)
            LOGGER.info("Quality plots -> %s", plot_path)
        except Exception as exc:
            LOGGER.error("Plot generation failed (%s: %s); metrics are still valid", type(exc).__name__, exc)

    result.warnings = _warnings_from_summary(df, result)

    summary = {
        "scene": cfg.scene.name,
        "source_type": cfg.scene.source_type,
        "raw_images_dir": str(raw_dir),
        "total_images": result.total_images,
        "accepted_images": result.accepted_images,
        "rejected_images": result.rejected_images,
        "unreadable_images": result.unreadable_images,
        "acceptance_rate_percent": round(100.0 * result.accepted_images / max(1, result.total_images), 2),
        "policy": result.policy,
        "duplicates": {k: v for k, v in result.duplicates.items() if k != "groups"},
        "duplicate_groups": result.duplicates.get("groups", []),
        "camera": _camera_summary(df),
        "statistics": _summarise(df),
        "materialisation": result.materialisation,
        "rejection_reasons": (
            df.loc[~df["accepted"], "rejection_reason"].value_counts().to_dict()
            if result.rejected_images else {}
        ),
        "warnings": result.warnings,
        "config": cfg.to_dict(),
    }
    summary_path = cfg.metrics_dir / "preprocess_summary.json"
    write_json(summary_path, summary)
    result.summary_path = str(summary_path)
    LOGGER.info("Summary -> %s", summary_path)

    return result
