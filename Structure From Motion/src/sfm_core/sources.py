"""Input sources: photographs and video, combined into one reconstruction.

A *source* is one capture device/session - a folder of photographs, or a video
file. Every source becomes a **camera group** with its own intrinsics, because
a phone's still camera and its video mode do not share a field of view: video
is typically cropped and rescaled, so a frame's focal length in pixels differs
from a photograph's even though the physical lens is identical. Calibrating
both with one matrix biases every pose that mixes them.

The mapper therefore carries a per-image intrinsics matrix, and bundle
adjustment refines **one focal length per group** rather than one globally.
Combining sources is genuinely useful: photographs contribute resolution and
texture detail, while video contributes dense temporal coverage that guarantees
the match graph stays connected.

Frame selection
---------------
Video is massively redundant - at 30 fps consecutive frames are nearly
identical, and near-duplicate views add cost without adding parallax. Frames
are therefore sampled at a target rate, then filtered for blur (video frames
have a fixed exposure and no per-shot stabilisation, so motion blur is common)
and for near-duplication against the previously accepted frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..utils import ensure_dir, get_logger

__all__ = ["SourceSpec", "InputSet", "extract_video_frames", "collect_inputs"]

LOGGER = get_logger()


@dataclass
class SourceSpec:
    """One input source and the frames it contributed."""

    kind: str                       # "images" | "video"
    origin: str
    group: int
    paths: List[Path] = field(default_factory=list)
    width: int = 0
    height: int = 0
    notes: str = ""


@dataclass
class InputSet:
    """All images for a reconstruction, with their camera-group assignment."""

    paths: List[Path] = field(default_factory=list)
    groups: List[int] = field(default_factory=list)
    sources: List[SourceSpec] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.paths)

    def summary(self) -> List[Dict[str, object]]:
        return [{"kind": s.kind, "origin": s.origin, "group": s.group,
                 "frames": len(s.paths), "resolution": [s.width, s.height],
                 "notes": s.notes} for s in self.sources]


def _laplacian_variance(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _dhash(gray: np.ndarray, size: int = 8) -> int:
    """64-bit difference hash, for cheap near-duplicate detection."""
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = small[:, 1:] > small[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return value


def extract_video_frames(
    video_path: Path,
    output_dir: Path,
    target_fps: float = 2.0,
    max_frames: int = 0,
    min_blur_ratio: float = 0.55,
    min_hamming: int = 6,
    jpeg_quality: int = 95,
    prefix: str = "vid",
) -> Tuple[List[Path], Dict[str, object]]:
    """Sample frames from a video, rejecting blurred and near-duplicate ones.

    ``min_blur_ratio`` is relative to the *median* sharpness of the sampled
    frames rather than an absolute threshold, because absolute Laplacian
    variance depends on scene texture and resolution and is not comparable
    across videos.

    Returns ``(frame_paths, report)``.
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, int(round(fps / max(target_fps, 1e-6))))

    LOGGER.info("Video %s: %dx%d, %.1f fps, %d frames -> sampling every %d frame(s)",
                video_path.name, width, height, fps, total, step)
    if width * height < 500_000:
        LOGGER.warning("  video resolution is only %.1f MP; heavily compressed video "
                       "carries little texture for stereo matching", width * height / 1e6)

    ensure_dir(output_dir)
    candidates: List[Tuple[int, np.ndarray, float, int]] = []
    index = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if index % step == 0:
            ok, frame = cap.retrieve()
            if ok and frame is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                candidates.append((index, frame, _laplacian_variance(gray), _dhash(gray)))
        index += 1
    cap.release()

    if not candidates:
        raise ValueError(f"No frames could be decoded from {video_path}")

    sharpness = np.array([c[2] for c in candidates])
    threshold = float(np.median(sharpness)) * min_blur_ratio

    kept: List[Path] = []
    last_hash: Optional[int] = None
    rejected_blur = rejected_dupe = 0
    for frame_index, frame, blur, hash_value in candidates:
        if blur < threshold:
            rejected_blur += 1
            continue
        if last_hash is not None and bin(last_hash ^ hash_value).count("1") < min_hamming:
            rejected_dupe += 1
            continue
        path = Path(output_dir) / f"{prefix}_{frame_index:06d}.jpg"
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        if not ok:
            continue
        buf.tofile(str(path))
        kept.append(path)
        last_hash = hash_value
        if max_frames and len(kept) >= max_frames:
            break

    report = {
        "video": str(video_path),
        "resolution": [width, height],
        "fps": round(float(fps), 3),
        "total_frames": total,
        "sampled": len(candidates),
        "kept": len(kept),
        "rejected_blur": rejected_blur,
        "rejected_near_duplicate": rejected_dupe,
        "blur_threshold": round(threshold, 2),
        "sample_step": step,
    }
    LOGGER.info("  kept %d frames (%d blurred, %d near-duplicate rejected)",
                len(kept), rejected_blur, rejected_dupe)
    return kept, report


def collect_inputs(
    image_dirs: Sequence[Path] = (),
    videos: Sequence[Path] = (),
    frame_dir: Optional[Path] = None,
    target_fps: float = 2.0,
    max_video_frames: int = 0,
    max_images_per_source: int = 0,
) -> InputSet:
    """Gather photographs and video frames into one grouped input set.

    Each image directory and each video becomes its own camera group.
    """
    from ..utils import iter_images

    inputs = InputSet()
    group = 0

    for directory in image_dirs:
        directory = Path(directory)
        if not directory.is_dir():
            LOGGER.warning("Image directory missing, skipping: %s", directory)
            continue
        paths = iter_images(directory)
        if not paths:
            continue
        if max_images_per_source and len(paths) > max_images_per_source:
            sel = np.linspace(0, len(paths) - 1, max_images_per_source).astype(int)
            paths = [paths[i] for i in sel]
        probe = cv2.imdecode(np.fromfile(str(paths[0]), dtype=np.uint8), cv2.IMREAD_COLOR)
        h, w = (probe.shape[:2] if probe is not None else (0, 0))
        spec = SourceSpec("images", str(directory), group, list(paths), w, h)
        inputs.sources.append(spec)
        inputs.paths.extend(paths)
        inputs.groups.extend([group] * len(paths))
        LOGGER.info("Source %d: %d photographs from %s (%dx%d)",
                    group, len(paths), directory.name, w, h)
        group += 1

    for video in videos:
        video = Path(video)
        out = Path(frame_dir or video.parent / "video_frames") / video.stem[:40]
        paths, report = extract_video_frames(
            video, out, target_fps=target_fps, max_frames=max_video_frames,
            prefix=f"g{group}",
        )
        if not paths:
            continue
        spec = SourceSpec("video", str(video), group, paths,
                          report["resolution"][0], report["resolution"][1],
                          notes=(f"{report['kept']}/{report['sampled']} sampled frames kept; "
                                 f"{report['rejected_blur']} blurred, "
                                 f"{report['rejected_near_duplicate']} near-duplicate"))
        inputs.sources.append(spec)
        inputs.paths.extend(paths)
        inputs.groups.extend([group] * len(paths))
        group += 1

    if not inputs.paths:
        raise ValueError("No usable input images or video frames were found")
    LOGGER.info("Collected %d images across %d camera group(s)",
                len(inputs.paths), len(inputs.sources))
    return inputs
