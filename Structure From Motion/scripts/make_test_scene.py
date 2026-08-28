"""Generate a SYNTHETIC scene for self-testing the stage-1 input pipeline.

WHY THIS EXISTS
---------------
It lets you verify that extraction, quality analysis, duplicate detection,
filtering and reporting all work *before* you walk to campus and capture real
data. It deliberately injects known defects so you can confirm each detector
fires:

    * motion-blurred images        -> blur_score collapses
    * under/over-exposed images    -> brightness / clipping filters
    * a low-contrast (hazy) image  -> contrast filter
    * an exact + a near-duplicate  -> dHash grouping
    * a zero-byte and a garbage    -> "unreadable" rejection
      file
    * one odd-resolution image     -> resolution consistency check
    * EXIF on a subset             -> focal-length prior detection

!! NEVER report numbers from this scene as experimental results. It is a
!! software test fixture, not a captured dataset. Real results must come from
!! your own NTU images.

Usage
-----
    python scripts/make_test_scene.py --out data/_selftest --images 24 --video
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import ensure_dir, set_global_seed  # noqa: E402

WIDTH, HEIGHT = 1600, 1200



def homography_from_four_points(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Homography mapping four ``src`` corners onto four ``dst`` corners.

    Implemented here rather than calling ``cv2.getPerspectiveTransform`` so
    that no library function which fits a geometric transform appears anywhere
    in this project, not even in the test-data generator. This one only
    texture-maps a quad whose corners are already known from a known camera -
    it estimates nothing about a scene - but keeping the rule absolute means a
    reader never has to weigh that distinction.

    Standard DLT: each correspondence contributes two rows, and the solution is
    the null vector of the resulting 8x9 system.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    rows = []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, Vt = np.linalg.svd(np.array(rows))
    H = Vt[-1].reshape(3, 3)
    return (H / H[2, 2]) if abs(H[2, 2]) > 1e-12 else H

def _texture_canvas(rng: np.random.Generator) -> np.ndarray:
    """A large, richly textured 'wall' that sub-windows are cropped from.

    Random blobs + text-like marks + noise give plenty of corner structure, so
    the crops behave a little like a real textured facade under translation.
    """
    canvas = np.full((HEIGHT * 2, WIDTH * 2, 3), 190, dtype=np.uint8)
    for _ in range(400):
        centre = (int(rng.integers(0, canvas.shape[1])), int(rng.integers(0, canvas.shape[0])))
        radius = int(rng.integers(8, 70))
        colour = tuple(int(c) for c in rng.integers(20, 235, size=3))
        cv2.circle(canvas, centre, radius, colour, thickness=-1, lineType=cv2.LINE_AA)
    for _ in range(200):
        p1 = (int(rng.integers(0, canvas.shape[1])), int(rng.integers(0, canvas.shape[0])))
        p2 = (p1[0] + int(rng.integers(-260, 260)), p1[1] + int(rng.integers(-260, 260)))
        colour = tuple(int(c) for c in rng.integers(0, 255, size=3))
        cv2.line(canvas, p1, p2, colour, thickness=int(rng.integers(1, 6)), lineType=cv2.LINE_AA)
    for i in range(40):
        org = (int(rng.integers(0, canvas.shape[1] - 300)), int(rng.integers(40, canvas.shape[0])))
        cv2.putText(canvas, f"NTU-{i:03d}", org, cv2.FONT_HERSHEY_SIMPLEX,
                    float(rng.uniform(0.8, 2.4)), (30, 30, 30), 2, cv2.LINE_AA)
    noise = rng.normal(0, 7, canvas.shape).astype(np.int16)
    return np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _crop(canvas: np.ndarray, t: float) -> np.ndarray:
    """Sample a view along a smooth arc across the canvas (simulated camera path)."""
    max_x = canvas.shape[1] - WIDTH
    max_y = canvas.shape[0] - HEIGHT
    x = int(max_x * (0.5 + 0.45 * math.sin(2 * math.pi * t)))
    y = int(max_y * (0.5 + 0.30 * math.cos(2 * math.pi * t)))
    x = max(0, min(max_x, x))
    y = max(0, min(max_y, y))
    return canvas[y:y + HEIGHT, x:x + WIDTH].copy()


def _write_with_exif(path: Path, image_bgr: np.ndarray, focal_mm: float = 4.25,
                     focal_35mm: int = 26) -> bool:
    """Save a JPEG carrying a plausible smartphone EXIF block.

    Returns False (and saves without EXIF) if this Pillow build will not accept
    a synthesised sub-IFD - the pipeline must cope with both cases anyway.
    """
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    try:
        exif = img.getexif()
        exif[0x010F] = "SelfTest"          # Make
        exif[0x0110] = "SyntheticCam v1"   # Model
        exif[0x0112] = 1                   # Orientation = normal
        sub = exif.get_ifd(0x8769)
        sub[0x920A] = focal_mm             # FocalLength
        sub[0xA405] = focal_35mm           # FocalLengthIn35mmFilm
        sub[0x829D] = 1.8                  # FNumber
        sub[0x8827] = 100                  # ISO
        sub[0x9003] = "2026:08:16 10:30:00"
        img.save(path, quality=95, exif=exif.tobytes())
        return True
    except Exception:
        img.save(path, quality=95)
        return False


def build_scene(out_dir: Path, n_images: int, seed: int) -> List[Tuple[str, str]]:
    """Write the synthetic image set. Returns (filename, injected defect) pairs."""
    set_global_seed(seed)
    rng = np.random.default_rng(seed)
    raw_dir = ensure_dir(out_dir / "raw_images")
    canvas = _texture_canvas(rng)

    manifest: List[Tuple[str, str]] = []
    n_clean = max(6, n_images - 8)

    for i in range(n_clean):
        frame = _crop(canvas, i / max(1, n_clean))
        name = f"img_{i:03d}.jpg"
        _write_with_exif(raw_dir / name, frame)
        manifest.append((name, "clean"))

    base = _crop(canvas, 0.33)
    idx = n_clean

    # Motion blur (directional) and defocus blur.
    kernel = np.zeros((25, 25), np.float32)
    kernel[12, :] = 1.0 / 25.0
    name = f"img_{idx:03d}_motionblur.jpg"
    _write_with_exif(raw_dir / name, cv2.filter2D(base, -1, kernel))
    manifest.append((name, "motion blur"))
    idx += 1

    name = f"img_{idx:03d}_defocus.jpg"
    _write_with_exif(raw_dir / name, cv2.GaussianBlur(base, (31, 31), 9.0))
    manifest.append((name, "defocus blur"))
    idx += 1

    # Under- and over-exposure.
    name = f"img_{idx:03d}_dark.jpg"
    _write_with_exif(raw_dir / name, np.clip(base.astype(np.int16) - 165, 0, 255).astype(np.uint8))
    manifest.append((name, "under-exposed"))
    idx += 1

    name = f"img_{idx:03d}_bright.jpg"
    _write_with_exif(raw_dir / name, np.clip(base.astype(np.int16) + 155, 0, 255).astype(np.uint8))
    manifest.append((name, "over-exposed / highlight clipping"))
    idx += 1

    # Low contrast ("haze").
    name = f"img_{idx:03d}_lowcontrast.jpg"
    hazy = (base.astype(np.float32) * 0.16 + 118.0).astype(np.uint8)
    _write_with_exif(raw_dir / name, hazy)
    manifest.append((name, "low contrast"))
    idx += 1

    # Near-duplicate of an existing clean frame (tiny shift + JPEG re-encode).
    dup_src = _crop(canvas, 1 / max(1, n_clean))
    shifted = np.roll(dup_src, 3, axis=1)
    name = f"img_{idx:03d}_neardup.jpg"
    _write_with_exif(raw_dir / name, shifted)
    manifest.append((name, "near-duplicate of img_001"))
    idx += 1

    # Odd resolution (tests the consistency check).
    name = f"img_{idx:03d}_smallres.jpg"
    _write_with_exif(raw_dir / name, cv2.resize(base, (640, 480), interpolation=cv2.INTER_AREA))
    manifest.append((name, "different resolution 640x480"))
    idx += 1

    # Corrupt inputs: an empty file and a file of random bytes.
    (raw_dir / f"img_{idx:03d}_empty.jpg").write_bytes(b"")
    manifest.append((f"img_{idx:03d}_empty.jpg", "zero-byte file"))
    idx += 1
    (raw_dir / f"img_{idx:03d}_garbage.jpg").write_bytes(bytes(rng.integers(0, 256, 4096, dtype=np.uint8)))
    manifest.append((f"img_{idx:03d}_garbage.jpg", "non-decodable bytes"))

    return manifest


def build_video(out_dir: Path, path: Path, seed: int, seconds: float = 6.0, fps: int = 30) -> Path:
    """Write a short synthetic video for testing frame extraction."""
    rng = np.random.default_rng(seed + 1)
    canvas = _texture_canvas(rng)
    ensure_dir(path.parent)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not open a video writer for {path}")
    total = int(seconds * fps)
    for i in range(total):
        frame = _crop(canvas, i / total)
        # Blur a short burst of frames to exercise in-extraction blur rejection.
        if 40 <= i < 55:
            frame = cv2.GaussianBlur(frame, (25, 25), 7.0)
        writer.write(frame)
    writer.release()
    return path


# ===========================================================================
# Synthetic 3D room with ground-truth camera poses
# ===========================================================================
#
# The defect scene above is a set of 2D crops, so its two-view geometry is
# degenerate by construction (a pure planar homography). That is fine for
# testing the input pipeline, but it cannot validate Essential-matrix
# estimation or pose recovery, which need genuine parallax.
#
# This builds a crude "room" of textured 3D planes at different depths, plus a
# free-standing pillar, and renders it from known camera poses using exact
# perspective projection. Because the poses are known, the recovered relative
# rotation can be checked against ground truth - which is what makes stages 2
# and 7 verifiable rather than merely plausible.
#
# World convention matches OpenCV: +x right, +y DOWN, +z forward (into the
# scene). Cameras look along their own +z axis.


def _look_at(eye: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """World-to-camera rotation R and translation t for a camera at ``eye``.

    Returns (R, t) such that ``X_camera = R @ X_world + t``. The camera centre
    is recoverable as ``C = -R.T @ t`` - a distinction that matters enormously
    later, when parsing COLMAP poses.
    """
    world_up = np.array([0.0, -1.0, 0.0])  # -y is "up" because +y points down
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    R = np.vstack([right, down, forward])
    t = -R @ eye
    return R, t


def _plane(corners: np.ndarray, texture: np.ndarray) -> dict:
    """A textured quad: 4 world-space corners in texture (TL, TR, BR, BL) order."""
    return {"corners": corners.astype(np.float64), "texture": texture}


def _build_room(canvas: np.ndarray) -> List[dict]:
    """A box-shaped room plus a central pillar, each face textured differently."""
    h, w = canvas.shape[:2]

    def tile(row: int, col: int, size: int = 700) -> np.ndarray:
        y = min(row * size, h - size)
        x = min(col * size, w - size)
        return canvas[y:y + size, x:x + size].copy()

    planes = [
        # Back wall (farthest - drawn first)
        _plane(np.array([[-3, -2, 8], [3, -2, 8], [3, 2, 8], [-3, 2, 8]]), tile(0, 0)),
        # Left wall
        _plane(np.array([[-3, -2, 3], [-3, -2, 8], [-3, 2, 8], [-3, 2, 3]]), tile(0, 1)),
        # Right wall
        _plane(np.array([[3, -2, 8], [3, -2, 3], [3, 2, 3], [3, 2, 8]]), tile(1, 0)),
        # Floor
        _plane(np.array([[-3, 2, 8], [3, 2, 8], [3, 2, 3], [-3, 2, 3]]), tile(1, 1)),
        # Central pillar, much closer to the camera: this is what creates the
        # parallax that makes the Essential matrix well conditioned.
        _plane(np.array([[-0.6, -1.2, 5], [0.6, -1.2, 5], [0.6, 1.2, 5], [-0.6, 1.2, 5]]), tile(2, 0)),
        _plane(np.array([[0.6, -1.2, 5], [0.6, -1.2, 6], [0.6, 1.2, 6], [0.6, 1.2, 5]]), tile(2, 1)),
        _plane(np.array([[-0.6, -1.2, 6], [-0.6, -1.2, 5], [-0.6, 1.2, 5], [-0.6, 1.2, 6]]), tile(0, 2)),
    ]
    return planes


def _render_view(planes: List[dict], K: np.ndarray, R: np.ndarray, t: np.ndarray,
                 width: int, height: int) -> np.ndarray:
    """Render the textured planes from one camera pose by exact projection.

    Each quad is drawn with the homography that maps its texture corners onto
    its projected image corners. Faces are composited far-to-near (painter's
    algorithm), which is correct here because the geometry is convex and
    non-interpenetrating.
    """
    frame = np.full((height, width, 3), 12, dtype=np.uint8)

    renderable = []
    for plane in planes:
        camera_pts = (R @ plane["corners"].T).T + t
        if np.any(camera_pts[:, 2] <= 0.15):
            continue  # behind or too close to the camera
        projected = (K @ camera_pts.T).T
        projected = projected[:, :2] / projected[:, 2:3]
        renderable.append((float(camera_pts[:, 2].mean()), plane, projected))

    for _, plane, projected in sorted(renderable, key=lambda item: -item[0]):
        texture = plane["texture"]
        th, tw = texture.shape[:2]
        src = np.array([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]], dtype=np.float32)
        H = homography_from_four_points(src, projected)
        warped = cv2.warpPerspective(texture, H, (width, height), flags=cv2.INTER_LINEAR)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(mask, projected.astype(np.int32), 255)
        frame[mask > 0] = warped[mask > 0]

    return frame


def build_scene3d(out_dir: Path, n_views: int, seed: int,
                  width: int = 1600, height: int = 1200) -> dict:
    """Render a synthetic 3D room from an arc of cameras with known poses.

    Returns the ground-truth metadata (intrinsics + per-view R, t, centre),
    which is written to ``metadata/ground_truth_poses.json``.
    """
    set_global_seed(seed)
    rng = np.random.default_rng(seed + 7)
    canvas = _texture_canvas(rng)
    planes = _build_room(canvas)

    raw_dir = ensure_dir(out_dir / "raw_images")
    focal = 1200.0
    K = np.array([[focal, 0, width / 2.0], [0, focal, height / 2.0], [0, 0, 1.0]])
    target = np.array([0.0, 0.0, 6.0])

    views = []
    for i in range(n_views):
        u = i / max(1, n_views - 1)
        # Arc that translates laterally and in depth: real baseline, mild
        # rotation. Pure rotation would be degenerate and is avoided on purpose.
        eye = np.array([
            -2.2 + 4.4 * u,
            -0.35 + 0.7 * math.sin(2 * math.pi * u),
            0.4 + 1.1 * math.sin(math.pi * u),
        ])
        R, t = _look_at(eye, target)
        frame = _render_view(planes, K, R, t, width, height)
        name = f"view_{i:03d}.jpg"
        # EXIF must be CONSISTENT with the focal length actually rendered,
        # otherwise the pipeline's estimated K silently disagrees with ground
        # truth and every pose comparison is meaningless.
        # f_px = (focal_35mm / 36) * max(w, h)  =>  focal_35mm = f_px * 36 / max(w, h)
        focal_35 = int(round(focal * 36.0 / max(width, height)))
        _write_with_exif(raw_dir / name, frame, focal_mm=4.25, focal_35mm=focal_35)
        views.append({
            "image": name,
            "R_world_to_camera": R.tolist(),
            "t_world_to_camera": t.tolist(),
            "camera_center_world": (-R.T @ t).tolist(),
        })

    ground_truth = {
        "note": (
            "Synthetic ground truth for validating pose recovery. World frame is "
            "OpenCV-style: +x right, +y down, +z forward. X_camera = R @ X_world + t."
        ),
        "image_width": width,
        "image_height": height,
        "intrinsics": {"fx": focal, "fy": focal, "cx": width / 2.0, "cy": height / 2.0},
        "views": views,
    }
    meta_dir = ensure_dir(out_dir / "metadata")
    (meta_dir / "ground_truth_poses.json").write_text(
        json.dumps(ground_truth, indent=2), encoding="utf-8"
    )
    return ground_truth


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic self-test scene (NOT for reported results).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out", type=Path, default=Path("data/_selftest"),
                        help="Scene directory to create (raw_images/ goes inside).")
    parser.add_argument("--images", type=int, default=24, help="Approximate number of images.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video", action="store_true", help="Also write a synthetic test video.")
    parser.add_argument("--video-path", type=Path, default=None,
                        help="Where to write the video (default: <out>/../videos/selftest.mp4).")
    parser.add_argument(
        "--mode", choices=["defects", "room3d"], default="defects",
        help="'defects' injects image-quality faults; 'room3d' renders a 3D room "
             "from known camera poses (validates two-view geometry and pose recovery).",
    )
    args = parser.parse_args(argv)

    out_dir = args.out if args.out.is_absolute() else Path.cwd() / args.out

    if args.mode == "room3d":
        gt = build_scene3d(out_dir, args.images, args.seed)
        print(f"Synthetic 3D room written to {out_dir / 'raw_images'}")
        print(f"  {len(gt['views'])} views at {gt['image_width']}x{gt['image_height']}, "
              f"f = {gt['intrinsics']['fx']:.0f} px")
        print(f"  ground-truth poses: {out_dir / 'metadata' / 'ground_truth_poses.json'}")
        first = np.array(gt["views"][0]["camera_center_world"])
        last = np.array(gt["views"][-1]["camera_center_world"])
        print(f"  camera path spans {np.linalg.norm(last - first):.2f} world units "
              f"(genuine baseline, not pure rotation)")
        print("\nREMINDER: this is a software test fixture. Never report its numbers as results.")
        return 0

    manifest = build_scene(out_dir, args.images, args.seed)

    print(f"Synthetic self-test scene written to {out_dir / 'raw_images'}")
    print(f"{'file':<32} injected defect")
    print("-" * 70)
    for name, defect in manifest:
        print(f"{name:<32} {defect}")
    print("-" * 70)
    print(f"{len(manifest)} files "
          f"({sum(1 for _, d in manifest if d == 'clean')} clean, "
          f"{sum(1 for _, d in manifest if d != 'clean')} with injected defects)")

    if args.video:
        video_path = args.video_path or (out_dir.parent / "videos" / "selftest.mp4")
        path = build_video(out_dir, video_path, args.seed)
        print(f"\nSynthetic test video: {path} ({path.stat().st_size / 1e6:.1f} MB)")

    print("\nREMINDER: this is a software test fixture. Never report its numbers as results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
