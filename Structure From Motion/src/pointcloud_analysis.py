"""Stage 5 - dense point-cloud analysis, filtering and rendering (Open3D).

Loads a fused dense cloud, measures it, optionally removes outliers, and
renders it to images for the report.

The original ``fused.ply`` is never modified. Every filtered variant is written
alongside it under a new name, so the raw MVS output stays available as the
reference for every comparison.

Density and scale
-----------------
Nearest-neighbour spacing is the honest way to describe how dense a cloud is:
"31 million points" means nothing without knowing the volume they occupy.
All distances are in the reconstruction's arbitrary units - monocular SfM has
no metric scale, so "points per cubic unit" only becomes "points per cubic
metre" once an external measurement fixes the scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .utils import ensure_dir, get_logger

__all__ = [
    "PointCloudStats",
    "load_cloud",
    "analyse_cloud",
    "remove_outliers",
    "render_views",
]

LOGGER = get_logger()


@dataclass
class PointCloudStats:
    """Measurements of one dense point cloud."""

    path: str = ""
    num_points: int = 0
    file_size_bytes: int = 0
    has_colors: bool = False
    has_normals: bool = False
    bbox_min: List[float] = field(default_factory=list)
    bbox_max: List[float] = field(default_factory=list)
    extent: List[float] = field(default_factory=list)
    diagonal: float = 0.0
    robust_extent: List[float] = field(default_factory=list)
    centroid: List[float] = field(default_factory=list)
    nn_mean: float = 0.0
    nn_median: float = 0.0
    nn_p95: float = 0.0
    nn_std: float = 0.0
    points_per_unit_volume: float = 0.0
    sampled_for_nn: int = 0


def load_cloud(path: Path):
    """Load a PLY point cloud with Open3D.

    Raises
    ------
    FileNotFoundError, ValueError
        If the file is missing or contains no points.
    """
    import open3d as o3d

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Point cloud not found: {path}\n"
            f"Run dense reconstruction first: run_reconstruction.py --scene <name> --dense"
        )
    cloud = o3d.io.read_point_cloud(str(path))
    if len(cloud.points) == 0:
        raise ValueError(f"{path} contains no points")
    return cloud


def analyse_cloud(cloud, path: Path, nn_sample: int = 50000, seed: int = 42) -> PointCloudStats:
    """Measure size, extent and nearest-neighbour spacing.

    Nearest-neighbour distances are computed on a random subsample: a full k-d
    tree query over tens of millions of points costs minutes, and the spacing
    distribution is already well estimated from a large sample.
    """
    import open3d as o3d

    points = np.asarray(cloud.points)
    stats = PointCloudStats(
        path=str(path),
        num_points=int(len(points)),
        file_size_bytes=int(Path(path).stat().st_size) if Path(path).is_file() else 0,
        has_colors=bool(cloud.has_colors()),
        has_normals=bool(cloud.has_normals()),
    )

    lo, hi = points.min(axis=0), points.max(axis=0)
    extent = hi - lo
    p2, p98 = np.percentile(points, [2, 98], axis=0)
    stats.bbox_min = lo.tolist()
    stats.bbox_max = hi.tolist()
    stats.extent = extent.tolist()
    stats.diagonal = float(np.linalg.norm(extent))
    stats.robust_extent = (p98 - p2).tolist()
    stats.centroid = points.mean(axis=0).tolist()

    rng = np.random.default_rng(seed)
    count = min(nn_sample, len(points))
    indices = rng.choice(len(points), size=count, replace=False)
    tree = o3d.geometry.KDTreeFlann(cloud)
    distances = np.full(count, np.nan, dtype=np.float64)
    for i, j in enumerate(indices):
        # k=2 because the first neighbour returned is the query point itself.
        _, _, squared = tree.search_knn_vector_3d(cloud.points[int(j)], 2)
        if len(squared) > 1:
            distances[i] = float(np.sqrt(squared[1]))
    distances = distances[np.isfinite(distances)]
    if distances.size:
        stats.nn_mean = float(distances.mean())
        stats.nn_median = float(np.median(distances))
        stats.nn_p95 = float(np.percentile(distances, 95))
        stats.nn_std = float(distances.std())
    stats.sampled_for_nn = int(distances.size)

    # Volume from the robust extent: a handful of stray points on the horizon
    # would otherwise inflate the box and make the density look far too low.
    robust_volume = float(np.prod(p98 - p2))
    if robust_volume > 0:
        stats.points_per_unit_volume = float(len(points) / robust_volume)
    return stats


def remove_outliers(
    cloud,
    method: str = "statistical",
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
    radius: float = 0.05,
    min_points: int = 16,
) -> Tuple[Any, int]:
    """Remove outliers, returning the filtered cloud and how many were dropped.

    ``statistical`` drops points whose mean neighbour distance exceeds the
    cloud average by ``std_ratio`` standard deviations. This suits the
    scattered noise MVS produces over sky and untextured regions.

    ``radius`` drops points with fewer than ``min_points`` neighbours within
    ``radius``. Better for isolated floating clusters, but the radius is in
    arbitrary reconstruction units, so it must be chosen per scene.

    Raises
    ------
    ValueError
        For an unknown method.
    """
    before = len(cloud.points)
    if method == "statistical":
        filtered, _ = cloud.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, std_ratio=std_ratio
        )
    elif method == "radius":
        filtered, _ = cloud.remove_radius_outlier(nb_points=min_points, radius=radius)
    else:
        raise ValueError(f"Unknown outlier method {method!r}. Use 'statistical' or 'radius'.")
    return filtered, before - len(filtered.points)


def crop_to_percentile(cloud, low: float = 1.0, high: float = 99.0):
    """Return a copy of the cloud with the extreme tails removed, for framing.

    MVS scatters a small number of points onto the sky and the far horizon.
    They are a negligible fraction of the cloud but they dominate its bounding
    box - on this project's own scene the raw extent was 46.7 units while the
    2-98 percentile extent was 5.5. Any renderer that frames on the raw box
    therefore zooms out until the actual subject is a few pixels wide.

    This is used for *rendering only*. Measurements and saved clouds always use
    the full data.
    """
    import open3d as o3d

    points = np.asarray(cloud.points)
    if len(points) == 0:
        return cloud
    lo = np.percentile(points, low, axis=0)
    hi = np.percentile(points, high, axis=0)
    box = o3d.geometry.AxisAlignedBoundingBox(min_bound=lo, max_bound=hi)
    return cloud.crop(box)


def render_views(
    cloud,
    out_dir: Path,
    prefix: str = "dense",
    width: int = 1600,
    height: int = 1200,
    point_size: float = 1.5,
    crop_for_framing: bool = True,
) -> List[Path]:
    """Render the cloud from several viewpoints to PNG.

    Tries Open3D's offscreen visualiser first, and falls back to a matplotlib
    orthographic projection when no OpenGL context is available - a report
    needs the picture either way.

    ``crop_for_framing`` discards the 1st/99th percentile tails before
    rendering so the subject fills the frame. It affects only the image.
    """
    import open3d as o3d

    out_dir = ensure_dir(Path(out_dir))
    if crop_for_framing:
        cloud = crop_to_percentile(cloud)
    views = [("front", 0.0, 0.0), ("top", 0.0, -88.0),
             ("left", -70.0, -18.0), ("right", 70.0, -18.0)]
    written: List[Path] = []

    try:
        vis = o3d.visualization.Visualizer()
        if not vis.create_window(visible=False, width=width, height=height):
            raise RuntimeError("could not create an OpenGL window")
        vis.add_geometry(cloud)
        option = vis.get_render_option()
        option.point_size = point_size
        option.background_color = np.asarray([1.0, 1.0, 1.0])

        for name, yaw, pitch in views:
            control = vis.get_view_control()
            control.set_zoom(0.72)
            # rotate() takes pixel deltas, roughly 5.5 px per degree by default.
            control.rotate(yaw * 5.5, pitch * 5.5)
            vis.poll_events()
            vis.update_renderer()
            path = out_dir / f"{prefix}_{name}.png"
            vis.capture_screen_image(str(path), do_render=True)
            written.append(path)
            control.rotate(-yaw * 5.5, -pitch * 5.5)  # restore for the next view
        vis.destroy_window()
        if written:
            return written
    except Exception as exc:
        LOGGER.warning("Open3D offscreen rendering unavailable (%s); using matplotlib", exc)

    return _render_matplotlib(cloud, out_dir, prefix)


def _render_matplotlib(cloud, out_dir: Path, prefix: str) -> List[Path]:
    """Fallback renderer: orthographic scatter projections via matplotlib."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = np.asarray(cloud.points)
    colors = np.asarray(cloud.colors) if cloud.has_colors() else None
    if len(points) > 400000:
        rng = np.random.default_rng(0)
        indices = rng.choice(len(points), 400000, replace=False)
        points = points[indices]
        colors = colors[indices] if colors is not None else None

    written: List[Path] = []
    for name, (a, b) in (("front", (0, 1)), ("top", (0, 2)), ("side", (2, 1))):
        fig, ax = plt.subplots(figsize=(11, 8.5), facecolor="white")
        ax.scatter(points[:, a], points[:, b], s=0.35,
                   c=colors if colors is not None else "#333333", linewidths=0)
        # Percentile limits, not min/max: a few surviving strays would still
        # push the axes out and shrink the subject.
        ax.set_xlim(np.percentile(points[:, a], 0.5), np.percentile(points[:, a], 99.5))
        ax.set_ylim(np.percentile(points[:, b], 0.5), np.percentile(points[:, b], 99.5))
        ax.set_aspect("equal", adjustable="box")
        if name in {"front", "side"}:
            ax.invert_yaxis()  # +y points down in the reconstruction frame
        ax.set_title(f"Dense point cloud - {name} view ({len(points):,} points shown)")
        ax.set_xlabel(f"{'XYZ'[a]} [arb. units]")
        ax.set_ylabel(f"{'XYZ'[b]} [arb. units]")
        ax.grid(alpha=0.2, linestyle="--")
        path = Path(out_dir) / f"{prefix}_{name}.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    return written
