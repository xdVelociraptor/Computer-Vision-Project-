"""Surface reconstruction by volumetric TSDF fusion.

Depth maps are integrated into a truncated signed-distance field, then the
zero-level isosurface is extracted as a triangle mesh.

Why TSDF rather than Poisson
----------------------------
Screened Poisson reconstruction is the usual choice, but it is a large,
subtle algorithm and using a library implementation would sit awkwardly beside
a project whose point is to implement the geometry itself. TSDF fusion is
simple enough to write honestly and has a real advantage here: it consumes the
*depth maps* directly, so it exploits per-view visibility and free-space
information that a bare point cloud has already discarded.

Method
------
For every voxel and every view: project the voxel centre into the camera, read
the measured depth at that pixel, and take the signed difference between the
measured surface depth and the voxel's own depth::

    sdf = depth_measured - depth_voxel

Positive means the voxel is in front of the observed surface (free space),
negative means behind it (occluded). Values are truncated to a narrow band
around zero - beyond it a single depth reading says nothing useful - and
accumulated as a running weighted mean across views. Colour is averaged with
the same weights.

The zero crossing of that field is the surface, extracted with marching cubes.
``skimage.measure.marching_cubes`` is a generic isosurface extractor: it knows
nothing about cameras, depth or reconstruction, and is used the same way one
would use an FFT.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..utils import get_logger

__all__ = ["TSDFConfig", "SurfaceResult", "fuse_tsdf", "extract_surface"]

LOGGER = get_logger()


@dataclass
class TSDFConfig:
    """Volumetric fusion parameters."""

    #: Voxels along the longest axis of the scene bounding box.
    resolution: int = 256
    #: Truncation distance as a multiple of the voxel size. Larger fills gaps
    #: but rounds off detail.
    truncation_voxels: float = 4.0
    #: Bounding box taken from this percentile range of the dense points, so a
    #: few stray points cannot blow up the grid.
    bbox_percentile: Tuple[float, float] = (1.0, 99.0)
    #: Ignore depth readings beyond this distance (unreliable far field).
    max_depth: float = 0.0
    #: Voxels seen by fewer than this many views are left empty.
    min_weight: float = 2.0
    #: Voxels processed per batch, to bound peak memory.
    chunk: int = 2_000_000


@dataclass
class SurfaceResult:
    vertices: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    faces: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.int32))
    normals: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    colors: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.uint8))
    stats: Dict[str, float] = field(default_factory=dict)


def fuse_tsdf(
    depth_maps: Sequence,
    bounds: Tuple[np.ndarray, np.ndarray],
    cfg: Optional[TSDFConfig] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Integrate depth maps into a TSDF volume.

    Returns ``(tsdf, weights, color_volume, origin, voxel_size)`` with the
    volumes shaped ``(nx, ny, nz)``.
    """
    cfg = cfg or TSDFConfig()
    lo, hi = bounds
    extent = hi - lo
    voxel_size = float(extent.max()) / cfg.resolution
    if voxel_size <= 0:
        raise ValueError("Degenerate scene bounds; cannot build a TSDF volume")
    truncation = cfg.truncation_voxels * voxel_size

    dims = np.maximum(np.ceil(extent / voxel_size).astype(int), 1)
    n_voxels = int(np.prod(dims))
    LOGGER.info("TSDF volume %s = %.1fM voxels, voxel size %.5f, truncation %.5f",
                tuple(dims), n_voxels / 1e6, voxel_size, truncation)

    tsdf = np.ones(n_voxels, dtype=np.float32)
    weights = np.zeros(n_voxels, dtype=np.float32)
    color = np.zeros((n_voxels, 3), dtype=np.float32)

    # Voxel centre coordinates, generated per chunk rather than all at once.
    ix, iy, iz = np.unravel_index(np.arange(n_voxels), dims)
    start = time.perf_counter()

    for dm in depth_maps:
        h, w = dm.depth.shape
        R, t, K = dm.R, dm.t, dm.K
        for begin in range(0, n_voxels, cfg.chunk):
            end = min(begin + cfg.chunk, n_voxels)
            sl = slice(begin, end)
            centres = np.stack([ix[sl], iy[sl], iz[sl]], axis=1).astype(np.float32)
            world = lo + (centres + 0.5) * voxel_size

            cam = world @ R.T.astype(np.float32) + t.astype(np.float32)
            z = cam[:, 2]
            in_front = z > 1e-6
            if not np.any(in_front):
                continue

            uv = (cam[in_front] / z[in_front, None]) @ K.T.astype(np.float32)
            u = np.round(uv[:, 0]).astype(np.int32)
            v = np.round(uv[:, 1]).astype(np.int32)
            inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            if not np.any(inside):
                continue

            global_idx = np.flatnonzero(in_front)[inside]
            u, v = u[inside], v[inside]
            measured = dm.depth[v, u]
            has_depth = measured > 0
            if cfg.max_depth > 0:
                has_depth &= measured < cfg.max_depth
            if not np.any(has_depth):
                continue

            global_idx = global_idx[has_depth]
            u, v = u[has_depth], v[has_depth]
            measured = measured[has_depth].astype(np.float32)
            voxel_depth = z[in_front][inside][has_depth].astype(np.float32)

            sdf = measured - voxel_depth
            # Behind the surface by more than the truncation band: this view
            # cannot see through the object, so it carries no information.
            band = sdf >= -truncation
            if not np.any(band):
                continue
            global_idx = global_idx[band]
            sdf = np.clip(sdf[band] / truncation, -1.0, 1.0)
            uu, vv = u[band], v[band]

            absolute = begin + global_idx
            w_old = weights[absolute]
            w_new = w_old + 1.0
            tsdf[absolute] = (tsdf[absolute] * w_old + sdf) / w_new
            rgb = dm.color[vv, uu].astype(np.float32)
            color[absolute] = (color[absolute] * w_old[:, None] + rgb) / w_new[:, None]
            weights[absolute] = w_new

    LOGGER.info("  integrated %d depth maps in %.1f s (%.1f%% of voxels observed)",
                len(depth_maps), time.perf_counter() - start,
                100.0 * float((weights > 0).mean()))

    return (tsdf.reshape(dims), weights.reshape(dims),
            color.reshape(*dims, 3), lo, voxel_size)


def extract_surface(
    depth_maps: Sequence,
    dense_points: np.ndarray,
    cfg: Optional[TSDFConfig] = None,
) -> SurfaceResult:
    """Fuse depth maps and extract the zero-level isosurface as a mesh."""
    cfg = cfg or TSDFConfig()
    result = SurfaceResult()
    if len(depth_maps) == 0 or len(dense_points) == 0:
        result.stats["note"] = "no depth maps or points to fuse"
        return result

    lo_pct, hi_pct = cfg.bbox_percentile
    lo = np.percentile(dense_points, lo_pct, axis=0).astype(np.float32)
    hi = np.percentile(dense_points, hi_pct, axis=0).astype(np.float32)
    pad = (hi - lo) * 0.03
    lo, hi = lo - pad, hi + pad

    start = time.perf_counter()
    tsdf, weights, color, origin, voxel_size = fuse_tsdf(depth_maps, (lo, hi), cfg)

    # Unobserved voxels must not create surface: force them to "outside" so
    # marching cubes finds no zero crossing there.
    tsdf = tsdf.copy()
    tsdf[weights < cfg.min_weight] = 1.0

    if not (tsdf.min() < 0.0 < tsdf.max()):
        result.stats["note"] = "TSDF has no zero crossing; nothing to extract"
        LOGGER.warning(result.stats["note"])
        return result

    from skimage import measure   # generic isosurface extraction

    verts, faces, normals, _ = measure.marching_cubes(tsdf, level=0.0,
                                                      spacing=(voxel_size,) * 3)
    verts_world = verts + origin

    # Colour each vertex from the fused colour volume at its nearest voxel.
    idx = np.clip(np.round(verts / voxel_size).astype(int), 0,
                  np.array(tsdf.shape) - 1)
    vert_colors = color[idx[:, 0], idx[:, 1], idx[:, 2]].astype(np.uint8)

    result.vertices = verts_world
    result.faces = faces.astype(np.int32)
    result.normals = normals
    result.colors = vert_colors
    result.stats = {
        "vertices": len(verts_world),
        "faces": len(faces),
        "voxel_size": voxel_size,
        "resolution": cfg.resolution,
        "observed_voxel_fraction": float((weights > 0).mean()),
        "time": time.perf_counter() - start,
    }
    LOGGER.info("  surface: %d vertices, %d faces in %.1f s",
                len(verts_world), len(faces), result.stats["time"])
    return result
