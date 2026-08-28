"""Build the interactive 3D viewer from reconstruction files already on disk.

Reads whichever of ``sparse.ply``, ``dense.ply``, ``surface.ply`` and
``cameras.json`` exist and writes a single self-contained HTML file. Useful for
regenerating or re-tuning the viewer without re-running the pipeline.

Examples
--------
    python scripts/build_viewer.py --scene scene_01
    python scripts/build_viewer.py --scene scene_01 --max-dense 150000 --title "NTU Carpark A"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.config import ConfigError, load_config  # noqa: E402
from src.sfm_core.viewer import build_viewer  # noqa: E402
from src.utils import human_size, setup_logging  # noqa: E402


def _parse_ply(path: Path):
    """Minimal binary/ASCII PLY reader for the files this project writes.

    Returns ``(vertices, colors, normals, faces)`` with missing parts as None.
    """
    with open(path, "rb") as fh:
        header_bytes = b""
        while b"end_header" not in header_bytes:
            chunk = fh.readline()
            if not chunk:
                raise ValueError(f"{path} has no PLY header")
            header_bytes += chunk
        header = header_bytes.decode("ascii", errors="replace").splitlines()

        binary = any("binary_little_endian" in line for line in header)
        n_vertices = n_faces = 0
        props: List[Tuple[str, str]] = []
        element = None
        type_map = {"float": "f4", "float32": "f4", "double": "f8",
                    "uchar": "u1", "uint8": "u1", "int": "i4", "int32": "i4",
                    "uint": "u4", "short": "i2", "ushort": "u2"}
        for line in header:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "element":
                element = parts[1]
                if element == "vertex":
                    n_vertices = int(parts[2])
                elif element == "face":
                    n_faces = int(parts[2])
            elif parts[0] == "property" and element == "vertex" and parts[1] != "list":
                props.append((parts[2], type_map.get(parts[1], "f4")))

        if not binary:
            raise ValueError(f"{path} is ASCII PLY; only binary is supported here")

        dtype = np.dtype([(name, fmt) for name, fmt in props])
        verts = np.frombuffer(fh.read(n_vertices * dtype.itemsize), dtype=dtype, count=n_vertices)

        faces = None
        if n_faces:
            raw = fh.read()
            # Each face record is one uchar count followed by three int32.
            rec = np.dtype([("n", "u1"), ("a", "i4"), ("b", "i4"), ("c", "i4")])
            usable = len(raw) // rec.itemsize
            arr = np.frombuffer(raw[: usable * rec.itemsize], dtype=rec, count=usable)
            faces = np.stack([arr["a"], arr["b"], arr["c"]], axis=1).astype(np.int32)

    names = verts.dtype.names
    xyz = np.stack([verts["x"], verts["y"], verts["z"]], axis=1).astype(np.float64)
    colors = (np.stack([verts["red"], verts["green"], verts["blue"]], axis=1).astype(np.uint8)
              if {"red", "green", "blue"} <= set(names) else None)
    normals = (np.stack([verts["nx"], verts["ny"], verts["nz"]], axis=1).astype(np.float64)
               if {"nx", "ny", "nz"} <= set(names) else None)
    return xyz, colors, normals, faces


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build the interactive 3D viewer from reconstruction files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=Path, action="append", default=None, metavar="YAML")
    p.add_argument("--scene", type=str, default=None)
    p.add_argument("--dir", type=Path, default=None,
                   help="Directory holding sparse.ply / dense.ply / surface.ply / cameras.json.")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--title", type=str, default=None)
    p.add_argument("--subtitle", type=str,
                   default="Sparse map, camera poses, and full-resolution geometry")
    p.add_argument("--max-dense", type=int, default=260000)
    p.add_argument("--max-faces", type=int, default=220000)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    overrides = [f"scene.name={args.scene}"] if args.scene else []
    try:
        cfg = load_config(args.config, overrides=overrides)
    except ConfigError as exc:
        print(f"CONFIGURATION ERROR\n{exc}", file=sys.stderr)
        return 2
    setup_logging(level="INFO")

    directory = Path(args.dir) if args.dir else (cfg.scene_output_dir / "scratch")
    if not directory.is_dir():
        print(f"Reconstruction directory not found: {directory}\n"
              f"Run: python scripts/run_scratch_sfm.py --scene {cfg.scene.name}", file=sys.stderr)
        return 3

    sparse_path = directory / "sparse.ply"
    if not sparse_path.is_file():
        print(f"{sparse_path} is required but missing", file=sys.stderr)
        return 3
    sparse_pts, sparse_cols, _, _ = _parse_ply(sparse_path)
    if sparse_cols is None:
        sparse_cols = np.full((len(sparse_pts), 3), 200, np.uint8)
    print(f"  sparse : {len(sparse_pts):,} points")

    dense_pts = dense_cols = None
    if (directory / "dense.ply").is_file():
        dense_pts, dense_cols, _, _ = _parse_ply(directory / "dense.ply")
        if dense_cols is None:
            dense_cols = np.full((len(dense_pts), 3), 190, np.uint8)
        print(f"  dense  : {len(dense_pts):,} points")

    mesh_v = mesh_f = mesh_c = None
    if (directory / "surface.ply").is_file():
        mesh_v, mesh_c, _, mesh_f = _parse_ply(directory / "surface.ply")
        print(f"  surface: {len(mesh_v):,} vertices, "
              f"{0 if mesh_f is None else len(mesh_f):,} faces")

    centers = rotations = None
    K = None
    image_size = (1600, 1200)
    cams_path = directory / "cameras.json"
    if cams_path.is_file():
        cams = json.loads(cams_path.read_text(encoding="utf-8"))
        entries = cams.get("cameras", [])
        if entries:
            centers = np.array([c["camera_center"] for c in entries])
            rotations = np.array([c["R_world_to_camera"] for c in entries])
            groups = cams.get("camera_groups") or {}
            first = groups.get(str(entries[0].get("camera_group", 0))) or \
                next(iter(groups.values()), None) or cams.get("intrinsics")
            if first:
                K = np.array([[first["fx"], 0, first["cx"]],
                              [0, first["fy"], first["cy"]], [0, 0, 1.0]])
                image_size = (int(first["width"]), int(first["height"]))
        print(f"  cameras: {0 if centers is None else len(centers)} poses")

    output = Path(args.output) if args.output else \
        directory / f"{cfg.scene.name}_reconstruction.html"
    title = args.title or cfg.scene.name.replace("_", " ").title()

    build_viewer(
        output, title=title, subtitle=args.subtitle,
        sparse_points=sparse_pts, sparse_colors=sparse_cols,
        dense_points=dense_pts, dense_colors=dense_cols,
        mesh_vertices=mesh_v, mesh_faces=mesh_f, mesh_colors=mesh_c,
        camera_centers=centers, camera_rotations=rotations,
        intrinsics=K, image_size=image_size,
        max_dense_points=args.max_dense, max_mesh_faces=args.max_faces,
    )
    print()
    print(f"  viewer : {output}")
    print(f"  size   : {human_size(output.stat().st_size)}")
    print("  open it by double-clicking, or drag it into a browser")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
