"""Stage 5 CLI - analyse, filter and render a dense point cloud.

Measures the fused MVS output (point count, extent, nearest-neighbour spacing,
density), optionally removes outliers, and renders images for the report.

The original ``fused.ply`` is never modified; filtered variants are written
alongside it under new names.

Examples
--------
    python scripts/analyze_pointcloud.py --scene scene_01
    python scripts/analyze_pointcloud.py --scene scene_01 --remove-outliers
    python scripts/analyze_pointcloud.py --scene scene_01 --ply path/to/other.ply
"""

from __future__ import annotations

import argparse
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.config import ConfigError, load_config  # noqa: E402
from src.pointcloud_analysis import (  # noqa: E402
    analyse_cloud,
    load_cloud,
    remove_outliers,
    render_views,
)
from src.utils import Stopwatch, ensure_dir, human_size, setup_logging, write_json  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 5: dense point-cloud statistics, filtering and rendering.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, action="append", default=None, metavar="YAML")
    parser.add_argument("--scene", type=str, default=None, help="Override scene.name.")
    parser.add_argument(
        "--ply", type=Path, default=None,
        help="Point cloud to analyse (default: outputs/<scene>/dense/fused.ply).",
    )
    parser.add_argument(
        "--remove-outliers", action="store_true",
        help="Also produce a filtered copy and compare it against the original.",
    )
    parser.add_argument(
        "--outlier-method", choices=["statistical", "radius"], default="statistical",
    )
    parser.add_argument("--nb-neighbors", type=int, default=20)
    parser.add_argument("--std-ratio", type=float, default=2.0)
    parser.add_argument("--no-render", action="store_true", help="Skip image rendering.")
    parser.add_argument(
        "--nn-sample", type=int, default=50000,
        help="Points sampled for the nearest-neighbour distance statistics.",
    )
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def _print_stats(stats, label: str) -> None:
    print()
    print(f"  {label}")
    print("  " + "-" * 62)
    print(f"    points            : {stats.num_points:,}")
    print(f"    file size         : {human_size(stats.file_size_bytes)}")
    print(f"    colours / normals : {stats.has_colors} / {stats.has_normals}")
    print(f"    extent (raw)      : {np.round(stats.extent, 3).tolist()}")
    print(f"    extent (2-98%)    : {np.round(stats.robust_extent, 3).tolist()}")
    print(f"    bbox diagonal     : {stats.diagonal:.3f}")
    print(f"    nearest neighbour : mean {stats.nn_mean:.5f}, median {stats.nn_median:.5f}, "
          f"p95 {stats.nn_p95:.5f}")
    print(f"    density           : {stats.points_per_unit_volume:,.0f} points per unit volume")


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    overrides = []
    if args.scene:
        overrides.append(f"scene.name={args.scene}")
    try:
        cfg = load_config(args.config, overrides=overrides)
    except ConfigError as exc:
        print(f"CONFIGURATION ERROR\n{exc}", file=sys.stderr)
        return 2

    logger = setup_logging(level=args.log_level)
    ply = Path(args.ply) if args.ply else (cfg.scene_output_dir / "dense" / "fused.ply")

    try:
        with Stopwatch("stage 5", logger) as sw:
            cloud = load_cloud(ply)
            stats = analyse_cloud(cloud, ply, nn_sample=args.nn_sample, seed=cfg.runtime.seed)

            payload = {"scene": cfg.scene.name, "original": asdict(stats)}
            rendered: List[Path] = []
            viz_dir = ensure_dir(cfg.visualizations_dir / "dense")

            if not args.no_render:
                rendered = render_views(cloud, viz_dir, prefix="dense")

            filtered_stats = None
            if args.remove_outliers:
                filtered, dropped = remove_outliers(
                    cloud, method=args.outlier_method,
                    nb_neighbors=args.nb_neighbors, std_ratio=args.std_ratio,
                )
                out_ply = ply.parent / f"fused_{args.outlier_method}_filtered.ply"
                import open3d as o3d
                o3d.io.write_point_cloud(str(out_ply), filtered)
                filtered_stats = analyse_cloud(filtered, out_ply, nn_sample=args.nn_sample,
                                               seed=cfg.runtime.seed)
                payload["filtered"] = asdict(filtered_stats)
                payload["points_removed"] = int(dropped)
                payload["outlier_method"] = args.outlier_method
                if not args.no_render:
                    rendered += render_views(filtered, viz_dir, prefix="dense_filtered")
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 3
    except Exception as exc:
        logger.error("Unexpected error: %s: %s", type(exc).__name__, exc)
        logger.error("%s", traceback.format_exc())
        return 5

    print()
    print("=" * 70)
    print(f"STAGE 5 COMPLETE - scene '{cfg.scene.name}' in {sw.elapsed:.1f} s")
    print("=" * 70)
    _print_stats(stats, f"ORIGINAL: {ply.name}")

    if filtered_stats is not None:
        _print_stats(filtered_stats, f"FILTERED ({args.outlier_method}): {Path(filtered_stats.path).name}")
        removed = payload["points_removed"]
        print()
        print(f"    removed {removed:,} points "
              f"({100.0 * removed / max(1, stats.num_points):.2f} % of the original)")
        print(f"    the original fused.ply is unchanged")

    json_path = cfg.metrics_dir / "dense_analysis.json"
    write_json(json_path, payload)
    print()
    print(f"  metrics JSON       : {json_path}")
    for path in rendered:
        print(f"  render             : {path}")
    print()
    print("  Note: all distances are in ARBITRARY units. Density figures become")
    print("  metric only once an external measurement fixes the scale.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
