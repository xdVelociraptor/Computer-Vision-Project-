"""Complete from-scratch SfM + MVS pipeline. No COLMAP, no SfM library.

    images -> SIFT -> matching -> geometric verification (own RANSAC + E)
           -> tracks -> seed pair -> incremental registration (own PnP)
           -> triangulation (own DLT) -> bundle adjustment (own residuals)
           -> plane-sweep MVS (own) -> TSDF fusion -> marching cubes
           -> PLY exports + interactive 3D viewer

OpenCV provides image I/O, SIFT and descriptor matching. NumPy provides linear
algebra, SciPy a generic least-squares solver, scikit-image a generic
isosurface extractor. Every piece of reconstruction geometry is implemented in
``src/sfm_core``.

Examples
--------
    python scripts/run_scratch_sfm.py --scene scene_01
    python scripts/run_scratch_sfm.py --scene scene_01 --max-images 60 --no-dense
    python scripts/run_scratch_sfm.py --scene scene_01 --dense-size 1000 --depths 128
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.camera import estimate_intrinsics  # noqa: E402
from src.config import ConfigError, load_config  # noqa: E402
from src.preprocess import read_exif  # noqa: E402
from src.sfm_core import IncrementalMapper, MapperConfig, extract_features  # noqa: E402
from src.sfm_core.sources import collect_inputs  # noqa: E402
from src.sfm_core.dense import DenseConfig, run_dense_stereo  # noqa: E402
from src.sfm_core.surface import TSDFConfig, extract_surface  # noqa: E402
from src.utils import (  # noqa: E402
    Stopwatch, ensure_dir, human_size, iter_images, set_global_seed,
    setup_logging, timestamp, write_json,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="From-scratch Structure from Motion and Multi-View Stereo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=Path, action="append", default=None, metavar="YAML")
    p.add_argument("--scene", type=str, default=None)
    p.add_argument("--images", type=Path, action="append", default=None,
                   help="Image folder. Repeatable; each folder is its own camera group.")
    p.add_argument("--video", type=Path, action="append", default=None,
                   help="Video file. Repeatable; each video is its own camera group, "
                        "reconstructed jointly with the photographs.")
    p.add_argument("--video-fps", type=float, default=2.0,
                   help="Frames sampled per second of video.")
    p.add_argument("--max-video-frames", type=int, default=0,
                   help="Cap on frames kept per video (0 = unlimited).")
    p.add_argument("--max-images", type=int, default=0,
                   help="Evenly subsample to at most this many images (0 = all).")
    p.add_argument("--feature-size", type=int, default=1600,
                   help="Long side for feature detection.")
    p.add_argument("--max-features", type=int, default=8000)
    p.add_argument("--feature", default="sift", choices=["sift", "akaze", "orb"],
                   help="Local feature detector. SIFT is the strongest baseline; "
                        "AKAZE uses a nonlinear scale space and a binary descriptor; "
                        "ORB is fastest but least robust to viewpoint change. "
                        "Detection and description estimate no geometry, so the "
                        "choice does not affect which reconstruction code runs.")
    p.add_argument("--no-root-sift", action="store_true",
                   help="Use raw SIFT descriptors instead of the Hellinger "
                        "(RootSIFT) transform. For ablation only.")
    p.add_argument("--focal", type=float, default=0.0,
                   help="Fix the focal length in pixels (at --feature-size scale) and skip "
                        "self-calibration. Self-calibration needs the focal-induced "
                        "reprojection signal to exceed feature-localisation noise; on "
                        "heavily compressed images it does not, so a known calibration is "
                        "both more accurate and more honest than a scan.")
    p.add_argument("--sequential-window", type=int, default=0,
                   help="0 = exhaustive matching; N = only match within N frames.")
    p.add_argument("--no-dense", action="store_true")
    p.add_argument("--no-surface", action="store_true")
    p.add_argument("--dense-size", type=int, default=800, help="Long side for MVS.")
    p.add_argument("--depths", type=int, default=96, help="Depth planes per view.")
    p.add_argument("--source-views", type=int, default=6)
    p.add_argument("--min-ncc", type=float, default=0.35)
    p.add_argument("--consistent-views", type=int, default=2,
                   help="How many other views must independently agree with a depth. "
                        "The strongest control on dense noise.")
    p.add_argument("--no-library-warp", action="store_true",
                   help="Use the project's own bilinear sampler instead of "
                        "cv2.warpPerspective. ~40x slower, no quality gain; provided to "
                        "demonstrate the library performs no geometric work.")
    p.add_argument("--tsdf-resolution", type=int, default=256)
    p.add_argument("--no-viewer", action="store_true")
    p.add_argument("--reuse-sparse", action="store_true",
                   help="Load poses and points from a previous run instead of re-running "
                        "SfM. Lets the dense stage be re-tuned in minutes.")
    p.add_argument("--multi-model", action="store_true",
                   help="Reconstruct disconnected groups as separate models and "
                        "align them with a similarity transform, instead of "
                        "discarding whatever the first model could not reach.")
    p.add_argument("--force", action="store_true", help="Ignore cached features.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def write_ply_points(path: Path, points: np.ndarray, colors: Optional[np.ndarray] = None,
                     normals: Optional[np.ndarray] = None) -> Path:
    """Binary little-endian PLY point cloud."""
    path = Path(path)
    ensure_dir(path.parent)
    n = len(points)
    fields = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}",
              "property float x", "property float y", "property float z"]
    if normals is not None:
        header += ["property float nx", "property float ny", "property float nz"]
        fields += [("nx", "f4"), ("ny", "f4"), ("nz", "f4")]
    if colors is not None:
        header += ["property uchar red", "property uchar green", "property uchar blue"]
        fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    header.append("end_header")

    arr = np.empty(n, dtype=fields)
    arr["x"], arr["y"], arr["z"] = points[:, 0], points[:, 1], points[:, 2]
    if normals is not None:
        arr["nx"], arr["ny"], arr["nz"] = normals[:, 0], normals[:, 1], normals[:, 2]
    if colors is not None:
        c = np.asarray(colors, np.uint8)
        arr["red"], arr["green"], arr["blue"] = c[:, 0], c[:, 1], c[:, 2]
    with open(path, "wb") as fh:
        fh.write(("\n".join(header) + "\n").encode("ascii"))
        arr.tofile(fh)
    return path


def write_ply_mesh(path: Path, vertices: np.ndarray, faces: np.ndarray,
                   colors: Optional[np.ndarray] = None) -> Path:
    """Binary little-endian PLY triangle mesh."""
    path = Path(path)
    ensure_dir(path.parent)
    header = ["ply", "format binary_little_endian 1.0",
              f"element vertex {len(vertices)}",
              "property float x", "property float y", "property float z"]
    fields = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    if colors is not None:
        header += ["property uchar red", "property uchar green", "property uchar blue"]
        fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    header += [f"element face {len(faces)}", "property list uchar int vertex_indices",
               "end_header"]

    varr = np.empty(len(vertices), dtype=fields)
    varr["x"], varr["y"], varr["z"] = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    if colors is not None:
        c = np.asarray(colors, np.uint8)
        varr["red"], varr["green"], varr["blue"] = c[:, 0], c[:, 1], c[:, 2]

    farr = np.empty(len(faces), dtype=[("n", "u1"), ("a", "i4"), ("b", "i4"), ("c", "i4")])
    farr["n"] = 3
    farr["a"], farr["b"], farr["c"] = faces[:, 0], faces[:, 1], faces[:, 2]

    with open(path, "wb") as fh:
        fh.write(("\n".join(header) + "\n").encode("ascii"))
        varr.tofile(fh)
        farr.tofile(fh)
    return path


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    overrides = [f"scene.name={args.scene}"] if args.scene else []
    try:
        cfg = load_config(args.config, overrides=overrides)
    except ConfigError as exc:
        print(f"CONFIGURATION ERROR\n{exc}", file=sys.stderr)
        return 2

    run_id = timestamp()
    out_dir = ensure_dir(cfg.scene_output_dir / "scratch")
    log_path = ensure_dir(cfg.logs_dir) / f"scratch_sfm_{run_id}.log"
    logger = setup_logging(log_path, level=args.log_level)
    set_global_seed(cfg.runtime.seed)

    image_dirs = [Path(d) for d in (args.images or [])]
    if not image_dirs and not args.video:
        default_dir = cfg.processed_images_dir
        if not default_dir.is_dir():
            default_dir = cfg.raw_images_dir
        image_dirs = [default_dir]

    try:
        inputs = collect_inputs(
            image_dirs=image_dirs,
            videos=[Path(v) for v in (args.video or [])],
            frame_dir=cfg.raw_images_dir.parent / "video_frames",
            target_fps=args.video_fps,
            max_video_frames=args.max_video_frames,
            max_images_per_source=args.max_images,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"INPUT ERROR: {exc}", file=sys.stderr)
        return 3
    paths, groups = inputs.paths, inputs.groups
    if len(paths) < 3:
        print(f"Need at least 3 images, found {len(paths)}", file=sys.stderr)
        return 3

    logger.info("=" * 70)
    logger.info("FROM-SCRATCH SfM  |  scene=%s  images=%d  sources=%d  run=%s",
                cfg.scene.name, len(paths), len(inputs.sources), run_id)
    logger.info("=" * 70)

    summary: Dict[str, object] = {"scene": cfg.scene.name, "run_id": run_id,
                                  "num_input_images": len(paths),
                                  "sources": inputs.summary(),
                                  "implementation": "from scratch (no COLMAP / no SfM library)"}

    with Stopwatch("total", logger) as total_sw:
        # --- features ------------------------------------------------------
        cache = None if args.force else (ensure_dir(cfg.cache_dir) / f"scratch_feats_{args.feature}_{args.feature_size}.pkl")
        t0 = time.perf_counter()
        features, feat_scale = extract_features(
            paths, max_features=args.max_features, max_image_size=args.feature_size,
            cache_path=cache, root_sift=not args.no_root_sift,
            method=args.feature,
        )
        feature_time = time.perf_counter() - t0
        logger.info("Features: %d keypoints across %d images in %.1f s",
                    sum(len(f) for f in features), len(features), feature_time)

        # --- intrinsics: one set per camera group ---------------------------
        # A phone's stills and its video do not share a field of view (video is
        # cropped and rescaled), so each source is calibrated separately from
        # its own frame dimensions. Bundle adjustment then refines one focal
        # length per group rather than one globally.
        group_intrinsics = {}
        for spec in inputs.sources:
            idx = groups.index(spec.group)
            probe = read_exif(paths[idx]) if cfg.camera.use_exif_focal_prior else {}
            gw, gh = features[idx].width, features[idx].height
            group_intrinsics[spec.group] = estimate_intrinsics(
                gw, gh,
                focal_length_35mm=probe.get("focal_length_35mm"),
                focal_length_mm=probe.get("focal_length_mm"),
                default_focal_factor=cfg.camera.default_focal_factor,
            )
            logger.info("Group %d (%s): %d images, f=%.1f px at %dx%d (%s)",
                        spec.group, spec.kind, len(spec.paths),
                        group_intrinsics[spec.group].K[0, 0], gw, gh,
                        group_intrinsics[spec.group].provenance)

        Ks_feat = [group_intrinsics[g].K.copy() for g in groups]
        K = Ks_feat[0]
        intr = group_intrinsics[groups[0]]
        w0, h0 = features[0].width, features[0].height

        # --- sparse reconstruction -----------------------------------------
        if args.focal > 0:
            for g in group_intrinsics:
                group_intrinsics[g].fx = group_intrinsics[g].fy = args.focal
            Ks_feat = [group_intrinsics[g].K.copy() for g in groups]
            K = Ks_feat[0]
            logger.info("Focal fixed at %.1f px by --focal; self-calibration disabled",
                        args.focal)

        mapper_cfg = MapperConfig(
            sequential_window=args.sequential_window,
            seed=cfg.runtime.seed,
            refine_focal=False,
            refine_distortion=False,
            auto_focal=(args.focal <= 0),
        )
        match_cache = None if args.force else (
            ensure_dir(cfg.cache_dir) / f"scratch_matches_{args.feature}_{args.feature_size}_"
            f"{args.max_features}_{args.sequential_window}.pkl")
        mapper = IncrementalMapper(features, Ks_feat, mapper_cfg, groups=groups,
                                   match_cache=match_cache)
        cams_file = out_dir / "cameras.json"
        if args.reuse_sparse and cams_file.is_file():
            from src.utils import read_json
            saved = read_json(cams_file)
            mapper.match_all()
            mapper.build_tracks()
            rec = mapper.rec
            rec.Ks = {i: Ks_feat[i].copy() for i in range(len(features))}
            rec.groups = {i: groups[i] for i in range(len(features))}
            by_name = {f.name: i for i, f in enumerate(features)}
            for entry in saved["cameras"]:
                idx = by_name.get(entry["image"])
                if idx is not None:
                    rec.poses[idx] = (np.array(entry["R_world_to_camera"]),
                                      np.array(entry["t_world_to_camera"]))
                    rec.registered_order.append(idx)
            mapper._triangulate_all()
            logger.info("Reused %d poses from %s; re-triangulated %d points",
                        rec.num_registered, cams_file.name, len(rec.points3d))
        elif args.multi_model:
            # Reconstruct every group the capture contains, then align them.
            # A group joined to the main model by a single verified pair cannot
            # be resected at all - resection needs 3D points, and a track only
            # becomes a point once two registered cameras see it - so growing
            # one model can never reach it however the thresholds are set.
            from src.sfm_core.multimodel import (
                finalise_merge, merge_models, reconstruct_models)
            try:
                models = reconstruct_models(features, Ks_feat, mapper_cfg,
                                            groups=groups, match_cache=match_cache)
            except RuntimeError as exc:
                logger.error("Reconstruction failed: %s", exc)
                return 4
            logger.info("Reconstructed %d model(s): %s", len(models),
                        [m.rec.num_registered for m in models])
            primary = models[0]
            for other in models[1:]:
                if merge_models(primary, other, primary.mapper.matches,
                                seed=mapper_cfg.seed):
                    finalise_merge(primary)
            mapper, rec = primary.mapper, primary.rec
            summary["models"] = {
                "count": len(models),
                "sizes": [m.rec.num_registered for m in models],
                "merged": sorted(primary.merged_from),
            }
        else:
            try:
                rec = mapper.run()
            except RuntimeError as exc:
                logger.error("Reconstruction failed: %s", exc)
                return 4

        centers = rec.camera_centers()
        sparse_pts, sparse_cols = rec.points_array()
        errs = np.array(list(rec.errors.values())) if rec.errors else np.zeros(1)
        lengths = np.array(list(rec.track_lengths.values())) if rec.track_lengths else np.zeros(1)

        summary["sparse"] = {
            "registered_images": rec.num_registered,
            "registration_rate": round(100.0 * rec.num_registered / len(paths), 2),
            "num_points": len(sparse_pts),
            "mean_reprojection_error_px": round(float(errs.mean()), 5),
            "median_reprojection_error_px": round(float(np.median(errs)), 5),
            "mean_track_length": round(float(lengths.mean()), 4),
            "refined_focal_px": round(float(rec.K[0, 0]), 4),
            "initial_focal_px": round(float(K[0, 0]), 4),
            "feature_time_s": round(feature_time, 2),
            "matching_time_s": round(mapper.stats.get("matching_time", 0.0), 2),
            "mapping_time_s": round(mapper.stats.get("mapping_time", 0.0), 2),
            "verified_pairs": int(mapper.stats.get("verified_pairs", 0)),
            "num_tracks": int(mapper.stats.get("num_tracks", 0)),
        }

        write_ply_points(out_dir / "sparse.ply", sparse_pts, sparse_cols)
        first_of_group = {}
        for i in sorted(rec.poses):
            first_of_group.setdefault(groups[i], i)
        poses_json = {
            "camera_groups": {
                str(g): {"fx": float(rec.Ks[i][0, 0]), "fy": float(rec.Ks[i][1, 1]),
                         "cx": float(rec.Ks[i][0, 2]), "cy": float(rec.Ks[i][1, 2]),
                         "width": features[i].width, "height": features[i].height,
                         "kind": next(sp.kind for sp in inputs.sources if sp.group == g)}
                for g, i in sorted(first_of_group.items())
            },
            "note": "world-to-camera: x_cam = R @ x_world + t; centre C = -R^T t",
            "cameras": [
                {"image": rec.image_names[i],
                 "index": int(i),
                 "R_world_to_camera": rec.poses[i][0].tolist(),
                 "t_world_to_camera": rec.poses[i][1].tolist(),
                 "camera_center": centers[i].tolist(),
                 "camera_group": int(groups[i])}
                for i in sorted(rec.poses)
            ],
        }
        write_json(out_dir / "cameras.json", poses_json)

        # --- dense MVS ------------------------------------------------------
        dense_result = None
        if not args.no_dense:
            shared: Dict[Tuple[int, int], int] = {}
            for tid in rec.points3d:
                obs = [i for i in mapper.tracks[tid].observations if i in rec.poses]
                for a in range(len(obs)):
                    for b in range(a + 1, len(obs)):
                        key = (min(obs[a], obs[b]), max(obs[a], obs[b]))
                        shared[key] = shared.get(key, 0) + 1

            dense_cfg = DenseConfig(
                max_image_size=args.dense_size,
                num_depths=args.depths,
                num_source_views=args.source_views,
                min_ncc=args.min_ncc,
                min_consistent_views=args.consistent_views,
                use_library_warp=not args.no_library_warp,
            )
            # Feature-frame intrinsics rescaled to each image's own native
            # resolution, which is what the dense stage loads from disk.
            native = {sp.group: (sp.width, sp.height) for sp in inputs.sources}
            K_full = {}
            for i in rec.poses:
                Ki = rec.Ks[i].copy()
                nat_w = native.get(groups[i], (features[i].width, 0))[0]
                ratio = nat_w / max(features[i].width, 1)
                if abs(ratio - 1.0) > 1e-9:
                    Ki = Ki * ratio
                    Ki[2, 2] = 1.0
                K_full[i] = Ki
            dense_result = run_dense_stereo(paths, rec.poses, K_full, sparse_pts,
                                            shared, dense_cfg)
            if len(dense_result.points):
                write_ply_points(out_dir / "dense.ply", dense_result.points,
                                 dense_result.colors, dense_result.normals)
            summary["dense"] = {
                "num_points": int(len(dense_result.points)),
                **{k: (round(float(v), 4) if isinstance(v, float) else v)
                   for k, v in dense_result.stats.items()},
                "image_size": args.dense_size,
                "num_depths": args.depths,
            }

        # --- surface --------------------------------------------------------
        surface = None
        if dense_result is not None and not args.no_surface and len(dense_result.points):
            surface = extract_surface(dense_result.depth_maps, dense_result.points,
                                      TSDFConfig(resolution=args.tsdf_resolution))
            if len(surface.faces):
                write_ply_mesh(out_dir / "surface.ply", surface.vertices,
                               surface.faces, surface.colors)
            summary["surface"] = {k: (round(float(v), 5) if isinstance(v, float) else v)
                                  for k, v in surface.stats.items()}

    summary["total_time_s"] = round(total_sw.elapsed, 2)
    write_json(out_dir / "summary.json", summary)

    # --- viewer -------------------------------------------------------------
    viewer_path = None
    if not args.no_viewer:
        from src.sfm_core.viewer import build_viewer
        viewer_path = build_viewer(
            out_dir / f"{cfg.scene.name}_reconstruction.html",
            title=cfg.scene.description.strip().split("\n")[0][:60] or cfg.scene.name,
            sparse_points=sparse_pts, sparse_colors=sparse_cols,
            dense_points=dense_result.points if dense_result else None,
            dense_colors=dense_result.colors if dense_result else None,
            mesh_vertices=surface.vertices if surface else None,
            mesh_faces=surface.faces if surface else None,
            mesh_colors=surface.colors if surface else None,
            camera_centers=np.array([centers[i] for i in sorted(centers)]),
            camera_rotations=np.array([rec.poses[i][0] for i in sorted(rec.poses)]),
            intrinsics=rec.K, image_size=(w0, h0),
        )

    # --- report -------------------------------------------------------------
    s = summary["sparse"]
    print()
    print("=" * 70)
    print(f"FROM-SCRATCH RECONSTRUCTION COMPLETE - {cfg.scene.name}")
    print("=" * 70)
    print("  no COLMAP, no SfM library: all geometry implemented in src/sfm_core")
    print()
    if len(inputs.sources) > 1:
        print("  input sources:")
        for spec in inputs.summary():
            print(f"    group {spec['group']}: {spec['frames']:>4} {spec['kind']:<7} "
                  f"{spec['resolution'][0]}x{spec['resolution'][1]}"
                  + (f"  ({spec['notes']})" if spec['notes'] else ""))
        print()
    print(f"  registered images   : {s['registered_images']} / {len(paths)} "
          f"({s['registration_rate']:.1f} %)")
    print(f"  verified pairs      : {s['verified_pairs']}")
    print(f"  tracks built        : {s['num_tracks']:,}")
    print(f"  sparse points       : {s['num_points']:,}")
    print(f"  mean reproj. error  : {s['mean_reprojection_error_px']:.4f} px")
    print(f"  mean track length   : {s['mean_track_length']:.2f}")
    print(f"  focal: {s['initial_focal_px']:.1f} px (prior) -> "
          f"{s['refined_focal_px']:.1f} px (bundle-adjusted)")
    print(f"  timings             : features {s['feature_time_s']:.0f}s, "
          f"matching {s['matching_time_s']:.0f}s, mapping {s['mapping_time_s']:.0f}s")
    if "dense" in summary:
        d = summary["dense"]
        print()
        print(f"  dense points        : {d['num_points']:,}")
        print(f"  MVS views           : {d.get('views', 0)} at {d['image_size']}px, "
              f"{d['num_depths']} depth planes")
        print(f"  dense time          : {d.get('total_time', 0):.0f} s")
    if "surface" in summary and summary["surface"].get("faces"):
        m = summary["surface"]
        print()
        print(f"  mesh                : {m['vertices']:,} vertices, {m['faces']:,} faces")
        print(f"  TSDF                : {m['resolution']}^3 grid, voxel {m['voxel_size']:.5f}")
    print()
    print(f"  outputs             : {out_dir}")
    for name in ("sparse.ply", "dense.ply", "surface.ply", "cameras.json", "summary.json"):
        f = out_dir / name
        if f.is_file():
            print(f"    {name:<16} {human_size(f.stat().st_size)}")
    if viewer_path:
        print(f"    {viewer_path.name:<16} {human_size(viewer_path.stat().st_size)}  <- open this")
    print(f"  total time          : {total_sw.elapsed:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
