"""Controlled ablations of the from-scratch mapper's own settings.

Several values in :class:`MapperConfig` were tuned while bundle adjustment was
failing to converge, against the objective "register more images". They were
partly compensating for that fault rather than for anything intrinsic to the
data, so they need re-deriving now that it is fixed. This script produces the
table to do that from.

Two experiment groups, split by cost:

``filters``
    The structure-quality filters are applied *after* the model is final, so
    every threshold can be evaluated on one reconstruction. One mapping run
    answers the whole sweep.

``growth``
    Settings that change how the model is built - triangulation angle, the PnP
    acceptance floor, the reprojection limit - need one reconstruction each.

Features and the verified match graph come from the usual caches and are
loaded once, so conditions differ only in the setting under test.

Examples
--------
    python scripts/run_scratch_ablation.py --scene scene_01 --focal 1150 --experiment filters
    python scripts/run_scratch_ablation.py --scene scene_01 --focal 1150 --experiment growth
"""

from __future__ import annotations

import argparse
import copy
import csv
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.camera import estimate_intrinsics  # noqa: E402
from src.config import ConfigError, load_config  # noqa: E402
from src.sfm_core import IncrementalMapper, MapperConfig, extract_features  # noqa: E402
from src.sfm_core.diagnostics import covisibility_components  # noqa: E402
from src.sfm_core.sources import collect_inputs  # noqa: E402
from src.utils import ensure_dir, setup_logging, timestamp, write_json  # noqa: E402


# ===========================================================================
# Metrics
# ===========================================================================


def cloud_metrics(rec, parallax: Optional[Dict[int, float]] = None) -> Dict[str, float]:
    """Scale-free descriptors of a sparse cloud.

    The bounding box is reported *relative to the camera path*, because the
    reconstruction has no metric scale - an absolute extent means nothing on
    its own, while the ratio says how far the structure sprawls compared with
    the distance actually walked.
    """
    out: Dict[str, float] = {}
    if not rec.points3d or len(rec.poses) < 2:
        return out
    pts = np.array(list(rec.points3d.values()))
    centres = np.array(list(rec.camera_centers().values()))

    def diag(a: np.ndarray) -> float:
        return float(np.linalg.norm(a.max(0) - a.min(0)))

    extent = diag(centres)
    out["camera_path_extent"] = round(extent, 4)
    out["bbox_diagonal"] = round(diag(pts), 4)
    out["bbox_ratio"] = round(diag(pts) / extent, 4) if extent > 1e-9 else 0.0

    d = np.linalg.norm(pts[:, None, :] - centres[None, :, :], axis=2).min(axis=1)
    if extent > 1e-9:
        out["dist_median_x"] = round(float(np.median(d)) / extent, 4)
        out["dist_p99_x"] = round(float(np.percentile(d, 99)) / extent, 4)
        out["dist_max_x"] = round(float(d.max()) / extent, 4)
    if parallax:
        vals = np.array(list(parallax.values()))
        out["parallax_median_deg"] = round(float(np.median(vals)), 4)
        out["parallax_p10_deg"] = round(float(np.percentile(vals, 10)), 4)
    return out


def model_metrics(mapper, rec) -> Dict[str, object]:
    errs = np.array(list(rec.errors.values())) if rec.errors else np.zeros(1)
    lens = np.array(list(rec.track_lengths.values())) if rec.track_lengths else np.zeros(1)
    comps = covisibility_components(rec.points3d, mapper.tracks, rec.poses,
                                    mapper.cfg.min_covisible_points)
    row: Dict[str, object] = {
        "registered": len(rec.poses),
        "num_points": len(rec.points3d),
        "mean_reprojection_px": round(float(errs.mean()), 5),
        "median_reprojection_px": round(float(np.median(errs)), 5),
        "mean_track_length": round(float(lens.mean()), 4),
        "num_components": len(comps),
        "largest_component": len(comps[0]) if comps else 0,
    }
    row.update(cloud_metrics(rec, mapper.point_parallax()))
    return row



# ===========================================================================
# Image subsets and match-graph slicing (for the brief's experiments)
# ===========================================================================


def select_subset(count: int, total: int) -> List[int]:
    """``count`` image indices spread evenly across the whole capture.

    A fixed stride, not random sampling, so the result is deterministic, and
    starting at 0 so the subset always spans the full capture. Taking the
    *first* ``count`` images instead would shrink the scene's spatial extent as
    well as its image count, confounding the experiment.
    """
    if count <= 0 or count >= total:
        return list(range(total))
    step = total / count
    idx = sorted({int(i * step) for i in range(count)})
    if len(idx) < count:
        spare = [i for i in range(total) if i not in set(idx)]
        idx = sorted(idx + spare[: count - len(idx)])
    return idx[:count]


def quality_subset(cfg, arm: str, names: Sequence[str]) -> List[int]:
    """Indices surviving a blur-based quality arm.

    ``all`` keeps everything; ``blur_pNN`` drops the blurriest NN percent by
    the stage-1 Laplacian score.
    """
    if arm == "all":
        return list(range(len(names)))
    import re
    import pandas as pd
    csv_path = cfg.metrics_dir / "image_quality.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"{csv_path} not found; the quality arms need stage 1: "
            f"python scripts/run_preprocess.py --scene {cfg.scene.name}")
    df = pd.read_csv(csv_path)
    match = re.fullmatch(r"blur_p(\d+)", arm)
    if not match:
        raise ValueError(f"unknown quality arm {arm!r}")
    pos = {n: i for i, n in enumerate(names)}
    df = df[df["image_name"].isin(pos)]
    threshold = float(np.percentile(df["blur_score"], float(match.group(1))))
    kept = df[df["blur_score"] >= threshold]["image_name"]
    return sorted(pos[n] for n in kept if n in pos)


def subset_matches(matches, keep: Sequence[int]):
    """Filter the verified match graph to ``keep`` and reindex to subset-local ids.

    Reusing the full graph rather than rematching each condition is both far
    cheaper and methodologically better: the matching stage becomes an exact
    constant across conditions, so any difference in the result is attributable
    to the variable under test rather than to RANSAC noise in matching.
    """
    from dataclasses import replace as _replace
    remap = {g: l for l, g in enumerate(keep)}
    out = []
    for m in matches:
        if m.i in remap and m.j in remap:
            out.append(_replace(m, i=remap[m.i], j=remap[m.j]))
    return out

# ===========================================================================
# Conditions
# ===========================================================================


def growth_conditions(base: MapperConfig) -> List[Dict[str, object]]:
    """One reconstruction each: these settings change how the model is built."""
    out: List[Dict[str, object]] = []
    for v in (0.5, 1.0, 1.5, 2.0):
        out.append({"name": f"tri_angle_{v}", "knob": "min_triangulation_angle_deg",
                    "value": v, "cfg": replace(base, min_triangulation_angle_deg=v)})
    for v in (8, 12, 16, 20):
        out.append({"name": f"pnp_min_inliers_{v}", "knob": "pnp_min_inliers",
                    "value": v, "cfg": replace(base, pnp_min_inliers=v)})
    for v in (8.0, 6.0, 4.0, 2.0):
        out.append({"name": f"max_reproj_{v}", "knob": "max_reprojection_px",
                    "value": v, "cfg": replace(base, max_reprojection_px=v)})
    return out


FILTER_CONDITIONS: List[Dict[str, object]] = (
    [{"name": "none", "parallax": 0.0, "distance": 0.0}]
    + [{"name": f"parallax_{v}", "parallax": v, "distance": 0.0}
       for v in (0.75, 1.0, 1.5, 2.0, 3.0)]
    + [{"name": f"distance_{v}", "parallax": 0.0, "distance": v}
       for v in (1.0, 1.5, 2.0, 3.0, 5.0)]
    + [{"name": f"both_{p}_{d}", "parallax": p, "distance": d}
       for p, d in ((1.0, 3.0), (1.5, 2.0), (2.0, 2.0))]
)


# ===========================================================================
# Runner
# ===========================================================================


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ablate the from-scratch mapper's settings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=Path, action="append", default=None, metavar="YAML")
    p.add_argument("--scene", type=str, default=None)
    p.add_argument("--experiment", required=True,
                   choices=["filters", "growth", "calibration", "brief", "detector", "all"])
    p.add_argument("--brief-part", default="all",
                   choices=["image_count", "matcher", "quality", "all"],
                   help="Which of the brief's three experiments to run.")
    p.add_argument("--image-counts", nargs="+", type=int, default=None,
                   metavar="N", help="Subset sizes; 0 means all images.")
    p.add_argument("--quality-arms", nargs="+", default=None, metavar="ARM",
                   help="e.g. all blur_p15 blur_p30")
    p.add_argument("--sequential-arm", type=int, default=10, metavar="W",
                   help="Window for the sequential matcher arm.")
    p.add_argument("--calibration-part", default="all",
                   choices=["focal", "distortion", "all"],
                   help="Which half of the calibration experiment to run. "
                        "'distortion' conditions cost roughly 13x a plain one, "
                        "so they are separable from the focal sweep.")
    p.add_argument("--knob", type=str, default=None, metavar="FIELD",
                   help="Sweep one MapperConfig field instead of the built-in growth "
                        "set, e.g. --knob min_triangulation_angle_deg. Incremental SfM "
                        "is path-dependent, so a single value can look good by luck; "
                        "this exists to check a promising one against its neighbours.")
    p.add_argument("--values", nargs="+", type=float, default=None, metavar="V",
                   help="Values for --knob.")
    p.add_argument("--focal", type=float, default=0.0)
    p.add_argument("--feature", default="sift", choices=["sift", "akaze", "orb"],
                   help="Detector for the non-detector experiments.")
    p.add_argument("--detectors", nargs="+", default=None, metavar="M",
                   help="Detectors to compare in --experiment detector.")
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="FIELD=VALUE",
                   help="Override a MapperConfig field for every condition, e.g. "
                        "--set seed_min_parallax_deg=1.5. Repeatable. Use this to "
                        "give a detector thresholds suited to it instead of "
                        "judging it under another detector's settings.")
    p.add_argument("--feature-size", type=int, default=1600)
    p.add_argument("--max-features", type=int, default=8000)
    p.add_argument("--sequential-window", type=int, default=0)
    p.add_argument("--log-level", default="WARNING",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def make_mapper(features, Ks_feat, groups, cfg, match_cache) -> IncrementalMapper:
    return IncrementalMapper(features, [K.copy() for K in Ks_feat], cfg,
                             groups=groups, match_cache=match_cache)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    overrides = [f"scene.name={args.scene}"] if args.scene else []
    try:
        cfg = load_config(args.config, overrides=overrides)
    except ConfigError as exc:
        print(f"CONFIGURATION ERROR\n{exc}", file=sys.stderr)
        return 2

    run_id = timestamp()
    setup_logging(ensure_dir(cfg.logs_dir) / f"ablation_{run_id}.log", level=args.log_level)

    images_dir = cfg.processed_images_dir
    if not images_dir.is_dir():
        images_dir = cfg.raw_images_dir
    inputs = collect_inputs(image_dirs=[images_dir], videos=[],
                            frame_dir=cfg.raw_images_dir.parent / "video_frames")
    paths, groups = inputs.paths, inputs.groups

    features, _ = extract_features(
        paths, max_features=args.max_features, max_image_size=args.feature_size,
        cache_path=ensure_dir(cfg.cache_dir) / f"scratch_feats_{args.feature}_{args.feature_size}.pkl",
        method=args.feature)

    group_intrinsics = {}
    for spec in inputs.sources:
        idx = groups.index(spec.group)
        group_intrinsics[spec.group] = estimate_intrinsics(
            features[idx].width, features[idx].height,
            default_focal_factor=cfg.camera.default_focal_factor)
        if args.focal > 0:
            group_intrinsics[spec.group].fx = group_intrinsics[spec.group].fy = args.focal
    Ks_feat = [group_intrinsics[g].K.copy() for g in groups]

    base = MapperConfig(
        sequential_window=args.sequential_window, seed=cfg.runtime.seed,
        refine_focal=False, refine_distortion=False, auto_focal=(args.focal <= 0),
    )
    for override in args.overrides:
        field, _, raw = override.partition("=")
        field = field.strip()
        if not hasattr(base, field):
            print(f"MapperConfig has no field {field!r}", file=sys.stderr)
            return 2
        current = getattr(base, field)
        value = (raw.strip().lower() in {"1", "true", "yes"}) if isinstance(current, bool)             else type(current)(raw.strip())
        base = replace(base, **{field: value})
        print(f"  override: {field} = {value}")

    match_cache = (ensure_dir(cfg.cache_dir) / f"scratch_matches_{args.feature}_{args.feature_size}_"
                   f"{args.max_features}_{args.sequential_window}.pkl")

    out_dir = ensure_dir(cfg.scene_output_dir / "ablations")
    rows: List[Dict[str, object]] = []
    csv_path = out_dir / f"ablation_{args.experiment}_{run_id}.csv"

    def flush_rows() -> None:
        """Persist after every condition.

        A sweep is hours long and each condition is expensive; writing only at
        the end means one crash discards everything already paid for. This ran
        as written once and lost seven completed reconstructions to a failure
        in the eighth.
        """
        if not rows:
            return
        columns: List[str] = []
        for r in rows:
            for k in r:
                if k not in columns:
                    columns.append(k)
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=columns)
            w.writeheader()
            w.writerows(rows)
        write_json(csv_path.with_suffix(".json"),
                   {"scene": cfg.scene.name, "run_id": run_id,
                    "experiment": args.experiment, "rows": rows})

    def run_filters() -> None:
        """One reconstruction; every filter threshold evaluated on it."""
        print("\n=== filters: reconstructing once, then sweeping thresholds ===")
        cfg_f = replace(base, final_min_triangulation_angle_deg=0.0,
                        max_point_distance_factor=0.0,
                        # Culling would confound the comparison by changing the
                        # camera set between thresholds.
                        enforce_single_component=False)
        mapper = make_mapper(features, Ks_feat, groups, cfg_f, match_cache)
        mapper.match_all()
        mapper.build_tracks()
        t0 = time.perf_counter()
        rec = mapper.run()
        print(f"    base model: {len(rec.poses)} cameras, {len(rec.points3d)} points, "
              f"{time.perf_counter() - t0:.0f} s")

        parallax = mapper.point_parallax()
        centres = np.array(list(rec.camera_centers().values()))
        extent = float(np.linalg.norm(centres.max(0) - centres.min(0)))
        tids = list(rec.points3d)
        pts = np.array([rec.points3d[t] for t in tids])
        dist = np.linalg.norm(pts[:, None, :] - centres[None, :, :], axis=2).min(axis=1)
        ang = np.array([parallax[t] for t in tids])
        err = np.array([rec.errors.get(t, 0.0) for t in tids])

        def diag(a: np.ndarray) -> float:
            return float(np.linalg.norm(a.max(0) - a.min(0)))

        for cond in FILTER_CONDITIONS:
            keep = np.ones(len(tids), bool)
            if cond["parallax"] > 0:
                keep &= ang >= cond["parallax"]
            if cond["distance"] > 0:
                keep &= dist <= cond["distance"] * extent
            if keep.sum() < 10:
                continue
            sub = pts[keep]
            rows.append({
                "experiment": "filters", "condition": cond["name"],
                "parallax_deg": cond["parallax"], "distance_factor": cond["distance"],
                "registered": len(rec.poses),
                "num_points": int(keep.sum()),
                "points_kept_pct": round(100.0 * keep.mean(), 2),
                "bbox_ratio": round(diag(sub) / extent, 4),
                "mean_reprojection_px": round(float(err[keep].mean()), 5),
                "median_reprojection_px": round(float(np.median(err[keep])), 5),
                "parallax_median_deg": round(float(np.median(ang[keep])), 4),
                "dist_p99_x": round(float(np.percentile(dist[keep], 99)) / extent, 4),
                "dist_max_x": round(float(dist[keep].max()) / extent, 4),
            })
            flush_rows()

    def run_growth() -> None:
        if args.knob:
            if not args.values:
                print("--knob needs --values", file=sys.stderr)
                return
            if not hasattr(base, args.knob):
                print(f"MapperConfig has no field '{args.knob}'", file=sys.stderr)
                return
            current = getattr(base, args.knob)
            conditions = [
                {"name": f"{args.knob}_{v}", "knob": args.knob, "value": v,
                 "cfg": replace(base, **{args.knob: type(current)(v)})}
                for v in args.values
            ]
        else:
            conditions = growth_conditions(base)

        print("\n=== growth: one reconstruction per condition ===")
        for i, cond in enumerate(conditions, 1):
            name = cond["name"]
            print(f"  [{i}] {name} ...", end="", flush=True)
            t0 = time.perf_counter()
            try:
                mapper = make_mapper(features, Ks_feat, groups, cond["cfg"], match_cache)
                mapper.match_all()
                mapper.build_tracks()
                rec = mapper.run()
                row: Dict[str, object] = {
                    "experiment": "growth", "condition": name,
                    "knob": cond["knob"], "value": cond["value"],
                    "time_s": round(time.perf_counter() - t0, 1),
                }
                row.update(model_metrics(mapper, rec))
                rows.append(row)
                flush_rows()
                # flush: redirected stdout is block-buffered, so without this a
                # long sweep looks stalled from outside until it finishes.
                print(f" {row['registered']} cams, {row['num_points']} pts, "
                      f"{row['mean_reprojection_px']:.3f} px, "
                      f"bbox {row.get('bbox_ratio', 0)}x, {row['time_s']:.0f} s",
                      flush=True)
            except Exception as exc:  # a failure is a result
                print(f" FAILED: {type(exc).__name__}: {exc}")
                traceback.print_exc(limit=2)
                rows.append({"experiment": "growth", "condition": name,
                             "knob": cond["knob"], "value": cond["value"],
                             "failure": f"{type(exc).__name__}: {exc}"})
                flush_rows()

    def run_calibration() -> None:
        """Distortion, focal refinement, and whether self-calibration works.

        All three were settled while bundle adjustment was not converging, and
        the focal conclusion in particular ("self-calibration does not work on
        this data") rests on reprojection error having been flat across a wide
        focal range. That flatness is exactly what a non-converging adjustment
        would produce regardless of the truth, so it has to be re-measured
        rather than inherited.

        The fixed-focal sweep is the decisive part: with structure re-solved
        for each candidate, a focal that is wrong should now cost reprojection
        error, because no single projective deformation satisfies three or more
        views sharing one calibration.
        """
        long_side = max(features[0].width, features[0].height)
        conditions: List[Dict[str, object]] = []

        # The fixed-focal sweep runs FIRST. It answers the question that
        # matters, and enabling distortion refinement costs roughly 13x the
        # runtime of a plain condition - not in the solver, but because every
        # triangulation then has to undistort its observations, on every one of
        # ~40k tracks, after every registration. Ordering the cheap decisive
        # conditions ahead of the expensive ones means a killed or interrupted
        # sweep still answers something.
        if args.calibration_part in ("focal", "all"):
            for f in (700.0, 900.0, 1150.0, 1450.0, 1800.0, 2300.0):
                conditions.append({
                    "name": f"fixed_focal_{int(f)}",
                    "focal": f, "auto": False,
                    "cfg": replace(base, refine_distortion=False,
                                   refine_focal=False, auto_focal=False),
                })
            conditions.append({
                "name": "refine_focal_on",
                "focal": args.focal or 1150.0, "auto": False,
                "cfg": replace(base, refine_distortion=False, refine_focal=True,
                               auto_focal=False),
            })
            conditions.append({
                "name": "self_calibration",
                "focal": 0.0, "auto": True,
                "cfg": replace(base, refine_distortion=False, refine_focal=False,
                               auto_focal=True),
            })
        if args.calibration_part in ("distortion", "all"):
            for rf in (False, True):
                conditions.append({
                    "name": f"distortion_on__refine_focal_{rf}",
                    "focal": args.focal or 1150.0, "auto": False,
                    "cfg": replace(base, refine_distortion=True, refine_focal=rf,
                                   auto_focal=False),
                })

        print("\n=== calibration: one reconstruction per condition ===")
        for i, cond in enumerate(conditions, 1):
            print(f"  [{i}/{len(conditions)}] {cond['name']} ...", end="", flush=True)
            t0 = time.perf_counter()
            try:
                if cond["auto"]:
                    # Start from the uncalibrated prior, not from --focal:
                    # seeding self-calibration with the answer would not test
                    # anything.
                    prior = {}
                    for spec in inputs.sources:
                        idx = groups.index(spec.group)
                        prior[spec.group] = estimate_intrinsics(
                            features[idx].width, features[idx].height,
                            default_focal_factor=cfg.camera.default_focal_factor)
                    Ks = [prior[g].K.copy() for g in groups]
                else:
                    Ks = []
                    for g in groups:
                        K = group_intrinsics[g].K.copy()
                        K[0, 0] = K[1, 1] = float(cond["focal"])
                        Ks.append(K)
                mapper = make_mapper(features, Ks, groups, cond["cfg"], match_cache)
                mapper.match_all()
                mapper.build_tracks()
                rec = mapper.run()

                focals = [0.5 * (K[0, 0] + K[1, 1]) for K in rec.Ks.values()]
                dist = list(rec.distortion.values())
                row: Dict[str, object] = {
                    "experiment": "calibration", "condition": cond["name"],
                    "focal_in": cond["focal"], "auto_focal": cond["auto"],
                    "refine_distortion": cond["cfg"].refine_distortion,
                    "refine_focal": cond["cfg"].refine_focal,
                    "focal_out": round(float(np.mean(focals)), 2) if focals else 0.0,
                    "focal_out_long_side_ratio": (
                        round(float(np.mean(focals)) / long_side, 4) if focals else 0.0),
                    "k1": round(float(dist[0][0]), 6) if dist else 0.0,
                    "k2": round(float(dist[0][1]), 6) if dist else 0.0,
                    "time_s": round(time.perf_counter() - t0, 1),
                }
                row.update(model_metrics(mapper, rec))
                rows.append(row); flush_rows()
                print(f" {row['registered']} cams, {row['num_points']} pts, "
                      f"{row['mean_reprojection_px']:.3f} px mean / "
                      f"{row['median_reprojection_px']:.3f} med, "
                      f"f={row['focal_out']}, {row['time_s']:.0f} s", flush=True)
            except Exception as exc:
                print(f" FAILED: {type(exc).__name__}: {exc}", flush=True)
                traceback.print_exc(limit=2)
                rows.append({"experiment": "calibration", "condition": cond["name"],
                             "failure": f"{type(exc).__name__}: {exc}"})
                flush_rows()

    def run_brief() -> None:
        """The brief's three experiments: image_count, matcher, quality.

        image_count and quality slice the *already computed* match graph, so
        matching is an exact constant across their conditions and every
        difference is attributable to the variable under test. The matcher
        experiment cannot do that - comparing exhaustive against sequential is
        partly a comparison of matching cost - so both of its arms match from
        scratch and both times are measured.
        """
        names = [f.name for f in features]
        total = len(features)
        full_matches = None

        def build(idx, cfg_used, matches_in, label, extra):
            sub_feats = [features[i] for i in idx]
            sub_Ks = [Ks_feat[i].copy() for i in idx]
            sub_groups = [groups[i] for i in idx]
            mp = IncrementalMapper(sub_feats, sub_Ks, cfg_used,
                                   groups=sub_groups, match_cache=None)
            if matches_in is not None:
                mp.matches = matches_in
                mp.build_tracks()
            t0 = time.perf_counter()
            rec = mp.run()
            row = {"experiment": label, **extra,
                   "images_used": len(idx),
                   "verified_pairs": len(mp.matches),
                   "matching_time_s": round(float(mp.stats.get("matching_time", 0.0)), 2),
                   "time_s": round(time.perf_counter() - t0, 1)}
            row.update(model_metrics(mp, rec))
            rows.append(row); flush_rows()
            print(f"    {len(idx):>4} img -> {row['registered']:>4} cams, "
                  f"{row['num_points']:>6} pts, {row['mean_reprojection_px']:.3f} px, "
                  f"{row['time_s']:.0f} s", flush=True)

        if args.brief_part in ("image_count", "all"):
            print("\n=== image_count ===")
            probe = IncrementalMapper(features, [K.copy() for K in Ks_feat], base,
                                      groups=groups, match_cache=match_cache)
            full_matches = probe.match_all()
            counts = args.image_counts or [25, 50, 75, 100, 0]
            for c in counts:
                idx = select_subset(c, total)
                print(f"  [{len(idx)} images]", end="", flush=True)
                try:
                    build(idx, base, subset_matches(full_matches, idx),
                          "image_count", {"condition": f"n{len(idx)}",
                                          "image_count": len(idx)})
                except Exception as exc:
                    print(f" FAILED: {type(exc).__name__}: {exc}", flush=True)
                    rows.append({"experiment": "image_count",
                                 "condition": f"n{len(idx)}",
                                 "failure": f"{type(exc).__name__}: {exc}"})
                    flush_rows()

        if args.brief_part in ("quality", "all"):
            print("\n=== quality ===")
            if full_matches is None:
                probe = IncrementalMapper(features, [K.copy() for K in Ks_feat], base,
                                          groups=groups, match_cache=match_cache)
                full_matches = probe.match_all()
            for arm in (args.quality_arms or ["all", "blur_p15", "blur_p30"]):
                try:
                    idx = quality_subset(cfg, arm, names)
                except (FileNotFoundError, ValueError) as exc:
                    print(f"  {arm}: SKIPPED ({exc})", flush=True)
                    continue
                print(f"  [{arm}: {len(idx)} images]", end="", flush=True)
                try:
                    build(idx, base, subset_matches(full_matches, idx),
                          "quality", {"condition": arm, "quality_arm": arm})
                except Exception as exc:
                    print(f" FAILED: {type(exc).__name__}: {exc}", flush=True)
                    rows.append({"experiment": "quality", "condition": arm,
                                 "failure": f"{type(exc).__name__}: {exc}"})
                    flush_rows()

        if args.brief_part in ("matcher", "all"):
            print("\n=== matcher (both arms match from scratch, times measured) ===")
            idx = list(range(total))
            for arm, window in (("exhaustive", 0), (f"sequential_w{args.sequential_arm}",
                                                    args.sequential_arm)):
                print(f"  [{arm}]", end="", flush=True)
                try:
                    build(idx, replace(base, sequential_window=window), None,
                          "matcher", {"condition": arm, "matcher": arm,
                                      "sequential_window": window})
                except Exception as exc:
                    print(f" FAILED: {type(exc).__name__}: {exc}", flush=True)
                    rows.append({"experiment": "matcher", "condition": arm,
                                 "failure": f"{type(exc).__name__}: {exc}"})
                    flush_rows()

    def run_detectors() -> None:
        """Compare local feature detectors end to end.

        Each detector gets its own feature extraction, its own match graph and
        its own reconstruction, because a detector changes what is *found* and
        therefore everything downstream. Nothing else varies.

        Keypoint count is not a quality ranking. ORB routinely finds several
        times more keypoints than SIFT and still registers fewer cameras: what
        matters is how many survive geometric verification and how long the
        resulting tracks are. Judge on registered cameras, track length and
        reprojection error together.
        """
        from src.sfm_core.features import extract_features as _extract

        methods = args.detectors or ["sift", "akaze", "orb"]
        print("\n=== detector comparison ===")
        for method in methods:
            print(f"  [{method}] extracting ...", end="", flush=True)
            t0 = time.perf_counter()
            try:
                feats, _ = _extract(
                    paths, max_features=args.max_features,
                    max_image_size=args.feature_size,
                    cache_path=ensure_dir(cfg.cache_dir)
                    / f"scratch_feats_{method}_{args.feature_size}.pkl",
                    method=method)
                kp = float(np.mean([len(f) for f in feats]))
                t_extract = time.perf_counter() - t0

                cache = (ensure_dir(cfg.cache_dir)
                         / f"scratch_matches_{method}_{args.feature_size}_"
                           f"{args.max_features}_{args.sequential_window}.pkl")
                mp = IncrementalMapper(feats, [K.copy() for K in Ks_feat], base,
                                       groups=groups, match_cache=cache)
                t1 = time.perf_counter()
                mp.match_all()
                mp.build_tracks()
                t_match = time.perf_counter() - t1
                t2 = time.perf_counter()
                rec = mp.run()

                row = {"experiment": "detector", "condition": method,
                       "detector": method,
                       "descriptor": str(feats[0].descriptors.dtype),
                       "binary": bool(feats[0].is_binary),
                       "mean_keypoints": round(kp, 1),
                       "verified_pairs": len(mp.matches),
                       "num_tracks": len(mp.tracks),
                       "extract_s": round(t_extract, 1),
                       "match_s": round(t_match, 1),
                       "map_s": round(time.perf_counter() - t2, 1)}
                row.update(model_metrics(mp, rec))
                rows.append(row); flush_rows()
                print(f" {row['registered']:>4} cams, {row['num_points']:>6} pts, "
                      f"{row['mean_reprojection_px']:.3f} px, track "
                      f"{row['mean_track_length']:.2f}, {row['map_s']:.0f} s",
                      flush=True)
            except Exception as exc:
                print(f" FAILED: {type(exc).__name__}: {exc}", flush=True)
                rows.append({"experiment": "detector", "condition": method,
                             "failure": f"{type(exc).__name__}: {exc}"})
                flush_rows()

    if args.experiment in ("filters", "all"):
        run_filters()
    if args.experiment in ("growth", "all"):
        run_growth()
    if args.experiment in ("calibration", "all"):
        run_calibration()
    if args.experiment in ("brief", "all"):
        run_brief()
    if args.experiment in ("detector", "all"):
        run_detectors()

    if not rows:
        print("no results")
        return 1

    flush_rows()

    print(f"\n  results: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
