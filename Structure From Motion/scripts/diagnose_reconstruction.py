"""Measure why a reconstruction registers fewer images than the capture holds.

Answers three questions that registration counts alone cannot:

1. How many images *could* be registered? The largest connected component of
   the verified match graph is a hard ceiling - an image outside it cannot be
   reached by growing a model, however good the mapper is. Comparing the
   ceiling against the count actually registered separates a data limit from
   an algorithmic one.

2. When the model splits into covisibility components, is that a genuine
   split or an artefact of the culling heuristic? Each verified image pair
   spanning two components is re-solved from its own correspondences and
   compared against the relative pose the model implies.

3. Where did the capture physically break? Consecutive frames with no
   verified pair mark the places the photographer moved without shooting.

Runs the mapper with component culling disabled, so the *unculled* model is
what gets examined. Features and matches come from the usual caches, so this
costs a mapping pass and nothing more.

Examples
--------
    python scripts/diagnose_reconstruction.py --scene scene_01 --focal 1150
    python scripts/diagnose_reconstruction.py --scene scene_01 --focal 1150 --graph-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.camera import estimate_intrinsics  # noqa: E402
from src.config import ConfigError, load_config  # noqa: E402
from src.sfm_core import IncrementalMapper, MapperConfig, extract_features  # noqa: E402
from src.sfm_core.diagnostics import (  # noqa: E402
    bridge_evidence, format_report, match_graph_components, model_report,
)
from src.sfm_core.sources import collect_inputs  # noqa: E402
from src.utils import ensure_dir, setup_logging, timestamp, write_json  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Diagnose registration shortfall and reconstruction splits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=Path, action="append", default=None, metavar="YAML")
    p.add_argument("--scene", type=str, default=None)
    p.add_argument("--focal", type=float, default=0.0,
                   help="Fix the focal in pixels, as in run_scratch_sfm.py.")
    p.add_argument("--feature-size", type=int, default=1600)
    p.add_argument("--max-features", type=int, default=8000)
    p.add_argument("--sequential-window", type=int, default=0)
    p.add_argument("--min-pair-inliers", type=int, default=0,
                   help="Inlier floor used when computing the match-graph ceiling. "
                        "0 takes it from MapperConfig, which is what the mapper "
                        "actually accepts - hard-coding a different value here "
                        "reports a ceiling the pipeline is not working against.")
    p.add_argument("--graph-only", action="store_true",
                   help="Report match-graph connectivity without running the mapper. "
                        "Seconds instead of minutes; answers question 1 and 3 only.")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


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

    run_id = timestamp()
    logger = setup_logging(ensure_dir(cfg.logs_dir) / f"diagnose_{run_id}.log",
                           level=args.log_level)

    images_dir = cfg.processed_images_dir
    if not images_dir.is_dir():
        images_dir = cfg.raw_images_dir
    inputs = collect_inputs(image_dirs=[images_dir], videos=[],
                            frame_dir=cfg.raw_images_dir.parent / "video_frames")
    paths, groups = inputs.paths, inputs.groups
    if len(paths) < 3:
        print(f"Need at least 3 images, found {len(paths)}", file=sys.stderr)
        return 3

    logger.info("Diagnosing %s: %d images", cfg.scene.name, len(paths))

    features, _ = extract_features(
        paths, max_features=args.max_features, max_image_size=args.feature_size,
        cache_path=ensure_dir(cfg.cache_dir) / f"scratch_feats_{args.feature_size}.pkl",
    )

    group_intrinsics = {}
    for spec in inputs.sources:
        idx = groups.index(spec.group)
        gw, gh = features[idx].width, features[idx].height
        group_intrinsics[spec.group] = estimate_intrinsics(
            gw, gh, default_focal_factor=cfg.camera.default_focal_factor)
        if args.focal > 0:
            group_intrinsics[spec.group].fx = group_intrinsics[spec.group].fy = args.focal
    Ks_feat = [group_intrinsics[g].K.copy() for g in groups]

    mapper_cfg = MapperConfig(
        sequential_window=args.sequential_window,
        seed=cfg.runtime.seed,
        refine_focal=False,
        refine_distortion=False,
        auto_focal=(args.focal <= 0),
        # The whole point is to inspect the model *before* culling.
        enforce_single_component=False,
    )
    # The ceiling must be computed against the floor the mapper actually
    # applies. With the ratio route enabled that is the lower floor, not
    # min_pair_inliers.
    min_pair_inliers = args.min_pair_inliers or (
        mapper_cfg.min_pair_floor_inliers if mapper_cfg.min_pair_inlier_ratio > 0
        else mapper_cfg.min_pair_inliers)

    match_cache = (ensure_dir(cfg.cache_dir) / f"scratch_matches_{args.feature_size}_"
                   f"{args.max_features}_{args.sequential_window}.pkl")
    mapper = IncrementalMapper(features, Ks_feat, mapper_cfg, groups=groups,
                               match_cache=match_cache)

    out_dir = ensure_dir(cfg.scene_output_dir / "diagnostics")

    if args.graph_only:
        matches = mapper.match_all()
        components = match_graph_components(matches, len(features), min_pair_inliers)
        print()
        print("=" * 70)
        print("MATCH-GRAPH CONNECTIVITY (no reconstruction run)")
        print("=" * 70)
        print(f"  images                  : {len(features)}")
        print(f"  verified pairs          : {len(matches)}")
        print(f"  components (>={min_pair_inliers} inliers): "
              f"{len(components)}  sizes {[len(c) for c in components][:8]}")
        print(f"  registration ceiling    : {len(components[0]) if components else 0}")
        unreachable = sorted(set(range(len(features))) - set(components[0] if components else []))
        if unreachable:
            print(f"  unreachable images      : "
                  f"{', '.join(features[i].name for i in unreachable[:12])}"
                  f"{' ...' if len(unreachable) > 12 else ''}")
        print("=" * 70)
        return 0

    matches = mapper.match_all()
    mapper.build_tracks()
    rec = mapper.run()

    report = model_report(
        features, matches, mapper.tracks, rec.poses, rec.points3d, rec.Ks,
        min_covisible_points=mapper_cfg.min_covisible_points,
        min_pair_inliers=min_pair_inliers,
        seed=mapper_cfg.seed,
    )
    report["scene"] = cfg.scene.name
    report["run_id"] = run_id
    report["mean_reprojection_error_px"] = round(
        float(np.mean(list(rec.errors.values()))) if rec.errors else 0.0, 5)
    report["num_points"] = len(rec.points3d)
    report["image_names"] = [f.name for f in features]

    print()
    print(format_report(report))

    # Pairwise evidence for every adjacent pair of covisibility components,
    # not only the largest two - a three-way split needs three verdicts.
    covis_sizes = report["covisibility"]["sizes"]  # type: ignore[index]
    if len(covis_sizes) > 1:
        from src.sfm_core.diagnostics import covisibility_components
        comps = covisibility_components(
            rec.points3d, mapper.tracks, rec.poses, mapper_cfg.min_covisible_points)
        detail = []
        print()
        print("  PAIRWISE COMPONENT EVIDENCE")
        for a in range(min(len(comps), 4)):
            for b in range(a + 1, min(len(comps), 4)):
                if len(comps[a]) < 3 or len(comps[b]) < 3:
                    continue
                ev = bridge_evidence(features, matches, rec.poses, rec.Ks,
                                     comps[a], comps[b], seed=mapper_cfg.seed)
                print(f"   [{len(comps[a])} vs {len(comps[b])} cameras] "
                      f"{ev.verdict.upper()}: {ev.detail}")
                detail.append({
                    "size_a": len(comps[a]), "size_b": len(comps[b]),
                    "verdict": ev.verdict, "detail": ev.detail,
                    "num_bridges": ev.num_bridges, "num_agreeing": ev.num_agreeing,
                    "bridges": [
                        {"i": x.i, "j": x.j, "inliers": x.num_inliers,
                         "rotation_error_deg": None if not np.isfinite(x.rotation_error_deg)
                         else round(x.rotation_error_deg, 3),
                         "translation_error_deg": None if not np.isfinite(x.translation_error_deg)
                         else round(x.translation_error_deg, 3),
                         "agrees": x.agrees, "note": x.note}
                        for x in ev.bridges[:20]],
                })
        report["pairwise_evidence"] = detail

    path = out_dir / f"diagnostics_{run_id}.json"
    write_json(path, report)
    print()
    print(f"  report written: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
