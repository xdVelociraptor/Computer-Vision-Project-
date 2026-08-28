"""Stage 1 CLI - build and validate the input image set for one scene.

What it does
------------
1. (video input only) extracts ordered frames from the source video
2. computes quality diagnostics for every raw image
3. detects near-duplicates
4. applies the configured accept/reject policy
5. copies accepted images to ``processed_images/`` and rejects to ``review/``
6. writes ``image_quality.csv``, ``preprocess_summary.json`` and a 9-panel plot

Raw input is never modified, moved or deleted.

Examples
--------
    # photographs already in data/scene_01/raw_images
    python scripts/run_preprocess.py --config configs/scene_01.yaml

    # video capture, sampled at 2 frames per second
    python scripts/run_preprocess.py --config configs/scene_01.yaml \
        --source video --video videos/scene_01.mp4 --fps 2

    # treatment arm of the image-quality experiment
    python scripts/run_preprocess.py --config configs/scene_01.yaml \
        --config configs/quality_filtered.yaml --force

    # arbitrary override of any config key
    python scripts/run_preprocess.py --config configs/scene_01.yaml \
        --set processing.resize_max_dim=3000
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ConfigError, load_config  # noqa: E402
from src.preprocess import run_preprocess  # noqa: E402
from src.utils import (  # noqa: E402
    Stopwatch,
    ensure_dir,
    set_global_seed,
    setup_logging,
    timestamp,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 1: input validation and image quality analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Raw images are never modified. Rejected images are COPIED to review/, not deleted.",
    )
    parser.add_argument(
        "--config", type=Path, action="append", default=None, metavar="YAML",
        help="Config layer merged onto configs/default.yaml. Repeatable; later layers win.",
    )
    parser.add_argument("--scene", type=str, default=None, help="Override scene.name.")
    parser.add_argument(
        "--source", choices=["images", "video"], default=None, help="Override scene.source_type.",
    )
    parser.add_argument(
        "--video", type=Path, default=None,
        help="Source video (relative to data_root or absolute). Implies --source video.",
    )
    parser.add_argument("--fps", type=float, default=None, help="Target extraction fps (video mode=fps).")
    parser.add_argument("--every-n", type=int, default=None, help="Keep every N-th frame (video).")
    parser.add_argument("--max-frames", type=int, default=None, help="Cap on extracted frames (video).")
    parser.add_argument(
        "--skip-extraction", action="store_true",
        help="Reuse frames already in raw_images/ instead of re-decoding the video.",
    )
    parser.add_argument(
        "--filter", action="store_true",
        help="Shorthand for quality.mode=filter with the blur/brightness/contrast rules enabled.",
    )
    parser.add_argument("--workers", type=int, default=None, help="Override runtime.workers.")
    parser.add_argument(
        "--resize", type=int, default=None, metavar="PX",
        help="Downscale processed images so the long side is at most PX (0 = keep native).",
    )
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
        help="Override any config key, e.g. --set quality.blur.percentile=20",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip figure generation.")
    parser.add_argument("--no-cache", action="store_true", help="Recompute all metrics, ignoring the cache.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Analyse and report, but do not write processed_images/ or review/.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Allow overwriting a non-empty processed_images/ or review/ directory.",
    )
    parser.add_argument("--log-level", default=None, choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def _overrides_from_args(args: argparse.Namespace) -> List[str]:
    """Translate the convenience flags into `--set`-style override strings."""
    overrides: List[str] = []
    if args.scene:
        overrides.append(f"scene.name={args.scene}")
    if args.video is not None:
        overrides.append("scene.source_type=video")
        overrides.append(f"scene.video_file={args.video.as_posix()}")
    if args.source:
        overrides.append(f"scene.source_type={args.source}")
    if args.fps is not None:
        overrides.append("video.mode=fps")
        overrides.append(f"video.target_fps={args.fps}")
    if args.every_n is not None:
        overrides.append("video.mode=every_n")
        overrides.append(f"video.every_n={args.every_n}")
    if args.max_frames is not None:
        overrides.append(f"video.max_frames={args.max_frames}")
        if args.fps is None and args.every_n is None:
            overrides.append("video.mode=max_frames")
    if args.filter:
        overrides.append("quality.mode=filter")
        overrides.append("quality.filters=[blur, brightness, contrast, resolution]")
    if args.workers is not None:
        overrides.append(f"runtime.workers={args.workers}")
    if args.resize is not None:
        overrides.append(f"processing.resize_max_dim={args.resize}")
    if args.no_cache:
        overrides.append("processing.cache=false")
    if args.log_level:
        overrides.append(f"runtime.log_level={args.log_level}")
    # User-supplied --set wins over the convenience flags.
    overrides.extend(args.overrides)
    return overrides


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        cfg = load_config(args.config, overrides=_overrides_from_args(args))
    except ConfigError as exc:
        print(f"CONFIGURATION ERROR\n{exc}", file=sys.stderr)
        return 2

    run_id = timestamp()
    log_path = ensure_dir(cfg.logs_dir) / f"preprocess_{run_id}.log"
    logger = setup_logging(log_path, level=cfg.runtime.log_level)
    set_global_seed(cfg.runtime.seed)

    logger.info("=" * 70)
    logger.info("STAGE 1 - INPUT VALIDATION AND QUALITY ANALYSIS")
    logger.info("scene=%s  source=%s  run_id=%s", cfg.scene.name, cfg.scene.source_type, run_id)
    logger.info("raw    : %s", cfg.raw_images_dir)
    logger.info("output : %s", cfg.scene_output_dir)
    logger.info("=" * 70)

    # The exact configuration used is stored next to the results, so any run can
    # be reproduced verbatim later.
    cfg.dump_yaml(cfg.metrics_dir / f"config_preprocess_{run_id}.yaml")

    try:
        with Stopwatch("stage 1", logger) as sw:
            result = run_preprocess(
                cfg,
                force=args.force,
                dry_run=args.dry_run,
                skip_extraction=args.skip_extraction,
                make_plots=not args.no_plots,
            )
    except FileExistsError as exc:
        logger.error("%s", exc)
        return 3
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        logger.error("Stage 1 failed: %s", exc)
        logger.debug("%s", traceback.format_exc())
        return 4
    except Exception as exc:  # unexpected: dump the traceback to the log file
        logger.error("Unexpected error: %s: %s", type(exc).__name__, exc)
        logger.error("%s", traceback.format_exc())
        return 5

    print()
    print("=" * 70)
    print(f"STAGE 1 COMPLETE - scene '{result.scene}' in {sw.elapsed:.1f} s")
    print("=" * 70)
    print(f"  total images       : {result.total_images}")
    print(f"  accepted           : {result.accepted_images}")
    print(f"  rejected           : {result.rejected_images}")
    print(f"  unreadable         : {result.unreadable_images}")
    print(f"  near-duplicate grps: {result.duplicates.get('num_groups', 0)} "
          f"({result.duplicates.get('num_duplicates', 0)} images)")
    print(f"  policy             : {result.policy.get('mode')} "
          f"filters={result.policy.get('active_filters')}")
    if result.policy.get("blur_threshold") is not None:
        print(f"  blur threshold     : {result.policy['blur_threshold']} "
              f"({result.policy.get('blur_threshold_mode')})")
    print()
    print(f"  quality CSV        : {result.csv_path}")
    print(f"  summary JSON       : {result.summary_path}")
    if result.plot_path:
        print(f"  overview figure    : {result.plot_path}")
    if result.extraction_report_path:
        print(f"  extraction report  : {result.extraction_report_path}")
    if not args.dry_run:
        print(f"  processed images   : {cfg.processed_images_dir} "
              f"({result.materialisation.get('copied', 0)} copied, "
              f"{result.materialisation.get('reencoded', 0)} re-encoded)")
        print(f"  review copies      : {cfg.review_dir} "
              f"({result.materialisation.get('rejected_copied', 0)} images)")
    print(f"  log file           : {log_path}")

    if result.warnings:
        print()
        print("  WARNINGS:")
        for warning in result.warnings:
            print(f"   ! {warning}")

    failures = result.materialisation.get("failed") or []
    if failures:
        print()
        print(f"  {len(failures)} file(s) could not be written:")
        for item in failures[:10]:
            print(f"   ! {item['image']}: {item['error']}")

    write_json(
        cfg.metrics_dir / f"preprocess_result_{run_id}.json",
        {"run_id": run_id, "elapsed_s": round(sw.elapsed, 3), **vars(result)},
    )
    print()
    print("  Next step: inspect the CSV and the overview figure, then run the COLMAP stage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
