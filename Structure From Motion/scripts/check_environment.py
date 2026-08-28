"""Verify that every dependency this project needs is installed and usable.

Run this FIRST, and again whenever something breaks. It probes the Python
version, every required and optional package, the OpenCV build (including
whether SIFT can actually be constructed), the COLMAP executable and its CUDA
support, the NVIDIA driver, FFmpeg, free disk space, and OneDrive sync risk.

Usage
-----
    python scripts/check_environment.py
    python scripts/check_environment.py --verbose

Exit codes
----------
    0  everything required is present (warnings may still be printed)
    1  at least one required component is missing or broken
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ConfigError, load_config  # noqa: E402
from src.environment import check_environment, format_report  # noqa: E402
from src.utils import ensure_dir, write_json  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify the SfM project environment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Optional scene config; used to check the configured data/output volumes.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Where to write the JSON report (default: <output_root>/_env/environment_report.json).",
    )
    parser.add_argument("--verbose", action="store_true", help="Print the raw data behind each check.")
    parser.add_argument(
        "--deep", action="store_true",
        help="Also run COLMAP's CUDA kernel test - the decisive check for dense MVS support.",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Exit non-zero on warnings as well as failures.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    check_paths = [Path.cwd()]
    output_path = args.output
    try:
        cfg = load_config(args.config)
        check_paths = [cfg.paths.data_root, cfg.paths.output_root]
        if output_path is None:
            output_path = cfg.paths.output_root / "_env" / "environment_report.json"
    except ConfigError as exc:
        print(f"[WARN] Could not load configuration ({exc}); checking the current directory instead.\n",
              file=sys.stderr)

    report = check_environment(check_paths=check_paths, deep=args.deep)
    print(format_report(report, verbose=args.verbose))

    if output_path is not None:
        ensure_dir(Path(output_path).parent)
        write_json(Path(output_path), report.to_dict())
        print(f"\nJSON report written to: {output_path}")

    if not report.ok:
        return 1
    if args.strict and report.warnings:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
