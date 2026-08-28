from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sfm_core.lidar_evaluation import evaluate_lidar_alignment  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Align custom SfM geometry to an iPhone LiDAR scan and compute metrics.")
    parser.add_argument("--sfm", type=Path, required=True)
    parser.add_argument("--lidar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--correspondences", type=Path)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.02, 0.05, 0.10])
    args = parser.parse_args()
    report = evaluate_lidar_alignment(
        args.sfm,
        args.lidar,
        args.output,
        correspondences_path=args.correspondences,
        thresholds=tuple(args.thresholds),
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
