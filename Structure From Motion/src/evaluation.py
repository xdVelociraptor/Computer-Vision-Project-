"""Stage 8b - results aggregation, statistical comparison and failure analysis.

Turns the raw ``results.csv`` files produced by :mod:`src.experiments` into the
tables and findings that go into the report:

* aggregation across scenes and repeats
* per-experiment comparison tables with run-to-run spread
* a structured failure report that checks each failure against the known causes
  of SfM breakdown, using evidence from the earlier stages where it exists

Why the spread matters
----------------------
COLMAP's multi-threaded bundle adjustment is not bit-deterministic (measured at
roughly 0.2% variation in 3D point count on an identical dataset - see README
section 11). A difference between two conditions smaller than that spread is
**not** a result. Where repeats exist, this module reports mean, standard
deviation and range so that claims can be checked against the noise floor.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .utils import ensure_dir, get_logger, write_json

__all__ = [
    "markdown_table",
    "flag_timing_outliers",
    "load_results",
    "aggregate",
    "compare_conditions",
    "build_failure_report",
    "summarise_cross_scene",
    "FAILURE_CAUSES",
]

LOGGER = get_logger()

#: The failure causes listed in the project brief, each with the evidence that
#: confirms or rules it out. ``check`` receives the merged evidence dict and
#: returns (verdict, detail) or None when no evidence is available.
FAILURE_CAUSES: List[Dict[str, Any]] = [
    {
        "cause": "insufficient overlap between images",
        "evidence": "stage 2 inliers at stride 1; stage 4 track length",
        "remedy": "Capture with smaller steps between shots; aim for >=70% overlap.",
    },
    {
        "cause": "weak or repetitive texture",
        "evidence": "stage 1 edge density; stage 2 ratio-test survival rate",
        "remedy": "Avoid blank walls and repeating patterns; include irregular detail.",
    },
    {
        "cause": "motion blur",
        "evidence": "stage 1 blur_score distribution",
        "remedy": "Brace the camera, shorten exposure, or filter the blurriest images.",
    },
    {
        "cause": "reflective or transparent surfaces",
        "evidence": "visual inspection; stage 2 inlier ratio",
        "remedy": "Reflections move with the camera and violate the rigid-scene assumption.",
    },
    {
        "cause": "insufficient geometric baseline (rotation-dominated capture)",
        "evidence": "stage 2 homography dominance; stage 4 baseline distribution",
        "remedy": "Translate between shots. Pure rotation cannot triangulate.",
    },
    {
        "cause": "degenerate planar configuration",
        "evidence": "stage 2 homography dominance",
        "remedy": "Include non-planar structure or vary the viewpoint more.",
    },
    {
        "cause": "too few feature matches",
        "evidence": "stage 2 mutual matches and RANSAC inliers",
        "remedy": "Lower mapper.min_num_matches, or improve capture density.",
    },
    {
        "cause": "incorrect camera intrinsics",
        "evidence": "stage 4 estimated focal length vs EXIF prior",
        "remedy": "Check camera.model and whether EXIF focal data is present.",
    },
    {
        "cause": "disconnected image graph",
        "evidence": "stage 3 number of sub-models",
        "remedy": "More than one model means the capture broke into pieces; add linking views.",
    },
    {
        "cause": "inappropriate matching strategy",
        "evidence": "matcher experiment comparison",
        "remedy": "Sequential matching suits ordered captures; exhaustive suits unordered sets.",
    },
    {
        "cause": "GPU / CUDA failure",
        "evidence": "COLMAP stage log",
        "remedy": "Run scripts/check_environment.py --deep; try --no-gpu.",
    },
    {
        "cause": "COLMAP configuration error",
        "evidence": "COLMAP stage log",
        "remedy": "Read the failing stage's log; check option names against the installed binary.",
    },
]


def markdown_table(df: pd.DataFrame, float_format: str = "{:.4g}") -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table.

    Written by hand rather than using ``DataFrame.to_markdown`` so the project
    does not gain a ``tabulate`` dependency just to format one table.
    """
    if df.empty:
        return "_(no data)_"

    def _cell(value: Any) -> str:
        if isinstance(value, float):
            return float_format.format(value)
        return str(value)

    headers = [str(c) for c in df.columns]
    rows = [[_cell(v) for v in row] for row in df.itertuples(index=False, name=None)]
    widths = [max(len(headers[i]), *(len(r[i]) for r in rows)) if rows else len(headers[i])
              for i in range(len(headers))]

    lines = ["| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |",
             "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(row))) + " |")
    return "\n".join(lines)


def load_results(paths: Sequence[Path]) -> pd.DataFrame:
    """Load and concatenate one or more ``results.csv`` files.

    Missing files are skipped with a warning rather than raising, so an
    aggregate across scenes still works when one scene has not been run.
    """
    frames: List[pd.DataFrame] = []
    for path in paths:
        path = Path(path)
        if not path.is_file():
            LOGGER.warning("Results file not found, skipping: %s", path)
            continue
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            LOGGER.warning("Could not read %s: %s", path, exc)
            continue
        if df.empty:
            continue
        df["source_file"] = str(path)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


#: A run whose wall-clock time exceeds this multiple of its condition's median
#: did not spend that time computing. The usual cause is the machine sleeping
#: or hibernating mid-suite: the timers keep counting, so a 4-minute mapping
#: stage can be recorded as 16 hours. Reconstruction metrics from such a run
#: are perfectly valid - only its timings are meaningless.
TIMING_OUTLIER_FACTOR = 5.0


def flag_timing_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``timing_reliable`` column, marking runs with implausible timings.

    Compares each run's ``total_time`` against the median for the same
    condition. Requires at least 3 repeats of a condition to judge, since a
    median of two is not robust.

    This keeps a suspect run's *measurements* in the analysis while excluding
    its *timings* - throwing the whole row away would discard good data, and
    keeping its timing would corrupt every runtime average.
    """
    if df.empty or "total_time" not in df.columns:
        return df
    df = df.copy()
    df["timing_reliable"] = True

    keys = [k for k in ("scene", "image_count", "matcher", "quality_arm") if k in df.columns]
    if not keys:
        return df

    for _, group in df.groupby(keys, dropna=False):
        if len(group) < 3:
            continue
        median = float(group["total_time"].median())
        if median <= 0:
            continue
        suspect = group.index[group["total_time"] > TIMING_OUTLIER_FACTOR * median]
        df.loc[suspect, "timing_reliable"] = False
    return df


#: Metrics aggregated when repeats exist.
_NUMERIC_METRICS = [
    "registered_images", "registration_rate", "sparse_points", "observations",
    "mean_track_length", "mean_reprojection_error", "dense_points",
    "feature_time", "matching_time", "mapping_time", "dense_time", "total_time",
]


def aggregate(df: pd.DataFrame, group_keys: Sequence[str] = ("scene", "experiment_type", "experiment_name")) -> pd.DataFrame:
    """Collapse repeats into mean / std / min / max per condition.

    Reporting the spread is not decoration: it is what makes a difference
    between two conditions interpretable.
    """
    if df.empty:
        return df
    keys = [k for k in group_keys if k in df.columns]
    metrics = [m for m in _NUMERIC_METRICS if m in df.columns]
    if not keys or not metrics:
        return df

    grouped = df.groupby(keys, dropna=False)
    out = grouped.agg({m: ["mean", "std", "min", "max"] for m in metrics})
    out.columns = [f"{metric}_{stat}" for metric, stat in out.columns]
    out["runs"] = grouped.size()
    if "success" in df.columns:
        out["success_rate"] = grouped["success"].mean()
    if "image_count" in df.columns:
        out["image_count"] = grouped["image_count"].first()
    return out.reset_index()


def compare_conditions(
    df: pd.DataFrame, experiment_type: str, metric: str = "registration_rate"
) -> pd.DataFrame:
    """Comparison table for one experiment type, sorted by the chosen metric."""
    if df.empty or "experiment_type" not in df.columns:
        return pd.DataFrame()
    sub = df[df["experiment_type"] == experiment_type]
    if sub.empty:
        return pd.DataFrame()

    columns = [c for c in (
        "scene", "experiment_name", "image_count", "matcher", "quality_arm",
        "registered_images", "registration_rate", "sparse_points", "observations",
        "mean_track_length", "mean_reprojection_error", "dense_points",
        "feature_time", "matching_time", "mapping_time", "total_time", "success",
    ) if c in sub.columns]
    table = sub[columns].copy()
    if metric in table.columns:
        table = table.sort_values(metric, ascending=False)
    return table


def _collect_evidence(scene_output_dir: Path) -> Dict[str, Any]:
    """Gather diagnostics from earlier stages to explain a failure.

    Reads whatever exists; a missing stage simply contributes no evidence.
    """
    evidence: Dict[str, Any] = {}
    metrics = Path(scene_output_dir) / "metrics"

    quality_csv = metrics / "image_quality.csv"
    if quality_csv.is_file():
        try:
            q = pd.read_csv(quality_csv)
            evidence["stage1"] = {
                "images": int(len(q)),
                "median_blur_score": float(q["blur_score"].median()) if "blur_score" in q else None,
                "median_edge_density": float(q["edge_density"].median()) if "edge_density" in q else None,
                "images_with_exif": int(q["has_exif"].sum()) if "has_exif" in q else None,
                "near_identical_consecutive": int(
                    ((q["hamming_to_prev"] >= 0) & (q["hamming_to_prev"] <= 2)).sum()
                ) if "hamming_to_prev" in q else None,
            }
        except Exception as exc:
            LOGGER.debug("Could not read stage-1 evidence: %s", exc)

    feature_json = metrics / "feature_analysis_summary.json"
    if feature_json.is_file():
        try:
            import json
            data = json.loads(feature_json.read_text(encoding="utf-8"))
            per_detector = data.get("per_detector", {})
            primary = per_detector.get("SIFT") or (
                next(iter(per_detector.values())) if per_detector else {}
            )
            evidence["stage2"] = {
                "mean_f_inliers": primary.get("mean_f_inliers"),
                "mean_f_inlier_ratio": primary.get("mean_f_inlier_ratio"),
                "degenerate_fraction": primary.get("degenerate_fraction"),
                "diagnostics": data.get("diagnostics", []),
                "per_stride": data.get("per_stride", {}),
            }
        except Exception as exc:
            LOGGER.debug("Could not read stage-2 evidence: %s", exc)
    return evidence


def build_failure_report(
    df: pd.DataFrame, scene_output_dirs: Dict[str, Path], path: Optional[Path] = None
) -> Dict[str, Any]:
    """Produce a structured report of every failed experiment.

    For each failure it records the stage, COLMAP's diagnosis, and the full
    checklist of possible causes from the project brief - annotated with the
    measured evidence for or against each one, where earlier stages provide it.
    """
    report: Dict[str, Any] = {
        "total_experiments": int(len(df)),
        "failures": [],
        "num_failures": 0,
        "cause_checklist": FAILURE_CAUSES,
    }
    if df.empty or "success" not in df.columns:
        return report

    failures = df[~df["success"].astype(bool)]
    report["num_failures"] = int(len(failures))
    report["failure_rate_percent"] = round(100.0 * len(failures) / max(1, len(df)), 2)

    evidence_cache: Dict[str, Dict[str, Any]] = {}
    for _, row in failures.iterrows():
        scene = str(row.get("scene", ""))
        if scene not in evidence_cache and scene in scene_output_dirs:
            evidence_cache[scene] = _collect_evidence(scene_output_dirs[scene])
        evidence = evidence_cache.get(scene, {})

        entry: Dict[str, Any] = {
            "experiment_name": row.get("experiment_name"),
            "experiment_type": row.get("experiment_type"),
            "scene": scene,
            "image_count": int(row.get("image_count", 0) or 0),
            "matcher": row.get("matcher"),
            "failure_stage": row.get("failure_stage"),
            "failure_reason": row.get("failure_reason"),
            "experiment_dir": row.get("experiment_dir"),
            "likely_causes": _rank_causes(row, evidence),
            "evidence": evidence,
        }
        report["failures"].append(entry)

    # A pattern across failures is more informative than any single failure.
    if report["failures"]:
        by_stage: Dict[str, int] = {}
        for f in report["failures"]:
            key = str(f.get("failure_stage") or "unknown")
            by_stage[key] = by_stage.get(key, 0) + 1
        report["failures_by_stage"] = by_stage
        smallest = min((f["image_count"] for f in report["failures"] if f["image_count"]),
                       default=0)
        successes = df[df["success"].astype(bool)]
        if smallest and not successes.empty and "image_count" in successes.columns:
            largest_success = int(successes["image_count"].max())
            if smallest > largest_success:
                report["pattern"] = (
                    f"All failures used at least {smallest} images while the largest successful "
                    f"run used {largest_success}; the failures are not simply a lack of images."
                )
            else:
                report["pattern"] = (
                    f"Failures occur at {smallest} images and below, while successes reach "
                    f"{largest_success}; this is consistent with too few images / too little "
                    f"overlap rather than a scene-property problem."
                )

    if path is not None:
        write_json(Path(path), report)
    return report


def _rank_causes(row: pd.Series, evidence: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Annotate the cause checklist with measured evidence for this failure."""
    ranked: List[Dict[str, Any]] = []
    stage2 = evidence.get("stage2", {})
    stage1 = evidence.get("stage1", {})
    reason = str(row.get("failure_reason", "")).lower()

    def _add(cause: str, verdict: str, detail: str) -> None:
        ranked.append({"cause": cause, "verdict": verdict, "detail": detail})

    # Environment failures must short-circuit. If the binary never started,
    # the images are irrelevant, and listing data-related causes underneath
    # would invite exactly the wrong conclusion.
    if re.search(r"could not start|failed to start|dll initialisation|0xc00001|"
                 r"exited immediately|resource-exhaustion", reason, re.IGNORECASE):
        _add("environment / process launch failure", "CONFIRMED",
             "COLMAP never started, so this failure says nothing about the image data. "
             "Re-run the failed experiments; completed ones are skipped automatically.")
        return ranked

    inliers = stage2.get("mean_f_inliers")
    if inliers is not None:
        if inliers < 100:
            _add("too few feature matches", "LIKELY",
                 f"stage 2 measured only {inliers:.0f} mean verified inliers per pair")
        else:
            _add("too few feature matches", "unlikely",
                 f"stage 2 measured {inliers:.0f} mean verified inliers per pair")

    degenerate = stage2.get("degenerate_fraction")
    if degenerate is not None:
        verdict = "LIKELY" if degenerate > 0.3 else "unlikely"
        _add("degenerate planar / rotation-dominated configuration", verdict,
             f"{degenerate:.0%} of analysed pairs were homography-dominant")

    near_identical = stage1.get("near_identical_consecutive")
    if near_identical:
        _add("insufficient geometric baseline", "POSSIBLE",
             f"{near_identical} consecutive image pairs were near-identical in stage 1")

    if stage1.get("images_with_exif") == 0:
        _add("incorrect camera intrinsics", "POSSIBLE",
             "no EXIF focal length available; COLMAP used a heuristic prior")

    if int(row.get("num_models", 0) or 0) > 1:
        _add("disconnected image graph", "CONFIRMED",
             f"{int(row['num_models'])} sub-models were produced")

    if "cuda" in reason or "kernel image" in reason or "gpu" in reason:
        _add("GPU / CUDA failure", "CONFIRMED", "the COLMAP error mentions CUDA/GPU")
    if "initial image pair" in reason:
        _add("insufficient overlap or degenerate initialisation", "CONFIRMED",
             "COLMAP could not find a valid initial image pair")

    if not ranked:
        _add("undetermined", "unknown",
             "Run stages 1 and 2 on this scene to gather diagnostic evidence.")
    return ranked


def summarise_cross_scene(df: pd.DataFrame) -> pd.DataFrame:
    """Best result per scene, for the generalisation/robustness comparison."""
    if df.empty or "scene" not in df.columns:
        return pd.DataFrame()
    successes = df[df["success"].astype(bool)] if "success" in df.columns else df
    if successes.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for scene, group in successes.groupby("scene"):
        # Compare each scene at its best, using the run with the most
        # registered images; comparing a scene's worst run against another's
        # best would say more about the sweep than about the scenes.
        best = group.loc[group["registered_images"].idxmax()]
        rows.append({
            "scene": scene,
            "best_experiment": best.get("experiment_name"),
            "image_count": int(best.get("image_count", 0) or 0),
            "registered_images": int(best.get("registered_images", 0) or 0),
            "registration_rate": float(best.get("registration_rate", 0.0) or 0.0),
            "sparse_points": int(best.get("sparse_points", 0) or 0),
            "observations": int(best.get("observations", 0) or 0),
            "mean_track_length": float(best.get("mean_track_length", 0.0) or 0.0),
            "mean_reprojection_error": float(best.get("mean_reprojection_error", 0.0) or 0.0),
            "dense_points": int(best.get("dense_points", 0) or 0),
            "total_time": float(best.get("total_time", 0.0) or 0.0),
            "runs_attempted": int(len(df[df["scene"] == scene])),
            "runs_successful": int(len(group)),
        })
    return pd.DataFrame(rows).sort_values("registration_rate", ascending=False)


def derive_findings(df: pd.DataFrame) -> List[str]:
    """Plain-language findings drawn from the results table.

    Deliberately conservative: differences are only called out when they exceed
    the ~0.2% run-to-run variation measured for COLMAP, and comparisons that
    rest on a single run each are flagged as such.
    """
    findings: List[str] = []
    if df.empty:
        return ["No results to analyse."]

    successes = df[df["success"].astype(bool)] if "success" in df.columns else df
    if successes.empty:
        return ["Every experiment failed. See the failure report for probable causes."]

    successes = flag_timing_outliers(successes)
    unreliable = successes[~successes["timing_reliable"]] if "timing_reliable" in successes else pd.DataFrame()
    if not unreliable.empty:
        names = ", ".join(f"{r.experiment_name}/rep{r.repeat_index}" for r in unreliable.itertuples())
        findings.append(
            f"TIMING WARNING: {len(unreliable)} run(s) recorded wall-clock times more than "
            f"{TIMING_OUTLIER_FACTOR:.0f}x their condition's median ({names}). The usual cause "
            f"is the machine sleeping mid-suite - the timers keep counting while nothing runs. "
            f"Their reconstruction metrics are valid and retained; their timings are excluded "
            f"from every runtime figure below."
        )
    # Timing statements use only trustworthy rows; quality statements use all.
    timed = successes[successes["timing_reliable"]] if "timing_reliable" in successes else successes

    # --- image count ----------------------------------------------------
    counts = timed[timed.get("experiment_type") == "image_count"] \
        if "experiment_type" in timed.columns else pd.DataFrame()
    if len(counts) >= 2:
        by_count = counts.groupby("image_count").agg(
            rate=("registration_rate", "mean"),
            points=("sparse_points", "mean"),
            error=("mean_reprojection_error", "mean"),
            time=("total_time", "mean"),
        ).sort_index()
        smallest, largest = by_count.index.min(), by_count.index.max()
        findings.append(
            f"Image count {smallest} -> {largest}: registration rate "
            f"{by_count.loc[smallest, 'rate']:.1f}% -> {by_count.loc[largest, 'rate']:.1f}%, "
            f"sparse points {by_count.loc[smallest, 'points']:.0f} -> "
            f"{by_count.loc[largest, 'points']:.0f}, runtime "
            f"{by_count.loc[smallest, 'time']:.1f}s -> {by_count.loc[largest, 'time']:.1f}s."
        )
        if by_count["time"].max() > 0 and len(by_count) >= 3:
            counts_arr = by_count.index.to_numpy(dtype=float)
            times = by_count["time"].to_numpy(dtype=float)
            positive = counts_arr > 0
            if positive.sum() >= 3 and times[positive].min() > 0:
                # Slope in log-log space estimates the growth exponent:
                # ~1 is linear, ~2 is quadratic (as exhaustive matching implies).
                slope = float(np.polyfit(np.log(counts_arr[positive]),
                                         np.log(times[positive]), 1)[0])
                findings.append(
                    f"Runtime scales as approximately O(n^{slope:.2f}) with image count "
                    f"(log-log fit over {int(positive.sum())} points)."
                )

    # --- matcher --------------------------------------------------------
    matchers = timed[timed.get("experiment_type") == "matcher"] \
        if "experiment_type" in timed.columns else pd.DataFrame()
    if len(matchers) >= 2:
        by_matcher = matchers.groupby("matcher").agg(
            rate=("registration_rate", "mean"),
            points=("sparse_points", "mean"),
            match_time=("matching_time", "mean"),
            total=("total_time", "mean"),
            runs=("registration_rate", "size"),
        )
        parts = [f"{m}: {r['rate']:.1f}% registered, {r['points']:.0f} points, "
                 f"{r['match_time']:.1f}s matching" for m, r in by_matcher.iterrows()]
        findings.append("Matcher comparison - " + "; ".join(parts) + ".")
        if by_matcher["runs"].max() == 1:
            findings.append(
                "Matcher comparison rests on a single run per condition. COLMAP is not "
                "bit-deterministic when multi-threaded, so compare any difference against the "
                "measured reproducibility spread below before calling it real; raise "
                "experiments.repeat to measure it directly."
            )

    # --- quality --------------------------------------------------------
    quality = successes[successes.get("experiment_type") == "quality"] \
        if "experiment_type" in successes.columns else pd.DataFrame()
    if len(quality) >= 2 and "quality_arm" in quality.columns:
        by_arm = quality.groupby("quality_arm").agg(
            rate=("registration_rate", "mean"),
            points=("sparse_points", "mean"),
            error=("mean_reprojection_error", "mean"),
            images=("image_count", "mean"),
        )
        if "all" in by_arm.index:
            control = by_arm.loc["all"]
            for arm, row in by_arm.iterrows():
                if arm == "all":
                    continue
                delta_points = 100.0 * (row["points"] - control["points"]) / max(1.0, control["points"])
                direction = "improved" if delta_points > 0 else "reduced"
                findings.append(
                    f"Quality filtering '{arm}' used {row['images']:.0f} of "
                    f"{control['images']:.0f} images and {direction} the sparse point count by "
                    f"{abs(delta_points):.1f}% ({control['points']:.0f} -> {row['points']:.0f}), "
                    f"with reprojection error {control['error']:.4f} -> {row['error']:.4f} px."
                )
            findings.append(
                "Note: filtering removes weak images but also removes overlap. The sign of the "
                "effect is an empirical result for this scene, not a general rule."
            )

    findings.extend(reproducibility_findings(successes))

    failures = df[~df["success"].astype(bool)] if "success" in df.columns else pd.DataFrame()
    if not failures.empty:
        findings.append(
            f"{len(failures)} of {len(df)} experiments failed. See failure_report.json for the "
            f"per-failure cause analysis."
        )
    return findings


def reproducibility_findings(successes: pd.DataFrame) -> List[str]:
    """Measure run-to-run variation from conditions that are actually identical.

    Different experiment types often contain the same underlying condition - a
    full-image exhaustive run appears in the image-count sweep, the matcher
    comparison, the quality control arm and the baseline. Those are free
    repeats, and grouping on the variables that actually define a condition
    turns them into a direct measurement of the noise floor.

    This matters because every comparison elsewhere in the report is only
    meaningful relative to this spread.
    """
    keys = [k for k in ("scene", "image_count", "matcher", "quality_arm")
            if k in successes.columns]
    if not keys or "sparse_points" not in successes.columns or len(successes) < 2:
        return []

    findings: List[str] = []
    for values, group in successes.groupby(keys, dropna=False):
        if len(group) < 2:
            continue
        points = group["sparse_points"].to_numpy(dtype=float)
        if points.mean() <= 0:
            continue
        spread = (points.max() - points.min()) / points.mean() * 100.0
        label = ", ".join(f"{k}={v}" for k, v in zip(keys, np.atleast_1d(values)))
        detail = ""
        if "mean_reprojection_error" in group.columns:
            errors = group["mean_reprojection_error"].to_numpy(dtype=float)
            detail = (f", reprojection error {errors.min():.5f}-{errors.max():.5f} px")
        findings.append(
            f"Reproducibility: {len(group)} identical runs ({label}) produced "
            f"{int(points.min())}-{int(points.max())} sparse points, a spread of "
            f"{spread:.2f}% of the mean{detail}. Treat differences smaller than this as noise."
        )
    return findings
