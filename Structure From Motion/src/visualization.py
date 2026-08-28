"""Publication-quality plotting.

All figures produced by the project go through this module so that fonts,
sizes, DPI and colour conventions stay consistent across the report.

Conventions
-----------
* Every axis is labelled **with units**.
* Histograms state the sample size in the title.
* Accepted/rejected use the same two colours in every figure.
* Figures are saved at 300 DPI PNG (print quality for a written report).
* Axes are never truncated to exaggerate a difference; bar charts start at zero.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # headless: never try to open a window from a CLI run

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from .utils import ensure_dir, get_logger

__all__ = [
    "apply_plot_style",
    "save_figure",
    "plot_quality_overview",
    "plot_feature_analysis",
    "plot_sparse_overview",
    "plot_camera_trajectory",
    "plot_two_view_sfm",
    "plot_image_count_experiment",
    "plot_matcher_comparison",
    "plot_quality_experiment",
    "plot_cross_scene",
    "plot_dense_comparison",
]

LOGGER = get_logger()

#: Fixed semantic colours reused across every figure in the report.
COLOR_ACCEPTED = "#2a7ab0"
COLOR_REJECTED = "#c1432c"
COLOR_NEUTRAL = "#5a5a5a"
COLOR_THRESHOLD = "#d95f02"

#: Stable per-detector colours so every figure in the report agrees.
DETECTOR_COLORS = {
    "SIFT": "#2a7ab0",
    "ORB": "#c1432c",
    "AKAZE": "#4a9e5c",
    "BRISK": "#8a5fb0",
    "KAZE": "#d9a02c",
}

DPI = 300


def _detector_color(name: str) -> str:
    """Colour for a detector, stable across every figure."""
    return DETECTOR_COLORS.get(str(name).upper(), COLOR_NEUTRAL)


def apply_plot_style() -> None:
    """Set the project-wide matplotlib rcParams."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "legend.frameon": False,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.autolayout": False,
        }
    )


def save_figure(fig: "matplotlib.figure.Figure", path: Path) -> Path:
    """Write a figure to PNG at report resolution and close it."""
    path = Path(path)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def _hist(
    ax: "matplotlib.axes.Axes",
    values: Sequence[float],
    *,
    title: str,
    xlabel: str,
    bins: int = 30,
    threshold: Optional[float] = None,
    threshold_label: str = "reject threshold",
    log_x: bool = False,
) -> None:
    """Histogram helper with an optional decision-threshold marker."""
    data = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if data.size == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return
    if log_x and data.min() > 0:
        edges = np.logspace(np.log10(data.min()), np.log10(data.max() + 1e-9), bins + 1)
        ax.set_xscale("log")
    else:
        edges = bins
    ax.hist(data, bins=edges, color=COLOR_ACCEPTED, edgecolor="white", linewidth=0.4)
    if threshold is not None and np.isfinite(threshold):
        ax.axvline(threshold, color=COLOR_THRESHOLD, linestyle="--", linewidth=1.4, label=threshold_label)
        ax.legend(loc="upper right")
    ax.set_title(f"{title} (n={data.size})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of images")


def plot_quality_overview(
    df: pd.DataFrame,
    path: Path,
    *,
    scene: str = "",
    policy: Optional[Dict[str, Any]] = None,
) -> Path:
    """Nine-panel diagnostic figure for the stage-1 input analysis.

    Panels
    ------
    1. Sharpness distribution (with the active rejection threshold, if any)
    2. Brightness distribution
    3. Contrast distribution
    4. Sharpness vs capture order, split by accept/reject
    5. Brightness vs contrast scatter (exposure quality map)
    6. Perceptual-hash distance to the previous image (baseline diagnostic)
    7. Texture (edge density) vs sharpness
    8. Exposure clipping (shadows vs highlights)
    9. Rejection-reason counts
    """
    apply_plot_style()
    policy = policy or {}
    accepted_mask = df["accepted"].astype(bool) if "accepted" in df else pd.Series(True, index=df.index)

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    title = f"Input image quality analysis{f' - {scene}' if scene else ''}"
    fig.suptitle(
        f"{title}\n{len(df)} images | {int(accepted_mask.sum())} accepted | "
        f"{int((~accepted_mask).sum())} rejected | policy: {policy.get('mode', 'n/a')}"
        + (f", filters={policy.get('active_filters')}" if policy.get("active_filters") else ""),
        fontsize=12,
    )

    # 1 - sharpness
    _hist(
        axes[0][0], df.get("blur_score", pd.Series(dtype=float)),
        title="Sharpness: variance of Laplacian",
        xlabel="Laplacian variance (long side normalised) [intensity$^2$]",
        threshold=policy.get("blur_threshold"),
    )

    # 2 - brightness
    _hist(
        axes[0][1], df.get("brightness", pd.Series(dtype=float)),
        title="Mean brightness", xlabel="Mean grey level [0-255]",
    )

    # 3 - contrast
    _hist(
        axes[0][2], df.get("contrast", pd.Series(dtype=float)),
        title="Global contrast", xlabel="Std. dev. of grey level [0-255]",
    )

    # 4 - sharpness vs capture order
    ax = axes[1][0]
    idx = df.index.to_numpy()
    if "blur_score" in df:
        ax.scatter(idx[accepted_mask.to_numpy()], df.loc[accepted_mask, "blur_score"],
                   s=14, c=COLOR_ACCEPTED, label="accepted", alpha=0.85)
        rejected = ~accepted_mask
        if rejected.any():
            ax.scatter(idx[rejected.to_numpy()], df.loc[rejected, "blur_score"],
                       s=22, c=COLOR_REJECTED, marker="x", label="rejected")
        threshold = policy.get("blur_threshold")
        if threshold is not None:
            ax.axhline(threshold, color=COLOR_THRESHOLD, linestyle="--", linewidth=1.2)
        ax.legend(loc="best")
    ax.set_title("Sharpness along the capture sequence")
    ax.set_xlabel("Image index (capture order)")
    ax.set_ylabel("Laplacian variance [intensity$^2$]")

    # 5 - exposure quality map
    ax = axes[1][1]
    if {"brightness", "contrast"} <= set(df.columns):
        ax.scatter(df.loc[accepted_mask, "brightness"], df.loc[accepted_mask, "contrast"],
                   s=16, c=COLOR_ACCEPTED, label="accepted", alpha=0.85)
        rejected = ~accepted_mask
        if rejected.any():
            ax.scatter(df.loc[rejected, "brightness"], df.loc[rejected, "contrast"],
                       s=24, c=COLOR_REJECTED, marker="x", label="rejected")
        ax.legend(loc="best")
    ax.set_title("Exposure map: brightness vs contrast")
    ax.set_xlabel("Mean grey level [0-255]")
    ax.set_ylabel("Std. dev. of grey level [0-255]")

    # 6 - baseline / motion diagnostic
    ax = axes[1][2]
    if "hamming_to_prev" in df:
        valid = df[df["hamming_to_prev"] >= 0]
        ax.plot(valid.index, valid["hamming_to_prev"], color=COLOR_NEUTRAL, linewidth=1.0, marker="o",
                markersize=2.5)
        ax.axhline(2, color=COLOR_THRESHOLD, linestyle="--", linewidth=1.0,
                   label="near-identical (d$\\leq$2)")
        ax.legend(loc="best")
    ax.set_title("Perceptual change between consecutive images")
    ax.set_xlabel("Image index (capture order)")
    ax.set_ylabel("dHash Hamming distance [bits]")

    # 7 - texture vs sharpness
    ax = axes[2][0]
    if {"edge_density", "blur_score"} <= set(df.columns):
        sc = ax.scatter(df["edge_density"] * 100.0, df["blur_score"],
                        s=16, c=df.get("entropy", pd.Series(0.0, index=df.index)),
                        cmap="viridis", alpha=0.9)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Intensity entropy [bits]")
    ax.set_title("Scene texture vs sharpness")
    ax.set_xlabel("Canny edge pixels [% of image]")
    ax.set_ylabel("Laplacian variance [intensity$^2$]")

    # 8 - clipping
    ax = axes[2][1]
    if {"shadow_clipped_frac", "highlight_clipped_frac"} <= set(df.columns):
        ax.scatter(df.loc[accepted_mask, "shadow_clipped_frac"] * 100.0,
                   df.loc[accepted_mask, "highlight_clipped_frac"] * 100.0,
                   s=16, c=COLOR_ACCEPTED, label="accepted", alpha=0.85)
        rejected = ~accepted_mask
        if rejected.any():
            ax.scatter(df.loc[rejected, "shadow_clipped_frac"] * 100.0,
                       df.loc[rejected, "highlight_clipped_frac"] * 100.0,
                       s=24, c=COLOR_REJECTED, marker="x", label="rejected")
            ax.legend(loc="best")
    ax.set_title("Exposure clipping")
    ax.set_xlabel("Pixels at/below grey 5 [%]")
    ax.set_ylabel("Pixels at/above grey 250 [%]")

    # 9 - rejection reasons
    ax = axes[2][2]
    if "rejection_reason" in df:
        reasons: Dict[str, int] = {}
        for entry in df.loc[~accepted_mask, "rejection_reason"].fillna(""):
            for token in str(entry).split(";"):
                token = token.strip()
                if not token:
                    continue
                key = token.split("(")[0]
                reasons[key] = reasons.get(key, 0) + 1
        if reasons:
            labels = list(reasons)
            counts = [reasons[k] for k in labels]
            ax.barh(labels, counts, color=COLOR_REJECTED)
            ax.set_xlim(left=0)
            # Counts are integers: suppress matplotlib's fractional ticks.
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            for i, c in enumerate(counts):
                ax.text(c, i, f" {c}", va="center", fontsize=8)
        else:
            ax.text(0.5, 0.5, "no images rejected", ha="center", va="center", transform=ax.transAxes)
            ax.set_yticks([])
    ax.set_title("Rejections by cause")
    ax.set_xlabel("Number of images")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save_figure(fig, path)


# ===========================================================================
# Stage 2 - feature detection / matching / two-view geometry
# ===========================================================================


def plot_feature_analysis(
    detector_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    path: Path,
    *,
    scene: str = "",
    num_images: Optional[int] = None,
) -> Path:
    """Nine-panel comparison of detectors and two-view geometry quality.

    Panels
    ------
    1. Keypoints per image, by detector
    2. Match attrition: raw -> ratio -> mutual -> RANSAC inlier
    3. Verified inliers vs baseline (index stride)
    4. Inlier ratio vs baseline
    5. Inlier-ratio distribution per detector
    6. Sampson epipolar error distribution
    7. Degeneracy map: homography dominance vs inlier ratio
    8. Runtime per stage, by detector
    9. Spatial coverage of keypoints
    """
    apply_plot_style()
    detectors = sorted(pair_df["detector"].unique()) if not pair_df.empty else []

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    if num_images is None and not detector_df.empty:
        num_images = int(detector_df["image"].nunique())
    fig.suptitle(
        f"Feature detection, matching and two-view geometry"
        f"{f' - {scene}' if scene else ''}\n"
        f"{num_images or 0} images | "
        f"{len(pair_df) // max(1, len(detectors)) if detectors else 0} pairs per detector | "
        f"detectors: {', '.join(detectors)}",
        fontsize=12,
    )

    # 1 - keypoints per image
    ax = axes[0][0]
    if not detector_df.empty:
        data = [detector_df.loc[detector_df["detector"] == d, "num_keypoints"].to_numpy()
                for d in detectors]
        data = [d for d in data if len(d)]
        if data:
            bp = ax.boxplot(data, tick_labels=[d for d in detectors
                                               if len(detector_df[detector_df["detector"] == d])],
                            patch_artist=True, showfliers=False)
            for patch, name in zip(bp["boxes"], detectors):
                patch.set_facecolor(_detector_color(name))
                patch.set_alpha(0.75)
    ax.set_title("Keypoints detected per image")
    ax.set_xlabel("Detector")
    ax.set_ylabel("Number of keypoints")
    ax.set_ylim(bottom=0)

    # 2 - attrition funnel
    ax = axes[0][1]
    stages = ["ratio_matches", "mutual_matches", "f_inliers"]
    labels = ["Ratio test", "Mutual NN", "RANSAC inliers"]
    x = np.arange(len(stages))
    width = 0.8 / max(1, len(detectors))
    for i, name in enumerate(detectors):
        sub = pair_df[pair_df["detector"] == name]
        values = [float(sub[s].mean()) if s in sub else 0.0 for s in stages]
        ax.bar(x + i * width - 0.4 + width / 2, values, width,
               label=name, color=_detector_color(name))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right")
    ax.set_title("Match attrition through the filtering pipeline")
    ax.set_xlabel("Filtering stage")
    ax.set_ylabel("Mean correspondences per pair")

    # 3 - inliers vs baseline
    ax = axes[0][2]
    for name in detectors:
        sub = pair_df[pair_df["detector"] == name].groupby("stride")["f_inliers"].mean()
        if not sub.empty:
            ax.plot(sub.index, sub.to_numpy(), marker="o", label=name, color=_detector_color(name))
    ax.set_ylim(bottom=0)
    ax.legend(loc="best")
    ax.set_title("Verified inliers vs baseline width")
    ax.set_xlabel("Image index stride (proxy for baseline)")
    ax.set_ylabel("Mean RANSAC inliers per pair")

    # 4 - inlier ratio vs baseline
    ax = axes[1][0]
    for name in detectors:
        sub = pair_df[pair_df["detector"] == name].groupby("stride")["f_inlier_ratio"].mean()
        if not sub.empty:
            ax.plot(sub.index, sub.to_numpy() * 100.0, marker="s", label=name,
                    color=_detector_color(name))
    ax.set_ylim(0, 100)
    ax.legend(loc="best")
    ax.set_title("Geometric verification rate vs baseline width")
    ax.set_xlabel("Image index stride (proxy for baseline)")
    ax.set_ylabel("Mean RANSAC inlier ratio [%]")

    # 5 - inlier ratio distribution
    ax = axes[1][1]
    for name in detectors:
        values = pair_df.loc[pair_df["detector"] == name, "f_inlier_ratio"].to_numpy() * 100.0
        if values.size:
            ax.hist(values, bins=20, range=(0, 100), alpha=0.55,
                    label=name, color=_detector_color(name))
    ax.legend(loc="best")
    ax.set_title("Distribution of RANSAC inlier ratio")
    ax.set_xlabel("Inlier ratio [%]")
    ax.set_ylabel("Number of pairs")

    # 6 - Sampson error
    ax = axes[1][2]
    for name in detectors:
        values = pair_df.loc[(pair_df["detector"] == name) & (pair_df["success"]),
                             "f_mean_sampson_px"].to_numpy()
        values = values[np.isfinite(values) & (values > 0)]
        if values.size:
            ax.hist(values, bins=20, alpha=0.55, label=name, color=_detector_color(name))
    ax.legend(loc="best")
    ax.set_title("Epipolar residual of verified inliers")
    ax.set_xlabel("Mean Sampson distance [px]")
    ax.set_ylabel("Number of pairs")

    # 7 - degeneracy map
    ax = axes[2][0]
    for name in detectors:
        sub = pair_df[pair_df["detector"] == name]
        if not sub.empty:
            ax.scatter(sub["homography_dominance"], sub["f_inlier_ratio"] * 100.0,
                       s=22, alpha=0.75, label=name, color=_detector_color(name))
    ax.axvline(0.45, color=COLOR_THRESHOLD, linestyle="--", linewidth=1.3,
               label="degeneracy threshold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 100)
    ax.legend(loc="best", fontsize=7)
    ax.set_title("Degeneracy map (right of the line = planar / pure rotation)")
    ax.set_xlabel("Homography dominance  H / (H + F) inliers")
    ax.set_ylabel("RANSAC inlier ratio [%]")

    # 8 - runtime
    ax = axes[2][1]
    stages = ["detect_ms", "describe_ms"]
    x = np.arange(len(stages) + 1)
    width = 0.8 / max(1, len(detectors))
    for i, name in enumerate(detectors):
        det_sub = detector_df[detector_df["detector"] == name] if not detector_df.empty else pd.DataFrame()
        pair_sub = pair_df[pair_df["detector"] == name]
        values = [float(det_sub[s].mean()) if not det_sub.empty else 0.0 for s in stages]
        values.append(float(pair_sub["match_ms"].mean()) if not pair_sub.empty else 0.0)
        ax.bar(x + i * width - 0.4 + width / 2, values, width, label=name,
               color=_detector_color(name))
    ax.set_xticks(x)
    ax.set_xticklabels(["Detect", "Describe", "Match"])
    ax.set_yscale("log")
    ax.legend(loc="best")
    ax.set_title("Runtime per stage (log scale)")
    ax.set_xlabel("Stage")
    ax.set_ylabel("Mean time [ms]")

    # 9 - spatial coverage
    ax = axes[2][2]
    if not detector_df.empty:
        names, means = [], []
        for name in detectors:
            sub = detector_df[detector_df["detector"] == name]
            if not sub.empty:
                names.append(name)
                means.append(float(sub["spatial_coverage"].mean()) * 100.0)
        if names:
            ax.bar(names, means, color=[_detector_color(n) for n in names])
            ax.set_ylim(0, 100)
            for i, value in enumerate(means):
                ax.text(i, value, f" {value:.0f}%", ha="center", va="bottom", fontsize=8)
    ax.set_title("Keypoint spatial coverage (16x16 grid)")
    ax.set_xlabel("Detector")
    ax.set_ylabel("Grid cells containing a keypoint [%]")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save_figure(fig, path)


# ===========================================================================
# Stage 4 - sparse model statistics and camera trajectory
# ===========================================================================


def plot_sparse_overview(
    stats: Dict[str, Any],
    pose_df: pd.DataFrame,
    model: Any,
    path: Path,
    *,
    scene: str = "",
) -> Path:
    """Six-panel statistical summary of a sparse reconstruction.

    Panels
    ------
    1. Reprojection-error distribution, with mean and median marked
    2. Track-length distribution (how many views see each 3D point)
    3. Registered 3D points per image, in capture order
    4. Reprojection error vs track length (do redundant points triangulate better?)
    5. Baseline between consecutive cameras
    6. Registration and scene-extent summary
    """
    apply_plot_style()
    errors = np.array([p.error for p in model.points.values()], dtype=float)
    tracks = np.array([p.track_length for p in model.points.values()], dtype=float)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    fig.suptitle(
        f"Sparse reconstruction analysis{f' - {scene}' if scene else ''}\n"
        f"{stats['registered_images']}/{stats['total_input_images']} images registered "
        f"({stats['registration_rate_percent']:.1f} %) | "
        f"{stats['num_points3D']:,} points | {stats['num_observations']:,} observations",
        fontsize=12,
    )

    # 1 - reprojection error
    ax = axes[0][0]
    if errors.size:
        ax.hist(errors, bins=50, color=COLOR_ACCEPTED, edgecolor="white", linewidth=0.3)
        ax.axvline(float(errors.mean()), color=COLOR_THRESHOLD, linestyle="--", linewidth=1.4,
                   label=f"mean {errors.mean():.3f} px")
        ax.axvline(float(np.median(errors)), color=COLOR_REJECTED, linestyle=":", linewidth=1.4,
                   label=f"median {np.median(errors):.3f} px")
        ax.legend(loc="upper right")
    ax.set_title(f"Reprojection error (n={errors.size:,} points)")
    ax.set_xlabel("Reprojection error [px]")
    ax.set_ylabel("Number of 3D points")

    # 2 - track length
    ax = axes[0][1]
    if tracks.size:
        top = int(min(tracks.max(), 30))
        ax.hist(tracks, bins=np.arange(1.5, top + 1.5, 1.0), color=COLOR_ACCEPTED,
                edgecolor="white", linewidth=0.3)
        ax.axvline(float(tracks.mean()), color=COLOR_THRESHOLD, linestyle="--", linewidth=1.4,
                   label=f"mean {tracks.mean():.2f}")
        ax.legend(loc="upper right")
    ax.set_title("Track length (views observing each point)")
    ax.set_xlabel("Number of observing images")
    ax.set_ylabel("Number of 3D points")

    # 3 - points per image
    ax = axes[0][2]
    if not pose_df.empty:
        ax.bar(pose_df["order"], pose_df["num_points3D"], color=COLOR_ACCEPTED, width=0.9)
        ax.axhline(float(pose_df["num_points3D"].mean()), color=COLOR_THRESHOLD,
                   linestyle="--", linewidth=1.2,
                   label=f"mean {pose_df['num_points3D'].mean():.0f}")
        ax.legend(loc="best")
    ax.set_ylim(bottom=0)
    ax.set_title("Triangulated points visible per image")
    ax.set_xlabel("Image index (capture order)")
    ax.set_ylabel("Number of 3D points")

    # 4 - error vs track length
    ax = axes[1][0]
    if tracks.size:
        top = int(min(tracks.max(), 20))
        data, labels = [], []
        for length in range(2, top + 1):
            selection = errors[tracks == length]
            if selection.size >= 5:
                data.append(selection)
                labels.append(str(length))
        if data:
            bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)
            for patch in bp["boxes"]:
                patch.set_facecolor(COLOR_ACCEPTED)
                patch.set_alpha(0.7)
    ax.set_title("Reprojection error vs track length")
    ax.set_xlabel("Track length [number of observing images]")
    ax.set_ylabel("Reprojection error [px]")

    # 5 - baseline between consecutive cameras
    ax = axes[1][1]
    if "baseline_to_prev" in pose_df:
        baselines = pose_df["baseline_to_prev"].to_numpy()[1:]
        if baselines.size:
            ax.plot(np.arange(1, len(baselines) + 1), baselines, marker="o", markersize=3,
                    color=COLOR_NEUTRAL, linewidth=1.0)
            ax.axhline(float(np.median(baselines)), color=COLOR_THRESHOLD, linestyle="--",
                       linewidth=1.2, label=f"median {np.median(baselines):.3f}")
            ax.legend(loc="best")
    ax.set_ylim(bottom=0)
    ax.set_title("Baseline between consecutive cameras")
    ax.set_xlabel("Image index (capture order)")
    ax.set_ylabel("Distance [arbitrary units]")

    # 6 - textual summary
    ax = axes[1][2]
    ax.axis("off")
    trajectory = stats.get("trajectory", {})
    bbox = stats.get("bounding_box", {})
    camera = (stats.get("cameras") or [{}])[0]
    lines = [
        f"Registered      : {stats['registered_images']} / {stats['total_input_images']} "
        f"({stats['registration_rate_percent']:.1f} %)",
        f"Cameras         : {stats['num_cameras']}",
        f"3D points       : {stats['num_points3D']:,}",
        f"Observations    : {stats['num_observations']:,}",
        f"Track length    : mean {stats['mean_track_length']:.2f}, "
        f"median {stats['median_track_length']:.0f}",
        f"Reproj. error   : mean {stats['mean_reprojection_error_px']:.3f} px, "
        f"median {stats['median_reprojection_error_px']:.3f} px",
        "",
        f"Camera model    : {camera.get('model', 'n/a')}",
        f"Focal length    : {camera.get('fx', 0):.1f} px",
        f"Principal point : ({camera.get('cx', 0):.1f}, {camera.get('cy', 0):.1f})",
        f"Horizontal FOV  : {camera.get('horizontal_fov_deg', 0):.1f} deg",
        "",
        f"Scene extent    : {np.round(bbox.get('robust_extent_p02_p98', [0, 0, 0]), 2).tolist()}",
        f"Trajectory len. : {trajectory.get('trajectory_length', 0):.3f}",
        f"Tortuosity      : {trajectory.get('tortuosity', 'n/a')}",
        f"Smoothness      : {trajectory.get('smoothness_index', 'n/a')}",
        "",
        "Lengths are in ARBITRARY units:",
        "monocular SfM has no metric scale.",
    ]
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=8.5,
            transform=ax.transAxes)
    ax.set_title("Reconstruction summary")

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return save_figure(fig, path)


def _frustum_lines(center: np.ndarray, R: np.ndarray, K: np.ndarray,
                   width: int, height: int, depth: float) -> list:
    """Line segments of a camera frustum in world coordinates.

    A pixel ``(u, v)`` back-projects to the camera-space ray ``K⁻¹[u, v, 1]``.
    Scaling it to ``depth`` and mapping to the world with ``X = Rᵀ·p + C`` gives
    the frustum corners - the same world-from-camera transform used for the
    camera centre, applied to a direction rather than the origin.
    """
    corners_px = np.array([[0, 0, 1], [width, 0, 1], [width, height, 1], [0, height, 1]],
                          dtype=float)
    K_inv = np.linalg.inv(K)
    rays = (K_inv @ corners_px.T).T
    rays = rays / np.linalg.norm(rays, axis=1, keepdims=True) * depth
    world = (R.T @ rays.T).T + center
    segments = [(center, world[i]) for i in range(4)]
    segments += [(world[i], world[(i + 1) % 4]) for i in range(4)]
    return segments


def plot_camera_trajectory(
    model: Any,
    pose_df: pd.DataFrame,
    path: Path,
    *,
    scene: str = "",
    max_points: int = 20000,
    draw_frustums: bool = True,
) -> Path:
    """3D camera trajectory over the sparse point cloud, plus orthogonal views.

    Camera positions are ``C = -Rᵀt``, never ``-t``. Frustums show where each
    camera was actually looking, which makes a mis-oriented pose obvious at a
    glance in a way that a bare position plot does not.
    """
    apply_plot_style()

    xyz = np.array([p.xyz for p in model.points.values()], dtype=float)
    rgb = np.array([p.rgb for p in model.points.values()], dtype=float) / 255.0
    errors = np.array([p.error for p in model.points.values()], dtype=float)

    # Trim the worst-triangulated 1% before plotting: a few wild outliers would
    # otherwise set the axis limits and squash the actual scene to a dot.
    if xyz.shape[0] > 100:
        keep = errors <= np.percentile(errors, 99)
        xyz, rgb = xyz[keep], rgb[keep]
    if xyz.shape[0] > max_points:
        picks = np.linspace(0, xyz.shape[0] - 1, max_points).astype(int)
        xyz, rgb = xyz[picks], rgb[picks]

    centers = pose_df[["center_x", "center_y", "center_z"]].to_numpy()

    fig = plt.figure(figsize=(15, 11))
    fig.suptitle(
        f"Camera trajectory and sparse point cloud{f' - {scene}' if scene else ''}\n"
        f"{len(centers)} registered cameras | {len(model.points):,} 3D points | "
        f"camera centres computed as C = -R\u1d40t",
        fontsize=12,
    )

    ax = fig.add_subplot(2, 2, 1, projection="3d")
    if xyz.size:
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=0.6, c=rgb, alpha=0.55,
                   linewidths=0, depthshade=False)
    if len(centers):
        ax.plot(centers[:, 0], centers[:, 1], centers[:, 2], color=COLOR_REJECTED,
                linewidth=1.6, label="camera path")
        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], s=26, c=COLOR_REJECTED,
                   marker="^", depthshade=False)
        ax.scatter(*centers[0], s=90, c="#1a9850", marker="o", label="first camera",
                   depthshade=False)
        ax.scatter(*centers[-1], s=90, c="#762a83", marker="s", label="last camera",
                   depthshade=False)

        if draw_frustums and model.cameras:
            camera = next(iter(model.cameras.values()))
            extent = float(np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0))) if xyz.size else 1.0
            depth = max(extent * 0.03, 1e-6)
            poses = sorted(model.images.values(), key=lambda p: p.name.lower())
            step = max(1, len(poses) // 12)  # ~12 frustums keeps the plot readable
            for pose in poses[::step]:
                for start, end in _frustum_lines(pose.center, pose.R, camera.K,
                                                 camera.width, camera.height, depth):
                    ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                            color="#1f78b4", linewidth=0.7, alpha=0.85)
        ax.legend(loc="upper left", fontsize=7)
    ax.set_title("3D view")
    ax.set_xlabel("X [arb. units]")
    ax.set_ylabel("Y [arb. units]")
    ax.set_zlabel("Z [arb. units]")
    _equal_aspect_3d(ax, xyz if xyz.size else centers)

    for index, (i, j, name) in enumerate([(0, 2, "Top-down (XZ)"), (0, 1, "Front (XY)")], start=2):
        ax = fig.add_subplot(2, 2, index)
        if xyz.size:
            ax.scatter(xyz[:, i], xyz[:, j], s=0.6, c=rgb, alpha=0.5, linewidths=0)
        if len(centers):
            ax.plot(centers[:, i], centers[:, j], color=COLOR_REJECTED, linewidth=1.5,
                    marker="^", markersize=4, label="camera path")
            ax.scatter(centers[0, i], centers[0, j], s=80, c="#1a9850", marker="o", zorder=5)
            ax.scatter(centers[-1, i], centers[-1, j], s=80, c="#762a83", marker="s", zorder=5)
            ax.legend(loc="best", fontsize=7)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(name)
        ax.set_xlabel(f"{'XYZ'[i]} [arb. units]")
        ax.set_ylabel(f"{'XYZ'[j]} [arb. units]")

    ax = fig.add_subplot(2, 2, 4)
    if "baseline_to_prev" in pose_df and len(pose_df) > 1:
        order = pose_df["order"].to_numpy()[1:]
        ax.plot(order, pose_df["baseline_to_prev"].to_numpy()[1:], marker="o", markersize=3,
                color=COLOR_ACCEPTED, label="baseline to previous")
        twin = ax.twinx()
        twin.plot(order, pose_df["rotation_to_prev_deg"].to_numpy()[1:], marker="s",
                  markersize=3, color=COLOR_THRESHOLD, label="rotation to previous")
        twin.set_ylabel("Relative rotation [deg]", color=COLOR_THRESHOLD)
        twin.tick_params(axis="y", labelcolor=COLOR_THRESHOLD)
        twin.set_ylim(bottom=0)
        twin.grid(False)
        ax.legend(loc="upper left", fontsize=7)
        twin.legend(loc="upper right", fontsize=7)
    ax.set_ylim(bottom=0)
    ax.set_title("Inter-camera motion along the capture")
    ax.set_xlabel("Image index (capture order)")
    ax.set_ylabel("Baseline [arb. units]", color=COLOR_ACCEPTED)
    ax.tick_params(axis="y", labelcolor=COLOR_ACCEPTED)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return save_figure(fig, path)


def _equal_aspect_3d(ax: Any, points: np.ndarray) -> None:
    """Force equal aspect on a 3D axis so geometry is not visually distorted.

    Matplotlib's 3D axes do not support ``set_aspect('equal')`` reliably, and an
    unequal aspect makes a straight camera path look curved - which would be a
    misleading figure in a report about geometry.
    """
    if points is None or len(points) == 0:
        return
    mins, maxs = points.min(axis=0), points.max(axis=0)
    centre = (mins + maxs) / 2.0
    radius = float((maxs - mins).max()) / 2.0
    if radius <= 0:
        return
    ax.set_xlim(centre[0] - radius, centre[0] + radius)
    ax.set_ylim(centre[1] - radius, centre[1] + radius)
    ax.set_zlim(centre[2] - radius, centre[2] + radius)


# ===========================================================================
# Stage 7 - custom two-view SfM
# ===========================================================================


def plot_two_view_sfm(
    result: Any,
    payload: Dict[str, Any],
    path: Path,
    *,
    scene: str = "",
) -> Path:
    """Six-panel summary of the from-scratch two-view reconstruction.

    Panels
    ------
    1. Triangulated point cloud in 3D, with both camera centres
    2. Top-down view showing the baseline and the depth spread
    3. Parallax-angle distribution (the conditioning of each depth estimate)
    4. Reprojection error before vs after bundle adjustment
    5. Reprojection error against parallax angle
    6. Numerical summary
    """
    apply_plot_style()
    points = payload["points_after"]
    colors = np.asarray(payload["colors"], dtype=float) / 255.0
    C1, C2 = payload["C1"], payload["C2_ba"]

    fig = plt.figure(figsize=(15, 9))
    ba = result.bundle_adjustment or {}
    fig.suptitle(
        f"Two-view SfM from scratch{f' - {scene}' if scene else ''}:  "
        f"{result.image_a}  ->  {result.image_b}\n"
        f"{result.final_points:,} points | rotation {result.rotation_deg:.2f} deg | "
        f"median parallax {result.median_parallax_deg:.2f} deg | "
        f"RMSE {ba.get('rmse_before', 0):.3f} -> {ba.get('rmse_after', 0):.3f} px",
        fontsize=12,
    )

    # 1 - 3D cloud
    ax = fig.add_subplot(2, 3, 1, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2.0, c=colors,
               alpha=0.75, linewidths=0, depthshade=False)
    for centre, colour, label in ((C1, "#1a9850", "camera 1"), (C2, "#762a83", "camera 2")):
        ax.scatter(*centre, s=110, c=colour, marker="^", label=label, depthshade=False)
    ax.plot([C1[0], C2[0]], [C1[1], C2[1]], [C1[2], C2[2]],
            color=COLOR_REJECTED, linewidth=2.0, label="baseline")
    ax.legend(loc="upper left", fontsize=7)
    ax.set_title("Triangulated point cloud")
    ax.set_xlabel("X [baselines]")
    ax.set_ylabel("Y [baselines]")
    ax.set_zlabel("Z [baselines]")
    _equal_aspect_3d(ax, np.vstack([points, C1[None, :], C2[None, :]]))

    # 2 - top-down
    ax = fig.add_subplot(2, 3, 2)
    ax.scatter(points[:, 0], points[:, 2], s=2.0, c=colors, alpha=0.7, linewidths=0)
    ax.scatter(C1[0], C1[2], s=110, c="#1a9850", marker="^", label="camera 1", zorder=5)
    ax.scatter(C2[0], C2[2], s=110, c="#762a83", marker="^", label="camera 2", zorder=5)
    ax.plot([C1[0], C2[0]], [C1[2], C2[2]], color=COLOR_REJECTED, linewidth=2.0, zorder=4)
    ax.legend(loc="best", fontsize=7)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("Top-down view (XZ)")
    ax.set_xlabel("X [baselines]")
    ax.set_ylabel("Z, depth [baselines]")

    # 3 - parallax
    ax = fig.add_subplot(2, 3, 3)
    parallax = payload["parallax"]
    ax.hist(parallax, bins=40, color=COLOR_ACCEPTED, edgecolor="white", linewidth=0.3)
    ax.axvline(float(np.median(parallax)), color=COLOR_THRESHOLD, linestyle="--",
               linewidth=1.4, label=f"median {np.median(parallax):.2f} deg")
    ax.axvline(2.0, color=COLOR_REJECTED, linestyle=":", linewidth=1.3,
               label="2 deg: poorly conditioned below")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_title("Parallax (triangulation) angle")
    ax.set_xlabel("Angle between viewing rays [deg]")
    ax.set_ylabel("Number of points")

    # 4 - BA before/after
    ax = fig.add_subplot(2, 3, 4)
    before, after = payload["errors_before"], payload["errors_after"]
    top = float(max(np.percentile(before, 99), np.percentile(after, 99)))
    bins = np.linspace(0, max(top, 1e-6), 40)
    ax.hist(before, bins=bins, alpha=0.6, color=COLOR_REJECTED,
            label=f"before BA (mean {before.mean():.3f} px)")
    ax.hist(after, bins=bins, alpha=0.6, color=COLOR_ACCEPTED,
            label=f"after BA (mean {after.mean():.3f} px)")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_title("Reprojection error: bundle adjustment effect")
    ax.set_xlabel("Reprojection error [px]")
    ax.set_ylabel("Number of points")

    # 5 - error vs parallax
    ax = fig.add_subplot(2, 3, 5)
    ax.scatter(parallax, after, s=6, alpha=0.5, color=COLOR_ACCEPTED)
    # Log scale only when the data actually spans more than a decade; forcing it
    # on a narrow range produces overlapping, unreadable tick labels.
    positive = parallax[parallax > 0]
    log_scale = positive.size > 0 and (positive.max() / positive.min()) > 10.0
    if log_scale:
        ax.set_xscale("log")
    ax.set_title("Reprojection error vs parallax angle")
    ax.set_xlabel(f"Parallax angle [deg{', log scale' if log_scale else ''}]")
    ax.set_ylabel("Reprojection error after BA [px]")

    # 6 - summary text
    ax = fig.add_subplot(2, 3, 6)
    ax.axis("off")
    lines = [
        f"Detector        : {result.detector}",
        f"Keypoints       : {result.keypoints_a:,} / {result.keypoints_b:,}",
        f"Ratio matches   : {result.ratio_matches:,}",
        f"Mutual matches  : {result.mutual_matches:,}",
        f"E inliers       : {result.essential_inliers:,} "
        f"({result.essential_inlier_ratio:.1%})",
        "",
        f"Triangulated    : {result.triangulated_raw:,}",
        f"  behind camera : -{result.rejected_cheirality:,}",
        f"  low parallax  : -{result.rejected_parallax:,}",
        f"  high reproj.  : -{result.rejected_reprojection:,}",
        f"  far depth     : -{result.rejected_depth:,}",
        f"  at infinity   : -{result.rejected_infinite:,}",
        f"  FINAL         : {result.final_points:,}",
        "",
        f"Rotation        : {result.rotation_deg:.3f} deg",
        f"Parallax median : {result.median_parallax_deg:.3f} deg",
    ]
    if result.dlt_vs_opencv_max_diff is not None:
        lines.append(f"DLT vs OpenCV   : {result.dlt_vs_opencv_max_diff:.2e} (relative)")
    if ba.get("ran"):
        lines += [
            "",
            "Bundle adjustment:",
            f"  RMSE before   : {ba['rmse_before']:.4f} px",
            f"  RMSE after    : {ba['rmse_after']:.4f} px",
            f"  improvement   : {ba['improvement_percent']:.2f} %",
            f"  rotation moved: {ba['rotation_change_deg']:.4f} deg",
            f"  iterations    : {ba['iterations']}",
        ]
    gt = result.ground_truth or {}
    if gt.get("compared"):
        lines += [
            "",
            "Ground truth:",
            f"  rotation err  : {gt['rotation_error_deg']:.4f} deg",
            f"  transl. dir   : {gt['translation_direction_error_deg']:.4f} deg",
        ]
    lines += ["", "Scale is arbitrary: distances are", "in units of one baseline."]
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace",
            fontsize=8, transform=ax.transAxes)
    ax.set_title("Pipeline summary")

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return save_figure(fig, path)


# ===========================================================================
# Stage 8 - experiment results
# ===========================================================================


def _errorbar_by_group(ax: Any, df: pd.DataFrame, x: str, y: str, color: str,
                       label: str = "") -> None:
    """Plot mean with min/max whiskers over repeats of each x value."""
    grouped = df.groupby(x)[y]
    xs = np.array(sorted(grouped.groups.keys()), dtype=float)
    means = np.array([grouped.get_group(v).mean() for v in xs])
    lows = np.array([grouped.get_group(v).min() for v in xs])
    highs = np.array([grouped.get_group(v).max() for v in xs])
    yerr = np.vstack([means - lows, highs - means])
    ax.errorbar(xs, means, yerr=yerr, marker="o", capsize=3, linewidth=1.5,
                color=color, label=label or None)


def plot_image_count_experiment(df: pd.DataFrame, out_dir: Path, scene: str = "") -> List[Path]:
    """Plots 1-5 of the brief: how reconstruction responds to image count.

    Error bars span the min/max across repeats. Where only one run exists there
    are no bars, and the caption should say so - a single run cannot separate a
    real effect from COLMAP's run-to-run variation.
    """
    apply_plot_style()
    out_dir = ensure_dir(Path(out_dir))
    paths: List[Path] = []
    sub = df[(df.get("experiment_type") == "image_count") & (df["success"].astype(bool))] \
        if "experiment_type" in df.columns else pd.DataFrame()
    if sub.empty:
        LOGGER.warning("No successful image_count experiments to plot")
        return paths

    suffix = f" - {scene}" if scene else ""

    specs = [
        ("registered_images", "Registered images", "Number of registered images",
         "01_images_vs_registered.png"),
        ("registration_rate", "Registration rate", "Registration rate [%]",
         "02_images_vs_registration_rate.png"),
        ("sparse_points", "Sparse point count", "Number of 3D points",
         "03_images_vs_sparse_points.png"),
        ("mean_reprojection_error", "Mean reprojection error", "Reprojection error [px]",
         "04_images_vs_reprojection_error.png"),
    ]
    for column, title, ylabel, filename in specs:
        if column not in sub.columns:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.6))
        _errorbar_by_group(ax, sub, "image_count", column, COLOR_ACCEPTED)
        if column == "registered_images":
            limit = float(sub["image_count"].max())
            ax.plot([0, limit], [0, limit], linestyle=":", color=COLOR_NEUTRAL,
                    linewidth=1.2, label="100 % registration")
            ax.legend(loc="best")
        if column == "registration_rate":
            ax.set_ylim(0, 105)
        else:
            ax.set_ylim(bottom=0)
        ax.set_title(f"{title} vs number of input images{suffix}")
        ax.set_xlabel("Number of input images")
        ax.set_ylabel(ylabel)
        paths.append(save_figure(fig, out_dir / filename))

    # 5 - runtime, broken down by stage
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    stages = [("feature_time", "feature extraction", "#4a9e5c"),
              ("matching_time", "matching", "#2a7ab0"),
              ("mapping_time", "mapping", "#d9a02c"),
              ("dense_time", "dense MVS", "#c1432c")]
    grouped = sub.groupby("image_count")
    xs = np.array(sorted(grouped.groups.keys()), dtype=float)
    bottom = np.zeros_like(xs)
    width = max(1.0, float(np.diff(xs).min()) * 0.6) if len(xs) > 1 else 1.0
    for column, label, colour in stages:
        if column not in sub.columns:
            continue
        values = np.array([grouped.get_group(v)[column].mean() for v in xs])
        if values.sum() <= 0:
            continue
        ax.bar(xs, values, bottom=bottom, width=width, label=label, color=colour)
        bottom += values
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")
    ax.set_title(f"Runtime vs number of input images{suffix}")
    ax.set_xlabel("Number of input images")
    ax.set_ylabel("Wall-clock time [s]")
    paths.append(save_figure(fig, out_dir / "05_images_vs_runtime.png"))
    return paths


def plot_matcher_comparison(df: pd.DataFrame, out_dir: Path, scene: str = "") -> List[Path]:
    """Plot 6 of the brief: exhaustive vs sequential matching."""
    apply_plot_style()
    out_dir = ensure_dir(Path(out_dir))
    sub = df[df.get("experiment_type") == "matcher"] if "experiment_type" in df.columns \
        else pd.DataFrame()
    if sub.empty:
        LOGGER.warning("No matcher experiments to plot")
        return []

    metrics = [
        ("registration_rate", "Registration rate [%]"),
        ("sparse_points", "Sparse 3D points"),
        ("mean_track_length", "Mean track length"),
        ("mean_reprojection_error", "Mean reprojection error [px]"),
        ("matching_time", "Matching time [s]"),
        ("total_time", "Total time [s]"),
    ]
    metrics = [(c, lbl) for c, lbl in metrics if c in sub.columns]
    matchers = sorted(sub["matcher"].unique())
    colours = {"exhaustive": "#2a7ab0", "sequential": "#c1432c",
               "vocab_tree": "#4a9e5c", "spatial": "#d9a02c"}

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    fig.suptitle(f"Matcher comparison{f' - {scene}' if scene else ''}  "
                 f"(identical images and settings; only the matcher differs)", fontsize=12)
    for ax, (column, label) in zip(axes.ravel(), metrics):
        means = [sub[sub["matcher"] == m][column].mean() for m in matchers]
        lows = [sub[sub["matcher"] == m][column].min() for m in matchers]
        highs = [sub[sub["matcher"] == m][column].max() for m in matchers]
        yerr = np.vstack([np.array(means) - np.array(lows),
                          np.array(highs) - np.array(means)])
        ax.bar(matchers, means, yerr=yerr, capsize=4,
               color=[colours.get(m, COLOR_NEUTRAL) for m in matchers])
        ax.set_ylim(bottom=0)
        ax.set_title(label)
        ax.set_xlabel("Matcher")
        ax.set_ylabel(label)
        for i, value in enumerate(means):
            ax.text(i, value, f" {value:.4g}", ha="center", va="bottom", fontsize=8)
    for ax in axes.ravel()[len(metrics):]:
        ax.axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return [save_figure(fig, out_dir / "06_matcher_comparison.png")]


def plot_quality_experiment(df: pd.DataFrame, out_dir: Path, scene: str = "") -> List[Path]:
    """Image-quality experiment: does discarding weak images help or hurt?"""
    apply_plot_style()
    out_dir = ensure_dir(Path(out_dir))
    sub = df[df.get("experiment_type") == "quality"] if "experiment_type" in df.columns \
        else pd.DataFrame()
    if sub.empty:
        return []

    arms = list(sub["quality_arm"].unique()) if "quality_arm" in sub.columns else []
    if not arms:
        return []
    # Keep the control first so every panel reads left-to-right as "baseline
    # then treatments".
    arms = (["all"] if "all" in arms else []) + [a for a in sorted(arms) if a != "all"]

    metrics = [("image_count", "Images used"),
               ("registered_images", "Registered images"),
               ("registration_rate", "Registration rate [%]"),
               ("sparse_points", "Sparse 3D points"),
               ("mean_track_length", "Mean track length"),
               ("mean_reprojection_error", "Mean reprojection error [px]")]
    metrics = [(c, lbl) for c, lbl in metrics if c in sub.columns]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    fig.suptitle(
        f"Image-quality filtering experiment{f' - {scene}' if scene else ''}\n"
        f"Does discarding low-quality images help or hurt? Measured, not assumed.",
        fontsize=12,
    )
    for ax, (column, label) in zip(axes.ravel(), metrics):
        means = [sub[sub["quality_arm"] == a][column].mean() for a in arms]
        colours = [COLOR_NEUTRAL if a == "all" else COLOR_ACCEPTED for a in arms]
        ax.bar(arms, means, color=colours)
        ax.set_ylim(bottom=0)
        ax.set_title(label)
        ax.set_xlabel("Filtering arm")
        ax.set_ylabel(label)
        for i, value in enumerate(means):
            ax.text(i, value, f" {value:.4g}", ha="center", va="bottom", fontsize=8)
    for ax in axes.ravel()[len(metrics):]:
        ax.axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return [save_figure(fig, out_dir / "07_quality_experiment.png")]


def plot_cross_scene(summary: pd.DataFrame, out_dir: Path) -> List[Path]:
    """Plot 13 of the brief: comparison across scenes (generalisation)."""
    apply_plot_style()
    out_dir = ensure_dir(Path(out_dir))
    if summary.empty:
        return []

    metrics = [("registration_rate", "Registration rate [%]"),
               ("sparse_points", "Sparse 3D points"),
               ("mean_track_length", "Mean track length"),
               ("mean_reprojection_error", "Mean reprojection error [px]")]
    metrics = [(c, lbl) for c, lbl in metrics if c in summary.columns]

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.0 * len(metrics), 4.4))
    axes = np.atleast_1d(axes)
    fig.suptitle("Cross-scene comparison: same pipeline, same settings, different scenes",
                 fontsize=12)
    scenes = summary["scene"].tolist()
    for ax, (column, label) in zip(axes, metrics):
        ax.bar(scenes, summary[column], color=COLOR_ACCEPTED)
        ax.set_ylim(bottom=0)
        if column == "registration_rate":
            ax.set_ylim(0, 105)
        ax.set_title(label)
        ax.set_xlabel("Scene")
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=20)
        for i, value in enumerate(summary[column]):
            ax.text(i, value, f" {value:.4g}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return [save_figure(fig, out_dir / "13_cross_scene_comparison.png")]


def plot_dense_comparison(df: pd.DataFrame, out_dir: Path, scene: str = "") -> List[Path]:
    """Plot 11 of the brief: dense point counts across conditions."""
    apply_plot_style()
    out_dir = ensure_dir(Path(out_dir))
    if "dense_points" not in df.columns:
        return []
    sub = df[(df["dense_points"] > 0) & (df["success"].astype(bool))]
    if sub.empty:
        return []

    fig, ax = plt.subplots(figsize=(8, 4.6))
    labels = [f"{r['experiment_type']}/{r['experiment_name']}" for _, r in sub.iterrows()]
    ax.bar(labels, sub["dense_points"], color=COLOR_ACCEPTED)
    ax.set_ylim(bottom=0)
    ax.set_title(f"Dense point count by condition{f' - {scene}' if scene else ''}")
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Fused dense points")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return [save_figure(fig, out_dir / "11_dense_point_comparison.png")]
