"""Typed, layered configuration system for the NTU SfM project.

Design goals
------------
1. **No hard-coded paths.** Every filesystem location is derived from
   ``paths.project_root`` / ``paths.data_root`` / ``paths.output_root``.
2. **Layered.** ``configs/default.yaml`` is the base; a scene/experiment YAML
   is deep-merged on top; ``--set a.b.c=value`` CLI overrides win last.
3. **Strict.** An unknown or mistyped key raises :class:`ConfigError` instead of
   being silently ignored. A typo in a YAML key must never cause a run to
   silently use a default value - that is how unreproducible experiments happen.
4. **Serialisable.** :meth:`ProjectConfig.to_dict` round-trips so that the exact
   configuration of every run is written next to its results.

The full schema for *all* pipeline stages is declared here up-front (features,
matching, mapper, dense, mesh, experiments), even though the early stages only
read part of it. That keeps the config file format stable so we never have to
rewrite earlier stages later.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import types
import typing
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import yaml

__all__ = [
    "ConfigError",
    "ProjectConfig",
    "load_config",
    "apply_overrides",
    "deep_merge",
]


class ConfigError(ValueError):
    """Raised for any malformed / unknown / out-of-range configuration value."""


# ===========================================================================
# Schema
# ===========================================================================


@dataclass
class PathsConfig:
    """Root locations. Relative values are resolved against ``project_root``.

    ``output_root`` is deliberately configurable: dense MVS output for a single
    scene can reach several GB, which you may not want inside a OneDrive-synced
    folder. Point it at another drive (e.g. ``D:/sfm_outputs``) if needed.
    """

    project_root: Path = Path(".")
    data_root: Path = Path("data")
    output_root: Path = Path("outputs")


@dataclass
class RuntimeConfig:
    """Process-level knobs: logging, parallelism, GPU selection, determinism."""

    log_level: str = "INFO"
    workers: int = 8
    use_gpu: bool = True
    #: COLMAP ``--*.gpu_index``. Comma-separated list, "-1" means all GPUs.
    gpu_index: str = "0"
    #: Threads handed to COLMAP (``--*.num_threads``); -1 = all cores.
    num_threads: int = -1
    seed: int = 42


@dataclass
class SceneConfig:
    """Identity and provenance of one captured scene."""

    name: str = "scene_01"
    #: ``images`` (a folder of photographs) or ``video`` (frames are extracted).
    source_type: str = "images"
    #: Path to the source video, relative to ``data_root`` unless absolute.
    video_file: Optional[Path] = None
    description: str = ""
    capture_notes: str = ""


@dataclass
class VideoConfig:
    """Frame-extraction policy for video inputs."""

    #: ``fps`` (resample to target_fps) | ``every_n`` | ``max_frames``.
    mode: str = "fps"
    target_fps: float = 2.0
    every_n: int = 10
    max_frames: int = 0
    start_time_s: float = 0.0
    #: 0.0 means "until the end of the video".
    end_time_s: float = 0.0
    output_format: str = "jpg"
    jpeg_quality: int = 95
    #: Reject a candidate frame whose normalised Laplacian variance is below
    #: this. 0 disables the in-extraction blur filter.
    min_blur_score: float = 0.0
    #: Skip a candidate frame whose dHash is within this Hamming distance of the
    #: last *kept* frame. 0 disables consecutive-similarity rejection.
    min_hamming_distance: int = 0
    #: Downscale the long side of extracted frames. 0 = keep native resolution.
    resize_max_dim: int = 0


@dataclass
class BlurFilterConfig:
    """Sharpness rejection rule.

    Absolute Laplacian-variance thresholds are notoriously dataset-dependent,
    so two data-driven modes are provided as well:

    * ``percentile`` - reject the sharpest-metric bottom *p* percent.
    * ``mad``        - reject below ``median - k * MAD`` (robust to outliers).
    """

    threshold_mode: str = "absolute"  # absolute | percentile | mad
    min_score: float = 60.0
    percentile: float = 10.0
    mad_k: float = 2.5


@dataclass
class BrightnessFilterConfig:
    min: float = 25.0
    max: float = 235.0


@dataclass
class ContrastFilterConfig:
    min: float = 15.0


@dataclass
class ResolutionFilterConfig:
    min_width: int = 640
    min_height: int = 480
    #: If true, images whose (w,h) differs from the dataset mode are rejected.
    require_consistent: bool = False


@dataclass
class DuplicateFilterConfig:
    hash_size: int = 8
    max_hamming_distance: int = 3
    #: Which member of a near-duplicate group to keep: ``sharpest`` | ``first``.
    keep: str = "sharpest"


@dataclass
class ClippingFilterConfig:
    """Fraction of pixels allowed to be crushed to black / blown to white."""

    shadow_frac_max: float = 0.60
    highlight_frac_max: float = 0.40


@dataclass
class QualityConfig:
    """Image-quality diagnostics and (optional) rejection policy.

    ``mode: report_only`` computes every metric and writes the CSV but accepts
    every readable image. This is the default *on purpose*: whether discarding
    low-quality images helps or hurts reconstruction is an experiment to run
    (Section 13 of the project brief), not an assumption to bake in.
    """

    mode: str = "report_only"  # report_only | filter
    #: Sharpness metrics are computed after downscaling the long side to this
    #: value so that images of different resolutions are comparable.
    blur_eval_long_side: int = 1024
    #: Active rejection rules when ``mode == "filter"``. Any subset of
    #: [blur, brightness, contrast, resolution, duplicate, clipping].
    filters: List[str] = field(default_factory=list)
    blur: BlurFilterConfig = field(default_factory=BlurFilterConfig)
    brightness: BrightnessFilterConfig = field(default_factory=BrightnessFilterConfig)
    contrast: ContrastFilterConfig = field(default_factory=ContrastFilterConfig)
    resolution: ResolutionFilterConfig = field(default_factory=ResolutionFilterConfig)
    duplicate: DuplicateFilterConfig = field(default_factory=DuplicateFilterConfig)
    clipping: ClippingFilterConfig = field(default_factory=ClippingFilterConfig)


@dataclass
class ProcessingConfig:
    """How accepted images are materialised into ``processed_images/``."""

    #: Downscale long side. 0 = keep native resolution.
    resize_max_dim: int = 0
    jpeg_quality: int = 95
    #: Bake EXIF orientation into pixels so OpenCV and COLMAP agree.
    apply_exif_transpose: bool = True
    #: ``auto`` copies bytes verbatim when no resize/rotation is needed (this
    #: preserves the EXIF focal length that COLMAP uses as a prior), and
    #: re-encodes otherwise. ``copy`` / ``reencode`` force one behaviour.
    copy_mode: str = "auto"
    #: Cache per-image metrics keyed by (size, mtime, parameters).
    cache: bool = True


@dataclass
class CameraConfig:
    """COLMAP camera model and intrinsic priors."""

    #: SIMPLE_PINHOLE | PINHOLE | SIMPLE_RADIAL | RADIAL | OPENCV | FULL_OPENCV
    model: str = "SIMPLE_RADIAL"
    #: One shared camera for the whole scene (correct for a single phone).
    single_camera: bool = True
    single_camera_per_folder: bool = False
    use_exif_focal_prior: bool = True
    #: Fallback f = factor * max(width, height) when EXIF has no focal length.
    default_focal_factor: float = 1.2


@dataclass
class FeatureConfig:
    """COLMAP SIFT feature extraction parameters."""

    max_image_size: int = 3200
    max_num_features: int = 8192
    first_octave: int = -1
    estimate_affine_shape: bool = False
    domain_size_pooling: bool = False
    peak_threshold: float = 0.0066666666666666671
    edge_threshold: float = 10.0


@dataclass
class SiftParams:
    """OpenCV ``cv2.SIFT_create`` parameters (independent of COLMAP's SIFT)."""

    n_features: int = 4000  # 0 = unlimited; capped for tractable brute-force matching
    n_octave_layers: int = 3
    contrast_threshold: float = 0.04
    edge_threshold: float = 10.0
    sigma: float = 1.6


@dataclass
class OrbParams:
    n_features: int = 5000
    scale_factor: float = 1.2
    n_levels: int = 8
    fast_threshold: int = 20


@dataclass
class AkazeParams:
    threshold: float = 0.001
    n_octaves: int = 4
    n_octave_layers: int = 4


@dataclass
class DescriptorMatchingConfig:
    """Descriptor matching policy for the OpenCV analysis stage."""

    #: ``bf`` is exact and deterministic; ``flann`` is faster but approximate.
    method: str = "bf"
    #: Lowe's ratio test threshold (d1/d2 < ratio).
    ratio: float = 0.75
    #: Additionally require mutual nearest-neighbour agreement in both directions.
    mutual_check: bool = True


@dataclass
class RansacConfig:
    """Robust estimation settings shared by F, E and H."""

    #: ransac | lmeds | magsac | usac_default | usac_accurate
    method: str = "magsac"
    threshold_px: float = 1.5
    confidence: float = 0.9999
    max_iters: int = 10000


@dataclass
class PairSelectionConfig:
    """Which image pairs the feature analysis is run on."""

    #: sequential | strided | explicit
    mode: str = "strided"
    #: Index gaps to sample, e.g. [1, 2, 5, 10] probes widening baselines.
    strides: List[int] = field(default_factory=lambda: [1, 2, 5, 10])
    max_pairs: int = 40
    #: ``["a.jpg:b.jpg", ...]`` when mode == "explicit".
    explicit: List[str] = field(default_factory=list)


@dataclass
class FeatureVisualizationConfig:
    max_figures: int = 6
    max_drawn_matches: int = 120
    num_epipolar_lines: int = 12


@dataclass
class FeatureAnalysisConfig:
    """Stage 2 - independent OpenCV feature/geometry analysis.

    This runs *alongside* COLMAP rather than feeding it. Its purpose is to make
    the matching behaviour of a scene measurable and visible: how many features
    survive the ratio test, how many survive geometric verification, and whether
    the two-view geometry is degenerate.
    """

    detectors: List[str] = field(default_factory=lambda: ["SIFT", "ORB", "AKAZE"])
    #: Analysis-time downscale. Keeps brute-force matching tractable.
    max_image_size: int = 1600
    sift: SiftParams = field(default_factory=SiftParams)
    orb: OrbParams = field(default_factory=OrbParams)
    akaze: AkazeParams = field(default_factory=AkazeParams)
    matching: DescriptorMatchingConfig = field(default_factory=DescriptorMatchingConfig)
    ransac: RansacConfig = field(default_factory=RansacConfig)
    pairs: PairSelectionConfig = field(default_factory=PairSelectionConfig)
    visualize: FeatureVisualizationConfig = field(default_factory=FeatureVisualizationConfig)
    #: Flag a pair as homography-dominant (planar / pure-rotation) above this.
    #: See Torr & Zisserman model selection; ORB-SLAM uses a similar test.
    homography_dominance_threshold: float = 0.45


@dataclass
class TriangulationConfig:
    """Filters applied to triangulated points before they are reported."""

    #: Reject points whose two viewing rays meet at less than this angle.
    #: Low-parallax points have very poorly conditioned depth: a sub-pixel
    #: measurement error translates into an enormous depth error.
    min_parallax_deg: float = 1.0
    max_reprojection_error_px: float = 4.0
    #: Drop the far tail of the depth distribution (near-infinity points).
    max_depth_percentile: float = 98.0


@dataclass
class BundleAdjustmentConfig:
    """scipy.optimize.least_squares bundle-adjustment demonstration."""

    enabled: bool = True
    #: linear | soft_l1 | huber | cauchy. Robust losses down-weight outliers
    #: instead of letting a few bad correspondences dominate the sum of squares.
    loss: str = "huber"
    #: Residual scale at which the robust loss starts down-weighting, in pixels.
    f_scale: float = 1.0
    max_iterations: int = 200
    #: Also refine the focal length. Off by default: with two views it is very
    #: weakly constrained and tends to absorb other errors.
    refine_focal_length: bool = False
    verbose: bool = False


@dataclass
class TwoViewSfMConfig:
    """Stage 7 - the from-scratch two-view reconstruction, for comparison."""

    detector: str = "SIFT"
    max_image_size: int = 1600
    ratio: float = 0.75
    mutual_check: bool = True
    ransac: "RansacConfig" = field(default_factory=lambda: RansacConfig(threshold_px=1.0))
    triangulation: TriangulationConfig = field(default_factory=TriangulationConfig)
    bundle_adjustment: BundleAdjustmentConfig = field(default_factory=BundleAdjustmentConfig)
    export_ply: bool = True


@dataclass
class SequentialMatchingConfig:
    overlap: int = 10
    quadratic_overlap: bool = True
    loop_detection: bool = False
    loop_detection_period: int = 10
    loop_detection_num_images: int = 50


@dataclass
class MatchingConfig:
    """COLMAP feature matching parameters."""

    method: str = "exhaustive"  # exhaustive | sequential | vocab_tree | spatial
    max_ratio: float = 0.8
    max_distance: float = 0.7
    cross_check: bool = True
    max_num_matches: int = 32768
    guided_matching: bool = False
    sequential: SequentialMatchingConfig = field(default_factory=SequentialMatchingConfig)
    vocab_tree_path: Optional[Path] = None


@dataclass
class MapperConfig:
    """COLMAP incremental mapper parameters."""

    min_num_matches: int = 15
    init_min_num_inliers: int = 100
    abs_pose_min_num_inliers: int = 30
    ba_refine_focal_length: bool = True
    ba_refine_principal_point: bool = False
    ba_refine_extra_params: bool = True
    ba_global_max_num_iterations: int = 50
    multiple_models: bool = True
    max_num_models: int = 50
    extract_colors: bool = True


@dataclass
class FusionConfig:
    min_num_pixels: int = 5
    max_reproj_error: float = 2.0
    max_depth_error: float = 0.01
    max_normal_error: float = 10.0


@dataclass
class DenseConfig:
    """COLMAP MVS (image_undistorter / patch_match_stereo / stereo_fusion)."""

    enabled: bool = False
    max_image_size: int = 1600
    #: Correlation window half-width. Larger = smoother and far more robust on
    #: weakly textured surfaces (concrete, painted walls), at the cost of
    #: blurring fine detail and runtime growing roughly with the window area.
    window_radius: int = 5
    window_step: int = 1
    num_samples: int = 15
    num_iterations: int = 5
    geom_consistency: bool = True
    filter: bool = True
    cache_size_gb: float = 4.0

    # --- filtering: the knobs that decide how clean the cloud is ----------
    #: Minimum normalised cross-correlation to keep a depth estimate. Raising
    #: it is the single most effective way to remove speckle noise.
    filter_min_ncc: float = 0.1
    #: Minimum triangulation angle in degrees. Low-parallax depths are
    #: numerically unstable - this is what removes the "smeared sky" artefacts.
    filter_min_triangulation_angle: float = 3.0
    #: How many source views must agree on a depth before it is kept.
    filter_min_num_consistent: int = 2
    filter_geom_consistency_max_cost: float = 1.0
    geom_consistency_regularizer: float = 0.3
    geom_consistency_max_cost: float = 3.0
    #: Source views considered per reference image during undistortion. More
    #: views means more redundancy for the consistency check.
    num_src_images: int = 20

    fusion: FusionConfig = field(default_factory=FusionConfig)


@dataclass
class MeshConfig:
    enabled: bool = False
    method: str = "poisson"  # poisson | delaunay
    poisson_depth: int = 11
    poisson_trim: float = 7.0


@dataclass
class ExperimentsConfig:
    subdir: str = "experiments"
    #: 0 means "all available images".
    image_counts: List[int] = field(default_factory=lambda: [25, 50, 75, 100, 125, 0])
    matchers: List[str] = field(default_factory=lambda: ["exhaustive", "sequential"])
    #: Arms of the image-quality experiment: ``all``, ``blur_pNN``, ``sharpest_NN``.
    quality_arms: List[str] = field(default_factory=lambda: ["all", "blur_p15", "blur_p30"])
    repeat: int = 1
    #: Skip an experiment whose completion marker already exists (unless --force).
    skip_completed: bool = True
    run_dense: bool = False


@dataclass
class ProjectConfig:
    """Root configuration object handed to every stage of the pipeline."""

    project_name: str = "ntu_sfm"
    paths: PathsConfig = field(default_factory=PathsConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    feature_analysis: FeatureAnalysisConfig = field(default_factory=FeatureAnalysisConfig)
    two_view: TwoViewSfMConfig = field(default_factory=TwoViewSfMConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    mapper: MapperConfig = field(default_factory=MapperConfig)
    dense: DenseConfig = field(default_factory=DenseConfig)
    mesh: MeshConfig = field(default_factory=MeshConfig)
    experiments: ExperimentsConfig = field(default_factory=ExperimentsConfig)

    # -- derived paths ------------------------------------------------------

    @property
    def scene_data_dir(self) -> Path:
        return self.paths.data_root / self.scene.name

    @property
    def raw_images_dir(self) -> Path:
        return self.scene_data_dir / "raw_images"

    @property
    def processed_images_dir(self) -> Path:
        return self.scene_data_dir / "processed_images"

    @property
    def metadata_dir(self) -> Path:
        return self.scene_data_dir / "metadata"

    @property
    def review_dir(self) -> Path:
        return self.scene_data_dir / "review"

    @property
    def scene_output_dir(self) -> Path:
        return self.paths.output_root / self.scene.name

    @property
    def logs_dir(self) -> Path:
        return self.scene_output_dir / "logs"

    @property
    def metrics_dir(self) -> Path:
        return self.scene_output_dir / "metrics"

    @property
    def visualizations_dir(self) -> Path:
        return self.scene_output_dir / "visualizations"

    @property
    def database_dir(self) -> Path:
        return self.scene_output_dir / "database"

    @property
    def sparse_dir(self) -> Path:
        return self.scene_output_dir / "sparse"

    @property
    def dense_dir(self) -> Path:
        return self.scene_output_dir / "dense"

    @property
    def meshes_dir(self) -> Path:
        return self.scene_output_dir / "meshes"

    @property
    def cache_dir(self) -> Path:
        return self.scene_output_dir / "cache"

    @property
    def feature_viz_dir(self) -> Path:
        return self.visualizations_dir / "features"

    @property
    def video_path(self) -> Optional[Path]:
        """Absolute path to the source video, or ``None``."""
        if self.scene.video_file is None:
            return None
        p = Path(self.scene.video_file)
        return p if p.is_absolute() else (self.paths.data_root / p)

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Plain-Python view suitable for JSON/YAML dumping."""
        return _to_plain(self)

    def dump_yaml(self, path: Path) -> Path:
        """Write the fully-resolved configuration next to a run's results."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False, allow_unicode=True)
        return path

    def dump_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


# ===========================================================================
# dict -> dataclass conversion (strict)
# ===========================================================================


def _is_optional(tp: Any) -> bool:
    origin = typing.get_origin(tp)
    return origin in (Union, types.UnionType) and type(None) in typing.get_args(tp)


def _strip_optional(tp: Any) -> Any:
    args = [a for a in typing.get_args(tp) if a is not type(None)]  # noqa: E721
    if len(args) != 1:
        raise ConfigError(f"Unsupported union type in schema: {tp!r}")
    return args[0]


def _coerce(value: Any, tp: Any, ctx: str) -> Any:
    """Convert a raw YAML value to the declared dataclass field type."""
    if _is_optional(tp):
        if value is None:
            return None
        tp = _strip_optional(tp)

    origin = typing.get_origin(tp)

    if is_dataclass(tp):
        if not isinstance(value, dict):
            raise ConfigError(f"{ctx}: expected a mapping, got {type(value).__name__}")
        return _from_dict(tp, value, ctx)

    if origin in (list, List):
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{ctx}: expected a list, got {type(value).__name__}")
        (item_tp,) = typing.get_args(tp) or (Any,)
        return [_coerce(v, item_tp, f"{ctx}[{i}]") for i, v in enumerate(value)]

    if origin in (dict, Dict):
        if not isinstance(value, dict):
            raise ConfigError(f"{ctx}: expected a mapping, got {type(value).__name__}")
        return dict(value)

    if tp is Path:
        return Path(str(value))
    if tp is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in {"true", "false", "yes", "no", "1", "0"}:
            return value.lower() in {"true", "yes", "1"}
        raise ConfigError(f"{ctx}: expected a boolean, got {value!r}")
    if tp is int:
        if isinstance(value, bool):
            raise ConfigError(f"{ctx}: expected an int, got a bool")
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{ctx}: expected an int, got {value!r}") from exc
    if tp is float:
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{ctx}: expected a float, got {value!r}") from exc
    if tp is str:
        if isinstance(value, (dict, list)):
            raise ConfigError(f"{ctx}: expected a string, got {type(value).__name__}")
        return str(value)
    if tp is Any:
        return value

    raise ConfigError(f"{ctx}: unsupported schema type {tp!r}")


def _from_dict(cls: Any, data: Dict[str, Any], ctx: str = "") -> Any:
    hints = typing.get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(
            f"{ctx or cls.__name__}: unknown configuration key(s) "
            f"{sorted(unknown)}. Valid keys: {sorted(known)}"
        )
    kwargs: Dict[str, Any] = {}
    for name, raw in data.items():
        sub_ctx = f"{ctx}.{name}" if ctx else name
        kwargs[name] = _coerce(raw, hints[name], sub_ctx)
    return cls(**kwargs)


def _to_plain(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    return obj


# ===========================================================================
# Merging / overrides / validation
# ===========================================================================


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (``base`` is not mutated)."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _parse_scalar(text: str) -> Any:
    """Interpret a CLI override value using YAML scalar rules."""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def apply_overrides(data: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """Apply ``dotted.key=value`` strings from the command line.

    Example::

        apply_overrides(cfg, ["quality.mode=filter", "quality.filters=[blur]"])
    """
    out = copy.deepcopy(data)
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"Malformed --set override {item!r}; expected key=value")
        dotted, raw = item.split("=", 1)
        keys = [k for k in dotted.strip().split(".") if k]
        if not keys:
            raise ConfigError(f"Malformed --set override {item!r}; empty key")
        node = out
        for key in keys[:-1]:
            nxt = node.get(key)
            if not isinstance(nxt, dict):
                nxt = {}
                node[key] = nxt
            node = nxt
        node[keys[-1]] = _parse_scalar(raw)
    return out


_VALID_SOURCE_TYPES = {"images", "video"}
_VALID_QUALITY_MODES = {"report_only", "filter"}
_VALID_FILTERS = {"blur", "brightness", "contrast", "resolution", "duplicate", "clipping"}
_VALID_BLUR_MODES = {"absolute", "percentile", "mad"}
_VALID_VIDEO_MODES = {"fps", "every_n", "max_frames"}
_VALID_COPY_MODES = {"auto", "copy", "reencode"}
_VALID_CAMERA_MODELS = {
    "SIMPLE_PINHOLE",
    "PINHOLE",
    "SIMPLE_RADIAL",
    "RADIAL",
    "OPENCV",
    "OPENCV_FISHEYE",
    "FULL_OPENCV",
    "FOV",
    "SIMPLE_RADIAL_FISHEYE",
    "RADIAL_FISHEYE",
    "THIN_PRISM_FISHEYE",
}
_VALID_MATCHERS = {"exhaustive", "sequential", "vocab_tree", "spatial"}
_VALID_DETECTORS = {"SIFT", "ORB", "AKAZE", "BRISK", "KAZE"}
_VALID_RANSAC_METHODS = {"ransac", "lmeds", "magsac", "usac_default", "usac_accurate"}
# open3d_poisson was removed: Poisson surface reconstruction from a library
# is a reconstruction step, and the deliverable implements TSDF fusion plus
# marching cubes itself in sfm_core/surface.py.
_VALID_MESH_METHODS = {"poisson", "delaunay"}


def _validate(cfg: ProjectConfig) -> None:
    """Fail loudly on semantically invalid combinations."""
    errors: List[str] = []

    if cfg.scene.source_type not in _VALID_SOURCE_TYPES:
        errors.append(f"scene.source_type must be one of {sorted(_VALID_SOURCE_TYPES)}")
    if cfg.scene.source_type == "video" and cfg.scene.video_file is None:
        errors.append("scene.source_type is 'video' but scene.video_file is not set")
    if not cfg.scene.name.strip():
        errors.append("scene.name must not be empty")

    if cfg.video.mode not in _VALID_VIDEO_MODES:
        errors.append(f"video.mode must be one of {sorted(_VALID_VIDEO_MODES)}")
    if cfg.video.target_fps <= 0:
        errors.append("video.target_fps must be > 0")
    if cfg.video.every_n < 1:
        errors.append("video.every_n must be >= 1")
    if not 1 <= cfg.video.jpeg_quality <= 100:
        errors.append("video.jpeg_quality must be in [1, 100]")
    if cfg.video.end_time_s and cfg.video.end_time_s <= cfg.video.start_time_s:
        errors.append("video.end_time_s must be greater than video.start_time_s")

    if cfg.quality.mode not in _VALID_QUALITY_MODES:
        errors.append(f"quality.mode must be one of {sorted(_VALID_QUALITY_MODES)}")
    bad_filters = set(cfg.quality.filters) - _VALID_FILTERS
    if bad_filters:
        errors.append(f"quality.filters has unknown entries {sorted(bad_filters)}")
    if cfg.quality.blur.threshold_mode not in _VALID_BLUR_MODES:
        errors.append(f"quality.blur.threshold_mode must be one of {sorted(_VALID_BLUR_MODES)}")
    if not 0.0 <= cfg.quality.blur.percentile < 100.0:
        errors.append("quality.blur.percentile must be in [0, 100)")
    if cfg.quality.blur_eval_long_side < 128:
        errors.append("quality.blur_eval_long_side must be >= 128")
    if not 1 <= cfg.quality.duplicate.hash_size <= 16:
        errors.append("quality.duplicate.hash_size must be in [1, 16]")
    max_bits = cfg.quality.duplicate.hash_size ** 2
    if not 0 <= cfg.quality.duplicate.max_hamming_distance <= max_bits:
        errors.append(f"quality.duplicate.max_hamming_distance must be in [0, {max_bits}]")
    if cfg.quality.duplicate.keep not in {"sharpest", "first"}:
        errors.append("quality.duplicate.keep must be 'sharpest' or 'first'")

    if cfg.processing.copy_mode not in _VALID_COPY_MODES:
        errors.append(f"processing.copy_mode must be one of {sorted(_VALID_COPY_MODES)}")
    if cfg.processing.resize_max_dim and cfg.processing.resize_max_dim < 256:
        errors.append("processing.resize_max_dim must be 0 (disabled) or >= 256")

    if cfg.camera.model not in _VALID_CAMERA_MODELS:
        errors.append(f"camera.model must be one of {sorted(_VALID_CAMERA_MODELS)}")
    if cfg.camera.default_focal_factor <= 0:
        errors.append("camera.default_focal_factor must be > 0")

    fa = cfg.feature_analysis
    bad_detectors = {d.upper() for d in fa.detectors} - _VALID_DETECTORS
    if bad_detectors:
        errors.append(f"feature_analysis.detectors has unknown entries {sorted(bad_detectors)}")
    if not fa.detectors:
        errors.append("feature_analysis.detectors must not be empty")
    if fa.matching.method not in {"bf", "flann"}:
        errors.append("feature_analysis.matching.method must be 'bf' or 'flann'")
    if not 0.0 < fa.matching.ratio <= 1.0:
        errors.append("feature_analysis.matching.ratio must be in (0, 1]")
    if fa.ransac.method not in _VALID_RANSAC_METHODS:
        errors.append(f"feature_analysis.ransac.method must be one of {sorted(_VALID_RANSAC_METHODS)}")
    if fa.ransac.threshold_px <= 0:
        errors.append("feature_analysis.ransac.threshold_px must be > 0")
    if not 0.0 < fa.ransac.confidence < 1.0:
        errors.append("feature_analysis.ransac.confidence must be in (0, 1)")
    if fa.pairs.mode not in {"sequential", "strided", "explicit"}:
        errors.append("feature_analysis.pairs.mode must be sequential | strided | explicit")
    if fa.pairs.mode == "explicit" and not fa.pairs.explicit:
        errors.append("feature_analysis.pairs.mode is 'explicit' but pairs.explicit is empty")
    if any(s < 1 for s in fa.pairs.strides):
        errors.append("feature_analysis.pairs.strides entries must be >= 1")
    if fa.pairs.max_pairs < 1:
        errors.append("feature_analysis.pairs.max_pairs must be >= 1")
    if fa.max_image_size and fa.max_image_size < 256:
        errors.append("feature_analysis.max_image_size must be 0 (native) or >= 256")

    tv = cfg.two_view
    if tv.detector.upper() not in _VALID_DETECTORS:
        errors.append(f"two_view.detector must be one of {sorted(_VALID_DETECTORS)}")
    if not 0.0 < tv.ratio <= 1.0:
        errors.append("two_view.ratio must be in (0, 1]")
    if tv.ransac.method not in _VALID_RANSAC_METHODS:
        errors.append(f"two_view.ransac.method must be one of {sorted(_VALID_RANSAC_METHODS)}")
    if tv.bundle_adjustment.loss not in {"linear", "soft_l1", "huber", "cauchy", "arctan"}:
        errors.append(
            "two_view.bundle_adjustment.loss must be one of "
            "['linear', 'soft_l1', 'huber', 'cauchy', 'arctan']"
        )
    if tv.bundle_adjustment.max_iterations < 1:
        errors.append("two_view.bundle_adjustment.max_iterations must be >= 1")
    if tv.triangulation.min_parallax_deg < 0:
        errors.append("two_view.triangulation.min_parallax_deg must be >= 0")
    if not 0 < tv.triangulation.max_depth_percentile <= 100:
        errors.append("two_view.triangulation.max_depth_percentile must be in (0, 100]")

    if cfg.matching.method not in _VALID_MATCHERS:
        errors.append(f"matching.method must be one of {sorted(_VALID_MATCHERS)}")
    if cfg.matching.method == "vocab_tree" and cfg.matching.vocab_tree_path is None:
        errors.append("matching.method is 'vocab_tree' but matching.vocab_tree_path is unset")
    bad_matchers = set(cfg.experiments.matchers) - _VALID_MATCHERS
    if bad_matchers:
        errors.append(f"experiments.matchers has unknown entries {sorted(bad_matchers)}")

    if cfg.mesh.method not in _VALID_MESH_METHODS:
        errors.append(f"mesh.method must be one of {sorted(_VALID_MESH_METHODS)}")

    if cfg.runtime.workers < 1:
        errors.append("runtime.workers must be >= 1")
    if cfg.runtime.log_level.upper() not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        errors.append("runtime.log_level must be a valid logging level name")

    if any(c < 0 for c in cfg.experiments.image_counts):
        errors.append("experiments.image_counts entries must be >= 0 (0 means 'all')")

    if errors:
        raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(errors))


def _resolve_paths(cfg: ProjectConfig, project_root: Path) -> None:
    """Make every root path absolute, relative to the project root."""
    cfg.paths.project_root = project_root.resolve()

    def _abs(p: Path) -> Path:
        p = Path(p)
        return p.resolve() if p.is_absolute() else (cfg.paths.project_root / p).resolve()

    cfg.paths.data_root = _abs(cfg.paths.data_root)
    cfg.paths.output_root = _abs(cfg.paths.output_root)
    if cfg.matching.vocab_tree_path is not None:
        cfg.matching.vocab_tree_path = _abs(cfg.matching.vocab_tree_path)
    cfg.runtime.log_level = cfg.runtime.log_level.upper()


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Configuration file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse YAML file {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level of a config file must be a mapping")
    return data


def find_project_root(start: Optional[Path] = None) -> Path:
    """Walk upwards from ``start`` looking for the ``configs/default.yaml`` marker."""
    here = (start or Path(__file__)).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "configs" / "default.yaml").is_file():
            return candidate
    # Fall back to the package parent (src/ -> project root).
    return Path(__file__).resolve().parents[1]


def load_config(
    config_path: Optional[Union[Path, Sequence[Path]]] = None,
    *,
    overrides: Optional[List[str]] = None,
    project_root: Optional[Path] = None,
    default_config: Optional[Path] = None,
) -> ProjectConfig:
    """Build a validated :class:`ProjectConfig`.

    Parameters
    ----------
    config_path:
        One scene/experiment YAML, or a sequence of them, deep-merged on top of
        the defaults in the order given (later files win). ``None`` uses the
        defaults alone.
    overrides:
        ``["section.key=value", ...]`` strings from ``--set``, applied last.
    project_root:
        Explicit project root; auto-detected from this file's location if omitted.
    default_config:
        Override the base ``configs/default.yaml`` location (used by tests).

    Raises
    ------
    ConfigError
        For a missing file, a YAML syntax error, an unknown key, a wrongly typed
        value, or a semantically invalid combination.
    """
    root = (project_root or find_project_root()).resolve()
    base_path = default_config or (root / "configs" / "default.yaml")
    data = _read_yaml(Path(base_path))

    if config_path is not None:
        layers = [config_path] if isinstance(config_path, (str, Path)) else list(config_path)
        for layer in layers:
            scene_path = Path(layer)
            if not scene_path.is_absolute() and not scene_path.is_file():
                scene_path = root / scene_path
            data = deep_merge(data, _read_yaml(scene_path))

    if overrides:
        data = apply_overrides(data, list(overrides))

    cfg: ProjectConfig = _from_dict(ProjectConfig, data)
    _resolve_paths(cfg, root)
    _validate(cfg)
    return cfg
