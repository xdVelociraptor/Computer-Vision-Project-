"""Reconstructing a capture that does not form a single connected model.

Incremental SfM grows one model from one seed. When a capture contains a group
of images that never links back, that group is simply lost - and the loss is
invisible in the usual metrics, because everything reported is computed over
the cameras that *did* register.

Why the images are unreachable rather than merely difficult
-----------------------------------------------------------
Registering an image needs 2D-3D correspondences, and a track only becomes a
3D point once **two** registered cameras observe it. An image joined to the
model by a single verified pair therefore cannot be resected at all: its
shared tracks have exactly one registered observer, so none of them is
triangulated, so it has zero correspondences no matter how many inliers that
one pair had. Measured on this project's scene_01, 18 of 20 unregistered
images had *no* verified pair to any registered camera and the remaining two
had exactly one each (20 and 15 inliers). No amount of threshold tuning
reaches them; the deadlock is structural.

The answer is to reconstruct such a group as its own model and then align the
models to each other. Two reconstructions of one scene agree only up to a
similarity - SfM fixes the world frame and scale arbitrarily - so merging
means estimating 7 degrees of freedom from whatever the two models share.

Where the correspondences come from
-----------------------------------
The models cover *disjoint* image sets, so they share no observations and
there are no direct 3D-3D correspondences. The link has to come from the
bridging image pairs: a feature in image ``i`` (triangulated in model A) is
matched to a feature in image ``j`` (triangulated in model B), and that pair
of 3D points is one correspondence. Those bridging matches are typically the
weak ones that were not good enough to reconstruct with, which is exactly why
the merge is estimated robustly and never trusted on residual alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..utils import get_logger
from .features import FeatureSet, PairMatch
from .geometry import ransac_similarity
from .mapper import IncrementalMapper, MapperConfig, Reconstruction

__all__ = ["ModelResult", "reconstruct_models", "merge_models", "finalise_merge"]

LOGGER = get_logger()


@dataclass
class ModelResult:
    """One reconstructed model, kept with the mapper that built it.

    The mapper is retained because merging needs its ``_track_of_feature``
    index: without it there is no way to ask which 3D point a given image
    feature became.
    """

    mapper: IncrementalMapper
    rec: Reconstruction
    index: int = 0
    merged_from: List[int] = field(default_factory=list)

    @property
    def images(self) -> Set[int]:
        return set(self.rec.poses)


def _point_for_feature(model: ModelResult, image: int, feat: int) -> Optional[np.ndarray]:
    tid = model.mapper._track_of_feature.get((image, feat))
    if tid is None:
        return None
    return model.rec.points3d.get(tid)


def _model_scale(rec: Reconstruction) -> float:
    """Characteristic size of a model, used to set a scale-free threshold."""
    centres = np.array(list(rec.camera_centers().values()))
    if len(centres) < 2:
        pts = np.array(list(rec.points3d.values()))
        if len(pts) < 2:
            return 1.0
        return float(np.linalg.norm(pts.max(0) - pts.min(0))) or 1.0
    return float(np.linalg.norm(centres.max(0) - centres.min(0))) or 1.0


def reconstruct_models(
    features: Sequence[FeatureSet],
    Ks: Sequence[np.ndarray],
    cfg: MapperConfig,
    groups: Optional[Sequence[int]] = None,
    match_cache=None,
    min_model_images: int = 5,
    max_models: int = 4,
) -> List[ModelResult]:
    """Reconstruct the capture as one or more models, largest first.

    The first model is built normally. Whatever fails to register is then
    offered to a fresh mapper restricted to those images, and so on, until too
    few images remain to form a model. Each sub-model reuses the already
    computed match graph - only the *subset* changes - so the extra cost is
    mapping time alone, never rematching.
    """
    primary = IncrementalMapper(features, [K.copy() for K in Ks], cfg,
                                groups=groups, match_cache=match_cache)
    primary.match_all()
    primary.build_tracks()
    models = [ModelResult(primary, primary.run(), index=0)]

    all_images = set(range(len(features)))
    remaining = all_images - models[0].images

    while len(remaining) >= min_model_images and len(models) < max_models:
        subset = set(remaining)
        sub_matches = [m for m in primary.matches
                       if m.i in subset and m.j in subset]
        if len(sub_matches) < min_model_images:
            break

        LOGGER.info("Attempting model %d from %d unregistered image(s), "
                    "%d verified pairs among them",
                    len(models), len(subset), len(sub_matches))
        # A fresh config: self-calibration must not run again, since the focal
        # is already settled by the primary model and re-scanning it on a
        # handful of images would be far less reliable.
        sub_cfg = MapperConfig(**{**cfg.__dict__, "auto_focal": False})
        sub = IncrementalMapper(features, [K.copy() for K in Ks], sub_cfg,
                                groups=groups, match_cache=None)
        sub.matches = sub_matches
        sub.build_tracks()
        try:
            rec = sub.run()
        except RuntimeError as exc:
            LOGGER.info("  model %d abandoned: %s", len(models), exc)
            break
        if rec.num_registered < min_model_images:
            LOGGER.info("  model %d registered only %d image(s); stopping",
                        len(models), rec.num_registered)
            break

        models.append(ModelResult(sub, rec, index=len(models)))
        newly = set(rec.poses)
        if not newly - (all_images - remaining):
            break                                  # made no progress; stop
        remaining -= newly

    models.sort(key=lambda m: -m.rec.num_registered)
    for i, m in enumerate(models):
        m.index = i
    return models


def merge_models(primary: ModelResult, other: ModelResult,
                 matches: Sequence[PairMatch],
                 min_correspondences: int = 12,
                 min_inliers: int = 8,
                 threshold_factor: float = 0.03,
                 seed: int = 0) -> bool:
    """Align ``other`` into ``primary``'s frame and absorb it. In-place.

    Returns True only when the alignment is supported by enough agreeing
    correspondences. A merge that cannot be justified is refused rather than
    forced: two models placed wrongly in one frame is precisely the failure
    that reprojection error cannot see, and reporting two honest models beats
    one confident-looking wrong one.
    """
    a_imgs, b_imgs = primary.images, other.images
    src: List[np.ndarray] = []
    dst: List[np.ndarray] = []

    for m in matches:
        if m.i in a_imgs and m.j in b_imgs:
            ia, ja, ib, jb = m.i, m.idx_i, m.j, m.idx_j
        elif m.j in a_imgs and m.i in b_imgs:
            ia, ja, ib, jb = m.j, m.idx_j, m.i, m.idx_i
        else:
            continue
        for fa, fb in zip(np.asarray(ja).ravel(), np.asarray(jb).ravel()):
            pa = _point_for_feature(primary, ia, int(fa))
            pb = _point_for_feature(other, ib, int(fb))
            if pa is not None and pb is not None:
                dst.append(pa)      # primary frame is the destination
                src.append(pb)

    if len(src) < min_correspondences:
        LOGGER.info("  models %d and %d share only %d usable 3D correspondence(s); "
                    "not merging", primary.index, other.index, len(src))
        return False

    src_arr, dst_arr = np.array(src), np.array(dst)
    threshold = threshold_factor * _model_scale(primary.rec)
    model, inliers = ransac_similarity(src_arr, dst_arr, threshold=threshold, seed=seed)
    if model is None or int(inliers.sum()) < min_inliers:
        LOGGER.warning("  models %d and %d: similarity not supported "
                       "(%d/%d correspondences agree); keeping them separate",
                       primary.index, other.index,
                       0 if model is None else int(inliers.sum()), len(src))
        return False

    scale, R, t = model
    LOGGER.info("  merging model %d into %d: %d/%d correspondences agree, "
                "scale %.4f", other.index, primary.index,
                int(inliers.sum()), len(src), scale)

    # Transform poses. A world point maps as X_a = s R X_b + t, so a camera
    # with X_cam = R_b X_b + t_b becomes R_b (R^T / s)(X_a - t) + t_b, i.e.
    # rotation R_b R^T and translation t_b - R_b R^T t / s, with the camera
    # translation carrying the scale.
    Rt = R.T
    for img, (R_b, t_b) in other.rec.poses.items():
        R_new = R_b @ Rt
        t_new = scale * t_b - R_new @ t
        primary.rec.poses[img] = (R_new, t_new)
        primary.rec.Ks[img] = other.rec.Ks[img].copy()
        primary.rec.groups[img] = other.rec.groups.get(img, 0)
        primary.rec.registered_order.append(img)

    primary.merged_from.append(other.index)
    return True


def finalise_merge(primary: ModelResult) -> None:
    """Re-solve the primary model after absorbing another one.

    Only *poses* are carried across by :func:`merge_models`, deliberately. The
    two models index their own track lists, so copying points would mean
    inventing an id mapping between them - and it is unnecessary: the primary
    mapper built its tracks from the whole match graph, so tracks covering the
    absorbed images already exist and merely lacked a registered camera to
    triangulate against. Supplying the poses is enough for them to become
    points.

    The adjustment afterwards matters as much as the transform. The similarity
    is fitted to correspondences drawn from the weakest pairs in the capture,
    so it lands the model in roughly the right place; bundle adjustment is what
    makes it exact.
    """
    primary.mapper._triangulate_all()
    primary.mapper._run_ba(global_ba=True)
    primary.mapper._filter_points()
    primary.mapper._triangulate_all()
    primary.mapper._run_ba(global_ba=True)
    primary.mapper._filter_points()
