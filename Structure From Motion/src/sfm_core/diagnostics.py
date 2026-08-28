"""Measurement tools for deciding whether a reconstruction has really split.

A reconstruction that has silently broken into two scenes sharing one
coordinate frame is the hardest failure in this project to see: reprojection
error is a *local* statistic, so both halves can reproject their own points
perfectly while sitting in the wrong place relative to each other. Track
length and registration count are equally blind to it.

This module supplies the evidence that reprojection error cannot. Nothing
here changes reconstruction behaviour; every function measures.

The central idea is that the verified match graph already knows the answer.
If two groups of cameras are genuinely one scene, verified image pairs bridge
them, and each bridging pair's own two-view geometry agrees with where the
reconstruction put those two cameras. If the reconstruction has split, either
no pair bridges the groups at all (a capture discontinuity - the photographer
moved without shooting) or the bridging pairs disagree with the model, which
is what an overlaid pair of scenes looks like.

Distinguishing those two cases matters, because they call for opposite
responses: a discontinuity is honest and the cameras should be kept, while a
disagreement means the model is wrong and must be split or re-solved.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .features import FeatureSet, PairMatch, Track
from .geometry import (
    decompose_essential,
    ransac_essential,
    rotation_angle_deg,
    select_pose_by_cheirality,
)

LOGGER = logging.getLogger("sfm.diagnostics")


# ===========================================================================
# Match-graph connectivity
# ===========================================================================


def match_graph_components(
    matches: Sequence[PairMatch],
    num_images: int,
    min_inliers: int = 0,
    restrict_to: Optional[Iterable[int]] = None,
) -> List[List[int]]:
    """Connected components of the verified match graph, largest first.

    Two images are adjacent when a verified pair joins them with at least
    ``min_inliers`` inliers. ``min_inliers`` of 0 accepts every pair that
    survived geometric verification.

    This is an upper bound on what incremental registration can achieve: an
    image outside the largest component cannot be reached by growing a model,
    however good the mapper is. Comparing this bound against the number
    actually registered separates a data limit from an algorithmic one.
    """
    keep: Optional[Set[int]] = None if restrict_to is None else set(restrict_to)
    nodes = range(num_images) if keep is None else sorted(keep)

    adjacency: Dict[int, Set[int]] = {i: set() for i in nodes}
    for m in matches:
        if m.num_inliers < min_inliers:
            continue
        if keep is not None and (m.i not in keep or m.j not in keep):
            continue
        if m.i in adjacency and m.j in adjacency:
            adjacency[m.i].add(m.j)
            adjacency[m.j].add(m.i)

    components: List[List[int]] = []
    unvisited = set(adjacency)
    while unvisited:
        start = unvisited.pop()
        component = {start}
        stack = [start]
        while stack:
            node = stack.pop()
            for neighbour in adjacency[node]:
                if neighbour not in component:
                    component.add(neighbour)
                    unvisited.discard(neighbour)
                    stack.append(neighbour)
        components.append(sorted(component))
    components.sort(key=len, reverse=True)
    return components


def connectivity_table(
    matches: Sequence[PairMatch],
    num_images: int,
    thresholds: Sequence[int] = (30, 50, 100, 150, 200),
) -> List[Dict[str, object]]:
    """Component structure of the match graph across inlier thresholds.

    A graph that fragments rapidly as the threshold rises is held together by
    weak pairs, and registration through it will be fragile.
    """
    rows = []
    for threshold in thresholds:
        components = match_graph_components(matches, num_images, threshold)
        rows.append({
            "min_inliers": int(threshold),
            "num_components": len(components),
            "sizes": [len(c) for c in components],
            "largest": len(components[0]) if components else 0,
        })
    return rows


def weak_links(
    matches: Sequence[PairMatch],
    num_images: int,
    min_inliers: int = 100,
) -> List[Tuple[int, int, int]]:
    """Consecutive capture pairs whose verification is weak or absent.

    Returned as ``(i, j, inliers)`` with zero meaning the pair did not survive
    verification at all. Assumes images are ordered by capture, which holds
    for a sequentially named walking capture. Zeros mark the points where the
    photographer moved without covering the gap - the places a sequential
    model must jump.
    """
    strength: Dict[Tuple[int, int], int] = {}
    for m in matches:
        strength[(min(m.i, m.j), max(m.i, m.j))] = m.num_inliers
    out = []
    for i in range(num_images - 1):
        count = strength.get((i, i + 1), 0)
        if count < min_inliers:
            out.append((i, i + 1, int(count)))
    return out


# ===========================================================================
# Do bridging pairs agree with the model?
# ===========================================================================


@dataclass
class BridgeEvidence:
    """One verified pair spanning two camera groups, checked against the model."""

    i: int
    j: int
    num_inliers: int
    #: Angle between the model's relative rotation and the pair's own.
    rotation_error_deg: float = float("nan")
    #: Angle between the model's baseline direction and the pair's own.
    translation_error_deg: float = float("nan")
    #: Corroborating only - see the note in :func:`bridge_evidence`.
    translation_agrees: bool = False
    agrees: bool = False
    note: str = ""


@dataclass
class SplitEvidence:
    """Why two groups of cameras are, or are not, one scene."""

    group_a: List[int] = field(default_factory=list)
    group_b: List[int] = field(default_factory=list)
    bridges: List[BridgeEvidence] = field(default_factory=list)
    verdict: str = ""
    detail: str = ""

    @property
    def num_bridges(self) -> int:
        return len(self.bridges)

    @property
    def num_agreeing(self) -> int:
        return sum(1 for b in self.bridges if b.agrees)


def _relative_pose_from_pair(
    features: Sequence[FeatureSet],
    match: PairMatch,
    Ks: Dict[int, np.ndarray],
    threshold_px: float,
    seed: int,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Recover (R, t) for one pair from its own correspondences alone.

    Deliberately independent of the reconstruction, so the result can be used
    as an outside check on it.
    """
    pts1 = features[match.i].keypoints[match.idx_i]
    pts2 = features[match.j].keypoints[match.idx_j]
    if len(pts1) < 8:
        return None
    result = ransac_essential(pts1, pts2, Ks[match.i], Ks[match.j],
                              threshold_px=threshold_px, seed=seed)
    if result.model is None or result.num_inliers < 8:
        return None
    inl = result.inliers
    R, t, mask, _ = select_pose_by_cheirality(
        decompose_essential(result.model), pts1[inl], pts2[inl],
        Ks[match.i], K2=Ks[match.j],
    )
    if R is None or int(mask.sum()) < 8:
        return None
    return R, t


def bridge_evidence(
    features: Sequence[FeatureSet],
    matches: Sequence[PairMatch],
    poses: Dict[int, Tuple[np.ndarray, np.ndarray]],
    Ks: Dict[int, np.ndarray],
    group_a: Iterable[int],
    group_b: Iterable[int],
    rotation_tolerance_deg: float = 8.0,
    translation_tolerance_deg: float = 25.0,
    threshold_px: float = 1.5,
    max_bridges: int = 40,
    seed: int = 0,
) -> SplitEvidence:
    """Check every verified pair spanning two camera groups against the model.

    For each bridging pair, the relative pose implied by the reconstruction is
    compared with the relative pose the pair yields on its own. Rotation is the
    trustworthy comparison; two-view translation is known only up to scale, so
    only its direction is compared and it is given a looser tolerance.

    The verdict distinguishes three situations that registration counts alone
    cannot tell apart:

    ``connected``
        Bridging pairs exist and agree. The groups are one scene and any
        culling that separates them is destroying good cameras.
    ``discontinuity``
        No bridging pair survives verification. The photographer moved without
        shooting the gap. Both groups may still be individually correct, but
        nothing observed ties them together.
    ``inconsistent``
        Bridging pairs exist and disagree with the model. This is the overlaid
        two-scenes failure, and the model is genuinely wrong.
    """
    a = set(group_a)
    b = set(group_b)
    evidence = SplitEvidence(group_a=sorted(a), group_b=sorted(b))

    spanning = [m for m in matches
                if (m.i in a and m.j in b) or (m.i in b and m.j in a)]
    spanning.sort(key=lambda m: -m.num_inliers)

    if not spanning:
        evidence.verdict = "discontinuity"
        evidence.detail = (
            "No verified image pair spans the two groups. Nothing observed "
            "connects them, so their relative placement rests entirely on "
            "indirect chaining."
        )
        return evidence

    for match in spanning[:max_bridges]:
        item = BridgeEvidence(i=match.i, j=match.j, num_inliers=int(match.num_inliers))
        if match.i not in poses or match.j not in poses:
            item.note = "one endpoint is not registered"
            evidence.bridges.append(item)
            continue

        pair_pose = _relative_pose_from_pair(
            features, match, Ks, threshold_px, seed + match.i * 977 + match.j)
        if pair_pose is None:
            item.note = "two-view pose not recoverable"
            evidence.bridges.append(item)
            continue

        R_pair, t_pair = pair_pose
        R_i, t_i = poses[match.i]
        R_j, t_j = poses[match.j]
        # Relative pose the model implies, in the same convention the pair
        # produces: camera i to camera j.
        R_model = R_j @ R_i.T
        t_model = t_j - R_model @ t_i

        item.rotation_error_deg = float(rotation_angle_deg(R_model.T @ R_pair))

        n_model = float(np.linalg.norm(t_model))
        n_pair = float(np.linalg.norm(t_pair))
        if n_model > 1e-9 and n_pair > 1e-9:
            cos = float(np.clip((t_model / n_model) @ (t_pair / n_pair), -1.0, 1.0))
            item.translation_error_deg = float(np.degrees(np.arccos(cos)))

        # Agreement is decided on rotation alone. Two-view translation is
        # recovered only up to scale and sign, and its *direction* is poorly
        # conditioned for the short, mostly-forward baselines a walking
        # capture produces - a 180 degree flip shows up routinely on pairs
        # whose rotation agrees to within a degree. Treating it as a hard
        # gate marks sound links as broken, so it is recorded as corroborating
        # evidence and reported, not used to decide.
        item.agrees = item.rotation_error_deg <= rotation_tolerance_deg
        if np.isfinite(item.translation_error_deg):
            flipped = 180.0 - item.translation_error_deg
            item.translation_agrees = bool(
                min(item.translation_error_deg, flipped) <= translation_tolerance_deg)
        evidence.bridges.append(item)

    checked = [x for x in evidence.bridges if np.isfinite(x.rotation_error_deg)]
    if not checked:
        evidence.verdict = "discontinuity"
        evidence.detail = (
            f"{len(spanning)} pair(s) span the groups but none yielded a "
            "usable two-view pose, so the link cannot be confirmed."
        )
        return evidence

    agreeing = [x for x in checked if x.agrees]
    median_rotation = float(np.median([x.rotation_error_deg for x in checked]))

    # Weighted by inlier count: a pair verified on 77 correspondences is much
    # stronger evidence about where two cameras sit than one scraping past the
    # threshold on 31, and counting them equally lets a crowd of marginal
    # pairs outvote the reliable ones.
    total_weight = sum(x.num_inliers for x in checked)
    agree_weight = sum(x.num_inliers for x in agreeing)
    fraction = agree_weight / total_weight if total_weight else 0.0

    if fraction >= 0.5:
        evidence.verdict = "connected"
        evidence.detail = (
            f"{len(agreeing)}/{len(checked)} bridging pairs agree with the model "
            f"({fraction:.0%} by inlier weight, median rotation disagreement "
            f"{median_rotation:.2f} deg). The groups are one scene."
        )
    else:
        evidence.verdict = "inconsistent"
        evidence.detail = (
            f"only {len(agreeing)}/{len(checked)} bridging pairs agree "
            f"({fraction:.0%} by inlier weight, median rotation disagreement "
            f"{median_rotation:.2f} deg). The model places these groups "
            "inconsistently with what the images show."
        )
    return evidence


# ===========================================================================
# Whole-model report
# ===========================================================================


def covisibility_components(
    points3d: Dict[int, np.ndarray],
    tracks: Sequence[Track],
    poses: Dict[int, Tuple[np.ndarray, np.ndarray]],
    min_shared: int,
) -> List[List[int]]:
    """Registered cameras grouped by shared triangulated points.

    Mirrors the mapper's own definition so a report can be produced without
    reaching into mapper internals.
    """
    shared: Dict[Tuple[int, int], int] = {}
    for tid in points3d:
        observers = sorted(i for i in tracks[tid].observations if i in poses)
        for a, b in combinations(observers, 2):
            shared[(a, b)] = shared.get((a, b), 0) + 1

    adjacency: Dict[int, Set[int]] = {i: set() for i in poses}
    for (a, b), count in shared.items():
        if count >= min_shared:
            adjacency[a].add(b)
            adjacency[b].add(a)

    components: List[List[int]] = []
    unvisited = set(adjacency)
    while unvisited:
        start = unvisited.pop()
        component = {start}
        stack = [start]
        while stack:
            node = stack.pop()
            for neighbour in adjacency[node]:
                if neighbour not in component:
                    component.add(neighbour)
                    unvisited.discard(neighbour)
                    stack.append(neighbour)
        components.append(sorted(component))
    components.sort(key=len, reverse=True)
    return components


def model_report(
    features: Sequence[FeatureSet],
    matches: Sequence[PairMatch],
    tracks: Sequence[Track],
    poses: Dict[int, Tuple[np.ndarray, np.ndarray]],
    points3d: Dict[int, np.ndarray],
    Ks: Dict[int, np.ndarray],
    min_covisible_points: int = 15,
    min_pair_inliers: int = 30,
    seed: int = 0,
) -> Dict[str, object]:
    """Everything needed to judge a reconstruction's structural health.

    Reports the registration ceiling implied by the match graph alongside what
    was actually registered, so the two can be compared directly, and runs the
    bridging-pair check on the largest covisibility split.
    """
    num_images = len(features)
    graph = match_graph_components(matches, num_images, min_pair_inliers)
    reachable = len(graph[0]) if graph else 0

    covis = covisibility_components(points3d, tracks, poses, min_covisible_points)

    report: Dict[str, object] = {
        "num_images": num_images,
        "registered_total": len(poses),
        "match_graph": {
            "min_pair_inliers": int(min_pair_inliers),
            "num_components": len(graph),
            "sizes": [len(c) for c in graph],
            "largest_component": reachable,
            "unreachable_images": sorted(
                set(range(num_images)) - set(graph[0] if graph else [])),
        },
        "covisibility": {
            "min_shared_points": int(min_covisible_points),
            "num_components": len(covis),
            "sizes": [len(c) for c in covis],
        },
        "connectivity_table": connectivity_table(matches, num_images),
        "weak_consecutive_links": weak_links(matches, num_images),
    }

    # The registration ceiling is what the graph allows, not the image count.
    # Reporting against the wrong denominator is how a data limit gets
    # mistaken for an algorithmic one, and vice versa.
    report["registration_ceiling"] = reachable
    report["registered_fraction_of_ceiling"] = (
        round(len(poses) / reachable, 4) if reachable else 0.0)

    if len(covis) > 1:
        evidence = bridge_evidence(
            features, matches, poses, Ks, covis[0], covis[1], seed=seed)
        report["split_evidence"] = {
            "group_a_size": len(evidence.group_a),
            "group_b_size": len(evidence.group_b),
            "num_bridges": evidence.num_bridges,
            "num_agreeing": evidence.num_agreeing,
            "verdict": evidence.verdict,
            "detail": evidence.detail,
        }
    else:
        report["split_evidence"] = {
            "verdict": "single_component",
            "detail": "The registered cameras form one covisibility component.",
        }
    return report


def format_report(report: Dict[str, object]) -> str:
    """Human-readable rendering of :func:`model_report`."""
    graph = report["match_graph"]          # type: ignore[index]
    covis = report["covisibility"]         # type: ignore[index]
    split = report["split_evidence"]       # type: ignore[index]

    lines = [
        "=" * 70,
        "RECONSTRUCTION DIAGNOSTICS",
        "=" * 70,
        f"  images                    : {report['num_images']}",
        f"  registered                : {report['registered_total']}",
        f"  registration ceiling      : {report['registration_ceiling']} "
        f"(largest match-graph component at >={graph['min_pair_inliers']} inliers)",
        f"  registered / ceiling      : {report['registered_fraction_of_ceiling']:.1%}",
        "",
        f"  match-graph components    : {graph['num_components']}  sizes {graph['sizes'][:6]}",
        f"  covisibility components   : {covis['num_components']}  sizes {covis['sizes'][:6]}",
        "",
        "  connectivity vs inlier threshold:",
    ]
    for row in report["connectivity_table"]:  # type: ignore[index]
        lines.append(f"     >={row['min_inliers']:>4} inliers : "
                     f"{row['num_components']:>3} components, sizes {row['sizes'][:6]}")

    weak = report["weak_consecutive_links"]  # type: ignore[index]
    broken = [w for w in weak if w[2] == 0]
    lines += [
        "",
        f"  weak consecutive links    : {len(weak)}  (of which unverified: {len(broken)})",
    ]
    if broken:
        lines.append("     capture gaps at frames: "
                     + ", ".join(f"{a + 1}->{b + 1}" for a, b, _ in broken[:12]))

    lines += [
        "",
        f"  SPLIT VERDICT             : {split['verdict'].upper()}",
        f"     {split['detail']}",
        "=" * 70,
    ]
    return "\n".join(lines)
