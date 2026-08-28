from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path
import numpy as np


PLY_TYPES = {
    "char": "i1",
    "uchar": "u1",
    "short": "<i2",
    "ushort": "<u2",
    "int": "<i4",
    "uint": "<u4",
    "float": "<f4",
    "double": "<f8",
}


def _read_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    properties: list[tuple[str, str]] = []
    vertex_count = 0
    file_format = ""
    current_element = ""
    with path.open("rb") as stream:
        if stream.readline().strip() != b"ply":
            raise ValueError(f"Not a PLY file: {path}")
        while True:
            raw = stream.readline()
            if not raw:
                raise ValueError(f"Incomplete PLY header: {path}")
            line = raw.decode("ascii", errors="strict").strip()
            parts = line.split()
            if parts[:1] == ["format"]:
                file_format = parts[1]
            elif parts[:2] == ["element", "vertex"]:
                current_element = "vertex"
                vertex_count = int(parts[2])
            elif parts[:1] == ["element"]:
                current_element = parts[1]
            elif parts[:1] == ["property"] and current_element == "vertex" and parts[1] != "list":
                properties.append((parts[2], PLY_TYPES[parts[1]]))
            elif line == "end_header":
                data_offset = stream.tell()
                break

    names = [name for name, _ in properties]
    if not all(name in names for name in ("x", "y", "z")):
        raise ValueError(f"PLY file has no x/y/z vertex properties: {path}")
    if file_format == "binary_little_endian":
        dtype = np.dtype(properties)
        records = np.memmap(path, mode="r", dtype=dtype, offset=data_offset, shape=(vertex_count,))
        points = np.column_stack((records["x"], records["y"], records["z"])).astype(np.float64)
        if all(name in names for name in ("red", "green", "blue")):
            colors = np.column_stack((records["red"], records["green"], records["blue"])).astype(np.uint8)
        else:
            colors = np.full((vertex_count, 3), 190, dtype=np.uint8)
    elif file_format == "ascii":
        rows: list[list[float]] = []
        with path.open("r", encoding="ascii", errors="strict") as stream:
            for line in stream:
                if line.strip() == "end_header":
                    break
            for _ in range(vertex_count):
                rows.append([float(value) for value in stream.readline().split()[: len(properties)]])
        values = np.asarray(rows, dtype=np.float64)
        points = values[:, [names.index("x"), names.index("y"), names.index("z")]]
        if all(name in names for name in ("red", "green", "blue")):
            colors = values[:, [names.index("red"), names.index("green"), names.index("blue")]].astype(np.uint8)
        else:
            colors = np.full((vertex_count, 3), 190, dtype=np.uint8)
    else:
        raise ValueError(f"Unsupported PLY format '{file_format}': {path}")
    finite = np.isfinite(points).all(axis=1)
    return points[finite], colors[finite]


def load_point_cloud(path: Path) -> tuple[np.ndarray, np.ndarray]:
    suffix = path.suffix.casefold()
    if suffix == ".ply":
        return _read_ply(path)
    if suffix == ".obj":
        points: list[list[float]] = []
        colors: list[list[int]] = []
        for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not raw_line.startswith("v "):
                continue
            values = raw_line.split()[1:]
            if len(values) < 3:
                continue
            points.append([float(value) for value in values[:3]])
            if len(values) >= 6:
                rgb = np.clip(np.asarray([float(value) for value in values[3:6]]) * 255.0, 0, 255)
                colors.append(rgb.astype(int).tolist())
            else:
                colors.append([190, 190, 190])
        return np.asarray(points, dtype=np.float64), np.asarray(colors, dtype=np.uint8)
    if suffix in {".xyz", ".txt", ".pts", ".csv"}:
        delimiter = "," if suffix == ".csv" else None
        values = np.loadtxt(path, delimiter=delimiter, comments="#", ndmin=2)
        points = values[:, :3].astype(np.float64)
        colors = values[:, 3:6].astype(np.uint8) if values.shape[1] >= 6 else np.full((len(points), 3), 190, np.uint8)
        finite = np.isfinite(points).all(axis=1)
        return points[finite], colors[finite]
    raise ValueError("Point cloud must be PLY, OBJ, XYZ, PTS, TXT, or CSV")


def write_binary_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    colors = np.asarray(colors, dtype=np.uint8)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    ).encode("ascii")
    records = np.empty(
        len(points),
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    records["x"], records["y"], records["z"] = np.asarray(points, dtype=np.float32).T
    records["red"], records["green"], records["blue"] = colors.T
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(records.tobytes())


def apply_similarity(points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return scale * (points @ rotation.T) + translation


def estimate_similarity(source: np.ndarray, target: np.ndarray, *, estimate_scale: bool = True) -> tuple[float, np.ndarray, np.ndarray]:
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or len(source) < 3:
        raise ValueError("Similarity estimation requires at least three paired 3D points")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    centered_source = source - source_center
    centered_target = target - target_center
    covariance = centered_target.T @ centered_source / len(source)
    left, singular, right = np.linalg.svd(covariance)
    signs = np.ones(3)
    if np.linalg.det(left @ right) < 0:
        signs[-1] = -1.0
    rotation = left @ np.diag(signs) @ right
    source_variance = float(np.mean(np.sum(centered_source * centered_source, axis=1)))
    scale = float(np.sum(singular * signs) / source_variance) if estimate_scale else 1.0
    translation = target_center - scale * (rotation @ source_center)
    return scale, rotation, translation


def nearest_neighbors(source: np.ndarray, target: np.ndarray, *, chunk_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    if len(target) == 0:
        raise ValueError("Nearest-neighbor target is empty")
    distances = np.empty(len(source), dtype=np.float64)
    indices = np.empty(len(source), dtype=np.int64)
    for start in range(0, len(source), chunk_size):
        stop = min(start + chunk_size, len(source))
        delta = source[start:stop, None, :] - target[None, :, :]
        squared = np.einsum("ijk,ijk->ij", delta, delta)
        local_indices = np.argmin(squared, axis=1)
        indices[start:stop] = local_indices
        distances[start:stop] = np.sqrt(squared[np.arange(stop - start), local_indices])
    return distances, indices


def _sample(points: np.ndarray, count: int, seed: int) -> np.ndarray:
    if len(points) <= count:
        return points.copy()
    indices = np.random.default_rng(seed).choice(len(points), count, replace=False)
    return points[indices]


def _pca_basis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    center = np.median(points, axis=0)
    centered = points - center
    covariance = np.cov(centered, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    basis = vectors[:, np.argsort(values)[::-1]]
    radius = float(np.quantile(np.linalg.norm(centered, axis=1), 0.8))
    return center, basis, max(radius, 1e-12)


def automatic_initial_alignment(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    source_sample = _sample(source, 3500, 7)
    target_sample = _sample(target, 3500, 11)
    source_center, source_basis, source_radius = _pca_basis(source_sample)
    target_center, target_basis, target_radius = _pca_basis(target_sample)
    scale = target_radius / source_radius
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for permutation in itertools.permutations(range(3)):
        permutation_matrix = np.eye(3)[:, permutation]
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            mapping = permutation_matrix @ np.diag(signs)
            rotation = target_basis @ mapping @ source_basis.T
            if np.linalg.det(rotation) < 0:
                continue
            translation = target_center - scale * (rotation @ source_center)
            transformed = apply_similarity(source_sample, scale, rotation, translation)
            distances, _ = nearest_neighbors(transformed, target_sample)
            score = float(np.mean(np.partition(distances, int(0.7 * len(distances)))[: int(0.7 * len(distances))]))
            if best is None or score < best[0]:
                best = (score, rotation, translation)
    if best is None:
        raise RuntimeError("Could not initialize point-cloud alignment")
    return scale, best[1], best[2]


def refine_similarity_icp(
    source: np.ndarray,
    target: np.ndarray,
    initial: tuple[float, np.ndarray, np.ndarray],
    *,
    iterations: int = 25,
    trim_fraction: float = 0.75,
) -> tuple[float, np.ndarray, np.ndarray, list[float]]:
    source_sample = _sample(source, 6000, 17)
    target_sample = _sample(target, 8000, 19)
    scale, rotation, translation = initial
    history: list[float] = []
    for _ in range(iterations):
        transformed = apply_similarity(source_sample, scale, rotation, translation)
        distances, indices = nearest_neighbors(transformed, target_sample)
        keep_count = max(3, int(trim_fraction * len(distances)))
        keep = np.argpartition(distances, keep_count - 1)[:keep_count]
        delta_scale, delta_rotation, delta_translation = estimate_similarity(
            transformed[keep], target_sample[indices[keep]], estimate_scale=True
        )
        scale = delta_scale * scale
        rotation = delta_rotation @ rotation
        translation = delta_scale * (delta_rotation @ translation) + delta_translation
        rmse = float(np.sqrt(np.mean(np.square(distances[keep]))))
        history.append(rmse)
        if len(history) > 2 and abs(history[-2] - history[-1]) < 1e-7:
            break
    return scale, rotation, translation, history


def load_correspondences(path: Path) -> tuple[np.ndarray, np.ndarray]:
    source: list[list[float]] = []
    target: list[list[float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            source.append([float(row[f"sfm_{axis}"]) for axis in "xyz"])
            target.append([float(row[f"lidar_{axis}"]) for axis in "xyz"])
    return np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)


def _distance_metrics(distances: np.ndarray, thresholds: tuple[float, ...]) -> dict[str, float]:
    metrics = {
        "mean": float(np.mean(distances)),
        "median": float(np.median(distances)),
        "rmse": float(np.sqrt(np.mean(np.square(distances)))),
        "p90": float(np.quantile(distances, 0.90)),
        "p95": float(np.quantile(distances, 0.95)),
    }
    for threshold in thresholds:
        metrics[f"within_{threshold:g}"] = float(np.mean(distances <= threshold))
    return metrics


def evaluate_lidar_alignment(
    sfm_path: Path,
    lidar_path: Path,
    output_dir: Path,
    *,
    correspondences_path: Path | None = None,
    thresholds: tuple[float, ...] = (0.02, 0.05, 0.10),
) -> dict[str, object]:
    sfm_points, sfm_colors = load_point_cloud(sfm_path)
    lidar_points, _ = load_point_cloud(lidar_path)
    if len(sfm_points) < 10 or len(lidar_points) < 10:
        raise ValueError("Both point clouds must contain at least ten finite points")
    if correspondences_path is not None:
        source_control, target_control = load_correspondences(correspondences_path)
        initial = estimate_similarity(source_control, target_control, estimate_scale=True)
        initialization = "manual_correspondences"
    else:
        initial = automatic_initial_alignment(sfm_points, lidar_points)
        initialization = "automatic_pca_hypotheses"
    scale, rotation, translation, history = refine_similarity_icp(sfm_points, lidar_points, initial)
    aligned = apply_similarity(sfm_points, scale, rotation, translation)

    sfm_eval = _sample(aligned, 15000, 29)
    lidar_eval = _sample(lidar_points, 15000, 31)
    accuracy_distances, _ = nearest_neighbors(sfm_eval, lidar_eval)
    completeness_distances, _ = nearest_neighbors(lidar_eval, sfm_eval)
    accuracy = _distance_metrics(accuracy_distances, thresholds)
    completeness = _distance_metrics(completeness_distances, thresholds)
    report: dict[str, object] = {
        "initialization": initialization,
        "alignment": {
            "scale": scale,
            "rotation": rotation.tolist(),
            "translation": translation.tolist(),
            "trimmed_icp_rmse_history": history,
        },
        "input": {"sfm_points": len(sfm_points), "lidar_points": len(lidar_points)},
        "accuracy_sfm_to_lidar": accuracy,
        "completeness_lidar_to_sfm": completeness,
        "symmetric_chamfer_l1": accuracy["mean"] + completeness["mean"],
        "symmetric_chamfer_l2": float(np.mean(np.square(accuracy_distances)) + np.mean(np.square(completeness_distances))),
        "units": "LiDAR coordinate units; normally metres for iPhone exports",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_binary_ply(output_dir / "sfm_aligned_to_lidar.ply", aligned, sfm_colors)
    lidar_in_sfm = ((lidar_points - translation) @ rotation) / scale
    write_binary_ply(
        output_dir / "lidar_aligned_to_sfm.ply",
        lidar_in_sfm,
        np.tile(np.array([[232, 151, 72]], dtype=np.uint8), (len(lidar_in_sfm), 1)),
    )
    overlay_lidar = _sample(lidar_points, min(30000, len(lidar_points)), 37)
    overlay_sfm = _sample(aligned, min(30000, len(aligned)), 41)
    overlay_points = np.vstack((overlay_lidar, overlay_sfm))
    overlay_colors = np.vstack(
        (
            np.tile(np.array([[232, 151, 72]], dtype=np.uint8), (len(overlay_lidar), 1)),
            np.tile(np.array([[54, 154, 201]], dtype=np.uint8), (len(overlay_sfm), 1)),
        )
    )
    write_binary_ply(output_dir / "comparison_overlay.ply", overlay_points, overlay_colors)
    (output_dir / "lidar_comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
