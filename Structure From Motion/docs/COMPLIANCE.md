# SfM implementation boundary

## Assignment condition

> Existing functions may be used, but existing functions that directly implement the SfM task may not be called.

This project treats feature detection and descriptor matching as allowed low-level image operations. Everything that estimates motion or structure is implemented in `src/sfm/custom_multiview.py` with NumPy.

The canonical `outputs/scene_cv_data/measurements/sift_reference.sqlite` is a neutral measurement store. Its reconstruction inputs are limited to camera intrinsics, image names, 2D keypoints, descriptors, and tentative descriptor-match index pairs. It does not contain essential or fundamental matrices, verified two-view geometry, relative poses, registered camera poses, 3D points, or externally refined parameters.

## Allowed external calls

| Operation | External call | Why it is allowed |
| --- | --- | --- |
| Image decoding/resizing | Pillow and OpenCV image I/O | Preprocessing only |
| Local features | OpenCV SIFT, AKAZE, ORB; optional ALIKED | Produces keypoints and descriptors, not camera motion or 3D structure |
| Descriptor matching | OpenCV FLANN or brute-force matcher | Produces tentative 2D correspondences only |
| Measurement migration | SQLite row/blob reading | Copies only the allowed low-level fields listed above |
| Linear algebra | NumPy SVD, eigendecomposition, least-squares solve | Generic numerical primitives |
| Viewer | Three.js | Displays exported points and poses only |

## Implemented here

The project directly implements:

1. Hartley normalization and normalized eight-point essential estimation.
2. Sampson error and random-sample geometric verification.
3. Essential decomposition and cheirality-based pose selection.
4. Conflict-free multi-image feature-track construction.
5. Linear DLT triangulation with best-pair and multiview variants.
6. Calibrated DLT camera resection with custom random sampling.
7. Gauss-Newton camera-pose refinement.
8. Incremental image registration and relative-pose graph bridging.
9. Alternating pose/structure bundle refinement.
10. Reconstruction export, point cleanup, and camera visualization.

## Explicitly excluded

The repository does not include or invoke an external mapper, reconstruction command, essential-matrix estimator, pose-recovery routine, triangulation routine, camera-resection routine, or bundle-adjustment routine. Run the static guard with:

```powershell
.\.venv\Scripts\python.exe scripts\audit_sfm_compliance.py
```

OpenSfM is a literature and architecture reference only. Calling its `reconstruct` or `bundle` commands would violate this project's boundary.
