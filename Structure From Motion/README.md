# NTU Structure-from-Motion Project

End-to-end 3D reconstruction from self-captured multi-view imagery of NTU campus
scenes: camera poses, sparse point cloud, dense multi-view stereo, and a
triangle mesh — with **every piece of reconstruction geometry implemented from
scratch**.

---

## 1. The constraint, and how it is satisfied

> *You may call some existing function in your development but cannot call
> existing functions that directly implement the SfM task.*

This rules out COLMAP, pycolmap, OpenSfM, and OpenCV's geometry solvers
(`findEssentialMat`, `recoverPose`, `findFundamentalMat`, `solvePnP`,
`triangulatePoints`, …). None of them is called anywhere in `src/` or
`scripts/`.

**What libraries do provide, and why each is a primitive rather than the task:**

| Library | Used for | Why this is not "the SfM task" |
|---|---|---|
| OpenCV | SIFT / AKAZE / ORB detection and description, `knnMatch`, image decode, resize, resample | Finds and describes interest points and resamples images. Estimates no camera motion and no 3D structure. |
| NumPy | linear algebra, SVD, least-squares primitives | General matrix mathematics. |
| SciPy | `least_squares` (trust-region solver), `lil_matrix`, `cKDTree`, `spatial.transform.Rotation` | A *generic* non-linear optimiser; a sparse matrix type; a spatial index. `Rotation` only converts between axis-angle and matrix form — it estimates nothing from data, and the conversion is elementary linear algebra. |
| Open3D | point-cloud file I/O, KD-tree, bounding box, display | Audited: no `registration_*`, no `TSDFVolume`, no `create_from_point_cloud_poisson`. Library surface reconstruction was removed as a mesh option — TSDF fusion and marching cubes are ours. |
| scikit-image | `marching_cubes` | A generic isosurface extractor over a scalar field. The TSDF it runs on is ours. |
| Pillow | EXIF, orientation | Metadata. |

**What is implemented here**, in `src/sfm_core/`:

| Component | File | Notes |
|---|---|---|
| Normalised 8-point fundamental matrix | `geometry.py` | Hartley normalisation |
| Essential matrix + RANSAC | `geometry.py` | Sampson distance, adaptive iteration count |
| E decomposition + cheirality | `geometry.py` | four-fold ambiguity resolved by depth *and* reprojection |
| DLT triangulation (2-view and n-view) | `geometry.py` | |
| P3P + non-linear pose refinement | `p3p.py` | Grunert's formulation |
| RANSAC PnP | `geometry.py` | |
| Bundle adjustment | `bundle.py` | axis-angle parameterisation, Jacobian sparsity, Huber loss, gauge fixing |
| Track construction | `features.py` | union-find over pairwise matches |
| Incremental mapper | `mapper.py` | seed selection, next-best-view, local/global BA |
| Plane-sweep MVS | `dense.py` | NCC over depth hypotheses, geometric consistency, sub-plane refinement |
| TSDF fusion | `surface.py` | |
| 7-DoF similarity (Umeyama) + RANSAC | `geometry.py` | for aligning independent models |
| Multi-model reconstruction | `multimodel.py` | |
| Split diagnostics | `diagnostics.py` | |

The earlier COLMAP-based comparison code has been moved **out of the project
tree** to `../sfm_excluded_colmap_reference/`, so it cannot ship with the
deliverable. Nothing in `src/` or `scripts/` imports it, and `pycolmap` has
been removed from `requirements.txt`.

**Enforced by a test, not by inspection.**

```bash
python scripts/audit_sfm_compliance.py     # exit 0 = clean
python -m unittest discover -s tests       # 16 tests, includes the audit
```

The audit parses each file into a syntax tree and inspects only *calls*,
*imports* and *subprocess invocations*, so comments and docstrings cannot raise
a false alarm and our own `bundle_adjust` is not mistaken for a library one. A
grep-based version reported 68 findings on this same compliant tree, every one
of them a comment, a docstring, or one of our own definitions — a result worse
than no check, because it teaches you to ignore it.

`tests/test_compliance.py` also plants a real violation and confirms the audit
rejects it, so a refactor that quietly defangs the check fails the suite rather
than a marker.

**Audit method.** The tree is scanned for calls to every OpenCV geometry
solver, for `pycolmap` / `opensfm` imports, and for Open3D's registration,
TSDF and Poisson-reconstruction entry points. The only remaining textual
matches are (a) the docstring table in `sfm_core/geometry.py` naming the
OpenCV functions this project implements *instead of*, and (b) one comment in
`environment.py` recording that COLMAP is deliberately not probed.

### Where exactly the line falls

**Anything that turns pixels into features is allowed. Anything that turns
features into geometry is ours.**

A detector takes an image and returns 2D keypoint locations and descriptor
vectors. It estimates no camera motion and no 3D structure, so swapping SIFT
for AKAZE changes *which corners are found*, never *how pose and structure are
computed*. The same reasoning admits descriptor matching, which produces
tentative 2D correspondences and nothing more.

Two decisions follow from holding that line strictly:

* The synthetic-scene generator computes its texture-mapping homography with
  its own DLT rather than `cv2.getPerspectiveTransform`. That call estimates
  nothing about a real scene — the quad corners come from a known camera and a
  known plane — but it is still a library function fitting a geometric
  transform, and keeping the rule absolute means no reader has to weigh the
  distinction. Validated against the function it replaced: maximum mapping
  difference 3.1e-05 px over 200 random quads.
* `warpPerspective` resamples an image using a homography *this project*
  derives from the plane-sweep geometry. `--no-library-warp` substitutes our
  own bilinear sampler — about 40x slower with no quality change — to
  demonstrate that the library performs image resampling and no geometric work.

**Every OpenCV function actually called**, extracted from the syntax tree so
docstrings cannot hide anything:

```
detectors / matchers : SIFT_create, AKAZE_create, ORB_create,
                       BFMatcher_create, FlannBasedMatcher
image processing     : GaussianBlur, Sobel, Laplacian, Canny, boxFilter,
                       filter2D, calcHist, cvtColor, resize, split,
                       warpPerspective
I/O and drawing      : imdecode, imencode, VideoCapture, VideoWriter,
                       VideoWriter_fourcc, circle, line, putText,
                       fillConvexPoly

of which estimate geometry: none
```

---

## Report results: interactive reconstructions

The five main reconstructions in the report are in [`results/`](results/), each
as a self-contained 3D viewer plus the run's `summary.json` (all metrics).
Download an `.html` file and open it in a browser. Toggle sparse points, dense points, the mesh and camera poses, or switch between the Home, Top, Front and Side views.

| Scene | Capture | Viewer | Metrics |
|---|---|---|---|
| Textured statue | 29 photos, iPhone 17 | [textured_statue.html](results/textured_statue.html) | [summary](results/textured_statue_summary.json) |
| Tall statue | 4K video, 27 mm, iPhone 12 Pro Max | [tall_statue.html](results/tall_statue.html) | [summary](results/tall_statue_summary.json) |
| Big building | 4K video, 27 mm, iPhone 12 Pro Max | [big_building.html](results/big_building.html) | [summary](results/big_building_summary.json) |
| Water dispenser | 1080p video, afternoon, Galaxy S23 FE | [water_dispenser.html](results/water_dispenser.html) | [summary](results/water_dispenser_summary.json) |
| Security robot | 1080p video, iPhone 17 | [security_robot.html](results/security_robot.html) | [summary](results/security_robot_summary.json) |

All five use the same frozen settings:
`--feature-size 2400 --min-ncc 0.35 --consistent-views 2 --multi-model`.
The ablation, lens and lighting runs are kept outside the repository.

---

## 2. Results

Measured 2026-08-22. Both scenes are 1.9 MP WhatsApp-compressed JPEGs with EXIF
stripped.

| Metric | scene_01 (carpark, 110 img) | scene_02 (vending machine, 146 img) |
|---|---|---|
| Registered | **90/110** (+20 in a 2nd model) | **144/146** |
| Mean reprojection error | 1.3275 px | **1.2609 px** |
| Median reprojection error | 1.0477 px | 1.0832 px |
| Mean track length | 3.90 | **7.06** |
| Sparse points | 18,080 | 17,192 |
| Dense points | 5,154,161 | **13,195,325** |
| Dense valid fraction | 0.125 | **0.191** |
| Mesh faces | 395,134 | **941,204** |
| Match-graph ceiling | 110/110 | 146/146 |

scene_02 is the better reconstruction on every axis, and the reason is visible
in the track length: 7.06 observations per point against 3.90. A rigid,
well-textured subject with genuine 3D structure is worth more than any amount
of tuning.

**scene_01 is reconstructed as two models** (90 + 20). The 20-image group is
joined to the main model by only two verified pairs, which yields just two
usable 3D correspondences — far too few to fit a 7-DoF similarity. The merge is
therefore *refused* rather than forced. See §8.

---

## 2b. Detectors

Three local feature detectors are supported. Detection and description estimate
no geometry, so the choice changes *which corners are found*, never how pose
and structure are computed.

Measured on scene_02 (146 images), each with its own features, match graph and
reconstruction:

| detector | descriptor | cameras | points | mean px | track | seed threshold |
|---|---|---|---|---|---|---|
| SIFT | float32 | 144/146 | 17,192 | **1.261** | **7.06** | 3.0 deg |
| AKAZE | uint8 | **146/146** | 13,496 | 1.378 | 6.46 | 3.0 deg |
| ORB | uint8 | 144/146 | 29,360 | 1.349 | 3.71 | 1.2 deg |

SIFT builds the best-constrained model. AKAZE reaches every camera but chains
shorter tracks. ORB's large point count comes with the shortest tracks — one
surface point split into fragments rather than more structure — so **point
count alone is not a quality measure**.

The matcher's metric follows the descriptor type automatically: L2 for SIFT's
float histograms, Hamming for AKAZE and ORB's bit strings. This is not
configurable by accident — comparing a bit string under L2 returns plausible
numbers rather than an error, so the mistake would surface only as an
inexplicably poor reconstruction.

**ORB needed `seed_min_parallax_deg` relaxed to 1.2**, and that says more about
the capture than the detector: over the top 200 seed candidates SIFT found
exactly **one** pair above 3 deg and ORB found none. scene_02 was shot at about
two frames a second, so nearly every pair is a near-duplicate. Judging a
detector under another detector's thresholds measures the thresholds — use
`--set FIELD=VALUE` to give each one settings it can work with. Full detail in
`docs/HANDOFF.md`.

---

## 3. Pipeline

```
  data/<scene>/raw_images/            (read-only)
        |
        v
  [1] preprocess.py ................. quality metrics, near-duplicate detection,
        |                             accept/reject -> processed_images/ + review/
        v
  [2] sfm_core/features.py .......... SIFT + RootSIFT transform
        |                             mutual-ratio matching
        |                             own RANSAC + F for verification
        v
  [3] sfm_core/features.py .......... union-find tracks
        |
        v
  [4] sfm_core/mapper.py ............ seed pair (inliers AND parallax)
        |                             -> E -> cheirality -> triangulate
        |                             -> incremental P3P registration
        |                             -> local + global bundle adjustment
        |                             -> pose refinement, polish, outlier filters
        v
  [5] sfm_core/multimodel.py ........ reconstruct unreachable groups separately,
        |                             align by similarity where justified
        v
  [6] sfm_core/dense.py ............. plane-sweep MVS -> depth maps -> fusion
        v
  [7] sfm_core/surface.py ........... TSDF -> marching cubes -> mesh
        v
  [8] sfm_core/viewer.py ............ interactive HTML viewer + PLY exports
```

---

## 4. Installation

### 4.1 Python 3.12 and the virtual environment

Python is pinned to **3.12** because Open3D 0.19 publishes no wheels above
cp312. The machine default is 3.14, which would silently lose the optional
point-cloud analysis.

```bash
cd "$HOME/OneDrive/Desktop/sfm_project"
py -3.12 -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt
```

No COLMAP install is required. The deliverable implements SfM itself.

### 4.2 Software stack

| Component | Version | Role |
|---|---|---|
| Python | 3.12 | pinned by Open3D 0.19 |
| OpenCV (contrib) | 4.14 | SIFT, descriptor matching, image I/O, video decode |
| NumPy / SciPy | 2.5 / 1.18 | linear algebra; generic least-squares |
| scikit-image | 0.26 | marching cubes |
| Open3D | 0.19 | optional point-cloud statistics |
| pandas / matplotlib | 3.0 / 3.11 | metrics tables, plots |
| Pillow | 12.3 | EXIF, orientation |

### 4.3 Storage

The project root is inside OneDrive. Multi-GB dense outputs should be
redirected via `paths.output_root` to a local disk.

### 4.4 Verify

```bash
./.venv/Scripts/python.exe scripts/check_environment.py
```

---

## 5. Commands

### Reconstruct

```bash
./.venv/Scripts/python.exe scripts/run_scratch_sfm.py --scene scene_01 \
    --focal 1150 --min-ncc 0.45 --consistent-views 4 --multi-model
```

* `--reuse-sparse` re-tunes the dense stage without redoing SfM.
* `--multi-model` reconstructs groups the main model cannot reach (§8).
* `--no-dense` / `--no-surface` / `--no-viewer` stop after the sparse stage.
* `--feature {sift,akaze,orb}` selects the detector. SIFT is the strongest
  baseline and the only one RootSIFT applies to; AKAZE uses a nonlinear scale
  space and a binary descriptor; ORB is fastest and least robust to viewpoint
  change. The matcher's metric follows the descriptor type automatically —
  L2 for SIFT, Hamming for the binary pair — because comparing a bit string
  under L2 returns numbers rather than an error, and the damage would surface
  only as an inexplicably poor reconstruction.
* `--no-root-sift` reverts to raw SIFT descriptors (ablation only).
* Features and the verified match graph are cached under `outputs/<scene>/cache/`.

**`--focal 1150` is currently required** — see §10.

### Diagnose

```bash
./.venv/Scripts/python.exe scripts/diagnose_reconstruction.py --scene scene_01 \
    --focal 1150 --graph-only
```

Reports the **registration ceiling** (largest connected component of the
verified match graph — a hard upper bound no mapper can exceed), where the
capture physically broke, and whether a covisibility split is genuine. Without
`--graph-only` it also runs the mapper with culling disabled and tests each
bridging pair against the model.

### Ablate

```bash
./.venv/Scripts/python.exe scripts/run_scratch_ablation.py --scene scene_01 \
    --focal 1150 --experiment filters      # one reconstruction, many thresholds
./.venv/Scripts/python.exe scripts/run_scratch_ablation.py --scene scene_01 \
    --focal 1150 --experiment growth       # one reconstruction per condition
./.venv/Scripts/python.exe scripts/run_scratch_ablation.py --scene scene_01 \
    --focal 1150 --experiment calibration --calibration-part focal
./.venv/Scripts/python.exe scripts/run_scratch_ablation.py --scene scene_01 \
    --focal 1150 --experiment growth --knob min_triangulation_angle_deg \
    --values 1.25 1.5 1.75
```

Features and the match graph are loaded once, so conditions differ only in the
setting under test. Results are written after **every** condition, so a crash
does not discard completed work.

### Test

```bash
python -m unittest discover -s tests
```

Sixteen tests over synthetic scenes with known ground truth: essential-matrix
recovery and outlier rejection, two-view and n-view triangulation, P3P, PnP
RANSAC under 40% contamination, reprojection error, the 7-DoF similarity
solver (including its reflection guard and 80%-outlier robustness), and the
compliance audit.

### Compare against a LiDAR reference

```bash
python scripts/evaluate_against_lidar.py --help
```

Every accuracy figure elsewhere in this project is *self-consistency*:
reprojection error cannot detect a model that is globally wrong but internally
consistent. A LiDAR scan supplies an external metric reference and absolute
scale, which SfM does not have. `src/sfm_core/lidar_evaluation.py` handles PLY
I/O, PCA-based initial alignment, similarity estimation, ICP refinement and
point-to-cloud distance metrics. It needs a scan exported as PLY or OBJ; none
is included yet.

### Build the viewer

```bash
./.venv/Scripts/python.exe scripts/build_viewer.py --scene scene_01
```

---

## 6. Folder structure

```
sfm_project/
  configs/            default.yaml + per-scene overrides
  data/<scene>/
    raw_images/       read-only originals
    processed_images/ accepted images (stage 1 output)
    review/           copies of rejected images, never deleted
  outputs/<scene>/
    cache/            feature and match caches
    scratch/          sparse.ply, dense.ply, surface.ply, cameras.json,
                      summary.json, <scene>_reconstruction.html
    diagnostics/      match-graph and split reports
    ablations/        experiment CSV/JSON
    logs/             timestamped run logs
  src/
    sfm_core/         all reconstruction geometry
    camera.py config.py preprocess.py utils.py environment.py
    pointcloud_analysis.py visualization.py
  scripts/            CLI entry points
  docs/HANDOFF.md     project state, measured findings, open work
  _quarantine_violates_constraint/   excluded from the deliverable
```

### Key design rules

| Rule | Where enforced |
|---|---|
| Raw input is read-only | `preprocess.py` only copies out of `raw_images/` |
| Rejected images are copied to `review/`, never deleted | `materialise_images()` |
| Non-empty output dirs are not clobbered without `--force` | `utils.prepare_output_dir()` |
| Unknown config keys are errors, not silent defaults | `config._from_dict()` |
| Every run dumps the exact config it used | `ProjectConfig.dump_yaml()` |
| No hard-coded paths | everything derives from `paths.*` |

---

## 7. Validation against ground truth

Every geometric component is checked against a synthetic scene with known
poses before being trusted on real data. Regenerate with:

```bash
./.venv/Scripts/python.exe scripts/make_test_scene.py --mode room3d \
    --out data/_selftest3d --images 20
```

| Component | Result |
|---|---|
| Essential matrix | 0.073° rotation error with 25% outliers |
| Triangulation | exact to 7e-15 |
| P3P | 400/400 recovered, max error 2.4e-06° |
| Full pipeline | 20/20 registered, ATE 0.054% of scene extent |
| Plane-sweep depth | 0.4% median error vs sparse ground truth |
| Similarity (Umeyama) | scale error 1.3e-16; exact inlier recovery at 80% outliers; reflection guard holds |
| Similarity pose transform | reprojection unchanged to 1.7e-13 px |

**A caveat that cost real time.** `_selftest3d` passes at 20/20 and ~0.27 px on
both broken and fixed bundle adjustment: 20 clean images are too few for BA
damage to become unrecoverable. It validates the *geometry* and says nothing
about *convergence at scale*. A regression test that passes on broken code is
worse than none.

---

## 8. Multi-model reconstruction

Registering an image needs 2D-3D correspondences, and a track becomes a 3D
point only once **two** registered cameras observe it. An image joined to the
model by a single verified pair therefore cannot be resected at all — its
shared tracks have one registered observer, so none is triangulated, so it has
zero correspondences however many inliers that pair had.

Measured on scene_01: of 20 unregistered images, **18 had no verified pair to
any registered camera** and two had exactly one each (20 and 15 inliers). No
threshold reaches them; the deadlock is structural.

`--multi-model` reconstructs such a group as its own model, then estimates a
7-DoF similarity from bridging pairs — a feature triangulated in model A
matched to one triangulated in model B. On scene_01 this recovers all 20 images
at 1.2999 px, and then **refuses the merge**: only 2 usable correspondences
exist, and 7 degrees of freedom need far more. Two honest models beat one
confidently fused wrong one — a wrongly merged model is exactly the failure
reprojection error cannot see.

---

## 9. Measured findings worth defending in the report

**Reprojection error is a local statistic.** It cannot detect a globally split
reconstruction: a model showing 1.86 px and healthy track lengths was two
scenes in one coordinate frame. Only the render revealed it.

**Watch the median, not the RMSE, under a robust loss.** The optimiser
minimises the Huber cost; a large RMSE that will not move usually means a few
gross outliers being correctly ignored.

**A low reprojection error can be survivorship.** Tightening
`max_reprojection_px` from 8.0 to 2.0 improves the error to 0.82 px — by
discarding the reconstruction (85 cameras → 22). The number improves because
the evidence was deleted.

**RANSAC sample size dominates robustness.** 6-point DLT vs 3-point P3P is
5e-8 vs 2e-4 usable samples at a 94% outlier rate.

**A threshold normalised by the wrong quantity looks scale-free and is not.**
The sparse outlier filter first scaled by camera-path extent — correct for a
walking capture (median viewing distance 0.13× the path) and wrong for a room
capture (1.18×). It now scales by the cloud's own median viewing distance.

**Descriptors matter more than tuning.** RootSIFT — three lines of Hellinger
normalisation — raised scene_01's match-graph ceiling from 90 to 110 and made
the whole capture one connected component. No mapper change achieved that.

**SfM has no notion of vertical.** The world frame is fixed arbitrarily; up is
estimated from the camera path in `viewer.estimate_up_axis`.

### Settings, and the measurements behind them

| Setting | Value | Evidence |
|---|---|---|
| `min_triangulation_angle_deg` | 1.5 | 0.5 → 85 cams @ 2.19 px; **1.5 → 87 @ 1.58 px**; 2.0 → 77 @ 1.76. Plateau verified across 1.25–1.75. |
| `pnp_min_inliers` | 8 | 12, 16 and 20 all collapse to the *same* 27 cameras — a bottleneck crossable only on 8–11 correspondences. A fragility, not a virtue. |
| `max_reprojection_px` | 8.0 | 6.0 → 42 cams, 4.0 → 41, 2.0 → 22. Tightening discards the reconstruction. |
| `min_pair_inlier_ratio` / `floor` | 0.25 / 15 | Pair 71–74 had 27 inliers from 70 ratio-matches (39%, genuine) and was rejected by a flat floor; 84–85 had 20 from 229 (9%, noise). Ratio separates them. |
| `seed_candidates` | 200 | At 40, a 146-image burst offered only near-duplicates; the best had 0.051° parallax and triangulated nothing. |
| `local_ba_window` | 12 | The "local" BA previously optimised the whole model every image, sharing a 40-evaluation budget: 67% of adjustments improved RMSE by <1%. |
| `ba_max_iterations_global` | 200 | At nfev=200 a global BA improves RMSE 44–64%; at 40, ~4%. |
| `max_point_distance_median_factor` | 8.0 | Bounding box 15.9× → 2.9× the camera path, dropping <1% of points, with median reprojection unchanged. |

---

## 10. Known limitations

* **`--focal 1150` is required.** The focal is now strongly observable — 1.56 px
  at 1150 against 2.78 px at 1800, with registration halving away from the
  optimum — but `calibrate_focal` still lands at 923, because it scans on only
  8 trial images. Raising `focal_trial_images` is the obvious fix and is
  **untested**.
* **Only the primary model is exported.** A second model is recorded in
  `summary.json` but not written to disk.
* **`refine_distortion` quality was never re-measured.** Enabling it costs ~13×
  the runtime — not in the solver, but because every triangulation must then
  undistort its observations across ~40k tracks after every registration. It is
  off by default for that reason as well as the original quality one.
* **Data quality is the binding constraint.** 1.9 MP, WhatsApp-recompressed,
  EXIF stripped. Re-transferring originals by USB (~12 MP, EXIF intact) is the
  highest-value capture-side action available.
* **§15 measured-distance scale experiment** is outstanding; it needs a real
  measurement on site (or a LiDAR reference).

---

## 11. Capture guidance

Measured on this project's own captures, not inherited advice.

* **Never stop shooting while you move.** scene_01's registration problem traced
  to 11 consecutive frame pairs with *zero* verified inliers — places the camera
  moved without covering the gap. Those directly produced three misplaced
  cameras.
* **Overlap ≥ 70%.** scene_02 reached 51% pair survival and 144/146 registered;
  scene_01 managed 15% and 90/110.
* **Translate, do not pivot.** Pure rotation gives no baseline. A 146-frame
  burst at ~2 frames/second produced near-duplicate pairs with 0.051° of
  parallax, which triangulate nothing.
* **Do not machine-gun the shutter.** 146 frames over a proper orbit carry far
  more information than 146 frames in 70 seconds.
* **Choose a subject with depth at different orientations.** A single flat
  façade is degenerate. scene_02 works because the machine, floor and wall face
  different ways.
* **Rigid subjects only.** Fabric, foliage and water violate the rigid-scene
  assumption.
* **Close the loop** if you can; it strongly constrains bundle adjustment.
* **Lock exposure and focus**, keep one lens, avoid zoom changes mid-capture.
* **Avoid** moving people, mirrors, and large blank walls. Reflective glass is
  survivable — scene_02's vending machine reconstructed cleanly — but direct
  reflections are not.
* **Transfer by USB, not messaging apps.** Both scenes arrived at 1.9 MP with
  EXIF stripped.

Record location, time of day, device and capture path in `scene.description` /
`scene.capture_notes`; they are copied into every results file.

---

## 12. Reproducibility

Every run records the resolved configuration, a timestamped log, and
machine-readable metrics. Randomness is seeded via `runtime.seed`
(`utils.set_global_seed`), and every RANSAC call receives an explicitly derived
seed.

**This pipeline is deterministic.** Identical inputs and settings produce
identical output — verified during the ablation sweeps, where
`min_covisible_points` at 4, 8 and 15 produced byte-identical results (which is
how a dead configuration knob was discovered).

That is a genuine contrast with COLMAP, whose multi-threaded bundle adjustment
is not bit-deterministic: repeated runs of one condition varied by 0.15–9.37%
in sparse point count. Those figures were measured earlier in this project with
the now-quarantined comparison code and are reported as historical.

When comparing conditions, remember that **a difference smaller than the
measured noise floor is not a difference**. Since this pipeline has no noise
floor of its own, differences between conditions here are real — but they are
still subject to path dependence: incremental SfM can reach a different basin
from a different seed pair, which is why `min_triangulation_angle_deg` was
verified as a *plateau* across 1.25–1.75 rather than trusted at a single value.
