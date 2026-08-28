# Start here — orientation for a new session

Read this first, then `HANDOFF.md` for the measured detail. Written 2026-08-29.

---

## 1. Which folder am I in?

There are five similarly-named folders on the Desktop. They are not copies of
each other and confusing them will waste an hour.

| Folder | What it is | Use it for |
|---|---|---|
| **`Structure From Motion`** | **Code only, ~825 KB.** The deliverable. No data, no outputs, no venv. | Handing in, editing, exporting |
| `sfm 3` | The same code **plus** `data/`, `outputs/` (1.3 GB of reconstructions) and a working `.venv` | Running experiments, regenerating results |
| `sfm 2` | Snapshot of the same work *before* the detector/LiDAR/test additions | Fallback only |
| `sfm_project` | **A different codebase entirely** — someone else's `src/sfm/`, SQLite feature store, alternating BA. Not ours. | Reference only. Do not edit. |
| `sfm_excluded_colmap_reference` | The old COLMAP comparison code, moved out of the tree because it violates the constraint | Never ship this |

A full backup lives at
`C:\Users\tharu\Documents\Codex\sfm_project_desktop_backup_20260827\`.

**`sfm 3` is where you run things.** `Structure From Motion` has no images or
caches, so a run there re-extracts and rematches from scratch (~25 min for 146
images).

---

## 2. The constraint — the single most important rule

> You may call some existing function in your development but cannot call
> existing functions that directly implement the SfM task.

**The line: anything that turns pixels into features is allowed; anything that
turns features into geometry is ours.**

Allowed and used: OpenCV SIFT/AKAZE/ORB detection and description, descriptor
matching, image I/O and resampling; NumPy linear algebra; SciPy's *generic*
least-squares solver; scikit-image's *generic* marching cubes.

Forbidden and absent: COLMAP, pycolmap, OpenSfM, and every OpenCV geometry
solver (`findEssentialMat`, `recoverPose`, `findFundamentalMat`, `solvePnP`,
`triangulatePoints`, …), plus Open3D's registration / TSDF / Poisson entry
points.

**Verify before believing:**

```bash
python scripts/audit_sfm_compliance.py     # exit 0 = clean
python -m unittest discover -s tests       # 16 tests
```

The audit parses the syntax tree, so comments and docstrings cannot raise false
alarms and our own `bundle_adjust` is not mistaken for a library one. A
grep-based predecessor reported 68 findings on this same compliant tree, every
one a false positive. `tests/test_compliance.py` plants a real violation and
checks the audit rejects it, so a refactor that defangs the check fails the
suite.

**Do not "helpfully" add** `cv2.findEssentialMat` or similar to fix a bug. Every
one of those is implemented in `src/sfm_core/`.

---

## 3. Results

| Metric | scene_01 (carpark, 110 img) | scene_02 (vending machine, 146 img) |
|---|---|---|
| Registered | 90/110 (+20 in a 2nd model) | **144/146** |
| Mean / median reprojection | 1.3275 / 1.0477 px | **1.2609 / 1.0832 px** |
| Mean track length | 3.90 | **7.06** |
| Dense points | 5,154,161 | **13,195,325** |
| Mesh faces | 395,134 | **941,204** |

scene_01 began this project at **41/110 and 1.85 px**. Everything since came
from measurement, not tuning by feel.

`--focal 1150` is currently required (see §6).

---

## 4. What is in the code

`src/sfm_core/` — all reconstruction geometry, written here:

| Component | File |
|---|---|
| 8-point F, essential + RANSAC, cheirality, DLT triangulation, RANSAC PnP, 7-DoF similarity | `geometry.py` |
| P3P (Grunert) + pose refinement | `p3p.py` |
| Bundle adjustment — axis-angle, sparse Jacobian, Huber | `bundle.py` |
| SIFT/AKAZE/ORB + RootSIFT, matching, tracks | `features.py` |
| Incremental mapper | `mapper.py` |
| Plane-sweep MVS | `dense.py` |
| TSDF + marching cubes | `surface.py` |
| Multi-model reconstruction + similarity merge | `multimodel.py` |
| Split / match-graph diagnostics | `diagnostics.py` |
| LiDAR comparison (ICP written from scratch) | `lidar_evaluation.py` |

---

## 5. Commands

```bash
# reconstruct
python scripts/run_scratch_sfm.py --scene scene_01 --focal 1150 \
    --min-ncc 0.45 --consistent-views 4 --multi-model

# is this capture even reconstructable? (minutes, run this FIRST on new data)
python scripts/diagnose_reconstruction.py --scene scene_01 --focal 1150 --graph-only

# experiments
python scripts/run_scratch_ablation.py --scene scene_01 --focal 1150 \
    --experiment {filters|growth|calibration|brief|detector}

# override any MapperConfig field
... --set seed_min_parallax_deg=1.2 --set min_triangulation_angle_deg=0.8
```

`--feature {sift,akaze,orb}` picks the detector. `--video clip.mp4 --video-fps 2`
adds video; each video is its own camera group with its own focal, because a
phone's video is cropped relative to its stills.

---

## 6. Open items

**Blocked on the user**

* **LiDAR comparison** — the highest-value item left. The module is ported,
  wired and verified working on our clouds (recovers scale to 1e-15 on a
  self-alignment test). It needs one scan exported as PLY or OBJ. Until then
  *every accuracy number in this project is self-consistency*, and reprojection
  error cannot detect a model that is globally wrong but internally consistent.
* §15 measured-distance experiment — a tape measure on site, or the LiDAR scan
  serves the same purpose better.

**In progress at time of writing**

* `focal_trial_images` was raised 8 → 24 to make self-calibration work; the
  verification run had not finished. If it recovers ~1150 on scene_02 (rather
  than the 923 the 8-image probe chose), `--focal` stops being mandatory.
  **Check this before trusting it.**

**Small and optional**

* scene_01's second model (20 cameras) exists in memory and is recorded in
  `summary.json`, but is never written to disk.
* `refine_distortion` quality has never been measured — enabling it costs ~13x
  the runtime, because every triangulation must then undistort its observations
  across ~40k tracks after every registration.

---

## 7. Traps that already cost time — do not rediscover them

* **RMSE is the wrong statistic under a robust loss.** The optimiser minimises
  the Huber cost; a large RMSE that will not move usually means a few gross
  outliers being correctly ignored. Watch the *median* residual. Reading RMSE
  alone produced a confident and wrong diagnosis.
* **A low reprojection error can be survivorship.** Tightening
  `max_reprojection_px` from 8.0 to 2.0 "improves" the error to 0.82 px by
  discarding the reconstruction (85 cameras → 22). The number improves because
  the evidence was deleted.
* **The synthetic fixture cannot catch a convergence bug.** `_selftest3d`
  passes at 20/20 and ~0.27 px on both broken and fixed bundle adjustment —
  20 clean images are too few for the damage to become unrecoverable.
* **A threshold normalised by the wrong quantity looks scale-free and is not.**
  The sparse outlier filter first scaled by camera-path extent: right for a
  walking capture (median viewing distance 0.13x the path), wrong for a room
  capture (1.18x). It now scales by median viewing distance.
* **End-of-pipeline filters must run after the last `_triangulate_all()`.**
  Placed earlier, their effect is silently undone by the re-triangulation
  inside `_keep_largest_component()`.
* **Judging a detector under another detector's thresholds measures the
  thresholds.** ORB "failed" on scene_02 until `seed_min_parallax_deg` was
  relaxed to 1.2 — and the reason was the *capture*: over the top 200 seed
  candidates SIFT found exactly **one** pair above 3 degrees and ORB found none.
* **Never report a wall-clock time without sanity-checking it.** A dense stage
  reported 15.8 hours for work that runs in ~35 minutes; the machine slept
  mid-run while the timer kept counting.

---

## 8. Working agreement

Build stage by stage and wait for the user's execution results before moving
on. **Never fabricate experimental numbers** — every figure reported must come
from an actual run. The `_selftest*` scenes are software fixtures and their
numbers are never results.
