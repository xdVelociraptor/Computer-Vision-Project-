# Project state — handoff

Read this first in a new session. It records what exists, what works, what is
known-broken, and what to do next.

## What this project is

NTU Structure-from-Motion coursework. **Constraint: no library that implements
the SfM task may be called.** COLMAP and pycolmap are therefore excluded from
the deliverable; everything geometric is implemented in `src/sfm_core/`.

Two pipelines exist:

| Pipeline | Entry point | Status |
|---|---|---|
| **From-scratch** (the deliverable) | `scripts/run_scratch_sfm.py` | Working |
| COLMAP-based (comparison only) | `scripts/run_reconstruction.py` | Working, **not submittable** |

The COLMAP one is kept solely to provide comparison numbers in the report.

## Working command

```bash
cd "$HOME/OneDrive/Desktop/sfm_project"
./.venv/Scripts/python.exe scripts/run_scratch_sfm.py --scene scene_01 \
    --focal 1150 --min-ncc 0.45 --consistent-views 4
```

Add `--reuse-sparse` to re-tune the dense stage in ~5 min without redoing SfM.
Features and the verified match graph are cached under `outputs/<scene>/cache/`.

Two diagnostic tools added 2026-08-21:

```bash
# Registration ceiling, capture gaps, and whether a split is real.
# Seconds with --graph-only; one mapping pass without it.
./.venv/Scripts/python.exe scripts/diagnose_reconstruction.py --scene scene_01     --focal 1150 --graph-only

# Controlled ablations of the mapper's own settings. 'filters' needs one
# reconstruction total; 'growth' needs one per condition.
./.venv/Scripts/python.exe scripts/run_scratch_ablation.py --scene scene_01     --focal 1150 --experiment filters
```

Changing the pair-acceptance rule or the feature settings invalidates the match
cache by design (they are part of its key), so the next run rematches — about
19 minutes for 5,995 pairs.

## Current results (scene_01, 110 WhatsApp-compressed images)

Measured 2026-08-21 after the bundle-adjustment fixes. The "before" column is
what this file used to report.

| Metric | From-scratch (before) | From-scratch (now) | COLMAP |
|---|---|---|---|
| Registered | 41/110 | **87/110** | 110/110 |
| Covisibility components | [41, 39, 1, 1], culled to 41 | **[87, 1]** | 1 |
| Sparse points | 11,933 | 17,967 | 31,607 |
| Reprojection error (mean) | 1.85 px | **1.560 px** | 1.08 px |
| Reprojection error (median) | — | **1.124 px** | — |
| Sparse bbox / camera path | 15.9x | **2.91x** | — |
| Dense points | 1,513,112 | **4,240,674** | 1,065,519 |
| Dense views | 38 | **86** | — |
| Dense valid fraction | 0.083 | **0.1027** | — |
| Mesh faces | 181,614 | **437,506** | 633,293 |
| Mapping time | ~470 s | 459 s | 209 s |

The mean reprojection error is now *below* the original figure while covering
more than twice as many cameras, so it is no longer a survivorship statistic:
the old 1.85 px described a 41-camera subset that had been culled precisely
because it fitted.

The registration ceiling — the largest connected component of the verified
match graph — is **90 of 110**. Frames 77–96 form their own 19-image
component and frame 85 is isolated; no mapper can reach them from the main
model. **87 of the 90 reachable images are registered.**

## Validated against ground truth (synthetic scene `_selftest3d`)

* Essential matrix: 0.073° rotation error with 25% outliers
* Triangulation: exact to 7e-15
* P3P: 400/400 exact recovery, max error 2.4e-06°
* Full pipeline: 20/20 registered, 0.264 px, ATE 0.054% of scene extent
* Plane-sweep depth: 0.4% median error vs sparse ground truth

Regenerate the fixtures with:
`python scripts/make_test_scene.py --mode room3d --out data/_selftest3d --images 20`

## Hard-won settings — do not "tidy" these

Each carries the measurement that justifies it in a code comment. **Several
were re-derived on 2026-08-21** once bundle adjustment was fixed — see the
next section. A measurement is only as good as the pipeline it was taken on,
and these were taken on a broken one.

* ~~`min_triangulation_angle_deg = 0.5` — the textbook 2° gave 19 registered
  images, 0.5 gave 65.~~ **Superseded: now 1.5.** That sweep was run while BA
  was not converging. Re-measured, a stricter floor gives *more* cameras and
  much better geometry.
* `pnp_min_inliers = 8` — P3P produces tighter inlier sets than the old DLT.
  Re-measured and still 8, but it is a fragility rather than a virtue: 12, 16
  and 20 all collapse the reconstruction to the same 27 cameras.
* `refine_distortion = False` — distortion trades against focal and structure;
  on gave 39 images, off gave 85.
* `refine_focal` frozen after calibration — BA drifts a weakly observable
  focal (1176 → 1543) and drags registration down with it.
* `--focal 1150` — **self-calibration does not work on this data.** Across a 2×
  focal range the reprojection error varies by only ~0.1 px, below the feature
  noise of recompressed images. Independently corroborated: COLMAP's estimate
  is 1147.5.
* Dense: outliers removed before the depth-range percentile, far/near ratio
  capped at 18 (was 1,722,915 — the cause of radial streaking).
* `min_consistent_views = 4` — the strongest control on dense noise.

The entries above that are *not* struck through were not re-measured after the
BA fix. `refine_distortion`, `refine_focal` and the dense settings were all
tuned under the same broken adjustment, so their justifications carry the same
risk as the two that turned out to be wrong. They are plausible, not
confirmed.

## Settings re-derived after the bundle-adjustment fix (2026-08-21)

The original list was tuned while BA was failing to converge, on the objective
"register more images". Each suspect was re-swept with
`scripts/run_scratch_ablation.py --experiment growth`. One changed; two were
vindicated. Being wrong about which is which is exactly why they were measured
rather than reasoned about.

### `min_triangulation_angle_deg`: 0.5 -> **1.5**

| value | cameras | points | mean reproj | bbox / path |
|---|---|---|---|---|
| 0.50 (old) | 85 | 22,275 | 2.189 px | 15.92x |
| 1.25 | 88 | 17,776 | 1.721 px | 16.31x |
| 1.40 | 89 | 16,679 | 1.785 px | 10.27x |
| **1.50** | 87 | 18,142 | **1.578 px** | 7.83x |
| 1.60 | 84 | 16,901 | 1.715 px | 5.72x |
| 1.75 | 88 | 14,614 | 1.964 px | 5.56x |
| 2.00 | 77 | 16,012 | 1.759 px | 4.34x |

The old comment claimed "2.0 -> 19 images registered, 0.5 -> 65". That was
true of the broken pipeline and is false now: a *stricter* floor gives more
cameras and much better geometry, because the near-parallel-ray points it
excludes were misleading the adjustment rather than supporting it.

1.25-1.75 is a plateau (84-89 cameras), so 1.5 is not a lucky single sample -
incremental SfM is path-dependent and the neighbours were checked on purpose.
It costs roughly twice the mapping time: better structure means larger,
better-conditioned adjustment problems.

### `pnp_min_inliers`: stays at **8**, but for a worse reason than assumed

| value | cameras | points |
|---|---|---|
| 8 | 85 | 22,275 |
| 12 | 27 | 6,927 |
| 16 | 27 | 6,927 |
| 20 | 27 | 6,927 |

12, 16 and 20 give *identical* results, which means growth stalls at the same
place every time. Past 27 cameras this capture's only way forward is an image
sharing 8-11 points with the model; cross that bottleneck and images with
300-1400 inliers become reachable again.

Treat this as a known fragility, not a good property. A pose accepted on 8
correspondences is thin. The value is low because the capture demands it, and
the principled fix is a stricter pose check at bottlenecks rather than a
lower bar everywhere.

### `max_reprojection_px`: stays at **8.0**

| value | cameras | points | mean reproj |
|---|---|---|---|
| 8.0 | 85 | 22,275 | 2.189 px |
| 6.0 | 42 | 8,073 | 1.661 px |
| 4.0 | 41 | 7,989 | 1.274 px |
| 2.0 | 22 | 3,483 | 0.818 px |

Tightening buys a better-looking error by discarding most of the
reconstruction. The low error at 2.0 px is survivorship, not accuracy - the
same trap that made the old 1.85 px figure look respectable. 8.0 is doing
necessary work for 1.9 MP recompressed images.

## Known issues

1. ~~**41/110 registration.** The capture genuinely splits into two arcs~~
   **CORRECTED 2026-08-21 — this diagnosis was wrong.** The capture does not
   split into 41 + 39. Connected components of the verified match graph at the
   pipeline's own `min_pair_inliers` are **[88, 19, 1, 1, 1]**: 88 images are
   reachable in one component, and the registered set was frames 1–33 + 97–110,
   i.e. the start and end of the walk joined by a *successful loop closure*.
   The missing part was the middle, frames 34–96.

   The real cause was bundle adjustment (see "Bundle adjustment" below).
   Registration is now 85/110 in a single covisibility component, with nothing
   culled. Re-measure any time with:

   ```
   python scripts/diagnose_reconstruction.py --scene scene_01 --focal 1150 --graph-only
   ```

2. **Data quality is still a real constraint**, but it is no longer the
   binding one for registration. Images are 1600×1200 (1.9 MP) with EXIF
   stripped by WhatsApp. Re-transferring the originals by USB (~12 MP, EXIF
   intact) remains the highest-value capture-side action. The genuinely
   unreachable images are frames 77–96, which form their own 19-image
   component, plus frame 85.
3. **Sparse cloud sprawl — measured and fixed 2026-08-21.** The cloud's
   bounding box was 15.9x the camera-path extent. See "Sparse outlier
   filtering" below.

## Bundle adjustment — the fault that caused all of this

Two compounding bugs, both fixed 2026-08-21:

1. **The "local" BA was never local.** `_run_ba` passed the entire model's
   poses and observations on every call; the `global_ba` flag only toggled
   distortion refinement. With `local_ba_every = 1`, a full 26k–56k-parameter
   global adjustment ran after *every registered image*.
2. **All of them shared a `max_nfev` budget of 40**, far too small for a
   problem that size. Measured: a global adjustment at `nfev=200` improved
   RMSE by 44–64% where `nfev=40` managed about 4%.

Across the 110-image run, 67% of adjustments improved RMSE by under 1%, the
median post-adjustment RMSE was 16.9 px, and 96% finished above 5 px. The
1.85 px that used to be reported came almost entirely from the final filter
discarding whatever did not fit, not from optimisation.

Fixes: a genuine local BA (`local_ba_window = 12`, neighbouring cameras held
fixed as anchors), `ba_max_iterations_global = 200`, calibration refined only
in global passes, and an adaptive robust-loss scale. The BA log now reports
termination reason, `nfev` and the median residual — all three were being
computed and none were shown, which is why this went unnoticed.

**Watch the median, not the RMSE.** Under a Huber loss the optimiser
minimises the robust cost; a large RMSE that refuses to move usually means a
few gross outliers being correctly ignored. Reading RMSE alone produces a
confident and wrong diagnosis.

**`_selftest3d` cannot catch this.** It passes at 20/20 and 0.2631 px on both
the broken and the fixed code — 20 clean images are too few for the damage to
become unrecoverable. It validates the geometry and says nothing about
convergence at scale.

## Experiments 11-13, run through the from-scratch pipeline (2026-08-27)

Produced by `run_scratch_ablation.py --experiment brief`. The earlier numbers
in this file came from the COLMAP path, which is now excluded from the
deliverable; these replace them.

image_count and quality slice the already-computed match graph rather than
rematching. That is equivalent - whether a pair verifies does not depend on
which other images are present - and it makes matching an exact constant
across conditions, so differences are attributable to the variable under test.
The matcher experiment cannot do that, since comparing exhaustive against
sequential is partly a comparison of matching cost, so both of its arms match
from scratch and both times are measured.

### 11. Image count

| images | verified pairs | registered | rate | points | mean px | track |
|---|---|---|---|---|---|---|
| 25 | 36 | 3 | 12% | 462 | 0.663 | 2.12 |
| 50 | 180 | 33 | 66% | 6,893 | 1.155 | 3.04 |
| 75 | 422 | 55 | 73% | 12,185 | 1.083 | 3.36 |
| 100 | 737 | 69 | 69% | 14,391 | 1.313 | 3.68 |
| 110 | 919 | **90** | **82%** | 18,080 | 1.327 | 3.90 |

Verified pairs grow much faster than image count - 4.4x the images gives 25x
the pairs - which is why registration collapses below 50 images: at 25, only
36 of 300 possible pairs verify and the mapper has nothing to traverse.

The step from 100 to 110 is non-monotonic in rate (69% -> 82%) for ten extra
images. Those are the frames that close the loop back to the start, and they
are worth far more than their count. Runtime is roughly flat from 75 onward
because mapping cost tracks *registered* cameras, not input images.

### 12. Quality filtering

| arm | images | pairs | registered | points | mean px |
|---|---|---|---|---|---|
| all | 110 | 919 | **90** | 18,080 | 1.327 |
| blur_p15 | 93 | 810 | 77 | 17,699 | **1.256** |
| blur_p30 | 77 | 663 | 63 | 13,184 | 1.295 |

Filtering improves accuracy and costs coverage. Dropping the blurriest 15%
gives the best reprojection error measured anywhere in this project (1.256 px)
while losing 13 cameras; at 30% the error stops improving and 27 cameras are
gone. Removing weak images also removes overlap - the same mechanism as the
image-count collapse. The sign of the effect is an empirical result for this
scene, not a general rule.

### 13. Matcher

| matcher | pairs | registered | points | mean px | matching time | total |
|---|---|---|---|---|---|---|
| exhaustive | 919 | **90** | 18,080 | 1.327 | **992 s** | 1,471 s |
| sequential (w=10) | 631 | 76 | 17,525 | **1.264** | **142 s** | 876 s |

Sequential matching is 7x cheaper and costs 16% of registrations. It finds 631
of the 919 pairs, missing precisely the non-sequential ones - which is where
loop closures live, and the loop closure is what §11 shows to be worth the
most per image.

## Detector comparison (2026-08-28)

Run with `run_scratch_ablation.py --experiment detector`. Each detector gets
its own feature extraction, its own match graph and its own reconstruction;
nothing else varies.

### scene_02 (vending machine, 146 images)

| detector | descriptor | keypoints | verified pairs | cameras | points | mean px | track | seed threshold |
|---|---|---|---|---|---|---|---|---|
| SIFT | float32 | 2,893 | 5,432 | 144/146 | 17,192 | **1.261** | **7.06** | 3.0 deg (default) |
| AKAZE | uint8 | 2,834 | 5,091 | **146/146** | 13,496 | 1.378 | 6.46 | 3.0 deg (default) |
| ORB | uint8 | 5,579 | 4,524 | 144/146 | **29,360** | 1.349 | 3.71 | **1.2 deg (relaxed)** |

**SIFT builds the best model**: longest tracks and lowest reprojection error.
**AKAZE reaches every camera** - its nonlinear scale space finds correspondences
on smooth painted panels where a Gaussian pyramid blurs weak gradients away -
but chains them into shorter tracks. It also had the *highest* median inlier
ratio (0.384 against SIFT's 0.365), so its matches are individually cleaner;
the deficit is track length, not match quality.

**Read ORB's point count carefully.** 29,360 points with a 3.71 track length is
not "more structure": short tracks mean one physical surface point is split
across several fragments instead of chaining into one. More points, each less
constrained. Some of the gain is also the relaxed 0.8 deg triangulation angle
admitting lower-parallax points. Point count alone is a bad quality metric.

### ORB needed its own seed threshold, and why

ORB first failed outright - `Seed pair triangulated only 2 points`. That was
**not** a verdict on ORB. Measured over the top 200 seed candidates:

| detector | max parallax | pairs >= 3 deg | pairs >= 1.5 deg |
|---|---|---|---|
| SIFT | 3.14 deg | **1** | 20 |
| ORB | 1.98 deg | **0** | 1 |

**SIFT cleared the default 3 deg seed requirement by exactly one pair out of
200.** scene_02 is 146 frames shot in about 70 seconds, so nearly every pair is
a near-duplicate with no baseline; ORB's slightly coarser corner localisation
was enough to push its best candidate below the bar and leave it nothing to
start from. Relaxing `seed_min_parallax_deg` to 1.2 made it reconstruct.

The whole scene_02 result therefore rests on a single qualifying seed pair.
That is worth stating in the report: it is a **capture** limitation surfaced by
a detector change, not a property of the detectors.

Reprojection was never the constraint. For both detectors, 100% of
cheirality-valid seed points reprojected under 8 px (median 0.20 px SIFT,
0.32 px ORB). Parallax alone was binding.

### Comparing detectors fairly

Judging a detector under another detector's thresholds measures the thresholds,
not the detector. `--set FIELD=VALUE` exists for this:

```
python scripts/run_scratch_ablation.py --scene scene_02 --focal 1150     --experiment detector --detectors orb     --set seed_min_parallax_deg=1.2 --set min_triangulation_angle_deg=0.8
```

Note also that match *times* are not comparable across arms when one is served
from cache: SIFT reported 2.5 s against AKAZE's genuine 2,034 s for the same
10,585 pairs.

## Outstanding from the original brief

* ~~scene_02 not captured~~ **DONE** — a vending machine, 146 images,
  144/146 registered at 1.2609 px. Rigid, well-textured, genuinely 3D; the
  best reconstruction in the project and a real cross-scene test, since it
  shares no tuning with scene_01.
* ~~Experiments §11-13 were run through COLMAP~~ **DONE** — re-run through the
  from-scratch pipeline, see the section above.
* **Measured-distance scale experiment (§15) not done** — needs a real
  measurement on site. An iPhone LiDAR scan would serve the same purpose and
  give a metric reference for the whole cloud rather than a single distance;
  the similarity solver needed to align it already exists and is validated.
### Self-calibration (2026-08-29)

`calibrate_focal` now chooses **f = 1079.9 px** on scene_02 against a true
~1150, an error of **-6.1%**. It previously chose 923 (-19.7%). Two stacked
faults, and fixing the first exposed the second:

1. `focal_trial_images` was **8**, too few for the minimum to be readable.
   Raised to 24, after which the probe shows f=1120 at **0.9593 px** against
   ~1.33 px either side - an unmistakable minimum.
2. The probe then **threw that minimum away**. It ranked candidates by
   registered view count *before* error, and f=1120 registered 23 of 24 views
   while its neighbours registered 24. One missing view outweighed a 26%
   better fit, so the golden-section search refined inside the wrong bracket
   and returned 1303.

The old ranking was not careless - it was written to stop a real failure, where
plain argmin-on-error picked f=1600 at 7-of-8 views (1.67 px) over f=1120 at
8-of-8 (1.00 px), because a partial reconstruction fits fewer constraints and
so reports a lower error. It simply over-corrected. `_rank_focal_trials` now
tolerates a view deficit proportional to probe size: one view out of eight is
meaningful, one out of twenty-four is not. Verified against both historical
failures and against a genuine partial reconstruction, which must still lose.

**It stops at 1080 rather than 1150 because the probe optimises a proxy** - the
24-image subset's error minimum sits slightly below the full 146-image
optimum. More trial images would close the gap at proportional cost; the probe
already takes **62 minutes**.

**Supply `--focal` when you know it.** Self-calibration is a usable fallback
for an unknown device, not an equal. And this is one measurement on one scene -
`refine_focal` helped on scene_02 and wrecked scene_01, so test before
generalising.

* **`--focal 1150` is still recommended.** The focal is strongly observable now
  (1.56 px at 1150 against 2.78 px at 1800, registration halving away from the
  optimum) but `calibrate_focal` lands at 923 because it scans on only 8 trial
  images. Raising `focal_trial_images` is the obvious fix and is untested.
* **Only the primary model is exported.** A second model is recorded in
  `summary.json` but not written to disk.
* **`refine_distortion` quality never re-measured** — enabling it costs ~13x
  the runtime, because every triangulation must then undistort its
  observations across ~40k tracks after every registration.

## Lessons that belong in the report

* Reprojection error is a **local** statistic and cannot detect a globally
  split reconstruction. A model showed 1.86 px and healthy track lengths while
  being two scenes in one coordinate frame. Only the render revealed it.
* RANSAC sample size dominates robustness: 6-point DLT vs 3-point P3P is
  5e-8 vs 2e-4 usable samples at a 94% outlier rate.
* Monocular self-calibration needs the focal-induced signal to exceed feature
  noise; on lossily recompressed images it does not, and no algorithm recovers
  information that is not in the data.
* SfM fixes the world frame arbitrarily, so a reconstruction has no notion of
  vertical. Up is estimated from the camera path in `viewer.estimate_up_axis`.

## Debugging practice that worked

Validate every geometric component against synthetic ground truth before
trusting it on real data, and **look at the render** — eleven bugs were found
by measurement, and the split was found by looking. Iterate on a ~25-image
subset (`--max-images 25 --no-dense`, ~3 min), not the full set.

## Sparse outlier filtering

Points triangulated along near-degenerate rays end up far from the scene while
still reprojecting correctly in the two or three views that see them - moving
a point along its viewing ray is exactly the direction the observations cannot
constrain. No reprojection test can catch them, which is why `_filter_points`
never did.

Two candidate criteria were ablated on one reconstruction (85 cameras, 22,275
points) with `scripts/run_scratch_ablation.py --experiment filters`:

Swept on the old permissive cloud (`min_triangulation_angle_deg = 0.5`,
22,275 points), where the distances are expressed against the camera-path
extent:

| filter | points kept | bbox / path extent | median reproj |
|---|---|---|---|
| none | 100% | **15.92x** | 1.662 px |
| parallax >= 0.75 deg | 86.4% | 13.37x | 1.695 px |
| parallax >= 1.5 deg | 74.3% | 6.83x | 1.662 px |
| parallax >= 3.0 deg | 61.2% | 4.92x | 1.700 px |
| distance <= 3.0x path | 98.4% | 6.78x | 1.650 px |
| distance <= 2.0x path | 91.5% | 5.13x | 1.649 px |
| distance <= 1.0x path | 85.2% | 3.31x | 1.624 px |

Re-swept on the final cloud (`min_triangulation_angle_deg = 1.5`), against the
median viewing distance, which is what the knob actually uses. The two fixes
overlap: the stricter growth-time angle prevents most far outliers from ever
being triangulated, so far less is left to remove.

| filter | points kept | bbox / path extent |
|---|---|---|
| 3x median | 98.3% | 1.79x |
| 5x median | 98.8% | 2.32x |
| **8x median** | **99.1%** | **2.91x** |
| 15x median | 99.5% | 3.96x |

**Parallax is a poor discriminator here.** The cloud's median parallax is
5.24 deg, which is healthy, and a distant point observed from two widely
separated cameras subtends a perfectly respectable angle. At matched retention
(~87%) the distance criterion reaches 4.37x where parallax manages only
13.37x.

Two independent signals confirm the removed points are genuinely bad rather
than legitimate far-field structure: median reprojection error *improves*
slightly as they go, and median parallax rises from 5.24 to 7.22 deg - so the
distant points are also the low-parallax ones, exactly the degenerate-ray
failure mode.

**There is no knee in the data**, so no threshold is derivable from it: the
bbox ratio falls smoothly across the whole sweep (p99 distance 3.37x, max
9.48x). The chosen setting is a judgement, to be revisited against the table
above rather than guessed at. `final_min_triangulation_angle_deg` exists for
the same stage but defaults to 0 (off), because the measurement above says
parallax is not the right axis for this scene.

### Ordering: the filters must run last

`_keep_largest_component()` calls `_triangulate_all()`, which rebuilds every
track it can. Running the structure filters before it removes points that are
then silently re-added — the first version of this did exactly that, dropping
103 points where it should have dropped 175, and leaving the bounding box at
7.83x instead of 2.91x. They now run after the final re-triangulation.

### Normalise by viewing distance, not by the camera path

The first version of this filter scaled the threshold by the camera-path
extent. On `_selftest3d` it discarded 36% of the points in one focal-
calibration sub-model while costing the final model only 3.4% (4,239 -> 4,094
points, 0.2631 -> 0.2736 px, still 20/20 registered). That *instability* is
the tell: one threshold meaning wildly different things across sub-models of a
single scene, because the ratio it normalises by is not a property of the
scene. The two captures differ in a way a path-relative threshold cannot
express:

| scene | median viewing distance | max / median |
|---|---|---|
| scene_01 (walking) | 0.13x the path extent | 71.7 |
| `_selftest3d` (room) | 1.18x the path extent | 3.4 |

A walking capture covers far more ground than it sees; a room capture barely
moves while viewing everything. So the knob is
`max_point_distance_median_factor`, a multiple of the cloud's own **median
viewing distance**, which means the same thing in both. At the default of
**15x**, scene_01 keeps 91.2% with its bounding box down from 15.9x to 5.1x,
and `_selftest3d` keeps 100% - correctly, since its cloud has no tail at all.

This is worth remembering generally: a threshold expressed against the wrong
reference quantity can be right on the scene it was tuned on and badly wrong
on the next one, while looking like a principled scale-free choice in both.

This should also reduce the dense stage's reliance on its own workarounds:
`depth_outlier_factor` and the `max_depth_range_ratio = 18` cap exist because
the raw sparse cloud once produced a far/near ratio of 1,722,915. Removing the
outliers at source addresses that where the cap only contained it.
