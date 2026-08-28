# From-scratch SfM: registration recall and accuracy — design

Date: 2026-08-21
Status: phases 1–2 complete and measured; see 'Outcome' at the end

## Problem

`run_scratch_sfm.py --scene scene_01` registers 41 of 110 images. The
HANDOFF attributes this to a capture that "splits into two arcs (41 + 39)
that never observe enough common structure", i.e. a data problem.

**Measurement contradicts that.** Building the connected components of the
cached verified-match graph at the pipeline's own `min_pair_inliers = 30`:

| min inliers | components | sizes |
|---|---|---|
| 30 | 5 | 88, 19, 1, 1, 1 |
| 50 | 8 | 87, 8, 7, 4, 1, 1 |
| 100 | 20 | 37, 27, 12, 6, 5, 4 |

At the threshold the pipeline actually uses, **88 images form one connected
component**. The genuinely separate pieces are frames 77–96 (19 images) and
three isolated frames (54, 74, 85).

The registered set is frames 1–33 plus 97–110 — the start and end of the
walk, joined by a *successful loop closure*. What is missing is the middle,
frames 34–96. The mapper registers 85 images and is then culled to 41.

So the shortfall is algorithmic, not photographic. Three causes:

1. **Contiguity test false-positives.** `_is_geometrically_contiguous`
   (mapper.py:694) single-linkage clusters camera centres at
   `median(step) × 8`. This capture has real position discontinuities —
   pairs 96↔97 and 76↔77 have zero verified inliers, so the photographer
   physically jumped. A legitimate gap is indistinguishable from a split
   under that test. It escalated `min_covisible_points` 15 → 30, breaking
   85 cameras into [41, 39] and discarding 44.

2. **`_next_image` abandons the whole set.** Returning `None` when the best
   candidate is below `pnp_min_inliers` drops every remaining image at once
   and records none of them in `failed`, so the retry pass never runs
   (`0 retry pass(es)` in the logs).

3. **No multi-model support.** A fragmented capture should yield several
   internally-consistent models, merged where they share structure. The
   mapper grows one model and discards the remainder.

## Constraint

No library function that implements the SfM task. OpenCV supplies image I/O,
SIFT and descriptor matching; NumPy linear algebra; SciPy a generic
least-squares solver; scikit-image a generic isosurface extractor. All
reconstruction geometry lives in `src/sfm_core/`.

## Open question

Whether the 85 registered cameras are correctly placed or genuinely two
overlaid scenes is **not settled** by the match graph. The HANDOFF records a
render showing two scenes in one frame. Phase 1 measures this rather than
assuming either way.

## Phase 1 — Instrumentation

New `src/sfm_core/diagnostics.py`. Pure measurement, no behaviour change.

* `match_graph_components(matches, n, min_inliers)` — connected components of
  the verified pair graph, the table above.
* `model_report(rec, tracks, matches)` — per-model covisibility components,
  camera-centre extent, and for each candidate split, whether the two halves
  are linked by verified pairs and whether those pairs' relative poses agree
  with the reconstructed ones. This is the evidence that decides the open
  question.
* Metrics recorded separately and honestly: `registered_total` (before any
  culling), `registered_component` (after), `num_components`.

## Phase 2 — Registration recall

* **`_next_image`**: sub-threshold candidates are added to `failed` and
  retried once the model has grown, instead of terminating the loop. The
  loop ends only when every remaining image has been tried against a model
  that has not advanced since.
* **Contiguity**: replace centre-spacing single-linkage with a match-graph
  test — two camera groups are contiguous when verified pairs bridge them
  *and* those pairs' relative poses are consistent with the reconstruction.
  A capture discontinuity has no bridging pairs to be inconsistent with, so
  it cannot false-positive; two overlaid scenes have bridging pairs whose
  relative poses disagree, which it detects.
* Both behaviours behind `MapperConfig` flags so the experiment runner can
  ablate them.

## Later phases (not yet designed in detail)

3. Multi-model reconstruction + 7-DoF similarity merge (Umeyama + RANSAC,
   written from scratch), replacing cull-to-largest.
4. Sparse accuracy: iterative BA↔filter with tightening threshold;
   statistical outlier removal for the 3.5× bounding-box inflation.
5. Dense: `mean_valid_fraction` is 0.083 — only 8.3% of pixels get a depth.
6. Scratch experiment runner (`src/scratch_experiments.py` +
   `scripts/run_scratch_experiments.py`) covering image_count, matcher,
   quality, scratch ablations, determinism, and the split-arc knobs.

## Testing

Phase 1 is validated against the numbers in the table above, which were
produced from the cached match graph and are reproducible.

Phase 2 is validated by re-running scene_01: registration must rise from 41
without the mean reprojection error degrading materially, and the diagnostics
must agree with what the render shows. `_selftest3d` must continue to pass
(20/20 registered, ATE 0.054% of extent) — a recall change that breaks the
synthetic ground truth is a regression, not an improvement.

## Expectations

Images are 1.9 MP, WhatsApp-recompressed, EXIF stripped. 41 → ~88 registered
is realistic. Matching COLMAP's 1.08 px reprojection error on this data may
not be. No reported figure comes from anywhere but an actual run.


## Outcome (measured 2026-08-21)

Phase 1 built `src/sfm_core/diagnostics.py` and
`scripts/diagnose_reconstruction.py`. Phase 2 went differently from the plan
above, and the differences are the useful part of the record.

### What the plan got wrong

**The `_next_image` fix was unnecessary.** The plan assumed abandonment was
costing registrations. Measurement showed the unregistered images were
exactly the match-graph-unreachable set `{54, 74, 77-96}`: the mapper was
already reaching 85 of the 88 reachable images, 97% of the ceiling. The
change was dropped.

**The contiguity test was a symptom, not the cause.** It escalated
`min_covisible_points` 15 -> 30 because the model genuinely *was* drifting
apart - not because the test was wrong. Fixing the drift stopped the
escalation firing at all.

### The actual cause: bundle adjustment

1. The "local" BA was never local. `_run_ba` passed the whole model's poses
   and observations every call; `global_ba` only toggled distortion
   refinement. With `local_ba_every=1`, a full 26k-56k-parameter global
   adjustment ran after every registered image.
2. All of them shared a `max_nfev` budget of 40. Measured: global BA at
   nfev=200 improved RMSE by 44-64% where nfev=40 managed ~4%.

Fixes: genuine local BA (`local_ba_window=12`, neighbouring cameras held
fixed as anchors), `ba_max_iterations_global=200`, calibration refined only
in global passes, an adaptive robust-loss scale, and BA termination reason,
`nfev` and median residual added to the log - all three were computed and
none were surfaced, which is why this survived so long.

### Result on scene_01

| metric | before | after |
|---|---|---|
| registered | 41/110 | 74/110 |
| registered before culling | 81-85 | 88 (100% of ceiling) |
| covisibility components | [67, 7, 4, 1, 1, 1] | [74, 14] |
| sparse points | 11,933 | 23,042 |
| mean reprojection | 1.85 px | 2.12 px |
| mapping time | ~470 s | 231 s |

Reprojection error rising is expected: the old figure was measured over a
much smaller culled subset.

`_selftest3d` is unchanged at 20/20 and 0.2631 px, before and after.

### Two measurement traps recorded for the report

* **RMSE is the wrong statistic to watch under a robust loss.** The optimiser
  minimises the Huber cost; a large RMSE that will not move usually means a
  few gross outliers being correctly ignored. The median residual is the one
  that shows whether the model is improving. Reading RMSE alone produced a
  confident but wrong first diagnosis in this very session.
* **A regression fixture can be too small to fail.** `_selftest3d` passes
  identically on the broken and fixed code, because 20 clean images never let
  the BA damage become unrecoverable. It validated the geometry correctly and
  said nothing at all about convergence at scale.
