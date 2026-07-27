# Phase 2 - Standard POS/CHROM Baselines

## Material Passport

- Artifact: formal Phase 2 experiment result
- Inputs: frozen Phase 1 windows and existing local RGB traces
- Dataset scope: 42 UBFC-rPPG subjects available in the local dataset
- Verification status: reproduced locally
- External data upload: none

## Scope

Phase 2 estimates window-level heart rate from facial RGB traces using standard
classical rPPG methods. It does not train a reliability model and does not use
reference error to select signals, regions, parameters, or windows.

The primary reference remains the Phase 1 time-domain contact-PPG estimate
(`time_hr_bpm`). The Phase 1 frequency estimate and reference category remain
quality-control fields.

## Frozen Processing Method

1. Load the existing `rgb_trace.csv` for each subject. Videos and face
   landmarks are not processed again.
2. Use five regions:

   - `full_face_inner`;
   - `forehead`;
   - `left_cheek`;
   - `right_cheek`;
   - `cheeks_mean`.

3. Resample valid RGB observations to the median video timestamp interval.
4. Extract a continuous signal for each region using:

   - CHROM: temporally normalized chrominance projections, projection-domain
     0.7-4.0 Hz filtering, 1.6-second internal windows, and 50% overlap-add;
   - POS: temporally normalized plane-orthogonal projections, 1.6-second
     sliding windows, alpha tuning, and sample-wise overlap-add.

5. Apply the same final linear detrending and zero-phase 0.7-4.0 Hz,
   third-order Butterworth filter to both method outputs.
6. Slice each continuous rPPG signal using the exact Phase 1 start and end
   timestamps.
7. Estimate heart rate with a 4096-point Hann periodogram and log-power
   parabolic peak interpolation.
8. Preserve every `(subject, window, ROI, method)` key. Windows that do not
   pass source-data quality rules receive an explicit status instead of being
   silently dropped.

The implementation follows the CHROM method of de Haan and Jeanne (2013),
DOI `10.1109/TBME.2013.2266196`, and the POS method described by Wang et al.
(2017), DOI `10.1109/TBME.2016.2609282`.

## Source-Only Quality Rules

A window is estimable when:

- video coverage is at least 95%;
- at least 80% of observed ROI frames are valid;
- the longest interpolation gap is no more than 0.50 seconds.

These rules read video/RGB quality only. They do not read reference HR or rPPG
error.

## Outputs

The formal outputs are under `results_v2/phase2/`:

- `rppg_window_results.csv`;
- `phase2_subject_qc.csv`;
- `method_roi_metrics.csv`;
- `phase2_runtime.csv`;
- `phase2_config.json`;
- `phase2_run_summary.json`.

The window file also contains reference-independent candidate reliability
features:

- spectral peak power fraction;
- spectral peak-to-median ratio;
- spectral entropy;
- rPPG signal standard deviation;
- face motion and ROI quality summaries;
- POS-CHROM heart-rate disagreement;
- cross-region disagreement;
- left-right cheek disagreement.

## Reproduction Result

- Subjects: 42/42
- Phase 1 reference windows: 2,407
- Expected rows: 24,070
- Written rows: 24,070
- Estimable rows: 23,842
- Insufficient-valid-frame rows: 168
- Excessive-interpolation-gap rows: 60
- Finite HR rows: 23,842
- Primary-reference evaluated rows: 22,968
- Runtime: 55.62 seconds
- Full tests: 29 passed
- Ruff: passed

All non-estimable rows came from the forehead region. They were concentrated in
subjects 5, 32, and 48.

## Descriptive Baseline Results

The following metrics use only Phase 1 concordant reference windows.

| Method | ROI | Available | MAE | RMSE | Median AE | Pearson r |
|---|---|---:|---:|---:|---:|---:|
| CHROM | cheeks mean | 100.00% | 1.587 | 2.321 | 1.157 | 0.992 |
| CHROM | full face inner | 100.00% | 1.613 | 2.461 | 1.161 | 0.991 |
| CHROM | right cheek | 100.00% | 1.775 | 3.034 | 1.202 | 0.986 |
| CHROM | left cheek | 100.00% | 1.877 | 4.235 | 1.227 | 0.972 |
| CHROM | forehead | 95.21% | 3.213 | 8.231 | 1.337 | 0.902 |
| POS | full face inner | 100.00% | 1.708 | 3.410 | 1.127 | 0.982 |
| POS | cheeks mean | 100.00% | 1.746 | 3.772 | 1.164 | 0.978 |
| POS | right cheek | 100.00% | 2.293 | 5.754 | 1.197 | 0.954 |
| POS | left cheek | 100.00% | 2.645 | 8.498 | 1.225 | 0.896 |
| POS | forehead | 95.21% | 4.594 | 12.824 | 1.354 | 0.785 |

These are overlapping-window descriptive results, not independent-sample
confidence estimates. They must not be used to claim generalization.

## Validation Decision

Phase 2 passed its implementation and data-integrity gates. The low median
errors coexist with large subject/region-specific outliers. This makes the
result suitable for Phase 3, where reliability labels and interpretable
features will be evaluated with subject-grouped splits.

No Phase 2 parameter may now be selected using these errors without declaring a
separate exploratory or sensitivity analysis.
