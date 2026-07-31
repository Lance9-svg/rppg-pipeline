# rPPG research pipeline

This project tests whether interpretable, reference-independent signal-quality
features can predict the reliability of classical rPPG heart-rate estimates
for unseen subjects. Evaluation is subject-independent and includes checks for
evaluation bias.

The repository currently contains the completed contact-PPG reference and
POS/CHROM evaluation stages.

## Dataset

The project uses Dataset 2 of the [UBFC-rPPG dataset](https://sites.google.com/view/ybenezeth/ubfcrppg). The videos and contact-PPG files are obtained from the dataset authors for research purposes and are not redistributed in this repository.

## Completed experiments

### Phase 1: contact-PPG reference

Phase 1 builds a timestamp-aware heart-rate reference without reading the rPPG
estimates:

1. Read the contact-PPG signal, sensor heart rate, and timestamps.
2. Resolve duplicate and non-uniform timestamps and resample the signal.
3. Detrend and filter the contact-PPG signal.
4. Estimate heart rate with time-domain and frequency-domain methods.
5. Classify their agreement and retain sensor heart rate as a separate
   quality-control field.
6. Write the aligned reference windows, subject manifest, quality-control
   table, and configuration.

### Phase 2: window-aligned POS/CHROM evaluation

Phase 2 reuses the existing facial RGB traces and the exact Phase 1 windows:

1. Load the RGB and source-quality traces for each subject.
2. Process `full_face_inner`, `forehead`, `left_cheek`, `right_cheek`, and
   `cheeks_mean`.
3. Extract POS and CHROM signals for every region.
4. Estimate window-level heart rate and preserve explicit non-estimable
   statuses.
5. Calculate spectral, motion, ROI-quality, cross-method, and cross-region
   information without using the reference error to select a candidate.
6. Write the window-level results, subject quality-control table, method/ROI
   summary, runtime, configuration, and run summary.

## Current status

| Phase | Status | Scope | Evidence |
|---|---|---|---|
| Phase 1 | Complete | Contact-PPG reference construction | [Validation report](docs/phase1_validation_report.md) |
| Phase 2 | Complete | Window-aligned POS/CHROM evaluation | [Validation report](docs/phase2_validation_report.md) |

## Automated verification

The test suite covers synthetic heart-rate recovery, timestamp preparation,
constant and invalid signals, ROI construction, RGB channel ordering, window
alignment, quality-gate behaviour, subject discovery, batch processing, and
expected output generation.

## Repository structure

```text
rppg_pipeline/
  ubfc.py             UBFC subject discovery and ground-truth parsing
  ppg_reference.py    Phase 1 contact-PPG reference
  run_phase1.py       Phase 1 dataset runner
  standard_rppg.py    window-aligned POS/CHROM processing
  run_phase2.py       Phase 2 dataset runner
  video.py            video metadata
  face_landmarks.py   MediaPipe face landmarks
  roi.py              fixed facial ROIs
  rgb_trace.py        per-frame RGB and source-quality traces
  rppg.py             legacy full-segment CHROM/POS comparator
  evaluation.py       legacy coarse-grid evaluation comparator
  run_single.py       single-video preprocessing and legacy analysis
  run_batch.py        dataset preprocessing and legacy batch analysis
tests/                automated tests
docs/                 phase validation reports
```

`ppg_reference.py` and `standard_rppg.py` contain the Phase 1 and Phase 2
implementations used for the current experiments. `rppg.py`, `evaluation.py`,
`run_single.py`, and `run_batch.py` remain available for preprocessing and
bias comparisons.

## Environment setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

## Run automated checks

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\ruff.exe check .
```

## Reproduce Phase 1

```powershell
.\.venv\Scripts\python.exe -m rppg_pipeline.run_phase1 `
  --dataset-root "<dataset-root>" `
  --out results_v2\phase1
```

The main outputs are:

- `subject_manifest.csv`
- `ppg_reference_qc.csv`
- `ppg_reference_windows.csv`
- `phase1_config.json`

## Reproduce Phase 2

Phase 2 reuses the RGB traces generated under `results/`.

```powershell
.\.venv\Scripts\python.exe -m rppg_pipeline.run_phase2 `
  --phase1-dir results_v2\phase1 `
  --trace-root results `
  --out results_v2\phase2
```

The main outputs are:

- `rppg_window_results.csv`
- `phase2_subject_qc.csv`
- `method_roi_metrics.csv`
- `phase2_runtime.csv`
- `phase2_config.json`
- `phase2_run_summary.json`

## References

- S. Bobbia, R. Macwan, Y. Benezeth, A. Mansouri, and J. Dubois,
  "Unsupervised skin tissue segmentation for remote photoplethysmography,"
  *Pattern Recognition Letters*, vol. 124, pp. 82-90, 2019.
  [doi:10.1016/j.patrec.2017.10.017](https://doi.org/10.1016/j.patrec.2017.10.017)
- G. de Haan and V. Jeanne, "Robust Pulse Rate From Chrominance-Based rPPG,"
  *IEEE Transactions on Biomedical Engineering*, vol. 60, no. 10,
  pp. 2878-2886, 2013.
  [doi:10.1109/TBME.2013.2266196](https://doi.org/10.1109/TBME.2013.2266196)
- W. Wang, A. C. den Brinker, S. Stuijk, and G. de Haan,
  "Algorithmic Principles of Remote PPG,"
  *IEEE Transactions on Biomedical Engineering*, vol. 64, no. 7,
  pp. 1479-1491, 2017.
  [doi:10.1109/TBME.2016.2609282](https://doi.org/10.1109/TBME.2016.2609282)
