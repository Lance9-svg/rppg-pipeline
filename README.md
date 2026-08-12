# Standalone UBFC rPPG reliability experiment

This branch contains one reproducible experiment: starting from the original
UBFC-rPPG Dataset 2 files, it builds window-aligned contact-PPG references,
extracts classical POS and CHROM estimates from five facial regions, and writes
one validated candidate table. The research question is whether
reference-independent signal-quality features can identify unreliable rPPG
heart-rate estimates for previously unseen subjects.

This repository contains code and tests only. Videos, contact-PPG data,
MediaPipe model files, traces, tables, figures, and run records remain local and
are excluded from Git when the documented `data/`, `models/`, and
`local_outputs/` directories are used. If different paths are used, keep them
outside the repository checkout.

## Inputs

Obtain Dataset 2 from the
[UBFC-rPPG project page](https://sites.google.com/view/ybenezeth/ubfcrppg).
Do not copy the dataset into Git. The input directory must retain the canonical
layout:

```text
<dataset-root>/
  subject1/
    vid.avi
    ground_truth.txt
  subject2/
    vid.avi
    ground_truth.txt
  ...
```

A local MediaPipe Face Landmarker `.task` model is also required. Its path is
passed explicitly; the model is not downloaded or committed by the program.

## Environment

The experiment is tested with Python 3.13. On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

## Run the experiment

Use a new local output directory. Existing output directories are rejected so
that an earlier experiment cannot be overwritten accidentally.

```powershell
.\.venv\Scripts\python.exe -m rppg_pipeline `
  --dataset-root "data\UBFC_DATASET_2" `
  --face-model "models\face_landmarker.task" `
  --output "local_outputs\ubfc-candidates"
```

For a fast pipeline check before the full run, select two original subject IDs:

```powershell
.\.venv\Scripts\python.exe -m rppg_pipeline `
  --dataset-root "data\UBFC_DATASET_2" `
  --face-model "models\face_landmarker.task" `
  --output "local_outputs\ubfc-smoke" `
  --subjects 1 3
```

The fixed experiment writes:

```text
<output>/
  inventory.csv
  traces/<subject>/
  reference_windows.csv
  candidate_windows.csv
  reliability_features.csv
  feature_dictionary.csv
  feature_audit.csv
  run.json
```

`inventory.csv` uses relative dataset names. `run.json` records input hashes,
the Git revision, fixed processing parameters, completed stages, and success or
failure; it deliberately contains no wall-clock timestamps.
`candidate_windows.csv` has one row for every subject, window, facial region,
and method combination. `reliability_features.csv` keeps only eligible rows,
the frozen M1--M4 input fields, grouping keys, and the strict `>3`, `>5`, and
`>10 bpm` reliability targets. Generated files must not be committed.

## Fixed processing path

The public command has no phase selector, retry, resume, model registry, or
report generator. It always executes the same sequence:

1. validate and inventory canonical UBFC subject inputs;
2. extract per-frame RGB and source-quality traces for five fixed facial ROIs;
3. construct 10-second contact-PPG reference windows at a 1-second step;
4. estimate window-level heart rate with POS and CHROM;
5. validate and export the candidate table;
6. export the leakage-controlled reliability feature table and field audit.

The numerical implementations and their regression tests live in
`ppg_reference.py`, `standard_rppg.py`, `roi.py`, `candidates.py`, and
`reliability_features.py`.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\python.exe -m rppg_pipeline --help
```

The test suite covers the single CLI contract, input inventory, success and
failure recording, synthetic contact-PPG recovery, synthetic POS/CHROM
recovery, ROI behaviour, candidate-key completeness, strict label boundaries,
feature allowlists, leakage rejection, and provenance hashes. It does not claim
that the formal UBFC experiment has run.

## Method references

The contact-PPG reference is a fixed signal-processing baseline rather than a
claimed reproduction of a separate named HR algorithm. It uses SciPy for
detrending, Butterworth filtering, peak detection, and spectral estimation;
all thresholds and window settings are explicit in `PPGReferenceConfig`.

- S. Bobbia, R. Macwan, Y. Benezeth, A. Mansouri, and J. Dubois,
  “Unsupervised Skin Tissue Segmentation for Remote Photoplethysmography,”
  *Pattern Recognition Letters*, 2019.
  [DOI: 10.1016/j.patrec.2017.10.017](https://doi.org/10.1016/j.patrec.2017.10.017)
- G. de Haan and V. Jeanne, “Robust Pulse Rate From Chrominance-Based rPPG,”
  *IEEE Transactions on Biomedical Engineering*, 60(10), 2878–2886, 2013.
  [DOI: 10.1109/TBME.2013.2266196](https://doi.org/10.1109/TBME.2013.2266196)
- W. Wang, A. C. den Brinker, S. Stuijk, and G. de Haan, “Algorithmic
  Principles of Remote PPG,” *IEEE Transactions on Biomedical Engineering*,
  64(7), 1479–1491, 2017.
  [DOI: 10.1109/TBME.2016.2609282](https://doi.org/10.1109/TBME.2016.2609282)
- P. Virtanen et al., “SciPy 1.0: Fundamental Algorithms for Scientific
  Computing in Python,” *Nature Methods*, 17, 261–272, 2020.
  [DOI: 10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2)

## Scope and limitations

This branch currently supports only the canonical UBFC-rPPG Dataset 2 layout
and classical POS/CHROM processing. It is not a general rPPG framework. It does
not include trained reliability models, cross-dataset validation, formal
42-subject results, or thesis figures. Those claims require later commits and
actual local experiment runs.

## License

No software license has been added to this repository. Source availability
alone does not grant reuse rights. The UBFC-rPPG dataset and the MediaPipe model
remain subject to their providers' separate terms and are not redistributed
here.
