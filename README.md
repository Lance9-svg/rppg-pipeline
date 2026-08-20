# rPPG Degradation and Reliability Pipeline

Interpretable reliability assessment for classical rPPG heart-rate estimation
under controlled video degradation.

This repository contains the experiment-processing source code and automated
tests for a master's project that evaluates how controlled video degradation
(frame-rate reduction and ROI drift) affects POS and CHROM heart-rate estimates
on the public UBFC-rPPG dataset, and whether reference-independent quality
features can identify unreliable estimates.

## Repository contents

- `rppg_pipeline/` — processing source code:
  - `dataset.py` — subject discovery and ground-truth reading
  - `video.py` — single-pass video decoding, face landmarks, ROI construction
  - `degradation.py` — frame-rate reduction and ROI-drift conditions
  - `rppg.py` — POS / CHROM estimation and quality features
  - `experiment.py` — candidate and audit table construction
  - `statistics.py` — per-condition metrics and paired subject bootstrap
  - `reliability.py` — subject-grouped out-of-fold reliability model
  - `formal_run.py` — single-pass formal entry point
- `tests/` — automated tests

Raw videos, ground-truth signals, model binaries, and formal result files are
intentionally not included in this repository.

## Setup

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Tests

```bash
.venv/bin/python -m pytest -q
```

## Usage

```bash
# Formal five-condition run
.venv/bin/python -m rppg_pipeline.formal_run \
  --dataset-root <dataset-root> \
  --face-model <face-landmarker-model> \
  --baseline-output <baseline-dir> \
  --output <output-dir>

# Degradation statistics and paired subject bootstrap
.venv/bin/python -m rppg_pipeline.statistics \
  --candidates <output-dir>/degradation_candidates.csv \
  --output <stats-dir>

# Reliability model
.venv/bin/python -m rppg_pipeline.reliability \
  --candidates <output-dir>/degradation_candidates.csv \
  --output <reliability-dir>
```

## Scope

The method follows a frozen experimental protocol. Conclusions are limited to
the current UBFC protocol and do not constitute clinical reliability or
cross-dataset generalisation.
