# Codex project instructions

Read this file before changing code, experiments, documentation, or Git state.

## Project purpose

This repository implements a reproducible, standalone UBFC-rPPG Dataset 2
experiment. It constructs contact-PPG references, extracts POS and CHROM heart
rate estimates from five facial regions, and builds leakage-controlled signal
reliability features. It is research software: reproducibility, provenance,
privacy, and accurate claims take precedence over convenience.

## Technical stack

- Python 3.13 (the recorded successful local run used 3.13.4).
- NumPy, pandas, SciPy, OpenCV contrib, and MediaPipe.
- pytest for tests and Ruff for linting.
- Windows PowerShell commands are documented, but implementation paths must
  remain portable and use `pathlib.Path`.

## Important paths

- `rppg_pipeline/`: implementation and CLI.
- `tests/`: unit and pipeline-contract tests.
- `README.md`: experiment contract and user instructions.
- `MIGRATION.md`: machine setup, current work, and non-Git assets.
- `data/` or an external path: licensed UBFC data; never commit it.
- `models/`: local MediaPipe model files; never commit them.
- `local_outputs/`: generated traces, tables, manifests, and figures; never
  commit them unless a later, explicit policy approves a small anonymized
  aggregate artifact.

## Development rules

1. Define one task, its inputs/outputs, affected files, method source, and
   verification before editing.
2. Preserve real Git history. Do not rewrite history, fabricate dates or
   results, or split old work into fictitious commits.
3. Keep research decisions, dataset adapters, algorithm implementations, and
   engineering utilities distinguishable. Cite sources for POS, CHROM,
   filtering, spectral estimation, and externally derived methods.
4. Record whether parameters are literature-defined, pre-specified,
   dataset-specific, or exploratory. Never silently tune a primary protocol
   after inspecting formal test results.
5. Never commit secrets, absolute local paths, caches, virtual environments,
   licensed/raw participant data, model binaries, or unrelated generated
   output.
6. Do not claim tests, experiments, manual review, or results that were not
   actually completed. AI output is not research evidence.
7. Do not alter the frozen experiment contract or generated formal results
   without an explicit research task and provenance plan.

## Verification

Before presenting a change as complete, run the applicable checks from the
repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\python.exe -m rppg_pipeline --help
git status
git diff
git diff --staged
```

If a check cannot run, state the reason and the unverified scope. Inspect diffs
for secrets, raw data, caches, local absolute paths, and unrelated files.
Keep code, tests, experiment records, and documentation in truthful logical
commits; do not commit automatically unless the user asks.

## Protected material

Do not modify or redistribute the UBFC dataset, participant data, provider
licenses, the MediaPipe model binary, or formal local experiment outputs by
default. The repository currently has no software license, so do not imply
that source availability grants reuse rights.
