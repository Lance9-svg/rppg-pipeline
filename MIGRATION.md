# Migration guide

Audit date: 2026-08-12 (Asia/Shanghai)

## Current state

The active development branch is
`codex/research/ubfc-reliability-experiment`. Its checked-out base commit is
`8588daf` (`refactor: build standalone dataset-to-candidate experiment`) and
that commit is present on `origin`. At audit time the working tree is dirty:

- modified: `README.md`, `rppg_pipeline/candidates.py`,
  `rppg_pipeline/experiment.py`, `rppg_pipeline/provenance.py`,
  `tests/test_candidates.py`, `tests/test_experiment.py`, and
  `tests/test_provenance.py`;
- untracked implementation: `rppg_pipeline/reliability_features.py`;
- untracked tests: `tests/test_reliability_features.py`;
- this migration preparation adds `.gitignore`, `AGENTS.md`, and
  `MIGRATION.md` changes.

These working-tree changes are **not available by cloning Git** until they are
reviewed and committed/pushed, or copied separately.

There is also an older checkout (`<old-phase1-checkout>`) on
`research/phase1-phase2-pipeline`, base commit `a7e547d`. At audit time it has
17 modified tracked files under `docs/` and `rppg_pipeline/`. Do not delete the
old computer or that checkout until those changes have been reviewed against
the active branch. The surrounding planning workspace (`<old-rppg-workspace>`)
is a separate, uninitialized Git repository with no commits; its planning
notes, slides, and policy files are not carried by the code repository.

## Project structure

```text
rppg-pipeline/
  AGENTS.md                 long-term Codex/development rules
  MIGRATION.md              this machine-transfer record
  README.md                 experiment contract and commands
  pyproject.toml            Ruff and pytest configuration
  requirements.txt         runtime dependencies
  requirements-dev.txt     runtime + test/lint dependencies
  rppg_pipeline/            Python package and CLI
  tests/                    automated tests
  data/                     local dataset location (ignored, optional layout)
  models/                   local model files (ignored)
  local_outputs/            generated experiment output (ignored)
```

## Environment and dependencies

The recorded successful full run used Python 3.13.4 with:

- `mediapipe==0.10.35`
- `numpy==2.5.0`
- `opencv-contrib-python==5.0.0.93`
- `pandas==3.0.3`
- `scipy==1.18.0`

The repository intentionally specifies compatible lower bounds in
`requirements.txt`, not a fully locked environment. `requirements-dev.txt`
adds pytest and Ruff. No Node.js application, `package.json`, Conda
`environment.yml`, Docker configuration, GPU/CUDA dependency, database, or
external API service was found. The CLI is CPU-based but full video processing
is compute- and storage-intensive.

The complete new-device recovery prompt is stored in
`NEW_DEVICE_BOOTSTRAP_PROMPT_ZH.md`. Project-specific Codex Skills are backed
up separately in the private repository
`https://github.com/Lance9-svg/codex-research-skills.git`. System and
plugin-provided Skills are not copied; install them through Codex on the new
device.

On a new Windows machine:

```powershell
git clone https://github.com/Lance9-svg/rppg-pipeline.git
Set-Location rppg-pipeline
git switch codex/research/ubfc-reliability-experiment
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\python.exe -m rppg_pipeline --help
```

If exact environment reproduction is required, install the recorded versions
above after confirming that wheels exist for the new machine's Python version
and architecture. Do not copy `.venv`; recreate it.

## Running the project

Preserve the canonical UBFC Dataset 2 layout (`subjectN/vid.avi` and
`subjectN/ground_truth.txt`) and supply paths at runtime:

```powershell
.\.venv\Scripts\python.exe -m rppg_pipeline `
  --dataset-root "<dataset-root>" `
  --face-model "models\face_landmarker.task" `
  --output "local_outputs\ubfc-smoke" `
  --subjects 1 3
```

Use a new output directory for every run; the program rejects an existing
output directory to prevent accidental overwrite. After smoke verification,
omit `--subjects` for the full dataset run.

## Completed work

- The pushed base branch provides a standalone dataset-to-candidate pipeline,
  canonical UBFC discovery, contact-PPG references, five facial ROIs,
  POS/CHROM estimates, provenance manifests, and tests.
- Local `run.json` evidence records a successful 42-subject run from dirty
  commit `8588daf`, using the environment versions above. It produced 2,407
  reference windows, 24,070 candidate rows, and 22,968 reliability-feature
  rows. This is a local run record, not proof that the uncommitted code has
  been reviewed or published.
- Local smoke runs exist for subjects 1/3 and 24/40.

## Work in progress and next steps

The uncommitted work adds leakage-controlled reliability features, strict
3/5/10-bpm targets, field audits, pipeline integration, and tests. Before
migration is considered complete:

1. review both dirty worktrees and determine whether the older branch changes
   are superseded, still needed, or should become a separate truthful commit;
2. run pytest, Ruff, CLI help, secret scanning, and Git diff checks on the
   active branch;
3. review and commit the reliability-feature code/tests and migration docs in
   logical commits, then push the branch;
4. clone the branch on the new machine and repeat the verification commands;
5. copy or reacquire the non-Git assets below, verify hashes where available,
   and run a two-subject smoke test before any full rerun.

## Files that must not be committed

Copy these through encrypted external storage or reacquire them under their
provider terms:

- UBFC-rPPG Dataset 2. The audited run used
  `<old-dataset-drive>\dataset\UBFC-RPPG Dataset`; the manifest records 42
  subject videos, each roughly 1.2--1.9 GB, plus contact-ground-truth files.
  Raw participant data must remain outside Git.
- MediaPipe `face_landmarker.task`. The audited file is 3,758,596 bytes with
  SHA-256 `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`.
- `local_outputs/` if prior traces/results must be retained. Four audited run
  directories total about 179 MB. They include the full run and three smoke
  runs and are intentionally ignored.
- Any still-needed content from the outer, no-commit workspace (planning
  Markdown, presentations, inspection files, and Codex skills/configuration).
  Treat `.codex/`, `.agents/`, and machine-specific worktree metadata as local
  tooling; migrate only reviewed, non-secret materials, not `.git` links.
- The old dirty checkout (`<old-phase1-checkout>`) until its 17 changes have
  been reconciled.

Do not copy `.venv`, `.pytest_cache`, `.ruff_cache`, `__pycache__`, IDE files,
or the `.git` file from a linked worktree. Clone normally on the new device.

## Known issues and migration risks

- Highest risk: uncommitted changes exist in two code worktrees; Git alone is
  not yet a complete backup.
- The active worktree is a linked worktree whose `.git` file contains an
  absolute path to old-machine metadata. Copying that directory verbatim will
  produce a broken checkout; commit/push or make a patch/bundle instead.
- The requirements use lower bounds, so a future install may resolve different
  versions. The recorded versions above are the current reproducibility anchor.
- Local manifests contain the old machine's dataset/model command paths. They
  are ignored outputs and should not be committed; use new runtime arguments.
- The dataset and model have separate provider terms, and this repository has
  no software license.
- No live secrets were found by the filename/content-pattern audit, but secret
  scanning is heuristic. Inspect staged diffs before every push.
