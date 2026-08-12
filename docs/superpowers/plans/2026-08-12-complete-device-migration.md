# Complete Device Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a complete, safe, one-conversation migration path for the rPPG project and its 23 project-specific Codex Skills.

**Architecture:** Keep research code/history in the existing private migration repository and place reviewed project Skills in a separate private repository. System and plugin Skills are restored through Codex on the destination device, while datasets, models, experiment outputs, credentials, caches, and machine metadata remain outside Git. A single bootstrap prompt coordinates clone, remote repair, Skill installation, environment recreation, and verification.

**Tech Stack:** Git, GitHub private repositories, Codex Skills, Windows PowerShell, Python 3.13, pytest, Ruff.

## Global Constraints

- Preserve truthful Git history; never rewrite or force-push.
- Do not upload raw participant data, licensed datasets, models, experiment outputs, secrets, credentials, caches, virtual environments, or plugin caches.
- The formal development remote remains `https://github.com/Lance9-svg/rppg-pipeline.git`.
- The project backup remote remains `https://github.com/Lance9-svg/rppg-pipeline-migration.git`.
- Install system and plugin Skills on the destination device rather than copying their caches.
- Verify every copied Skill includes all files referenced by its `SKILL.md`.

---

### Task 1: Audit the project-specific Skill source

**Files:**
- Read: `.codex/skills/**`
- Create: separate `codex-research-skills` repository

- [ ] Inventory Skill names, files, sizes, nested repositories, and links.
- [ ] Scan filenames and contents for secrets without printing secret values.
- [ ] Identify local absolute paths and distinguish executable configuration from documentation/examples.
- [ ] Exclude nested `.git`, caches, credentials, and machine metadata.

### Task 2: Build and publish the private Skill repository

**Files:**
- Create: `README.md`
- Create: `.gitignore`
- Copy: reviewed project Skill directories

- [ ] Copy all 23 complete Skill directories into an isolated repository.
- [ ] Verify every copied `SKILL.md` and every referenced local resource exists.
- [ ] Run final secrets, absolute-path, large-file, and Git diff audits.
- [ ] Commit with a truthful migration message.
- [ ] Create a private GitHub repository and push `main`.

### Task 3: Add the one-prompt destination bootstrap

**Files:**
- Create: `NEW_DEVICE_BOOTSTRAP_PROMPT_ZH.md`
- Modify: `MIGRATION.md`

- [ ] Document project clone and `origin`/`migration` remote configuration.
- [ ] Document installation of project Skills from the private Skill repository.
- [ ] Document system/plugin Skill restoration without copying caches.
- [ ] Document Python environment recreation, tests, Ruff, CLI, and manual assets.
- [ ] Commit and push the migration documentation to both the migration backup and formal project remotes.

### Task 4: Final verification

**Files:**
- Verify both private GitHub repositories and the formal project branch.

- [ ] Run pytest, Ruff, CLI help, Git diff checks, and secret scans.
- [ ] Verify remote branch commit hashes for all intended destinations.
- [ ] Verify the Skill repository file count, Skill count, and absence of excluded paths.
- [ ] Report manual-only assets and remaining risks.
