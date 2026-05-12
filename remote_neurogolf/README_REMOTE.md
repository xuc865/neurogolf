# Remote NeuroGolf Runbook

This folder is for a clean remote server run. It does not depend on the broken
local Python environment.

## Why this route

Public work worth reusing is solver-based:

- `rogermt/neurogolf-solver`: modular v5 solver, reports v4/v4.3 at 307 local solved tasks and estimated LB around 650-670.
- `ashhhhhh26/neurogolf-2026`: single-file enhanced solver based on `rogermt`, with analytical solvers plus least-squares conv solvers.

The runner treats those as candidate generators, also scans your existing
`submission*` folders/zips, validates every candidate model per task, keeps the
lowest-cost valid model, and builds one final zip.

## Setup

On the remote server, from the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r remote_neurogolf/requirements.txt
```

If the server uses Kaggle CLI submission, create `~/.kaggle/kaggle.json` in the
usual Kaggle username/key format and run:

```bash
chmod 600 ~/.kaggle/kaggle.json
```

Do not hardcode tokens in scripts. The token pasted in chat should be rotated
after this run.

## Run

## Recommended Submission Workflow

Because Kaggle's hidden leaderboard score is the source of truth, use the known
good submission as the baseline and preserve it unless a model fails ONNX
processing. This avoids replacing high-LB models with candidates that only look
better under an imperfect local estimate.

```bash
python remote_neurogolf/run_remote.py \
  --repo-root . \
  --data-dir data \
  --work-dir remote_runs/preserve_baseline \
  --output-zip remote_runs/preserve_baseline/final_submission.zip \
  --preserve-source submission_opt2.zip \
  --conv-budget 180 \
  --workers 8
```

If your known 4763.83 submission zip has a different filename, use that file for
`--preserve-source`.

Basic ensemble run for leaderboard-oriented selection:

```bash
python remote_neurogolf/run_remote.py \
  --repo-root . \
  --data-dir data \
  --work-dir remote_runs/b60 \
  --output-zip remote_runs/b60/final_submission.zip \
  --conv-budget 60 \
  --workers 8
```

Longer leaderboard-oriented run, usually better but slower:

```bash
python remote_neurogolf/run_remote.py \
  --repo-root . \
  --data-dir data \
  --work-dir remote_runs/b180 \
  --output-zip remote_runs/b180/final_submission.zip \
  --conv-budget 180 \
  --workers 8
```

Stricter research run, useful for robustness diagnostics but often too
conservative for leaderboard selection:

```bash
python remote_neurogolf/run_remote.py \
  --repo-root . \
  --data-dir data \
  --work-dir remote_runs/b180_arcgen \
  --output-zip remote_runs/b180_arcgen/final_submission.zip \
  --conv-budget 180 \
  --include-arcgen \
  --workers 8
```

Submit automatically after packaging:

```bash
python remote_neurogolf/run_remote.py \
  --repo-root . \
  --data-dir data \
  --work-dir remote_runs/b180 \
  --output-zip remote_runs/b180/final_submission.zip \
  --conv-budget 180 \
  --submit
```

The submit step refuses to upload if the local estimate is below `6000`. To
change that guardrail, pass `--min-submit-score 6500` or another value.

If you already have a public or private high-scoring zip/model directory, add it
as another candidate source:

```bash
python remote_neurogolf/run_remote.py \
  --repo-root . \
  --data-dir data \
  --work-dir remote_runs/ensemble \
  --output-zip remote_runs/ensemble/final_submission.zip \
  --extra-source /path/to/other_submission.zip \
  --extra-source /path/to/other_models_dir \
  --workers 8
```

## What to submit manually

Submit this file:

```text
remote_runs/b180/final_submission.zip
```

Use `remote_runs/b180/final_report.json` to inspect which source won each task
and the estimated local score.

## If Kaggle Reports Task 101 Processing Error

That means the selected `task101.onnx` is not accepted by the platform's ONNX
processing stage, even if it may run locally. Rerun with the updated
`run_remote.py`; it now rejects candidates that fail strict ONNX processing.

For an immediate safe resubmission, replace only task 101 with an identity
fallback:

```bash
python remote_neurogolf/fix_task101_zip.py \
  remote_runs/b180/final_submission.zip \
  remote_runs/b180/final_submission_no101error.zip
```

Submit `remote_runs/b180/final_submission_no101error.zip`. This may lose task
101's task-specific points, but it avoids the whole submission failing.

## Final Processing Gate

Before any manual submission, run the full 400-model audit:

```bash
python remote_neurogolf/audit_submission.py \
  remote_runs/b180/final_submission.zip \
  --data-dir data \
  --include-arcgen \
  --workers 8
```

If it reports failures, repair every processing-failing task with a guaranteed
safe identity fallback and create a clean zip:

```bash
python remote_neurogolf/audit_submission.py \
  remote_runs/b180/final_submission.zip \
  --data-dir data \
  --include-arcgen \
  --workers 8 \
  --repair \
  --output-zip remote_runs/b180/final_submission_processing_clean.zip
```

Submit `final_submission_processing_clean.zip` only after the audit reports
`failure_count: 0`.

## Efficiency Notes

The runner is CPU-only. It now caches one-hot encoded examples and evaluates
each candidate with one official-style profiled pass instead of a separate
validation pass plus a scoring pass. `--workers` controls parallel candidate
validation; a good starting value is physical CPU cores or slightly less. If the
server becomes memory-bound, lower `--workers`.
