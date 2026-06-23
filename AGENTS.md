# Agent Instructions

## Compute Policy

- Never execute repository scripts, tests, dataset generation, model training,
  OCR inference, embedding extraction, or other project compute on the login
  node. This includes lightweight syntax and unit checks: submit them through
  Slurm.
- Use Slurm and request an A100 node for heavy jobs.
- Prefer batch scripts under `scripts/` and submit them with `sbatch`. Do not
  invoke repository submit scripts with `bash` on the login node; either issue
  their `sbatch` commands directly or submit the orchestration itself as a job.
- Do not assume the partition hard-coded in a batch script currently has A100 nodes. Before submitting GPU work, check live Slurm resources with `sinfo -o '%P %G %D %t'` or `sinfo -p <partition> -o '%P %G %D %t %N'`.
- If a script requests `--gres=gpu:a100:1` but its partition has no A100 GRES, override only the partition at submit time, for example `sbatch --partition=mit_preemptable scripts/<job>.sbatch`. Keep the A100 GRES request intact.
- If Slurm commands fail with controller connection errors from a sandboxed agent environment, retry the same Slurm command outside the sandbox before changing job scripts.
- Before running any command that may train, generate full datasets, run OCR, or process large Parquet files, confirm it is inside an allocated A100 Slurm job or submit it as one.
- Only read-only inspections, file edits, `git` commands, and Slurm scheduler
  commands are allowed on the login node. Run all smoke checks through Slurm.

## Dataset Cleanup Policy

- Keep source datasets, cleaned source datasets, final generated datasets, and cache artifacts required to rebuild generation.
- Remove stale logs, notebooks, audit-only outputs, bytecode caches, OS metadata, and scratch directories unless explicitly requested otherwise.
- Treat cleanup as part of every experiment cycle, not as a separate end-of-project task:
  - after a failed or cancelled job, remove its partial output directory and logs once the failure has been diagnosed;
  - after a replacement run is confirmed, remove superseded smoke outputs, numbered retry directories, and obsolete evaluation reports;
  - after recording a successful run in `NEXT_LLM_REMINDER.txt`, retain only logs still needed to diagnose or reproduce the active result.
- Do not decide that an artifact is useful solely because its Slurm job completed. Preserve it only when it is an input to the active workflow, a reviewed result, or required to rebuild one.
- Before deleting a dataset or cache, check manifests, batch-script defaults, README/DATASET_CARD references, and the active handoff. Never delete raw/clean source data, reviewed final datasets, the active substitution atlases, model/font assets, or an environment needed by the current evaluation.
- Prefer stable purpose-based directory names for accepted outputs. Temporary and smoke outputs must live under `.cache/` and should be deleted when the iteration is superseded.
- Record material cleanup in `NEXT_LLM_REMINDER.txt`, including any retained artifact whose purpose is not obvious.

## Handoff Reminder Policy

- When pausing or completing a run, leave a short reminder file in the repo root that states:
  - the current objective,
  - the last confirmed job IDs or artifact paths,
  - the next concrete step to resume from,
  - any open blockers or missing scheduler state.
- Prefer a plain text file named `NEXT_LLM_REMINDER.txt` unless the user asked for a different format.
- Keep the reminder concise and update it when the run state changes.
- For Slurm jobs, prefer submit-and-stop handoffs: record the job ID and output path, then stop unless the user explicitly asks you to keep polling, inspect logs, or chain the next step.
