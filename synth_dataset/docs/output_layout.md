# Output Layout

This project keeps generated-data artifacts in a small number of named folders:

- `generated_datasets/mix65/`: active generated train/test/validation parquet files and generation metrics.
- `model_results/mix65/`: trained model checkpoints, metrics, predictions, and paper-ready confusion matrices for the mix65 dataset.
- `lookup_tables/in_use/`: lookup tables used by the current generation pipeline.
- `lookup_tables/archive/`: older lookup tables kept for reproducibility or reference.
- `outputs/slurm_logs_archive/`: archived Slurm logs from previous experiments.
- `outputs/reference/`: uploaded reference images or screenshots.
- `docs/legacy_next_llm_reminder.txt`: old run-state reminder retained for historical context.

Protected inputs remain in `BASE_DATASETS_DO_NOT_EVER_DELETE/`, and the original instruction snapshot remains in `ORIGINAL_INSTRUCTIONS_DO_NOT_EVER_DELETE`.
