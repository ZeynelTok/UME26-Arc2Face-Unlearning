# Paper artifacts

This folder contains the saved evidence behind the paper. You can recompute
the tables without downloading CelebA, training adapters, or generating images.

- `run_metrics.csv` contains one row per run.
- `per_sample_predictions.csv.gz` contains the ArcFace results used to calculate
  those metrics. `generation_attempts.csv.gz` also includes detection failures.
- `adapters.npz` stores the 120 selected adapters, with their training summaries
  in `adapter_summaries.jsonl`.
- `target_selections.csv`, `target_candidates.csv`, and
  `hardness_top3_neighbors.csv.gz` record how targets and groups were chosen.
- `quality_scores.csv.gz`, `recognizer_runs.csv`, `adaface_predictions.csv.gz`,
  and `large_gallery_predictions.csv.gz` support the extra evaluations.
- `sensitivity_policy_metrics.csv` contains the top-10 and top-20 results.
- `protocol.json` records the groups, references, settings, and calibration.
- `recomputed_paper_tables.json` is the saved output of the table script.

To rebuild the bundle from completed experiment outputs and check the tables:

```powershell
venv\Scripts\python.exe scripts\reproducibility\build_paper_artifacts.py --overwrite
venv\Scripts\python.exe scripts\reproducibility\recompute_paper_tables.py `
  --json-output paper_artifacts\recomputed_paper_tables.json
```
