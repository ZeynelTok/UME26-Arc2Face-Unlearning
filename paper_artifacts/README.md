# Paper artifacts

These files are enough to inspect and recompute the published results without
rerunning adapter training or image generation. CelebA, generated images, and
pretrained model weights are not included.

Files in this directory:

- `run_metrics.csv` and `per_sample_predictions.csv.gz`: run-level metrics and
  the detected samples from which they are calculated.
- `generation_attempts.csv.gz`: all 20,160 attempted generations, including
  detection failures.
- `adapter_summaries.jsonl` and `adapters.npz`: the selected checkpoints for
  all 120 runs. `generation_runs.jsonl` records the generation settings without
  runner-specific orchestration fields.
- `target_candidates.csv`, `target_selections.csv`, and
  `hardness_top3_neighbors.csv.gz`: target-policy and neighbour evidence.
- `quality_scores.csv.gz`, `recognizer_runs.csv`, `adaface_predictions.csv.gz`,
  and `large_gallery_predictions.csv.gz`: additional evaluation evidence.
- `sensitivity_policy_metrics.csv`: the top-10/top-20 nested-cut experiment.
- `protocol.json`: exact groups, references, split, and calibration protocol.
- `reported_summaries.json` and `paper_table_values.tex`: derived summaries.

To rebuild these files and recompute the tables from completed experiment
outputs:

```powershell
venv\Scripts\python.exe scripts\reproducibility\build_paper_artifacts.py --overwrite
venv\Scripts\python.exe scripts\reproducibility\recompute_paper_tables.py `
  --json-output paper_artifacts\recomputed_paper_tables.json
```
