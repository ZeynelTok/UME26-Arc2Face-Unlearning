# The Nearest Target Is the Wrong One

This is the code and saved evidence for *The Nearest Target Is the Wrong One:
Target Separation in Arc2Face Identity Unlearning*, accepted at the [3rd
Workshop and Challenge on Unlearning and Model Editing (U&ME), ECCV
2026](https://sites.google.com/view/u-and-me-workshop/). A proceedings link will
be added when it is available.

The repository is a companion to the paper. It contains the experiment code,
figures, identity split, and results used for the tables. CelebA, model weights,
generated images, checkpoints, and working outputs are left out because of
their size or licence terms.

Project page: <https://zeyneltok.github.io/UME26-Arc2Face-Unlearning/>

## Main result

Moving the target farther from the forgotten identity sharply reduces
forgetting failures and identity leakage. Retention stays close to 89% across
all four target policies.

| Policy | FA=0 groups | Mean FA ↓ | Mean RA ↑ | Mean ERB ↑ | Mean Leak@8 ↓ | Mean sim. | RA_target ↑ | RA_hard ↑ | RA_random ↑ | SER-FIQ ≥ real ↑ | GraFIQs ≥ real ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nearest-hard | 9/30 | 51.94 | 88.79 | 50.34 | 66.67 | 0.8689 | 53.89 | 79.81 | 97.85 | 70.28% | 10.64% |
| random-hard | 20/30 | 25.83 | 88.86 | 71.81 | 31.11 | 0.5460 | 77.08 | 79.96 | 97.85 | 70.48% | 10.77% |
| median-hard | 28/30 | 2.64 | 88.75 | 92.35 | 6.67 | 0.3403 | 92.22 | 79.78 | 97.80 | 69.62% | 10.81% |
| least-sim-hard | 30/30 | 0.00 | 88.86 | 93.84 | 0.00 | 0.2874 | 94.28 | 79.90 | 97.89 | 69.07% | 11.12% |

FA and Leak@8 are better when lower. RA, ERB, and the quality comparisons are
better when higher. Values use the same rounding as the paper.

<p align="center">
  <a href="paper_assets/figures/intro_teaser.pdf">
    <img src="paper_assets/figures/intro_teaser.png" width="900" alt="A nearby target can remain inside the forgotten identity's verification region, while a separated target leaves it." />
  </a>
</p>

<p align="center"><em>A nearby target can still verify as the forgotten identity. Click the figure for the PDF.</em></p>

<p align="center">
  <a href="paper_assets/figures/expanded_mechanistic_summary.pdf">
    <img src="paper_assets/figures/expanded_mechanistic_summary.png" width="900" alt="Results for target similarity, forgetting accuracy, target arrival, and Leak at 8." />
  </a>
</p>

<p align="center"><em>Results across 30 groups and four target policies. Click the figure for the PDF.</em></p>

## Recompute the paper tables

The quickest way to check the results is to recompute the tables from the
files in [`paper_artifacts/`](paper_artifacts/README.md). This does not train a
model or generate images.

```powershell
py -3.12 -m venv venv
venv\Scripts\python.exe -m pip install -r paper_artifacts\reproducibility\requirements-paper-cu128.lock
venv\Scripts\python.exe scripts\reproducibility\recompute_paper_tables.py `
  --json-output paper_artifacts\recomputed_paper_tables.json
```

The script prints all three tables and writes their full-precision values to
`paper_artifacts/recomputed_paper_tables.json`.

<details>
<summary><strong>Run the full experiment</strong></summary>

Run these commands from the repository root. The scripts create `configs/`
and `outputs/`, which Git ignores.

### 1. Add CelebA and pretrained models

Download aligned CelebA images and annotations from the [official CelebA
page](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), subject to its access
and non-commercial research terms. Put the files here:

```text
Data/
|-- validation-splits.json          # included
|-- list_eval_partition.txt
|-- Anno/
|   |-- identity_CelebA.txt
|   `-- list_landmarks_align_celeba.txt
`-- img_align_celeba/
    `-- *.jpg
```

The identity annotation may require a request to the dataset maintainers. A
complete copy has 202,599 aligned images and 10,177 identities. Preparation
stops if images or annotations are missing.

Clone Arc2Face and its matching revision:

```powershell
git clone https://github.com/foivospar/Arc2Face models\Arc2Face
git -C models\Arc2Face checkout 8f3acd701d17fda7fde4c9f1d170fc88fecbe9ad
venv\Scripts\python.exe scripts\setup\download_arc2face.py `
  --out-dir models\Arc2Face `
  --arcface-recognizer-dir models\insightface\models\antelopev2
```

Download InsightFace
[`antelopev2`](https://github.com/deepinsight/insightface/tree/master/python-package)
to `models/insightface/models/antelopev2/`. Keep `arcface.onnx` with the four
standard antelopev2 ONNX files, and remove or rename `glintr100.onnx` so that
InsightFace loads the intended recognizer.

For the second recognizer, clone [AdaFace](https://github.com/mk-minchul/AdaFace)
to `models/adaface/` and place the authors' R50 WebFace4M checkpoint at
`models/adaface/weights/adaface_ir50_webface4m.ckpt`.

The quality comparison also uses the official SER-FIQ and GraFIQs repositories:

```powershell
git clone https://github.com/pterhoer/FaceImageQuality.git models\official_quality\ser_fiq_source
git -C models\official_quality\ser_fiq_source checkout 611296605db57b8d50518fd5911d5111eeb52747
git clone https://github.com/jankolf/grafiqs.git models\official_quality\grafiqs_source
git -C models\official_quality\grafiqs_source checkout 8d11a1f8506fb0f86ebd7738afc31467e2a01058
```

See [`extras/serfiq/README.md`](extras/serfiq/README.md) for the SER-FIQ model
conversion and GraFIQs checkpoint paths. Stable Diffusion v1.5 is downloaded
through the Hugging Face cache. Model revisions and hashes are recorded in
`paper_artifacts/reproducibility/models.lock.json`.

### 2. Prepare the benchmark

```powershell
venv\Scripts\python.exe scripts\data_prep\prepare_identities.py `
  --data-dir Data `
  --out-dir outputs\prep `
  --public-splits Data\validation-splits.json `
  --arcface-model-root models\insightface `
  --device cuda `
  --partitions 0 `
  --min-images-per-identity 15 `
  --expanded-forget-count 30 `
  --fit-reference-count 3 `
  --eval-reference-count 3

venv\Scripts\python.exe scripts\configs\build_group_configs.py `
  --benchmark outputs\prep\benchmark.json `
  --out-dir configs\expanded

venv\Scripts\python.exe scripts\evaluation\validate_gallery_references.py
```

This prepares the identity splits, embeddings, verification threshold, 30
groups, and held-out gallery.

### 3. Run the four policies

```powershell
foreach ($policy in "nearest-hard", "random-hard", "median-hard", "least-sim-hard") {
  venv\Scripts\python.exe scripts\sweeps\run_expanded_target_selection_sweep.py `
    --policy $policy `
    --evaluation-device cuda `
    --reference-det-thresh 0.5 `
    --evaluation-det-thresh 0.1 `
    --random-seed 2026 `
    --skip-if-exists
}
```

Each policy trains 30 rank-8 adapters, generates 168 images per group, and
evaluates them against the held-out gallery.

### 4. Build the remaining results

```powershell
venv\Scripts\python.exe scripts\evaluation\build_expanded_mechanistic_summary.py
venv\Scripts\python.exe scripts\configs\prepare_expanded_sensitivity_variants.py `
  --base-benchmark outputs\prep\benchmark.json `
  --counts 10 20 `
  --benchmarks-root outputs\sensitivity\benchmarks `
  --configs-root configs\sensitivity
venv\Scripts\python.exe scripts\evaluation\compare_expanded_sensitivity.py `
  --benchmarks-root outputs\sensitivity\benchmarks `
  --results-root outputs\sensitivity\runs `
  --canonical-results-root outputs `
  --counts 10 20 `
  --policies nearest-hard random-hard median-hard least-sim-hard `
  --out-dir outputs\sensitivity\comparison
venv\Scripts\python.exe scripts\evaluation\calibrate_adaface_threshold.py
venv\Scripts\python.exe scripts\evaluation\verify_with_adaface.py
venv\Scripts\python.exe scripts\evaluation\validate_large_gallery_leakage.py
venv\Scripts\python.exe scripts\evaluation\build_official_quality_scores.py `
  --scope all `
  --output-dir outputs\quality_official_v1 `
  --ser-fiq-trunk-model models\official_quality\ser_fiq_arcface_trunk_b1.onnx `
  --ser-fiq-head-model models\official_quality\ser_fiq_arcface_head_t100.onnx
venv\Scripts\python.exe scripts\evaluation\build_official_quality_comparison.py `
  --input-dir outputs\quality_official_v1
```

Build a fresh artefact bundle after the runs finish:

```powershell
venv\Scripts\python.exe scripts\reproducibility\build_paper_artifacts.py --overwrite
venv\Scripts\python.exe scripts\reproducibility\recompute_paper_tables.py `
  --json-output paper_artifacts\recomputed_paper_tables.json
```

</details>

<details>
<summary><strong>Hardware and reproducibility notes</strong></summary>

The 120-run adapter, generation, and ArcFace experiment took about 12 hours on
an NVIDIA GeForce RTX 5070. SER-FIQ and GraFIQs scoring took about 65 minutes.
Working outputs used roughly 17.6 GB; the checked-in artefacts are under 10 MB.

The experiment fixes identity selection, reference splits, calibration
sampling, target selection, training seeds, and generation seeds. Diffusion
images may still vary across GPUs, drivers, and CUDA versions. The saved
per-image results reproduce the paper tables without regenerating those images.

To check the recorded environment and external models:

```powershell
venv\Scripts\python.exe scripts\reproducibility\verify_reproducibility.py
venv\Scripts\python.exe scripts\reproducibility\verify_reproducibility.py --verify-model-hashes
```

</details>

## Tests and figures

Run the tests with:

```powershell
venv\Scripts\python.exe -m unittest discover -s tests -v
```

The other paper figures are available as PDFs:

- [Hardness distribution](paper_assets/figures/hardness_histogram.pdf)
- [Forget-identity resemblance](paper_assets/figures/forget_resemblance.pdf)
- [Secondary-recognizer results](paper_assets/figures/second_recognizer.pdf)

Their source scripts are in `scripts/figures/`.
