# Official SER-FIQ and GraFIQs evaluation

This directory documents the paper's post-hoc face-image-quality analysis. It
scores existing generated images and does not train adapters or generate new
images.

## Frozen evaluation protocol

Every generated and real-reference image uses the same preprocessing:

1. InsightFace `antelopev2` SCRFD detection at `det_thresh=0.1`.
2. Select the face with largest bounding-box area.
3. Five-landmark `norm_crop` alignment to 112x112.
4. Do not use a fallback crop. Record unreadable/no-face/error inputs as rows.

The quality methods are:

- **SER-FIQ (on the authors' released ArcFace):** upstream commit
  `611296605db57b8d50518fd5911d5111eeb52747`, its dropout-trained ArcFace,
  Dropout(0.5), 100 stochastic passes, L2-normalised embeddings, mean unique
  pairwise Euclidean distance, paper transform `2/(1+exp(d))`, and the
  upstream `alpha=130`, `r=0.88` logistic normalisation. A repository-relative
  image key and per-image SHA-256 seed make masks independent of clone location
  and traversal order. Following the authors' recommendation, the deterministic
  trunk is evaluated once and only the final Dropout/embedding head is repeated.
- **GraFIQs (on the authors' released ArcFace ResNet-100):** upstream commit
  `8d11a1f8506fb0f86ebd7738afc31467e2a01058` and the released
  `resnet100_ms1mv2_arcface.pth`. The primary endpoint is the absolute gradient
  sum at B2, which the GraFIQs paper selected for its SOTA comparison. Image,
  B1, B3, and B4 gradients are also retained. Higher gradient magnitude means
  lower face-recognition utility. OpenCV BGR input is retained, matching the
  upstream command's default.

The real baseline contains 960 unique CelebA photographs. The historical
1,260 count is the number of group-membership rows after shared references are
repeated; it must not be described as 1,260 distinct photographs.

Raw SER-FIQ values and the manuscript's probability-superiority percentages
are different quantities. For each group, every detected generated image is
compared only with the six unique real references of its intended identity.
The paper reports the equal-weight mean of these 30 group-level probabilities.
For GraFIQs-B2, lower raw gradient is treated as better. Thus table values are
identity-matched `P(quality(generated) >= quality(real))`; 50% means no
stochastic-ordering advantage in this comparison, not perceptual
indistinguishability or statistical equivalence. The former all-generated
versus all-real global pool is retained in `comparison.json` only as a
sensitivity statistic.

## Official sources and checkpoints

The source trees and weights are kept under the ignored `models/` directory.

```powershell
git clone https://github.com/pterhoer/FaceImageQuality.git models\official_quality\ser_fiq_source
git -C models\official_quality\ser_fiq_source checkout 611296605db57b8d50518fd5911d5111eeb52747

git clone https://github.com/jankolf/grafiqs.git models\official_quality\grafiqs_source
git -C models\official_quality\grafiqs_source checkout 8d11a1f8506fb0f86ebd7738afc31467e2a01058
```

Download the SER-FIQ model archive from the link in the upstream README:

`https://drive.google.com/file/d/17fEWczMzTUDzRTv9qN3hFwVbkqRD7HE7/view?usp=sharing`

Place its two model files at:

```text
models/official_quality/ser_fiq_source/insightface/model/insightface-symbol.json
models/official_quality/ser_fiq_source/insightface/model/insightface-0000.params
```

Download GraFIQs weights from the authors' link
`https://share.jankolf.de/s/WWCXmNkj7FTcRpR` and place
`resnet100_ms1mv2_arcface.pth` in `models/official_quality/`.

Expected SHA-256 hashes are:

| File | SHA-256 |
|---|---|
| SER-FIQ archive | `fdf6169c23bb6db96509709f8d7ed9ff4b7f2f2ec6686e9271a2f91337211cb5` |
| `insightface-symbol.json` | `836045e55b546e87409d4877dbefaec94cffabf03c626eac611ff7af4404a6c5` |
| `insightface-0000.params` | `8acca797c0a0649f8b00e698349462ebd81f467cbe2012df56cfa30794655e45` |
| `resnet100_ms1mv2_arcface.pth` | `f2153932123dbb8695fca2d307e9eab618361c16d416f82ea83b81f3ef33cecb` |

## Export and independent validation

MXNet 1.8 is isolated from the paper environment in a small CPU Docker image.
The production ONNX graph replaces upstream's always-on Dropout node with an
explicit inverted Bernoulli mask input. This preserves the distribution while
making each image reproducible.

```powershell
docker build -t genmu-serfiq-official:mxnet1.8 extras\serfiq

docker run --rm -v "${PWD}:/repo" -w /repo genmu-serfiq-official:mxnet1.8 `
  python extras/serfiq/export_official_serfiq_onnx.py `
  --symbol models/official_quality/ser_fiq_source/insightface/model/insightface-symbol.json `
  --params models/official_quality/ser_fiq_source/insightface/model/insightface-0000.params `
  --output models/official_quality/ser_fiq_arcface_t100_mask.onnx `
  --num-passes 100
```

The expected exported-model SHA-256 is
`ec6e2774de56e491d1f42fbecd4f02233ceed90f42ea3a45cf5b0eeda82355b8`.

Create the optimized trunk/head pair used by the production scorer:

```powershell
venv\Scripts\python.exe extras\serfiq\split_official_serfiq_onnx.py `
  --full-model models\official_quality\ser_fiq_arcface_t100_mask.onnx `
  --trunk-output models\official_quality\ser_fiq_arcface_trunk_b1.onnx `
  --head-output models\official_quality\ser_fiq_arcface_head_t100.onnx
```

Their expected SHA-256 hashes are
`4569740d430a0ca19a3d39d642f2efd37fc188f041f8d1f74f0d7b33456018a2`
(trunk) and
`a772d0b0adfb696a552413c1eb2ae2b454bc39fb75b8c44addde18ced94f9339`
(head). With identical masks, full and split CPU scores are bit-identical. The
observed CUDA difference on the validation face was `4.0523e-05` in normalized
quality, caused by different batch-1/batch-100 convolution kernels.

The following compares every non-stochastic operation and parameter against
an official MXNet CPU oracle. The tested maximum absolute embedding-component
difference was `1.07288361e-06`.

```powershell
venv\Scripts\python.exe extras\serfiq\make_official_quality_validation_fixture.py `
  --output outputs\quality_official_v1_validation\deterministic_aligned_bgr.npy

docker run --rm -v "${PWD}:/repo" -w /repo genmu-serfiq-official:mxnet1.8 `
  python extras/serfiq/serfiq_mxnet_reference.py `
  --symbol models/official_quality/ser_fiq_source/insightface/model/insightface-symbol.json `
  --params models/official_quality/ser_fiq_source/insightface/model/insightface-0000.params `
  --aligned-bgr outputs/quality_official_v1_validation/deterministic_aligned_bgr.npy `
  --output outputs/quality_official_v1_validation/mxnet_no_dropout_t100.npy `
  --batch-size 100

venv\Scripts\python.exe extras\serfiq\validate_official_serfiq_onnx.py `
  --model models\official_quality\ser_fiq_arcface_t100_mask.onnx `
  --aligned-bgr outputs\quality_official_v1_validation\deterministic_aligned_bgr.npy `
  --mxnet-output outputs\quality_official_v1_validation\mxnet_no_dropout_t100.npy `
  --device cpu
```

`score_serfiq_unmodified.py` additionally runs the authors' untouched
`SER_FIQ.get_score` to compare the accelerated scorer with native T=100 Monte
Carlo variation. GraFIQs needs no conversion: `OfficialGraFIQsScorer` imports
the pinned upstream `extract_grafiqs.py` model, transforms, BN loss, and
gradient calculation directly.

## Full scoring and table comparison

```powershell
venv\Scripts\python.exe -m unittest tests.test_official_quality

venv\Scripts\python.exe scripts\evaluation\build_official_quality_scores.py `
  --scope all `
  --output-dir outputs\quality_official_v1 `
  --ser-fiq-trunk-model models\official_quality\ser_fiq_arcface_trunk_b1.onnx `
  --ser-fiq-head-model models\official_quality\ser_fiq_arcface_head_t100.onnx

venv\Scripts\python.exe scripts\evaluation\build_official_quality_comparison.py `
  --input-dir outputs\quality_official_v1
```

The scorer expects 30 groups, 168 generated images per policy
and group, 5,040 images per policy, and 20,160 generated images in total. Its
CSV output is append-only and resumable. `comparison.json` rejects unreadable
or scoring-error rows. It retains `no_face` rows, reports attempted/scored
denominators, and gives both the identity-matched, group-balanced conditional
comparison and a conservative end-to-end comparison that treats generated
no-face cases as losses. Every real identity is required to have six scored
references. The JSON additionally records the superseded global-pool value as
a labelled sensitivity statistic.
