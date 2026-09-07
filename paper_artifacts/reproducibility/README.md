# Paper environment

This directory records the software, hardware, and external pretrained files
used for the paper experiments.

Create a Python 3.12 environment, install the recorded package versions, and
run the checks from the repository root:

```powershell
py -3.12 -m venv venv
venv\Scripts\python.exe -m pip install -r paper_artifacts\reproducibility\requirements-paper-cu128.lock
venv\Scripts\python.exe scripts\reproducibility\verify_reproducibility.py
```

The first check covers installed versions, CUDA availability, the ONNX Runtime
CUDA provider, and required external model file sizes. To also check the
external model hashes:

```powershell
venv\Scripts\python.exe scripts\reproducibility\verify_reproducibility.py --verify-model-hashes
```

`models.lock.json` identifies Arc2Face, AdaFace, the InsightFace ONNX files, and
the locally resolved Stable Diffusion v1.5 snapshot. Its hashes apply only to
external model and third-party source files, not repository outputs or local
scientific source files.

These records describe the Windows/CUDA environment used for the experiments.
Per-image seeds control reruns, but diffusion output may differ across GPU
architectures, CUDA libraries, or driver versions. The saved artifacts reproduce
the reported table values without regenerating the images.

To update the environment and external-model records after a change:

```powershell
venv\Scripts\python.exe scripts\reproducibility\capture_reproducibility_lock.py
```
