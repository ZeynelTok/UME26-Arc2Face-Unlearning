# Paper environment

This folder records the software, hardware, and pretrained files used for the
paper experiments.

Create a Python 3.12 environment, install the recorded package versions, and
run the checks from the repository root:

```powershell
py -3.12 -m venv venv
venv\Scripts\python.exe -m pip install -r paper_artifacts\reproducibility\requirements-paper-cu128.lock
venv\Scripts\python.exe scripts\reproducibility\verify_reproducibility.py
```

This checks package versions, CUDA, the ONNX Runtime CUDA provider, and the
expected model files. Add the model-hash check with:

```powershell
venv\Scripts\python.exe scripts\reproducibility\verify_reproducibility.py --verify-model-hashes
```

`models.lock.json` identifies Arc2Face, AdaFace, the InsightFace ONNX files, and
the Stable Diffusion v1.5 snapshot used in the experiments. The hashes cover
external models and third-party code.

The recorded environment used Windows and CUDA. Per-image seeds control reruns,
but generated images may still vary across GPUs, CUDA libraries, and drivers.
The saved results reproduce the tables without regenerating images.

To refresh these records:

```powershell
venv\Scripts\python.exe scripts\reproducibility\capture_reproducibility_lock.py
```
