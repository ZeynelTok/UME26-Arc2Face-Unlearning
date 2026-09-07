FROM python:3.8-slim-bullseye

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 libquadmath0 \
    && rm -rf /var/lib/apt/lists/*

# The upstream SER-FIQ repository targets Python 3.7/3.8 and MXNet 1.8.
# Keeping it in a container avoids changing the paper's PyTorch environment.
RUN python -m pip install --no-cache-dir \
    numpy==1.21.6 \
    mxnet==1.8.0.post0 \
    opencv-python-headless==4.5.3.56 \
    onnx==1.12.0 \
    protobuf==3.20.1 \
    scipy==1.7.3 \
    scikit-learn==0.24.2 \
    scikit-image==0.18.3

WORKDIR /workspace
