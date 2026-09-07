"""Produce a deterministic MXNet reference output for SER-FIQ conversion.

Run inside the pinned ``genmu-serfiq-official:mxnet1.8`` image.  The official
graph's sole Dropout node is temporarily set to p=0.  Comparing this output
with the converted ONNX graph supplied an all-ones mask verifies every
non-stochastic operation and parameter independently of random-number APIs.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import cv2
import mxnet as mx
import numpy as np
from mxnet import gluon


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", type=Path, required=True)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--aligned-bgr", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    graph = json.loads(args.symbol.read_text(encoding="utf-8"))
    dropout = [node for node in graph["nodes"] if node["op"] == "Dropout"]
    if len(dropout) != 1 or dropout[0].get("attrs", {}).get("p") != "0.5":
        raise RuntimeError(f"Unexpected official Dropout nodes: {dropout}")
    dropout[0]["attrs"]["p"] = "0"

    aligned_bgr = np.load(args.aligned_bgr)
    if aligned_bgr.shape != (112, 112, 3):
        raise ValueError(f"Expected a 112x112 BGR crop, got {aligned_bgr.shape}")
    rgb_chw = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
    batch = np.repeat(rgb_chw[None, :, :, :], args.batch_size, axis=0).astype(np.float32)

    with tempfile.TemporaryDirectory(prefix="serfiq_reference_") as temporary:
        symbol_path = Path(temporary) / "no_dropout-symbol.json"
        symbol_path.write_text(json.dumps(graph), encoding="utf-8")
        model = gluon.nn.SymbolBlock.imports(
            str(symbol_path),
            ["data"],
            str(args.params),
            ctx=mx.cpu(),
        )
        output = model(mx.nd.array(batch, ctx=mx.cpu())).asnumpy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, output)
    print(f"Wrote {args.output} with shape {output.shape}")


if __name__ == "__main__":
    main()
