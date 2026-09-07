"""Export the pinned official SER-FIQ MXNet graph to reproducible ONNX.

Run this script inside ``genmu-serfiq-official:mxnet1.8``.  The exported graph
uses the authors' exact parameters and operations.  Its always-on Dropout(0.5)
node is replaced by an explicit mask input so production inference can use an
order-independent, per-image random seed.
"""
from __future__ import annotations

import argparse
import hashlib
import tempfile
from pathlib import Path

import numpy as np
import onnx
from mxnet.contrib import onnx as onnx_mxnet
from onnx import TensorProto, helper, numpy_helper


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", type=Path, required=True)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-passes", type=int, default=100)
    args = parser.parse_args()
    if args.num_passes < 2:
        raise ValueError("--num-passes must be at least 2")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="serfiq_export_") as temporary:
        raw_path = Path(temporary) / "serfiq_raw.onnx"
        onnx_mxnet.export_model(
            str(args.symbol),
            str(args.params),
            [(args.num_passes, 3, 112, 112)],
            np.float32,
            str(raw_path),
            opset_version=12,
        )
        model = onnx.load(str(raw_path))

    initializers = {item.name: item for item in model.graph.initializer}
    dropout_nodes = [node for node in model.graph.node if node.op_type == "Dropout"]
    if len(dropout_nodes) != 1:
        raise RuntimeError(f"Expected one Dropout node, found {len(dropout_nodes)}")
    dropout = dropout_nodes[0]
    ratio = float(numpy_helper.to_array(initializers[dropout.input[1]]))
    if ratio != 0.5:
        raise RuntimeError(f"Expected Dropout(0.5), found {ratio}")

    # MXNet PReLU parameters are channel vectors.  ONNX requires explicit
    # channel/spatial broadcasting for NCHW tensors.
    for node in model.graph.node:
        if node.op_type != "PRelu":
            continue
        slope = initializers[node.input[1]]
        values = numpy_helper.to_array(slope)
        slope.CopyFrom(numpy_helper.from_array(values.reshape(-1, 1, 1), name=slope.name))

    dropout_index = list(model.graph.node).index(dropout)
    explicit_mask = helper.make_node(
        "Mul",
        [dropout.input[0], "dropout_mask"],
        [dropout.output[0]],
        name="dropout0_explicit_mask",
    )
    del model.graph.node[dropout_index]
    model.graph.node.insert(dropout_index, explicit_mask)
    model.graph.input.append(
        helper.make_tensor_value_info(
            "dropout_mask",
            TensorProto.FLOAT,
            [args.num_passes, 512, 7, 7],
        )
    )

    # Old MXNet exporters list every initializer as an overridable graph input.
    # Removing those duplicates leaves only the image and mask as true inputs.
    initializer_names = set(initializers)
    true_inputs = [item for item in model.graph.input if item.name not in initializer_names]
    del model.graph.input[:]
    model.graph.input.extend(true_inputs)

    # Remove the no-longer-used Dropout ratio initializer.
    used_inputs = {name for node in model.graph.node for name in node.input}
    kept_initializers = [item for item in model.graph.initializer if item.name in used_inputs]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_initializers)

    metadata = {
        "ser_fiq_source_commit": "611296605db57b8d50518fd5911d5111eeb52747",
        "mxnet_symbol_sha256": sha256(args.symbol),
        "mxnet_params_sha256": sha256(args.params),
        "num_stochastic_passes": str(args.num_passes),
        "dropout_probability": "0.5",
        "dropout_protocol": "explicit order-independent Bernoulli mask; inverted scaling",
    }
    del model.metadata_props[:]
    for key, value in metadata.items():
        prop = model.metadata_props.add()
        prop.key = key
        prop.value = value

    onnx.checker.check_model(model)
    onnx.save(model, str(args.output))
    print(f"Wrote {args.output}")
    print(f"SHA256 {sha256(args.output)}")


if __name__ == "__main__":
    main()
