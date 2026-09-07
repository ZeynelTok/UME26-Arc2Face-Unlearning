"""Split SER-FIQ before Dropout for the optimization endorsed upstream.

The authors note that when Dropout is at the last layer, the deterministic
trunk need only be evaluated once.  This script derives a batch-1 trunk and a
T-pass stochastic head from the already validated full ONNX graph.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import onnx
from onnx import utils


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _set_batch(value_info, batch: int) -> None:
    dimension = value_info.type.tensor_type.shape.dim[0]
    dimension.ClearField("dim_param")
    dimension.dim_value = batch


def _replace_fixed_batch(model, old_batch: int, new_batch: int) -> None:
    for value_info in [*model.graph.input, *model.graph.output, *model.graph.value_info]:
        shape = value_info.type.tensor_type.shape
        if shape.dim and shape.dim[0].dim_value == old_batch:
            _set_batch(value_info, new_batch)


def _metadata(model, values: dict[str, str]) -> None:
    existing = {item.key: item.value for item in model.metadata_props}
    existing.update(values)
    del model.metadata_props[:]
    for key, value in existing.items():
        item = model.metadata_props.add()
        item.key = key
        item.value = value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-model", type=Path, required=True)
    parser.add_argument("--trunk-output", type=Path, required=True)
    parser.add_argument("--head-output", type=Path, required=True)
    args = parser.parse_args()
    args.trunk_output.parent.mkdir(parents=True, exist_ok=True)
    args.head_output.parent.mkdir(parents=True, exist_ok=True)

    utils.extract_model(
        str(args.full_model),
        str(args.trunk_output),
        ["data"],
        ["bn1"],
        check_model=True,
        infer_shapes=True,
    )
    utils.extract_model(
        str(args.full_model),
        str(args.head_output),
        ["bn1", "dropout_mask"],
        ["fc1"],
        check_model=True,
        infer_shapes=True,
    )

    parent_hash = _sha256(args.full_model)
    trunk = onnx.load(str(args.trunk_output))
    _replace_fixed_batch(trunk, old_batch=100, new_batch=1)
    _metadata(
        trunk,
        {
            "split_role": "deterministic_trunk_batch_1",
            "parent_model_sha256": parent_hash,
            "optimization": "evaluate pre-Dropout trunk once per image",
        },
    )
    onnx.checker.check_model(trunk)
    onnx.save(trunk, str(args.trunk_output))

    head = onnx.load(str(args.head_output))
    _metadata(
        head,
        {
            "split_role": "stochastic_head_T_passes",
            "parent_model_sha256": parent_hash,
            "optimization": "repeat only last Dropout layer and embedding head",
        },
    )
    onnx.checker.check_model(head)
    onnx.save(head, str(args.head_output))

    print(f"Trunk SHA256 {_sha256(args.trunk_output)}")
    print(f"Head SHA256  {_sha256(args.head_output)}")


if __name__ == "__main__":
    main()
