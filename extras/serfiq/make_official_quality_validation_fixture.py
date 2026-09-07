"""Create the deterministic 112x112 BGR fixture used by SER-FIQ checks."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fixture = np.random.default_rng(7).integers(0, 256, size=(112, 112, 3), dtype=np.uint8)
    np.save(args.output, fixture)
    print(f"Wrote {args.output} with shape {fixture.shape} and dtype {fixture.dtype}")


if __name__ == "__main__":
    main()
