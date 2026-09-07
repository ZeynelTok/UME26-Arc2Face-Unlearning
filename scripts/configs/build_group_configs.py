from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from genmu_face_unlearning.config import save_experiment_config


def _slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _load_benchmark(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate experiment config JSON files from expanded benchmark groups.")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    benchmark = _load_benchmark(args.benchmark)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    expected_names = {
        f"{_slugify(str(group['set_name']))}.json"
        for group in benchmark["expanded_groups"]
    }
    for stale_path in sorted(args.out_dir.glob("expanded_*.json")):
        if stale_path.name not in expected_names:
            stale_path.unlink()
            print(f"Removed stale generated config {stale_path}")

    for group in benchmark["expanded_groups"]:
        set_name = str(group["set_name"])
        experiment_name = _slugify(set_name)
        config = {
            "experiment_name": experiment_name,
            "forget_id": int(group["forget_id"]),
            "retain_ids": [int(x) for x in group.get("retain_ids", [])],
            "hard_retain_ids": [int(x) for x in group.get("hard_retain_ids", [])],
            "random_retain_ids": [int(x) for x in group.get("random_retain_ids", [])],
        }

        out_path = args.out_dir / f"{experiment_name}.json"
        save_experiment_config(config, out_path)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
