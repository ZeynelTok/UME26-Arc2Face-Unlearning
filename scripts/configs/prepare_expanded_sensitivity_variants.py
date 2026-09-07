from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from genmu_face_unlearning.benchmark import load_benchmark
from genmu_face_unlearning.config import save_experiment_config


def _slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _write_group_configs(benchmark: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    expected_names = {
        f"{_slugify(str(group['set_name']))}.json"
        for group in benchmark.get("expanded_groups", [])
    }
    for stale_path in sorted(out_dir.glob("expanded_*.json")):
        if stale_path.name not in expected_names:
            stale_path.unlink()
    for group in benchmark.get("expanded_groups", []):
        experiment_name = _slugify(str(group["set_name"]))
        save_experiment_config(
            {
                "experiment_name": experiment_name,
                "forget_id": int(group["forget_id"]),
                "retain_ids": [int(x) for x in group.get("retain_ids", [])],
                "hard_retain_ids": [int(x) for x in group.get("hard_retain_ids", [])],
                "random_retain_ids": [int(x) for x in group.get("random_retain_ids", [])],
            },
            out_dir / f"{experiment_name}.json",
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build strict top-10/top-20 cuts of the canonical expanded benchmark."
    )
    parser.add_argument(
        "--base-benchmark",
        type=Path,
        default=ROOT / "outputs" / "prep" / "benchmark.json",
    )
    parser.add_argument("--counts", type=int, nargs="+", default=[10, 20])
    parser.add_argument(
        "--benchmarks-root",
        type=Path,
        default=ROOT / "outputs" / "sensitivity" / "benchmarks",
    )
    parser.add_argument(
        "--configs-root",
        type=Path,
        default=ROOT / "configs" / "sensitivity",
    )
    args = parser.parse_args()

    if not args.base_benchmark.is_file():
        raise FileNotFoundError(
            f"Canonical benchmark not found: {args.base_benchmark}. "
            "Run identity preparation before building sensitivity variants."
        )
    base_benchmark = load_benchmark(args.base_benchmark)
    available = base_benchmark.get("expanded_groups", [])
    if not available:
        raise ValueError("Canonical benchmark contains no expanded groups.")

    counts = sorted({int(count) for count in args.counts})
    if any(count <= 0 for count in counts):
        raise ValueError(f"All counts must be positive. Received: {counts}")
    if counts[-1] > len(available):
        raise ValueError(
            f"Requested top-{counts[-1]}, but the canonical benchmark contains "
            f"only {len(available)} groups."
        )

    for count in counts:
        benchmark = copy.deepcopy(base_benchmark)
        benchmark["expanded_groups"] = copy.deepcopy(available[:count])
        benchmark["expanded_selection_policy"] = {
            **benchmark.get("expanded_selection_policy", {}),
            "expanded_forget_count": count,
            "selection_note": (
                f"Strict top-{count} nested cut of the canonical split-first expanded cohort."
            ),
        }

        benchmark_dir = args.benchmarks_root / f"top_{count}"
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        with (benchmark_dir / "benchmark.json").open("w", encoding="utf-8") as handle:
            json.dump(benchmark, handle, indent=2)
            handle.write("\n")

        _write_group_configs(
            benchmark,
            args.configs_root / f"top_{count}" / "expanded",
        )
        print(f"Wrote canonical top-{count} benchmark cut and configs")


if __name__ == "__main__":
    main()
