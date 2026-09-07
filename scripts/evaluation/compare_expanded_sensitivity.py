from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path


POLICY_SLUGS = {
    "nearest-hard": "nearest_hard",
    "least-sim-hard": "least_sim_hard",
    "random-hard": "random_hard",
    "median-hard": "median_hard",
}


def _safe_mean(values: list[float]) -> float | None:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


def _load_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _load_benchmark(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _canonical_policy_rows(canonical_root: Path, selected_ids: list[int], policy_slug: str) -> list[dict] | None:
    rows = []
    missing = []
    for gid in selected_ids:
        group_dir = canonical_root / f"expanded_{gid}"
        metrics_path = group_dir / f"eval_{policy_slug}" / "metrics.json"
        adapter_path = group_dir / f"u_projection_adapter_{policy_slug}" / "projection_adapter_summary.json"
        if not metrics_path.exists() or not adapter_path.exists():
            missing.append(str(group_dir))
            continue
        metrics = _load_summary(metrics_path)
        adapter = _load_summary(adapter_path)
        rows.append(
            {
                "experiment_name": f"expanded_{gid}",
                "forget_id": int(gid),
                "target_identity": adapter.get("target_identity"),
                "target_similarity_to_forget": adapter.get("target_similarity_to_forget"),
                "fa": metrics.get("FA"),
                "ra": metrics.get("RA"),
                "erb": metrics.get("ERB"),
                "identity_leak_at_8": metrics.get("IdentityLeak@8"),
                "metrics_path": str(metrics_path),
            }
        )
    if missing:
        return None
    return rows


def _format_float(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate top-10/top-20 expanded sensitivity sweeps across target-selection policies."
    )
    parser.add_argument("--benchmarks-root", type=Path, default=Path("outputs") / "sensitivity" / "benchmarks")
    parser.add_argument("--results-root", type=Path, default=Path("outputs") / "sensitivity" / "runs")
    parser.add_argument("--canonical-results-root", type=Path, default=Path("outputs"))
    parser.add_argument("--counts", type=int, nargs="+", default=[10, 20])
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["nearest-hard", "random-hard", "median-hard", "least-sim-hard"],
        choices=sorted(POLICY_SLUGS),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs") / "sensitivity" / "comparison")
    args = parser.parse_args()

    counts = sorted({int(count) for count in args.counts})
    benchmark_rows = []
    benchmark_sets: dict[int, set[int]] = {}
    metrics_rows = []

    for count in counts:
        benchmark_path = args.benchmarks_root / f"top_{count}" / "benchmark.json"
        selected_ids = []
        if benchmark_path.exists():
            benchmark = _load_benchmark(benchmark_path)
            selected_ids = [int(group["forget_id"]) for group in benchmark.get("expanded_groups", [])]
            benchmark_sets[count] = set(selected_ids)
            benchmark_rows.append(
                {
                    "count": count,
                    "group_count": len(selected_ids),
                    "forget_ids": " ".join(str(x) for x in selected_ids),
                    "selection_note": benchmark.get("expanded_selection_policy", {}).get("selection_note", ""),
                }
            )

        for policy in args.policies:
            policy_slug = POLICY_SLUGS[policy]
            summary_path = args.results_root / f"top_{count}" / f"expanded_target_selection_{policy_slug}_summary.json"
            if not summary_path.exists():
                rows = _canonical_policy_rows(args.canonical_results_root, selected_ids, policy_slug) if selected_ids else None
                if rows is None:
                    continue
                summary_source = f"derived from {args.canonical_results_root}"
            else:
                summary = _load_summary(summary_path)
                rows = summary.get("rows", [])
                summary_source = str(summary_path)
            fa_values = [row.get("fa") for row in rows]
            ra_values = [row.get("ra") for row in rows]
            erb_values = [row.get("erb") for row in rows]
            leak_values = [row.get("identity_leak_at_8") for row in rows]
            sim_values = [row.get("target_similarity_to_forget") for row in rows]
            metrics_rows.append(
                {
                    "count": count,
                    "policy": policy,
                    "group_count": len(rows),
                    "fa_zero_count": sum(1 for value in fa_values if value == 0.0),
                    "fa_mean": _safe_mean(fa_values),
                    "ra_mean": _safe_mean(ra_values),
                    "erb_mean": _safe_mean(erb_values),
                    "identity_leak_at_8_mean": _safe_mean(leak_values),
                    "target_similarity_mean": _safe_mean(sim_values),
                    "summary_path": summary_source,
                }
            )

    overlap_rows = []
    for left, right in combinations(counts, 2):
        if left not in benchmark_sets or right not in benchmark_sets:
            continue
        left_ids = benchmark_sets[left]
        right_ids = benchmark_sets[right]
        overlap = sorted(left_ids & right_ids)
        overlap_rows.append(
            {
                "left_count": left,
                "right_count": right,
                "overlap_count": len(overlap),
                "left_only_count": len(left_ids - right_ids),
                "right_only_count": len(right_ids - left_ids),
                "overlap_forget_ids": " ".join(str(x) for x in overlap),
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(
        args.out_dir / "selection_summary.csv",
        benchmark_rows,
        ["count", "group_count", "forget_ids", "selection_note"],
    )
    _write_csv(
        args.out_dir / "selection_overlap.csv",
        overlap_rows,
        ["left_count", "right_count", "overlap_count", "left_only_count", "right_only_count", "overlap_forget_ids"],
    )
    _write_csv(
        args.out_dir / "policy_metrics.csv",
        [
            {
                **row,
                "fa_mean": _format_float(row["fa_mean"]),
                "ra_mean": _format_float(row["ra_mean"]),
                "erb_mean": _format_float(row["erb_mean"]),
                "identity_leak_at_8_mean": _format_float(row["identity_leak_at_8_mean"]),
                "target_similarity_mean": _format_float(row["target_similarity_mean"]),
            }
            for row in metrics_rows
        ],
        [
            "count",
            "policy",
            "group_count",
            "fa_zero_count",
            "fa_mean",
            "ra_mean",
            "erb_mean",
            "identity_leak_at_8_mean",
            "target_similarity_mean",
            "summary_path",
        ],
    )

    report = {
        "counts": counts,
        "policies": list(args.policies),
        "selection_summary": benchmark_rows,
        "selection_overlap": overlap_rows,
        "policy_metrics": metrics_rows,
    }
    with (args.out_dir / "expanded_count_sensitivity_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    lines = [
        "# Expanded-Count Sensitivity Summary",
        "",
        "This report compares top-10/top-20 expanded benchmark variants under the same locked pipeline.",
        "",
        "## Selection Overlap",
        "",
        "| Left | Right | Overlap | Left Only | Right Only |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in overlap_rows:
        lines.append(
            f"| top-{row['left_count']} | top-{row['right_count']} | {row['overlap_count']} | "
            f"{row['left_only_count']} | {row['right_only_count']} |"
        )
    if not overlap_rows:
        lines.append("| n/a | n/a | n/a | n/a | n/a |")

    lines.extend(
        [
            "",
            "## Policy Metrics",
            "",
            "| Count | Policy | Groups | FA=0 | Mean FA | Mean RA | Mean ERB | Mean Leak@8 | Mean Target Sim |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in metrics_rows:
        lines.append(
            f"| {row['count']} | {row['policy']} | {row['group_count']} | {row['fa_zero_count']} | "
            f"{_format_float(row['fa_mean'])} | {_format_float(row['ra_mean'])} | {_format_float(row['erb_mean'])} | "
            f"{_format_float(row['identity_leak_at_8_mean'])} | {_format_float(row['target_similarity_mean'])} |"
        )
    if not metrics_rows:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")

    with (args.out_dir / "expanded_count_sensitivity_report.md").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")

    print(f"Wrote comparison outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
