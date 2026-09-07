"""Rebuild expanded-sweep summaries from per-group outputs."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

POLICIES = ["nearest_hard", "random_hard", "median_hard", "least_sim_hard"]
POLICY_LABELS = {
    "nearest_hard": "nearest-hard",
    "random_hard": "random-hard",
    "median_hard": "median-hard",
    "least_sim_hard": "least-sim-hard",
}


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _adapter_summary_path(out_root: Path, experiment_name: str, policy: str) -> Path:
    return out_root / experiment_name / f"u_projection_adapter_{policy}" / "projection_adapter_summary.json"


def _eval_metrics_path(out_root: Path, experiment_name: str, policy: str) -> Path:
    return out_root / experiment_name / f"eval_{policy}" / "metrics.json"


def _per_sample_path(out_root: Path, experiment_name: str, policy: str) -> Path:
    return out_root / experiment_name / f"eval_{policy}" / "per_sample_results.csv"


def _discover_groups(out_root: Path) -> list[str]:
    names = []
    for group_dir in sorted(out_root.glob("expanded_*")):
        gid = group_dir.name.replace("expanded_", "")
        if not gid.isdigit():
            continue
        if all(
            _adapter_summary_path(out_root, group_dir.name, policy).exists()
            and _eval_metrics_path(out_root, group_dir.name, policy).exists()
            for policy in POLICIES
        ):
            names.append(group_dir.name)
    return sorted(names, key=lambda name: int(name.replace("expanded_", "")))


def _percent_true(rows: list[dict]) -> float | None:
    if not rows:
        return None
    return 100.0 * sum(1 for row in rows if str(row["verifies_as_intended"]).lower() == "true") / len(rows)


def _target_retention_metrics(out_root: Path, experiment_name: str, policy: str, target_identity: int) -> dict:
    path = _per_sample_path(out_root, experiment_name, policy)
    if not path.exists():
        return {
            "RA_target": None,
            "RA_hard_nontarget": None,
            "RA_target_n": 0,
            "RA_hard_nontarget_n": 0,
        }

    with path.open("r", encoding="utf-8", newline="") as handle:
        samples = list(csv.DictReader(handle))

    hard_retain = [
        row
        for row in samples
        if row["role"] == "retain" and row["retain_group"] == "hard"
    ]
    target_rows = [row for row in hard_retain if int(row["intended_identity"]) == target_identity]
    nontarget_rows = [row for row in hard_retain if int(row["intended_identity"]) != target_identity]
    return {
        "RA_target": _percent_true(target_rows),
        "RA_hard_nontarget": _percent_true(nontarget_rows),
        "RA_target_n": len(target_rows),
        "RA_hard_nontarget_n": len(nontarget_rows),
    }


def build_group_features(out_root: Path) -> list[dict]:
    experiment_names = _discover_groups(out_root)
    if not experiment_names:
        raise FileNotFoundError(f"No expanded_* groups with all four policies found under {out_root}")

    rows = []
    for experiment_name in experiment_names:
        row = {
            "experiment_name": experiment_name,
            "forget_id": int(experiment_name.replace("expanded_", "")),
        }
        for policy in POLICIES:
            adapter = _load_json(_adapter_summary_path(out_root, experiment_name, policy))
            metrics = _load_json(_eval_metrics_path(out_root, experiment_name, policy))
            target_identity = int(adapter["target_identity"])
            target_retention = _target_retention_metrics(out_root, experiment_name, policy, target_identity)
            row[f"{policy}_target_identity"] = target_identity
            row[f"{policy}_target_similarity"] = adapter["target_similarity_to_forget"]
            row[f"{policy}_FA"] = metrics["FA"]
            row[f"{policy}_RA"] = metrics["RA"]
            row[f"{policy}_ERB"] = metrics["ERB"]
            row[f"{policy}_IdentityLeak8"] = metrics["IdentityLeak@8"]
            row[f"{policy}_RA_hard"] = metrics.get("RA_hard")
            row[f"{policy}_RA_random"] = metrics.get("RA_random")
            row[f"{policy}_RA_target"] = target_retention["RA_target"]
            row[f"{policy}_RA_hard_nontarget"] = target_retention["RA_hard_nontarget"]
            row[f"{policy}_RA_target_n"] = target_retention["RA_target_n"]
            row[f"{policy}_RA_hard_nontarget_n"] = target_retention["RA_hard_nontarget_n"]
            row[f"{policy}_hard_neighbor_retention_gap"] = metrics.get("hard_neighbor_retention_gap")
            row[f"{policy}_evaluated"] = metrics.get("evaluated_sample_count")
            row[f"{policy}_dropped"] = metrics.get("dropped_no_face_count")
        row["target_similarity_gap"] = row["nearest_hard_target_similarity"] - row["least_sim_hard_target_similarity"]
        rows.append(row)
    return rows


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def build_analysis(threshold_path: Path, group_rows: list[dict]) -> dict:
    from scipy import stats

    threshold = _load_json(threshold_path)["threshold"]

    per_policy = {}
    pooled_sims: list[float] = []
    pooled_fas: list[float] = []
    fa_matrix: list[list[float]] = []
    for policy in POLICIES:
        sims = [row[f"{policy}_target_similarity"] for row in group_rows]
        fas = [row[f"{policy}_FA"] for row in group_rows]
        ras = [row[f"{policy}_RA"] for row in group_rows]
        erbs = [row[f"{policy}_ERB"] for row in group_rows]
        leaks = [row[f"{policy}_IdentityLeak8"] for row in group_rows]
        ra_hards = [row[f"{policy}_RA_hard"] for row in group_rows if row[f"{policy}_RA_hard"] is not None]
        ra_randoms = [row[f"{policy}_RA_random"] for row in group_rows if row[f"{policy}_RA_random"] is not None]
        ra_targets = [row[f"{policy}_RA_target"] for row in group_rows if row[f"{policy}_RA_target"] is not None]
        ra_hard_nontargets = [
            row[f"{policy}_RA_hard_nontarget"]
            for row in group_rows
            if row[f"{policy}_RA_hard_nontarget"] is not None
        ]
        per_policy[policy] = {
            "target_similarity_mean": _mean(sims),
            "target_similarity_min": min(sims),
            "target_similarity_max": max(sims),
            "fa_mean": _mean(fas),
            "fa_min": min(fas),
            "fa_max": max(fas),
            "fa_zero_count": sum(1 for fa in fas if fa == 0.0),
            "ra_mean": _mean(ras),
            "ra_min": min(ras),
            "ra_max": max(ras),
            "ra_ge_80_count": sum(1 for ra in ras if ra >= 80.0),
            "erb_mean": _mean(erbs),
            "erb_min": min(erbs),
            "erb_max": max(erbs),
            "erb_ge_80_count": sum(1 for erb in erbs if erb >= 80.0),
            "identity_leak8_mean": _mean(leaks),
            "identity_leak8_min": min(leaks),
            "identity_leak8_max": max(leaks),
            "ra_hard_mean": _mean(ra_hards) if ra_hards else None,
            "ra_random_mean": _mean(ra_randoms) if ra_randoms else None,
            "ra_target_mean": _mean(ra_targets) if ra_targets else None,
            "ra_hard_nontarget_mean": _mean(ra_hard_nontargets) if ra_hard_nontargets else None,
        }
        pooled_sims.extend(sims)
        pooled_fas.extend(fas)
        fa_matrix.append(fas)

    pearson = stats.pearsonr(pooled_sims, pooled_fas)
    spearman = stats.spearmanr(pooled_sims, pooled_fas)
    friedman = stats.friedmanchisquare(*fa_matrix)

    nearest_fa = fa_matrix[POLICIES.index("nearest_hard")]
    least_fa = fa_matrix[POLICIES.index("least_sim_hard")]
    wilcoxon = stats.wilcoxon(nearest_fa, least_fa)

    nearest_zero = [fa == 0.0 for fa in nearest_fa]
    least_zero = [fa == 0.0 for fa in least_fa]
    b = sum(1 for nz, lz in zip(nearest_zero, least_zero) if (not nz) and lz)
    c = sum(1 for nz, lz in zip(nearest_zero, least_zero) if nz and (not lz))
    binom_p = stats.binomtest(min(b, c), b + c, 0.5).pvalue if (b + c) > 0 else None

    least = per_policy["least_sim_hard"]
    least["hard_gap_mean"] = (
        least["ra_hard_mean"] - least["ra_random_mean"]
        if least["ra_hard_mean"] is not None and least["ra_random_mean"] is not None
        else None
    )

    return {
        "threshold": threshold,
        "group_count": len(group_rows),
        "policies": POLICIES,
        "per_policy": per_policy,
        "pooled_dose_response": {
            "n": len(pooled_sims),
            "pearson_r": pearson.statistic,
            "pearson_p": pearson.pvalue,
            "spearman_rho": spearman.statistic,
            "spearman_p": spearman.pvalue,
        },
        "paired_significance_nearest_vs_least_sim": {
            "friedman_chi2": friedman.statistic,
            "friedman_p": friedman.pvalue,
            "wilcoxon_statistic": wilcoxon.statistic,
            "wilcoxon_p": wilcoxon.pvalue,
            "discordant_nearest_fail_least_success": b,
            "discordant_nearest_success_least_fail": c,
            "exact_binomial_p": binom_p,
        },
    }


def build_long_format_rows(group_rows: list[dict]) -> list[dict]:
    rows = []
    for policy in POLICIES:
        for group in group_rows:
            rows.append(
                {
                    "policy": policy,
                    "experiment_name": group["experiment_name"],
                    "forget_id": group["forget_id"],
                    "target_identity": group[f"{policy}_target_identity"],
                    "target_similarity_to_forget": group[f"{policy}_target_similarity"],
                    "FA": group[f"{policy}_FA"],
                    "RA": group[f"{policy}_RA"],
                    "ERB": group[f"{policy}_ERB"],
                    "IdentityLeak@8": group[f"{policy}_IdentityLeak8"],
                    "RA_hard": group[f"{policy}_RA_hard"],
                    "RA_random": group[f"{policy}_RA_random"],
                    "RA_target": group[f"{policy}_RA_target"],
                    "RA_hard_nontarget": group[f"{policy}_RA_hard_nontarget"],
                    "RA_target_n": group[f"{policy}_RA_target_n"],
                    "RA_hard_nontarget_n": group[f"{policy}_RA_hard_nontarget_n"],
                    "evaluated": group[f"{policy}_evaluated"],
                    "dropped": group[f"{policy}_dropped"],
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--threshold", type=Path, default=ROOT / "outputs" / "prep" / "threshold_arcface.json")
    parser.add_argument("--summary-dir", type=Path, default=ROOT / "outputs" / "expanded_summary")
    args = parser.parse_args()

    group_rows = build_group_features(args.out_root)
    analysis = build_analysis(args.threshold, group_rows)

    args.summary_dir.mkdir(parents=True, exist_ok=True)
    features_path = args.summary_dir / "expanded_mechanistic_group_features.csv"
    with features_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(group_rows[0].keys()))
        writer.writeheader()
        writer.writerows(group_rows)

    analysis_path = args.summary_dir / "expanded_mechanistic_analysis.json"
    with analysis_path.open("w", encoding="utf-8") as handle:
        json.dump(analysis, handle, indent=2)
        handle.write("\n")

    policy_summary = {
        "group_count": len(group_rows),
        "policies": {
            policy: {
                "FA_mean": analysis["per_policy"][policy]["fa_mean"],
                "FA_min": analysis["per_policy"][policy]["fa_min"],
                "FA_max": analysis["per_policy"][policy]["fa_max"],
                "RA_mean": analysis["per_policy"][policy]["ra_mean"],
                "RA_min": analysis["per_policy"][policy]["ra_min"],
                "RA_max": analysis["per_policy"][policy]["ra_max"],
                "ERB_mean": analysis["per_policy"][policy]["erb_mean"],
                "ERB_min": analysis["per_policy"][policy]["erb_min"],
                "ERB_max": analysis["per_policy"][policy]["erb_max"],
                "IdentityLeak@8_mean": analysis["per_policy"][policy]["identity_leak8_mean"],
                "IdentityLeak@8_min": analysis["per_policy"][policy]["identity_leak8_min"],
                "IdentityLeak@8_max": analysis["per_policy"][policy]["identity_leak8_max"],
                "FA_zero_count": analysis["per_policy"][policy]["fa_zero_count"],
                "ERB_ge_80_count": analysis["per_policy"][policy]["erb_ge_80_count"],
                "RA_ge_80_count": analysis["per_policy"][policy]["ra_ge_80_count"],
                "target_similarity_mean": analysis["per_policy"][policy]["target_similarity_mean"],
                "ra_hard_mean": analysis["per_policy"][policy]["ra_hard_mean"],
                "ra_random_mean": analysis["per_policy"][policy]["ra_random_mean"],
                "ra_target_mean": analysis["per_policy"][policy]["ra_target_mean"],
                "ra_hard_nontarget_mean": analysis["per_policy"][policy]["ra_hard_nontarget_mean"],
            }
            for policy in POLICIES
        },
        "pooled_dose_response": analysis["pooled_dose_response"],
        "paired_significance_nearest_vs_least_sim": analysis["paired_significance_nearest_vs_least_sim"],
    }
    with (args.summary_dir / "expanded_policy_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(policy_summary, handle, indent=2)
        handle.write("\n")

    long_rows = build_long_format_rows(group_rows)
    group_metrics_path = args.summary_dir / "expanded_group_metrics.csv"
    with group_metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(long_rows[0].keys()))
        writer.writeheader()
        writer.writerows(long_rows)

    print(f"Groups: {analysis['group_count']}")
    for policy in POLICIES:
        p = analysis["per_policy"][policy]
        print(
            f"  {POLICY_LABELS[policy]:15s} FA0={p['fa_zero_count']:2d}/{analysis['group_count']} "
            f"meanFA={p['fa_mean']:6.2f} meanRA={p['ra_mean']:6.2f} "
            f"RA_hard={p['ra_hard_mean']} RA_target={p['ra_target_mean']} RA_rand={p['ra_random_mean']} "
            f"sim={p['target_similarity_mean']:.4f}"
        )
    d = analysis["pooled_dose_response"]
    print(f"  pooled n={d['n']} Pearson r={d['pearson_r']:.4f} (p={d['pearson_p']:.2e}) Spearman rho={d['spearman_rho']:.4f}")
    print(f"Wrote {features_path}")
    print(f"Wrote {analysis_path}")
    print(f"Wrote {args.summary_dir / 'expanded_policy_summary.json'}")
    print(f"Wrote {group_metrics_path}")


if __name__ == "__main__":
    main()
