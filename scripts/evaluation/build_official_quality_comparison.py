"""Aggregate the SER-FIQ and GraFIQs quality results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

POLICIES = ["nearest_hard", "random_hard", "median_hard", "least_sim_hard"]
METRICS = {
    "ser_fiq_on_arcface": ("ser_fiq_normalized_score", True),
    "grafiqs_b2_on_arcface": ("grafiqs_block2", False),
}


def _probability_superiority(generated: np.ndarray, real: np.ndarray, higher_is_better: bool) -> float:
    """Probability that a random generated image has at least real-image utility."""

    if higher_is_better:
        statistic = stats.mannwhitneyu(generated, real, alternative="two-sided").statistic
    else:
        statistic = stats.mannwhitneyu(real, generated, alternative="two-sided").statistic
    return float(statistic / (len(generated) * len(real)))


def identity_matched_group_superiority(
    generated_scored: pd.DataFrame,
    generated_attempted: pd.DataFrame,
    real_scored: pd.DataFrame,
    column: str,
    higher_is_better: bool,
) -> pd.DataFrame:
    """Compute one identity-matched superiority probability per group.

    Every scored generated image is compared only with the real references of
    its intended identity. The conditional denominator contains those detected
    generated/real pairs. The end-to-end denominator additionally contains the
    corresponding pairs for generated no-face attempts, which contribute zero
    wins. Returning group-level values makes the 30 benchmark groups the equal-
    weight aggregation and inferential units.
    """

    required_generated = {"experiment_name", "intended_identity", column}
    required_real = {"real_identity", "image_key", column}
    if missing := required_generated.difference(generated_scored.columns):
        raise ValueError(f"Generated quality rows lack columns: {sorted(missing)}")
    if missing := required_generated.difference(generated_attempted.columns):
        raise ValueError(f"Generated attempt rows lack columns: {sorted(missing)}")
    if missing := required_real.difference(real_scored.columns):
        raise ValueError(f"Real quality rows lack columns: {sorted(missing)}")
    if real_scored["image_key"].duplicated().any():
        raise ValueError("Real-reference quality rows must be unique by image_key")

    real_by_identity = {
        int(identity): frame[column].to_numpy()
        for identity, frame in real_scored.groupby("real_identity", sort=True)
    }
    if not real_by_identity or any(len(values) == 0 for values in real_by_identity.values()):
        raise ValueError("Every real identity must have at least one scored reference")

    scored_groups = set(generated_scored["experiment_name"])
    attempted_groups = set(generated_attempted["experiment_name"])
    if scored_groups != attempted_groups:
        raise ValueError("Scored and attempted generated rows cover different groups")

    rows = []
    for experiment_name in sorted(scored_groups):
        scored_group = generated_scored.loc[
            generated_scored["experiment_name"] == experiment_name
        ]
        attempted_group = generated_attempted.loc[
            generated_attempted["experiment_name"] == experiment_name
        ]
        numerator = 0.0
        conditional_denominator = 0
        attempted_denominator = 0

        for identity, attempted_identity in attempted_group.groupby(
            "intended_identity", sort=True
        ):
            identity = int(identity)
            if identity not in real_by_identity:
                raise ValueError(f"No real references for intended identity {identity}")
            real_values = real_by_identity[identity]
            scored_values = scored_group.loc[
                scored_group["intended_identity"].astype(int) == identity, column
            ].to_numpy()
            if len(scored_values):
                numerator += (
                    stats.mannwhitneyu(scored_values, real_values, alternative="two-sided").statistic
                    if higher_is_better
                    else stats.mannwhitneyu(real_values, scored_values, alternative="two-sided").statistic
                )
            conditional_denominator += len(scored_values) * len(real_values)
            attempted_denominator += len(attempted_identity) * len(real_values)

        if conditional_denominator == 0 or attempted_denominator == 0:
            raise ValueError(f"Empty identity-matched comparison for {experiment_name}")
        rows.append(
            {
                "experiment_name": experiment_name,
                "matched_wins": float(numerator),
                "conditional_pair_count": int(conditional_denominator),
                "attempted_pair_count": int(attempted_denominator),
                "probability_superiority_conditional_on_detected": float(
                    numerator / conditional_denominator
                ),
                "probability_superiority_no_face_as_loss": float(
                    numerator / attempted_denominator
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("experiment_name").reset_index(drop=True)


def _real_group_means(real: pd.DataFrame, column: str) -> pd.Series:
    rows = []
    for record in real[["real_group_ids", column]].to_dict("records"):
        for group_id in json.loads(record["real_group_ids"]):
            rows.append({"experiment_name": f"expanded_{int(group_id)}", column: record[column]})
    return pd.DataFrame(rows).groupby("experiment_name", sort=True)[column].mean()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/quality_official_v1"))
    args = parser.parse_args()

    generated_path = args.input_dir / "generated_official_quality_per_image.csv"
    real_path = args.input_dir / "real_official_quality_per_image.csv"
    generated = pd.read_csv(generated_path)
    real = pd.read_csv(real_path)
    if len(generated) != 20_160 or len(real) != 960:
        raise RuntimeError(f"Unexpected attempted counts: generated={len(generated)}, real={len(real)}")
    if generated["image_key"].duplicated().any() or real["image_key"].duplicated().any():
        raise RuntimeError("Quality inputs must have one row per unique image key")
    bad_generated = generated.loc[~generated["status"].isin(["scored", "no_face"])]
    bad_real = real.loc[~real["status"].isin(["scored", "no_face"])]
    if len(bad_generated) or len(bad_real):
        raise RuntimeError(
            f"Unexpected scoring errors: generated={len(bad_generated)}, real={len(bad_real)}"
        )

    generated_attempted = generated.copy()
    real_attempted = real.copy()
    generated = generated.loc[generated["status"] == "scored"].copy()
    real = real.loc[real["status"] == "scored"].copy()
    real_counts = real.groupby("real_identity")["image_key"].nunique()
    if len(real_counts) != 160 or not (real_counts == 6).all():
        raise RuntimeError(
            "Identity-matched quality comparison requires exactly six unique "
            f"real references for each of 160 identities; observed {real_counts.value_counts().to_dict()}"
        )

    result = {
        "interpretation": (
            "The primary endpoint is the equal-weight mean of 30 group-level, identity-matched "
            "probabilities: each generated image is compared only with the six unique real "
            "references of its intended identity. The conditional value uses successfully "
            "aligned inputs; the conservative end-to-end value counts every generated no-face "
            "attempt as a loss. A value of 0.5 denotes no stochastic-ordering advantage, not "
            "equivalence or indistinguishability. The former globally pooled comparison is "
            "retained only as a sensitivity statistic."
        ),
        "primary_aggregation": {
            "matching": "generated intended_identity equals real_identity",
            "real_references_per_identity": 6,
            "within_group": "Mann-Whitney probability superiority over identity-matched pairs",
            "across_groups": "unweighted arithmetic mean of 30 group probabilities",
            "ties": "half a win",
            "gra_fiqs_direction": "lower raw B2 gradient is better",
        },
        "counts": {
            "generated_scored_total": int(len(generated)),
            "generated_attempted_total": int(len(generated_attempted)),
            "generated_no_face_total": int((generated_attempted["status"] == "no_face").sum()),
            "generated_per_policy": {
                policy: {
                    "attempted": int((generated_attempted["policy"] == policy).sum()),
                    "scored": int((generated["policy"] == policy).sum()),
                    "no_face": int(
                        (
                            (generated_attempted["policy"] == policy)
                            & (generated_attempted["status"] == "no_face")
                        ).sum()
                    ),
                }
                for policy in POLICIES
            },
            "real_unique_attempted": int(len(real_attempted)),
            "real_unique_scored": int(len(real)),
            "real_unique_no_face": int((real_attempted["status"] == "no_face").sum()),
        },
        "metrics": {},
    }

    for metric_name, (column, higher_is_better) in METRICS.items():
        generated[column] = pd.to_numeric(generated[column], errors="raise")
        real[column] = pd.to_numeric(real[column], errors="raise")
        real_values = real[column].to_numpy()
        real_by_group = _real_group_means(real, column)
        policy_group_means = generated.pivot_table(
            index="experiment_name",
            columns="policy",
            values=column,
            aggfunc="mean",
        )[POLICIES]
        raw_friedman = stats.friedmanchisquare(
            *(policy_group_means[policy] for policy in POLICIES)
        )
        matched_by_policy = {}
        for policy in POLICIES:
            matched_by_policy[policy] = identity_matched_group_superiority(
                generated.loc[generated["policy"] == policy],
                generated_attempted.loc[generated_attempted["policy"] == policy],
                real,
                column,
                higher_is_better,
            )
            if len(matched_by_policy[policy]) != 30:
                raise RuntimeError(
                    f"Expected 30 identity-matched group probabilities for {policy}, "
                    f"found {len(matched_by_policy[policy])}"
                )
        matched_friedman = stats.friedmanchisquare(
            *(
                matched_by_policy[policy][
                    "probability_superiority_conditional_on_detected"
                ]
                for policy in POLICIES
            )
        )
        metric_result = {
            "column": column,
            "direction": "higher_is_better" if higher_is_better else "lower_is_better",
            "real_mean": float(real_values.mean()),
            "real_std": float(real_values.std(ddof=1)),
            "friedman_across_policy_identity_matched_group_probabilities": {
                "statistic": float(matched_friedman.statistic),
                "p": float(matched_friedman.pvalue),
                "n_groups": int(len(policy_group_means)),
            },
            "global_pool_sensitivity": {
                "description": "Generated images compared with every unique real image, without identity matching",
            },
            "raw_group_mean_sensitivity": {
                "friedman_statistic": float(raw_friedman.statistic),
                "friedman_p": float(raw_friedman.pvalue),
                "n_groups": int(len(policy_group_means)),
            },
            "per_policy": {},
        }
        for policy in POLICIES:
            subset = generated.loc[generated["policy"] == policy]
            values = subset[column].to_numpy()
            policy_attempted = generated_attempted.loc[generated_attempted["policy"] == policy]
            generated_by_group = subset.groupby("experiment_name", sort=True)[column].mean()
            paired = pd.concat(
                [real_by_group.rename("real"), generated_by_group.rename("generated")],
                axis=1,
                join="inner",
            ).dropna()
            matched = matched_by_policy[policy]
            matched_conditional = matched[
                "probability_superiority_conditional_on_detected"
            ]
            matched_no_face = matched["probability_superiority_no_face_as_loss"]
            matched_vs_half = stats.wilcoxon(
                matched_conditional - 0.5, alternative="two-sided"
            )
            global_conditional = _probability_superiority(
                values, real_values, higher_is_better
            )
            metric_result["per_policy"][policy] = {
                "n_attempted": int(len(policy_attempted)),
                "n_images": int(len(values)),
                "n_no_face": int((policy_attempted["status"] == "no_face").sum()),
                "generated_mean": float(values.mean()),
                "generated_std": float(values.std(ddof=1)),
                "probability_superiority_conditional_on_detected": float(
                    matched_conditional.mean()
                ),
                "probability_superiority_no_face_as_loss": float(
                    matched_no_face.mean()
                ),
                "identity_matched_group_probability_std": float(
                    matched_conditional.std(ddof=1)
                ),
                "identity_matched_group_pair_count": int(
                    matched["conditional_pair_count"].sum()
                ),
                "wilcoxon_identity_matched_group_probability_vs_half": {
                    "statistic": float(matched_vs_half.statistic),
                    "p": float(matched_vs_half.pvalue),
                    "n_groups": int(len(matched)),
                },
                "global_pool_sensitivity": {
                    "probability_superiority_conditional_on_detected": global_conditional,
                    "probability_superiority_no_face_as_loss": float(
                        global_conditional * len(values) / len(policy_attempted)
                    ),
                },
                "raw_group_mean_sensitivity": {
                    "paired_wilcoxon_statistic": float(
                        stats.wilcoxon(
                            paired["real"], paired["generated"], alternative="two-sided"
                        ).statistic
                    ),
                    "paired_wilcoxon_p": float(
                        stats.wilcoxon(
                            paired["real"], paired["generated"], alternative="two-sided"
                        ).pvalue
                    ),
                    "n_groups": int(len(paired)),
                },
            }
        result["metrics"][metric_name] = metric_result

    output_path = args.input_dir / "comparison.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
