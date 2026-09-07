"""Recompute the three main paper tables from saved artifacts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation.build_official_quality_comparison import (  # noqa: E402
    identity_matched_group_superiority,
)


POLICIES = ("nearest_hard", "random_hard", "median_hard", "least_sim_hard")


def markdown_table(frame: pd.DataFrame) -> str:
    def format_value(value) -> str:
        if isinstance(value, float):
            return f"{value:.15g}"
        return str(value)

    header = [str(column) for column in frame.columns]
    rows = [[format_value(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    widths = [
        max(len(header[index]), *(len(row[index]) for row in rows))
        for index in range(len(header))
    ]
    render = lambda row: "| " + " | ".join(
        value.ljust(widths[index]) for index, value in enumerate(row)
    ) + " |"
    separator = ["-" * width for width in widths]
    return "\n".join([render(header), render(separator), *[render(row) for row in rows]])


def recompute(bundle: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(bundle / "run_metrics.csv")
    predictions = pd.read_csv(bundle / "per_sample_predictions.csv.gz")
    targets = pd.read_csv(bundle / "target_selections.csv")
    quality = pd.read_csv(bundle / "quality_scores.csv.gz", low_memory=False)
    recognizers = pd.read_csv(bundle / "recognizer_runs.csv")

    target_fields = targets[["run_key", "target_identity", "target_similarity_to_forget"]]
    metrics = metrics.merge(target_fields, on="run_key", validate="one_to_one")
    target_predictions = predictions.merge(
        target_fields[["run_key", "target_identity"]], on="run_key", validate="many_to_one"
    )
    target_predictions = target_predictions[
        (target_predictions["role"] == "retain")
        & (target_predictions["intended_identity"] == target_predictions["target_identity"])
    ]
    target_ra = (
        target_predictions.groupby("run_key")["verifies_as_intended"].mean().mul(100.0).rename("RA_target")
    )
    metrics = metrics.merge(target_ra, on="run_key", validate="one_to_one")

    table1_rows = []
    table2_rows = []
    real = quality[(quality["source"] == "real_reference") & (quality["status"] == "scored")]
    generated_attempted = quality[quality["source"] == "generated"]
    generated = generated_attempted[generated_attempted["status"] == "scored"]
    for policy in POLICIES:
        run = metrics[metrics["policy"] == policy]
        gen = generated[generated["policy"] == policy]
        attempted = generated_attempted[generated_attempted["policy"] == policy]
        ser_fiq_groups = identity_matched_group_superiority(
            gen, attempted, real, "ser_fiq_normalized_score", True
        )
        grafiqs_groups = identity_matched_group_superiority(
            gen, attempted, real, "grafiqs_block2", False
        )
        table1_rows.append(
            {
                "policy": policy.replace("_", "-"),
                "FA=0 groups": int((run["FA"] == 0).sum()),
                "Mean FA": run["FA"].mean(),
                "Mean RA": run["RA"].mean(),
                "Mean ERB": run["ERB"].mean(),
                "Mean Leak@8": run["IdentityLeak@8"].mean(),
                "Mean sim.": run["target_similarity_to_forget"].mean(),
            }
        )
        table2_rows.append(
            {
                "policy": policy.replace("_", "-"),
                "Mean RA": run["RA"].mean(),
                "RA_target": run["RA_target"].mean(),
                "RA_hard": run["RA_hard"].mean(),
                "RA_random": run["RA_random"].mean(),
                "SER-FIQ identity-matched P (%)": 100.0
                * ser_fiq_groups[
                    "probability_superiority_conditional_on_detected"
                ].mean(),
                "GraFIQs-B2 identity-matched P (%)": 100.0
                * grafiqs_groups[
                    "probability_superiority_conditional_on_detected"
                ].mean(),
            }
        )

    table3_rows = []
    for policy in POLICIES:
        row = {"policy": policy.replace("_", "-")}
        for recognizer in ("arcface", "adaface"):
            subset = recognizers[
                (recognizers["policy"] == policy) & (recognizers["recognizer"] == recognizer)
            ]
            row[f"{recognizer} FA=0 groups"] = int((subset["FA"] == 0).sum())
            row[f"{recognizer} Mean FA"] = subset["FA"].mean()
            row[f"{recognizer} Mean RA"] = subset["RA"].mean()
        table3_rows.append(row)
    return pd.DataFrame(table1_rows), pd.DataFrame(table2_rows), pd.DataFrame(table3_rows)


def _macro_component(value: object) -> str:
    words = re.findall(r"[A-Za-z]+|\d+", str(value).replace("=", " equals "))
    number_names = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "8": "Eight"}
    return "".join(number_names.get(word, word.title()) for word in words)


def latex_constants(tables: tuple[pd.DataFrame, ...]) -> str:
    """Render full-precision, artifact-derived values as reusable LaTeX macros."""
    lines = [
        "% Generated by scripts/reproducibility/recompute_paper_tables.py.",
        "% Do not edit values manually; rebuild from the verified paper_artifacts bundle.",
    ]
    for table_number, table in enumerate(tables, start=1):
        lines.append(f"% Main table {table_number}")
        for row in table.to_dict(orient="records"):
            policy = _macro_component(row["policy"])
            for column, value in row.items():
                if column == "policy":
                    continue
                if isinstance(value, float):
                    rendered = f"{value:.15g}"
                else:
                    rendered = str(value)
                macro = f"PaperTable{table_number}{policy}{_macro_component(column)}"
                lines.append(f"\\newcommand{{\\{macro}}}{{{rendered}}}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "paper_artifacts",
    )
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--latex-output", type=Path)
    args = parser.parse_args()
    tables = recompute(args.bundle.resolve())
    for number, table in enumerate(tables, start=1):
        print(f"\nTable {number} (full precision)\n")
        print(markdown_table(table))
    if args.json_output:
        payload = {
            f"table_{number}": table.to_dict(orient="records")
            for number, table in enumerate(tables, start=1)
        }
        args.json_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.latex_output:
        args.latex_output.write_text(latex_constants(tables), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
