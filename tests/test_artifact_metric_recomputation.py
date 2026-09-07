from __future__ import annotations

import unittest

import numpy as np
import pandas as pd


def _harmonic_mean(left: float, right: float) -> float:
    if left <= 0 or right <= 0:
        return 0.0
    return 2.0 * left * right / (left + right)


def derive_identity_metrics(frame: pd.DataFrame) -> dict[str, float]:
    """Independent known-answer calculation used only by this synthetic test."""
    forget = frame.loc[frame["role"] == "forget"]
    retain = frame.loc[frame["role"] == "retain"]
    hard = retain.loc[retain["retain_group"] == "hard"]
    random = retain.loc[retain["retain_group"] == "random"]
    fa = 100.0 * float(forget["verifies_as_target"].mean())
    ea = 100.0 - fa
    ra = 100.0 * float(retain["verifies_as_intended"].mean())
    result = {
        "FA": fa,
        "EA": ea,
        "RA": ra,
        "ERB": _harmonic_mean(ea, ra),
        "RA_hard": 100.0 * float(hard["verifies_as_intended"].mean()),
        "RA_random": 100.0 * float(random["verifies_as_intended"].mean()),
        "mean_target_similarity": float(forget["target_similarity"].mean()),
        "max_target_similarity": float(forget["target_similarity"].max()),
    }
    result["hard_neighbor_retention_gap"] = result["RA_hard"] - result["RA_random"]
    grouped = forget.groupby(["intended_identity", "reference_image"], sort=True)
    for k in (1, 4, 8):
        hits = [
            bool(group.sort_values("seed").head(k)["verifies_as_target"].any())
            for _, group in grouped
        ]
        result[f"IdentityLeak@{k}"] = 100.0 * float(np.mean(hits))
    return result


class IdentityMetricRecomputationTests(unittest.TestCase):
    def test_known_fa_ra_erb_and_leakage(self) -> None:
        rows = [
            {
                "role": "forget",
                "intended_identity": 1,
                "reference_image": "a",
                "seed": 1,
                "verifies_as_target": False,
                "verifies_as_intended": False,
                "target_similarity": 0.1,
                "retain_group": "forget",
            },
            {
                "role": "forget",
                "intended_identity": 1,
                "reference_image": "a",
                "seed": 2,
                "verifies_as_target": True,
                "verifies_as_intended": True,
                "target_similarity": 0.4,
                "retain_group": "forget",
            },
            {
                "role": "forget",
                "intended_identity": 1,
                "reference_image": "b",
                "seed": 1,
                "verifies_as_target": False,
                "verifies_as_intended": False,
                "target_similarity": 0.2,
                "retain_group": "forget",
            },
            {
                "role": "forget",
                "intended_identity": 1,
                "reference_image": "b",
                "seed": 2,
                "verifies_as_target": False,
                "verifies_as_intended": False,
                "target_similarity": 0.3,
                "retain_group": "forget",
            },
            {
                "role": "retain",
                "intended_identity": 2,
                "reference_image": "c",
                "seed": 1,
                "verifies_as_target": False,
                "verifies_as_intended": True,
                "target_similarity": 0.0,
                "retain_group": "hard",
            },
            {
                "role": "retain",
                "intended_identity": 2,
                "reference_image": "c",
                "seed": 2,
                "verifies_as_target": False,
                "verifies_as_intended": False,
                "target_similarity": 0.0,
                "retain_group": "hard",
            },
            {
                "role": "retain",
                "intended_identity": 3,
                "reference_image": "d",
                "seed": 1,
                "verifies_as_target": False,
                "verifies_as_intended": True,
                "target_similarity": 0.0,
                "retain_group": "random",
            },
            {
                "role": "retain",
                "intended_identity": 3,
                "reference_image": "d",
                "seed": 2,
                "verifies_as_target": False,
                "verifies_as_intended": True,
                "target_similarity": 0.0,
                "retain_group": "random",
            },
        ]
        metrics = derive_identity_metrics(pd.DataFrame(rows))
        self.assertEqual(metrics["FA"], 25.0)
        self.assertEqual(metrics["EA"], 75.0)
        self.assertEqual(metrics["RA"], 75.0)
        self.assertEqual(metrics["ERB"], 75.0)
        self.assertEqual(metrics["RA_hard"], 50.0)
        self.assertEqual(metrics["RA_random"], 100.0)
        self.assertEqual(metrics["hard_neighbor_retention_gap"], -50.0)
        self.assertEqual(metrics["IdentityLeak@1"], 0.0)
        self.assertEqual(metrics["IdentityLeak@4"], 50.0)
        self.assertEqual(metrics["IdentityLeak@8"], 50.0)
        self.assertAlmostEqual(metrics["mean_target_similarity"], 0.25)
        self.assertEqual(metrics["max_target_similarity"], 0.4)


if __name__ == "__main__":
    unittest.main()
