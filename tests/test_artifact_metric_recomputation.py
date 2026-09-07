from __future__ import annotations

import unittest

import pandas as pd

from genmu_face_unlearning.evaluation import _aggregate_metrics


class IdentityMetricTests(unittest.TestCase):
    def test_known_fa_ra_erb_and_leakage(self) -> None:
        # One of four forget images matches; three of four retain images match.
        # Only reference a leaks, and only on its second seed.
        samples = pd.DataFrame(
            [
                ("forget", 1, "a", 1, False, False, 0.1, "forget"),
                ("forget", 1, "a", 2, True,  True,  0.4, "forget"),
                ("forget", 1, "b", 1, False, False, 0.2, "forget"),
                ("forget", 1, "b", 2, False, False, 0.3, "forget"),
                ("retain", 2, "c", 1, False, True,  0.0, "hard"),
                ("retain", 2, "c", 2, False, False, 0.0, "hard"),
                ("retain", 3, "d", 1, False, True,  0.0, "random"),
                ("retain", 3, "d", 2, False, True,  0.0, "random"),
            ],
            columns=[
                "role", "intended_identity", "reference_image", "seed",
                "verifies_as_target", "verifies_as_intended", "target_similarity", "retain_group",
            ],
        )
        metrics = _aggregate_metrics(samples)
        expected = {
            "FA": 25.0,
            "EA": 75.0,
            "RA": 75.0,
            "ERB": 75.0,
            "RA_hard": 50.0,
            "RA_random": 100.0,
            "hard_neighbor_retention_gap": -50.0,
            "IdentityLeak@1": 0.0,
            "IdentityLeak@4": 50.0,
            "IdentityLeak@8": 50.0,
            "mean_target_similarity": 0.25,
            "max_target_similarity": 0.4,
        }
        for name, value in expected.items():
            with self.subTest(metric=name):
                self.assertAlmostEqual(metrics[name], value)

        # Complete forgetting failure must zero-floor ERB even with good retention.
        samples.loc[samples["role"] == "forget", "verifies_as_target"] = True
        self.assertEqual(_aggregate_metrics(samples)["ERB"], 0.0)


if __name__ == "__main__":
    unittest.main()
