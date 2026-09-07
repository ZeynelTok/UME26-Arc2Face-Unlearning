from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd

from genmu_face_unlearning.official_quality import (
    deterministic_image_seed,
    ser_fiq_scores_from_embeddings,
)
from scripts.evaluation.build_official_quality_comparison import (
    _probability_superiority,
    identity_matched_group_superiority,
)


class SerFiqFormulaTests(unittest.TestCase):
    def test_identical_embeddings_have_zero_distance(self) -> None:
        embeddings = np.repeat(np.array([[3.0, 4.0]], dtype=np.float64), 3, axis=0)
        scores = ser_fiq_scores_from_embeddings(embeddings)
        self.assertEqual(scores["ser_fiq_mean_euclidean_distance"], 0.0)
        self.assertEqual(scores["ser_fiq_paper_score"], 1.0)
        expected = 1.0 / (1.0 + math.exp(-130.0 * (1.0 - 0.88)))
        self.assertAlmostEqual(scores["ser_fiq_normalized_score"], expected, places=15)

    def test_two_orthogonal_embeddings_have_known_score(self) -> None:
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        scores = ser_fiq_scores_from_embeddings(embeddings)
        expected_distance = math.sqrt(2.0)
        expected_score = 2.0 / (1.0 + math.exp(expected_distance))
        self.assertAlmostEqual(scores["ser_fiq_mean_euclidean_distance"], expected_distance, places=15)
        self.assertAlmostEqual(scores["ser_fiq_paper_score"], expected_score, places=15)

    def test_seed_is_stable_and_key_specific(self) -> None:
        first = deterministic_image_seed("abc", base_seed=7)
        self.assertEqual(first, deterministic_image_seed("abc", base_seed=7))
        self.assertNotEqual(first, deterministic_image_seed("abd", base_seed=7))

    def test_zero_embedding_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ser_fiq_scores_from_embeddings(np.array([[0.0, 0.0], [1.0, 0.0]]))


class ProbabilitySuperiorityTests(unittest.TestCase):
    def test_higher_is_better_direction(self) -> None:
        value = _probability_superiority(np.array([2.0]), np.array([1.0]), True)
        self.assertEqual(value, 1.0)

    def test_lower_is_better_direction(self) -> None:
        value = _probability_superiority(np.array([1.0]), np.array([2.0]), False)
        self.assertEqual(value, 1.0)

    def test_tie_is_half(self) -> None:
        value = _probability_superiority(np.array([1.0]), np.array([1.0]), True)
        self.assertEqual(value, 0.5)

    def test_identity_matching_excludes_other_identities(self) -> None:
        real = pd.DataFrame(
            {
                "image_key": ["r1", "r2"],
                "real_identity": [1, 2],
                "score": [0.0, 100.0],
            }
        )
        attempted = pd.DataFrame(
            {
                "experiment_name": ["expanded_1", "expanded_1"],
                "intended_identity": [1, 2],
                "score": [1.0, 99.0],
            }
        )
        observed = identity_matched_group_superiority(
            attempted, attempted, real, "score", True
        )
        # Identity 1 wins and identity 2 loses. Cross-identity pairs never enter.
        self.assertEqual(
            observed.loc[0, "probability_superiority_conditional_on_detected"], 0.5
        )

    def test_groups_receive_equal_weight(self) -> None:
        real = pd.DataFrame(
            {"image_key": ["r1"], "real_identity": [1], "score": [0.0]}
        )
        scored = pd.DataFrame(
            {
                "experiment_name": ["expanded_1", "expanded_2", "expanded_2", "expanded_2"],
                "intended_identity": [1, 1, 1, 1],
                "score": [1.0, -1.0, -1.0, -1.0],
            }
        )
        observed = identity_matched_group_superiority(
            scored, scored, real, "score", True
        )
        self.assertEqual(observed["probability_superiority_conditional_on_detected"].mean(), 0.5)

    def test_no_face_attempt_is_a_loss_only_in_sensitivity(self) -> None:
        real = pd.DataFrame(
            {"image_key": ["r1"], "real_identity": [1], "score": [0.0]}
        )
        scored = pd.DataFrame(
            {
                "experiment_name": ["expanded_1"],
                "intended_identity": [1],
                "score": [1.0],
            }
        )
        attempted = pd.DataFrame(
            {
                "experiment_name": ["expanded_1", "expanded_1"],
                "intended_identity": [1, 1],
                "score": [1.0, np.nan],
            }
        )
        observed = identity_matched_group_superiority(
            scored, attempted, real, "score", True
        )
        self.assertEqual(
            observed.loc[0, "probability_superiority_conditional_on_detected"], 1.0
        )
        self.assertEqual(observed.loc[0, "probability_superiority_no_face_as_loss"], 0.5)


if __name__ == "__main__":
    unittest.main()
