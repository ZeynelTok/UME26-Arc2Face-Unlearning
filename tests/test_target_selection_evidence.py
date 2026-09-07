from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

from scripts.sweeps.run_expanded_target_selection_sweep import (
    _has_complete_candidate_evidence,
    _load_existing_selections,
    _selection_from_adapter_summary,
)


class TargetSelectionEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parent / "_tmp_target_selection_evidence"
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir()
        self.complete = {
            "experiment_name": "expanded_1",
            "forget_id": 1,
            "policy": "least-sim-hard",
            "target_identity": 4,
            "target_similarity_to_forget": 0.2,
            "candidate_similarities": [
                {"identity": 2, "similarity": 0.8},
                {"identity": 3, "similarity": 0.5},
                {"identity": 4, "similarity": 0.2},
            ],
        }

    def tearDown(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)

    def test_complete_candidate_evidence_is_required(self) -> None:
        self.assertTrue(_has_complete_candidate_evidence(self.complete))
        incomplete = {**self.complete, "candidate_similarities": []}
        self.assertFalse(_has_complete_candidate_evidence(incomplete))
        inconsistent = {**self.complete, "target_identity": 3}
        self.assertFalse(_has_complete_candidate_evidence(inconsistent))

    def test_incomplete_frozen_summary_is_not_reused(self) -> None:
        path = self.root / "summary.json"
        path.write_text(
            json.dumps(
                {
                    "policy": "least-sim-hard",
                    "rows": [
                        self.complete,
                        {
                            **self.complete,
                            "experiment_name": "expanded_2",
                            "candidate_similarities": [],
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        loaded = _load_existing_selections(path, "least-sim-hard")
        self.assertEqual(set(loaded), {"expanded_1"})

    def test_adapter_summary_without_candidates_forces_reconstruction(self) -> None:
        (self.root / "projection_adapter_summary.json").write_text(
            json.dumps(
                {
                    "target_identity": 4,
                    "target_similarity_to_forget": 0.2,
                }
            ),
            encoding="utf-8",
        )
        selection = _selection_from_adapter_summary(
            self.root,
            {"forget_id": 1},
            "least-sim-hard",
        )
        self.assertIsNone(selection)


if __name__ == "__main__":
    unittest.main()
