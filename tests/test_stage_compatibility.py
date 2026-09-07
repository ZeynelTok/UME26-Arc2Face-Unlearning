from __future__ import annotations

import csv
import json
import shutil
import unittest
from pathlib import Path

import numpy as np

from genmu_face_unlearning.generation_plan import expected_generation_keys
from genmu_face_unlearning.config import normalize_experiment_config
from scripts.sweeps.run_expanded_target_selection_sweep import _run_is_compatible


class StageCompatibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parent / "_tmp_stage_compatibility"
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir()
        self.config_path = self.root / "config.json"
        self.benchmark_path = self.root / "benchmark.json"
        self.threshold_path = self.root / "threshold.json"
        self.metadata_path = self.root / "metadata.csv"
        self.embeddings_path = self.root / "embeddings.npy"
        self.adapter_dir = self.root / "adapter"
        self.generation_dir = self.root / "generation"
        self.eval_dir = self.root / "evaluation"
        for directory in (self.adapter_dir, self.generation_dir, self.eval_dir):
            directory.mkdir()

        self.config = normalize_experiment_config(
            {
                "experiment_name": "expanded_1",
                "forget_id": 1,
                "retain_ids": [2, 3, 4],
                "hard_retain_ids": [2, 3, 4],
                "random_retain_ids": [5, 6, 7],
            }
        )
        self.config_path.write_text(
            json.dumps(
                {
                    "experiment_name": "expanded_1",
                    "forget_id": 1,
                    "retain_ids": [2, 3, 4],
                    "hard_retain_ids": [2, 3, 4],
                    "random_retain_ids": [5, 6, 7],
                }
            ),
            encoding="utf-8",
        )
        self.benchmark_path.write_text('{"version":1}', encoding="utf-8")
        self.threshold_path.write_text('{"threshold":0.1}', encoding="utf-8")
        self.metadata_path.write_text("identity,image_path\n", encoding="utf-8")
        self.embeddings_path.write_bytes(b"embedding fixture")
        self.group = {
            "forget_id": 1,
            "fit_reference_images": {
                str(identity): [f"Data/{identity}_{index}.jpg" for index in range(3)]
                for identity in range(1, 8)
            },
            "eval_reference_images": {
                str(identity): [f"Data/{identity}_eval_{index}.jpg" for index in range(3)]
                for identity in range(1, 8)
            },
        }
        with (self.root / "gallery_reference_detections.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["identity", "reference_image", "detector_success"],
            )
            writer.writeheader()
            for identity, paths in self.group["eval_reference_images"].items():
                for path in paths:
                    writer.writerow(
                        {
                            "identity": identity,
                            "reference_image": str(Path(path).resolve()),
                            "detector_success": True,
                        }
                    )
            for index in range(459):
                writer.writerow(
                    {
                        "identity": 1000 + index,
                        "reference_image": str(Path(f"Data/filler_{index}.jpg").resolve()),
                        "detector_success": True,
                    }
                )
        self.expected_keys = expected_generation_keys(self.config, self.group)
        self.assertEqual(len(self.expected_keys), 168)

        adapter_path = self.adapter_dir / "projection_adapter.npz"
        self.write_adapter(target_identity=2)
        summary = {
            "checkpoint_selection_timing": "pre_update_state_scored_and_saved_before_optimizer_step",
            "forget_id": 1,
            "target_identity": 2,
            "reference_det_thresh": 0.5,
            "rank": 8,
        }
        (self.adapter_dir / "projection_adapter_summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )

        images = []
        for key in self.expected_keys:
            image_path = self.generation_dir / str(key["relative_image_path"])
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(b"png")
            images.append(
                {
                    "role": key["role"],
                    "identity": key["identity"],
                    "reference_image": f"Data/{key['reference_stem']}.jpg",
                    "seed": key["seed"],
                    "image_path": str(image_path),
                }
            )
        manifest = {
            "forget_id": 1,
            "target_identity": 2,
            "rank": 8,
            "adapter_path": str(adapter_path),
            "reference_det_thresh": 0.5,
            "retain_ids": self.config["retain_ids"],
            "hard_retain_ids": self.config["hard_retain_ids"],
            "random_retain_ids": self.config["random_retain_ids"],
            "eval_seeds": self.config["eval_seeds"],
            "generator": self.config["generator"],
            "generation": self.config["generation"],
            "expected_generation_key_count": 168,
            "images": images,
        }
        manifest_path = self.generation_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        predictions_path = self.eval_dir / "per_sample_results.csv"
        with predictions_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["role", "intended_identity", "reference_image", "seed"],
            )
            writer.writeheader()
            for key in self.expected_keys:
                writer.writerow(
                    {
                        "role": key["role"],
                        "intended_identity": key["identity"],
                        "reference_image": key["reference_stem"],
                        "seed": key["seed"],
                    }
                )
        metrics = {
            "det_thresh": 0.1,
            "threshold": 0.1,
            "attempted_sample_count": 168,
            "evaluated_sample_count": 168,
            "expected_generation_key_count": 168,
        }
        (self.eval_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    def tearDown(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)

    def write_adapter(self, target_identity: int) -> None:
        np.savez(
            self.adapter_dir / "projection_adapter.npz",
            forget_centroid=np.ones(512, dtype=np.float32),
            target_embedding=np.ones(512, dtype=np.float32),
            down=np.zeros((8, 512), dtype=np.float32),
            up=np.zeros((512, 8), dtype=np.float32),
            rank=np.array(8, dtype=np.int64),
            target_identity=np.array(target_identity, dtype=np.int64),
        )

    def compatible(self) -> bool:
        return _run_is_compatible(
            adapter_dir=self.adapter_dir,
            generation_dir=self.generation_dir,
            eval_dir=self.eval_dir,
            config=self.config,
            benchmark_path=self.benchmark_path,
            threshold_path=self.threshold_path,
            group=self.group,
            expected_target_identity=2,
            reference_det_thresh=0.5,
            evaluation_det_thresh=0.1,
        )

    def test_exact_bound_run_is_reused(self) -> None:
        self.assertTrue(self.compatible())

    def test_changed_threshold_is_rejected(self) -> None:
        self.threshold_path.write_text('{"threshold":0.2}', encoding="utf-8")
        self.assertFalse(self.compatible())

    def test_changed_adapter_is_rejected(self) -> None:
        self.write_adapter(target_identity=3)
        self.assertFalse(self.compatible())

    def test_missing_expected_png_is_rejected(self) -> None:
        first = self.expected_keys[0]["relative_image_path"]
        (self.generation_dir / str(first)).unlink()
        self.assertFalse(self.compatible())


if __name__ == "__main__":
    unittest.main()
