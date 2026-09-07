from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from genmu_face_unlearning.benchmark import build_benchmark


class PublicIdentityFilteringTests(unittest.TestCase):
    def test_public_neighbor_is_removed_before_reranking(self) -> None:
        metadata_rows = []
        for identity in range(1, 11):
            for index in range(6):
                metadata_rows.append(
                    {
                        "identity": identity,
                        "partition": 0,
                        "image_name": f"{identity:02d}_{index}.jpg",
                        "image_path": f"Data/{identity:02d}_{index}.jpg",
                    }
                )
        metadata = pd.DataFrame(metadata_rows)
        centroids = pd.DataFrame(
            {
                "identity": list(range(1, 11)),
                "num_images": [6] * 10,
                "partition": [0] * 10,
                "centroid": [np.eye(10, dtype=np.float32)[index] for index in range(10)],
            }
        )
        neighbor_rows = []
        non_public = list(range(1, 9))
        for target in non_public:
            ordered = [identity for identity in non_public if identity != target]
            if target == 1:
                ordered = [9, 2, 3, 4, *[identity for identity in ordered if identity not in {2, 3, 4}]]
            for rank, retain in enumerate(ordered, start=1):
                neighbor_rows.append(
                    {
                        "target_identity": target,
                        "retain_identity": retain,
                        "target_num_images": 6,
                        "retain_num_images": 6,
                        "cosine_distance": (0.01 * rank if target == 1 else 0.5 + 0.01 * rank),
                        "rank": rank,
                    }
                )
        neighbors = pd.DataFrame(neighbor_rows)
        benchmark = build_benchmark(
            metadata=metadata,
            centroids=centroids,
            neighbors=neighbors,
            min_images_per_identity=6,
            public_face_groups=[
                {
                    "set_name": "public",
                    "forget_id": 9,
                    "retain_ids": [10],
                }
            ],
            expanded_forget_count=1,
            hard_retain_count=3,
            fit_reference_count=3,
            eval_reference_count=3,
        )
        group = benchmark["expanded_groups"][0]
        self.assertEqual(group["forget_id"], 1)
        self.assertEqual(group["hard_retain_ids"], [2, 3, 4])
        self.assertEqual(len(group["random_retain_ids"]), 3)
        self.assertFalse({9, 10}.intersection(group["random_retain_ids"]))
        self.assertFalse(set(group["hard_retain_ids"]).intersection(group["random_retain_ids"]))


if __name__ == "__main__":
    unittest.main()
