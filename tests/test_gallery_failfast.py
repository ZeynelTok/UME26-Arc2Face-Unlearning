from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from genmu_face_unlearning.benchmark import group_reference_map
from genmu_face_unlearning.evaluation import _build_gallery


class FakeEmbedder:
    def __init__(self, missing: str | None = None) -> None:
        self.missing = missing

    def detect_and_embed_image(self, path: str):
        if path == self.missing:
            return None
        value = float(int(path[-1]) + 1)
        return SimpleNamespace(embedding=np.asarray([value, 1.0], dtype=np.float32))


class GalleryFailFastTests(unittest.TestCase):
    def test_missing_explicit_evaluation_split_is_rejected(self) -> None:
        group = {"fit_reference_images": {"10": ["a0", "a1", "a2"]}}
        with self.assertRaisesRegex(KeyError, "held-out evaluation references"):
            group_reference_map(group, split="eval")

    def test_gallery_requires_all_references_and_normalizes_centroids(self) -> None:
        references = {"10": ["a0", "a1", "a2"], "20": ["b0", "b1", "b2"]}
        gallery, provenance = _build_gallery(FakeEmbedder(), references)
        self.assertEqual(set(gallery), {10, 20})
        self.assertEqual(len(provenance), 6)
        self.assertTrue(all(row["detector_success"] for row in provenance))
        self.assertTrue(
            all(np.isclose(np.linalg.norm(centroid), 1.0) for centroid in gallery.values())
        )

    def test_gallery_fails_on_missing_detection(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Unable to detect required gallery reference"):
            _build_gallery(FakeEmbedder(missing="a1"), {"10": ["a0", "a1", "a2"]})

    def test_gallery_fails_on_wrong_reference_count(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "has 2 references; expected 3"):
            _build_gallery(FakeEmbedder(), {"10": ["a0", "a1"]})
