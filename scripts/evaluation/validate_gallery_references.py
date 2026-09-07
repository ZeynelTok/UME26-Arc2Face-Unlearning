"""Validate every held-out gallery reference used by the 30-group evaluation.

The evaluation requires three successfully detected reference images for every
identity.  This preflight scores each unique reference once, preserves detection
provenance, and fails if the gallery would otherwise be silently incomplete.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from genmu_face_unlearning.arcface_utils import ArcFaceEmbedder  # noqa: E402


def expected_references(benchmark: dict) -> list[tuple[int, str]]:
    rows: set[tuple[int, str]] = set()
    groups = benchmark.get("expanded_groups", [])
    for group in groups:
        reference_map = group.get("eval_reference_images", {})
        expected_identities = {
            int(group["forget_id"]),
            *map(int, group.get("hard_retain_ids", group.get("retain_ids", []))),
            *map(int, group.get("random_retain_ids", [])),
        }
        if {int(identity) for identity in reference_map} != expected_identities:
            raise RuntimeError(
                f"Group {group['forget_id']} evaluation-reference identities are incomplete"
            )
        for identity_text, paths in reference_map.items():
            if len(paths) != 3:
                raise RuntimeError(
                    f"Identity {identity_text} has {len(paths)} evaluation references; expected 3"
                )
            rows.update((int(identity_text), str(Path(path).resolve())) for path in paths)
    return sorted(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=ROOT / "outputs/prep/benchmark.json")
    parser.add_argument("--model-root", type=Path, default=ROOT / "models/insightface")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--det-thresh", type=float, default=0.1)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs/prep/gallery_reference_detections.csv",
    )
    args = parser.parse_args()

    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8-sig"))
    references = expected_references(benchmark)
    if len(references) != 480 or len({identity for identity, _ in references}) != 160:
        raise RuntimeError(
            f"Expected 480 unique references from 160 identities, found "
            f"{len(references)} from {len({identity for identity, _ in references})}"
        )

    embedder = ArcFaceEmbedder(
        model_root=args.model_root,
        device=args.device,
        det_thresh=args.det_thresh,
        use_fallback=False,
    )
    rows = []
    for identity, image_path in references:
        result = embedder.detect_and_embed_image(image_path)
        bbox = None if result is None or result.bbox is None else result.bbox.tolist()
        rows.append(
            {
                "identity": identity,
                "reference_image": image_path,
                "detector_success": result is not None,
                "det_thresh": args.det_thresh,
                "bbox": json.dumps(bbox, separators=(",", ":")),
            }
        )

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    failures = frame.loc[~frame["detector_success"]]
    if not failures.empty:
        raise RuntimeError(
            f"{len(failures)} required gallery references failed detection; see {args.out}"
        )
    print(
        f"Validated {len(frame)} held-out gallery references from "
        f"{frame['identity'].nunique()} identities at det_thresh={args.det_thresh}"
    )
    print(args.out)


if __name__ == "__main__":
    main()
