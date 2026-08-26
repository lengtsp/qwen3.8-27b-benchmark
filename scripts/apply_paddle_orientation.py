#!/usr/bin/env python3
"""Apply only high-confidence Paddle document-orientation corrections.

Run PaddleOCR's ``doc_img_orientation_classification`` first.  Its JSON result
contains the *corrective* counter-clockwise angle (0, 90, 180, or 270).  This
script turns those classifications into upright PNGs plus an auditable manifest
for PP-DocLayoutV3.  Low-confidence results are deliberately not rotated.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


VALID_ANGLES = {0, 90, 180, 270}


@dataclass(frozen=True)
class OrientationResult:
    angle: int
    confidence: float
    margin: float
    result_path: Path | None


def page_paths(input_dir: Path) -> list[Path]:
    pages = sorted(input_dir.glob("page-*.png"))
    if not pages:
        raise FileNotFoundError(f"No page-*.png files under {input_dir}")
    return pages


def parse_approval(value: str) -> tuple[int, int]:
    try:
        page_text, angle_text = value.split("=", 1)
        page, angle = int(page_text), int(angle_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Approval must be PAGE=ANGLE") from exc
    if page < 1 or angle not in VALID_ANGLES:
        raise argparse.ArgumentTypeError("PAGE must be positive and ANGLE 0, 90, 180, or 270")
    return page, angle


def load_result(orientation_dir: Path, page_path: Path) -> OrientationResult:
    result_path = orientation_dir / f"{page_path.stem}_res.json"
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    labels = result["label_names"]
    scores = result["scores"]
    if len(labels) < 2 or len(scores) < 2:
        raise ValueError(f"Expected top-2 orientation scores in {result_path}")
    angle = int(labels[0])
    if angle not in VALID_ANGLES:
        raise ValueError(f"Unexpected orientation label {labels[0]!r} in {result_path}")
    return OrientationResult(
        angle=angle,
        confidence=float(scores[0]),
        margin=float(scores[0]) - float(scores[1]),
        result_path=result_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--orientation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-confidence", type=float, default=0.80)
    parser.add_argument("--min-margin", type=float, default=0.20)
    parser.add_argument(
        "--approved-orientation",
        type=parse_approval,
        action="append",
        default=[],
        help="Manual, source-image-reviewed override, e.g. 20=0.",
    )
    parser.add_argument(
        "--allow-review",
        action="store_true",
        help="Copy unresolved pages unchanged but mark them review_required in the manifest.",
    )
    args = parser.parse_args()

    if not 0 < args.min_confidence <= 1 or not 0 <= args.min_margin <= 1:
        raise ValueError("Confidence and margin thresholds must be in [0, 1]")
    approvals = dict(args.approved_orientation)
    records: list[dict] = []
    unresolved: list[int] = []

    for page_number, source in enumerate(page_paths(args.input_dir), start=1):
        result = load_result(args.orientation_dir, source)
        manual_angle = approvals.get(page_number)
        if manual_angle is not None:
            applied_angle = manual_angle
            decision = "manual_source_review"
        elif result.confidence >= args.min_confidence and result.margin >= args.min_margin:
            applied_angle = result.angle
            decision = "classifier_accepted"
        else:
            applied_angle = 0
            decision = "review_required"
            unresolved.append(page_number)

        records.append(
            {
                "page": page_number,
                "source": str(source),
                "orientation_result": str(result.result_path),
                "classifier_correction_ccw_degrees": result.angle,
                "confidence": round(result.confidence, 6),
                "top2_margin": round(result.margin, 6),
                "applied_rotation_ccw_degrees": applied_angle,
                "decision": decision,
                "output": str(args.output_dir / source.name),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "orientation-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "min_confidence": args.min_confidence,
                "min_margin": args.min_margin,
                "pages": records,
                "review_required_pages": unresolved,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if unresolved and not args.allow_review:
        raise RuntimeError(
            "Orientation review required for pages "
            f"{unresolved}; inspect the source and pass --approved-orientation PAGE=ANGLE."
        )

    for record in records:
        with Image.open(record["source"]) as source_file:
            image = source_file.convert("RGB")
        angle = record["applied_rotation_ccw_degrees"]
        if angle:
            image = image.rotate(angle, expand=True)
        image.save(record["output"], format="PNG", optimize=True)

    print(manifest_path)


if __name__ == "__main__":
    main()
