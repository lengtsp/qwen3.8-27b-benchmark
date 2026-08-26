#!/usr/bin/env python3
"""Make OCR detail crops from PP-DocLayoutV3 geometry, never fixed coordinates.

This is the *candidate* stage of a conservative OCR repair pipeline.  It reads
the ``layout-manifest.json`` emitted by ``prepare_layout_ocr_images.py`` and
creates image crops only from regions detected by PP-DocLayoutV3:

* semantic blocks (title, paragraph title, text, table, figure, formula),
* whole header/footer/page-number lines reconstructed by vertical alignment,
* tiles only when a *detected* text block is too tall for the selected model.

The crop coordinates, padding, resizing and detector-box provenance are saved
in ``candidate-manifest.json``.  A candidate is evidence for an independent
OCR reader; this program never substitutes its text into a full-page OCR
transcript.  That separation makes it safe to use with documents whose layout
is unlike the calibration PDF.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


BLOCK_LABELS = {
    "doc_title",
    "title",
    "paragraph_title",
    "text",
    "table",
    "figure",
    "formula",
}
LINE_LABELS = {"header", "footer", "page_number"}


@dataclass(frozen=True)
class Box:
    """One PP-DocLayoutV3 detection in source-image pixel coordinates."""

    index: int
    label: str
    score: float
    left: float
    top: float
    right: float
    bottom: float
    order: int | None

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @property
    def center_y(self) -> float:
        return (self.top + self.bottom) / 2

    def as_list(self) -> list[float]:
        return [self.left, self.top, self.right, self.bottom]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_box(index: int, raw: dict[str, Any], width: int, height: int) -> Box | None:
    try:
        left, top, right, bottom = (float(value) for value in raw["coordinate"])
        label = str(raw["label"])
        score = float(raw["score"])
    except (KeyError, TypeError, ValueError):
        return None
    left, right = sorted((max(0.0, left), min(float(width), right)))
    top, bottom = sorted((max(0.0, top), min(float(height), bottom)))
    if right <= left or bottom <= top:
        return None
    raw_order = raw.get("order")
    order = int(raw_order) if isinstance(raw_order, (int, float)) else None
    return Box(index, label, score, left, top, right, bottom, order)


def dynamic_crop(
    bounds: Iterable[float], image_width: int, image_height: int, padding_ratio: float
) -> tuple[int, int, int, int]:
    """Pad a detected region proportionally; no page-specific coordinate exists."""
    left, top, right, bottom = bounds
    region_width, region_height = right - left, bottom - top
    # At least one source pixel retains border strokes for small labels; all
    # larger padding is calculated from the detector's own region geometry.
    padding = max(1, math.ceil(min(region_width, region_height) * padding_ratio))
    return (
        max(0, math.floor(left - padding)),
        max(0, math.floor(top - padding)),
        min(image_width, math.ceil(right + padding)),
        min(image_height, math.ceil(bottom + padding)),
    )


def resize_for_model(image: Image.Image, max_side: int) -> Image.Image:
    """Apply Typhoon's fixed-long-side policy without changing aspect ratio."""
    longest = max(image.size)
    # Typhoon OCR 1.5's supplied local helper leaves only tiny images alone;
    # every other image is resized to its trained 1,800-pixel long side.  The
    # policy applies equally to a small detected field and a full text block.
    if longest <= 300:
        return image
    scale = max_side / longest
    width = max(1, round(image.width * scale))
    height = max(1, round(image.height * scale))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def vertically_aligned_lines(boxes: list[Box]) -> list[list[Box]]:
    """Group same-label fragments into lines by their detected vertical alignment."""
    if not boxes:
        return []
    pending = sorted(boxes, key=lambda box: (box.center_y, box.left))
    lines: list[list[Box]] = []
    for box in pending:
        for line in lines:
            line_top = min(item.top for item in line)
            line_bottom = max(item.bottom for item in line)
            line_height = line_bottom - line_top
            overlap = max(0.0, min(box.bottom, line_bottom) - max(box.top, line_top))
            # Fragments belong to a line if their vertical overlap is robust
            # relative to the smaller detected extent.  This adapts to scans,
            # different DPIs and documents with unrelated margins.
            if overlap >= 0.45 * min(box.height, line_height):
                line.append(box)
                break
        else:
            lines.append([box])
    return [sorted(line, key=lambda box: box.left) for line in lines]


def text_tiles(box: Box, page_height: int, max_height_ratio: float, overlap_ratio: float) -> list[tuple[float, float, float, float]]:
    """Tile an unusually tall detected text block, retaining local overlap."""
    maximum = max(1.0, page_height * max_height_ratio)
    if box.height <= maximum:
        return [(box.left, box.top, box.right, box.bottom)]
    stride = max(1.0, maximum * (1.0 - overlap_ratio))
    tiles: list[tuple[float, float, float, float]] = []
    top = box.top
    while top < box.bottom:
        bottom = min(box.bottom, top + maximum)
        tiles.append((box.left, top, box.right, bottom))
        if bottom >= box.bottom:
            break
        top += stride
    return tiles


def candidate_record(
    *,
    candidate_id: str,
    page_number: int,
    kind: str,
    bounds: tuple[float, float, float, float],
    source_box_indexes: list[int],
    source_labels: list[str],
    source_orders: list[int],
    image: Image.Image,
    source_width: int,
    source_height: int,
    padding_ratio: float,
    max_side: int,
    output_dir: Path,
) -> dict[str, Any]:
    crop_bounds = dynamic_crop(bounds, source_width, source_height, padding_ratio)
    cropped = image.crop(crop_bounds)
    model_image = resize_for_model(cropped, max_side)
    relative = Path(f"page-{page_number:02d}") / f"{candidate_id}.png"
    destination = output_dir / "candidates" / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    model_image.save(destination, format="PNG", optimize=True)
    return {
        "id": candidate_id,
        "page": page_number,
        "kind": kind,
        "detector_bounds": [round(value, 3) for value in bounds],
        "crop_bounds": list(crop_bounds),
        "source_box_indexes": source_box_indexes,
        "source_labels": source_labels,
        "reading_orders": source_orders,
        "crop_dimensions": [cropped.width, cropped.height],
        "model_input_dimensions": [model_image.width, model_image.height],
        "image": str(Path("candidates") / relative),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-score", type=float, default=0.45)
    parser.add_argument(
        "--padding-ratio",
        type=float,
        default=0.08,
        help="Padding derived from each detected region's shorter side (default: 0.08).",
    )
    parser.add_argument(
        "--max-text-height-ratio",
        type=float,
        default=0.25,
        help="Split a detected text region taller than this fraction of the page (default: 0.25).",
    )
    parser.add_argument(
        "--tile-overlap-ratio",
        type=float,
        default=0.12,
        help="Overlap for automatic tiles within an over-height detected text block.",
    )
    parser.add_argument(
        "--model-max-side",
        type=int,
        default=1800,
        help="Resize each crop above 300 px to this long side; 1800 matches Typhoon OCR's documented policy.",
    )
    args = parser.parse_args()

    if not 0 <= args.min_score <= 1:
        raise ValueError("--min-score must be in [0, 1]")
    if not 0 <= args.padding_ratio <= 1:
        raise ValueError("--padding-ratio must be in [0, 1]")
    if not 0 < args.max_text_height_ratio <= 1:
        raise ValueError("--max-text-height-ratio must be in (0, 1]")
    if not 0 <= args.tile_overlap_ratio < 1:
        raise ValueError("--tile-overlap-ratio must be in [0, 1)")
    if args.model_max_side < 1:
        raise ValueError("--model-max-side must be positive")

    source_manifest = json.loads(args.layout_manifest.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pages_output: list[dict[str, Any]] = []

    for page_record in source_manifest["pages"]:
        page_number = int(page_record["page"])
        source_path = Path(page_record["source"])
        with Image.open(source_path) as source_file:
            source_image = source_file.convert("RGB")
        width, height = source_image.size
        parsed = [
            item
            for index, raw in enumerate(page_record["boxes"])
            if (item := parse_box(index, raw, width, height)) is not None
            and item.score >= args.min_score
        ]
        candidates: list[dict[str, Any]] = []
        kind_counts: dict[str, int] = defaultdict(int)

        def add(
            kind: str,
            bounds: tuple[float, float, float, float],
            contributors: list[Box],
        ) -> None:
            kind_counts[kind] += 1
            candidate_id = f"{kind}-{kind_counts[kind]:03d}"
            candidates.append(
                candidate_record(
                    candidate_id=candidate_id,
                    page_number=page_number,
                    kind=kind,
                    bounds=bounds,
                    source_box_indexes=[box.index for box in contributors],
                    source_labels=sorted({box.label for box in contributors}),
                    source_orders=[box.order for box in contributors if box.order is not None],
                    image=source_image,
                    source_width=width,
                    source_height=height,
                    padding_ratio=args.padding_ratio,
                    max_side=args.model_max_side,
                    output_dir=args.output_dir,
                )
            )

        for label in sorted(LINE_LABELS):
            for line in vertically_aligned_lines([box for box in parsed if box.label == label]):
                add(
                    f"{label}_line",
                    (
                        min(box.left for box in line),
                        min(box.top for box in line),
                        max(box.right for box in line),
                        max(box.bottom for box in line),
                    ),
                    line,
                )

        for box in sorted(
            (box for box in parsed if box.label in BLOCK_LABELS),
            key=lambda item: (item.order is None, item.order if item.order is not None else 10**9, item.top, item.left),
        ):
            tiles = (
                text_tiles(
                    box,
                    page_height=height,
                    max_height_ratio=args.max_text_height_ratio,
                    overlap_ratio=args.tile_overlap_ratio,
                )
                if box.label == "text"
                else [(box.left, box.top, box.right, box.bottom)]
            )
            for tile_number, bounds in enumerate(tiles, start=1):
                kind = box.label if len(tiles) == 1 else f"text_tile"
                add(kind, bounds, [box])

        pages_output.append(
            {
                "page": page_number,
                "source": str(source_path),
                "source_sha256": sha256_file(source_path),
                "source_dimensions": [width, height],
                "candidates": candidates,
            }
        )

    output_manifest = {
        "source_layout_manifest": str(args.layout_manifest),
        "policy": {
            "name": "ppdoclayoutv3_dynamic_region_candidates",
            "version": 1,
            "fixed_bounding_boxes": False,
            "minimum_detector_score": args.min_score,
            "padding_ratio_of_detected_short_side": args.padding_ratio,
            "text_tiling": {
                "maximum_detected_region_height_ratio": args.max_text_height_ratio,
                "overlap_ratio": args.tile_overlap_ratio,
            },
            "model_image_target_long_side_when_over_300px": args.model_max_side,
            "warning": "Candidates are independently read evidence only. Do not automatically overwrite the canonical full-page OCR text.",
        },
        "pages": pages_output,
    }
    manifest_path = args.output_dir / "candidate-manifest.json"
    manifest_path.write_text(
        json.dumps(output_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "pages": len(pages_output),
                "candidates": sum(len(page["candidates"]) for page in pages_output),
                "manifest": str(manifest_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
