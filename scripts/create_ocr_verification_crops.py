#!/usr/bin/env python3
"""Create provenance-preserving, field-level OCR verification crops.

The full-page result remains the canonical transcript.  These crop images are
for a second OCR pass on an explicitly named field such as a gazette line,
date, statistic, or table row.  The JSON manifest records the original page
and rectangle so an accepted crop result can replace only that field.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass(frozen=True)
class Crop:
    name: str
    page: int
    left: int
    top: int
    right: int
    bottom: int


def parse_crop(value: str) -> Crop:
    try:
        name, location = value.split("=", 1)
        page_text, rectangle = location.split(":", 1)
        left, top, right, bottom = (int(number) for number in rectangle.split(","))
        crop = Crop(name, int(page_text), left, top, right, bottom)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Crop must be NAME=PAGE:LEFT,TOP,RIGHT,BOTTOM"
        ) from exc
    if not crop.name or crop.page < 1 or crop.left >= crop.right or crop.top >= crop.bottom:
        raise argparse.ArgumentTypeError("Crop name/page/rectangle is invalid")
    return crop


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--crop", type=parse_crop, action="append", required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    for crop in args.crop:
        source = args.input_dir / f"page-{crop.page:02d}.png"
        if not source.is_file():
            raise FileNotFoundError(source)
        with Image.open(source) as source_file:
            image = source_file.convert("RGB")
        if crop.right > image.width or crop.bottom > image.height:
            raise ValueError(f"{crop.name}: rectangle lies outside {source}")
        destination = args.output_dir / f"{crop.name}.png"
        image.crop((crop.left, crop.top, crop.right, crop.bottom)).save(
            destination, format="PNG", optimize=True
        )
        manifest.append(
            {
                "name": crop.name,
                "page": crop.page,
                "source": str(source),
                "rectangle": [crop.left, crop.top, crop.right, crop.bottom],
                "dimensions": [crop.right - crop.left, crop.bottom - crop.top],
                "image": str(destination),
            }
        )
    (args.output_dir / "verification-crops.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
