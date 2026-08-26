#!/usr/bin/env python3
"""Prepare reproducible Qwen OCR inputs with PP-DocLayoutV3 layout metadata.

The script deliberately keeps two image variants for each page:

* ``full`` is the complete rendered PDF page.  It is the fair way to compare
  pixel budgets because no document content is removed.
* ``layout-content`` is the tight union of detected semantic content blocks.
  It removes only page margins, headers, footers, and page numbers.  It is a
  separate crop experiment, not a replacement for the full-page benchmark.

Use the local PP-DocLayoutV3 safetensors model so no detector weights need be
downloaded during a benchmark run.  The model is used only for geometry and
reading order; Qwen remains the Thai OCR engine.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw
from paddleocr import LayoutDetection


EXCLUDED_FROM_CONTENT_CROP = {"header", "footer", "page_number"}


@dataclass(frozen=True)
class Profile:
    """An exact Qwen-friendly image grid (multiples of 32 pixels)."""

    name: str
    width: int
    height: int


def parse_profile(value: str) -> Profile:
    try:
        name, dimensions = value.split("=", 1)
        width_text, height_text = dimensions.lower().split("x", 1)
        width, height = int(width_text), int(height_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Profile must look like name=WIDTHxHEIGHT, e.g. 1800=1280x1792"
        ) from exc
    if not name or width < 32 or height < 32 or width % 32 or height % 32:
        raise argparse.ArgumentTypeError(
            "Dimensions must be positive multiples of Qwen's 32-pixel grid"
        )
    return Profile(name=name, width=width, height=height)


def page_paths(input_dir: Path) -> list[Path]:
    pages = sorted(input_dir.glob("page-*.png"))
    if not pages:
        raise FileNotFoundError(f"No page-*.png files found under {input_dir}")
    return pages


def clip_box(
    coordinates: Iterable[float], image_width: int, image_height: int, padding: int
) -> tuple[int, int, int, int]:
    left, top, right, bottom = (round(float(value)) for value in coordinates)
    return (
        max(0, left - padding),
        max(0, top - padding),
        min(image_width, right + padding),
        min(image_height, bottom + padding),
    )


def content_crop_box(
    boxes: list[dict], image_width: int, image_height: int, padding: int
) -> tuple[int, int, int, int]:
    selected = [
        box
        for box in boxes
        if box["label"] not in EXCLUDED_FROM_CONTENT_CROP and box["score"] >= 0.45
    ]
    if not selected:
        return (0, 0, image_width, image_height)
    left = min(float(box["coordinate"][0]) for box in selected)
    top = min(float(box["coordinate"][1]) for box in selected)
    right = max(float(box["coordinate"][2]) for box in selected)
    bottom = max(float(box["coordinate"][3]) for box in selected)
    return clip_box((left, top, right, bottom), image_width, image_height, padding)


def overlay(source: Image.Image, boxes: list[dict], destination: Path) -> None:
    image = source.convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    colors = {
        "table": "#d62828",
        "title": "#00509d",
        "paragraph_title": "#00509d",
        "header": "#6a4c93",
        "footer": "#6a4c93",
        "text": "#2a9d8f",
    }
    for box in boxes:
        left, top, right, bottom = (round(float(value)) for value in box["coordinate"])
        color = colors.get(box["label"], "#f4a261")
        draw.rectangle((left, top, right, bottom), outline=color, width=5)
        draw.text((left + 4, max(0, top - 22)), box["label"], fill=color, stroke_width=1)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG", optimize=True)


def resize_to_profile(source: Image.Image, profile: Profile) -> Image.Image:
    """Resize without distorting glyphs, while retaining the profile's budget.

    A layout crop can have a very different aspect ratio from an A4 page.  A
    direct ``resize((width, height))`` would stretch Thai glyphs and invalidate
    the comparison.  The exact grid below keeps the area at or below the
    full-page profile and caps the long side, both on Qwen's 32-pixel grid.
    """
    target_area = profile.width * profile.height
    scale_for_area = math.sqrt(target_area / (source.width * source.height))
    scale_for_side = max(profile.width, profile.height) / max(source.size)
    scale = min(scale_for_area, scale_for_side)
    width = max(32, round((source.width * scale) / 32) * 32)
    height = max(32, round((source.height * scale) / 32) * 32)
    while width * height > target_area:
        if width / source.width >= height / source.height:
            width -= 32
        else:
            height -= 32
    return source.resize((width, height), Image.Resampling.LANCZOS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--layout-nms", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--crop-padding", type=int, default=32)
    parser.add_argument(
        "--profile",
        type=parse_profile,
        action="append",
        required=True,
        help="Repeatable exact target grid such as 1800=1280x1792",
    )
    args = parser.parse_args()

    pages = page_paths(args.input_dir)
    profiles: list[Profile] = args.profile
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # PP-DocLayoutV3 is a public safetensors (Transformers-engine) release.
    # Supplying both fields catches accidental fall-back to PaddleOCR's default
    # PP-DocLayout model.
    detector = LayoutDetection(
        model_name="PP-DocLayoutV3",
        model_dir=str(args.model_dir),
        engine="transformers",
        device=args.device,
    )

    manifest_pages: list[dict] = []
    for index, page_path in enumerate(pages, start=1):
        with Image.open(page_path) as source_file:
            source = source_file.convert("RGB")
        result = detector.predict(str(page_path), layout_nms=args.layout_nms)[0].json["res"]
        boxes = result["boxes"]
        crop = content_crop_box(boxes, source.width, source.height, args.crop_padding)
        crop_image = source.crop(crop)
        overlay(source, boxes, output_dir / "layout-overlays" / page_path.name)

        page_record = {
            "page": index,
            "source": str(page_path),
            "source_dimensions": [source.width, source.height],
            "content_crop": list(crop),
            "content_crop_dimensions": [crop_image.width, crop_image.height],
            "boxes": boxes,
        }
        manifest_pages.append(page_record)

        generated_dimensions: dict[str, dict[str, list[int]]] = {}
        for profile in profiles:
            for variant, image in (("full", source), ("layout-content", crop_image)):
                destination = output_dir / "images" / profile.name / variant / page_path.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                resized = resize_to_profile(image, profile)
                generated_dimensions.setdefault(profile.name, {})[variant] = [
                    resized.width,
                    resized.height,
                ]
                resized.save(destination, format="PNG", optimize=True)
        page_record["generated_dimensions"] = generated_dimensions

    manifest = {
        "detector": {
            "model_name": "PP-DocLayoutV3",
            "model_dir": str(args.model_dir),
            "engine": "transformers",
            "device": args.device,
            "layout_nms": args.layout_nms,
        },
        "crop_policy": {
            "name": "content_union",
            "excluded_labels": sorted(EXCLUDED_FROM_CONTENT_CROP),
            "minimum_box_confidence": 0.45,
            "padding_pixels_at_source_resolution": args.crop_padding,
            "warning": "layout-content intentionally omits excluded header/footer/page-number regions; full remains the canonical OCR input.",
        },
        "profiles": [profile.__dict__ for profile in profiles],
        "pages": manifest_pages,
    }
    (output_dir / "layout-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "pages": len(pages),
                "profiles": [profile.__dict__ for profile in profiles],
                "output_dir": str(output_dir),
                "manifest": str(output_dir / "layout-manifest.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
