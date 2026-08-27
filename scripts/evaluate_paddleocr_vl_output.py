#!/usr/bin/env python3
"""Measure PaddleOCR-VL page output only against healthy PDF prose text.

The companion evaluator for Typhoon expects one JSON document containing all
pages.  PaddleOCR-VL writes one ``page-XX_res.json`` document per page, so
this adapter joins its recognised blocks in layout order and applies the same
clean-reference / normalised-CER policy.  The PDF text layer is evaluation
only; this program never produces patch text for OCR output.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path
from typing import Any


def load_common_evaluator() -> Any:
    path = Path(__file__).with_name("evaluate_ocr_against_pdf_text.py")
    spec = importlib.util.spec_from_file_location("ocr_evaluator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def page_number(path: Path) -> int:
    match = re.fullmatch(r"page-(\d+)_res\.json", path.name)
    if match is None:
        raise ValueError(f"Unexpected PaddleOCR-VL result name: {path.name}")
    return int(match.group(1))


def page_hypothesis(data: dict[str, Any]) -> tuple[str, bool]:
    blocks = data.get("parsing_res_list", [])
    if not isinstance(blocks, list):
        raise ValueError("PaddleOCR-VL result has no parsing_res_list")
    text = "\n".join(str(block.get("block_content", "")) for block in blocks)
    has_table = any(block.get("block_label") == "table" for block in blocks)
    return text, has_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pdftotext", default="pdftotext")
    parser.add_argument(
        "--include-table-pages",
        action="store_true",
        help="Score table pages despite PDF text-stream / table-geometry mismatch.",
    )
    args = parser.parse_args()
    common = load_common_evaluator()
    result_paths = sorted(args.results_dir.glob("page-*_res.json"), key=page_number)
    if not result_paths:
        raise FileNotFoundError(f"No page-XX_res.json files in {args.results_dir}")

    pages: list[dict[str, Any]] = []
    total_reference = 0
    total_edits = 0
    for result_path in result_paths:
        page = page_number(result_path)
        source = json.loads(result_path.read_text(encoding="utf-8"))
        hypothesis, has_table = page_hypothesis(source)
        reference = common.extracted_page_text(args.pdftotext, args.pdf, page)
        health = common.text_health(reference)
        record: dict[str, Any] = {
            "page": page,
            "pdf_text_health": health,
            "paddle_detected_table": has_table,
        }
        if not health["clean"]:
            record["status"] = "skipped_corrupt_pdf_text"
        elif has_table and not args.include_table_pages:
            record["status"] = "skipped_table_geometry"
        else:
            clean_reference = common.normalise_for_cer(reference)
            clean_hypothesis = common.normalise_for_cer(hypothesis)
            edits = common.edit_distance(clean_reference, clean_hypothesis)
            record.update(
                {
                    "status": "scored_clean_prose_reference",
                    "reference_characters": len(clean_reference),
                    "ocr_characters": len(clean_hypothesis),
                    "edits": edits,
                    "ncer": round(edits / max(1, len(clean_reference)), 6),
                }
            )
            total_reference += len(clean_reference)
            total_edits += edits
        pages.append(record)

    output = {
        "metric": {
            "name": "normalised_character_error_rate",
            "not_a_word_error_rate": True,
            "reference_policy": "Only clean embedded PDF prose text is used; corrupt and table pages are excluded by default.",
            "safety": "The PDF text layer is evaluation-only and cannot patch PaddleOCR-VL output.",
        },
        "inputs": {"pdf": str(args.pdf), "results_dir": str(args.results_dir)},
        "summary": {
            "scored_pages": sum(p["status"] == "scored_clean_prose_reference" for p in pages),
            "skipped_corrupt_pages": sum(p["status"] == "skipped_corrupt_pdf_text" for p in pages),
            "skipped_table_pages": sum(p["status"] == "skipped_table_geometry" for p in pages),
            "reference_characters": total_reference,
            "edits": total_edits,
            "ncer": round(total_edits / max(1, total_reference), 6),
        },
        "pages": pages,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
