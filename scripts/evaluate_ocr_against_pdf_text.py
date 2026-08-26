#!/usr/bin/env python3
"""Evaluate OCR text only where the PDF's embedded text is demonstrably clean.

The supplied PDF has a corrupted Thai text layer on some pages.  This script
first rejects a page with C0/C1 control-character contamination, then excludes
detected table pages by default because visual table geometry and text-stream
order are not equivalent.  On the remaining clean prose pages it reports a
format-insensitive normalised character error rate (nCER), not a misleading
Thai whitespace-token WER.

The PDF text layer is *evaluation reference only*.  It is never sent to the
OCR model and is not eligible to patch an OCR response.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import unicodedata
from pathlib import Path
from typing import Any


def extracted_page_text(pdftotext: str, pdf: Path, page: int) -> str:
    command = [pdftotext, "-f", str(page), "-l", str(page), "-layout", str(pdf), "-"]
    return subprocess.check_output(command, text=True, errors="replace")


def text_health(text: str) -> dict[str, float | int | bool]:
    nonspace = [character for character in text if not character.isspace()]
    controls = sum(
        ord(character) < 32 or 0x7F <= ord(character) <= 0x9F
        for character in nonspace
    )
    thai = sum("\u0E00" <= character <= "\u0E7F" for character in nonspace)
    ratio = controls / max(1, len(nonspace))
    return {
        "nonspace_characters": len(nonspace),
        "control_characters": controls,
        "control_ratio": round(ratio, 6),
        "thai_characters": thai,
        "clean": ratio <= 0.005 and thai >= 100,
    }


def normalise_for_cer(text: str) -> str:
    """Retain language characters, numerals and combining marks; discard markup."""
    text = re.sub(r"<figure>.*?</figure>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = unicodedata.normalize("NFC", text)
    return "".join(
        character
        for character in text
        if unicodedata.category(character)[0] in {"L", "M", "N"}
    )


def edit_distance(reference: str, hypothesis: str) -> int:
    """Levenshtein distance using two rows to keep the evaluator dependency-free."""
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for reference_index, reference_character in enumerate(reference, start=1):
        current = [reference_index]
        for hypothesis_index, hypothesis_character in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[hypothesis_index] + 1,
                    previous[hypothesis_index - 1]
                    + (reference_character != hypothesis_character),
                )
            )
        previous = current
    return previous[-1]


def table_pages(layout_manifest: dict[str, Any], minimum_score: float) -> set[int]:
    return {
        int(page["page"])
        for page in layout_manifest["pages"]
        if any(
            box["label"] == "table" and float(box["score"]) >= minimum_score
            for box in page["boxes"]
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--ocr-results", type=Path, required=True)
    parser.add_argument("--layout-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pdftotext", default="pdftotext")
    parser.add_argument("--table-min-score", type=float, default=0.45)
    parser.add_argument(
        "--include-table-pages",
        action="store_true",
        help="Include table pages despite text-stream/geometry mismatch risk.",
    )
    args = parser.parse_args()
    if not 0 <= args.table_min_score <= 1:
        raise ValueError("--table-min-score must be in [0, 1]")

    ocr_data = json.loads(args.ocr_results.read_text(encoding="utf-8"))
    layout_data = json.loads(args.layout_manifest.read_text(encoding="utf-8"))
    results_by_page = {int(result["page"]): result for result in ocr_data["results"]}
    tables = table_pages(layout_data, args.table_min_score)
    page_records: list[dict[str, Any]] = []
    total_reference = 0
    total_edits = 0

    for page in sorted(results_by_page):
        reference = extracted_page_text(args.pdftotext, args.pdf, page)
        health = text_health(reference)
        record: dict[str, Any] = {
            "page": page,
            "pdf_text_health": health,
            "layout_detected_table": page in tables,
        }
        if not health["clean"]:
            record["status"] = "skipped_corrupt_pdf_text"
        elif page in tables and not args.include_table_pages:
            record["status"] = "skipped_table_geometry"
        else:
            clean_reference = normalise_for_cer(reference)
            hypothesis = normalise_for_cer(str(results_by_page[page]["text"]))
            edits = edit_distance(clean_reference, hypothesis)
            record.update(
                {
                    "status": "scored_clean_prose_reference",
                    "reference_characters": len(clean_reference),
                    "ocr_characters": len(hypothesis),
                    "edits": edits,
                    "ncer": round(edits / max(1, len(clean_reference)), 6),
                }
            )
            total_reference += len(clean_reference)
            total_edits += edits
        page_records.append(record)

    output = {
        "metric": {
            "name": "normalised_character_error_rate",
            "not_a_word_error_rate": True,
            "reason": "Thai does not reliably use whitespace word boundaries; complete human ground truth is unavailable.",
            "reference_policy": "Only clean embedded PDF prose text is used; corrupt and table-geometry pages are excluded by default.",
        },
        "inputs": {
            "pdf": str(args.pdf),
            "ocr_results": str(args.ocr_results),
            "layout_manifest": str(args.layout_manifest),
        },
        "summary": {
            "scored_pages": sum(record["status"] == "scored_clean_prose_reference" for record in page_records),
            "skipped_corrupt_pages": sum(record["status"] == "skipped_corrupt_pdf_text" for record in page_records),
            "skipped_table_pages": sum(record["status"] == "skipped_table_geometry" for record in page_records),
            "reference_characters": total_reference,
            "edits": total_edits,
            "ncer": round(total_edits / max(1, total_reference), 6),
        },
        "pages": page_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
